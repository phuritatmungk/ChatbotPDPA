import os
import json
import argparse
from typing import List, Dict, Tuple
from dotenv import load_dotenv

HAS_RAGAS = False
_RAGAS_IMPORT_ERR = None
try:
    import ragas  
    from datasets import Dataset  

    from ragas import evaluate  

    from ragas import metrics as _rmetrics  
    context_precision = getattr(_rmetrics, "context_precision", None)
    context_recall = getattr(_rmetrics, "context_recall", None)
    context_utilization = getattr(_rmetrics, "context_utilization", None)
 
    if context_precision or context_recall or context_utilization:
        HAS_RAGAS = True
    else:
        HAS_RAGAS = False
        _RAGAS_IMPORT_ERR = "no known metrics found in ragas.metrics"
except Exception as _e:
    HAS_RAGAS = False
    _RAGAS_IMPORT_ERR = str(_e)

try:
    from openai import OpenAI  
except Exception:
    OpenAI = None  

from .qdrant_storage import QdrantStorage, MyEmbedder
import numpy as np
from ..crew import build_langgraph_workflow


def _build_storage(collection_name: str, embed_model: str = None, normalize: bool = None) -> QdrantStorage:
   
    try:
        load_dotenv()
    except Exception:
        pass
 
    type_name = collection_name
    if collection_name.startswith("rag_"):
        type_name = collection_name[len("rag_") :]

    if embed_model:
        os.environ["RAG_EMBED_MODEL"] = embed_model
    if normalize is not None:
        os.environ["RAG_NORMALIZE_EMBEDDINGS"] = "true" if normalize else "false"

    storage = QdrantStorage(
        type=type_name,
        qdrant_location=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        embedder=MyEmbedder(os.getenv("RAG_EMBED_MODEL")),
    )
    return storage


def _maybe_generate_answer(question: str, contexts: List[str]) -> str:
    enable_llm = os.getenv("EVAL_ENABLE_LLM", "false").lower() in ("1", "true", "yes")
    if not enable_llm or OpenAI is None:
        return ""
    try:
        base_url = os.getenv("EVAL_BASE_URL", os.getenv("LLAMA_CPP_BASE_URL"))
        api_key = os.getenv("OPENAI_API_KEY", "not-needed")
        model = os.getenv("EVAL_MODEL", os.getenv("LLAMA_CPP_MODEL", "gpt-4o-mini"))
        client = OpenAI(base_url=base_url, api_key=api_key) if base_url else OpenAI()
        joined = "\n\n".join(contexts[:5])
        prompt = (
            "Answer the question using only the provided context. If not answerable, say 'ไม่พบข้อมูล'.\n\n"
            f"Context:\n{joined}\n\nQuestion: {question}"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=512,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def load_questions(path: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            if "question" in obj and "ground_truth" in obj:
                items.append({
                    "question": obj["question"],
                    "ground_truth": obj["ground_truth"],
                })
    return items


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


def evaluate_lite(storage: QdrantStorage, items: List[Dict[str, str]], top_k: int) -> Dict[str, float]:
    """Lightweight retrieval eval without extra dependencies.
    Metrics:
      - precision@k: fraction of retrieved contexts deemed relevant
      - recall@k: fraction of relevant contexts captured (approx via top-k vs GT embedding)
      - mrr: mean reciprocal rank w.r.t. first relevant context
    Relevance is determined by cosine similarity between ground truth and context >= threshold.
    Threshold configurable via EVAL_LITE_SIM_THR (default 0.30).
    """
    embedder = storage.embedder
    sim_thr = float(os.getenv("EVAL_LITE_SIM_THR", "0.30"))

    precisions: List[float] = []
    recalls: List[float] = []
    rr_list: List[float] = []

    gt_vec_cache: Dict[str, np.ndarray] = {}

    for item in items:
        q = item["question"]
        gt = item["ground_truth"]
        results = storage.search(q, limit=top_k)
        contexts = [r.get("text", "") for r in results if r.get("text")]
        if not contexts:
            precisions.append(0.0)
            recalls.append(0.0)
            rr_list.append(0.0)
            continue

  
        if gt not in gt_vec_cache:
            gt_vec_cache[gt] = np.array(embedder.model.encode(gt, normalize_embeddings=True))
        gt_vec = gt_vec_cache[gt]

      
        ctx_vecs = [np.array(embedder.model.encode(c, normalize_embeddings=True)) for c in contexts]
        sims = [
            _cosine(v, gt_vec)
            for v in ctx_vecs
        ]
        relevant_flags = [1 if s >= sim_thr else 0 for s in sims]

        precision_k = sum(relevant_flags) / max(len(contexts), 1)
 
        recall_k = 1.0 if any(relevant_flags) else 0.0

        rr = 0.0
        for idx, flag in enumerate(relevant_flags, start=1):
            if flag:
                rr = 1.0 / idx
                break

        precisions.append(precision_k)
        recalls.append(recall_k)
        rr_list.append(rr)

    return {
        "precision@k": float(np.mean(precisions) if precisions else 0.0),
        "recall@k": float(np.mean(recalls) if recalls else 0.0),
        "mrr": float(np.mean(rr_list) if rr_list else 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval (Ragas if available, else lite)")
    parser.add_argument("--collection", required=True, help="Qdrant collection name (as seen in dashboard)")
    parser.add_argument("--questions", required=True, help="Path to JSONL with fields: question, ground_truth")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--lite", action="store_true", help="Force lightweight eval (no extra installs)")
    parser.add_argument("--embed_model", default=None, help="Override embed model for queries (e.g., 'BAAI/bge-m3')")
    parser.add_argument("--normalize", choices=["true", "false"], default=None, help="Normalize embeddings (default: env or true)")
    parser.add_argument("--show", action="store_true", help="Print question, contexts, LLM answer, and ground truth")
    parser.add_argument("--show_k", type=int, default=2, help="How many contexts to show per question when --show is used")
    parser.add_argument("--agentic", action="store_true", help="Use Agentic RAG (LangGraph workflow) instead of direct Qdrant search")
    args = parser.parse_args()

    normalize = None if args.normalize is None else (args.normalize.lower() == "true")
    storage = _build_storage(args.collection, embed_model=args.embed_model, normalize=normalize)
    data = load_questions(args.questions)

    if args.agentic:
      
        wf = build_langgraph_workflow()
        preds = []
        for item in data:
            question = item["question"]
            gt = item["ground_truth"]
        
            last = None
            stream = wf.stream({"query": question, "context": ""}, stream_mode="values")
            for chunk in stream:
                last = chunk
            retrieved_text = (last or {}).get("retrieved", "") if isinstance(last, dict) else ""
      
            contexts = []
            if isinstance(retrieved_text, str) and retrieved_text:
                sep = "\n____\n"
                contexts = [c for c in retrieved_text.split(sep) if c.strip()]
              
                if "[Web]:" in retrieved_text and "[PDF/Knowledge]:" in retrieved_text:
                    pdf_part = retrieved_text.split("[Web]:")[0].replace("[PDF/Knowledge]:", "").strip()
                    contexts = [c for c in pdf_part.split(sep) if c.strip()]
            answer = (last or {}).get("best_answer") or (last or {}).get("response") or ""
            preds.append({
                "question": question,
                "contexts": contexts[: args.top_k],
                "answer": answer,
                "ground_truth": gt,
            })

            if args.show:
                try:
                    print("\n---")
                    print(f"Q: {question}")
                    to_show = contexts[: max(0, args.show_k)]
                    for i, c in enumerate(to_show, 1):
                        snippet = c if len(c) <= 500 else (c[:500] + "...")
                        print(f"Context[{i}]: {snippet}")
                    if answer:
                        print(f"Answer: {answer}")
                    print(f"Ground truth: {gt}")
                except Exception:
                    pass


        if HAS_RAGAS and len(preds) > 0:
            ds = Dataset.from_list(preds)
            available_metrics = [m for m in [context_precision, context_recall, context_utilization] if m is not None]
            report = evaluate(ds, metrics=available_metrics)
            print("\n=== Retrieval Metrics (Ragas, Agentic) ===")
            print(report)
        else:

            storage = _build_storage(args.collection, embed_model=args.embed_model, normalize=normalize)
            scores = evaluate_lite(storage, preds, args.top_k)
            print("\n=== Retrieval Metrics (Lite, Agentic) ===")
            for k, v in scores.items():
                print(f"{k}: {v:.4f}")
        return

    if args.lite or not HAS_RAGAS:
        scores = evaluate_lite(storage, data, args.top_k)
        print("\n=== Retrieval Metrics (Lite) ===")
        if not HAS_RAGAS and not args.lite:
            print(f"(fallback: ragas not available: {_RAGAS_IMPORT_ERR})")
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")
    else:
   
        try:
            base_url = os.getenv("EVAL_BASE_URL", os.getenv("LLAMA_CPP_BASE_URL"))
            if base_url:
                os.environ["OPENAI_BASE_URL"] = base_url
                os.environ["OPENAI_API_BASE"] = base_url
                if not os.getenv("OPENAI_API_KEY"):
                    os.environ["OPENAI_API_KEY"] = "not-needed"
        except Exception:
            pass

        preds = []
        for item in data:
            question = item["question"]
            gt = item["ground_truth"]
            results = storage.search(question, limit=args.top_k)
            contexts = [r.get("text", "") for r in results if r.get("text")]
            answer = _maybe_generate_answer(question, contexts)
            preds.append({
                "question": question,
                "contexts": contexts,
                "answer": answer,
                "ground_truth": gt,
            })

            if args.show:
                try:
                    print("\n---")
                    print(f"Q: {question}")
                    to_show = contexts[: max(0, args.show_k)]
                    for i, c in enumerate(to_show, 1):

                        snippet = c if len(c) <= 500 else (c[:500] + "...")
                        print(f"Context[{i}]: {snippet}")
                    if answer:
                        print(f"Answer: {answer}")
                    print(f"Ground truth: {gt}")
                except Exception:
                    pass

        ds = Dataset.from_list(preds)

        available_metrics = [m for m in [context_precision, context_recall, context_utilization] if m is not None]
        report = evaluate(ds, metrics=available_metrics)
        print("\n=== Retrieval Metrics (Ragas) ===")
        print(report)

        enable_llm = os.getenv("EVAL_ENABLE_LLM", "false").lower() in ("1", "true", "yes")
        if enable_llm:
            try:
                from ragas.metrics import faithfulness, answer_relevancy  # type: ignore
                llm_report = evaluate(ds, metrics=[faithfulness, answer_relevancy])
                print("\n=== LLM-based Metrics ===")
                print(llm_report)
            except Exception as e:
                print(f"LLM-based evaluation skipped: {e}")


if __name__ == "__main__":
    main()


