import argparse
import json
import os
import random
import sys
import re
from typing import Any, Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from tqdm import tqdm
from colorama import Fore, Style, init as colorama_init


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from agentic_rag.tools.qdrant_storage import QdrantStorage, MyEmbedder

colorama_init()

DEFAULT_TARGET = 150
DEFAULT_OUTPUT_XLSX = "eval/pdpa_generated.xlsx"
DEFAULT_OUTPUT_JSONL = "eval/pdpa_generated.jsonl"


def _call_llm(prompt: str, temperature: float = 0.3, max_tokens: int = 8192) -> str:
    base_url = (
        os.getenv("LLAMA_CPP_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "http://localhost:8080/v1"
    )
    model = (
        os.getenv("LLAMA_CPP_MODEL")
        or os.getenv("EVAL_MODEL")
        or "hf.co/scb10x/typhoon2.1-gemma3-4b-gguf:Q4_K_M"
    )
    api_key = os.getenv("OPENAI_API_KEY", "not-needed")

    client = OpenAI(base_url=base_url, api_key=api_key)

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "คุณเป็นผู้เชี่ยวชาญด้าน PDPA ที่เคร่งครัดเรื่องความถูกต้อง "
                        "ห้ามเดา ห้ามสมมติ ต้องตอบเป็นภาษาไทย "
                        "คำตอบต้องมีความยาว ละเอียด ครอบคลุมเนื้อหาในบริบท และถูกต้องตามหลักกฎหมาย PDPA "
                        "ห้ามใช้ความรู้ภายนอกหรือแต่งเติมนอกเหนือข้อมูลที่ส่งให้"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            timeout=120.0,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(Fore.RED + f"✘ API Error: {e}" + Style.RESET_ALL)
        return ""


def _safe_get(payload: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        if k in payload and payload[k] is not None:
            return str(payload[k])
    return default



def fetch_contexts(
    client: QdrantClient,
    max_per_collection: int = 0,
    min_text_len: int = 50,
) -> List[Dict[str, Any]]:

    print(Fore.CYAN + "📁 เริ่มดึงข้อมูลจาก Qdrant Collections (เพื่อใช้เป็น Seed)..." + Style.RESET_ALL)
    contexts: List[Dict[str, Any]] = []

    cols_resp = client.get_collections()
    collections = getattr(cols_resp, "collections", []) or []

    print(f"🔍 พบ {len(collections)} collections\n")

    for col in collections:
        name = getattr(col, "name", None) or col.get("name")
        if not name:
            continue

        print(Fore.YELLOW + f"➡ กำลังอ่าน Collection: {name}" + Style.RESET_ALL)

        next_offset = None
        fetched = 0

        while max_per_collection <= 0 or fetched < max_per_collection:
            points, next_offset = client.scroll(
                collection_name=name,
                with_payload=True,
                limit=64,
                offset=next_offset,
            )
            if not points:
                break

            for p in points:
                payload = getattr(p, "payload", None) or p.get("payload")
                if not isinstance(payload, dict):
                    continue

                text_val = payload.get("text")
                if not text_val or len(str(text_val)) < min_text_len:
                    continue

                contexts.append(
                    {
                        "text": str(text_val),
                        "doc_title": _safe_get(
                            payload,
                            ["source_file", "doc_title", "document_title", "file_name", "filename"],
                            "ไม่ระบุไฟล์",
                        ),
                        "page": _safe_get(
                            payload,
                            ["page_number", "page", "page_index", "page_idx"],
                            "ไม่ระบุหน้า",
                        ),
                        "collection": name,
                    }
                )

                fetched += 1
                if max_per_collection > 0 and fetched >= max_per_collection:
                    break

            if not next_offset:
                break

        print(f"   ✔ ดึงได้ {fetched} passages\n")

    print(Fore.GREEN + f"📦 รวมทั้งหมด: {len(contexts)} passages\n" + Style.RESET_ALL)
    return contexts



def generate_question(seed_chunk: Dict[str, Any]) -> str:
    prompt = (
        "จากข้อความต่อไปนี้ ให้สร้าง 'คำถาม' เกี่ยวกับ PDPA 1 ข้อ "
        "โดยสมมติว่าคุณเป็นผู้ใช้งานทั่วไปที่สงสัย "
        "ให้ตั้งคำถามสั้นๆ กระชับ เป็นธรรมชาติ (เหมือนคนถามจริง ไม่ต้องเป็นทางการมาก) "
        "และสามารถตอบได้โดยใช้ข้อมูลในข้อความนี้เป็นหลัก "
        "ขอเฉพาะตัวคำถาม ไม่ต้องมีคำตอบ\n\n"
        f"ข้อความ:\n{seed_chunk['text']}\n\n"
        "คำถาม (สั้นๆ กระชับ):"
    )
    question = _call_llm(prompt, temperature=0.7, max_tokens=256)

    return question.strip().strip('"').strip("'")


def retrieve_contexts(storage: QdrantStorage, question: str, top_k: int = 5) -> List[Dict[str, Any]]:

    client = storage.client
    cols_resp = client.get_collections()
    collections = getattr(cols_resp, "collections", []) or []
    
    all_results = []
    query_vector = storage.embedder.encode(question)

    for col in collections:
        name = getattr(col, "name", None) or col.get("name")
        if not name: 
            continue
            
        try:
            results = client.search(
                collection_name=name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True
            )
            for r in results:
                payload = r.payload or {}
                score = r.score
                all_results.append({
                    "text": payload.get("text", ""),
                    "doc_title": _safe_get(payload, ["source_file", "doc_title", "file_name"], "Unknown"),
                    "page": _safe_get(payload, ["page_number", "page"], "Unknown"),
                    "score": score
                })
        except Exception:
            continue

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


def generate_answer(question: str, contexts: List[Dict[str, Any]]) -> str:
    context_text = ""
    for idx, ctx in enumerate(contexts, 1):
        context_text += f"[{idx}] เอกสาร: {ctx['doc_title']} | หน้า: {ctx['page']}\n{ctx['text']}\n\n"

    prompt = (
        f"คำถาม: {question}\n\n"
        "จากข้อมูลบริบทที่รวบรวมมาด้านล่างนี้ (Retrieved Contexts) "
        "ให้เขียน 'คำตอบ' ที่ละเอียด ครอบคลุม และถูกต้องตามหลัก PDPA "
        "ต้องตอบเป็นภาษาไทย เหมือนการตอบคำถามทั่วไปอย่างเป็นธรรมชาติ "
        "**ห้าม** ขึ้นต้นประโยคด้วยคำว่า 'จากข้อมูลที่ให้มา', 'จากเอกสาร', หรือคำที่แสดงว่าอ่านมาจากบริบท ให้ตอบเข้าเนื้อหาเลย "
        "**ห้าม** ใส่ชื่อเอกสารหรือเลขหน้าลงในเนื้อหาคำตอบ (เช่น ห้ามเขียนว่า 'อ้างอิงจากหน้า 5') "
        "ห้ามเดา ห้ามสมมติ ห้ามใช้ความรู้ภายนอก "
        "หากข้อมูลในบริบทไม่เพียงพอที่จะตอบ ให้ตอบเท่าที่มี\n\n"
        f"บริบท:\n{context_text}\n\n"
        "คำตอบ (ละเอียด ครอบคลุม เป็นธรรมชาติ และ**ไม่มี**คำเกริ่นนำว่าจากข้อมูล):"
    )
    return _call_llm(prompt, temperature=0.3, max_tokens=2048)



def generate_rag_pairs(
    client: QdrantClient,
    seed_contexts: List[Dict[str, Any]],
    target: int,
    top_k: int
) -> List[Dict[str, str]]:

   
    storage = QdrantStorage(
        type="temp",
        qdrant_location=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        embedder=MyEmbedder(os.getenv("RAG_EMBED_MODEL"))
    )

    results: List[Dict[str, str]] = []
    
    print(Fore.CYAN + "🧠 เริ่มสร้าง Q&A แบบ RAG (Seed -> Gen Q -> Retrieve -> Gen A)...\n" + Style.RESET_ALL)
    pbar = tqdm(total=target, desc="Q&A Generated", colour="green")


    random.shuffle(seed_contexts)
    seed_iter = iter(seed_contexts)

    while len(results) < target:
        try:
            seed = next(seed_iter)
        except StopIteration:
 
            random.shuffle(seed_contexts)
            seed_iter = iter(seed_contexts)
            seed = next(seed_iter)

     
        question = generate_question(seed)
        if not question or len(question) < 10:
            continue

        retrieved = retrieve_contexts(storage, question, top_k=top_k)
        if not retrieved:
            continue

        answer = generate_answer(question, retrieved)
        if not answer or len(answer) < 20:
            continue


        doc_titles = sorted(list(set(r["doc_title"] for r in retrieved)))
        pages = sorted(list(set(r["page"] for r in retrieved)))
        
 
        context_str = "\n\n".join([f"Source: {c['doc_title']} (Page {c['page']})\n{c['text']}" for c in retrieved])

        results.append({
            "question": question,
            "ground_truth": answer,
            "doc_title": ", ".join(doc_titles),
            "page": ", ".join(pages),
            "contexts": context_str
        })
        
        pbar.update(1)

    pbar.close()
    return results


def save_outputs(pairs: List[Dict[str, str]], xlsx_path: str, jsonl_path: Optional[str]):
    print(Fore.CYAN + "💾 บันทึกไฟล์ผลลัพธ์..." + Style.RESET_ALL)

    df = pd.DataFrame(pairs, columns=["question", "ground_truth", "doc_title", "page", "contexts"])
    os.makedirs(os.path.dirname(xlsx_path), exist_ok=True)

    df.to_excel(xlsx_path, index=False)
    print(Fore.GREEN + f"✔ เขียน Excel: {xlsx_path}" + Style.RESET_ALL)

    if jsonl_path:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in pairs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(Fore.GREEN + f"✔ เขียน JSONL: {jsonl_path}" + Style.RESET_ALL)

    print(Fore.BLUE + "\n📌 เสร็จสิ้นทั้งหมด!" + Style.RESET_ALL)


def main():
    parser = argparse.ArgumentParser(description="Generate PDPA Q&A using RAG approach.")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_per_collection", type=int, default=0)
    parser.add_argument("--output_xlsx", default=DEFAULT_OUTPUT_XLSX)
    parser.add_argument("--output_jsonl", default=DEFAULT_OUTPUT_JSONL)
    args = parser.parse_args()

    load_dotenv()
    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY")
    )


    contexts = fetch_contexts(client, max_per_collection=args.max_per_collection)

    if not contexts:
        print(Fore.RED + "✘ ERROR: ไม่พบข้อมูลใน Qdrant" + Style.RESET_ALL)
        sys.exit(1)

    pairs = generate_rag_pairs(
        client=client,
        seed_contexts=contexts,
        target=args.target,
        top_k=args.top_k
    )

    save_outputs(pairs, args.output_xlsx, args.output_jsonl)


if __name__ == "__main__":
    main()
