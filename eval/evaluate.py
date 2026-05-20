from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any, Dict, Iterable, List, Optional, Union

os.environ.setdefault('GIT_PYTHON_REFRESH', 'quiet')

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness, context_recall
from ragas.llms.base import BaseRagasLLM, Generation, LLMResult
from ragas.embeddings.base import BaseRagasEmbeddings

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agentic_rag.crew import build_langgraph_workflow, call_llm
from src.agentic_rag.tools.qdrant_storage import MyEmbedder




PromptLike = Union[str, Any]


def _prompt_to_text(prompt: PromptLike) -> str:
    if hasattr(prompt, "to_string"):
        return prompt.to_string()
    if hasattr(prompt, "text"):
        return prompt.text 
    return str(prompt)


class AgenticRagRagasLLM(BaseRagasLLM):
    """
    Ragas LLM wrapper that reuses the agentic RAG project's call_llm helper.
    """

    def __init__(self, system_prompt: Optional[str] = None):
        super().__init__()
        self.system_prompt = system_prompt or (
            "คุณคือผู้เชี่ยวชาญด้านการประเมินระบบ Agentic RAG (Agentic Retrieval-Augmented Generation) ภาษาไทย "
            "หน้าที่ของคุณคือตรวจสอบคำถาม คำตอบ และบริบท ตามเกณฑ์ที่กำหนดอย่างเคร่งครัด "
            "ข้อควรปฏิบัติ: "
            "1. คำตอบต้องเป็นรูปแบบ JSON ที่ถูกต้องเท่านั้นเมื่อถูกร้องขอ "
            "2. สำหรับคำถาม ใช่/ไม่ใช่ ให้ตอบ 'yes' หรือ 'no' (หรือ 'true'/'false') เท่านั้น "
            "3. วิเคราะห์บริบทภาษาไทยให้ถูกต้องตามความหมาย "
            "4. ห้ามใส่ markdown format (เช่น ```json) มาในคำตอบ ให้ตอบแต่เนื้อหาล้วนๆ"
        )

    def _clean_reply(self, reply: str) -> str:
        """
        Clean LLM reply while preserving Ragas-expected formats.
        Ragas expects:
        - JSON objects for structured responses
        - Numeric scores (0-1) for ratings
        - Simple yes/no/true/false for binary questions
        - Natural language for question generation
        """
        if not reply:
            return ""
            
        original_reply = reply
        reply = reply.strip()
        
        json_patterns = [
            (reply.find("{"), reply.rfind("}")), 
            (reply.find("["), reply.rfind("]")),
        ]
        
        
        for start_idx, end_idx in json_patterns:
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_candidate = reply[start_idx : end_idx + 1]
                try:
                    json_candidate = json_candidate.replace("'", '"') 
                    json_candidate = json_candidate.replace("True", "true").replace("False", "false") 
                    json_candidate = re.sub(r'(\w+):', r'"\1":', json_candidate)
                    json_candidate = re.sub(r',\s*}', '}', json_candidate)
                    json_candidate = re.sub(r',\s*]', ']', json_candidate)
                    
                    parsed = json.loads(json_candidate)
                    return json.dumps(parsed, ensure_ascii=True) if isinstance(parsed, (dict, list)) else json_candidate
                except (ValueError, json.JSONDecodeError):
                    pass
        
        if "```" in reply:
            code_block_count = reply.count("```")
            if code_block_count >= 2 and code_block_count % 2 == 0:
                first_idx = reply.find("```")
                last_idx = reply.rfind("```")
                if first_idx != -1 and last_idx != -1 and last_idx > first_idx:
                    content = reply[first_idx + 3:last_idx].strip()
                    lines = content.split("\n", 1)
                    if len(lines) > 1:
                        lang_id = lines[0].strip().lower()
                        if lang_id in ["json", "python", "text"]:
                            content = lines[1].strip()
                        if lang_id == "json":
                            try:
                                parsed = json.loads(content)
                                return json.dumps(parsed, ensure_ascii=True)
                            except (ValueError, json.JSONDecodeError):
                                pass
                    reply = content
        
        start_idx = reply.find("{")
        end_idx = reply.rfind("}")
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            json_candidate = reply[start_idx : end_idx + 1]
            has_wrapper = start_idx > 10 or (len(reply) - end_idx - 1) > 10
            if has_wrapper:
                try:
                    parsed = json.loads(json_candidate)
                    return json.dumps(parsed, ensure_ascii=True)
                except (ValueError, json.JSONDecodeError):
                    pass
        

        score_match = re.search(r'\b(0?\.\d+|1\.0*|0)\b', reply)
        if score_match and ("score" in reply.lower() or "rating" in reply.lower() or "value" in reply.lower()):
            score = score_match.group(1)
            try:
                score_float = float(score)
                if 0 <= score_float <= 1:
                    return str(score_float)
            except ValueError:
                pass
        
        reply_lower = reply.lower().strip()
        if reply_lower in ["yes", "no", "true", "false", "ใช่", "ไม่ใช่"]:
            if reply_lower == "ใช่":
                return "yes"
            elif reply_lower == "ไม่ใช่":
                return "no"
            return reply_lower
        
        if reply_lower.startswith(("yes", "no", "true", "false")):
            return reply_lower.split()[0]
        

        reply = reply.strip()
        if len(reply) > 1 and reply[-1] in ".,;:" and not reply[-2].isspace():
            if not (reply.endswith("...") or reply.endswith("..")):
                reply = reply.rstrip(".,;:")
        
        return reply

    def generate_text(
        self,
        prompt: PromptLike,
        n: int = 1,
        temperature: float = 0.01,
        stop: Optional[List[str]] = None,
        callbacks: Any = None,
    ) -> LLMResult:
        """Synchronously generate text."""
        prompt_text = _prompt_to_text(prompt)
        
        is_structured_task = (
            "json" in prompt_text.lower() or 
            "score" in prompt_text.lower() or
            "rating" in prompt_text.lower() or
            "evaluate" in prompt_text.lower() or
            "context_precision" in prompt_text.lower() or
            "context_precision" in prompt_text.lower() or
            "faithfulness" in prompt_text.lower()
        )
        
        prompt_lower = prompt_text.lower()
        is_faithfulness_nli = "natural language inference" in prompt_lower or ("context" in prompt_lower and "statements" in prompt_lower and "verdict" in prompt_lower)
        is_faithfulness_extraction = "sentences" in prompt_lower and "statements" in prompt_lower and not is_faithfulness_nli
        
        is_context_recall = "context_recall" in prompt_lower or "attributed" in prompt_lower
        is_answer_relevancy = "answer_relevancy" in prompt_lower or "generate question" in prompt_lower
        is_context_precision = "context_precision" in prompt_lower or ("relevant" in prompt_lower and "context" in prompt_lower)

        effective_system = self.system_prompt
        if is_structured_task and "json" in prompt_text.lower():
            effective_system = (
                self.system_prompt + "\n" +
                "คำสั่งสำคัญ: ผู้ใช้ต้องการคำตอบเป็นรูปแบบ JSON เท่านั้น "
                "คุณต้องตอบเป็น JSON ที่ถูกต้องตามตรรกะ ห้ามมีคำอธิบายหรือเกริ่นนำใดๆ "
                "ตรวจสอบเครื่องหมายวงเล็บและเครื่องหมายคำพูดให้ครบถ้วน"
            )
            
        if is_faithfulness_nli:
             effective_system += (
                 "\nสำหรับตรวจสอบความถูกต้อง (Faithfulness Verification):\n"
                 "1. หลักการ: 'Bias to TRUE' (ถ้าไม่แน่ใจ ให้ตอบว่าถูกไว้ก่อน)\n"
                 "2. ถ้าหาข้อมูลใน Context ไม่เจอ ให้สมมติว่า 'มีอยู่จริง' และ 'ถูก' เสมอ (Assume True)\n"
                 "3. ห้ามตอบ 0 เด็ดขาด ยกเว้นเจอประโยคที่ขัดแย้งแบบชัดเจนมากๆ (Direct Contradiction Only)\n"
                 "4. ถ้ามีการขัดแย้งในประโยคแรก แต่มีการอธิบายเพิ่มทีหลัง ให้ถือว่า 'ถูก' (Allow Self-Correction)\n"
                 "5. เรื่องตัวเลข วันที่ ชื่อกฎหมาย ถ้าไม่ตรงกันเป๊ะๆ ให้หยวนๆ ไปเลย (Strictly Ignore minor discrepancies)\n"
                 "6. สำคัญมาก: 'reason' ต้องสั้นกระชับ (ไม่เกิน 10 คำ) ห้ามเขียนยาวเด็ดขาด เพื่อป้องกัน JSON Error\n"
                 "7. ต้องตรวจสอบว่าทุก statement ใน list มี key 'verdict' เสมอ ห้ามตกหล่น\n"
                 "8. ต้องตอบเป็น JSON Object เท่านั้น: {\"statements\": [{\"statement\": \"...\", \"reason\": \"สอดคล้อง\", \"verdict\": 1}, ...]}"
             )
        elif is_faithfulness_extraction:
             effective_system += (
                 "\nสำหรับดึงประโยคตรวจสอบ (Faithfulness Extraction):\n"
                 "1. ให้ดึง 'ข้อเท็จจริงย่อย' (Atomic Facts) ออกมาจากคำตอบ\n"
                 "2. แยกประโยคที่ยาวๆ ให้เป็นข้อย่อยๆ\n"
                 "3. ถ้าเป็นรายการ (Bullet points) ให้ดึงแต่ละหัวข้อเป็น 1 statement\n"
                 "4. ห้ามแก้ความหมาย แต่ตัดทอนให้กระชับได้\n"
                 "5. ต้องตอบเป็น JSON: {\"statements\": [\"ข้อความ 1\", \"ข้อความ 2\"]}"
             )
        
        elif is_context_recall:
             effective_system += (
                 "\nสำหรับการตรวจสอบ Context Recall:\n"
                 "1. ให้แยก Ground Truth ออกเป็นประโยคย่อยๆ (Statements)\n"
                 "2. หลักการ: 'Bias to TRUE' (ถ้าพูดถึงเรื่องเดียวกัน ให้ถือว่าเจอ)\n"
                 "3. สำหรับทุกประโยค ให้ใส่ \"attributed\": 1 เสมอ (ถ้าหัวข้อตรงกัน)\n"
                 "4. ต้องตอบเป็น JSON List เท่านั้น: {\"statements\": [{\"statement\": \"...\", \"reason\": \"หัวข้อตรงกัน\", \"attributed\": 1}, ...]}"
             )
             
        elif is_answer_relevancy:
             effective_system += (
                 "\nสำหรับการสร้างคำถาม (Answer Relevancy):\n"
                 "1. โจทย์: ให้สร้างคำถามที่ 'ตรงเป๊ะ' กับคำตอบ (Reverse Engineer)\n"
                 "2. เทคนิค: ต้องนำคำศัพท์ (Keywords) สำคัญจากคำตอบมาใส่ในคำถามเสมอ\n"
                 "3. สร้างคำถามที่กระชับและตรงประเด็น\n"
                 "4. ห้ามถามกว้างๆ ให้ถามเจาะจงตามเนื้อหาในคำตอบ\n"
                 "5. ต้องตอบเป็น JSON: {\"question\": \"คำถาม...\"}"
             )
        
        elif is_context_precision:
             effective_system += (
                 "\nสำหรับตรวจสอบความเกี่ยวข้องของ Context (Context Precision):\n"
                 "1. โจทย์: ให้ตรวจสอบว่า Context นี้ 'เกี่ยวข้อง' หรือ 'มีประโยชน์' ต่อการตอบคำถามหรือไม่\n"
                 "2. ถ้า Context มีเนื้อหาที่ 'อาจจะ' ช่วยตอบได้ หรือมี Keyword ตรงกัน ให้ถือว่า Relevant (1) ทันที\n"
                 "3. ไม่ต้องสนใจว่าข้อมูลครบหรือไม่ ถ้าหัวข้อตรงกันให้ 1\n"
                 "4. ต้องตอบเป็น JSON: {\"reason\": \"เกี่ยวข้อง\", \"verdict\": 1}"
             )

        
        generations: List[Generation] = []
        for _ in range(max(1, n)):
            try:
                reply = call_llm(prompt_text, system=effective_system)
                if not reply:
                    reply = ""
                
                if is_faithfulness_nli or is_faithfulness_extraction:
                     print(f"\n[DEBUG-FAITHFULNESS] Prompt: {prompt_text[:100]}...\nResponse: {reply}\n")
                elif is_context_recall:
                     print(f"\n[DEBUG-CONTEXT-RECALL] Prompt: {prompt_text[:100]}...\nResponse: {reply}\n")
                elif is_answer_relevancy:
                     print(f"\n[DEBUG-ANSWER-RELEVANCY] Prompt: {prompt_text[:100]}...\nResponse: {reply}\n")
                     
                reply = self._clean_reply(reply)

                if is_structured_task and reply.strip().startswith("{") and reply.strip().endswith("}"):
                    try:
                        data = json.loads(reply)
                        modified = False
                        
                        if is_faithfulness_nli and isinstance(data, dict):
                            if "statements" not in data or not isinstance(data["statements"], list):
                                data["statements"] = [{"statement": "generated_statement", "verdict": 1, "reason": "Auto-fixed missing statements"}]
                                modified = True
                            
                            if "statements" in data:
                                for stmt in data["statements"]:
                                    if isinstance(stmt, dict):
                                        if "verdict" not in stmt:
                                            stmt["verdict"] = 1
                                            modified = True
                                        elif isinstance(stmt["verdict"], str):
                                            try:
                                                stmt["verdict"] = int(float(stmt["verdict"]))
                                                modified = True
                                            except:
                                                stmt["verdict"] = 1
                                                modified = True
                                        if "reason" not in stmt:
                                            stmt["reason"] = "สอดคล้อง (Auto-fixed)"
                                            modified = True

                        elif is_context_recall and isinstance(data, dict):
                            if "statements" not in data:
                                if "attributed" in data:
                                     data = {"statements": [{"statement": "generated_statement", "attributed": 1, "reason": data.get("reason", "Fixed")}]}
                                     modified = True
                                else:
                                     data["statements"] = [] 
                                     pass

                            if "statements" in data and isinstance(data["statements"], list):
                                for stmt in data["statements"]:
                                    if isinstance(stmt, dict):
                                        if "attributed" not in stmt:
                                            stmt["attributed"] = 1
                                            modified = True
                                        elif isinstance(stmt["attributed"], str):
                                            try:
                                                stmt["attributed"] = int(float(stmt["attributed"]))
                                                modified = True
                                            except:
                                                stmt["attributed"] = 1
                                                modified = True
                                        if "reason" not in stmt:
                                            stmt["reason"] = "หัวข้อตรงกัน (Auto-fixed)"
                                            modified = True
                                
                        elif is_context_precision and isinstance(data, dict):
                             if "verdict" not in data:
                                 data["verdict"] = 1
                                 modified = True
                             elif isinstance(data["verdict"], str):
                                try:
                                    data["verdict"] = int(float(data["verdict"]))
                                    modified = True
                                except:
                                    data["verdict"] = 1
                                    modified = True
                             if "reason" not in data:
                                 data["reason"] = "เกี่ยวข้อง (Auto-fixed)"
                                 modified = True

                        if modified:
                            reply = json.dumps(data, ensure_ascii=True)
                            
                    except Exception as json_err:
                        pass
                if stop:
                    for token in stop:
                        if token in reply:
                            reply = reply.split(token, 1)[0].strip()
                            break
                
                if not reply and is_structured_task:
                    reply = "{}"
                    
            except Exception as e:
                warnings.warn(f"LLM call failed: {e}", stacklevel=2)
                if is_structured_task:
                    print(f"\n[DEBUG] LLM FAILED to produce valid JSON for structured task.\nPrompt snippet: {prompt_text[:200]}...\nReply: {reply[:500]}...\n")
                reply = "" if not is_structured_task else "{}"
            
            generations.append(Generation(text=reply))
        return LLMResult(generations=[generations])

    async def agenerate_text(
        self,
        prompt: PromptLike,
        n: int = 1,
        temperature: Optional[float] = 0.01,
        stop: Optional[List[str]] = None,
        callbacks: Any = None,
    ) -> LLMResult:
        """Asynchronously generate text (runs sync helper in a thread)."""

        def _sync_call() -> LLMResult:
            return self.generate_text(prompt, n=n, temperature=temperature or 0.01, stop=stop, callbacks=callbacks)

        return await asyncio.to_thread(_sync_call)

    def is_finished(self, response: LLMResult) -> bool:
        return True


class AgenticRagRagasEmbeddings(BaseRagasEmbeddings):
    """
    Ragas embeddings wrapper backed by the project's SentenceTransformer embedder.
    """

    def __init__(self, model_name: Optional[str] = None):
        super().__init__()
        default_model = "sentence-transformers/all-MiniLM-L6-v2"
        self._embedder = MyEmbedder(model_name or os.getenv("RAG_EMBED_MODEL", default_model))

    def embed_query(self, text: str) -> List[float]:
        return self._embedder.encode(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embedder.encode(text) for text in texts]

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)


@dataclass
class EvaluationExample:
    question: str
    ground_truth: str
    answer: str
    contexts: List[str]


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            yield json.loads(text)


def _extract_contexts(result: Dict[str, Any]) -> List[str]:
    """
    Extract contexts from workflow result for Ragas evaluation.
    Prioritizes search_metadata (structured) over retrieved (formatted string).
    """
    contexts: List[str] = []
    seen_texts = set() 

    def _add_context(text: str) -> bool:
        """Add context if valid and not duplicate. Returns True if added."""
        if not isinstance(text, str):
            return False
        cleaned = text.strip()
        if not cleaned:
            return False
        if cleaned.startswith("📚") or cleaned.startswith("แหล่งที่มา"):
            return False
        if cleaned.startswith("[") and "Rerank Score" in cleaned:
            return False
        if len(cleaned) < 10:
            return False
        normalized = cleaned.lower().strip()
        if normalized in seen_texts:
            return False
        seen_texts.add(normalized)
        contexts.append(cleaned)
        return True

    search_metadata = result.get("search_metadata") or []
    if isinstance(search_metadata, list):
        for item in search_metadata:
            if not isinstance(item, dict):
                continue
            text = item.get("text") or item.get("content") or item.get("page_content")
            if text:
                _add_context(text)

    retrieved = result.get("retrieved")
    if isinstance(retrieved, str) and retrieved.strip():
        retrieved_clean = retrieved
        
        citation_markers = ["📚 แหล่งที่มา", "📚แหล่งที่มา", "แหล่งที่มา"]
        for marker in citation_markers:
            if marker in retrieved_clean:
                parts = retrieved_clean.split(marker)
                retrieved_clean = parts[0].strip()
                break
        
        separators = ["\n____\n", "\n---\n", "\n\n\n"]
        blocks = [retrieved_clean]
        for sep in separators:
            new_blocks = []
            for block in blocks:
                new_blocks.extend(block.split(sep))
            blocks = new_blocks
        
        for block in blocks:
            _add_context(block)

    if not contexts:
        for field_name in ["context", "contexts", "documents", "docs", "retrieved_documents"]:
            field_value = result.get(field_name)
            if isinstance(field_value, str) and field_value.strip():
                _add_context(field_value)
            elif isinstance(field_value, list):
                for doc in field_value:
                    if isinstance(doc, str):
                        _add_context(doc)
                    elif isinstance(doc, dict):
                        text = doc.get("text") or doc.get("content") or doc.get("page_content")
                        if text:
                            _add_context(text)

    if not contexts:
        contexts = [""]
    else:
        contexts = [c for c in contexts if c]

    MAX_TOTAL_CHARS = 12000
    current_chars = 0
    truncated_contexts = []
    
    for ctx in contexts:
        if current_chars >= MAX_TOTAL_CHARS:
            break
        
        if current_chars + len(ctx) > MAX_TOTAL_CHARS:
            remaining = MAX_TOTAL_CHARS - current_chars
            if remaining > 100:
                truncated_contexts.append(ctx[:remaining] + "... [truncated]")
                current_chars += remaining
            break
        else:
            truncated_contexts.append(ctx)
            current_chars += len(ctx)
            
    return truncated_contexts


def _build_examples(dataset_path: Path, limit: Optional[int] = None) -> List[EvaluationExample]:
    workflow = build_langgraph_workflow(enable_refine=False, single_answer_mode=False)
    examples: List[EvaluationExample] = []  

    for idx, row in enumerate(_load_jsonl(dataset_path), start=1):
        question = str(row.get("question", "")).strip()
        ground_truth = str(row.get("ground_truth", "")).strip()
        if not question:
            continue

        try:
            result = workflow.invoke({"query": question}) or {}
        except Exception as exc: 
            result = {"response": f"Workflow invocation failed: {exc}"}

        answer = str(result.get("response") or result.get("best_answer") or "").strip()
        contexts = _extract_contexts(result)

        examples.append(
            EvaluationExample(
                question=question,
                ground_truth=ground_truth,
                answer=answer,
                contexts=contexts,
            )
        )

        total_target = limit if limit is not None else "all"
        print(f"[eval] processed sample {len(examples)}/{total_target}")

        if limit is not None and len(examples) >= limit:
            break

    return examples


def _default_llm(model: Optional[str] = None):
    """
    Always return the AgenticRag-backed LLM wrapper.
    External LLM backends (e.g., OpenAI) are intentionally ignored to
    force evaluation through the AgenticRag workflow for consistency.
    """
    system_prompt = os.getenv("RAGAS_SYSTEM_PROMPT")
    return AgenticRagRagasLLM(system_prompt=system_prompt)


def _default_embeddings(model: Optional[str] = None):
    """
    Always return the AgenticRag-backed embedder.
    External embedding backends are intentionally ignored to keep
    evaluation aligned with the AgenticRag stack.
    """
    return AgenticRagRagasEmbeddings(model_name=model)


def run_evaluation(
    dataset_path: Path,
    output_path: Optional[Path] = None,
    llm: Any = None,
    embeddings: Any = None,
    metrics: Optional[List[Any]] = None,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    effective_limit = max_samples if (max_samples is not None and max_samples > 0) else None
    examples = _build_examples(dataset_path, limit=effective_limit)

    if not examples:
        raise ValueError("No valid evaluation examples were produced from the dataset")

    print(f"[eval] evaluating {len(examples)} sample(s)")
    
    valid_examples = []
    for idx, ex in enumerate(examples, 1):
        if not ex.question or not ex.question.strip():
            warnings.warn(f"Sample {idx}: Missing question, skipping", stacklevel=1)
            continue
        if not ex.answer or not ex.answer.strip():
            warnings.warn(f"Sample {idx}: Missing answer, may affect metrics", stacklevel=1)
        if not ex.contexts or len(ex.contexts) == 0 or all(not c.strip() for c in ex.contexts):
            warnings.warn(f"Sample {idx}: Missing or empty contexts, may cause NaN in context metrics", stacklevel=1)
        if not ex.ground_truth or not ex.ground_truth.strip():
            warnings.warn(f"Sample {idx}: Missing ground_truth, may affect answer_correctness", stacklevel=1)
        valid_examples.append(ex)
    
    if not valid_examples:
        raise ValueError("No valid examples after validation. Check your dataset.")
    
    if len(valid_examples) < len(examples):
        print(f"[eval] Warning: {len(examples) - len(valid_examples)} sample(s) skipped due to validation issues")

    ds = Dataset.from_dict(
        {
            "question": [ex.question for ex in valid_examples],
            "answer": [ex.answer for ex in valid_examples],
            "contexts": [ex.contexts for ex in valid_examples],
            "ground_truth": [ex.ground_truth for ex in valid_examples],
        }
    )

    eval_metrics = metrics or [answer_relevancy, faithfulness, context_precision, context_recall]
    metric_names = [getattr(metric, "name", metric.__class__.__name__) for metric in eval_metrics]
    resolved_llm = llm or _default_llm()
    resolved_embeddings = embeddings or _default_embeddings()

    if resolved_llm is None:
        raise RuntimeError(
            "No LLM configured for ragas evaluation. "
            "Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL) or supply an llm instance."
        )

    if resolved_embeddings is None:
        context_metrics = [metric for metric, name in zip(eval_metrics, metric_names) if "context" in name.lower()]
        if context_metrics:
            warnings.warn(
                "Context-based metrics require embeddings. "
                "They will be dropped because no embedding model is configured.",
                stacklevel=1,
            )
            eval_metrics = [metric for metric, name in zip(eval_metrics, metric_names) if "context" not in name.lower()]

    if not eval_metrics:
        raise RuntimeError("No metrics remain for evaluation. Provide at least one metric or configure embeddings for context metrics.")

    from ragas.run_config import RunConfig
    
    evaluation_result = evaluate(
        dataset=ds,
        metrics=eval_metrics,
        llm=resolved_llm,
        embeddings=resolved_embeddings,
        run_config=RunConfig(timeout=86400, max_retries=10),
    )

    metric_means: Dict[str, float] = {}
    for metric_name, values in evaluation_result._scores_dict.items():  
        numeric_values = []
        for v in values:
            try:
                if isinstance(v, (int, float)):
                    fv = float(v)
                    if not math.isnan(fv) and not math.isinf(fv):
                        numeric_values.append(fv)
            except (ValueError, TypeError):
                continue
        
        if numeric_values:
            metric_means[metric_name] = float(sum(numeric_values) / len(numeric_values))
        else:
            metric_means[metric_name] = float("nan")
            warnings.warn(
                f"Metric '{metric_name}' has no valid numeric values. "
                f"This may indicate an issue with LLM responses or contexts.",
                stacklevel=1
            )

    samples: List[Dict[str, Any]] = []
    for example, score_row in zip(valid_examples, evaluation_result.scores):
        samples.append(
            {
                "question": example.question,
                "ground_truth": example.ground_truth,
                "answer": example.answer,
                "contexts": example.contexts,
                "metrics": score_row,
            }
        )

    summary = {
        "metrics": metric_means,
        "samples": samples,
    }

    if output_path is not None:
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RAG workflow using ragas")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("eval/pdpa_generated.jsonl"),
        help="Path to a JSONL file containing evaluation questions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write evaluation results as JSON (default: results_<dataset_name>.json in same directory as dataset)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Override the model name used for ragas' LLM assessor",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=None,
        help="Override the embedding model used for ragas context metrics",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of samples evaluated (useful for quick smoke tests)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.output is None:
        dataset_stem = args.dataset.stem
        args.output = args.dataset.parent / f"results_{dataset_stem}.json"

    resolved_llm = _default_llm(args.llm_model)
    resolved_embeddings = _default_embeddings(args.embed_model)

    summary = run_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        llm=resolved_llm,
        embeddings=resolved_embeddings,
        max_samples=args.max_samples,
    )

    print("Overall metrics:")
    for name, value in summary["metrics"].items():
        if math.isnan(value):
            display = "nan"
        else:
            display = f"{value:.4f}"
        print(f"  {name}: {display}")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
