import os
import yaml
from .tools.custom_tool import DocumentSearchTool
from .tools.qdrant_storage import QdrantStorage, MyEmbedder
from langgraph.graph import StateGraph
import logging
from typing import Dict, Any
from .tools.security_filter import SecurityFilter

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logging.warning("OpenAI client not installed. Please install with 'pip install openai'.")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.opentyphoon.ai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "typhoon-v2.1-12b-instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("TYPHOON_API_KEY") or os.getenv("OPENAI_API_KEY") or "not-needed"

AGENTS_YAML = os.path.join(os.path.dirname(__file__), 'config', 'agents.yaml')
TASKS_YAML = os.path.join(os.path.dirname(__file__), 'config', 'tasks.yaml')

def call_llm(prompt, system=None):
    if OpenAI is None:
        raise ImportError("OpenAI client not installed. Run: pip install openai")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=8192
    )
    return response.choices[0].message.content


def build_langgraph_workflow(pdf_tool=None, use_knowledge_base=True, enable_refine: bool = True, single_answer_mode: bool = False):
    with open(AGENTS_YAML, 'r', encoding='utf-8') as f:
        agents_config = yaml.safe_load(f)
    with open(TASKS_YAML, 'r', encoding='utf-8') as f:
        tasks_config = yaml.safe_load(f)

    security_filter = SecurityFilter()

    def append_progress(state, message):
        progress = state.get("progress_log", [])
        return progress + [message]

    def refine_question_node(state):
        query = state.get("query", "")
        context = state.get("context", "")
        progress_log = state.get("progress_log", [])

        try:
            filter_result = security_filter.filter_user_input(query or "")
        except Exception:
            filter_result = {"should_respond": False, "response_message": "เกิดข้อผิดพลาดในการตรวจสอบความปลอดภัยของข้อความ กรุณาลองใหม่อีกครั้ง"}

        if not filter_result.get("should_respond", True):
            progress_log = append_progress({"progress_log": progress_log}, "🔴 [Guardrail] บล็อกคำถามเนื่องจากพบคำหยาบ/ไม่เหมาะสม")
            warn_msg = filter_result.get("response_message") or "ตรวจพบเนื้อหาไม่เหมาะสมในคำถาม ⚠️ กรุณาพิมพ์ใหม่โดยใช้ถ้อยคำที่สุภาพ"
            return {**state, "response": warn_msg, "best_answer": "", "blocked": True, "progress_log": progress_log}

        progress_log = append_progress({"progress_log": progress_log}, "🟡 [LangGraph] กำลังปรับคำถาม (Refining question)...")
        system = agents_config['question_refiner_agent']['role'] + "\n" + agents_config['question_refiner_agent']['goal']
        
        if context and context.strip():
            prompt = (
                f"ช่วย 'ปรับคำถาม' ให้ชัดเจนและเฉพาะเจาะจงขึ้น โดยไม่ตอบคำถาม\n"
                f"บริบทก่อนหน้า:\n{context}\n"
                f"คำถามปัจจุบัน: {query}\n\n"
                f"ข้อกำหนดการตอบ:\n"
                f"- ตอบเพียง 1 บรรทัด เป็นคำถามที่ปรับแล้วเท่านั้น\n"
                f"- ห้ามอธิบายเพิ่มเติม ห้ามใส่ bullet ห้ามสรุปหรือยกตัวอย่าง\n"
                f"- ห้ามตอบเนื้อหาของคำถาม ให้ทำแค่ปรับถ้อยคำของคำถาม\n"
                f"- ภาษาไทยเท่านั้น"
            )
        else:
            prompt = (
                f"ช่วย 'ปรับคำถาม' ให้ชัดเจนและเฉพาะเจาะจงขึ้น โดยไม่ตอบคำถาม\n"
                f"คำถาม: {query}\n\n"
                f"ข้อกำหนดการตอบ:\n"
                f"- ตอบเพียง 1 บรรทัด เป็นคำถามที่ปรับแล้วเท่านั้น\n"
                f"- ห้ามอธิบายเพิ่มเติม ห้ามใส่ bullet ห้ามสรุปหรือยกตัวอย่าง\n"
                f"- ห้ามตอบเนื้อหาของคำถาม ให้ทำแค่ปรับถ้อยคำของคำถาม\n"
                f"- ภาษาไทยเท่านั้น"
            )

        refined_raw = call_llm(prompt, system=system)

        try:
            import re
            first_line = refined_raw.strip().splitlines()[0]
            first_line = re.sub(r"^(คำถามที่ปรับแล้ว|คำถามที่ชัดเจน|คำถาม|Refined Question|Refined)[:\s-]*", "", first_line, flags=re.I)
            refined = first_line.strip()
        except Exception:
            refined = refined_raw.strip()

        try:
            score_system = (
                "คุณเป็นผู้ประเมินคุณภาพคำถาม ให้คะแนนความชัดเจน ความเฉพาะเจาะจง และความพร้อมต่อการค้นข้อมูล "
                "บนสเกล 0-100 ตอบเป็น JSON เท่านั้น เช่น {\"score\": 87, \"reason\": \"ชัดเจนและเฉพาะเจาะจง\"}"
            )
            score_prompt = (
                f"โปรดประเมินคำถามที่ปรับแล้วต่อไปนี้:\n\n{refined}\n\n"
                f"ให้คะแนนและเหตุผลสั้นๆ"
            )
            score_raw = call_llm(score_prompt, system=score_system)
            import json
            score_obj = {}
            try:
                score_obj = json.loads(score_raw)
            except Exception:
                import re
                m = re.search(r"(\d{1,3})", score_raw)
                score_val = int(m.group(1)) if m else 0
                score_obj = {"score": score_val, "reason": score_raw.strip()[:200]}
            refined_score = int(score_obj.get("score", 0))
            refined_reason = str(score_obj.get("reason", "")).strip()
        except Exception:
            refined_score = None
            refined_reason = ""


        try:
            print("\n===== Refined Question =====")
            print(refined)
            print("===== Score =====")
            if refined_score is not None:
                print(f"Refined Question Score: {refined_score}/100")
                if refined_reason:
                    print(f"Reason: {refined_reason}")
            else:
                print("Scoring failed.")
            print("============================\n")
        except Exception:
            pass

        progress_log = append_progress({"progress_log": progress_log}, "🟢 [LangGraph] ปรับคำถามเสร็จแล้ว (Refined question)")
        return {**state, "refined_question": refined, "refined_question_score": refined_score, "progress_log": progress_log}

    def planning_node(state):
        progress_log = append_progress(state, "🟡 [LangGraph] กำลังวางแผน (Planning)...")
        refined = state.get("refined_question") or state.get("query", "")
        context = state.get("context", "")
        system = agents_config['planning_agent']['role'] + "\n" + agents_config['planning_agent']['goal']
        

        if context and context.strip():
            prompt = (
                f"เข้าใจแล้วครับ! เดี๋ยวช่วยวางแผนการตอบให้นะ 😊\n"
                f"จากบริบทการสนทนาที่ผ่านมา:\n{context}\n"
                f"คำถาม: {refined}\n"
                f"\nช่วยวางแผนการตอบคำถามนี้อย่างเป็นระบบ โดยใช้ภาษาที่เป็นกันเอง\n"
                f"เริ่มต้นด้วยการแสดงความเข้าใจในคำถาม เช่น 'เข้าใจแล้วครับ' หรือ 'คำถามดีมากเลย'\n"
                f"แบ่งออกเป็นขั้นตอนที่เข้าใจง่าย และใช้ emoji ที่เหมาะสม\n"
                f"โปรดตอบเป็นภาษาไทยเท่านั้น"
            )
        else:
            prompt = (
                f"เข้าใจแล้วครับ! เดี๋ยวช่วยวางแผนการตอบให้นะ 😊\n"
                f"คำถาม: {refined}\n"
                f"\nช่วยวางแผนการตอบคำถามนี้อย่างเป็นระบบ โดยใช้ภาษาที่เป็นกันเอง\n"
                f"เริ่มต้นด้วยการแสดงความเข้าใจในคำถาม เช่น 'เข้าใจแล้วครับ' หรือ 'คำถามดีมากเลย'\n"
                f"แบ่งออกเป็นขั้นตอนที่เข้าใจง่าย และใช้ emoji ที่เหมาะสม\n"
                f"โปรดตอบเป็นภาษาไทยเท่านั้น"
            )
            
        plan = call_llm(prompt, system=system)
        progress_log = append_progress({"progress_log": progress_log}, "🟢 [LangGraph] วางแผนเสร็จแล้ว (Planning done)")
        return {**state, "plan": plan, "progress_log": progress_log}

    def retrieval_node(state):
        progress_log = append_progress(state, "🟡 [LangGraph] กำลังค้นข้อมูล (Retrieving from PDF/Knowledge)...")
        query = state.get("query", "") 
        tool = pdf_tool if pdf_tool else DocumentSearchTool(file_path=os.path.join(os.path.dirname(__file__), '../../knowledge/pdpa.pdf'))
        
        search_results_with_metadata = tool.get_search_results_with_metadata(query)
        retrieved = tool._run(query)
        
        try:
            print("\n===== DocumentSearchTool Result (truncated) =====")
            if isinstance(retrieved, str):
                preview = retrieved[:2000]
                print(preview)
                if len(retrieved) > len(preview):
                    print(f"... [truncated, total {len(retrieved)} chars]")
            else:
                print(str(retrieved))
            
            if search_results_with_metadata:
                print("\n===== Rerank Scores =====")
                for i, result in enumerate(search_results_with_metadata):
                    rerank_score = result.get('rerank_score', None)
                    if rerank_score is not None:
                        source_file = result.get('source_file', 'ไม่ระบุไฟล์')
                        page_number = result.get('page_number', 'ไม่ระบุหน้า')
                        print(f"[{i+1}] {source_file}, หน้า {page_number} - Rerank Score: {rerank_score:.4f} SBERT")
                print("===== End Rerank Scores =====\n")
            
            print("===== End DocumentSearchTool Result =====\n")
        except Exception as _:
            pass
        
        progress_log = append_progress({"progress_log": progress_log}, "🟢 [LangGraph] ค้นข้อมูลเสร็จแล้ว (Retrieval done)")
        return {
            **state, 
            "retrieved": retrieved, 
            "retrieval_source": "pdf", 
            "search_metadata": search_results_with_metadata,
            "progress_log": progress_log
        }

    def judge_info_node(state):
        progress_log = append_progress(state, "🟡 [LangGraph] LLM ประเมินความเพียงพอของข้อมูล (Judging info sufficiency)...")
        refined = state.get("refined_question") or state.get("query", "")
        context = state.get("retrieved", "")

        if not context or context.strip() in ["ไม่พบผลลัพธ์ที่เกี่ยวข้อง", "โปรดตั้งคำถามเฉพาะเกี่ยวกับ PDPA เท่านั้น"]:
            print("🟡 [LangGraph] ข้อมูลไม่เพียงพอ - ใช้ข้อมูลที่มีเท่านั้น")
            return {**state, "info_sufficient": False, "judge_reason": "ข้อมูลไม่เพียงพอ (web search ถูกนำออก)", "progress_log": progress_log}

        system = "คุณเป็นผู้ช่วยที่เชี่ยวชาญในการประเมินความครบถ้วนของข้อมูลสำหรับการตอบคำถาม"
        prompt = (
            f"คำถาม: {refined}\n"
            f"ข้อมูลที่ค้นพบ: {context}\n"
            f"ข้อมูลนี้เพียงพอสำหรับการตอบคำถามหรือไม่?\n"
            f"ถ้าเพียงพอ ตอบว่า 'เพียงพอ'\nถ้าไม่เพียงพอ ตอบว่า 'ไม่เพียงพอ' และระบุว่าข้อมูลขาดอะไร\nโปรดตอบเป็นภาษาไทยเท่านั้น"
        )
        judge = call_llm(prompt, system=system)
        progress_log = append_progress({"progress_log": progress_log}, f"🟢 [LangGraph] LLM ประเมินแล้ว: {judge.strip()}")
        is_sufficient = ('เพียงพอ' in judge and 'ไม่เพียงพอ' not in judge)
        return {**state, "info_sufficient": is_sufficient, "judge_reason": judge.strip(), "progress_log": progress_log}

    def generate_answers_node(state):
        if single_answer_mode:
            progress_log = append_progress(state, "🟡 [LangGraph] กำลังสร้างคำตอบ (Generating answer)...")
        else:
            progress_log = append_progress(state, "🟡 [LangGraph] กำลังสร้างคำตอบหลายแบบ (Generating multiple answers)...")
        refined = state.get("refined_question") or state.get("query", "")
        retrieved_context = state.get("retrieved", "")
        conversation_context = state.get("context", "")
        search_metadata = state.get("search_metadata", []) 
        system = agents_config['answer_candidate_agent']['role'] + "\n" + agents_config['answer_candidate_agent']['goal']

        num_candidates = 1 if single_answer_mode else 3
        candidates = []
        for i in range(num_candidates):
            source_info = ""
            if search_metadata:
                sources = []
                for j, metadata in enumerate(search_metadata[:3]):
                    source_file = metadata.get('source_file', 'ไม่ระบุไฟล์')
                    page_number = metadata.get('page_number', 'ไม่ระบุหน้า')
                    if source_file != 'ไม่ระบุไฟล์' and page_number != 'ไม่ระบุหน้า':
                        sources.append(f"[{j+1}] {source_file}, หน้า {page_number}")
                    elif source_file != 'ไม่ระบุไฟล์':
                        sources.append(f"[{j+1}] {source_file}")
                
                if sources:
                    source_info = f"\n\n📚 แหล่งที่มาของข้อมูล:\n" + "\n".join(sources)
            
            if conversation_context and conversation_context.strip():
                prompt = (
                    f"คำถาม: {refined}\n"
                    f"ข้อมูลที่ค้นพบ: {retrieved_context}\n"
                    f"บริบทการสนทนาก่อนหน้า: {conversation_context}\n"
                    f"{source_info}\n"
                    f"\nสร้างคำตอบที่:\n"
                    f"- ตอบคำถามโดยตรงและครบถ้วน\n"
                    f"- **สำคัญมาก**: วิเคราะห์คำถามเพื่อระบุแนวคิดหลัก (key concepts) ที่ควรมีในคำตอบ\n"
                    f"- **สำคัญมาก**: ตรวจสอบให้แน่ใจว่าคำตอบครอบคลุมแนวคิดหลักทั้งหมด\n"
                    f"- **ตัวอย่างแนวคิดหลัก**:\n"
                    f"  * ถ้าถามเกี่ยวกับ 'ฐานกฎหมาย' ต้องรวม: ความยินยอม, สัญญา, หรือประโยชน์โดยชอบธรรม\n"
                    f"  * ถ้าถามเกี่ยวกับ 'การแจ้งเหตุ' ต้องรวม: การประเมินเบื้องต้น, รายงานเท่าที่ทราบ\n"
                    f"- ใช้ข้อมูลจากบริบทที่ให้มาเท่านั้น พร้อมสังเคราะห์ให้ถูกต้องตามหลักวิจัย/กฎหมาย\n"
                    f"- ระบุหลักฐานตามมาตรา/หมวด PDPA ที่เกี่ยวข้อง\n"
                    f"- ตอบให้ครบประเด็น ชัดเจน เข้าใจง่าย ใช้ภาษาไทย และใส่ emoji นำหน้าหัวข้อสำคัญ\n"
                    f"- หากข้อมูลไม่เพียงพอ ให้บอกอย่างตรงไปตรงมาและระบุสิ่งที่ขาด\n"
                    f"\n**ห้ามใส่คำแนะนำการจัดรูปแบบหรือคำสั่งใดๆ ในคำตอบ**\n"
                )
            else:
                prompt = (
                    f"คำถาม: {refined}\n"
                    f"ข้อมูลที่ค้นพบ: {retrieved_context}\n"
                    f"{source_info}\n"
                    f"\nสร้างคำตอบที่:\n"
                    f"- ตอบคำถามโดยตรงและครบถ้วน\n"
                    f"- **สำคัญมาก**: วิเคราะห์คำถามเพื่อระบุแนวคิดหลัก (key concepts) ที่ควรมีในคำตอบ\n"
                    f"- **สำคัญมาก**: ตรวจสอบให้แน่ใจว่าคำตอบครอบคลุมแนวคิดหลักทั้งหมด\n"
                    f"- **ตัวอย่างแนวคิดหลัก**:\n"
                    f"  * ถ้าถามเกี่ยวกับ 'ฐานกฎหมาย' ต้องรวม: ความยินยอม, สัญญา, หรือประโยชน์โดยชอบธรรม\n"
                    f"  * ถ้าถามเกี่ยวกับ 'การแจ้งเหตุ' ต้องรวม: การประเมินเบื้องต้น, รายงานเท่าที่ทราบ\n"
                    f"- ใช้ข้อมูลจากบริบทที่ให้มาเท่านั้น พร้อมสังเคราะห์ให้ถูกต้องตามหลักวิจัย/กฎหมาย\n"
                    f"- ระบุหลักฐานตามมาตรา/หมวด PDPA ที่เกี่ยวข้อง\n"
                    f"- ตอบให้ครบประเด็น ชัดเจน เข้าใจง่าย ใช้ภาษาไทย และใส่ emoji นำหน้าหัวข้อสำคัญ\n"
                    f"- หากข้อมูลไม่เพียงพอ ให้บอกอย่างตรงไปตรงมาและระบุสิ่งที่ขาด\n"
                    f"\n**ห้ามใส่คำแนะนำการจัดรูปแบบหรือคำสั่งใดๆ ในคำตอบ**\n"
                )
            try:
                answer = call_llm(prompt, system=system).strip()
            except Exception as e:
                answer = f"ไม่สามารถสร้างคำตอบลำดับที่ {i+1} ได้: {e}"
            candidates.append(answer)

        if single_answer_mode:
            best_answer = candidates[0] if candidates else ""
            progress_log = append_progress({"progress_log": progress_log}, "🟢 [LangGraph] สร้างคำตอบเสร็จแล้ว (Answer ready)")
            return {**state, "candidates": candidates, "best_answer": best_answer, "ranked": candidates, "progress_log": progress_log}
        else:
            progress_log = append_progress({"progress_log": progress_log}, f"🟢 [LangGraph] สร้างคำตอบเสร็จแล้ว {len(candidates)} แบบ (Candidates ready)")
            return {**state, "candidates": candidates, "progress_log": progress_log}

    def decision_ranking_node(state):
        progress_log = append_progress(state, "🟡 [LangGraph] จัดอันดับคำตอบ (Ranking candidates)...")
        candidates = state.get("candidates", [])
        refined = state.get("refined_question") or state.get("query", "")
        if not candidates:
            progress_log = append_progress({"progress_log": progress_log}, "🟡 [LangGraph] ไม่มี candidates สำหรับจัดอันดับ")
            return {**state, "ranked": [], "best_answer": state.get("best_answer", ""), "progress_log": progress_log}

        system = agents_config['decision_ranking_agent']['role'] + "\n" + agents_config['decision_ranking_agent']['goal']
        
        prompt = (
            f"คำถาม: {refined}\n"
            f"คำตอบที่ต้องเลือก:\n"
            f"{chr(10).join([f'{i+1}. {candidate}' for i, candidate in enumerate(candidates)])}\n\n"
            f"เลือกคำตอบที่ดีที่สุดโดยพิจารณา:\n"
            f"1. **ความเกี่ยวข้อง**: คำตอบตอบคำถามโดยตรงหรือไม่ (สำคัญที่สุด)\n"
            f"2. **การครอบคลุมแนวคิดหลัก**: คำตอบครอบคลุมแนวคิดหลักที่ควรมีหรือไม่\n"
            f"   - ถ้าถามเกี่ยวกับ 'ฐานกฎหมาย' ต้องมี: ความยินยอม, สัญญา, หรือประโยชน์โดยชอบธรรม\n"
            f"   - ถ้าถามเกี่ยวกับ 'การแจ้งเหตุ' ต้องมี: การประเมินเบื้องต้น, รายงานเท่าที่ทราบ\n"
            f"3. **ความครบถ้วน**: ครอบคลุมประเด็นสำคัญของคำถามหรือไม่\n"
            f"4. **ความถูกต้อง**: ข้อมูลถูกต้องตาม PDPA หรือไม่\n"
            f"5. **ความกระชับ**: คำตอบกระชับเหมาะสมกับความซับซ้อนของคำถามหรือไม่\n"
            f"6. **ความชัดเจน**: เข้าใจง่าย ไม่วกวน ไม่มี meta-commentary\n"
            f"7. **การอ้างอิง**: มีการอ้างอิงแหล่งที่มาหรือไม่\n\n"
            f"**สำคัญมาก**: เลือกคำตอบที่ครอบคลุมแนวคิดหลัก (key concepts) ที่ควรมีในคำตอบ\n"
            f"**สำหรับคำถามง่ายๆ**: เลือกคำตอบที่สั้นๆ แต่ครอบคลุมแนวคิดหลัก\n"
            f"**สำหรับคำถามซับซ้อน**: เลือกคำตอบที่อธิบายชัดเจนและครอบคลุมทุกประเด็นสำคัญ\n"
            f"**หลีกเลี่ยงคำตอบที่มี**: คำแนะนำการจัดรูปแบบ, meta-commentary, หรือคำสั่งใดๆ\n"
            f"**หลีกเลี่ยงคำตอบที่**: ขาดแนวคิดหลักที่ควรมี (CRITICAL)\n\n"
            f"ตอบแค่หมายเลขของคำตอบที่ดีที่สุด (เช่น 1, 2, หรือ 3)\n"
            f"โปรดตอบเป็นภาษาไทยเท่านั้น"
        )
        
        try:
            ranking_result = call_llm(prompt, system=system).strip()
            import re
            number_match = re.search(r'(\d+)', ranking_result)
            if number_match:
                selected_index = int(number_match.group(1)) - 1
                if 0 <= selected_index < len(candidates):
                    best_answer = candidates[selected_index]
                    ranked = [best_answer] + [c for i, c in enumerate(candidates) if i != selected_index]
                else:
                    best_answer = candidates[0]
                    ranked = candidates
            else:
                best_answer = candidates[0]
                ranked = candidates
        except Exception as e:
            best_answer = candidates[0]
            ranked = candidates
        
        progress_log = append_progress({"progress_log": progress_log}, "🟢 [LangGraph] จัดอันดับคำตอบเสร็จแล้ว (Ranking done)")
        return {**state, "ranked": ranked, "candidates": ranked, "best_answer": best_answer, "progress_log": progress_log}

    def response_node(state):
        progress_log = append_progress(state, "🟡 [LangGraph] กำลังสรุปคำตอบ (Synthesizing response)...")
        ranked = state.get("ranked", [])
        best_answer = state.get("best_answer", "")
        conversation_context = state.get("context", "")
        refined = state.get("refined_question") or state.get("query", "")
        search_metadata = state.get("search_metadata", [])

        if best_answer:
            system = agents_config['response_synthesizer_agent']['role'] + "\n" + agents_config['response_synthesizer_agent']['goal']
            
            source_info = ""
            if search_metadata:
                sources = []
                for j, metadata in enumerate(search_metadata[:3]):
                    source_file = metadata.get('source_file', 'ไม่ระบุไฟล์')
                    page_number = metadata.get('page_number', 'ไม่ระบุหน้า')
                    if source_file != 'ไม่ระบุไฟล์' and page_number != 'ไม่ระบุหน้า':
                        sources.append(f"[{j+1}] {source_file}, หน้า {page_number}")
                    elif source_file != 'ไม่ระบุไฟล์':
                        sources.append(f"[{j+1}] {source_file}")
                
                if sources:
                    source_info = f"\n\n📚 แหล่งที่มาของข้อมูล:\n" + "\n".join(sources)
            
            if conversation_context and conversation_context.strip():
                prompt = (
                    f"คำตอบเดิม: {best_answer}\n"
                    f"คำถาม: {refined}\n"
                    f"บริบทการสนทนา: {conversation_context}\n"
                    f"{source_info}\n"
                    f"\nปรับปรุงคำตอบให้:\n"
                    f"- ตอบคำถามโดยตรงและครบถ้วน\n"
                    f"- **สำคัญมาก**: ตรวจสอบให้แน่ใจว่าคำตอบครอบคลุมแนวคิดหลัก (key concepts) ทั้งหมด\n"
                    f"- **สำคัญมาก**: อย่าลบแนวคิดหลักออกเมื่อทำให้คำตอบกระชับขึ้น\n"
                    f"- **ตัวอย่างแนวคิดหลัก**:\n"
                    f"  * ถ้าถามเกี่ยวกับ 'ฐานกฎหมาย' ต้องมี: ความยินยอม, สัญญา, หรือประโยชน์โดยชอบธรรม\n"
                    f"  * ถ้าถามเกี่ยวกับ 'การแจ้งเหตุ' ต้องมี: การประเมินเบื้องต้น, รายงานเท่าที่ทราบ\n"
                    f"- เก็บเฉพาะข้อมูลสำคัญที่ตอบคำถาม แต่ต้องรวมแนวคิดหลักทั้งหมด\n"
                    f"- ตอบให้ตรงคำถาม ชัดเจน เข้าใจง่าย อิงมาตรา/หมวด PDPA ที่เกี่ยวข้อง\n"
                    f"- ใช้หัวข้อสำคัญพร้อม emoji นำหน้าเพื่อให้อ่านง่าย\n"
                    f"- ลบคำแนะนำการจัดรูปแบบหรือคำสั่งใดๆ ออก\n"
                    f"- ลบ meta-commentary หรือคำอธิบายเกี่ยวกับการจัดรูปแบบออก\n"
                    f"- เริ่มต้นด้วยคำตอบหลักทันที ไม่ต้องทักทายหรือเกริ่นนำ\n"
                    f"- ไม่เพิ่มข้อมูลใหม่ที่ไม่มีในคำตอบเดิม และใช้ภาษาไทย\n"
                    f"\n**ตอบเฉพาะเนื้อหาคำตอบเท่านั้น ไม่ต้องใส่คำแนะนำหรือคำสั่งใดๆ**\n"
                )
            else:
                prompt = (
                    f"คำตอบเดิม: {best_answer}\n"
                    f"คำถาม: {refined}\n"
                    f"{source_info}\n"
                    f"\nปรับปรุงคำตอบให้:\n"
                    f"- ตอบคำถามโดยตรงและครบถ้วน\n"
                    f"- **สำคัญมาก**: ตรวจสอบให้แน่ใจว่าคำตอบครอบคลุมแนวคิดหลัก (key concepts) ทั้งหมด\n"
                    f"- **สำคัญมาก**: อย่าลบแนวคิดหลักออกเมื่อทำให้คำตอบกระชับขึ้น\n"
                    f"- **ตัวอย่างแนวคิดหลัก**:\n"
                    f"  * ถ้าถามเกี่ยวกับ 'ฐานกฎหมาย' ต้องมี: ความยินยอม, สัญญา, หรือประโยชน์โดยชอบธรรม\n"
                    f"  * ถ้าถามเกี่ยวกับ 'การแจ้งเหตุ' ต้องมี: การประเมินเบื้องต้น, รายงานเท่าที่ทราบ\n"
                    f"- เก็บเฉพาะข้อมูลสำคัญที่ตอบคำถาม แต่ต้องรวมแนวคิดหลักทั้งหมด\n"
                    f"- ตอบให้ตรงคำถาม ชัดเจน เข้าใจง่าย อิงมาตรา/หมวด PDPA ที่เกี่ยวข้อง\n"
                    f"- ใช้หัวข้อสำคัญพร้อม emoji นำหน้าเพื่อให้อ่านง่าย\n"
                    f"- ลบคำแนะนำการจัดรูปแบบหรือคำสั่งใดๆ ออก\n"
                    f"- ลบ meta-commentary หรือคำอธิบายเกี่ยวกับการจัดรูปแบบออก\n"
                    f"- เริ่มต้นด้วยคำตอบหลักทันที ไม่ต้องทักทายหรือเกริ่นนำ\n"
                    f"- ไม่เพิ่มข้อมูลใหม่ที่ไม่มีในคำตอบเดิม และใช้ภาษาไทย\n"
                    f"\n**ตอบเฉพาะเนื้อหาคำตอบเท่านั้น ไม่ต้องใส่คำแนะนำหรือคำสั่งใดๆ**\n"
                )
            response = call_llm(prompt, system=system)
        else:
            system = agents_config['response_synthesizer_agent']['role'] + "\n" + agents_config['response_synthesizer_agent']['goal']
            
            source_info = ""
            if search_metadata:
                sources = []
                for j, metadata in enumerate(search_metadata[:3]): 
                    source_file = metadata.get('source_file', 'ไม่ระบุไฟล์')
                    page_number = metadata.get('page_number', 'ไม่ระบุหน้า')
                    if source_file != 'ไม่ระบุไฟล์' and page_number != 'ไม่ระบุหน้า':
                        sources.append(f"[{j+1}] {source_file}, หน้า {page_number}")
                    elif source_file != 'ไม่ระบุไฟล์':
                        sources.append(f"[{j+1}] {source_file}")
                
                if sources:
                    source_info = f"\n\n📚 แหล่งที่มาของข้อมูล:\n" + "\n".join(sources)
            
            prompt = (
                f"คำตอบที่จัดอันดับแล้ว: {ranked}\n"
                f"{source_info}\n"
                f"\nเลือกคำตอบที่ดีที่สุดและปรับปรุงให้:\n"
                f"- ตอบคำถามโดยตรงและครบถ้วน\n"
                f"- ตรวจสอบให้แน่ใจว่าครอบคลุมแนวคิดหลักครบถ้วนและอิงหลักฐานจาก metadata (ชื่อไฟล์/หน้า) โดยเล่าชื่อแหล่งชัดเจน ไม่ต้องใส่ [1]/[2]/[3]\n"
                f"- ใช้หัวข้อสำคัญพร้อม emoji นำหน้าเพื่อให้อ่านง่าย\n"
                f"- ลบคำแนะนำการจัดรูปแบบหรือคำสั่งใดๆ ออก\n"
                f"- ลบ meta-commentary หรือคำอธิบายเกี่ยวกับการจัดรูปแบบออก\n"
                f"- ใช้ภาษาที่เข้าใจง่ายและชัดเจน เป็นภาษาไทย\n"
                f"\n**ตอบเฉพาะเนื้อหาคำตอบเท่านั้น ไม่ต้องใส่คำแนะนำหรือคำสั่งใดๆ**\n"
            )
            response = call_llm(prompt, system=system)

        progress_log = append_progress({"progress_log": progress_log}, "🟢 [LangGraph] สรุปคำตอบเสร็จแล้ว (Response ready)")
        return {
            **state, 
            "response": response, 
            "best_answer": best_answer, 
            "search_metadata": search_metadata, 
            "progress_log": progress_log
        }

    graph = StateGraph(Dict[str, Any])
    graph.add_node("refine_question", refine_question_node)
    graph.add_node("planning", planning_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("generate_answers", generate_answers_node)
    graph.add_node("decision_ranking", decision_ranking_node)
    graph.add_node("response", response_node)

    if enable_refine:
        graph.set_entry_point("refine_question")
        graph.add_conditional_edges(
            "refine_question",
            lambda state: "response" if state.get("blocked") else "planning"
        )
    else:
        graph.set_entry_point("planning")

    graph.add_edge("planning", "retrieval")
    graph.add_edge("retrieval", "generate_answers")
    if single_answer_mode:
        graph.add_edge("generate_answers", "response")
    else:
        graph.add_edge("generate_answers", "decision_ranking")
        graph.add_edge("decision_ranking", "response")
    graph.set_finish_point("response")

    return graph.compile()
