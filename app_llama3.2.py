import streamlit as st
import os
import uuid
import atexit
import signal
import gc
import base64
import time
import tempfile
import yaml
from dotenv import load_dotenv

load_dotenv()

try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

from src.agentic_rag.tools.custom_tool import DocumentSearchTool
from src.agentic_rag.crew import build_langgraph_workflow
try:
    from src.agentic_rag.tools.chat_history import ChatHistoryStore
except Exception:
    ChatHistoryStore = None

import base64
logo_base64 = base64.b64encode(open("assets/Typhoon2.png", "rb").read()).decode()

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

html, body, [class^='css'] {{
    font-family: 'Inter', 'Prompt', sans-serif !important;
}}

.stApp {{
    background: linear-gradient(120deg, #181c2f 0%, #232946 100%);
    color: #f3f6fa;
    min-height: 100vh;
}}

/* 🔹 Sidebar Styling - Hidden */
section[data-testid="stSidebar"] {{
    display: none !important;
}}

/* 🔹 File Uploader */
.stFileUploader {{
    background: #1d1f33;
    border: 1.5px dashed #8f94fb;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    color: #ccc;
    font-weight: 500;
}}

.stFileUploader button {{
    background: linear-gradient(90deg, #4e54c8, #8f94fb);
    color: white;
    font-weight: bold;
    padding: 10px 22px;
    border-radius: 12px;
    border: none;
    margin-top: 12px;
}}

.stFileUploader button:hover {{
    background: linear-gradient(90deg, #8f94fb, #4e54c8);
    transform: scale(1.03);
    transition: all 0.2s ease-in-out;
}}

/* 🔹 Reset Chat Button */
.stButton > button {{
    background: linear-gradient(90deg, #4e54c8, #8f94fb);
    color: white;
    font-weight: bold;
    padding: 10px 20px;
    border-radius: 14px;
    border: none;
    margin-top: 20px;
}}

.stButton > button:hover {{
    background: linear-gradient(90deg, #8f94fb, #4e54c8);
    transform: scale(1.03);
    box-shadow: 0 4px 14px rgba(78, 84, 200, 0.3);
}}

/* 🔹 Chat Messages */
.stChatMessage {{
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 20px 24px;
    margin: 12px 0;
    border: 1px solid rgba(255,255,255,0.08);
    color: #f3f6fa;
}}

.stChatMessage[data-testid="user"] {{
    background: linear-gradient(90deg, #4e54c8, #8f94fb);
    color: white;
    font-weight: 600;
    box-shadow: 0 6px 18px rgba(78, 84, 200, 0.2);
}}

.stChatMessage[data-testid="assistant"] {{
    background: rgba(36, 40, 59, 0.85);
    border: 1px solid rgba(255,255,255,0.06);
    color: #f3f6fa;
}}

/* 🔹 Input Bar Floating Bottom */
.stChatInputContainer {{
    position: fixed;
    left: 0 !important;
    right: 0 !important;
    bottom: 0;
    background: rgba(36, 40, 59, 0.92);
    padding: 16px 20px !important;
    box-shadow: 0 -2px 16px rgba(31, 38, 135, 0.18);
    z-index: 100;
    display: flex;
    justify-content: center;
    width: 100% !important;
}}

.stChatInputContainer > div {{
    width: 100% !important;
    max-width: 1200px !important;
    margin: 0 auto !important;
}}

.stTextInput {{
    width: 100% !important;
    max-width: 100% !important;
}}

/* 🔹 Scrollbar */
::-webkit-scrollbar {{
    width: 10px;
    background: #232526;
}}

::-webkit-scrollbar-thumb {{
    background: #4e54c8;
    border-radius: 8px;
}}

/* 🔹 Main Logo and Title */
.main-header {{
    text-align: center;
    margin: 50px 0 30px 0;
    position: relative;
}}

.main-logo {{
    width: 100px;
    margin-bottom: 16px;
    filter: drop-shadow(0 0 16px rgba(142, 148, 251, 0.5));
}}

.main-title {{
    font-size: 2.5rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 1.2px;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}}

/* 🔹 Main content full width */
.block-container {{
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
    padding-top: 1rem !important;
}}

/* 🔹 Adjust main content area when sidebar is hidden */
.main .block-container {{
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}}

/* 🔹 Expander Styling for Alternative Answers */
.streamlit-expanderHeader {{
    background: linear-gradient(90deg, #4e54c8, #8f94fb) !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    border: none !important;
    margin: 8px 0 !important;
    transition: all 0.3s ease !important;
}}

.streamlit-expanderHeader:hover {{
    background: linear-gradient(90deg, #8f94fb, #4e54c8) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(78, 84, 200, 0.3) !important;
}}

.streamlit-expanderContent {{
    background: rgba(36, 40, 59, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-top: 8px !important;
    backdrop-filter: blur(10px) !important;
}}

/* 🔹 Alternative Answer Styling */
.alternative-answer {{
    background: rgba(255, 255, 255, 0.03);
    border-left: 3px solid #8f94fb;
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 8px;
    border-top-left-radius: 0;
    border-bottom-left-radius: 0;
}}
</style>

<!-- 🔹 Injected Logo + Title -->
<div class="main-header">
    <img class="main-logo" src="data:image/png;base64,{logo_base64}" />
    <div class="main-title">PDPA Assistant</div>
</div>
""", unsafe_allow_html=True)


def is_pdpa_related(document_tool):
    """
    Checks if the uploaded file is related to PDPA by searching for PDPA-related terms in the document.
    
    Args:
        document_tool: The DocumentSearchTool instance initialized with the file
        
    Returns:
        bool: True if the file is likely PDPA-related, False otherwise
    """
    pdpa_keywords = [
        "PDPA", "Personal Data Protection Act", "คุ้มครองข้อมูลส่วนบุคคล", "พ.ร.บ. คุ้มครองข้อมูลส่วนบุคคล", 
        "ข้อมูลส่วนบุคคล", "data controller", "data processor", "ผู้ควบคุมข้อมูล", "ผู้ประมวลผลข้อมูล",
        "สิทธิเจ้าของข้อมูล", "การประมวลผลข้อมูล", "การเก็บรวบรวมข้อมูล", "ฐานทางกฎหมาย"
    ]
    
    if hasattr(document_tool, 'raw_text') and document_tool.raw_text:
        text = document_tool.raw_text.lower()
        for keyword in pdpa_keywords:
            if keyword.lower() in text:
                return True
    
    return False


def create_agents_and_tasks(pdf_tool, use_knowledge_base=True, file_query_mode=False):
    """สร้าง LangGraph workflow ที่ประกอบด้วย agent สำหรับค้นคว้าและสังเคราะห์คำตอบเกี่ยวกับ PDPA โดยใช้เครื่องมือค้นหา PDF/ฐานความรู้เท่านั้น"""
    workflow = build_langgraph_workflow(pdf_tool=pdf_tool, use_knowledge_base=use_knowledge_base)
    return workflow


if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    try:
        CHAT_SESSION_ID = st.session_state.session_id
    except Exception:
        CHAT_SESSION_ID = None

if "chat_store" not in st.session_state:
    try:
        if ChatHistoryStore is not None:
            qdrant_url = os.getenv("QDRANT_URL2")
            qdrant_api_key = os.getenv("QDRANT_API_KEY2")
            if qdrant_url:
                st.session_state.chat_store = ChatHistoryStore(
                    collection_name="rag_chat_history",
                    qdrant_url=qdrant_url,
                    qdrant_api_key=qdrant_api_key,
                )
                try:
                    CHAT_STORE_REF = st.session_state.chat_store
                except Exception:
                    CHAT_STORE_REF = None
                st.session_state.messages = st.session_state.chat_store.list_messages(st.session_state.session_id)
            else:
                st.session_state.chat_store = None
                st.info("Chat history disabled. Set QDRANT_URL2 to enable Qdrant-backed history.")
        else:
            st.session_state.chat_store = None
    except Exception as e:
        st.warning(f"Chat history store unavailable: {e}")
        st.session_state.chat_store = None

if "pdf_tool" not in st.session_state:
    st.session_state.pdf_tool = None

if "knowledge_base_tool" not in st.session_state:
    knowledge_files = os.path.join("knowledge")
    if os.path.exists(knowledge_files) and os.listdir(knowledge_files):
        try:
            st.session_state.knowledge_base_tool = DocumentSearchTool(file_path=knowledge_files)
        except Exception as e:
            st.error(f"Error loading knowledge base: {str(e)}")
            st.session_state.knowledge_base_tool = None
    else:
        st.session_state.knowledge_base_tool = None

if "langgraph_workflow" not in st.session_state:
    st.session_state.langgraph_workflow = build_langgraph_workflow()

if "using_uploaded_file" not in st.session_state:
    st.session_state.using_uploaded_file = False

if "is_pdpa_related" not in st.session_state:
    st.session_state.is_pdpa_related = False

def build_conversation_context(messages, max_turns=3):
    """รวมประวัติการสนทนาล่าสุดเพื่อให้บริบทเพิ่มเติม"""
    if not messages:
        return ""
    
    start_idx = max(0, len(messages) - (max_turns * 2))
    recent_messages = messages[start_idx:]
    
    context = []
    for msg in recent_messages:
        role_prefix = "ผู้ใช้: " if msg["role"] == "user" else "ผู้ช่วย: "
        context.append(f"{role_prefix}{msg['content']}")
    
    return "\n".join(context)

def build_conversation_context_from_store(chat_store, session_id, max_turns=3):
    """สร้างบริบทการสนทนาจาก ChatHistoryStore"""
    if not chat_store or not session_id:
        return ""
    
    try:
        return chat_store.get_conversation_context(session_id, max_turns=max_turns, max_chars=4000)
    except Exception as e:
        print(f"Error building conversation context from store: {e}")
        return ""

def reset_chat():
    """ล้างประวัติการสนทนา"""
    try:
        if st.session_state.get("chat_store") and st.session_state.get("session_id"):
            st.session_state.chat_store.reset_session(st.session_state.session_id)
    except Exception as e:
        st.warning(f"ไม่สามารถล้างแชตบน Qdrant ได้: {e}")
    st.session_state.messages = []
    perform_periodic_gc()



def _cleanup_on_exit():

    try:
        mode = os.getenv("CHAT_HISTORY_CLEANUP_MODE", "session").lower()
        chat_store = globals().get("CHAT_STORE_REF")
        session_id = globals().get("CHAT_SESSION_ID")
        if chat_store:
            if mode == "collection":
                chat_store.drop_collection()
            elif session_id:
                chat_store.reset_session(session_id)
    except Exception:
        pass


def _signal_handler(signum, frame):
    _cleanup_on_exit()
    raise SystemExit(0)


atexit.register(_cleanup_on_exit)
try:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
except Exception:
    pass

def perform_periodic_gc():
    """ทำ garbage collection เพื่อลดการใช้หน่วยความจำ"""
    try:
        if st.session_state.pdf_tool and hasattr(st.session_state.pdf_tool, "_perform_gc"):
            st.session_state.pdf_tool._perform_gc()
        gc.collect()
    except Exception as e:
        st.error(f"Error during garbage collection: {str(e)}")

def display_pdf(file_bytes: bytes, file_name: str):
    """แสดงไฟล์ PDF ใน iframe"""
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""
    <div style="display: flex; justify-content: center; margin: 20px 0;">
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}" 
            width="100%" 
            height="600px" 
            style="border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; box-shadow: 0 4px 30px rgba(0,0,0,0.1);"
            type="application/pdf"
        >
        </iframe>
    </div>
    """
    st.markdown(f"<h3 style='text-align: center; margin-bottom: 16px; color: #fff;'>รายละเอียดเอกสาร: {file_name}</h3>", unsafe_allow_html=True)
    st.markdown(pdf_display, unsafe_allow_html=True)


if st.session_state.langgraph_workflow is None or st.session_state.using_uploaded_file:
    if st.session_state.pdf_tool is not None:
        if hasattr(st.session_state.pdf_tool, "release_resources"):
            st.session_state.pdf_tool.release_resources()
        st.session_state.pdf_tool = None
    st.session_state.langgraph_workflow = create_agents_and_tasks(
        st.session_state.knowledge_base_tool,
        use_knowledge_base=True
    )
    st.session_state.using_uploaded_file = False
    st.session_state.is_pdpa_related = True


for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else None):
        st.markdown(message["content"])


prompt = st.chat_input("พิมพ์คำถามเกี่ยวกับ PDPA ของคุณ...")

if prompt:
    print(f"🔍 App: Processing prompt: {prompt}")
    
    raw_prompt = prompt

    _SecurityFilter = None
    try:
        from src.agentic_rag.tools.security_filter import SecurityFilter as _SecurityFilter
        print("✅ App: SecurityFilter imported successfully")
    except Exception as e:
        print(f"❌ App: SecurityFilter import failed: {e}")
        try:
            from agentic_rag.tools.security_filter import SecurityFilter as _SecurityFilter
            print("✅ App: SecurityFilter imported successfully (fallback)")
        except Exception as e2:
            print(f"❌ App: SecurityFilter import failed (fallback): {e2}")
            _SecurityFilter = None
    if _SecurityFilter is not None:
        try:
            print(f"🔍 SecurityFilter: Processing prompt: {prompt}")
            _ui_sf = _SecurityFilter()
            _ui_filter = _ui_sf.filter_user_input(prompt or "")
            print(f"🔍 SecurityFilter result: {_ui_filter}")
            
            if not _ui_filter.get("should_respond", True):
                print("🔴 SecurityFilter: BLOCKING prompt")
                with st.chat_message("user", avatar="👤"):
                    st.markdown(raw_prompt)
                with st.chat_message("assistant"):
                    st.markdown(_ui_filter.get("response_message") or "ตรวจพบเนื้อหาไม่เหมาะสมในคำถาม ⚠️ กรุณาพิมพ์ใหม่โดยใช้ถ้อยคำที่สุภาพ")
                st.session_state.messages.append({"role": "assistant", "content": _ui_filter.get("response_message") or "ตรวจพบเนื้อหาไม่เหมาะสมในคำถาม ⚠️ กรุณาพิมพ์ใหม่โดยใช้ถ้อยคำที่สุภาพ"})
                prompt = None
            else:
                print("✅ SecurityFilter: ALLOWING prompt")
        except Exception as e:
            print(f"❌ SecurityFilter error: {e}")
            st.error(f"SecurityFilter error: {e}")
            pass
    else:
        print("❌ SecurityFilter: Not available")

if prompt:

    st.session_state.messages.append({"role": "user", "content": raw_prompt})
    try:
        if st.session_state.get("chat_store"):
            st.session_state.chat_store.add_message(
                session_id=st.session_state.session_id,
                role="user",
                content=raw_prompt,
            )
    except Exception as e:
        st.warning(f"บันทึกประวัติผู้ใช้ไม่สำเร็จ: {e}")
    with st.chat_message("user", avatar="👤"):
        st.markdown(raw_prompt)

    if st.session_state.get("chat_store") and st.session_state.get("session_id"):
        conversation_context = build_conversation_context_from_store(
            st.session_state.chat_store, 
            st.session_state.session_id, 
            max_turns=3
        )
    else:
        conversation_context = build_conversation_context(st.session_state.messages)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        start_time = time.time()
        
        with st.spinner("กำลังประมวลผล..."):
            print("\n" + "="*50)
            print(f"User Query: {prompt}")
            print(f"Conversation Context: {conversation_context[:200]}...")
            print("="*50 + "\n")
            print("🚀 LangGraph is kicking off the process...")
            conversation_history = f"Previous conversation:\n{conversation_context}\n\nNew question:"
            inputs = {"query": prompt, "context": conversation_history}
            stream = st.session_state.langgraph_workflow.stream(inputs, stream_mode="values")
            progress_placeholder = st.empty()
            progress_log = []
            result = None
            last_with_answer = None
            for chunk in stream:
                result = chunk
                if "progress_log" in chunk and chunk["progress_log"]:
                    progress_log = chunk["progress_log"]
                    progress_placeholder.markdown(
                        "<div style='color: #888; opacity: 0.7; font-size: 0.92em;'>"
                        + "<br>".join([f"• {step}" for step in progress_log])
                        + "</div>", unsafe_allow_html=True
                    )
                if ("response" in chunk and chunk["response"]) or ("candidates" in chunk and chunk["candidates"]):
                    last_with_answer = chunk
            progress_placeholder.empty()
            if last_with_answer is not None:
                result = last_with_answer
            print("\n" + "="*50)
            print("✅ LangGraph process finished.")
            print(f"🏁 Final Result: {result}")
            print("="*50 + "\n")
            def _extract_best_answer(res):
                try:
                    if not isinstance(res, dict):
                        return ""
                    for key in ["response", "best_answer"]:
                        val = res.get(key)
                        if isinstance(val, str) and val.strip():
                            return val.strip()
                    for key in ["ranked", "candidates"]:
                        arr = res.get(key)
                        if isinstance(arr, list) and arr:
                            first_val = arr[0]
                            if isinstance(first_val, str) and first_val.strip():
                                return first_val.strip()
                    return ""
                except Exception:
                    return ""
            full_response = _extract_best_answer(result)
            if not full_response:
                full_response = "ข้อมูลไม่เพียงพอในการสรุปคำตอบ โปรดระบุคำถามให้ชัดเจนหรืออัปโหลดเอกสารที่เกี่ยวข้องมากขึ้น"
        
        processing_time = time.time() - start_time
        
        best_answer = full_response
        if "candidates" in result and len(result["candidates"]) > 0:
            if isinstance(result["candidates"][0], str) and result["candidates"][0].strip():
                best_answer = result["candidates"][0].strip()
        
        if "progress_log" in result and result["progress_log"]:
            with st.expander("🛠️ ขั้นตอนการคิด/ทำงานของ Agent (คลิกเพื่อดู)", expanded=False):
                for step in result["progress_log"]:
                    st.markdown(
                        f"""
                        <div style=\"margin-bottom: 8px; padding: 8px 12px; border-radius: 6px; color: #888; font-size: 0.97em;\">
                            {step}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        if "candidate_metrics" in result and "scores" in result["candidate_metrics"] and len(result["candidate_metrics"]["scores"]) > 0:
            best_candidate = result["candidate_metrics"]["scores"][0]  # First candidate is the best
            best_score = best_candidate['scores']['overall_score']
            
            if best_score >= 80:
                score_color = "#4CAF50"  
                score_emoji = "🥇"
            elif best_score >= 70:
                score_color = "#FF9800"  
                score_emoji = "🥈"
            else:
                score_color = "#F44336" 
                score_emoji = "🥉"
            
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 12px; padding: 8px 12px; background: rgba(255, 255, 255, 0.05); border-radius: 8px; border-left: 4px solid {score_color};">
                <span style="font-size: 18px; margin-right: 8px;">{score_emoji}</span>
                <span style="color: #8f94fb; font-size: 14px; margin-right: 12px;">คำตอบที่ดีที่สุด</span>
                <div style="background: {score_color}; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; font-size: 12px;">
                    {best_score:.1f}/100
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        lines = best_answer.split('\n') if isinstance(best_answer, str) else [str(best_answer)]
        for i, line in enumerate(lines):
            full_response_so_far = '\n'.join(lines[:i+1])
            message_placeholder.markdown(full_response_so_far + "▌")
            time.sleep(0.05)  
        
        message_placeholder.markdown(best_answer)
        
        if "search_metadata" in result and result["search_metadata"]:
            with st.expander("📚 แหล่งที่มาของข้อมูล (คลิกเพื่อดู)", expanded=False):
                st.markdown("""
                <div style="background: rgba(78, 84, 200, 0.1); padding: 12px; border-radius: 8px; margin-bottom: 16px;">
                    <p style="margin: 0; color: #8f94fb; font-size: 14px;">
                        💡 ข้อมูลที่ใช้ในการตอบคำถามนี้มาจากเอกสารอ้างอิงดังนี้
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                sources = result["search_metadata"]
                for i, source in enumerate(sources[:5], 1):
                    source_file = source.get('source_file', 'ไม่ระบุไฟล์')
                    page_number = source.get('page_number', 'ไม่ระบุหน้า')
                    
                    if source_file != 'ไม่ระบุไฟล์' and page_number != 'ไม่ระบุหน้า':
                        source_display = f"**[{i}]** {source_file}, หน้า {page_number}"
                    elif source_file != 'ไม่ระบุไฟล์':
                        source_display = f"**[{i}]** {source_file}"
                    else:
                        source_display = f"**[{i}]** เอกสารอ้างอิง"
                    
                    st.markdown(source_display)
        
        

        
        info_container = st.container()
        with info_container:
            col1, col2, col3 = st.columns(3)
            with col1:
                if "retrieval_source" in result:
                    if result["retrieval_source"] == "pdf":
                        source_info = "📚 ฐานความรู้ PDPA"
                    
                    else:
                        source_info = "💭 คำตอบทั่วไป"
                    
                    st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 8px 12px; border-radius: 6px; margin-top: 8px;">
                        <small style="color: #8f94fb;">{source_info}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 8px 12px; border-radius: 6px; margin-top: 8px;">
                        <small style="color: #8f94fb;">💭 คำตอบทั่วไป</small>
                    </div>
                    """, unsafe_allow_html=True)
            with col2:
                if "candidates" in result and len(result["candidates"]) > 1:
                    st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 8px 12px; border-radius: 6px; margin-top: 8px;">
                        <small style="color: #8f94fb;">🎯 สร้างคำตอบ {len(result["candidates"])} แบบ</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: rgba(255, 255, 255, 0.05); padding: 8px 12px; border-radius: 6px; margin-top: 8px;">
                        <small style="color: #8f94fb;">💬 คำตอบเดียว</small>
                    </div>
                    """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div style="background: rgba(255, 255, 255, 0.05); padding: 8px 12px; border-radius: 6px; margin-top: 8px;">
                    <small style="color: #8f94fb;">⏱️ ใช้เวลา {processing_time:.1f} วินาที</small>
                </div>
                """, unsafe_allow_html=True)
        
        if "candidates" in result and len(result["candidates"]) > 1:
            st.markdown("---")
            with st.expander("🔍 ดูคำตอบอื่นๆ พร้อมคะแนนและอันดับ", expanded=False):
                if "candidate_metrics" in result:
                    metrics = result["candidate_metrics"]
                    
                    scores = [c['scores']['overall_score'] for c in metrics['scores']]
                    high_scores = len([s for s in scores if s >= 80])
                    medium_scores = len([s for s in scores if 70 <= s < 80])
                    low_scores = len([s for s in scores if s < 70])
                    
                    st.markdown("### 📊 สรุปผลการประเมิน")
                    
                    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                    with col1:
                        st.metric("คะแนนเฉลี่ย", f"{metrics['average_score']:.1f}/100")
                    with col2:
                        st.metric("คะแนนสูง (80+)", high_scores, delta=None)
                    with col3:
                        st.metric("คะแนนปานกลาง (70-79)", medium_scores, delta=None)
                    with col4:
                        st.metric("คะแนนต่ำ (<70)", low_scores, delta=None)
                    
                    st.info(f"💡 ระบบได้สร้างคำตอบ {metrics['total_candidates']} แบบให้คุณเลือก ด้านบนคือคำตอบที่ดีที่สุด")
                else:
                    st.info("💡 ระบบได้สร้างคำตอบหลายแบบให้คุณเลือก ด้านบนคือคำตอบที่ดีที่สุด คุณสามารถดูคำตอบอื่นๆ ได้ด้านล่างนี้")
                
                if "candidate_metrics" in result and "scores" in result["candidate_metrics"]:
                    st.markdown("---")
                    st.markdown("### 🏆 คำตอบที่จัดอันดับแล้ว")
                    
                    for candidate_data in result["candidate_metrics"]["scores"]:
                        rank = candidate_data['rank']
                        scores = candidate_data['scores']
                        answer = candidate_data['answer']
                        
                        if scores['overall_score'] >= 80:
                            score_color = "🟢"
                            rank_emoji = "🥇"
                        elif scores['overall_score'] >= 70:
                            score_color = "🟡"
                            rank_emoji = "🥈"
                        else:
                            score_color = "🔴"
                            rank_emoji = "🥉"
                        
                        with st.container():
                            st.markdown(f"#### {rank_emoji} คำตอบอันดับที่ {rank} - คะแนน: {scores['overall_score']:.1f}/100 {score_color}")
                            
                            st.markdown("**📊 คะแนนรายละเอียด:**")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("ความเกี่ยวข้อง", f"{scores['relevance']}/100")
                                st.metric("ความครบถ้วน", f"{scores['completeness']}/100")
                            with col2:
                                st.metric("ความถูกต้อง", f"{scores['accuracy']}/100")
                                st.metric("ความชัดเจน", f"{scores['clarity']}/100")
                            with col3:
                                st.metric("การอ้างอิง", f"{scores['legal_citation']}/100")
                                st.metric("คะแนนรวม", f"{scores['overall_score']:.1f}/100")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**✅ จุดเด่น:**")
                                for strength in scores.get('strengths', ['ไม่มีข้อมูล']):
                                    st.markdown(f"• {strength}")
                            
                            with col2:
                                st.markdown("**⚠️ จุดที่ควรปรับปรุง:**")
                                for weakness in scores.get('weaknesses', ['ไม่มีข้อมูล']):
                                    st.markdown(f"• {weakness}")
                            
                            st.markdown("**📝 คำตอบ:**")
                            st.markdown(answer)
                            
                            st.markdown("---")  
                else:
                    st.markdown("---")
                    st.markdown("### 💡 คำตอบอื่นๆ")
                    for i, candidate in enumerate(result["candidates"][1:], 2):
                        st.markdown(f"#### 💡 คำตอบที่ {i}")
                        st.markdown(candidate)
                        st.markdown("---")
    
    st.session_state.messages.append({"role": "assistant", "content": best_answer})
    try:
        if st.session_state.get("chat_store"):
            st.session_state.chat_store.add_message(
                session_id=st.session_state.session_id,
                role="assistant",
                content=best_answer,
            )
    except Exception as e:
        st.warning(f"บันทึกประวัติผู้ช่วยไม่สำเร็จ: {e}")
    
    perform_periodic_gc()

