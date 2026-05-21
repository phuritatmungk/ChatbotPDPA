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

if "theme" not in st.session_state:
    st.session_state.theme = "dark"
is_light = st.session_state.theme == "light"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sarabun:wght@400;500;600;700&display=swap');

:root {{
    --bg-0: #0d1124;
    --bg-1: #151a35;
    --bg-2: #1d2347;
    --surface: rgba(255,255,255,0.04);
    --surface-strong: rgba(255,255,255,0.07);
    --border: rgba(255,255,255,0.09);
    --border-strong: rgba(255,255,255,0.16);
    --text-1: #f1f4ff;
    --text-2: #b8bedb;
    --text-3: #7f86a8;
    --brand: #8b93ff;
    --brand-2: #c4a8ff;
    --brand-glow: rgba(139,147,255,0.35);
    --success: #4ade80;
    --warn: #fbbf24;
    --danger: #f87171;
    --radius-sm: 10px;
    --radius-md: 14px;
    --radius-lg: 20px;
}}

html, body, [class^='css'] {{
    font-family: 'Inter', 'Sarabun', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}}

.stApp {{
    background:
        radial-gradient(1200px 600px at 12% -10%, rgba(139,147,255,0.18), transparent 60%),
        radial-gradient(1000px 500px at 90% 0%, rgba(196,168,255,0.14), transparent 55%),
        linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 60%, var(--bg-2) 100%);
    color: var(--text-1);
    min-height: 100vh;
}}

@keyframes auroraDrift {{
    0%, 100% {{ transform: translate3d(0,0,0); }}
    50% {{ transform: translate3d(2%, -1%, 0); }}
}}
.stApp::before {{
    content: "";
    position: fixed; inset: -20% -10% auto -10%;
    height: 60vh;
    background: radial-gradient(60% 60% at 50% 0%, rgba(139,147,255,0.18), transparent 70%);
    filter: blur(40px);
    animation: auroraDrift 14s ease-in-out infinite;
    pointer-events: none; z-index: 0;
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, rgba(13,17,36,0.92) 0%, rgba(21,26,53,0.88) 100%) !important;
    border-right: 1px solid var(--border) !important;
}}
section[data-testid="stSidebar"] > div {{ padding-top: 12px !important; }}
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {{ display: none; }}
.sidebar-title {{
    font-size: 0.72rem; color: var(--text-3); letter-spacing: 1px;
    text-transform: uppercase; font-weight: 600;
    padding: 6px 4px 8px;
}}
.session-item {{
    display: block;
    padding: 10px 12px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    color: var(--text-2);
    margin: 2px 0;
    font-size: 0.86rem; line-height: 1.35;
    cursor: pointer;
    transition: all 140ms ease;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.session-item:hover {{
    background: var(--surface);
    color: var(--text-1);
}}
.session-item.active {{
    background: rgba(139,147,255,0.14);
    border-color: rgba(139,147,255,0.35);
    color: var(--text-1);
}}
.session-meta {{
    font-size: 0.68rem; color: var(--text-3);
    margin-top: 2px; font-weight: 400;
}}
section[data-testid="stSidebar"] .stButton > button {{
    width: 100%;
    background: transparent;
    border: 1px solid var(--border);
    box-shadow: none;
    padding: 10px 12px;
    font-weight: 500;
    font-size: 0.86rem;
    text-align: left;
    color: var(--text-2);
    line-height: 1.35;
    white-space: normal;
    height: auto;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: var(--surface);
    color: var(--text-1);
    border-color: var(--border-strong);
    transform: none;
    box-shadow: none;
}}
section[data-testid="stSidebar"] .new-chat-btn .stButton > button {{
    background: linear-gradient(135deg, #5b63ff 0%, #8b93ff 100%);
    color: white;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 4px 18px rgba(91,99,255,0.25);
    text-align: center;
    font-weight: 600;
    margin-bottom: 8px;
}}
section[data-testid="stSidebar"] .del-btn .stButton > button {{
    background: rgba(248,113,113,0.12);
    color: var(--danger);
    border: 1px solid rgba(248,113,113,0.32);
    text-align: center;
    padding: 10px 0;
}}
section[data-testid="stSidebar"] .del-btn .stButton > button:hover {{
    background: rgba(248,113,113,0.22);
    color: #fff;
    border-color: var(--danger);
}}

/* Compact top bar */
.topbar {{
    position: sticky; top: 3rem; z-index: 50;
    display: flex; align-items: center; gap: 14px;
    padding: 14px 8px;
    margin-bottom: 18px;
    background: linear-gradient(180deg, rgba(13,17,36,0.92), rgba(13,17,36,0.6) 70%, transparent);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
}}
.topbar-logo {{
    width: 36px; height: 36px;
    filter: drop-shadow(0 0 12px var(--brand-glow));
}}
.topbar-title {{
    font-weight: 700; font-size: 1.05rem;
    letter-spacing: 0.2px; color: var(--text-1);
}}
.topbar-sub {{
    font-size: 0.78rem; color: var(--text-3);
}}
.topbar-spacer {{ flex: 1; }}
.status-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 12px; border-radius: 999px;
    background: var(--surface);
    border: 1px solid var(--border);
    font-size: 0.76rem; color: var(--text-2);
}}
.status-dot {{
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 8px var(--success);
}}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg, #5b63ff 0%, #8b93ff 100%);
    color: white; font-weight: 600;
    padding: 9px 18px;
    border-radius: var(--radius-md);
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow: 0 4px 18px rgba(91,99,255,0.25);
    transition: transform 120ms ease, box-shadow 180ms ease;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(91,99,255,0.4);
}}
.stButton > button:active {{ transform: translateY(0); }}

/* Chat messages */
.stChatMessage {{
    background: var(--surface);
    backdrop-filter: blur(10px);
    border-radius: var(--radius-lg);
    padding: 18px 22px;
    margin: 10px 0;
    border: 1px solid var(--border);
    color: var(--text-1);
    line-height: 1.65;
}}
.stChatMessage[data-testid="user"] {{
    background: linear-gradient(135deg, rgba(91,99,255,0.18), rgba(139,147,255,0.10));
    border-color: rgba(139,147,255,0.35);
}}
.stChatMessage[data-testid="assistant"] {{
    background: var(--surface-strong);
}}

/* Chat input */
.stChatInputContainer {{
    position: fixed;
    left: 0 !important; right: 0 !important; bottom: 0;
    background: linear-gradient(180deg, transparent, rgba(13,17,36,0.92) 30%);
    padding: 16px 20px 22px !important;
    z-index: 100;
    display: flex; justify-content: center;
    width: 100% !important;
}}
.stChatInputContainer > div {{
    width: 100% !important;
    max-width: 920px !important;
    margin: 0 auto !important;
}}
.stChatInputContainer textarea {{
    background: var(--surface-strong) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-1) !important;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 10px; background: transparent; }}
::-webkit-scrollbar-thumb {{
    background: rgba(139,147,255,0.4);
    border-radius: 8px;
    border: 2px solid transparent;
    background-clip: padding-box;
}}
::-webkit-scrollbar-thumb:hover {{ background: rgba(139,147,255,0.7); background-clip: padding-box; }}

/* Block container */
.block-container {{
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    padding-bottom: 120px !important;
    max-width: 980px !important;
}}

/* Expander */
.streamlit-expanderHeader, [data-testid="stExpander"] details > summary {{
    background: var(--surface) !important;
    color: var(--text-1) !important;
    font-weight: 500 !important;
    border-radius: var(--radius-md) !important;
    padding: 11px 16px !important;
    border: 1px solid var(--border) !important;
    margin: 8px 0 !important;
    transition: all 180ms ease !important;
}}
.streamlit-expanderHeader:hover, [data-testid="stExpander"] details > summary:hover {{
    background: var(--surface-strong) !important;
    border-color: var(--border-strong) !important;
}}
.streamlit-expanderContent, [data-testid="stExpander"] details > div {{
    background: rgba(13,17,36,0.5) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 16px !important;
    margin-top: 8px !important;
}}

/* Empty state */
.empty-hero {{
    display: flex; flex-direction: column; align-items: center;
    text-align: center;
    padding: 56px 16px 24px;
    color: var(--text-2);
}}
.empty-hero .badge {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px; border-radius: 999px;
    background: rgba(139,147,255,0.10);
    border: 1px solid rgba(139,147,255,0.30);
    color: var(--brand); font-size: 0.78rem;
    margin-bottom: 18px;
}}
.empty-hero h1 {{
    font-size: 2.1rem; font-weight: 700;
    margin: 0 0 10px;
    background: linear-gradient(135deg, #fff, #c4a8ff);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.empty-hero p {{
    max-width: 520px; line-height: 1.6; margin: 0 0 28px;
    color: var(--text-2);
}}
.suggest-grid {{
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 10px; width: 100%; max-width: 680px;
}}
@media (max-width: 640px) {{ .suggest-grid {{ grid-template-columns: 1fr; }} }}
.suggest-chip {{
    text-align: left;
    padding: 14px 16px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    color: var(--text-1);
    font-size: 0.92rem; line-height: 1.45;
    transition: all 180ms ease;
}}
.suggest-chip:hover {{
    border-color: rgba(139,147,255,0.55);
    background: var(--surface-strong);
    transform: translateY(-1px);
    box-shadow: 0 6px 22px rgba(91,99,255,0.18);
}}
.suggest-chip .label {{
    display: block; font-size: 0.72rem; color: var(--brand);
    letter-spacing: 0.5px; text-transform: uppercase;
    margin-bottom: 4px; font-weight: 600;
}}

/* Source card */
.src-card {{
    display: flex; gap: 12px; align-items: flex-start;
    padding: 10px 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    margin: 6px 0;
}}
.src-card .src-num {{
    flex: 0 0 28px; height: 28px;
    border-radius: 8px;
    background: linear-gradient(135deg, rgba(91,99,255,0.4), rgba(139,147,255,0.2));
    color: var(--text-1);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.78rem; font-weight: 700;
}}
.src-card .src-body {{ flex: 1; min-width: 0; }}
.src-card .src-file {{
    color: var(--text-1); font-weight: 500; font-size: 0.92rem;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.src-card .src-page {{
    color: var(--text-3); font-size: 0.78rem; margin-top: 2px;
}}
</style>

<div class="topbar">
    <img class="topbar-logo" src="data:image/png;base64,{logo_base64}" />
    <div>
        <div class="topbar-title">PDPA Assistant</div>
        <div class="topbar-sub">ผู้ช่วยตอบคำถาม พ.ร.บ. คุ้มครองข้อมูลส่วนบุคคล</div>
    </div>
    <div class="topbar-spacer"></div>
    <div class="status-pill"><span class="status-dot"></span>Typhoon · Local</div>
</div>
""", unsafe_allow_html=True)

if is_light:
    st.markdown("""
    <style>
    :root {
        --bg-0: #f4f6fb;
        --bg-1: #ffffff;
        --bg-2: #eef1f8;
        --surface: rgba(15,18,40,0.04);
        --surface-strong: rgba(15,18,40,0.07);
        --border: rgba(15,18,40,0.10);
        --border-strong: rgba(15,18,40,0.18);
        --text-1: #15172a;
        --text-2: #4a4e6b;
        --text-3: #7a82a3;
        --brand: #5b63ff;
        --brand-2: #8b93ff;
        --brand-glow: rgba(91,99,255,0.30);
    }
    .stApp {
        background:
            radial-gradient(1200px 600px at 12% -10%, rgba(91,99,255,0.10), transparent 60%),
            radial-gradient(1000px 500px at 90% 0%, rgba(196,168,255,0.10), transparent 55%),
            linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 60%, var(--bg-2) 100%) !important;
        color: var(--text-1) !important;
    }
    .stApp::before {
        background: radial-gradient(60% 60% at 50% 0%, rgba(91,99,255,0.12), transparent 70%) !important;
    }
    .topbar {
        background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.6) 70%, transparent) !important;
    }
    .topbar-title { color: var(--text-1) !important; }
    .stChatInputContainer {
        background: linear-gradient(180deg, transparent, rgba(244,246,251,0.92) 30%) !important;
    }
    .stChatInputContainer textarea {
        background: #fff !important;
        color: var(--text-1) !important;
    }
    .stChatMessage { color: var(--text-1) !important; }
    .stChatMessage[data-testid="user"] {
        background: linear-gradient(135deg, rgba(91,99,255,0.10), rgba(139,147,255,0.06)) !important;
        border-color: rgba(91,99,255,0.30) !important;
    }
    .stChatMessage[data-testid="assistant"] {
        background: #fff !important;
        border-color: var(--border) !important;
    }
    .empty-hero h1 {
        background: linear-gradient(135deg, #15172a, #5b63ff) !important;
        -webkit-background-clip: text !important;
        background-clip: text !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(238,241,248,0.95) 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    .streamlit-expanderHeader, [data-testid="stExpander"] details > summary { color: var(--text-1) !important; }
    .streamlit-expanderContent, [data-testid="stExpander"] details > div {
        background: rgba(255,255,255,0.7) !important;
    }
    ::-webkit-scrollbar-thumb { background: rgba(91,99,255,0.35) !important; background-clip: padding-box; }
    </style>
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


if "preset_prompt" not in st.session_state:
    st.session_state.preset_prompt = None


def _format_relative_ts(ts: float) -> str:
    if not ts:
        return ""
    now = time.time()
    delta = max(0, now - ts)
    if delta < 60:
        return "เมื่อสักครู่"
    if delta < 3600:
        return f"{int(delta // 60)} นาทีก่อน"
    if delta < 86400:
        return f"{int(delta // 3600)} ชั่วโมงก่อน"
    if delta < 86400 * 7:
        return f"{int(delta // 86400)} วันก่อน"
    return time.strftime("%d %b", time.localtime(ts))


def _load_session(sid: str):
    st.session_state.session_id = sid
    try:
        if st.session_state.get("chat_store"):
            st.session_state.messages = st.session_state.chat_store.list_messages(sid)
        else:
            st.session_state.messages = []
    except Exception:
        st.session_state.messages = []


def _new_chat():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []


with st.sidebar:
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("✨  สนทนาใหม่", key="sidebar_new_chat", use_container_width=True):
        _new_chat()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    theme_label = "🌙  โหมดมืด" if is_light else "☀️  โหมดสว่าง"
    if st.button(theme_label, key="theme_toggle", use_container_width=True):
        st.session_state.theme = "dark" if is_light else "light"
        st.rerun()

    st.markdown('<div class="sidebar-title">ประวัติการสนทนา</div>', unsafe_allow_html=True)

    sessions = []
    if st.session_state.get("chat_store"):
        try:
            sessions = st.session_state.chat_store.list_sessions(limit=50)
        except Exception as e:
            st.caption(f"โหลดประวัติไม่สำเร็จ: {e}")

    if not sessions:
        st.markdown(
            '<div style="color: var(--text-3); font-size: 0.82rem; padding: 6px 4px;">ยังไม่มีประวัติ — เริ่มสนทนาเพื่อบันทึก</div>',
            unsafe_allow_html=True,
        )
    else:
        current_sid = st.session_state.get("session_id")
        for s in sessions:
            sid = s["session_id"]
            is_active = sid == current_sid
            col_load, col_del = st.columns([5, 1], gap="small")
            with col_load:
                if is_active:
                    st.markdown(
                        f'<div class="session-item active">{s["title"]}'
                        f'<div class="session-meta">{_format_relative_ts(s["last_ts"])} · {s["message_count"]} ข้อความ · กำลังใช้งาน</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    label = f"{s['title']}\n{_format_relative_ts(s['last_ts'])} · {s['message_count']} ข้อความ"
                    if st.button(label, key=f"session_{sid}", use_container_width=True):
                        _load_session(sid)
                        st.rerun()
            with col_del:
                st.markdown('<div class="del-btn">', unsafe_allow_html=True)
                if st.button("🗑", key=f"del_{sid}", help="ลบการสนทนานี้", use_container_width=True):
                    try:
                        st.session_state.chat_store.reset_session(sid)
                    except Exception as e:
                        st.warning(f"ลบไม่สำเร็จ: {e}")
                    if is_active:
                        _new_chat()
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown("""
    <div class="empty-hero">
        <span class="badge">● PDPA · พ.ร.บ. คุ้มครองข้อมูลส่วนบุคคล</span>
        <h1>ถามคำถามเกี่ยวกับ PDPA ได้เลย</h1>
        <p>ผู้ช่วยอ้างอิงจากตัวบทกฎหมาย พร้อมระบุมาตราและหน้าเอกสารต้นทาง ตอบเป็นภาษาไทย พร้อมเหตุผลและตัวอย่างประกอบ</p>
    </div>
    """, unsafe_allow_html=True)

    suggestions = [
        "เจ้าของข้อมูลส่วนบุคคลมีสิทธิอะไรบ้างตาม PDPA?",
        "ฐานทางกฎหมายในการประมวลผลข้อมูลมีอะไรบ้าง?",
        "เกิดเหตุข้อมูลรั่วไหล ต้องแจ้งใครและภายในกี่วัน?",
        "การถ่ายภาพหรือ CCTV ที่ติดบุคคลอื่นต้องขอความยินยอมไหม?",
    ]
    sc1, sc2 = st.columns(2, gap="small")
    for idx, q in enumerate(suggestions):
        target_col = sc1 if idx % 2 == 0 else sc2
        with target_col:
            if st.button(q, key=f"suggest_{idx}", use_container_width=True):
                st.session_state.preset_prompt = q
                st.rerun()

    rc1, rc2, rc3 = st.columns([1, 1, 1])
    with rc2:
        if st.button("🔄 เริ่มการสนทนาใหม่", key="new_chat_empty", use_container_width=True):
            reset_chat()
            st.rerun()
else:
    nc1, nc2 = st.columns([6, 1])
    with nc2:
        if st.button("🔄 ใหม่", key="new_chat_top", help="เริ่มการสนทนาใหม่", use_container_width=True):
            reset_chat()
            st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else None):
        st.markdown(message["content"])


prompt = st.chat_input("พิมพ์คำถามเกี่ยวกับ PDPA ของคุณ...")
if st.session_state.preset_prompt and not prompt:
    prompt = st.session_state.preset_prompt
    st.session_state.preset_prompt = None

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
            with st.expander(f"📚 แหล่งอ้างอิง · {len(result['search_metadata'][:5])} รายการ", expanded=False):
                cards_html = []
                for i, source in enumerate(result["search_metadata"][:5], 1):
                    source_file = source.get('source_file', 'ไม่ระบุไฟล์')
                    page_number = source.get('page_number', None)
                    page_label = f"หน้า {page_number}" if page_number and page_number != 'ไม่ระบุหน้า' else "ไม่ระบุหน้า"
                    cards_html.append(
                        f'<div class="src-card">'
                        f'  <div class="src-num">{i}</div>'
                        f'  <div class="src-body">'
                        f'    <div class="src-file">{source_file}</div>'
                        f'    <div class="src-page">{page_label}</div>'
                        f'  </div>'
                        f'</div>'
                    )
                st.markdown("\n".join(cards_html), unsafe_allow_html=True)
        
        

        
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

