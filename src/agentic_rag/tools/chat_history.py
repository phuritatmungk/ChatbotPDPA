from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue, VectorParams, Distance, PayloadSchemaType
import uuid
import time
import json


class ChatHistoryStore:
    """
    Enhanced Qdrant-backed chat history with conversation context management.
    - Uses a tiny 1-dim dummy vector [0.0] for compatibility
    - Filters by session_id; sorts by timestamp client-side
    - Supports conversation context building and memory management
    """

    def __init__(self, collection_name: str, qdrant_url: str, qdrant_api_key: Optional[str] = None):
        self.collection_name = collection_name
        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1, distance=Distance.COSINE),
            )
      
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="session_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            
            pass

    def add_message(self, session_id: str, role: str, content: str, ts: Optional[float] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        """เพิ่มข้อความใหม่ลงในประวัติการสนทนา"""
        payload: Dict[str, Any] = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "ts": float(ts if ts is not None else time.time()),
            "message_id": str(uuid.uuid4()),
        }
        if extra:
            payload.update(extra)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=[0.0],
            payload=payload,
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])

    def list_messages(self, session_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        """ดึงรายการข้อความทั้งหมดในเซสชัน"""
      
        flt = Filter(must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))])
        all_payloads: List[Dict[str, Any]] = []
        next_page = None
        while True:
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=flt,
                with_payload=True,
                with_vectors=False,
                limit=min(256, limit - len(all_payloads)) if limit else 256,
                offset=next_page,
            )
            points, next_page = result
            for p in points:
                if p.payload:
                    all_payloads.append(p.payload)
            if not next_page or (limit and len(all_payloads) >= limit):
                break
        all_payloads.sort(key=lambda x: x.get("ts", 0.0))
        return all_payloads[:limit] if limit else all_payloads

    def get_conversation_context(self, session_id: str, max_turns: int = 5, max_chars: int = 4000) -> str:
        """
        สร้างบริบทการสนทนาจากประวัติล่าสุด
        
        Args:
            session_id: ID ของเซสชันการสนทนา
            max_turns: จำนวนรอบการสนทนาสูงสุดที่จะรวม (user + assistant = 1 turn)
            max_chars: จำนวนตัวอักษรสูงสุดในบริบท
            
        Returns:
            บริบทการสนทนาในรูปแบบข้อความ
        """
        messages = self.list_messages(session_id, limit=max_turns * 2)
        
        if not messages:
            return ""
        
    
        context_parts = []
        char_count = 0
        
       
        messages.sort(key=lambda x: x.get("ts", 0.0))
        
      
        for message in reversed(messages):
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "user":
                context_line = f"ผู้ใช้: {content}"
            elif role == "assistant":
                context_line = f"ผู้ช่วย: {content}"
            else:
                continue
            
           
            if char_count + len(context_line) > max_chars:
                break
                
            context_parts.insert(0, context_line)  
            char_count += len(context_line)
        
        return "\n".join(context_parts)

    def get_recent_context(self, session_id: str, last_n_messages: int = 6) -> List[Dict[str, Any]]:
        """
        ดึงข้อความล่าสุดจำนวนที่กำหนด
        
        Args:
            session_id: ID ของเซสชันการสนทนา
            last_n_messages: จำนวนข้อความล่าสุดที่ต้องการ
            
        Returns:
            รายการข้อความล่าสุด
        """
        messages = self.list_messages(session_id, limit=last_n_messages)
        return messages[-last_n_messages:] if len(messages) > last_n_messages else messages

    def build_conversation_prompt(self, session_id: str, current_query: str, max_turns: int = 3) -> str:
        """
        สร้าง prompt ที่รวมประวัติการสนทนาและคำถามปัจจุบัน
        
        Args:
            session_id: ID ของเซสชันการสนทนา
            current_query: คำถามปัจจุบัน
            max_turns: จำนวนรอบการสนทนาที่จะรวม
            
        Returns:
            prompt ที่พร้อมส่งให้ LLM
        """
        context = self.get_conversation_context(session_id, max_turns=max_turns)
        
        if not context:
            return f"คำถาม: {current_query}"
        
        prompt = f"""ประวัติการสนทนาก่อนหน้า:
{context}

คำถามปัจจุบัน: {current_query}

โปรดตอบคำถามปัจจุบันโดยพิจารณาจากประวัติการสนทนาข้างต้น หากคำถามปัจจุบันเกี่ยวข้องกับคำถามก่อนหน้า ให้อ้างอิงถึงบริบทที่เกี่ยวข้อง"""
        
        return prompt

    def reset_session(self, session_id: str) -> None:
        """ลบประวัติการสนทนาทั้งหมดในเซสชัน"""
        flt = Filter(must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))])
        self.client.delete(collection_name=self.collection_name, points_selector=flt)

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """ดึงสถิติของเซสชันการสนทนา"""
        messages = self.list_messages(session_id)
        
        user_messages = [m for m in messages if m.get("role") == "user"]
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        
        return {
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "conversation_turns": min(len(user_messages), len(assistant_messages)),
            "session_duration": messages[-1].get("ts", 0) - messages[0].get("ts", 0) if messages else 0
        }

    def drop_collection(self) -> None:
        """Dangerous: deletes the entire chat history collection."""
        self.client.delete_collection(self.collection_name)


