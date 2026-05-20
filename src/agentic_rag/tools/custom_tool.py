import os
import warnings
from typing import Type, List, Optional, Dict
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv
import hashlib
import time
import gc
import traceback
import logging
from .qdrant_storage import QdrantStorage, MyEmbedder
try:
    import pdfplumber
except Exception:
    pdfplumber = None


try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
    FlagReranker = None  
except ImportError:
    CrossEncoder = None
    FlagReranker = None
    RERANKER_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Reranking will be disabled. Install with: pip install sentence-transformers")


warnings.filterwarnings("ignore", category=DeprecationWarning)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DocumentSearchTool")

load_dotenv()



class DocumentSearchToolInput(BaseModel):
    """
    สคีมาสำหรับรับข้อมูลอินพุตสำหรับการค้นหาในเอกสาร PDF
    """
    query: str = Field(..., description="คำถามสำหรับค้นหาในเอกสาร")
    context: Optional[str] = Field(None, description="ประวัติการสนทนาก่อนหน้า")
    model_config = ConfigDict(extra="allow")


class DocumentSearchTool(object):
    name: str = "DocumentSearchTool"
    description: str = "ค้นหาข้อมูลเกี่ยวกับ PDPA จากเอกสาร PDF โดยใช้ pythainlp, fitz, pdfplumber และ OCR"
    args_schema: Type[BaseModel] = DocumentSearchToolInput
    model_config = ConfigDict(extra="allow")

    def __init__(self, file_path: str):
        """
        เริ่มต้น DocumentSearchTool ด้วยไฟล์หรือโฟลเดอร์ที่ระบุ
        """
        super().__init__()
        self.file_path = file_path
        self.raw_text = ""
        self.chunks = []
        self.initialized = False
        self.use_vector_db = True  
     
        self.disable_indexing = os.getenv("RAG_DISABLE_INDEXING", "1").lower() in ("1", "true", "yes", "y")

        self.require_rag_doc = os.getenv("RAG_REQUIRE_RAG_DOC", "1").lower() in ("1", "true", "yes", "y")
 
        self.search_all_collections = True
        self.vector_db = None

        self.fallback_chunks: List[Dict[str, any]] = []
        self.fallback_loaded = False

        self.query_cache = {} 
        self.last_cache_cleanup = time.time()
        self.cache_ttl = 3600  
        self.last_gc_time = time.time()
        self.gc_interval = 300  
        
 
        self.reranker = None
        self.use_reranker = RERANKER_AVAILABLE and os.getenv("RAG_USE_RERANKER", "1").lower() in ("1", "true", "yes", "y")
        if self.use_reranker:
            try:
                self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
                logger.info("BGE reranker initialized successfully using CrossEncoder")
            except Exception as e:
                logger.warning(f"Failed to initialize BGE reranker: {e}")
                self.use_reranker = False
                self.reranker = None
        else:
            logger.info("Reranking disabled (set RAG_USE_RERANKER=1 to enable)")
        

    def _ensure_initialized(self):
        """
        Initialize vector DB lazily; skip any file/OCR/chunk work.
        """
        if not self.initialized:
            logger.info("Initializing DocumentSearchTool (read-only, Qdrant search)...")
            if self.vector_db is None:
                self._initialize_vector_db()

            self.initialized = True

    def _load_directory(self):
        """
        Deprecated in read-only mode: no file loading. Ensure vector DB only.
        """
        try:
            if self.vector_db is None:
                self._initialize_vector_db()
            self.initialized = True
        except Exception as e:
            logger.error(f"Error in _load_directory: {str(e)}")
            logger.error(traceback.format_exc())

    def _load_single_file(self):
        """
        Deprecated in read-only mode: no file loading. Ensure vector DB only.
        """
        try:
            if self.vector_db is None:
                self._initialize_vector_db()
            self.initialized = True
        except Exception as e:
            logger.error(f"Error in _load_single_file: {str(e)}")
            logger.error(traceback.format_exc())

    def _initialize_tool(self):
        """
        Deprecated in read-only mode: no chunking/indexing.
        """
        self.initialized = True

    def _initialize_vector_db(self):
        """
        Initialize Qdrant vector database
        """
        try:
            import os
            self.embedder = MyEmbedder(os.getenv("RAG_EMBED_MODEL"))

            explicit_collection = os.getenv("RAG_COLLECTION_NAME")
            allow_create_env = os.getenv("RAG_ALLOW_CREATE_COLLECTION", "0").lower() in ("1", "true", "yes", "y")
  
            check_version = False
            derived_type = "doc_default"
            q_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            q_key = os.getenv("QDRANT_API_KEY", None)

        
            if not explicit_collection:
                try:
                    from qdrant_client import QdrantClient as _QClient
                    _qc = _QClient(url=q_url, api_key=q_key)
                    collections = getattr(_qc.get_collections(), 'collections', [])
                    best_name = None
                    first_match = None
                    best_count = -1
                    for col in collections:
                        name = getattr(col, 'name', None) or col.get('name') if isinstance(col, dict) else None
                        if not name:
                            continue
                        if first_match is None:
                            first_match = name
                        try:
                            cnt = _qc.count(name, exact=False)
                            count_val = getattr(cnt, 'count', None)
                            if count_val is None and isinstance(cnt, dict):
                                count_val = cnt.get('count', 0)
                            count_val = int(count_val or 0)
                        except Exception:
                            count_val = 0
                        if count_val > best_count:
                            best_count = count_val
                            best_name = name
                 
                    if best_name and best_count > 0:
                        explicit_collection = best_name
                        logger.info(f"Auto-selected existing Qdrant collection: {explicit_collection} (points≈{best_count})")
                    elif first_match:
                        explicit_collection = first_match
                        logger.info(f"Auto-selected first matching Qdrant collection: {explicit_collection}")
                except Exception as _e:
                    logger.warning(f"Auto-discovery of Qdrant collection failed: {_e}")

          
            allow_create = False
            self.vector_db = QdrantStorage(
                type=derived_type,
                qdrant_location=q_url,
                qdrant_api_key=q_key,
                embedder=self.embedder,
                collection_name=(explicit_collection if explicit_collection else None),
                allow_create=allow_create,
            )
            logger.info(f"Qdrant vector database initialized successfully (collection: {self.vector_db.collection_name})")
        except Exception as e:
            logger.error(f"Error initializing Qdrant vector DB: {str(e)}")
            logger.error(traceback.format_exc())
            self.use_vector_db = False

    def _index_chunks(self):
        """
        Disabled in read-only mode.
        """
        return

    def _is_vector_db_ready(self) -> bool:
        return (
            self.use_vector_db and 
            self.vector_db is not None and 
            self.embedder is not None
        )

    def _process_context(self, context: Optional[str], max_length: int = 1000) -> Optional[str]:
        """
        ประมวลผล context โดยจำกัดขนาดและทำความสะอาด
        """
        if not context:
            return None
            
   
        if len(context) > max_length:
        
            context = context[-max_length:]
            
            newline_pos = context.find('\n')
            if newline_pos > 0:
                context = context[newline_pos+1:]
                
     
        return self._process_thai_text(context)

    def _search_chunks(self, query: str, context: Optional[str] = None) -> List[Dict[str, any]]:
        """
        ค้นหาชิ้นส่วนข้อความ (chunks) จาก Qdrant หรือ fallback ที่อ่านจาก PDF โดยตรง
        """
        fallback_results: List[Dict[str, any]] = []

 
        if self._is_vector_db_ready():
            try:
                self._cleanup_cache()
                cache_key = self._get_cache_key((query or "") + (context or ""))
                if cache_key in self.query_cache:
                    timestamp, result = self.query_cache[cache_key]
                    if time.time() - timestamp <= self.cache_ttl:
                        return result

                search_query = self._process_thai_text(query)
                if search_query:
                    vector = self.embedder.encode(search_query)
                    qc = self.vector_db.client
                    cols_resp = qc.get_collections()
                    collections = getattr(cols_resp, 'collections', [])
                    aggregated_results: List[Dict[str, any]] = []
                    searched_cols: List[str] = []

                    if collections:
                        for col in collections:
                            collection_name = getattr(col, 'name', None) or (col.get('name') if isinstance(col, dict) else None)
                            if not collection_name:
                                continue
                            try:
                                search_result = qc.search(
                                    collection_name=collection_name,
                                    query_vector=vector,
                                    limit=5,
                                    with_payload=True,
                                )
                                searched_cols.append(collection_name)
                                for point in search_result:
                                    payload = getattr(point, 'payload', None) or (point.get('payload') if isinstance(point, dict) else None)
                                    if isinstance(payload, dict):
                                        text_val = payload.get('text')
                                        if text_val:
                                            chunk_data = {
                                                'text': text_val,
                                                'source_file': payload.get('source_file', 'ไม่ระบุไฟล์'),
                                                'page_number': payload.get('page_number', 'ไม่ระบุหน้า'),
                                                'chunk_id': payload.get('chunk_id', 'ไม่ระบุ'),
                                                'collection_name': collection_name,
                                                'score': getattr(point, 'score', 0.0) if hasattr(point, 'score') else 0.0
                                            }
                                            aggregated_results.append(chunk_data)
                                    elif isinstance(payload, str):
                                        chunk_data = {
                                            'text': payload,
                                            'source_file': 'ไม่ระบุไฟล์',
                                            'page_number': 'ไม่ระบุหน้า',
                                            'chunk_id': 'ไม่ระบุ',
                                            'collection_name': collection_name,
                                            'score': 0.0
                                        }
                                        aggregated_results.append(chunk_data)
                            except Exception as e:
                                logger.warning(f"Could not search in collection '{collection_name}': {e}")
                                continue

                        if searched_cols:
                            logger.info(f"Qdrant searched in {len(searched_cols)} collections: {', '.join(searched_cols)}")

                 
                    if aggregated_results and self.use_reranker:
                        try:
                            aggregated_results = self._rerank_results(search_query, aggregated_results, top_k=10)
                        except Exception as e:
                            logger.warning(f"Reranking failed, using original results: {e}")

                    self.query_cache[cache_key] = (time.time(), aggregated_results)
                    if aggregated_results:
                        return aggregated_results
            except Exception as e:
                logger.error(f"An unexpected error occurred in _search_chunks (Qdrant): {str(e)}")
                logger.error(traceback.format_exc())

     
        fallback_results = self._fallback_search(query)
        if fallback_results:
            return fallback_results

        return []

    def _fallback_search(self, query: str) -> List[Dict[str, any]]:
        """
        ค้นหาแบบง่ายจากไฟล์ PDF (lexical) กรณี Qdrant ใช้ไม่ได้หรือไม่มีข้อมูล
        """
        if not pdfplumber:
            return []

        try:
            if not self.fallback_loaded:
                self._load_fallback_chunks()

            processed_query = (query or "").strip()
            if not processed_query or not self.fallback_chunks:
                return []

            q_lower = processed_query.lower()

            def score_chunk(text: str) -> float:
                t = text.lower()
                hit = t.count(q_lower)
                token_hits = sum(1 for tok in processed_query.split() if tok and tok.lower() in t)
                return float(hit * 2 + token_hits)

            scored = []
            for chunk in self.fallback_chunks:
                txt = chunk.get("text", "")
                sc = score_chunk(txt)
                if sc > 0:
                    scored.append((sc, chunk))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = [dict(item[1], rerank_score=item[0]) for item in scored[:5]]
            return top
        except Exception as e:
            logger.warning(f"Fallback PDF search failed: {e}")
            return []

    def _load_fallback_chunks(self):
        """
        โหลดเนื้อหาจาก PDF เป็นชิ้น (chunk) ง่ายๆ เพื่อใช้ค้นหาแบบ lexical
        """
        if self.fallback_loaded:
            return

        path_obj = Path(self.file_path)
        pdf_files: List[Path] = []
        if path_obj.is_file() and path_obj.suffix.lower() == ".pdf":
            pdf_files = [path_obj]
        elif path_obj.is_dir():
            pdf_files = sorted([p for p in path_obj.glob("*.pdf")])

        if not pdf_files:
            self.fallback_loaded = True
            return

        chunks: List[Dict[str, any]] = []
        for pdf_path in pdf_files:
            try:
                with pdfplumber.open(str(pdf_path)) as pdf:
                    for idx, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        if not text.strip():
                            continue
                        for piece in self._split_to_chunks(text, chunk_size=900, overlap=200):
                            chunks.append(
                                {
                                    "text": piece,
                                    "source_file": pdf_path.name,
                                    "page_number": idx + 1,
                                    "collection_name": "fallback_pdf",
                                    "score": 0.0,
                                }
                            )
            except Exception as e:
                logger.warning(f"Cannot read PDF {pdf_path}: {e}")
                continue

        self.fallback_chunks = chunks
        self.fallback_loaded = True
        logger.info(f"Fallback PDF chunks loaded: {len(self.fallback_chunks)}")

    def _split_to_chunks(self, text: str, chunk_size: int = 900, overlap: int = 200) -> List[str]:
        """
        แบ่งข้อความเป็นชิ้นขนาดคงที่เพื่อใช้ค้นหาแบบง่าย
        """
        cleaned = text.strip()
        if not cleaned:
            return []

        chunks = []
        start = 0
        length = len(cleaned)
        while start < length:
            end = min(start + chunk_size, length)
            chunks.append(cleaned[start:end])
            if end == length:
                break
            start = max(0, end - overlap)
        return chunks

    def get_search_results_with_metadata(self, query: str, context: Optional[str] = None) -> List[Dict[str, any]]:
        """
        ดึงผลการค้นหาพร้อม metadata สำหรับ Source Citation
        """
        try:
            self._ensure_initialized()
            
            if not self.initialized:
                return []
            
            self._cleanup_cache()
            processed_query = self._process_thai_text(query)
            
            if not processed_query:
                return []
            
            search_results = self._search_chunks(query, context)
            return search_results
            
        except Exception as e:
            logger.error(f"Error in get_search_results_with_metadata: {str(e)}")
            return []

    def _run(self, query: str, context: Optional[str] = None) -> str:
        """
        รันการค้นหาข้อมูล:
         - ตรวจสอบว่าคำถามมีคีย์เวิร์ด PDPA หรือไม่
         - ค้นหาและส่งผลลัพธ์เป็นข้อความภาษาไทยพร้อม metadata
        """
        try:
            self._ensure_initialized()

        
            if not self.initialized:
                return "เครื่องมือค้นหาเอกสารยังไม่พร้อมใช้งาน กรุณาลองใหม่อีกครั้ง"
            
        
            self._cleanup_cache()
            
            processed_query = self._process_thai_text(query)

            try:
                search_results = self._search_chunks(query, context)
                if search_results:
                 
                    result_parts = []
                    sources = []
                    
                    for i, chunk_data in enumerate(search_results):
                        text = chunk_data.get('text', '')
                        source_file = chunk_data.get('source_file', 'ไม่ระบุไฟล์')
                        page_number = chunk_data.get('page_number', 'ไม่ระบุหน้า')
                        rerank_score = chunk_data.get('rerank_score', None)
                        
                        result_parts.append(text)
                        
                 
                        if source_file != 'ไม่ระบุไฟล์' and page_number != 'ไม่ระบุหน้า':
                            source_citation = f"[{i+1}] {source_file}, หน้า {page_number}"
                        elif source_file != 'ไม่ระบุไฟล์':
                            source_citation = f"[{i+1}] {source_file}"
                        else:
                            source_citation = f"[{i+1}] เอกสารอ้างอิง"
                        
          
                        if rerank_score is not None:
                            source_citation += f" (Rerank Score: {rerank_score:.4f})"
                        
                        sources.append(source_citation)
                    
                
                    result = "\n____\n".join(result_parts)
                    if sources:
                        result += f"\n\n📚 แหล่งที่มา:\n" + "\n".join(sources)
                else:
                    result = "ไม่พบผลลัพธ์ที่เกี่ยวข้อง"
            except Exception as e:
                logger.error(f"Error in search_chunks: {str(e)}")
                logger.error(traceback.format_exc())
                result = "เกิดข้อผิดพลาดในการค้นหาข้อมูล กรุณาลองใหม่อีกครั้ง"
            
       
            self._perform_gc()
            
            return result
        except Exception as e:
            logger.error(f"Error in _run: {str(e)}")
            logger.error(traceback.format_exc())
            return f"เกิดข้อผิดพลาดในการค้นหาข้อมูล: {str(e)}"
    
    def release_resources(self):
        """
        ปล่อยทรัพยากรที่ใช้ในการประมวลผลเอกสาร
        """
        try:
         
            self.image_cache.clear()
            self.query_cache.clear()
            
     
            self.raw_text = ""
            self.chunks = []
            
         
            gc.collect()
            
            logger.info("Resources released successfully")
            return True
        except Exception as e:
            logger.error(f"Error in release_resources: {str(e)}")
            return False

    def _cleanup_cache(self):
        """
        ลบแคชที่หมดอายุเพื่อป้องกันการใช้หน่วยความจำมากเกินไป
        """
        try:
            current_time = time.time()
            if current_time - self.last_cache_cleanup > 300: 
                expired_keys = []
                for key, (timestamp, _) in self.query_cache.items():
                    if current_time - timestamp > self.cache_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.query_cache[key]
                
                self.last_cache_cleanup = current_time
        except Exception as e:
            logger.error(f"Error in _cleanup_cache: {str(e)}")

    def _get_cache_key(self, query: str) -> str:
        """
        สร้างคีย์สำหรับแคชจากคำถาม
        """
        return hashlib.md5(query.encode()).hexdigest()

    def _process_thai_text(self, text: str) -> str:
        """
        Simplified text processing for search-only mode.
        """
        return text or ""

    def _rerank_results(self, query: str, search_results: List[Dict[str, any]], top_k: int = 10) -> List[Dict[str, any]]:
        """
        Rerank search results using BGE reranker
        """
        if not self.use_reranker or not self.reranker or not search_results:
            return search_results[:top_k]
        
        try:
            
            pairs = []
            for result in search_results:
                text = result.get('text', '')
                if text:
                    pairs.append([query, text])
            
            if not pairs:
                return search_results[:top_k]
            
           
            scores = self.reranker.predict(pairs)
            
        
            import numpy as np
            if isinstance(scores, np.ndarray):
                scores = scores.tolist()
            elif isinstance(scores, float):
                scores = [scores]
            elif not isinstance(scores, list):
                scores = [float(scores)]
            
         
            if len(scores) != len(search_results):
                logger.warning(f"Score count mismatch: {len(scores)} scores for {len(search_results)} results")
                scores = scores[:len(search_results)] if len(scores) > len(search_results) else scores + [0.0] * (len(search_results) - len(scores))
            
        
            reranked_results = []
            for i, (result, score) in enumerate(zip(search_results, scores)):
                result_copy = result.copy()
                try:
                    result_copy['rerank_score'] = float(score)
                except (ValueError, TypeError):
                    result_copy['rerank_score'] = 0.0
                reranked_results.append(result_copy)
            
         
            reranked_results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            
            logger.info(f"Reranked {len(reranked_results)} results using BGE reranker")
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.warning(f"Reranking failed, using original results: {e}")
            return search_results[:top_k]

    def is_reranker_available(self) -> bool:
        """
        Check if reranker is available and working
        """
        return self.use_reranker and self.reranker is not None

   

    def _perform_gc(self):
        """
        ทำ garbage collection เป็นระยะเพื่อป้องกันการใช้หน่วยความจำมากเกินไป
        """
        try:
            current_time = time.time()
            if current_time - self.last_gc_time > self.gc_interval:
                gc.collect()
                self.last_gc_time = current_time
        except Exception as e:
            logger.error(f"Error in _perform_gc: {str(e)}")