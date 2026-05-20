import os
import io
import hashlib
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading

import pdfplumber
import fitz  # PyMuPDF
from typhoon_ocr import ocr_document
from PIL import Image, ImageEnhance
import cv2
import numpy as np

from src.agentic_rag.tools.qdrant_storage import QdrantStorage, MyEmbedder
from dotenv import load_dotenv


load_dotenv()

_ocr_cache = {}
_cache_lock = threading.Lock()


OCR_CONFIG = {
    'max_workers': 4,
    'fast_mode': True,
    'image_resize_threshold': 1500,
    'denoise_threshold': 500000,
    'contrast_enhancement': 1.5,
    'brightness_enhancement': 1.1,
    'dpi': 300,
}

@lru_cache(maxsize=100)
def _extract_text_with_typhoon_ocr(image_path: str, page_num: Optional[int] = None) -> str:
    """Extract text from image or PDF using typhoon-ocr with caching."""
    try:
        cache_key = f"{image_path}_{page_num}"
        
        with _cache_lock:
            if cache_key in _ocr_cache:
                return _ocr_cache[cache_key]
        
        import os
        os.environ['TYPHOON_OCR_API_KEY'] = 'sk-Cf5igH8dpj3LobZVlKACkrsCLqGvrKgm4KLt0zgt4sW85icr'
        
        if image_path.lower().endswith('.pdf'):
            task_type = "default"
        else:
            task_type = "structure"
        
        markdown = ocr_document(
            pdf_or_image_path=image_path,
            task_type=task_type,
            page_num=page_num
        )
        
        result = markdown if markdown else ""
        
        with _cache_lock:
            _ocr_cache[cache_key] = result
        
        return result
        
    except Exception as e:
        print(f"Error in typhoon-ocr processing: {e}")
        return ""


def _preprocess_image(image: Image.Image) -> Image.Image:
    """
    ปรับปรุงคุณภาพของภาพก่อนทำ OCR เพื่อเพิ่มความแม่นยำ (เวอร์ชันเร็ว)
    """
    if image.mode != 'L':
        image = image.convert('L')
    
    width, height = image.size
    threshold = OCR_CONFIG['image_resize_threshold']
    if width > threshold or height > threshold:
        scale = min(threshold/width, threshold/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(OCR_CONFIG['contrast_enhancement'])
    
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(OCR_CONFIG['brightness_enhancement'])
    
    img_array = np.array(image)
    
    if OCR_CONFIG['fast_mode'] and img_array.size > OCR_CONFIG['denoise_threshold']:
        img_array = cv2.fastNlMeansDenoising(img_array, h=10)
    
    return Image.fromarray(img_array)


def _extract_embedded_images(doc: fitz.Document):
    images = {}
    for page_num, page in enumerate(doc):
        page_images = []
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image")
                if image_bytes:
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    page_images.append(pil_image)
            except Exception:
                continue
        if page_images:
            images[page_num] = page_images
    return images


def extract_text_from_pdf_with_metadata(path: str, ocr_lang: str = "tha+eng") -> Tuple[str, Dict[int, str]]:
    """
    สกัดข้อความจาก PDF พร้อมเก็บข้อมูลหน้าที่ map กับข้อความ
    
    Returns:
        Tuple[str, Dict[int, str]]: (ข้อความรวม, dict ที่ map หน้าที่กับข้อความ)
    """
    def extract_with_pdfplumber_metadata(pdf_path: str) -> Tuple[str, Dict[int, str]]:
        text_pp = ""
        page_texts = {}
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_pp += page_text + "\n"
                        page_texts[page_num + 1] = page_text
                    else:
                        try:
                            ocr_text = _extract_text_with_typhoon_ocr(pdf_path, page_num + 1)
                            if ocr_text.strip():
                                text_pp += ocr_text + "\n"
                                page_texts[page_num + 1] = ocr_text
                        except Exception:
                            pass
        except Exception:
            pass
        return text_pp, page_texts

    def extract_with_fitz_metadata(pdf_path: str) -> Tuple[str, Dict[int, str]]:
        text_fitz = ""
        page_texts = {}
        try:
            doc = fitz.open(pdf_path)
            embedded = _extract_embedded_images(doc)
            for page_num, page in enumerate(doc):
                page_text = (page.get_text() or "").strip()
                if not page_text:
                    try:
                        page_images = embedded.get(page_num, [])
                        if page_images:
                            for img_idx, img_data in enumerate(page_images):
                                try:
                                    img_path = f"/tmp/page_{page_num}_img_{img_idx}.png"
                                    img_data.save(img_path)
                                    ocr_text = _extract_text_with_typhoon_ocr(img_path)
                                    if ocr_text.strip():
                                        page_text += ocr_text + "\n"
                                    os.unlink(img_path)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                if page_text:
                    text_fitz += page_text + "\n"
                    page_texts[page_num + 1] = page_text
            doc.close()
        except Exception:
            pass
        return text_fitz, page_texts

    text_pp, page_texts_pp = extract_with_pdfplumber_metadata(path)
    text_fitz, page_texts_fitz = extract_with_fitz_metadata(path)
    
    if len(text_fitz) > len(text_pp):
        return text_fitz, page_texts_fitz
    else:
        return text_pp, page_texts_pp

def extract_text_from_pdf(path: str, ocr_lang: str = "tha+eng") -> str:
    def extract_with_pdfplumber(pdf_path: str) -> str:
        text_pp = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_pp += page_text + "\n"
                    else:
                        try:
                            ocr_text = _extract_text_with_typhoon_ocr(pdf_path, page_num + 1)
                            if ocr_text.strip():
                                text_pp += ocr_text + "\n"
                        except Exception:
                            pass
        except Exception:
            pass
        return text_pp

    def extract_with_fitz(pdf_path: str) -> str:
        text_fitz = ""
        try:
            doc = fitz.open(pdf_path)
            embedded = _extract_embedded_images(doc)
            for page_num, page in enumerate(doc):
                page_text = (page.get_text() or "").strip()
                if not page_text:
                    try:
                        page_images = embedded.get(page_num, [])
                        if page_images:
                            for img_idx, img in enumerate(page_images):
                                temp_path = f"temp_page_{page_num}_img_{img_idx}.png"
                                img.save(temp_path)
                                ocr_text = _extract_text_with_typhoon_ocr(temp_path)
                                if ocr_text.strip():
                                    page_text += ocr_text + "\n"
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                        else:
                            ocr_text = _extract_text_with_typhoon_ocr(pdf_path, page_num + 1)
                            page_text = ocr_text
                    except Exception:
                        page_text = ""
                if page_text:
                    text_fitz += page_text + "\n"
            doc.close()
        except Exception:
            pass
        return text_fitz

    return extract_with_pdfplumber(path) + "\n" + extract_with_fitz(path)


def extract_text_from_path(path: str) -> str:
    if os.path.isdir(path):
        raise ValueError("Directory provided. Use extract_texts_from_dir() for per-file ingestion.")
    print(f"Extracting single file: {path}")
    text = extract_text_from_pdf(path)
    print(f"Extracted text length: {len(text)} chars")
    return text


def extract_texts_from_dir(path: str) -> List[Tuple[str, str]]:
    pdfs = [f for f in sorted(os.listdir(path)) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdfs)} PDF(s) under: {path}")
    results: List[Tuple[str, str]] = []
    total_len = 0
    for filename in pdfs:
        full = os.path.join(path, filename)
        print(f"Extracting: {filename}")
        text = extract_text_from_pdf(full)
        results.append((filename, text))
        total_len += len(text)
    print(f"Total extracted text length (all files): {total_len} chars")
    return results


def chunk_text_semantically(raw_text: str, source_file: str = "ไม่ระบุไฟล์", page_info: Dict[int, str] = None) -> List[Dict[str, any]]:
    """
    แบ่งข้อความเป็น chunks พร้อม metadata สำหรับ Source Citation
    
    Args:
        raw_text: ข้อความที่จะแบ่ง
        source_file: ชื่อไฟล์ต้นฉบับ
        page_info: ข้อมูลหน้าที่ map กับข้อความ (optional)
    
    Returns:
        List[Dict] ที่มี text, source_file, page_number, chunk_id
    """
    try:
        from chonkie import SemanticChunker
        import os
        chunk_model = os.getenv(
            "RAG_CHUNK_EMBED_MODEL",
            os.getenv("RAG_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        )
        sim_threshold = float(os.getenv("RAG_CHUNK_SIM_THRESHOLD", "0.68"))
        chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "128"))
        min_sentences = int(os.getenv("RAG_CHUNK_MIN_SENTENCES", "2"))
        min_chars_keep = int(os.getenv("RAG_MIN_CHUNK_CHARS", "60"))
        chunker = SemanticChunker(
            embedding_model=chunk_model,
            threshold=sim_threshold,
            chunk_size=chunk_size,
            min_sentences=min_sentences,
        )
        raw_chunks = chunker.chunk(raw_text)
        chunks = []
        for idx, ch in enumerate(raw_chunks):
            text = ch["text"] if isinstance(ch, dict) else (ch if isinstance(ch, str) else getattr(ch, "text", ""))
            if text and len(text.strip()) >= min_chars_keep:
                chunks.append({"id": idx, "text": text})
        normalized = []
        for idx, ch in enumerate(chunks):
            if isinstance(ch, dict) and "text" in ch:
                text = ch["text"]
            elif isinstance(ch, str):
                text = ch
            else:
                text = getattr(ch, "text", None) or ""
            if text.strip():
                page_number = "ไม่ระบุหน้า"
                if page_info:
                    best_page = 1
                    max_overlap = 0
                    for page_num, page_text in page_info.items():
                        chunk_words = set(text.lower().split())
                        page_words = set(page_text.lower().split())
                        overlap = len(chunk_words.intersection(page_words))
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_page = page_num
                    if max_overlap > 0:
                        page_number = str(best_page)
                
                chunk_data = {
                    "id": idx,
                    "text": text,
                    "source_file": source_file,
                    "page_number": page_number,
                    "chunk_id": f"{source_file}_{idx}"
                }
                normalized.append(chunk_data)
        return normalized
    except Exception:
        parts = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        return [{
            "id": i, 
            "text": p, 
            "source_file": source_file,
            "page_number": "ไม่ระบุหน้า",
            "chunk_id": f"{source_file}_{i}"
        } for i, p in enumerate(parts)]


def upload_chunks_to_qdrant(
    chunks: List[Dict[str, any]],
    collection_suffix: str,
    qdrant_url: str,
    qdrant_api_key: Optional[str] = None,
) -> str:
    content_hash = hashlib.md5("\n".join([c.get("text", "") for c in chunks]).encode("utf-8")).hexdigest()
    collection_name = f"doc_{collection_suffix}_{content_hash[:8]}"
    print(f"Connecting to Qdrant: {qdrant_url}")
    storage = QdrantStorage(
        type=collection_name,
        qdrant_location=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        embedder=MyEmbedder(os.getenv("RAG_EMBED_MODEL")),
    )
    print(f"Using collection: {storage.collection_name}")
    success = 0
    failed = 0
    for ch in chunks:
        try:
            storage.add(ch)
            success += 1
        except Exception as e:
            failed += 1
            print(f"Failed to upsert chunk id={ch.get('id')}: {e}")
    try:
        count_result = storage.client.count(storage.collection_name, exact=False)
        total = getattr(count_result, 'count', None)
        if total is None:
            try:
                total = count_result.get('count', 0)
            except Exception:
                total = 0
        print(f"Upload done. Success={success}, Failed={failed}, Collection points now={total}")
    except Exception as e:
        print(f"Could not query collection count: {e}")
    return storage.collection_name


def ingest_path_to_qdrant_with_metadata(
    path: str,
    collection_suffix: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> str:
    """
    Ingest ไฟล์หรือโฟลเดอร์ไปยัง Qdrant พร้อม metadata สำหรับ Source Citation
    """
    url = (
        qdrant_url
        or os.getenv("QDRANT_URL")
        or "http://localhost:6333"
    )
    key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

    if os.path.isdir(path):
        created: List[str] = []
        for filename in sorted(os.listdir(path)):
            if not filename.lower().endswith(".pdf"):
                continue
            full_path = os.path.join(path, filename)
            print(f"Processing: {filename}")
            
            raw_text, page_info = extract_text_from_pdf_with_metadata(full_path)
            if not raw_text.strip():
                print(f"No text extracted from {filename}. Skipping upload.")
                continue
            
            chunks = chunk_text_semantically(raw_text, source_file=filename, page_info=page_info)
            print(f"{filename}: Chunked into {len(chunks)} segment(s)")
            base = os.path.splitext(os.path.basename(filename))[0]
            safe = ''.join(ch if ch.isalnum() or ch in ['-', '_'] else '_' for ch in base)
            suffix = collection_suffix or safe
            collection = upload_chunks_to_qdrant(chunks, suffix, url, key)
            print(collection)
            created.append(collection)
        return created[-1] if created else ""
    else:
        print(f"Processing: {path}")
        
        raw_text, page_info = extract_text_from_pdf_with_metadata(path)
        if not raw_text.strip():
            print("No text extracted. Skipping upload.")
            return ""
        
        source_file = os.path.basename(path) or "uploaded"
        chunks = chunk_text_semantically(raw_text, source_file=source_file, page_info=page_info)
        print(f"Chunked into {len(chunks)} segment(s)")
        suffix = collection_suffix or source_file
        return upload_chunks_to_qdrant(chunks, suffix, url, key)

def ingest_path_to_qdrant(
    path: str,
    collection_suffix: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> str:
    url = (
        qdrant_url
        or os.getenv("QDRANT_URL")
        or "http://localhost:6333"
    )
    key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

    if os.path.isdir(path):
        created: List[str] = []
        for filename, raw_text in extract_texts_from_dir(path):
            if not raw_text.strip():
                print(f"No text extracted from {filename}. Skipping upload.")
                continue
            chunks = chunk_text_semantically(raw_text, source_file=filename)
            print(f"{filename}: Chunked into {len(chunks)} segment(s)")
            base = os.path.splitext(os.path.basename(filename))[0]
            safe = ''.join(ch if ch.isalnum() or ch in ['-', '_'] else '_' for ch in base)
            suffix = collection_suffix or safe
            collection = upload_chunks_to_qdrant(chunks, suffix, url, key)
            print(collection)
            created.append(collection)
        return created[-1] if created else ""
    else:
        raw_text = extract_text_from_path(path)
        if not raw_text.strip():
            print("No text extracted. Skipping upload.")
            return ""
        source_file = os.path.basename(path) or "uploaded"
        chunks = chunk_text_semantically(raw_text, source_file=source_file)
        print(f"Chunked into {len(chunks)} segment(s)")
        suffix = collection_suffix or source_file
        return upload_chunks_to_qdrant(chunks, suffix, url, key)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract, chunk and upload PDFs to Qdrant Cloud with Source Citation metadata")
    parser.add_argument("--path", required=True, help="Path to a PDF file or a directory containing PDFs")
    parser.add_argument("--suffix", default=None, help="Optional collection suffix label")
    parser.add_argument("--qdrant-url", dest="qdrant_url", default=None, help="Qdrant URL (fallback to env QDRANT_URL)")
    parser.add_argument("--qdrant-api-key", dest="qdrant_api_key", default=None, help="Qdrant API key (fallback to env QDRANT_API_KEY)")
    parser.add_argument("--with-metadata", action="store_true", help="Use enhanced metadata extraction (recommended for Source Citation)")
    args = parser.parse_args()

    if args.with_metadata:
        print("🚀 ใช้ระบบ Source Citation metadata")
        collection = ingest_path_to_qdrant_with_metadata(
            path=args.path,
            collection_suffix=args.suffix,
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
        )
    else:
        print("📚 ใช้ระบบ ingest แบบเดิม")
        collection = ingest_path_to_qdrant(
            path=args.path,
            collection_suffix=args.suffix,
            qdrant_url=args.qdrant_url,
            qdrant_api_key=args.qdrant_api_key,
        )
    print(f"Collection created: {collection}")


