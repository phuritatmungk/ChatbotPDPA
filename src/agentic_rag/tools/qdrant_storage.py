from typing import Optional, List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue, Distance, VectorParams
from sentence_transformers import SentenceTransformer
import hashlib
import uuid

class MyEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.vector_size = self.model.get_sentence_embedding_dimension()

    def encode(self, text: str):
        return self.model.encode(text).tolist()

class QdrantStorage:
    """
    Handles embeddings for memory entries using Qdrant.
    """
    def __init__(
        self,
        type: str,
        qdrant_location: str,
        qdrant_api_key: str,
        embedder: Optional[MyEmbedder] = None,
        collection_name: Optional[str] = None,
        allow_create: bool = True,
    ):
        self.type = type
        self.embedder = embedder or MyEmbedder()
        self.client = QdrantClient(
            url=qdrant_location,
            api_key=qdrant_api_key,
            timeout=60,
            prefer_grpc=False,
        )
     
        self.collection_name = collection_name or f"rag_{type}"
        self.vector_size = self.embedder.vector_size
        self._allow_create = allow_create
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            if not self._allow_create:

                return
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )

    def add(self, chunk: dict):
        vector = self.embedder.encode(chunk['text'])
        point_id = chunk.get('id')
  
        if isinstance(point_id, int):
            valid_id = point_id
        else:
            stable_hash = self._generate_id(chunk)
            valid_id = str(uuid.UUID(hex=stable_hash[:32]))
        self.client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=valid_id,
                vector=vector,
                payload=chunk
            )]
        )

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0,
    ) -> List[Dict[str, Any]]:
        vector = self.embedder.encode(query)
        qdrant_filter = None
        if filter:
            qdrant_filter = Filter(
                must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filter.items()]
            )
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=limit,
            query_filter=qdrant_filter,
            score_threshold=score_threshold
        )
        return [r.payload for r in results]

    def reset(self) -> None:
        self.client.delete_collection(self.collection_name)
        self._ensure_collection()

    def _generate_id(self, chunk: dict) -> str:
        return hashlib.sha1(chunk['text'].encode('utf-8')).hexdigest()

    def has_data(self) -> bool:
        """
        Returns True if the Qdrant collection contains any points, False otherwise.
        """
        try:
            if not self.client.collection_exists(self.collection_name):
                return False
          
            count_result = self.client.count(self.collection_name, exact=False)
      
            total = getattr(count_result, 'count', None)
            if total is None:
             
                try:
                    total = count_result.get('count', 0)
                except Exception:
                    total = 0
            return (total or 0) > 0
        except Exception:
            return False 