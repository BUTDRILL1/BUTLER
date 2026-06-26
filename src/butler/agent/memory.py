import json
import time
from typing import Any, List, Dict, Optional
import numpy as np
from pydantic import BaseModel

class MemoryChunk(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: float

class MemoryStore:
    def __init__(self, db: Any, embedding_provider: Any):
        self.db = db
        self.provider = embedding_provider

    def add(self, content: str, metadata: Dict[str, Any] = None) -> str:
        chunk_id = f"mem_{int(time.time() * 1000)}"
        embedding = self.provider.get_embedding(content)
        
        # Store embedding as JSON array string for Supabase
        emb_str = json.dumps(embedding) if embedding else None
        
        self.db.client.table("memory_chunks").insert({
            "id": chunk_id,
            "content": content,
            "metadata": json.dumps(metadata or {}),
            "embedding": emb_str,
            "created_at_ms": int(time.time() * 1000)
        }).execute()
        
        return chunk_id

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.provider.get_embedding(query)
        if not query_embedding:
            return []
            
        query_vec = np.array(query_embedding, dtype=np.float32)
        
        results = []
        
        # Fetch all memory chunks from Supabase
        response = self.db.client.table("memory_chunks").select("*").execute()
        rows = response.data
        
        for row in rows:
            if row.get("embedding"):
                try:
                    stored_emb = json.loads(row["embedding"])
                    stored_vec = np.array(stored_emb, dtype=np.float32)
                    # Simple cosine similarity
                    score = np.dot(query_vec, stored_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec))
                    
                    results.append({
                        "id": row["id"],
                        "content": row["content"],
                        "metadata": json.loads(row["metadata"]) if row.get("metadata") else {},
                        "score": float(score),
                        "created_at": row.get("created_at_ms", 0) / 1000.0
                    })
                except Exception as e:
                    continue
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
