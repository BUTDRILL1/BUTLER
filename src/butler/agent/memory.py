import json
import sqlite3
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
    def __init__(self, db_path: str, embedding_provider: Any):
        self.db_path = db_path
        self.provider = embedding_provider
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_chunks (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    embedding BLOB,
                    created_at REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_created ON memory_chunks(created_at)")

    def add(self, content: str, metadata: Dict[str, Any] = None) -> str:
        chunk_id = f"mem_{int(time.time() * 1000)}"
        embedding = self.provider.get_embedding(content)
        
        # Store embedding as binary for SQLite
        emb_blob = np.array(embedding, dtype=np.float32).tobytes() if embedding else None
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO memory_chunks (id, content, metadata, embedding, created_at) VALUES (?, ?, ?, ?, ?)",
                (chunk_id, content, json.dumps(metadata or {}), emb_blob, time.time())
            )
        return chunk_id

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.provider.get_embedding(query)
        if not query_embedding:
            return []
            
        query_vec = np.array(query_embedding, dtype=np.float32)
        
        results = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT id, content, metadata, embedding, created_at FROM memory_chunks").fetchall()
            
            for row in rows:
                if row["embedding"]:
                    stored_vec = np.frombuffer(row["embedding"], dtype=np.float32)
                    # Simple cosine similarity: (A . B) / (||A|| * ||B||)
                    # For normalized embeddings, this is just A . B
                    score = np.dot(query_vec, stored_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(stored_vec))
                    
                    results.append({
                        "id": row["id"],
                        "content": row["content"],
                        "metadata": json.loads(row["metadata"]),
                        "score": float(score),
                        "created_at": row["created_at"]
                    })
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
