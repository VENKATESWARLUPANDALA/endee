from pydantic import BaseModel
from typing import Any, Optional, List, Dict


class CreateIndexRequest(BaseModel):
    name: str
    dim: int
    metric: str = "cosine"
    ef_construction: int = 200
    M: int = 16

class IndexInfo(BaseModel):
    name: str
    dim: int
    metric: str
    count: int
    ef_construction: int
    M: int
    created_at: float

class Document(BaseModel):
    id: str
    vector: List[float]
    sparse_vector: Optional[Dict[str, float]] = None
    payload: Optional[Dict[str, Any]] = None

class UpsertRequest(BaseModel):
    documents: List[Document]

class UpsertResponse(BaseModel):
    upserted: int
    index: str

class FilterCondition(BaseModel):
    field: str
    op: str
    value: Any

class SearchRequest(BaseModel):
    vector: List[float]
    sparse_vector: Optional[Dict[str, float]] = None
    top_k: int = 10
    filters: Optional[List[FilterCondition]] = None
    alpha: float = 0.7
    ef_search: int = 100
    include_payload: bool = True

class SearchHit(BaseModel):
    id: str
    score: float
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    payload: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    hits: List[SearchHit]
    total: int
    query_time_ms: float
    index: str
    mode: str

class StatsResponse(BaseModel):
    total_indexes: int
    total_documents: int
    indexes: List[IndexInfo]
    uptime_seconds: float
    server: str = "Endee Vector DB v1.0"