"""
Endee Vector DB — Core Engine
Implements: HNSW-based dense ANN (via usearch), sparse retrieval,
            hybrid fusion (RRF), and payload filtering.
"""

import time
import threading
import numpy as np
from typing import Any, Optional

from usearch.index import Index as USearchIndex, MetricKind

from models import (
    CreateIndexRequest, IndexInfo, Document,
    FilterCondition, SearchHit
)


# ── Payload Filter ─────────────────────────────────────────────────────────────

def _eval_filter(payload: dict, condition: FilterCondition) -> bool:
    val = payload.get(condition.field)
    if val is None:
        return False
    op, cv = condition.op, condition.value
    try:
        if op == "eq":       return val == cv
        if op == "ne":       return val != cv
        if op == "gt":       return val > cv
        if op == "gte":      return val >= cv
        if op == "lt":       return val < cv
        if op == "lte":      return val <= cv
        if op == "in":       return val in cv
        if op == "contains": return cv in val
    except TypeError:
        return False
    return False


def _passes_filters(payload, filters):
    if not filters:
        return True
    if payload is None:
        return False
    return all(_eval_filter(payload, f) for f in filters)


# ── Sparse ─────────────────────────────────────────────────────────────────────

def _sparse_score(doc_sparse, query_sparse):
    return sum(qw * doc_sparse[t] for t, qw in query_sparse.items() if t in doc_sparse)


# ── RRF ────────────────────────────────────────────────────────────────────────

def _rrf_fusion(dense_hits, sparse_hits, alpha, k=60):
    scores = {}
    for rank, (doc_id, _) in enumerate(dense_hits):
        scores[doc_id] = scores.get(doc_id, 0.0) + alpha * (1.0 / (k + rank + 1))
    for rank, (doc_id, _) in enumerate(sparse_hits):
        scores[doc_id] = scores.get(doc_id, 0.0) + (1 - alpha) * (1.0 / (k + rank + 1))
    return scores


# ── Metric map ─────────────────────────────────────────────────────────────────

_METRIC_MAP = {
    "cosine": MetricKind.Cos,
    "l2":     MetricKind.L2sq,
    "ip":     MetricKind.IP,
}


# ── VectorIndex ────────────────────────────────────────────────────────────────

class VectorIndex:
    def __init__(self, config: CreateIndexRequest):
        self.name = config.name
        self.dim = config.dim
        self.metric = config.metric
        self.ef_construction = config.ef_construction
        self.M = config.M
        self.created_at = time.time()
        self._lock = threading.RLock()

        self._id_to_int: dict[str, int] = {}
        self._int_to_id: dict[int, str] = {}
        self._next_int = 0
        self._payloads: dict[str, Optional[dict]] = {}
        self._sparse_vectors: dict[str, dict[str, float]] = {}

        self._index = USearchIndex(
            ndim=config.dim,
            metric=_METRIC_MAP.get(config.metric, MetricKind.Cos),
            connectivity=config.M,
            expansion_add=config.ef_construction,
            expansion_search=max(50, config.ef_construction),
        )

    @property
    def count(self):
        return len(self._id_to_int)

    def info(self) -> IndexInfo:
        return IndexInfo(
            name=self.name, dim=self.dim, metric=self.metric,
            count=self.count, ef_construction=self.ef_construction,
            M=self.M, created_at=self.created_at
        )

    def upsert(self, docs: list[Document]) -> int:
        with self._lock:
            for doc in docs:
                if len(doc.vector) != self.dim:
                    raise ValueError(f"Doc '{doc.id}' has dim {len(doc.vector)}, expected {self.dim}")
                if doc.id not in self._id_to_int:
                    iid = self._next_int
                    self._next_int += 1
                    self._id_to_int[doc.id] = iid
                    self._int_to_id[iid] = doc.id
                else:
                    iid = self._id_to_int[doc.id]
                self._payloads[doc.id] = doc.payload
                if doc.sparse_vector:
                    self._sparse_vectors[doc.id] = doc.sparse_vector
                self._index.add(iid, np.array(doc.vector, dtype=np.float32))
            return len(docs)

    def search(self, query_vector, query_sparse, top_k, filters, alpha, ef_search, include_payload):
        with self._lock:
            if self.count == 0:
                return [], "dense"

            q = np.array(query_vector, dtype=np.float32)
            fetch_k = min(self.count, top_k * 10 if filters else top_k * 2)
            results = self._index.search(q, fetch_k)

            dense_hits = []
            for key, dist in zip(results.keys, results.distances):
                doc_id = self._int_to_id.get(int(key))
                if doc_id is None:
                    continue
                if not _passes_filters(self._payloads.get(doc_id), filters or []):
                    continue
                score = float(1 - dist) if self.metric == "cosine" else float(-dist)
                dense_hits.append((doc_id, score))

            sparse_hits = []
            mode = "dense"
            if query_sparse and (1 - alpha) > 0:
                mode = "hybrid"
                raw = []
                for doc_id, sv in self._sparse_vectors.items():
                    if not _passes_filters(self._payloads.get(doc_id), filters or []):
                        continue
                    sc = _sparse_score(sv, query_sparse)
                    if sc > 0:
                        raw.append((doc_id, sc))
                raw.sort(key=lambda x: -x[1])
                sparse_hits = raw[: top_k * 3]

            if mode == "hybrid":
                fused = _rrf_fusion(dense_hits, sparse_hits, alpha)
                dm, sm = dict(dense_hits), dict(sparse_hits)
                ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
                hits = [
                    SearchHit(
                        id=did, score=round(sc, 6),
                        dense_score=round(dm.get(did, 0.0), 6),
                        sparse_score=round(sm.get(did, 0.0), 6),
                        payload=self._payloads.get(did) if include_payload else None,
                    )
                    for did, sc in ranked
                ]
            else:
                dense_hits.sort(key=lambda x: -x[1])
                hits = [
                    SearchHit(
                        id=did, score=round(sc, 6), dense_score=round(sc, 6),
                        payload=self._payloads.get(did) if include_payload else None,
                    )
                    for did, sc in dense_hits[:top_k]
                ]
            return hits, mode

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self._id_to_int:
                return False
            iid = self._id_to_int.pop(doc_id)
            self._int_to_id.pop(iid, None)
            self._payloads.pop(doc_id, None)
            self._sparse_vectors.pop(doc_id, None)
            self._index.remove(iid)
            return True


# ── EndeeDB ────────────────────────────────────────────────────────────────────

class EndeeDB:
    def __init__(self):
        self._indexes: dict[str, VectorIndex] = {}
        self._lock = threading.Lock()
        self.started_at = time.time()

    def create_index(self, config: CreateIndexRequest) -> VectorIndex:
        with self._lock:
            if config.name in self._indexes:
                raise ValueError(f"Index '{config.name}' already exists")
            idx = VectorIndex(config)
            self._indexes[config.name] = idx
            return idx

    def get_index(self, name: str) -> VectorIndex:
        idx = self._indexes.get(name)
        if idx is None:
            raise KeyError(f"Index '{name}' not found")
        return idx

    def delete_index(self, name: str) -> bool:
        with self._lock:
            if name not in self._indexes:
                return False
            del self._indexes[name]
            return True

    def list_indexes(self):
        return [idx.info() for idx in self._indexes.values()]

    @property
    def uptime(self):
        return time.time() - self.started_at

    @property
    def total_documents(self):
        return sum(idx.count for idx in self._indexes.values())