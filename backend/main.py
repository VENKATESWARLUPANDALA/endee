"""
Endee Vector DB — FastAPI REST Server
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    CreateIndexRequest, IndexInfo,
    UpsertRequest, UpsertResponse,
    SearchRequest, SearchResponse,
    StatsResponse,
)
from vector_db import EndeeDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("endee")

db = EndeeDB()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Endee Vector DB starting up")
    yield
    logger.info("🛑 Endee Vector DB shutting down")


app = FastAPI(
    title="Endee Vector DB",
    description="High-performance open-source vector database for AI search and retrieval.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "uptime_seconds": round(db.uptime, 2)}


@app.get("/stats", tags=["System"])
def stats():
    return {
        "total_indexes": len(db.list_indexes()),
        "total_documents": db.total_documents,
        "indexes": [i.model_dump() for i in db.list_indexes()],
        "uptime_seconds": round(db.uptime, 2),
        "server": "Endee Vector DB v1.0"
    }


# ── Indexes ────────────────────────────────────────────────────────────────────

@app.post("/indexes", status_code=201, tags=["Indexes"])
def create_index(req: CreateIndexRequest):
    try:
        idx = db.create_index(req)
        return idx.info().model_dump()
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/indexes", tags=["Indexes"])
def list_indexes():
    return [i.model_dump() for i in db.list_indexes()]


@app.get("/indexes/{index_name}", tags=["Indexes"])
def get_index(index_name: str):
    try:
        return db.get_index(index_name).info().model_dump()
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/indexes/{index_name}", tags=["Indexes"])
def delete_index(index_name: str):
    if not db.delete_index(index_name):
        raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")
    return {"deleted": index_name}


# ── Documents ──────────────────────────────────────────────────────────────────

@app.post("/indexes/{index_name}/upsert", tags=["Documents"])
def upsert(index_name: str, req: UpsertRequest):
    try:
        idx = db.get_index(index_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    try:
        n = idx.upsert(req.documents)
        logger.info(f"Upserted {n} docs into '{index_name}'")
        return {"upserted": n, "index": index_name}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.delete("/indexes/{index_name}/documents/{doc_id}", tags=["Documents"])
def delete_document(index_name: str, doc_id: str):
    try:
        idx = db.get_index(index_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if not idx.delete(doc_id):
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return {"deleted": doc_id, "index": index_name}


# ── Search ─────────────────────────────────────────────────────────────────────

@app.post("/indexes/{index_name}/search", tags=["Search"])
def search(index_name: str, req: SearchRequest):
    try:
        idx = db.get_index(index_name)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    t0 = time.perf_counter()
    try:
        hits, mode = idx.search(
            query_vector=req.vector,
            query_sparse=req.sparse_vector,
            top_k=req.top_k,
            filters=req.filters,
            alpha=req.alpha,
            ef_search=req.ef_search,
            include_payload=req.include_payload,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    logger.info(f"Search '{index_name}' mode={mode} hits={len(hits)} time={elapsed_ms}ms")

    return {
        "hits": [h.model_dump() for h in hits],
        "total": len(hits),
        "query_time_ms": elapsed_ms,
        "index": index_name,
        "mode": mode,
    }


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)