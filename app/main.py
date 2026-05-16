from contextlib import asynccontextmanager
from fastapi import FastAPI
import httpx
from .config import settings
from .rag.store import VectorStore
from .rag.retriever import Retriever
from .routers import ocr_router, search_router, ingest_router
from .routers.search import set_retriever
from .routers.ingest import set_store

@asynccontextmanager
async def lifespan(app: FastAPI):
    store = VectorStore()
    retriever = Retriever(store)
    doc_count = store.collection.count()
    if doc_count > 0:
        print(f"Building BM25 index from {doc_count} existing documents...")
        retriever.build_bm25_index()
        print("BM25 index ready.")
    else:
        print("ChromaDB is empty. Run POST /ingest to load data first.")

    set_retriever(retriever)
    set_store(store)
    yield

app = FastAPI(title="Vision RAG", version="0.1.0", lifespan=lifespan)

app.include_router(ocr_router)
app.include_router(search_router)
app.include_router(ingest_router)

@app.get("/")
async def root():
    return {
        "service": "vision_rag",
        "primary_model": settings.MODEL_NAME,
        "backup_model": settings.BACKUP_MODEL_NAME,
    }

@app.get("/health")
async def health():
    primary = await _check_model(settings.MODEL_ENDPOINT, settings.MODEL_NAME)
    backup = await _check_model(settings.BACKUP_MODEL_ENDPOINT, settings.BACKUP_MODEL_NAME)

    primary_ok = primary["status"] == "online"
    backup_ok = backup["status"] == "online"

    if primary_ok and backup_ok:
        overall = "healthy"
    elif primary_ok or backup_ok:
        overall = "degraded"
    else:
        overall = "unhealthy"

    return {
        "status": overall,
        "primary_model": primary,
        "backup_model": backup,
    }


async def _check_model(endpoint: str, model_name: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{endpoint}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if model_name in models:
                return {"status": "online", "model": model_name}
            return {
                "status": "online (model not loaded)",
                "model": model_name,
                "available_models": models,
            }
    except httpx.ConnectError:
        return {"status": "offline", "model": model_name, "error": "connection refused"}
    except Exception as exc:
        return {"status": "offline", "model": model_name, "error": str(exc)}
