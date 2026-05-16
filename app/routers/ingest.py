from fastapi import APIRouter
from app.rag.store import VectorStore
from app.rag.ingest import ingest_csv
from app.routers.search import retriever

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

store: VectorStore = None

def set_store(s: VectorStore):
    global store
    store = s

@router.post("/")
async def ingest(csv_path: str = "app/rag/data/preprocessed.csv"):
    result = await ingest_csv(csv_path=csv_path, store=store)

    # Rebuild BM25 index after new data is ingested
    if retriever is not None:
        retriever.build_bm25_index()

    return {"status": "success", **result}
