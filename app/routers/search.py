from fastapi import APIRouter, Depends
from app.schemas import SearchQuery
from app.rag.retriever import Retriever
from app.rag.generator import generate

router = APIRouter(prefix="/search", tags=["Search"])

retriever: Retriever = None

def set_retriever(r: Retriever):
    global retriever
    retriever = r

@router.post("/")
async def search(body: SearchQuery):
    results = await retriever.retrieve(
        query=body.query,
        top_k=body.top_k,
        min_age=body.min_age,
        max_age=body.max_age,
        gender=body.gender,
    )

    answer = await generate(query=body.query, retrieved_results=results)

    return {
        "answer": answer,
        "sources": [
            {
                "id": r["id"],
                "metadata": r["metadata"],
                "rrf_score": r.get("rrf_score"),
            }
            for r in results
        ],
    }

@router.post("/retrieve")
async def retrieve_only(body: SearchQuery):
    """Return raw retrieved documents without LLM generation."""
    results = await retriever.retrieve(
        query=body.query,
        top_k=body.top_k,
        min_age=body.min_age,
        max_age=body.max_age,
        gender=body.gender,
    )
    return {"results": results}
