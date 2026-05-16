import asyncio
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from app.config import settings

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.EMBED_MODEL)

async def embed(text: str) -> list[float]:
    loop = asyncio.get_event_loop()
    model = get_model()
    result = await loop.run_in_executor(None, lambda: model.encode([text]))
    return result[0].tolist()

async def embed_batch(texts: list[str]) -> list[list[float]]:
    loop = asyncio.get_event_loop()
    model = get_model()
    result = await loop.run_in_executor(None, lambda: model.encode(texts, batch_size=64, show_progress_bar=True))
    return result.tolist()