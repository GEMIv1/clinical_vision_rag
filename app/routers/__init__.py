from app.routers.ocr import router as ocr_router
from app.routers.search import router as search_router
from app.routers.ingest import router as ingest_router

__all__ = ["ocr_router", "search_router", "ingest_router"]
