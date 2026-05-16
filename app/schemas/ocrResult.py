from pydantic import BaseModel, Field
from typing import Optional

class OCRResult(BaseModel):
    text: str
    doc_id: Optional[str] = None
    tokens_used: Optional[int] = None