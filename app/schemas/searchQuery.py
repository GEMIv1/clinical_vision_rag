from pydantic import BaseModel, Field
from typing import Optional

class SearchQuery(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    min_age: Optional[float] = None
    max_age: Optional[float] = None
    gender: Optional[str] = None