from pydantic import BaseModel

class SearchQueryResult(BaseModel):
    doc_id: str
    text: str
    score: float