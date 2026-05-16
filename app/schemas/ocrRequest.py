from pydantic import BaseModel, Field
from typing import Optional
from app.schemas.outputFormat import OutputFormat

class OCRRequest(BaseModel):
    prompt: Optional[str] = Field(default=None, description="Custom instruction override. Fall back to default system prompt if none.")
    output_format: OutputFormat = OutputFormat.plain
    language_hint: Optional[str] = Field(default=None, description="e.g. 'Arabic', 'English'.")