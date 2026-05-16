from fastapi import APIRouter, UploadFile, File, Depends
from app.schemas import OCRRequest, OCRResult
from app.vision import extract
from app.preprocessing.pipeline import preprocess


router = APIRouter(prefix="/ocr", tags=["OCR"])

@router.post("/", response_model=OCRResult)
async def ocr_extract(file: UploadFile = File(...), prompt: str = None, output_format: str = "plain", language_hint: str = None):
    image_bytes = await file.read()
    processed_bytes = preprocess(image_bytes)


    request = OCRRequest(
        prompt=prompt,
        output_format=output_format,
        language_hint=language_hint,
    )

    text = await extract(processed_bytes, request)
    return OCRResult(text=text)
