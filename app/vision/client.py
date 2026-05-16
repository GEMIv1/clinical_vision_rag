import base64
import httpx
from app.config import settings
from app.schemas.ocrRequest import OCRRequest
from app.schemas.outputFormat import OutputFormat
from app.vision.prompts import build_prompt
from app.vision.parser import parse_response


async def extract(image_bytes: bytes, request: OCRRequest) -> str:
    prompt = build_prompt(
        prompt=request.prompt,
        output_format=request.output_format,
        language_hint=request.language_hint,
    )
    image_b64 = base64.b64encode(image_bytes).decode()
    try:
        return await _call_model(image_b64, settings.MODEL_NAME, prompt, request.output_format)
    except Exception as exc:
        print(f"Primary model failed with error: {exc}. Falling back to backup model")
        return await _call_model(image_b64, settings.BACKUP_MODEL_NAME, prompt, request.output_format)


async def _call_model(image_b64: str, model_name: str, prompt: str, fmt: OutputFormat) -> str:
    url = "https://router.huggingface.co/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 4096,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            url,
            headers={"Authorization": f"Bearer {settings.HF_TOKEN}"},
            json=payload,
        )
        if not response.is_success:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise ValueError(f"[{model_name}] HTTP {response.status_code}: {detail}")
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        return parse_response(raw, fmt)