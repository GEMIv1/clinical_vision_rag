import json
import re
from app.schemas.outputFormat import OutputFormat

def parse_response(raw: str, output_format: OutputFormat) -> str:
   
    text = raw.strip()
    if output_format == OutputFormat.json:
        cleaned = re.sub(
            r"^```(?:json)?\s*\n?", 
            "",
            text,
        )
        cleaned = re.sub(
            r"\n?```\s*$", 
            "",
            cleaned,
        ).strip()

        try:
            data = json.loads(cleaned)
            return json.dumps(data, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, AttributeError):
            try:
                data = json.loads(text)
                return json.dumps(data, indent=2, ensure_ascii=False)
            except (json.JSONDecodeError, AttributeError):
                return text
    return text
            