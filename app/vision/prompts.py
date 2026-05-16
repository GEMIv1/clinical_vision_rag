from app.schemas.outputFormat import OutputFormat

SYSTEM_PROMPT = "Extract all text from this image exactly as it appears. Preserve layout and formatting."

def build_prompt(prompt: str | None, output_format: OutputFormat, language_hint: str | None) -> str:
    result = prompt if prompt else SYSTEM_PROMPT
    if language_hint:
        result += f" The document language is {language_hint}."
    if output_format == OutputFormat.json:
        result += " Return ONLY a JSON object. No markdown, no explanation."
    return result