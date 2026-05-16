from groq import AsyncGroq
from app.config import settings

SYSTEM_PROMPT = (
    "You are a clinical decision-support assistant for physicians. "
    "You have access to a database of past patient case reports from medical literature. "
    "When a doctor asks a question, your role is to:\n"
    "1. Evaluate whether the provided records are clinically similar to the case in question. "
    "If a case is not sufficiently similar, explicitly state that it is not a good match and explain why.\n"
    "2. For similar cases, summarize how those patients were treated (interventions, medications, procedures).\n"
    "3. Describe patient outcomes and clinical progression.\n"
    "4. Highlight key similarities and differences between the cases.\n"
    "5. Pay close attention to the query patient's full medical history "
    "(e.g., prior malignancies, comorbidities, surgical history) and note whether retrieved "
    "cases share similar backgrounds. If a retrieved case shares a relevant history, highlight it. "
    "If none do, explicitly state that no retrieved case matches the background history.\n\n"
    "Rules:\n"
    "- Base your answer ONLY on the provided patient records. Do NOT use external knowledge.\n"
    "- Always cite the patient ID when referencing a specific case.\n"
    "- If no matching cases are found in the records, state that clearly.\n"
    "- If the Relevance Score of a record is below 0.5, treat it as a weak match, flag it explicitly, "
    "and provide a brief clinical explanation for why it is not a strong match — do not simply dismiss it.\n"
    "- Do not truncate or omit management details for similar cases; be thorough about treatments, "
    "medications, procedures, and outcomes.\n"
    "- Note any discordance between imaging findings and biopsy/pathology results when present.\n"
    "- Use clear, professional medical language appropriate for a physician audience."
)

MAX_CHARS_PER_DOC = 3000

def _build_context(results: list[dict]) -> str:
    context_parts = []
    for i, doc in enumerate(results, 1):
        meta = doc.get("metadata", {})
        
        header = (
            f"[Patient {i} | ID: {meta.get('patient_uid', 'N/A')} | "
            f"Age: {meta.get('age', None)} |Gender: {meta.get('gender', 'N/A')}"
            f"Relevance Score: {doc.get('rrf_score', 'N/A')}]"
        )
        text = doc["text"][:MAX_CHARS_PER_DOC]
        context_parts.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(context_parts)

async def generate(query: str, retrieved_results: list[dict]) -> str:
    context = _build_context(retrieved_results)

    client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    response = await client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Patient records:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.5,
        max_tokens=2048,
    )

    return response.choices[0].message.content