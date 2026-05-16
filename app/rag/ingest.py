import pandas as pd
from app.rag.store import VectorStore

def chunk_text(text: str, max_words: int = 256, overlap_words: int = 50) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    step = max_words - overlap_words

    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += step

    return chunks

async def ingest_csv(csv_path: str, store: VectorStore, batch_size: int = 64, max_words: int = 256, overlap_words: int = 50):
    
    df = pd.read_csv(csv_path)

    all_ids: list[str] = []
    all_texts: list[str] = []
    all_metadatas: list[dict] = []

    for _, row in df.iterrows():
        patient_uid = str(row["patient_uid"])
        narrative = str(row["patient"])
        chunks = chunk_text(narrative, max_words=max_words, overlap_words=overlap_words)

        for chunk_idx, chunk in enumerate(chunks):
            doc_id = f"{patient_uid}_chunk_{chunk_idx}"
            metadata = {
                "patient_uid": patient_uid,
                "title": str(row["title"]),
                "age": str(row["age"]),
                "gender": str(row["gender"]),
                "chunk_index": chunk_idx,
                "total_chunks": len(chunks),
            }

            all_ids.append(doc_id)
            all_texts.append(chunk)
            all_metadatas.append(metadata)

    total = len(all_ids)
    ingested = 0

    for i in range(0, total, batch_size):
        batch_ids = all_ids[i : i + batch_size]
        batch_texts = all_texts[i : i + batch_size]
        batch_metas = all_metadatas[i : i + batch_size]

        await store.add_documents(
            ids=batch_ids,
            texts=batch_texts,
            metadatas=batch_metas,
        )

        ingested += len(batch_ids)
        print(f"Ingested {ingested}/{total} chunks")

    print(f"Ingestion complete: {total} chunks from {len(df)} patients")
    return {"total_patients": len(df), "total_chunks": total}
