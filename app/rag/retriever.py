import logging
import math
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from app.rag.store import VectorStore

class Retriever:
    def __init__(self, store: VectorStore):
        self.store = store
        self.bm25: BM25Okapi | None = None
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self.doc_metadatas: list[dict] = []
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


    def build_bm25_index(self):
        all_docs = self.store.get_all_documents()
        self.doc_ids = all_docs["ids"]
        self.doc_texts = all_docs["documents"]
        self.doc_metadatas = all_docs["metadatas"]

        tokenized = [doc.lower().split() for doc in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized)
    async def retrieve(self, query: str, top_k: int = 5, min_age: float = None, max_age: float = None, gender: str = None) -> list[dict]:
        """
        Steps:
          1&2. Dual retrieval: semantic search via vector store + keyword search via BM25
          3. Reciprocal Rank Fusion to combine signals from both retrievers into a single ranked list
          4. Cross-encoder re-ranking to further refine the order based on (query, doc) relevance
          5. Deduplicate results by patient_uid, keeping only the top-ranked chunk per unique patient
          6. Expand each chunk back to the full concatenated text of all chunks for that patient so downstream LLM calls have complete context
        """
        fetch_k = top_k * 3

        semantic_results = await self.store.query(
            text=query,
            top_k=fetch_k,
            min_age=min_age,
            max_age=max_age,
            gender=gender,
        )

        bm25_results = self._bm25_search(
            query=query,
            top_k=fetch_k,
            min_age=min_age,
            max_age=max_age,
            gender=gender,
        )

        fused = self._rrf_fuse(semantic_results, bm25_results)

        reranked = self._rerank(query, fused)

        deduplicated = self._deduplicate_by_patient(reranked, top_k)

        #full_results = self._expand_to_full_text(deduplicated)

        return deduplicated

    def _rerank(self, query: str, docs: list[dict]) -> list[dict]:
        """
        Score each (query, doc) pair with the cross-encoder and attach the
        result as `cross_encoder_score`.  The original `rrf_score` is kept
        untouched so callers can inspect both signals.

        The cross-encoder returns raw logits; we apply sigmoid to map them
        into [0, 1] so they are easier to read, but note that most medical /
        clinical text will produce values well below 0.5 — that is normal.
        """
        if not docs:
            return []

        pairs = [[query, doc["text"]] for doc in docs]
        raw_scores = self.cross_encoder.predict(pairs)

        for doc, raw in zip(docs, raw_scores):
            doc["cross_encoder_score"] = round(self._sigmoid(float(raw)), 4)
            doc["cross_encoder_raw"] = round(float(raw), 4)

        return sorted(docs, key=lambda x: x["cross_encoder_score"], reverse=True)

    def _bm25_search(self, query: str, top_k: int, min_age: float = None, max_age: float = None, gender: str = None) -> list[dict]:
        if self.bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        results = []
        for i, score in enumerate(scores):
            if score < 1.0:
                continue

            metadata = self.doc_metadatas[i] if self.doc_metadatas else {}

            if not self._passes_filters(metadata, min_age, max_age, gender):
                continue

            results.append({
                "id": self.doc_ids[i],
                "text": self.doc_texts[i],
                "metadata": metadata,
                "bm25_score": float(score),
            })

        results.sort(key=lambda x: x["bm25_score"], reverse=True)
        return results[:top_k]

    @staticmethod
    def _rrf_fuse( semantic_results: list[dict], bm25_results: list[dict], k: int = 30) -> list[dict]:

        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        for rank, doc in enumerate(semantic_results):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc_id] = doc

        for rank, doc in enumerate(bm25_results):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

        return [
            {
                "id": doc_id,
                "text": doc_map[doc_id]["text"],
                "metadata": doc_map[doc_id]["metadata"],
                "rrf_score": round(rrf_scores[doc_id], 6),
                "relevance_pct": round((rrf_scores[doc_id] / (2 / (k + 1))) * 100, 1),

            }
            for doc_id in sorted_ids
        ]
        
    @staticmethod
    def _deduplicate_by_patient(docs: list[dict], top_k: int) -> list[dict]:
        seen: set[str] = set()
        unique: list[dict] = []
        for doc in docs:
            uid = doc["metadata"].get("patient_uid")
            if uid not in seen:
                seen.add(uid)
                unique.append(doc)
            if len(unique) >= top_k:
                break
        return unique

    # def _expand_to_full_text(self, docs: list[dict]) -> list[dict]:
    #     full_results = []
    #     for doc in docs:
    #         patient_uid = doc["metadata"].get("patient_uid")
    #         if patient_uid is not None:
    #             all_chunks = self.store.get_by_patient_id(patient_uid)
    #             full_text = " ".join(chunk["text"] for chunk in all_chunks)
    #         else:
    #             full_text = doc["text"]

    #         full_results.append({
    #             "id": doc["id"],
    #             "text": full_text,
    #             "metadata": doc["metadata"],
    #             "rrf_score": doc.get("rrf_score"),
    #             "cross_encoder_score": doc.get("cross_encoder_score"),
    #             "cross_encoder_raw": doc.get("cross_encoder_raw"),
    #         })

    #     return full_results

    @staticmethod
    def _passes_filters(metadata: dict, min_age: float | None, max_age: float | None, gender: str | None) -> bool:
        if min_age is not None:
            doc_age = metadata.get("age")
            if doc_age is None or float(doc_age) < min_age:
                return False
        if max_age is not None:
            doc_age = metadata.get("age")
            if doc_age is None or float(doc_age) > max_age:
                return False
        if gender is not None:
            if metadata.get("gender") != gender:
                return False
        return True

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))