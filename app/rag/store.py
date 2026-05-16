import chromadb
from app.rag.embedder import embed, embed_batch
from app.config import settings

class VectorStore:
    def __init__(self, collection_name: str = "patients"):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    async def query(self, text: str, top_k: int = 5, min_age: float = None, max_age: float = None, gender: str = None) -> list[dict]:
        query_embedding = await embed(text)
        
        where_filter = None
        conditions = []
        if min_age is not None:
            conditions.append({"age": {"$gte": min_age}})
        if max_age is not None:
            conditions.append({"age": {"$lte": max_age}})
        if gender is not None:
            conditions.append({"gender": {"$eq": gender}})
        if conditions:
            where_filter = {"$and": conditions} if len(conditions) > 1 else conditions[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )
        return [
            {"text": doc, "id": id, "metadata": metadata}
            for doc, id, metadata in zip(results["documents"][0], results["ids"][0], results["metadatas"][0])
        ]
    
    async def add_documents(self, ids: list[str], texts: list[str], metadatas: list[dict] = None):
        embeddings = await embed_batch(texts)
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def get_all_documents(self) -> dict:
        result = self.collection.get()
        return {
            "ids": result["ids"],
            "documents": result["documents"],
            "metadatas": result["metadatas"]
        }

    def get_by_patient_id(self, patient_id: str) -> list[dict]:
        result = self.collection.get(
            where={"patient_uid": {"$eq": patient_id}}
        )
        docs = [
            {"text": doc, "id": id, "metadata": metadata}
            for doc, id, metadata in zip(result["documents"], result["ids"], result["metadatas"])
        ]
        docs.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
        return docs