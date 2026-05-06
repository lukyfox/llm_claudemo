import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from pathlib import Path
from models.file_processor import FileProcessor
import uuid


class KBase:
    """
    ChromaDB-backed knowledge base.
    Handles chunking, embedding and similarity search.
    """

    CHUNK_SIZE = 500       # characters per chunk
    CHUNK_OVERLAP = 80     # overlap between chunks

    def __init__(self, persist_dir: str = "data/chroma"):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="documents",
            embedding_function=DefaultEmbeddingFunction()
        )

    def add_document(self, file_path: str) -> dict:
        """
        Process a file and store its chunks in the collection.
        :param file_path: Path to the file to process.
        """
        result = FileProcessor.process(file_path)
        if result["type"] != "text_content":
            # Currently only text-based files are supported
            # TODO: add PDF support (pypdf reader)
            raise ValueError(f"Cannot index binary file: {result.get('filename')}")

        text = result["text"]
        filename = result["filename"]
        chunks = self._chunk(text)

        ids = [str(uuid.uuid4()) for _ in chunks]
        self._collection.add(
            documents=chunks,
            ids=ids,
            metadatas=[{"source": filename, "chunk": i} for i, _ in enumerate(chunks)]
        )
        return {"filename": filename, "chunks_added": len(chunks)}

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """
        Return the most relevant chunks for a query
        """
        results = self._collection.query(
            query_texts=[query],
            n_results=min(n_results, self._collection.count() or 1)
        )
        output = []
        for doc, meta, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            output.append({
                "text": doc,
                "source": meta["source"],
                "chunk": meta["chunk"],
                "relevance": round(1 - distance, 3)
            })
        return output

    def list_documents(self) -> list[str]:
        """
        Return unique filenames currently indexed.
        """
        if self._collection.count() == 0:
            return []
        all_meta = self._collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in all_meta})

    def delete_document(self, filename: str) -> int:
        """
        Remove all chunks belonging to a given filename
        """
        existing = self._collection.get(
            where={"source": filename},
            include=["metadatas"]
        )
        ids = existing["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def _chunk(self, text: str) -> list[str]:
        chunks, start = [], 0
        while start < len(text):
            end = start + self.CHUNK_SIZE
            chunks.append(text[start:end])
            start += self.CHUNK_SIZE - self.CHUNK_OVERLAP
        return [c.strip() for c in chunks if c.strip()]