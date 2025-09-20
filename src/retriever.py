import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self):
        BASE_PATH = os.path.dirname(os.path.abspath(__file__))
        index_path      = os.path.join(BASE_PATH,"..","Data","index.faiss")
        chunks_path     = os.path.join(BASE_PATH,"..","Data","chunks.npy")
        model_name      = "all-MiniLM-L6-v2"

        self.index      = faiss.read_index(index_path)
        self.chunks     = np.load(chunks_path, allow_pickle=True)
        self.model      = SentenceTransformer(model_name)

    def retrieve(self, query, top_k=3):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(q_emb, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            chunk_text = self.chunks[idx]
            results.append({"text": chunk_text, "score": float(dist)})

        return results


"""
# if __name__ == "__main__":
#     retriever = Retriever()
#     query = "Why we need central tendency?"
#     results = retriever.retrieve(query, top_k=2)

#     print("🔍 Retrieved chunks:")
#     for r in results:
#         print(f"- {r['text']} (score: {r['score']:.4f})")
"""
