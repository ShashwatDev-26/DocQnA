import os
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

class Preprocessor:
    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=200, overlap=50):
        self.model        = SentenceTransformer(model_name)
        self.chunk_size   = chunk_size
        self.overlap      = overlap
        self.basepath     = None

    def load_document(self, filepath):
        if os.path.exists(filepath)==False:
            print("[*] Incorrect file path!")
            return None
        self.basepath = os.path.dirname(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return text

    def chunk_text(self, text):
        words = text.split()
        chunks = []
        for i in range(0,len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks

    def embed_chunks(self, chunks):
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        return np.array(embeddings, dtype="float32")

    def build_faiss_index(self, embeddings):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype("float32"))   # 🔥 enforce float32
        return index


    def save_index(self, index, path="index.faiss"):
        path=os.path.join(self.basepath,path)
        faiss.write_index(index, path)

    def save_chunks(self, chunks, path="chunks.npy"):
        path=os.path.join(self.basepath,path)
        np.save(path, np.array(chunks))


if __name__ == "__main__":
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    textpath   = os.path.join(BASE_DIR,"..","Data","Article.txt")

    pre        = Preprocessor(chunk_size=200,overlap=50)

    text       = pre.load_document(textpath)
    chunks     = pre.chunk_text(text)
    embeddings = pre.embed_chunks(chunks)
    index      = pre.build_faiss_index(embeddings)

    # Save artifacts
    pre.save_index(index)
    pre.save_chunks(chunks)

    print(f"✅ Preprocessing complete! {len(chunks)} chunks stored.")
