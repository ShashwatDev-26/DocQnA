from src.retriever import Retriever
from src.generator import Generator
from src.preprocess import Preprocessor
import os

class RAGPipeline:
    def __init__(self, top_k=3):
        self.retriever = False
        self.generator = False
        self.SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR  = os.path.join(self.SRC_DIR,"..","Data")
        self.top_k     = top_k
        self.__preprocesscheck()

    def __preprocesscheck(self):
        filelist = os.listdir(self.DATA_DIR)
        add      = 0
        for i in filelist:
            if i.endswith(".txt"):
                add+=1
            if i.endswith(".npy"):
                add+=1
            if i.endswith(".faiss"):
                add+=1
        if add<3:
            print("[*] Found unprocessed")
            print("[*] Initiating preprocessing..")
            print("[*] This may take time please wait...")
            self.preprocess()
        print("[*] Pre-process ✅")
        self.retriever = Retriever()
        self.generator = Generator()


    def preprocess(self,size=200,lap=50):
        textpath   = os.path.join(self.DATA_DIR,"Article.txt")
        pre        = Preprocessor(chunk_size=size,overlap=lap)
        text       = pre.load_document(textpath)
        chunks     = pre.chunk_text(text)
        embeddings = pre.embed_chunks(chunks)
        index      = pre.build_faiss_index(embeddings)

        # Save artifacts
        pre.save_index(index)
        pre.save_chunks(chunks)

        print(f"✅ Preprocessing complete! {len(chunks)} chunks stored.")


    def ask(self, query):
        retrieved_chunks = self.retriever.retrieve(query, top_k=self.top_k)
        answer = self.generator.generate_answer(query, retrieved_chunks)

        return answer

if __name__ == "__main__":
    rag = RAGPipeline(top_k=2)
    query = input("❓Enter your question: ")
    answer = rag.ask(query)

    print("\n🤖 Final Answer:")
    print(answer)
