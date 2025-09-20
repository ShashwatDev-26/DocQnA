from transformers import pipeline

class Generator:
    def __init__(self, model_name="google/flan-t5-base"):
        self.generator = pipeline("text2text-generation", model=model_name)

    def generate_answer(self, query, retrieved_chunks, max_length=200):

        context = "\n".join([chunk["text"] for chunk in retrieved_chunks])

        prompt = f"Question: {query}\nContext: {context}\nAnswer:"

        response = self.generator(prompt, max_length=max_length, clean_up_tokenization_spaces=True)
        
        return response[0]["generated_text"]

"""
# if __name__ == "__main__":
#     gen = Generator()
#     query = "What are the symptoms of vitamin D deficiency?"
#     chunks = [
#         {"text": "Vitamin D deficiency causes fatigue and bone pain.", "score": 0.1},
#         {"text": "It may also lead to muscle weakness and depression.", "score": 0.2}
#     ]

#     answer = gen.generate_answer(query, chunks)
#     print("🤖 Final Answer:", answer)
"""
