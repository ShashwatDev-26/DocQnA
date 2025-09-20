from flask import Flask, request,render_template
from src.rag_pipeline import RAGPipeline

rag = RAGPipeline(top_k=3)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    if request.method == "POST":
        query = request.form.get("query")
        if query:
            answer = rag.ask(query)
    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)












































# import streamlit as st
# from src.rag_pipeline import RAGPipeline


# rag = RAGPipeline(top_k=3)

# # Streamlit UI
# st.set_page_config(page_title="RAG Q&A System", layout="centered")
# st.title("🤖 Domain-Aware Q&A with RAG")


# query = st.text_input("Ask a question based on your article:")

# if st.button("Get Answer") and query:
#     with st.spinner("Retrieving and generating answer..."):
#         answer = rag.ask(query)

#     st.subheader("Answer:")
#     st.write(answer)
