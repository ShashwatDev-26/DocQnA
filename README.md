# Domain-Aware Q&A System using Retrieval-Augmented Generation (RAG)

A lightweight **RAG-based question answering system** that retrieves relevant context from a domain-specific article and generates grounded answers using a language model. The project includes a simple **Flask web interface** for querying the system in a browser.

---

## Features

- Retrieval-Augmented Generation (RAG) pipeline:
  - **Retriever:** Searches for the most relevant text chunks from your article using **FAISS**.
  - **Generator:** Uses **Hugging Face FLAN-T5** to generate natural language answers.
- Supports **single or multiple articles** (can be extended).
- Simple, interactive **web interface** using Flask.
- Modular design for **easy extension or deployment**.

---

## Project Structure

```

rag\_project/
│
├── data/
│   ├── article.txt       # Your source article(s)
│   ├── index.faiss       # FAISS index built by preprocess.py
│   └── chunks.npy        # Preprocessed text chunks
│
├── src/
│   ├── preprocess.py     # Preprocessing: chunking + embeddings + FAISS index
│   ├── retriever.py      # Retriever module
│   ├── generator.py      # Generator module
│   └── rag\_pipeline.py   # Orchestrates retrieval + generation
│
├── app.py                # Flask web app
├── requirements.txt      # Python dependencies
└── README.md

````

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/ShashwatDev-26/DocQnA.git
cd DocQnA
````

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Preprocess your article

Put your article in `Data/Article.txt` and run:

```bash
src/preprocess.py
```

This will:

* Split the text into chunks
* Generate embeddings
* Build a FAISS index for retrieval

---

### 2. Launch the Flask app

```bash
python app.py
```

Open your browser at: `http://127.0.0.1:5000`
Type a question and click **Get Answer**. The system will fetch relevant chunks and generate a grounded response.

---

## How It Works

1. **User Query:** Input question in the web interface.
2. **Retriever:** FAISS searches the preprocessed article chunks and returns top-k relevant passages.
3. **Generator:** FLAN-T5 takes the retrieved passages and the query to generate a natural language answer.
4. **Answer:** Displayed in the web interface.

---

## Dependencies

* Python 3.9+
* [transformers](https://huggingface.co/docs/transformers/index)
* [sentence-transformers](https://www.sbert.net/)
* [faiss-cpu](https://github.com/facebookresearch/faiss)
* Flask

---

## Future Improvements

* Support **multiple articles** or PDF uploads.
* Add **chat history** for multi-turn conversations.
* Upgrade generator to **larger models** (e.g., GPT-4, LLaMA) for better answers.
* Deploy on cloud services like **Heroku / AWS / GCP**.

---

