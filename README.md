# DocMind — AI Document Chat

A RAG-powered chatbot that lets you upload any PDF and ask questions about it in natural language. Built with LangChain, FAISS, and GPT-4o.

**Live demo:** https://rag-chatbot-zz.streamlit.app/

---

## What it does

- Upload any PDF document
- Ask questions about the content in plain English
- Get accurate answers grounded in the document with source page citations
- Maintains conversation history across multiple questions

## How it works

1. The PDF is loaded and split into chunks
2. Each chunk is converted into a vector embedding using OpenAI's embedding model
3. Chunks are stored in a FAISS vector store for fast similarity search
4. When you ask a question, the most relevant chunks are retrieved
5. GPT-4o uses those chunks as context to generate a grounded answer

## Tech stack

- **LangChain** — RAG pipeline orchestration
- **FAISS** — local vector store for semantic search
- **OpenAI** — embeddings + GPT-4o for answer generation
- **Streamlit** — frontend UI
- **PyPDF** — PDF loading and parsing

## Run locally

```bash
git clone https://github.com/nattering/rag-chatbot.git
cd rag-chatbot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

You'll need an OpenAI API key — enter it in the app when prompted.

## Author

Built by Aryan — CS with AI student, year 1.