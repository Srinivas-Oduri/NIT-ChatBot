# AI-Powered PDF Analysis & Retrieval Backend

This project provides a backend system for intelligent PDF document analysis and retrieval using modern AI and vector search technologies. It leverages [LangChain](https://python.langchain.com/), [Ollama](https://ollama.com/), [FAISS](https://github.com/facebookresearch/faiss), and [pdfplumber](https://github.com/jsvine/pdfplumber) to enable advanced document chunking, semantic search, and LLM-powered analysis.

## Features

-**PDF Text Extraction:** Extracts and cleans text from PDF files using `pdfplumber`.

-**Document Chunking:** Splits documents into manageable chunks for efficient retrieval.

-**Vector Store:** Stores document embeddings using FAISS for fast similarity search.

-**LLM Integration:** Uses Ollama LLMs for generating sub-queries, synthesis, and analysis.

-**RAG (Retrieval-Augmented Generation):** Combines search and LLMs for context-aware answers.

-**API Endpoints:** Designed to be used as a backend for web or desktop applications.

## Technology Stack

-**Python 3.10+**

-**LangChain** for orchestration and chains

-**Ollama** for LLM and embeddings

-**FAISS** for vector similarity search

-**pdfplumber** for PDF text extraction

-**Logging** for robust error and event tracking

## Project Structure

```

backend/

  ai_core.py         # Main AI logic: extraction, chunking, vector store, RAG, LLM

  app.py             # API endpoints and server logic

  config.py          # Configuration for models, paths, and parameters

  utils.py           # Helper functions

  ...

uploads/             # User-uploaded PDFs

default_pdfs/        # Default/canned PDFs

```

## Setup

1.**Install dependencies:**

```bash

   pip install -r requirements.txt

```

2.**Configure Ollama and models** in `config.py`.

3.**Run the backend:**

```bash

   python backend/app.py

```

## Usage

- Upload PDF documents via the API or UI.
- Query the system for semantic search, document analysis, or synthesis.
- Retrieve context-rich answers powered by RAG and LLM.

## Customization

- Adjust chunk size, overlap, and model parameters in `config.py`.
- Extend analysis prompts and synthesis logic as needed.

## Troubleshooting

- Ensure Ollama and required models are running.
- Check logs for errors related to PDF extraction or LLM invocation.
- For PDF extraction issues, verify the file is not password-protected or corrupted.

## License

MIT License

---

**For more details, see the code and inline documentation.**
