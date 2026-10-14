# RAG Document Question-Answering System

A Retrieval-Augmented Generation (RAG) application that retrieves relevant document chunks from a persistent vector database and uses a large language model to generate context-grounded answers.

## Features

* Load text documents from a local directory.
* Split documents into overlapping chunks.
* Generate local text embeddings using Sentence Transformers.
* Store embeddings, text, and metadata in ChromaDB.
* Retrieve relevant chunks using vector similarity search.
* Generate answers using the Groq API.
* Persist the local vector database between application runs.

## Technology Stack

* Python
* Sentence Transformers
* ChromaDB
* Groq API
* python-dotenv

## Project Structure

```text
interview/
├── documents/
│   └── ai_fundamentals.txt
├── .env.example
├── .gitignore
├── ingest.py
├── rag.py
├── verify_db.py
├── requirements.txt
└── README.md
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the API key

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_actual_api_key
```

Do not commit `.env` to version control.

### 4. Add documents

Place non-empty `.txt` files inside the `documents/` directory.

### 5. Ingest documents

```bash
python ingest.py
```

This loads documents, creates overlapping chunks, generates embeddings, and stores them in ChromaDB.

### 6. Ask questions

```bash
python rag.py
```

Enter a question at the command-line prompt. Type `exit` to stop the application.

### 7. Verify storage

```bash
python verify_db.py
```

## Architecture

1. Documents are loaded from the local filesystem.
2. Text is divided into chunks with overlap.
3. A local embedding model converts each chunk into a vector.
4. ChromaDB stores the vectors, documents, and metadata.
5. User questions are converted into embeddings using the same model.
6. ChromaDB retrieves the nearest stored chunks.
7. The retrieved context is passed to a Groq-hosted LLM.
8. The LLM generates a response based on the supplied context.

## Limitations

* The ingestion script currently supports plain-text files.
* Retrieval always returns the nearest available chunks unless the query is rejected as empty.
* Semantic similarity does not guarantee factual relevance.
* Prompt instructions alone cannot eliminate hallucinations.
* Answer quality depends on document quality, retrieval quality, and the selected LLM.
* API availability and rate limits depend on the provider.

## Future Improvements

* Add PDF and DOCX ingestion.
* Implement configurable chunk sizes and overlap.
* Evaluate retrieval quality using a labelled test dataset.
* Add relevance filtering and reranking.
* Introduce hybrid keyword and semantic search.
* Add FastAPI endpoints.
* Implement structured logging and targeted API error handling.
* Add automated tests for ingestion, retrieval, and generation.

## Security

API credentials must be stored in environment variables and excluded from version control. Do not commit secrets, private documents, or local databases containing sensitive information.
