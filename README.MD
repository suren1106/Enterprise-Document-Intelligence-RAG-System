# 🤖 Enterprise Document Intelligence RAG

An end-to-end Retrieval-Augmented Generation (RAG) system for answering questions from enterprise PDF documents using semantic search, vector retrieval, and a locally hosted LLM.

## Architecture

PDF → Text Extraction → Chunking → Embeddings → FAISS → Semantic Retrieval → Context Construction → Llama 3.2 → Grounded Answer

## Key Features

* PDF document ingestion
* Intelligent overlapping text chunking
* Dense semantic embeddings
* FAISS vector database
* Top-K similarity retrieval
* Relevance thresholding
* Grounded prompt generation
* Local LLM inference with Ollama
* Source-page attribution
* Retrieval transparency
* Streamlit interface
* No paid API required

## Tech Stack

Python • Streamlit • Sentence Transformers • FAISS • PyPDF • Ollama • Llama 3.2

## RAG Pipeline

1. Upload a PDF.
2. Extract text from each page.
3. Split the document into overlapping chunks.
4. Generate embeddings using `all-MiniLM-L6-v2`.
5. Store vectors in FAISS.
6. Convert the user question into an embedding.
7. Retrieve the most semantically relevant chunks.
8. Apply a relevance threshold.
9. Inject retrieved context into a grounded prompt.
10. Generate the answer using Llama 3.2.
11. Display retrieved source pages and similarity scores.

## Example

**Question**

> How many annual leave days are available?

**System**

```text
Question
   ↓
Query Embedding
   ↓
FAISS Similarity Search
   ↓
Relevant Document Chunks
   ↓
Grounded Prompt
   ↓
Llama 3.2
   ↓
Answer + Sources
```

## Why RAG?

RAG allows an LLM to retrieve information from an external knowledge base at query time instead of relying exclusively on information learned during model training.

This architecture is useful for enterprise knowledge assistants, policy assistants, technical documentation search, internal knowledge systems, and document question-answering applications.

## Future Enhancements

* Hybrid BM25 + vector retrieval
* Cross-encoder reranking
* Query rewriting
* Multi-query retrieval
* Conversational memory
* Metadata filtering
* RAG evaluation
* Answer faithfulness evaluation
* Hallucination detection
* LangGraph orchestration
* Production API with FastAPI
* Docker deployment
* Observability with MLflow/Langfuse
