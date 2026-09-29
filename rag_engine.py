import os
import pickle
import requests
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434/api/generate"

LLM_MODEL = "llama3.2:3b"

VECTOR_DB = "vectorstore/index.faiss"

METADATA_FILE = "vectorstore/metadata.pkl"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_path):

    reader = PdfReader(pdf_path)

    documents = []

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:

            documents.append({
                "page": page_number + 1,
                "text": text
            })

    return documents


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(text, chunk_size=800, overlap=150):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# BUILD DOCUMENT CHUNKS
# ============================================================

def create_chunks(pdf_path):

    pages = extract_pdf_text(pdf_path)

    chunks = []

    for page in pages:

        page_chunks = chunk_text(
            page["text"]
        )

        for chunk in page_chunks:

            chunks.append({
                "text": chunk,
                "page": page["page"],
                "source": os.path.basename(pdf_path)
            })

    return chunks


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

def build_vector_database(pdf_path):

    print("Reading PDF...")

    chunks = create_chunks(pdf_path)

    texts = [
        item["text"]
        for item in chunks
    ]

    print(
        f"Creating embeddings for {len(texts)} chunks..."
    )

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    os.makedirs(
        "vectorstore",
        exist_ok=True
    )

    faiss.write_index(
        index,
        VECTOR_DB
    )

    with open(
        METADATA_FILE,
        "wb"
    ) as f:

        pickle.dump(
            chunks,
            f
        )

    print("Vector database created.")

    return len(chunks)


# ============================================================
# LOAD VECTOR DATABASE
# ============================================================

def load_vector_database():

    index = faiss.read_index(
        VECTOR_DB
    )

    with open(
        METADATA_FILE,
        "rb"
    ) as f:

        metadata = pickle.load(f)

    return index, metadata


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def retrieve_documents(
    query,
    top_k=4
):

    index, metadata = load_vector_database()

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        result = metadata[idx].copy()

        result["score"] = float(score)

        results.append(
            result
        )

    return results


# ============================================================
# PROMPT ENGINEERING
# ============================================================

def create_prompt(
    question,
    retrieved_docs
):

    context_parts = []

    for i, doc in enumerate(
        retrieved_docs
    ):

        context_parts.append(
            f"""
SOURCE {i + 1}
Page: {doc['page']}
Content:
{doc['text']}
"""
        )

    context = "\n".join(
        context_parts
    )

    prompt = f"""
You are an enterprise document intelligence assistant.

Answer the user's question using ONLY the provided context.

Rules:

1. Do not invent information.
2. If the answer is not present in the context,
   say "I could not find this information in the document."
3. Give a concise but useful answer.
4. Mention the relevant source page.
5. Do not use outside knowledge.

CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    return prompt


# ============================================================
# CALL OLLAMA
# ============================================================

def generate_answer(prompt):

    payload = {

        "model": LLM_MODEL,

        "prompt": prompt,

        "stream": False,

        "options": {

            "temperature": 0.1,

            "num_ctx": 4096

        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    return result["response"]


# ============================================================
# COMPLETE RAG PIPELINE
# ============================================================

def ask_rag(
    question,
    top_k=4
):

    retrieved_docs = retrieve_documents(
        question,
        top_k=top_k
    )

    if not retrieved_docs:

        return (
            "No relevant information found.",
            []
        )

    # --------------------------------------------------------
    # Relevance threshold
    # --------------------------------------------------------

    relevant_docs = [
        doc
        for doc in retrieved_docs
        if doc["score"] >= 0.25
    ]

    if not relevant_docs:

        return (
            "I could not find relevant information "
            "in the uploaded document.",
            retrieved_docs
        )

    prompt = create_prompt(
        question,
        relevant_docs
    )

    answer = generate_answer(
        prompt
    )

    return answer, retrieved_docs