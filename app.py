import os
import streamlit as st

from rag_engine import (
    build_vector_database,
    ask_rag
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Enterprise RAG",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "🤖 Enterprise Document Intelligence"
)

st.markdown(
    """
### Advanced Retrieval-Augmented Generation System

Upload a PDF → Build knowledge base → Ask questions → 
Retrieve relevant information → Generate grounded answers.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ RAG Configuration")

    top_k = st.slider(
        "Retrieved Documents",
        min_value=1,
        max_value=8,
        value=4
    )

    st.info(
        """
        Pipeline:

        PDF
        ↓
        Chunking
        ↓
        Embeddings
        ↓
        FAISS
        ↓
        Semantic Search
        ↓
        Llama
        """
    )


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📄 Upload Knowledge Base PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file:

    os.makedirs(
        "data",
        exist_ok=True
    )

    pdf_path = os.path.join(
        "data",
        uploaded_file.name
    )

    with open(
        pdf_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    if st.button(
        "🚀 Build Knowledge Base"
    ):

        with st.spinner(
            "Processing document..."
        ):

            number_of_chunks = (
                build_vector_database(
                    pdf_path
                )
            )

        st.success(
            f"""
            Knowledge base created successfully!

            Total chunks: {number_of_chunks}
            """
        )


# ============================================================
# QUESTION
# ============================================================

st.divider()

st.subheader(
    "💬 Ask your document"
)

question = st.text_input(
    "Enter your question"
)


# ============================================================
# RAG QUERY
# ============================================================

if st.button(
    "🔍 Ask AI"
):

    if not question:

        st.warning(
            "Please enter a question."
        )

    elif not os.path.exists(
        "vectorstore/index.faiss"
    ):

        st.error(
            "Please upload a PDF and "
            "build the knowledge base first."
        )

    else:

        with st.spinner(
            "Searching knowledge base..."
        ):

            answer, documents = ask_rag(
                question,
                top_k
            )

        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        st.subheader(
            "🤖 Answer"
        )

        st.write(
            answer
        )

        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📚 Retrieved Sources"
        )

        for i, doc in enumerate(
            documents
        ):

            with st.expander(
                f"""
                Source {i + 1} |
                Page {doc['page']} |
                Similarity {doc['score']:.3f}
                """
            ):

                st.write(
                    doc["text"]
                )