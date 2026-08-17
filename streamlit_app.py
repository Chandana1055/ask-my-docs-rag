import os
import re
import fitz
import faiss
import numpy as np
import streamlit as st

from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Ask My Docs",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# LOAD GROQ API
# =========================================================

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()

client = Groq(api_key=api_key)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📄 Ask My Docs</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI Document Assistant powered by RAG + FAISS + Groq'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type="pdf",
        accept_multiple_files=True
    )

    st.divider()

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} document(s) uploaded"
        )

        for file in uploaded_files:
            st.write(f"📄 {file.name}")

    else:
        st.info(
            "Upload one or more PDFs to begin."
        )

    st.divider()

    # CLEAR CHAT
    if st.button(
        "Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    # DOWNLOAD CHAT HISTORY
    if st.session_state.messages:

        chat_text = ""

        for message in st.session_state.messages:

            role = message["role"].upper()

            chat_text += f"{role}:\n"
            chat_text += message["content"]
            chat_text += "\n\n"

        st.download_button(
            label="📥 Download Chat History",
            data=chat_text,
            file_name="ask_my_docs_chat.txt",
            mime="text/plain",
            use_container_width=True
        )


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


with st.spinner(
    "Loading AI embedding model..."
):

    model = load_embedding_model()


# =========================================================
# PROCESS DOCUMENTS
# =========================================================

def process_documents(uploaded_files):

    all_chunks = []
    all_sources = []

    for uploaded_file in uploaded_files:

        pdf_bytes = uploaded_file.getvalue()

        doc = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        for page_number, page in enumerate(doc):

            text = page.get_text()

            if not text.strip():
                continue

            # CHUNKING
            chunk_size = 300
            overlap = 50

            step = chunk_size - overlap

            for i in range(
                0,
                len(text),
                step
            ):

                chunk = text[
                    i:i + chunk_size
                ]

                if chunk.strip():

                    all_chunks.append(
                        chunk
                    )

                    all_sources.append(
                        f"{uploaded_file.name} "
                        f"(Page {page_number + 1})"
                    )

        doc.close()

    if not all_chunks:

        return [], [], None

    # CREATE EMBEDDINGS
    embeddings = model.encode(
        all_chunks
    )

    embedding_array = np.array(
        embeddings
    ).astype("float32")

    # CREATE FAISS INDEX
    index = faiss.IndexFlatL2(
        embedding_array.shape[1]
    )

    index.add(
        embedding_array
    )

    return (
        all_chunks,
        all_sources,
        index
    )


# =========================================================
# KEYWORD SEARCH
# =========================================================

def keyword_search(
    question,
    chunks
):

    question_lower = question.lower()

    question_words = re.findall(
        r"[a-zA-Z0-9]+",
        question_lower
    )

    useful_words = [
        word
        for word in question_words
        if len(word) >= 4
    ]

    matched_indices = []

    for index, chunk in enumerate(chunks):

        chunk_lower = chunk.lower()

        score = 0

        for word in useful_words:

            if word in chunk_lower:

                score += 1

        if score >= 1:

            matched_indices.append(
                (index, score)
            )

    matched_indices.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        item[0]
        for item in matched_indices
    ]


# =========================================================
# MAIN APPLICATION
# =========================================================

if uploaded_files:

    # =====================================================
    # PROCESS DOCUMENTS
    # =====================================================

    with st.spinner(
        "Processing your documents..."
    ):

        chunks, sources, index = process_documents(
            uploaded_files
        )

    if not chunks:

        st.error(
            "No readable text was found in the PDFs."
        )

        st.stop()


    # =====================================================
    # DOCUMENT ANALYTICS
    # =====================================================

    total_documents = len(
        uploaded_files
    )

    total_chunks = len(
        chunks
    )

    total_words = sum(
        len(chunk.split())
        for chunk in chunks
    )

    total_pages = 0

    for uploaded_file in uploaded_files:

        pdf_bytes = uploaded_file.getvalue()

        doc = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        total_pages += len(doc)

        doc.close()


    # =====================================================
    # DOCUMENT OVERVIEW
    # =====================================================

    st.subheader(
        "Document Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Documents",
            total_documents
        )

    with col2:

        st.metric(
            "Pages",
            total_pages
        )

    with col3:

        st.metric(
            "Chunks",
            total_chunks
        )

    with col4:

        st.metric(
            "Words",
            total_words
        )

    st.divider()


    # =====================================================
    # PER-DOCUMENT ANALYTICS
    # =====================================================

    st.subheader(
        "Per-Document Analytics"
    )

    document_stats = []

    for uploaded_file in uploaded_files:

        pdf_bytes = uploaded_file.getvalue()

        doc = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        page_count = len(doc)

        document_text = ""

        for page in doc:

            document_text += page.get_text()

        doc.close()

        word_count = len(
            document_text.split()
        )

        document_chunk_count = sum(
            1
            for source in sources
            if source.startswith(
                uploaded_file.name + " "
            )
        )

        document_stats.append(
            {
                "Document": uploaded_file.name,
                "Pages": page_count,
                "Words": word_count,
                "Chunks": document_chunk_count
            }
        )


    st.dataframe(
        document_stats,
        use_container_width=True,
        hide_index=True
    )


    # =====================================================
    # DOCUMENT COMPARISON
    # =====================================================

    st.subheader(
        "Document Comparison"
    )

    chart_data = {
        "Document": [
            item["Document"]
            for item in document_stats
        ],

        "Words": [
            item["Words"]
            for item in document_stats
        ]
    }

    st.bar_chart(
        chart_data,
        x="Document",
        y="Words"
    )

    st.divider()


    # =====================================================
    # UPLOADED DOCUMENTS
    # =====================================================

    st.subheader(
        "Uploaded Documents"
    )

    for uploaded_file in uploaded_files:

        st.write(
            f"📄 {uploaded_file.name}"
        )

    st.divider()


    # =====================================================
    # CHAT HISTORY
    # =====================================================

    st.subheader(
        "Chat History"
    )

    if st.session_state.messages:

        for message in st.session_state.messages:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

    else:

        st.caption(
            "No questions asked yet. "
            "Start a conversation below."
        )


    # =====================================================
    # CHAT INPUT
    # =====================================================

    question = st.chat_input(
        "Ask something about your documents..."
    )


    if question:

        # =================================================
        # USER MESSAGE
        # =================================================

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        # =================================================
        # QUESTION EMBEDDING
        # =================================================

        question_embedding = model.encode(
            [question]
        )

        question_embedding = np.array(
            question_embedding
        ).astype("float32")


        # =================================================
        # FAISS SEMANTIC SEARCH
        # =================================================

        semantic_k = min(
            8,
            len(chunks)
        )

        distance, semantic_result = index.search(
            question_embedding,
            k=semantic_k
        )

        semantic_indices = list(
            semantic_result[0]
        )


        # =================================================
        # KEYWORD SEARCH
        # =================================================

        keyword_indices = keyword_search(
            question,
            chunks
        )


        # =================================================
        # COMBINE SEARCH RESULTS
        # =================================================

        combined_indices = []

        # Keyword results first
        for i in keyword_indices:

            if i not in combined_indices:

                combined_indices.append(i)

        # Semantic results
        for i in semantic_indices:

            if i not in combined_indices:

                combined_indices.append(i)

        # Limit context
        combined_indices = combined_indices[
            :min(
                12,
                len(combined_indices)
            )
        ]


        # =================================================
        # BUILD DOCUMENT CONTEXT
        # =================================================

        context_parts = []

        for i in combined_indices:

            context_parts.append(
                f"""
SOURCE:
{sources[i]}

CONTENT:
{chunks[i]}
"""
            )

        context = "\n".join(
            context_parts
        )


        # =================================================
        # GROQ RESPONSE
        # =================================================

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Searching documents..."
            ):

                try:

                    response = client.chat.completions.create(

                        # AVAILABLE MODEL FROM YOUR GROQ ACCOUNT
                        model="openai/gpt-oss-120b",

                        messages=[

                            {
                                "role": "system",

                                "content":
                                """
                                You are an AI document assistant.

                                Answer ONLY using the provided
                                document context.

                                Carefully search ALL provided
                                context before answering.

                                Look for:

                                - Certificate names
                                - Certificate IDs
                                - Credential IDs
                                - Registration numbers
                                - Course names
                                - Dates
                                - Names
                                - Organizations
                                - Skills
                                - Technologies
                                - Project names
                                - Education details
                                - Work experience

                                For ID questions:

                                Return the EXACT ID from
                                the document.

                                Never modify, change, or invent
                                an ID.

                                Always mention:

                                1. The answer
                                2. Document name
                                3. Page number

                                If multiple documents contain
                                relevant information, mention
                                all relevant documents.

                                Do not guess.

                                If the information is not present,
                                say:

                                "I couldn't find that information
                                in the uploaded documents."
                                """
                            },

                            {
                                "role": "user",

                                "content":
                                f"""
                                DOCUMENT CONTEXT:

                                {context}

                                QUESTION:

                                {question}

                                IMPORTANT:

                                Search every SOURCE and CONTENT
                                section before answering.

                                Return exact information from
                                the documents.

                                Do not guess.
                                """
                            }

                        ]
                    )

                    answer = (
                        response
                        .choices[0]
                        .message
                        .content
                    )

                except Exception as e:

                    answer = (
                        "Sorry, I couldn't generate an answer "
                        "from Groq right now.\n\n"
                        f"Error: {str(e)}"
                    )

            st.markdown(
                answer
            )


        # =================================================
        # SAVE ASSISTANT RESPONSE
        # =================================================

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        # =================================================
        # RETRIEVED SOURCES
        # =================================================

        with st.expander(
            "🔎 View Retrieved Sources"
        ):

            for i in combined_indices:

                st.markdown(
                    f"**📄 {sources[i]}**"
                )

                st.write(
                    chunks[i]
                )

                st.divider()


else:

    # =====================================================
    # WELCOME SCREEN
    # =====================================================

    st.info(
        "Upload one or more PDF documents "
        "from the sidebar to start."
    )

    st.markdown(
        """
        ### What can Ask My Docs do?

        **Multiple PDF Upload**  
        Upload several documents at once.

        **Hybrid Search**  
        Combines semantic FAISS search with
        exact keyword matching.

        **RAG Pipeline**  
        Retrieves relevant document context
        before generating answers.

        **Groq LLM**  
        Generates fast AI responses.

        **Source Tracking**  
        Shows document name and page.

        **Document Analytics**  
        Shows documents, pages, chunks,
        and word statistics.

        **Document Comparison**  
        Compare documents by word count.

        **Chat History**  
        Continue asking questions in the
        same conversation.

        **Download Chat History**  
        Download your questions and answers
        as a text file.
        """
    )