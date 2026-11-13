
import os
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# -------------------- SETTINGS --------------------

PDF_PATH = Path("softskill.pdf")
FAISS_PATH = Path("faiss_index")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-5.6"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
RETRIEVAL_K = 4


# -------------------- STREAMLIT CONFIG --------------------

st.set_page_config(
    page_title="Soft Skills PDF RAG Chatbot",
    page_icon="📘"
)

st.title("📘 Soft Skills PDF RAG Chatbot")

st.caption(
    "LangChain + Hugging Face + FAISS + OpenAI"
)


# -------------------- HUGGING FACE EMBEDDINGS --------------------

@st.cache_resource
def get_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings


# -------------------- BUILD FAISS DATABASE --------------------

def build_vector_database():

    if not PDF_PATH.exists():

        st.error("softskill.pdf file not found!")

        return None

    with st.spinner("Reading PDF and creating vector database..."):

        # Load PDF
        loader = PyPDFLoader(str(PDF_PATH))

        documents = loader.load()

        # Split PDF into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        chunks = text_splitter.split_documents(documents)

        if not chunks:

            st.error("No readable text found in PDF.")

            return None

        # Create FAISS vector database
        vector_db = FAISS.from_documents(
            documents=chunks,
            embedding=get_embeddings()
        )

        # Save database locally
        vector_db.save_local(str(FAISS_PATH))

    st.success(
        f"Database created successfully! Total chunks: {len(chunks)}"
    )

    return vector_db


# -------------------- LOAD EXISTING DATABASE --------------------

def load_vector_database():

    index_file = FAISS_PATH / "index.faiss"
    metadata_file = FAISS_PATH / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():

        return None

    vector_db = FAISS.load_local(
        str(FAISS_PATH),
        get_embeddings(),
        allow_dangerous_deserialization=True
    )

    return vector_db


# -------------------- OPENAI LLM --------------------

def get_openai_model():

    api_key = (
        st.session_state.get("openai_api_key")
        or os.getenv("OPENAI_API_KEY")
    )

    if not api_key:

        return None

    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=api_key
    )

    return llm


# -------------------- RAG QUESTION ANSWERING --------------------

def answer_question(question, vector_db, llm):

    # Retrieve relevant documents from FAISS
    retrieved_docs = vector_db.similarity_search(
        question,
        k=RETRIEVAL_K
    )

    # Prepare context
    context_parts = []

    for doc in retrieved_docs:

        page_number = doc.metadata.get("page", 0) + 1

        context_parts.append(
            f"[PDF Page {page_number}]\n{doc.page_content}"
        )

    context = "\n\n".join(context_parts)

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            """
            You are a helpful assistant answering questions
            from the uploaded Soft Skills PDF.

            Follow these instructions:

            1. Use the retrieved PDF context as your primary source.
            2. Do not invent information.
            3. Do not provide unrelated answers.
            4. If the answer is not available in the PDF,
               say:
               "I could not find that information
               in the uploaded PDF."
            5. Explain answers in simple English.
            6. Mention PDF page numbers whenever possible.

            PDF CONTEXT:

            {context}
            """
        ),

        ("human", "{question}")

    ])

    # Generate answer using OpenAI
    chain = prompt | llm

    response = chain.invoke({

        "context": context,

        "question": question

    })

    # Get source pages
    source_pages = sorted({

        doc.metadata.get("page", 0) + 1

        for doc in retrieved_docs

    })

    return response.content, source_pages


# -------------------- SIDEBAR --------------------

with st.sidebar:

    st.header("Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password"
    )

    if api_key:

        st.session_state["openai_api_key"] = api_key

    st.write("LLM:", OPENAI_MODEL)

    st.write("Embedding:", EMBEDDING_MODEL)

    # Build database button
    if st.button("Build / Rebuild FAISS Database"):

        st.session_state["vector_db"] = (
            build_vector_database()
        )

    # Clear chat button
    if st.button("Clear Chat"):

        st.session_state["messages"] = []

        st.rerun()


# -------------------- INITIALIZE SESSION --------------------

if "vector_db" not in st.session_state:

    st.session_state["vector_db"] = load_vector_database()


if "messages" not in st.session_state:

    st.session_state["messages"] = []


# -------------------- CHAT INTERFACE --------------------

if st.session_state["vector_db"] is None:

    st.info(
        "Put softskill.pdf in this folder and click "
        "'Build / Rebuild FAISS Database'."
    )

else:

    st.success("FAISS Vector Database is ready!")


# Display chat history
for message in st.session_state["messages"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# Chat input
question = st.chat_input(
    "Ask a question from your Soft Skills PDF..."
)


if question:

    api_key = (
        st.session_state.get("openai_api_key")
        or os.getenv("OPENAI_API_KEY")
    )

    if not api_key:

        st.warning(
            "Please enter your OpenAI API key in the sidebar."
        )

        st.stop()

    if st.session_state["vector_db"] is None:

        st.warning(
            "Please build the FAISS database first."
        )

        st.stop()

    # Display user question
    st.session_state["messages"].append({

        "role": "user",

        "content": question

    })

    with st.chat_message("user"):

        st.markdown(question)

    # Generate assistant response
    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Searching PDF and generating answer with OpenAI..."
            ):

                llm = get_openai_model()

                answer, source_pages = answer_question(
                    question,
                    st.session_state["vector_db"],
                    llm
                )

            st.markdown(answer)

            st.caption(
                "Source PDF pages: "
                + ", ".join(map(str, source_pages))
            )

            st.session_state["messages"].append({

                "role": "assistant",

                "content": answer

            })

        except Exception as error:

            st.error(f"Error: {error}")

            st.info(
                "Check your API key, internet connection, "
                "model access and API billing."
            )
