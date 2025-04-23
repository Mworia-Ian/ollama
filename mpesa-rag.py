# app.py

import streamlit as st
import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DEFAULT_DOC_PATH = "./data/MPESA_Statement_2025-03-16.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_mpesa_db"


def ingest_pdf(doc_path, password=None):
    """Load PDF documents with optional password protection."""
    if os.path.exists(doc_path):
        try:
            # Using PyPDFLoader instead of UnstructuredPDFLoader for password support
            loader = PyPDFLoader(file_path=doc_path, password=password)
            data = loader.load()
            logging.info("PDF loaded successfully.")
            return data, None
        except Exception as e:
            error_msg = str(e)
            if "password required" in error_msg.lower() or "incorrect password" in error_msg.lower():
                logging.error("PDF is password protected or incorrect password provided.")
                return None, "Password required or incorrect password provided."
            else:
                logging.error(f"Error loading PDF: {error_msg}")
                return None, f"Error loading PDF: {error_msg}"
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        return None, "PDF file not found."


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


def create_or_load_vector_db(doc_path, password=None):
    """Load or create the vector database."""
    # Pull the embedding model if not already available
    ollama.pull(EMBEDDING_MODEL)

    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Generate a unique persist directory based on the document path
    doc_name = os.path.basename(doc_path).replace(".", "_")
    persist_dir = f"./chroma_{doc_name}_db"

    if os.path.exists(persist_dir):
        vector_db = Chroma(
            embedding_function=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=persist_dir,
        )
        logging.info("Loaded existing vector database.")
        return vector_db, None
    else:
        # Load and process the PDF document
        data, error = ingest_pdf(doc_path, password)
        if data is None:
            return None, error

        # Split the documents into chunks
        chunks = split_documents(data)

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=persist_dir,
        )
        vector_db.persist()
        logging.info("Vector database created and persisted.")
        return vector_db, None


def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create the chain with preserved syntax."""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created with preserved syntax.")
    return chain


def main():
    st.title("Password-Protected Document Assistant")
    
    # File upload option
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    # Password input for protected PDFs
    password = st.text_input("PDF Password (leave empty if none):", type="password")
    if password == "":
        password = None
    
    # Determine which document to use
    doc_path = DEFAULT_DOC_PATH
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_path = f"./uploaded_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        doc_path = temp_path
    
    # Initialize database
    vector_db = None
    error_msg = None
    
    if st.button("Process Document"):
        with st.spinner("Processing document..."):
            vector_db, error_msg = create_or_load_vector_db(doc_path, password)
            
            if error_msg:
                st.error(error_msg)
            else:
                st.session_state.vector_db = vector_db
                st.success("Document processed successfully!")

    # User input
    user_input = st.text_input("Enter your question:", "")

    if user_input and 'vector_db' in st.session_state:
        with st.spinner("Generating response..."):
            try:
                # Initialize the language model
                llm = ChatOllama(model=MODEL_NAME)

                # Create the retriever
                retriever = create_retriever(st.session_state.vector_db, llm)

                # Create the chain
                chain = create_chain(retriever, llm)

                # Get the response
                response = chain.invoke(input=user_input)

                st.markdown("**Assistant:**")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    elif user_input and 'vector_db' not in st.session_state:
        st.warning("Please process a document first before asking questions.")
    else:
        st.info("Upload a document, enter a password if needed, process the document, and then ask questions.")


if __name__ == "__main__":
    main()