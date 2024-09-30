import os
import certifi
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# ----------------------------
# 1. Load Environment Variables
# ----------------------------

# Load environment variables from .env file
load_dotenv()

# Retrieve the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is provided
if not openai_api_key:
    st.error("Please set the OPENAI_API_KEY environment variable in the .env file.")
    st.stop()

# ----------------------------
# 2. Configure SSL Certificates
# ----------------------------

# Set the SSL certificate path
os.environ['SSL_CERT_FILE'] = certifi.where()

# ----------------------------
# 3. Define Helper Functions
# ----------------------------

def get_pdf_text(pdf_docs):
    """
    Extract text from a list of PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.warning(f"Failed to read {pdf.name}: {e}")
    return text

def get_text_chunks(text):
    """
    Split text into manageable chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vectorstore(text_chunks):
    """
    Create a FAISS vector store from text chunks.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def initialize_conversation_chain(vector_store):
    """
    Initialize the Conversational Retrieval Chain.
    """
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Initialize the ConversationalRetrievalChain
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,  # Disable source documents to avoid multiple output keys
        verbose=True
    )
    
    return conversation_chain

# ----------------------------
# 4. Streamlit Application
# ----------------------------

def main():
    st.set_page_config(page_title="Ask Mr. DM", page_icon=":books:")
    st.title("Ask Mr. Data Modeler")
    
    # Initialize session state variables
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    def initialize_chain():
        if st.session_state.vector_store and st.session_state.conversation_chain is None:
            st.session_state.conversation_chain = initialize_conversation_chain(st.session_state.vector_store)
            st.success("Conversation chain initialized.")
    
    # ----------------------------
    # 4.1 Load PDFs from Folder
    # ----------------------------
    if st.session_state.vector_store is None:
        pdf_docs_path = './pdf-docs'  # Folder containing PDF documents
        if os.path.exists(pdf_docs_path) and os.path.isdir(pdf_docs_path):
            st.write(f"Loading PDF documents from: `{pdf_docs_path}`")
            pdf_files = [
                os.path.join(pdf_docs_path, pdf) 
                for pdf in os.listdir(pdf_docs_path) 
                if pdf.lower().endswith('.pdf')
            ]
            if not pdf_files:
                st.warning(f"No PDF files found in `{pdf_docs_path}`.")
            else:
                pdf_docs = [open(pdf_file, 'rb') for pdf_file in pdf_files]
                raw_text = get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = create_vectorstore(text_chunks)
                    st.success("Vector store created from PDFs.")
                else:
                    st.error("No text extracted from the provided PDFs.")
        else:
            st.warning(f"Directory `{pdf_docs_path}` does not exist. You can upload PDFs below.")
    
    initialize_chain()
    
    # ----------------------------
    # 4.3 Chat Interface
    # ----------------------------
    if st.session_state.conversation_chain:
        user_question = st.chat_input("Type your message here...")
        if user_question:
            with st.spinner("Generating response..."):
                try:
                    response = st.session_state.conversation_chain({'question': user_question})
                    answer = response['answer']
                    st.session_state.chat_history.append({"role": "user", "content": user_question})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                    # Display the chat history
                    for message in st.session_state.chat_history:
                        if message["role"] == "user":
                            with st.chat_message("user"):
                                st.write(message["content"])
                        else:
                            with st.chat_message("assistant"):
                                st.write(message["content"])
                except Exception as e:
                    st.error(f"Error generating response: {e}")
    
    # ----------------------------
    # 4.4 Sidebar for Uploading PDFs
    # ----------------------------
    with st.sidebar:
        st.header("Add Your Documents")
        uploaded_pdfs = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
        if uploaded_pdfs:
            with st.spinner("Processing uploaded PDFs..."):
                raw_text = get_pdf_text(uploaded_pdfs)
                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    new_vectorstore = create_vectorstore(text_chunks)
                    if st.session_state.vector_store:
                        # Merge the new vector store with the existing one
                        st.session_state.vector_store.merge_from(new_vectorstore)
                    else:
                        st.session_state.vector_store = new_vectorstore
                    initialize_chain()
                    st.success("Uploaded PDFs have been added and processed.")
                else:
                    st.error("No text extracted from the uploaded PDFs.")

if __name__ == "__main__":
    main()