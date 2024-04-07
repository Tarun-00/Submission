from dotenv import load_dotenv, find_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.docstore.document import Document
import openai
import os

# Function to get OpenAI API key from environment variables
def get_openai_key():
    _ = load_dotenv(find_dotenv())
    return os.environ['OPENAI_API_KEY']

# Function to extract text content from PDF
def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

# Function to set up Streamlit page configuration
def st_first():
    st.set_page_config(
        page_title="Compare PDFs",
        page_icon=":books:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
    )
    st.markdown("""
    <style>
    body {
        background-color: #F5F5F5;
        color: #333333;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stButton>button {
        background-color: #0072C6;
        color: #FFFFFF;
    }
    .stButton>button:hover {
        background-color: #005A9E;
    }
    .stTextInput>div>div>input {
        border: 1px solid #CCCCCC;
        border-radius: 4px;
    }
    .sidebar .sidebar-content {
        padding: 20px;
    }
    .sidebar .sidebar-nav {
        margin-bottom: 20px;
    }
    #input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        background-color: #f0f0f0;
    }
    </style>
    """, unsafe_allow_html=True)
    st.header("Compare Invoices")

# Function to get AI-generated answer using OpenAI's API
def get_ai_answer(question, text_chunks):
    openai.api_key = get_openai_key()
    model_engine = "gpt-3.5-turbo"

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(text_chunks, embeddings)

    # Load the question-answering chain
    qa_chain = load_qa_chain(OpenAI(temperature=0.7, model_name=model_engine), chain_type="stuff")

    # Get the answer
    with get_openai_callback() as cb:
        result = qa_chain({"question": question, "input_documents": text_chunks}, return_only_outputs=True)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")

    if result:
        return result["output_text"]
    else:
        return "Sorry, I couldn't find an answer to your question."

# Main function
def main():
    st_first()

    # Add a navigation sidebar
    nav = st.sidebar.radio("Navigation", ["Compare Invoices", "About"])

    if nav == "Compare Invoices":
        st.title("Compare Invoices")

        # Create a container for the chat
        chat_container = st.container()

        mode = st.checkbox("Compare Two Invoices", value=True)

        all_text_chunks = []  # Initialize with an empty list

        if mode:
            st.subheader("Upload Invoice 1")
            pdf1 = st.file_uploader("Click to Upload Invoice 1", type="pdf")
            st.subheader("Upload Invoice 2")
            pdf2 = st.file_uploader("Click to Upload Invoice 2", type="pdf")

            if pdf1 is not None and pdf2 is not None:
                text1 = get_pdf_text(pdf1)
                text_chunks1 = get_text_chunks(text1)
                text2 = get_pdf_text(pdf2)
                text_chunks2 = get_text_chunks(text2)
                all_text_chunks = text_chunks1 + text_chunks2
        else:
            st.subheader("Upload Invoice")
            pdf = st.file_uploader("Click to Upload Invoice", type="pdf")

            if pdf is not None:
                text = get_pdf_text(pdf)
                text_chunks = get_text_chunks(text)
                all_text_chunks = text_chunks

        # Add the text input container
        input_container = st.container()
        with input_container:
            user_question = st.text_input("Ask a question:", key="user_input")

        if user_question and all_text_chunks:
            ai_answer = get_ai_answer(user_question, all_text_chunks)
            chat_container.markdown(f"**User:** {user_question}")
            chat_container.markdown(f"**AI:** {ai_answer}")

    elif nav == "About":
        st.write("Made By Team CoooooooL!")

if __name__ == '__main__':
    main()
