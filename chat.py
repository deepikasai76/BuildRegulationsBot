from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import os
import nltk
import config
import logging
import chromadb
from chromadb.config import Settings
import streamlit as st
import app

# # Set the page configuration
# st.set_page_config(page_title="🦜️🔗Langchain PDF Chatbot 🤖", layout='centered')


OPENAI_API_KEY = st.sidebar.text_input("Type your OpenAI API key and press ENTER", placeholder="YOUR_API_KEY")

# Creating a logger object
logger = logging.getLogger(__name__)

# Set the logging level to INFO
logger.setLevel(logging.INFO)

# Create a file handler to log messages to a file
file_handler = logging.FileHandler("logs_file/logs.log")

# Set the file handler's format
file_handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))

# Add the file handler to the logger
logger.addHandler(file_handler)

# Log a message to indicate that the logger has been created
logger.info("Logger Created")

# Load documents from the specified directory using a DirectoryLoader object
loader = DirectoryLoader(config.FILE_DIR, glob='*.pdf')
documents = loader.load()

# split the text to chuncks of of size 1000
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# Split the documents into chunks of size 1000 using a CharacterTextSplitter object
texts = text_splitter.split_documents(documents)

# Create a vector store from the chunks using an OpenAIEmbeddings object and a Chroma object
embeddings = OpenAIEmbeddings(openai_api_key= OPENAI_API_KEY)
docsearch = Chroma.from_documents(texts, embeddings)

# Create a persistent client for the Chroma database
client = chromadb.PersistentClient(path="persist")

# Define answer generation function
def answer(prompt: str, persist_directory = "persist") -> str:
    
    # Log a message indicating that the function has started
    logger.info(f"Start answering based on prompt: {prompt}.")
    
    # Create a prompt template using a template from the config module and input variables
    # representing the context and question.
    prompt_template = PromptTemplate(template=config.prompt_template, input_variables=["context", "question"])
    
    # Load a QA chain using an OpenAI object, a chain type, and a prompt template.
    doc_chain = load_qa_chain(
        llm=OpenAI(
            openai_api_key = config.OPENAI_API_KEY,
            model_name="text-davinci-003",
            temperature=0,
            max_tokens=300,
        ),
        chain_type="stuff",
        prompt=prompt_template,
    )
    
    # Log a message indicating the number of chunks to be considered when answering the user's query.
    logger.info(f"The top {config.k} chunks are considered to answer the user's query.")
    
    # Create a VectorDBQA object using a vector store, a QA chain, and a number of chunks to consider.
    qa = VectorDBQA(vectorstore=docsearch, combine_documents_chain=doc_chain, k=config.k)
    
    # Call the VectorDBQA object to generate an answer to the prompt.
    result = qa({"query": prompt})
    answer = result["result"]
    
    # Log a message indicating the answer that was generated
    logger.info(f"The returned answer is: {answer}")
    
    # Log a message indicating that the function has finished and return the answer.
    logger.info(f"Answering module over.")
    return answer
