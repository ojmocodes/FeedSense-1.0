import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import (PyPDFLoader, DataFrameLoader, GitLoader)
import pandas as pd
import os
import tempfile
from datetime import datetime
from langchain.text_splitter import SpacyTextSplitter
import new_csvs

#13_11_2023 is missing

# folder paths for my local setup
#vector_stores_folder_path = "/Users/olivermorris/Documents/GitHub/FeedSense-1.0/newVectorStores"
vector_stores_folder_path = "/Users/olivermorris/Documents/GitHub/FeedSense-1.0/SecondCharSplitVectorStores"
data_folder_path = "/Users/olivermorris/Documents/GitHub/FeedSense-1.0/new_csvs"


def list_files_in_folder(folder_name):
    # Get the current directory
    current_directory = os.getcwd()

    # Create the full path to the specified folder
    folder_path = os.path.join(current_directory, folder_name)

    # Check if the folder exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # List all files in the folder
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        if ".DS_Store" in files:
            files.remove(".DS_Store")
        return files
    else:
        print(f"The folder '{folder_name}' does not exist.")
        return []

class a_csv_doc():
    def __init__(self, file_name, file_path):
        self.file_name = file_name
        self.file_path = file_path

def csv_to_vectorstore_pipeline(list_of_file_names, data_folder_path):
    for the_file_name in list_of_file_names:
        file_name_for_vectorstore = the_file_name.rsplit(".")[0] # don't want file type in name
        csv_as_doc = a_csv_doc(the_file_name, f"{data_folder_path}/{the_file_name}")
        csv_text = get_csv_text(csv_as_doc)
        text_chunks = get_text_chunks(csv_text)
        vectorstore = create_vectorstore(text_chunks)
        save_vectorstore_locally(vectorstore, file_name_for_vectorstore, vector_stores_folder_path)


# this is only for one csv, but when needed (all owl farm csv's are in data folder) can create a function that loops
def get_csv_text(the_csv_doc):
    loader = CSVLoader(file_path=the_csv_doc.file_path)
    loaded_csvdoc = loader.load()

    all_csv_text = []
    for document in loaded_csvdoc:
        loaded_csvdoc_content = document.page_content
        all_csv_text.append(loaded_csvdoc_content)
    return ''.join(all_csv_text)


def get_text_chunks(text):

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=50,
        length_function=len
    )

    spacy_splitter = SpacyTextSplitter(chunk_size=1000)

    recursive_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )

    text_chunks = text_splitter.split_text(text)
    return text_chunks

#can do .from_documents too
def create_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key = "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")
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def save_vectorstore_locally(vectorstore, desired_name, vector_stores_folder_path):
    vectorstore.save_local(f"{vector_stores_folder_path}/{desired_name}")

def load_vectorstore_locally(vectorstore_name, vector_stores_folder_path):
    embeddings = OpenAIEmbeddings(openai_api_key = "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")
    st.success(f"heres")
    vectorstore = FAISS.load_local(f"{vector_stores_folder_path}/{vectorstore_name}", embeddings)
    return vectorstore

# running the final pipeline
folder_name = "new_csvs"
list_of_file_names = list_files_in_folder(folder_name)
csv_to_vectorstore_pipeline(list_of_file_names, data_folder_path)