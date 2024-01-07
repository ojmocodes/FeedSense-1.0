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



# folder paths for my local setup
vector_stores_folder_path = "/Users/olivermorris/Documents/GitHub/FeedSense-1.0/vectorStores"
data_folder_path = "/Users/olivermorris/Documents/GitHub/FeedSense-1.0/csvData"


list_of_file_names = [
    "general_nutrition_info.csv", "Farm_info_from_25-12-2024_to_01-01-2024.csv",
    "Farm_info_from_18-12-2023_to_25-12-2023.csv",
    "Farm_info_from_11-12-2023_to_18-12-2023.csv",
    "Farm_info_from_04-12-2023_to_11-12-2023.csv"
]

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
        chunk_size=100,
        chunk_overlap=20,
        length_function=len
    )

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
    vectorstore = FAISS.load_local(f"{vector_stores_folder_path}/{vectorstore_name}", embeddings)
    return vectorstore

# running the final pipeline
csv_to_vectorstore_pipeline(list_of_file_names, data_folder_path)