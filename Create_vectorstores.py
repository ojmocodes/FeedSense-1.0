import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
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



class a_csv_doc():
    def __init__(self, file_name, file_path):
        self.file_name = file_name
        self.file_path = file_path



nutritional_info = a_csv_doc("Detailed_nutrition_info", "/Users/olivermorris/PycharmProjects/pythonProject2/ask-multiple-pdfs/data/Detailed_nutrition_info.csv")



# this is only for one csv, but when needed (all owl farm csv's are in data folder) can create a function that loops
def get_csv_text(the_csv_doc):
    all_csv_text = ""

    loader = CSVLoader(file_path=the_csv_doc.file_path)
    loaded_csvdoc = loader.load()

    for document in loaded_csvdoc:
        loaded_csvdoc_content = document.page_content
        all_csv_text += loaded_csvdoc_content
    return all_csv_text


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

def save_vectorstore_locally(vectorstore, desired_name):
    vectorstore.save_local(f"{desired_name}")

def load_vectorstore_locally(vectorstore_name):
    embeddings = OpenAIEmbeddings(openai_api_key = "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")
    vectorstore = FAISS.load_local(f"{vectorstore_name}", embeddings)
    return vectorstore