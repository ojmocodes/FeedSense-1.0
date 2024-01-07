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
from createVectorStores import load_vectorstore_locally

def test_querying_vectorstore():
    # route_choice = vectorstore_name
    print("running")
    vector_stores_folder_path = "/Users/olivermorris/Documents/GitHub/FeedSense-1.0/vectorStores"
    route_choice = "Farm_info_from_25_12_2024_to_01_01_2024"
    user_question = "What was the average growth rate this week?"

    relevant_vector_store = load_vectorstore_locally(route_choice, vector_stores_folder_path)
    result = relevant_vector_store.similarity_search(user_question, k=3)[0].page_content
    return result

print(test_querying_vectorstore())