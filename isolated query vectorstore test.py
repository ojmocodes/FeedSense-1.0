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

OPEN_API_KEY = "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"
openai_api_key = OPEN_API_KEY
chat = ChatOpenAI(temperature=0, openai_api_key=OPEN_API_KEY)

def test_querying_vectorstore():
    # route_choice = vectorstore_name
    vector_stores_folder_path = "/Users/olivermorris/Documents/GitHub/FeedSense-1.0/vectorStores"
    route_choice = "Farm_info_from_11_12_2023_to_18_12_2023"
    user_question = "What is my protein to fat ratio currently?"

    relevant_vector_store = load_vectorstore_locally(route_choice, vector_stores_folder_path)
    result = relevant_vector_store.similarity_search(user_question, k=3)[0].page_content
    return result

rephrase_q = chat(
        messages = [SystemMessage(content = "You are a helpful assistant that rephrases question into present tense."),
        HumanMessage(
            content=f"""
            Transform this question: "{"What was my protein to fat ratio 3 weeks ago?"}" into the exact same question but rephrased into the present tense.
            For example: turn "...two weeks ago?" into "...this week?"
            """
        ),
    ] 
    )

print(test_querying_vectorstore())