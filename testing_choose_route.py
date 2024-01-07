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

# I know this is probably the wrong/unorthodox way of doing unit tests

now = datetime.now()
# print("Current date and time:", now) -> Current date and time: 2024-01-04 15:53:11.348154

# OpenAI info (global), not needed curently because I am explicity giving in params
OPEN_API_KEY = "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"
openai_api_key = OPEN_API_KEY
chat = ChatOpenAI(temperature=0, openai_api_key=OPEN_API_KEY)

vector_stores_folder_path = "/Users/olivermorris/Documents/GitHub/FeedSense-1.0/vectorStores"



def testing_choose_route(user_question, now, chat):
    print(user_question)
    print(now)
    choose_route = chat(
        messages = [SystemMessage(content = "You are a helpful assistant that chooses an appropriate folder name. The correct date format for you to always use is day_month_year. NEVER respond with anything other than the folder name in your response"),
        HumanMessage(
            content=f"""
            Given: {user_question} and {now}, respond, without any other text, with the most accurate of ONE OF THESE folder names, not including the "" marks:
            "Farm_info_from_25_12_2024_to_01_01_2024"
            "Farm_info_from_18_12_2023_to_25_12_2023"
            "Farm_info_from_11_12_2023_to_18_12_2023"
            "Farm_info_from_04_12_2023_to_11_12_2023"
            "general_nutrition_info"
            """
        ),
    ] 
    )
    route_choice = choose_route.content
    return route_choice

print("running")
print(testing_choose_route("What was the average growth rate 2 weeks ago?", now, chat=chat))