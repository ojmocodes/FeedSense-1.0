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

now = datetime.now()
# print("Current date and time:", now) -> Current date and time: 2024-01-04 15:53:11.348154

# OpenAI info (global), not needed curently because I am explicity giving in params
OPEN_API_KEY = "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"
openai_api_key = OPEN_API_KEY
chat = ChatOpenAI(temperature=0, openai_api_key=OPEN_API_KEY)

vector_stores_folder_path = "/Users/olivermorris/Documents/GitHub/FeedSense-1.0/vectorStores"


def handle_userinput(user_question, now, chat):
    # route to relevant db
    # get all results
    # feed them into completion
    #return completion.content

    choose_route = chat(
        messages = [SystemMessage(content = "You are a helpful assistant that chooses an appropriate folder name. The correct date format for you to always use is day_month_year. NEVER respond with anything other than the folder name in your response. If you're given a date range, always respond with the most recent date."),
        HumanMessage(
            content=f"""
            Given: {user_question} and that the current date is "{now}", respond, without any other text, with the most relevant of ONE OF THESE folder names, make sure to only reply with the name, nothing else:
            Farm_info_from_25_12_2023_to_01_01_2024
            Farm_info_from_18_12_2023_to_25_12_2023
            Farm_info_from_11_12_2023_to_18_12_2023
            Farm_info_from_04_12_2023_to_11_12_2023
            general_nutrition_info


            """
        ),
    ] 
    )
    route_choice = choose_route.content

    # here am rephrasing the q into present tense (only want "2 weeks ago" when routing)

    rephrase_q = chat(
        messages = [SystemMessage(content = "You are a helpful assistant that rephrases question into present tense."),
        HumanMessage(
            content=f"""
            Transform this question: "{user_question}" into the exact same question but rephrased into the present tense.
            For example: turn "...two weeks ago?" into "...this week?"
            """
        ),
    ] 
    )
    q_as_present_tense = rephrase_q.content



    relevant_vector_store = load_vectorstore_locally(route_choice, vector_stores_folder_path)
    result = relevant_vector_store.similarity_search(q_as_present_tense, k=3)[0].page_content



    completion = chat(
        messages = [
        SystemMessage(
            content=f"You are a helpful farming assistant. Take a deep breath, work step by step. Don't say 'According to the information provided...' or anything similar. Reply with full context, including assumptions, what figures mean, etc. Be polite and kind."
        ),
        HumanMessage(
            content=f"Use this information: '{result}' to answer the question: {q_as_present_tense}."
        ),
    ]
    )

    FINAL = completion.content
    return FINAL


    # legacy code that may be useful soon for having a chat history (allows follow ups)

    #response = st.session_state.conversation({'question': FINAL})
    #st.session_state.chat_history = response['chat_history']

    #for i, message in enumerate(st.session_state.chat_history):
    #    if i % 2 == 0:
    #        st.write(user_template.replace(
    #            "{{MSG}}", message.content), unsafe_allow_html=True)
    #    else:
    #        st.write(bot_template.replace(
    #            "{{MSG}}", message.content), unsafe_allow_html=True)

def main():

    load_dotenv()
    st.set_page_config(page_title="Chat with FeedSense - your digital farming brain",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with FeedSense!")

    user_question = st.text_input("Ask a question!:")

    if st.button("Process"):
        # Call the function to process the input
        processed_input = handle_userinput(user_question, now=now, chat=chat)

        # Display the processed input
        st.success(f"{processed_input}")


if __name__ == '__main__':
    main()