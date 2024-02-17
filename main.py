import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceHub
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.document_loaders import GitLoader
import pandas as pd
import os
import tempfile
from datetime import datetime
from datetime import datetime, timedelta

now = datetime.now()
now = str(now).split()[0]
now = f"{now[-2:]}_{now[-5:-3]}_{now[:4]}"
# print("Current date and time:", now) -> Current date and time: 2024-01-04 15:53:11.348154

# OpenAI info (global), not needed curently because I am explicity giving in params
OPEN_API_KEY = "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"
openai_api_key = OPEN_API_KEY
chat = ChatOpenAI(temperature=0, openai_api_key=OPEN_API_KEY)

#vector_stores_folder_path = "/Users/olivermorris/Documents/GitHub/FeedSense-1.0/newSecondCharSplitVectorStores"
vector_stores_folder_path = "FeedSense-1.0/SecondCharSplitVectorStores"

first = '''def handle_userinput(user_question, now, chat):
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
    '''

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

list_of_file_names = list_files_in_folder("new_csvs")

def load_vectorstore_locally(vectorstore_name, vector_stores_folder_path):
    embeddings = OpenAIEmbeddings(openai_api_key = "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")
    st.success(f"heres")
    vectorstore = FAISS.load_local(f"{vector_stores_folder_path}/{vectorstore_name}", embeddings)
    return vectorstore

def date_to_route(date, date_ranges=list_of_file_names):
    input_date = datetime.strptime(date, "%d_%m_%Y")

    for range in date_ranges:
        range = range.split(".")[0]
        end_of_week = datetime.strptime(range, "%d_%m_%Y")
        start_of_week = end_of_week - timedelta(days=6)

        if start_of_week <= input_date <= end_of_week:
            print(start_of_week)
            print(end_of_week)
            return range
    return None

def handle_userinput2(user_question, now, chat):
    # route to relevant db
    # get all results
    # feed them into completion
    # return completion.content

    # how route to relevant db?

    # query_to_date() - take in {now} and {user_question}, output -> relevant date in the form XX_YY_ZZZZ - here the focus is prompt engineering to get reliable output formatting

    query_to_date = chat(
        messages = [
            SystemMessage(content=
                          
            """
            ALWAYS RESPOND IN THE FORMAT: DAY_MONTH_YEAR. You are a helpful assistant that returns the date relevant to the question. Always return in the same format of XX_YY_ZZZZ.
            For example if the current date is 15_02_2024 and the question is asking about 12 weeks ago, return 06_11_2023
            For example if the current date is 17_02_2024 and the question is reffering to last July, return 17_07_2024
            """),
            
            HumanMessage(content=f"""
                         
            Given: The question "{user_question}" and that the current date is "{now}", respond, without any other text, the date relevant to the question. ALWAYS RESPOND IN THE FORMAT: DAY_MONTH_YEAR.

            """
        ),
    ] 
    )
    date_choice = query_to_date.content
    
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

    # need to create some python function that goes from date_choice to route_choice, in practice this means going from specific date to choosing a range

    print(date_choice)
    route_choice = date_to_route(date_choice)

    #relevant_vector_store = load_vectorstore_locally(route_choice, "/Users/olivermorris/Documents/GitHub/FeedSense-1.0/newVectorStores")
    #result = relevant_vector_store.similarity_search(q_as_present_tense, k=3)[0].page_content

    embeddings = OpenAIEmbeddings(openai_api_key = "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")
    st.success(f"DEV NOTES: Date choice: {date_choice}")
    st.success(f"DEV NOTES: Route choice: {route_choice}")
    #relevant_vector_store = FAISS.load_local(f"/Users/olivermorris/Documents/GitHub/FeedSense-1.0/SecondCharSplitVectorStores/{route_choice}", embeddings)
    #relevant_vector_store = FAISS.load_local(f"FeedSense-1.0/SecondCharSplitVectorStores/{route_choice}", embeddings)
    folder_path = os.path.join(os.path.dirname(__file__), 'SecondCharSplitVectorStores')
    file_path = os.path.join(folder_path, route_choice)
    relevant_vector_store = FAISS.load_local(f"{file_path}", embeddings)
    result = relevant_vector_store.similarity_search(q_as_present_tense, k=5)[0].page_content
    st.success(f'DEV NOTES: Similarity search result: {result}')


    completion = chat(
        messages = [
        SystemMessage(
            content=f"You are a helpful farming assistant. Take a deep breath, work step by step. Don't say 'According to the information provided...' or anything similar. All data is in the form kg N applied / ha YTD,60. As in the label is on the left, and the data is on the right, sepearted by a comma. Be polite and kind."
        ),
        HumanMessage(
            content=f"Use this information: '{result}' to answer the question: {user_question}."
        ),
    ]
    )

    FINAL = completion.content
    return FINAL

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
        processed_input = handle_userinput2(user_question, now=now, chat=chat)

        # Display the processed input
        st.success(f"{processed_input}")


if __name__ == '__main__':
    main()