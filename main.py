import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from htmlTemplates import css, bot_template, user_template
import pandas as pd
import os
from datetime import datetime
from datetime import datetime, timedelta

#reformatting datetime given into day/month/year and making it a string instead of a datetime object
now = datetime.now()
now = str(now).split()[0]
now = f"{now[-2:]}_{now[-5:-3]}_{now[:4]}"
# print("Current date and time:", now) -> Current date and time: 2024-01-04 15:53:11.348154

# OpenAI info (global), not needed curently because I am explicity giving in params
OPEN_API_KEY = "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"
openai_api_key = OPEN_API_KEY
chat = ChatOpenAI(temperature=0, openai_api_key=OPEN_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key = "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")

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

# takes date and list of file names and returns the file name relevant to the date
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

    # from user question to relevant date
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
    
    # here am rephrasing the q into present tense for similarity search and presentation
    rephrase_q = chat(
        messages = [SystemMessage(content = "You are a helpful assistant that rephrases question into present tense, and proper acronyms."),
        HumanMessage(
            content=f"""
            Transform this question: "{user_question}" into the exact same question but rephrased into the present tense.
            For example: turn "...two weeks ago?" into "...this week?"

            please rephrase "palm kernel" into "PKE"


            """
        ),
    ] 
    )
    q_as_present_tense = rephrase_q.content

    date_choice = query_to_date.content
    route_choice = date_to_route(date_choice)
    st.success(f"DEV NOTES: Date choice: {date_choice}")
    st.success(f"DEV NOTES: Route choice: {route_choice}")

    # these blocks here are the magic
    
    folder_path = os.path.join(os.path.dirname(__file__), 'GPTCharSplitVectorStores')
    file_path = os.path.join(folder_path, route_choice)
    relevant_vector_store = FAISS.load_local(f"{file_path}", embeddings)
    result = relevant_vector_store.similarity_search(q_as_present_tense, k=3)[0].page_content
    st.success(f'DEV NOTES: Similarity search result: {result}')


    completion = chat(
        messages = [
        SystemMessage(
            content=f"You are a helpful farming assistant. Take a deep breath. Don't say 'According to the information provided...' or anything similar."
        ),
        HumanMessage(
            content=f"""
            Answer the question: {user_question} given the relevant data '{result}', this data is related to the time period. Be kind, and try be as helpful as possible. The data is related to the relevant time period.
            """
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

    user_question = st.text_input("Ask a question about Owl Farm data from 01-05-2023 to 22-01-2024.")

    if st.button("Process"):
        # Call the function to process the input
        processed_input = handle_userinput2(user_question, now=now, chat=chat)

        # Display the processed input
        st.success(f"{processed_input}")


if __name__ == '__main__':
    main()