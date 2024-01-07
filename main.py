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

#hey does this work



# OpenAI info (global), not needed curently because I am explicity giving in params
OPEN_API_KEY = "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"
openai_api_key = OPEN_API_KEY


# this is the function which I have customised the most so
# there is a high chance it is the source of potential issues
# it is intended to loop over various inputted csv's and return text
# from all of them combined, and problems will prob arise from doc/text distincition
def get_csv_text(csv_docs):
    all_csv_text = ""
    for csv_file in csv_docs:
    # loops over all csvs and returns text

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(csv_file.getvalue())
            tmp_file_path = tmp_file.name

        csvLoader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
        csvdoc = csvLoader.load()
        for document in csvdoc:
            csvdoc_content = document.page_content
            all_csv_text += csvdoc_content
    return all_csv_text


# Alejandro's function that iterates over pdf's and returns the text of them all
def get_pdf_text(pdf_docs):
    all_pdf_text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            all_pdf_text += page.extract_text()
    return all_pdf_text

# Alejandro's function that creates chunks from a block of text
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

    chunks = text_splitter.split_text(text)
    return chunks


# Alejandro's function that does text embedding
def get_vectorstore_pdf(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key = "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")
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# The same as Alejandro's function for chunk embedding, but
# I have seperated because might want to change to FAISS.from_documents or something
def get_vectorstore_csv(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key = "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")
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Standard Alejandro function
def get_conversation_chain(csv_vectorstore, pdf_vectorstore):
    llm = ChatOpenAI(openai_api_key = "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")
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    pdf_vectorstore.merge_from(csv_vectorstore)
    pdf_vectorstore.docstore

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=pdf_vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Standard Alejandro function
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


# This function sets up the env, and works with streamlit to get documents
# it then executes the get text, get chunks, embedding, and merging functions
# It then initiates a conversation chain from the final vector store

def main():

    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents (pdf)")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process PDF's'", accept_multiple_files=True)

        st.subheader("Your documents (csv)")
        csv_docs = st.file_uploader(
            "Upload your CSVs here and click on 'Process CSV's'", accept_multiple_files=True)

        if st.button("Process PDF's"):
            with st.spinner("Processing"):

                # get text chunks
                pdf_raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                pdf_text_chunks = get_text_chunks(pdf_raw_text)

                # create vector store
                pdf_vectorstore = get_vectorstore_pdf(pdf_text_chunks)

        if st.button("Process CSV's"):
            with st.spinner("Processing"):
                # get text chunks
                csv_raw_text = get_csv_text(csv_docs)

                # get the text chunks
                csv_text_chunks = get_text_chunks(csv_raw_text)

                # create vector store
                csv_vectorstore = get_vectorstore_csv(csv_text_chunks)

        if st.button("Done, lets go!"):
            with st.spinner("Processing"):
                # get text chunks
                csv_raw_text = get_csv_text(csv_docs)

                # get the text chunks
                csv_text_chunks = get_text_chunks(csv_raw_text)

                # create vector store
                csv_vectorstore = get_vectorstore_csv(csv_text_chunks)
                csv_vectorstore.docstore._dict

                # get text chunks
                pdf_raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                pdf_text_chunks = get_text_chunks(pdf_raw_text)

                # create vector store
                pdf_vectorstore = get_vectorstore_pdf(pdf_text_chunks)
                pdf_vectorstore.docstore._dict

                #merging and creating final vector store
                #final_vectorstore = merge_vector_stores(pdf_vectorstore, csv_vectorstore)
                #final_vectorstore = pdf_vectorstore.merge_from(csv_vectorstore)

                # create conversation chain
                #st.session_state.conversation = get_conversation_chain(final_vectorstore)
                st.session_state.conversation = get_conversation_chain(csv_vectorstore, pdf_vectorstore)



if __name__ == '__main__':
    main()
