from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import (PyPDFLoader, DataFrameLoader, GitLoader)
import pandas as pd
import os
import tempfile
from datetime import datetime
import csv

#13_11_2023 is missing

# folder paths for my local setup
#vector_stores_folder_path = "/Users/olivermorris/Documents/GitHub/FeedSense-1.0/newVectorStores"
vector_stores_folder_path = "/Users/olivermorris/Documents/GitHub/FeedSense-1.0/GPTCharSplitVectorStores"
data_folder_path = "/Users/olivermorris/Documents/GitHub/FeedSense-1.0/new_csvs"

def csv_to_text_chunks(csv_file):
    # this function takes a csv file and converts it into a list of lines (each line is a chunk to sim search for)
    text_chunks = []
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            # Convert row to text chunk
            text_chunk = ', '.join(row)
            text_chunks.append(text_chunk)
    return text_chunks

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

def adapted_csv_to_vectorstore_pipeline(list_of_file_names, data_folder_path):
    for the_file_name in list_of_file_names:
        file_name_for_vectorstore = the_file_name.rsplit(".")[0] # don't want file type in name
        text_chunks = csv_to_text_chunks(f"{data_folder_path}/{the_file_name}")
        vectorstore = create_vectorstore(text_chunks)
        save_vectorstore_locally(vectorstore, file_name_for_vectorstore, vector_stores_folder_path)

#can do .from_documents too
def create_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key = "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")
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def save_vectorstore_locally(vectorstore, desired_name, vector_stores_folder_path):
    vectorstore.save_local(f"{vector_stores_folder_path}/{desired_name}")

# running the final pipeline, takes folder name, iterates over and 
folder_name = "new_csvs"
list_of_file_names = list_files_in_folder(folder_name)
adapted_csv_to_vectorstore_pipeline(list_of_file_names, data_folder_path)