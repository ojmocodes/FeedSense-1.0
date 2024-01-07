def test_querying_vectorstore():
    vector_stores_folder_path = "/Users/olivermorris/Documents/GitHub/FeedSense-1.0/vectorStores"
    embeddings = OpenAIEmbeddings(openai_api_key = "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")

    new_db = FAISS.load_local("vectorStores/Farm_info_from_25-12-2024_to_01-01-2024", embeddings)

    docs = new_db.similarity_search("What was the SCC this week?", k=3)
    print(docs[0].page_content)



def test_streamlit():

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
        processed_input = "hey man just testing here"

        # Display the processed input
        st.success(f"{processed_input}")


# streamlit working


def testing_handle_userinput(user_question, now, chat):
    # route to relevant db
    # get all results
    # feed them into completion
    #return completion.content

    choose_route = chat(
        messages = [
        HumanMessage(
            content=f"""
            The correct date format for you to always use is day_month_year.
            Given: {user_question} and {now}, respond, without any other text, with the most accurate of ONE OF THESE TITLES, not including the "" marks:
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

    relevant_vector_store = load_vectorstore_locally(route_choice, vector_stores_folder_path)
    result = relevant_vector_store.similarity_search(user_question, k=1)[0].page_content

    # here want to get "result" by similarity searching  the route_choice, and then pass that into completion, as part of this the vector store needs to be loaded with the load_vectorstore_locally function in Create_vectorstores


    completion = chat(
        messages = [
        SystemMessage(
            content=f"You are a helpful farming assistant. Take a deep breath, work step by step. Don't say 'According to the information provided...' or anything similar. Reply with full context, including assumptions, what figures mean, etc."
        ),
        HumanMessage(
            content=f"Use this information: '{result}' to answer the question: {user_question}. Answer Concisely."
        ),
    ]
    )

    FINAL = completion.content
    return FINAL