import os
import streamlit as st
import utils
from retriever import retriever
from LLM import model
from langchain_core.runnables import RunnablePassthrough

AUTH_TOKEN= "Enter your authorization token of your huggingface account"

# Page configuration
st.set_page_config(page_title="Chat with Your Documents", layout="wide")

# Initialize session state for chat history and context
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = ""
if "document_context" not in st.session_state:
    st.session_state.document_context = ""
if "retriever_initialized" not in st.session_state:
    st.session_state.retriever_initialized = False

# File uploader for multiple PDF and DOCX files
st.sidebar.title("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload your PDF or DOCX files", type=["pdf"], accept_multiple_files=True
)

# Process the uploaded files
if uploaded_files and not st.session_state.retriever_initialized:
    try:
        all_extracted_text = []

        # Iterate over each uploaded file and extract text
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.name.split(".")[-1].lower()
            if file_type == "pdf":
                st.sidebar.info(f"Processing PDF file: {uploaded_file.name}")
                extracted_text = utils.extract_pdf_to_documents(uploaded_file)
            else:
                st.sidebar.error(f"Unsupported file type: {uploaded_file.name}")
                extracted_text = None

            # Append extracted text if successful
            if extracted_text:
                all_extracted_text.extend(extracted_text)

        # Split the combined extracted text into chunks and initialize the retriever
        if all_extracted_text:
            chunks = utils.split_text(all_extracted_text)
            persist_directory = "chroma_data"
            os.makedirs(persist_directory, exist_ok=True)

            # Initialize retriever and LLM
            odj = retriever(chunks, persist_directory)
            st.session_state.retriever = odj.get_retriever()
            llm = model('meta-llama/Llama-2-7b-chat-hf', AUTH_TOKEN)
            st.session_state.llm_pipeline = llm.initialize_pipeline()
            st.session_state.rag_chain = {"context": st.session_state.retriever, "question": RunnablePassthrough()} | st.session_state.llm_pipeline

            st.sidebar.success("Files processed successfully!")
            st.session_state.retriever_initialized = True
            st.info("Document context has been successfully loaded. You can now ask questions about it.")
    except Exception as e:
        st.sidebar.error(f"An error occurred while processing the files: {e}")

# Chat interface
st.title("Chat with Your Documents 📄🤖")

# Text input for user query
user_query = st.text_input("Ask your question about the documents:", key="user_input")

# Function to handle user input and update chat history
def handle_user_input(query):
    try:
        # Combine user query with the conversation context
        conversation_input = st.session_state.conversation_context + f"\nUser: {query}\nAssistant:"
        response = st.session_state.rag_chain.invoke(conversation_input)

        # Extract the latest response
        response_text = response.split("<|assistant|>")[-1].strip()

        # Update chat history and conversation context
        st.session_state.chat_history.append((query, response_text))
        st.session_state.conversation_context += f"\nUser: {query}\nAssistant: {response_text}"
    except Exception as e:
        st.error(f"An error occurred while generating the response: {e}")

# Handle user input when the 'Submit' button is clicked
if st.button("Submit"):
    if user_query:
        handle_user_input(user_query)
    else:
        st.warning("Please enter a question.")

# Display chat history dynamically
if st.session_state.chat_history:
    for i, (query, reply) in enumerate(st.session_state.chat_history):
        # User message (left-aligned)
        st.markdown(
            f"""
            <div style='display: flex; align-items: flex-start; margin: 15px 0;'>
                <div style='padding: 12px; border: 1px solid #ddd; border-radius: 10px; max-width: 80%;'>
                    <strong>User:</strong> {query}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Assistant message (right-aligned)
        st.markdown(
            f"""
            <div style='display: flex; align-items: flex-start; justify-content: flex-end; margin: 15px 0;'>
                <div style='padding: 12px; border: 1px solid #ddd; border-radius: 10px; max-width: 80%;'>
                    <strong>Assistant:</strong> {reply}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

