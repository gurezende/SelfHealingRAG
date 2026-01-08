import os
import streamlit as st
from langgraph_agent.graph import build_graph
from langgraph_agent.document_loader import load_document
from langgraph_agent.retrieve_docs import *
from langgraph_agent.nodes import *
from IPython.display import display, Image


# Config page
st.set_page_config(page_title="Self-Healing RAG",
                   page_icon='🤖',
                   layout="wide",
                   initial_sidebar_state="expanded")


# Add a place to enter the API key
with st.sidebar:
    api_key = st.text_input("OPENAI_API_KEY", type="password")

    # Save the API key to the environment variable
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    # Clear
    if st.button('Clear'):
        st.rerun()
    
    st.divider()
    # Load document to streamlit
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    # Print File Uploaded
    if uploaded_file is not None:
        st.caption("File uploaded")

    # If a file is uploaded, create the TextSplitter and vector database
    if uploaded_file :

        # Code to work around document loader from Streamlit and make it readable by langchain
        temp_file = "./temp.pdf"
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

        # Load document and split it into chunks for efficient retrieval.
        documents = load_document(temp_file)

        # Message user that document is being processed with time emoji
        st.spinner("Processing document... :watch:")

    st.divider()
    # About
    st.write("Designed with :heart: by [Gustavo R. Santos](https://gustavorsantos.me)")

# Title and Instructions
if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar.")
    
st.title('Self-Healing RAG Agent | 🤖')
st.markdown('This AI Agent is trained to answer questions about the content from PDF File loaded.')
st.markdown("""
            The agent will then search the answer in the PDF file and try to generate an answer based on the content.   
            In case of failure, the agent tries to **:blue[self-heal] the RAG with the following strategr:**   
            * Increase the number of documents to retrieve from the PDF file
            * Reranking the results.
            """)
st.caption('Ask questions like: "Who is the author of this document?"')

st.divider()


# User question
question = st.text_input(label="Ask me something from your document:",
                         placeholder= "e.g. What is the definition of A/B testing?")


# Run the graph
if st.button('Search'):

    with st.spinner("Thinking..", show_time=True):

        # Log
        st.write(":file_folder: | Log of execution:")

        # Build the graph
        graph = build_graph()
        result = graph.invoke({
            "query": question,
            "retrieval_mode": "original",
            "retrieval_budget": 2,
            "retry_count": 1,
            "max_retries": 2,
            "healing_trace": []
        })


        # Print the result
        st.subheader("📖 Answer:")
        st.write(result["answer"])