

import streamlit as st
import os
from llms import ChatModelSelector
from embeddings import EmbeddingsSelector
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
from langchain.chains import RetrievalQA

# Default API keys
default_groq_api_key = "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"
default_cohere_api_key = "sC9sHbmVCoQr9xDHfruo8eXaEJwRodGxfga1TAj0"

# Streamlit interface setup
st.set_page_config(page_title="QA Chat Interface", page_icon=":books:", layout="wide")
st.title("📚 QA Chat Interface")

# Sidebar for settings and history
st.sidebar.header("📚 QA Chat")
st.sidebar.markdown("## Configuration")
st.markdown("-----------")


# Input fields for API keys
groq_api_key = st.sidebar.text_input("Enter Groq API Key", default_groq_api_key)
cohere_api_key = st.sidebar.text_input("Enter Cohere API Key", default_cohere_api_key)

# Checkbox to toggle book1_querying
book1_querying = st.sidebar.checkbox("Query Book 1", value=False)

# Display selected book information
if book1_querying:
    st.sidebar.success("Selected Book 1. You can start asking questions.")
else:
    st.sidebar.success("Selected Book 2. You can start asking questions.")

# Environment variables setup
os.environ["NEO4J_USERNAME"] = 'neo4j'
os.environ["NEO4J_PASSWORD"] = 'ZntmjOVGRvINTa24EnbWf6L3DtkXz3OStbZgc8OUOvU'
os.environ['NEO4J_URL'] = "neo4j+s://242aed3d.databases.neo4j.io"
os.environ["COHERE_API_KEY"] = cohere_api_key

# Initialize the model and vectorstore
llm = ChatModelSelector(model_type='groq', temperature=0.4, model_name="mixtral-8x7b-32768", groq_api_key=groq_api_key).get_chat_model()
option = "huggingface_bge"
selector = EmbeddingsSelector(option)
embeddings = selector.get_embeddings()

if book1_querying:
    vectorstore = Neo4jVector.from_existing_index(
        url=os.environ['NEO4J_URL'],
        username=os.environ['NEO4J_USERNAME'],
        password=os.environ['NEO4J_PASSWORD'],
        embedding=embeddings,
        index_name="PWS",
        node_label="PdfBotChunk_PWS",
    )
else:
    vectorstore = Neo4jVector.from_existing_index(
        url=os.environ['NEO4J_URL'],
        username=os.environ['NEO4J_USERNAME'],
        password=os.environ['NEO4J_PASSWORD'],
        embedding=embeddings,
        index_name="PRS",
        node_label="PdfBotChunk_PRS",
    )

base_retriever = vectorstore.as_retriever()
compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=base_retriever
)

# Main interface
st.write("### Ask your questions below:")
query = st.text_input("Enter your question/query")

# Display response
if st.button("Submit"):
    if not query.strip():  # Check if query is empty or only contains whitespace
        st.error("Please enter a valid query.")
    else:
        res = qa.invoke(query)
        st.write("#### Answer:")
        st.write(res['result'])
        # Add the question and its answer to the history
        st.sidebar.markdown("## History")
        st.sidebar.markdown(f"**Question:** {query}")
        st.sidebar.markdown(f"**Answer:** {res['result']}")

# Footer with additional information or branding
st.markdown("---")
st.markdown("Developed by [Your Name](https://your-website.com). Powered by Neo4j, Groq, and Cohere.")
