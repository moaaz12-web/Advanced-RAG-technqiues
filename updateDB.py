import os
from llms import ChatModelSelector
from embeddings import EmbeddingsSelector
from chunkify import DocumentProcessor
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain.prompts import ChatPromptTemplate

# Define the prompt
prompt_str = """You are a helpful assistant that answers to users questions based on provided context
in detail. You should mention about the technique used as well, if present in the context. Answer truthfully based on below context. If you don't know the answer, say you don't know.

Contexte : {contexte}

Question : {question}

Réponse : """
prompt = ChatPromptTemplate.from_template(prompt_str)

print("\n\n")
print("Updating vector database...")

# Set the API key for Groq
groq_api_key = "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"
llm = ChatModelSelector(model_type='groq', temperature=0.5, model_name="mixtral-8x7b-32768", groq_api_key=groq_api_key).get_chat_model()

# Initialize embeddings
option = "huggingface_bge"
selector = EmbeddingsSelector(option)
embeddings = selector.get_embeddings()

# Initialize the document processor
processor = DocumentProcessor(embeddings)

# Load documents from JSON files
chunks_PWS = processor.load_documents_from_json_file(os.path.join("stored_Chunks", 'chunks.json'))
chunks_PRS = processor.load_documents_from_json_file(os.path.join("stored_Chunks", 'chunks_PRS.json'))

# Update the vectorstore for PWS
try:
    vectorstore_PWS = Neo4jVector.from_documents(
        chunks_PWS,
        url="neo4j+s://242aed3d.databases.neo4j.io",
        username="neo4j",
        password="ZntmjOVGRvINTa24EnbWf6L3DtkXz3OStbZgc8OUOvU",
        embedding=embeddings,
        index_name="PWS",
        node_label="PdfBotChunk_PWS",
    )
    print("Vectorstore PWS updated successfully.")
    del vectorstore_PWS # Delete the vectorstore object
except Exception as e:
    print(f"Failed to update vectorstore PWS: {e}")

# Update the vectorstore for PRS
try:
    vectorstore_PRS = Neo4jVector.from_documents(
        chunks_PRS,
        url="neo4j+s://242aed3d.databases.neo4j.io",
        username="neo4j",
        password="ZntmjOVGRvINTa24EnbWf6L3DtkXz3OStbZgc8OUOvU",
        embedding=embeddings,
        index_name="PRS",
        node_label="PdfBotChunk_PRS",
    )
    print("Vectorstore PRS updated successfully.")
    del vectorstore_PRS # Delete the vectorstore object
except Exception as e:
    print(f"Failed to update vectorstore PRS: {e}")

print("Vector database update process complete.")
print("\n\n")
