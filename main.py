import os
os.environ["NEO4J_USERNAME"] = 'neo4j'
os.environ["NEO4J_PASSWORD"] = 'ZntmjOVGRvINTa24EnbWf6L3DtkXz3OStbZgc8OUOvU'
os.environ['NEO4J_URL'] = "neo4j+s://242aed3d.databases.neo4j.io"


from llms import ChatModelSelector
groq_api_key = "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"
llm = ChatModelSelector(model_type='groq', temperature=0.4, model_name="mixtral-8x7b-32768", groq_api_key=groq_api_key).get_chat_model()

from embeddings import EmbeddingsSelector
option = "huggingface_bge"
selector = EmbeddingsSelector(option)
embeddings = selector.get_embeddings()



from langchain_community.vectorstores.neo4j_vector import Neo4jVector


book1_querying=False
if book1_querying:
        vectorstore = Neo4jVector.from_existing_index(
        url="neo4j+s://242aed3d.databases.neo4j.io",
        username="neo4j",
        password="ZntmjOVGRvINTa24EnbWf6L3DtkXz3OStbZgc8OUOvU",
        embedding=embeddings,
        index_name="PWS",
        node_label="PdfBotChunk_PWS",
    )

if not book1_querying:
    vectorstore = Neo4jVector.from_existing_index(
        url="neo4j+s://242aed3d.databases.neo4j.io",
        username="neo4j",
        password="ZntmjOVGRvINTa24EnbWf6L3DtkXz3OStbZgc8OUOvU",
        embedding=embeddings,
        index_name="PRS",
        node_label="PdfBotChunk_PRS",
    )


base_retreiver = vectorstore.as_retriever()



from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere

os.environ["COHERE_API_KEY"] = "sC9sHbmVCoQr9xDHfruo8eXaEJwRodGxfga1TAj0"

compressor = CohereRerank()
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retreiver
)


from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=base_retreiver
)

# Accept user questions/query
while True:
    query = input("Enter your question/query (type 'quit' to exit): ")
    if query.lower() == "quit":
        break
    if not query.strip():  # Check if query is empty or only contains whitespace
        print("Please enter a valid query.")
        continue
    res = qa.invoke(query)
    print(res['result'])

