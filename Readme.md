# Advanced RAG techniques for document question answering


## FEATURES
1. Semnatic chunking rather than typical chunking strategies of LangChain.
2. Different embedding models, you can use FastEmbeddings, HuggingFace embeddings, or HuggingFace Inference API embeddings.
3. Differet LLMS, either from OpenAI or from Groq (open source HuggingFace models)
4. Neo4j as the vector database for storing and retreiving embeddings
5. Applied Contextual Compression that enhances  the retrieval of relevant documents by integrating a base retriever with a document compressor. The document compressor further filters out irrelevant stuff from the retrieved documents and only keeps relevant information related to the user query.
6. Implemented Reranking which enhances the retrieval by reordering the list of retrieved documents based on their relevance to the user question. After an initial set of documents is fetched by a base retriever, a reranking model evaluates and scores these documents to prioritize the most relevant ones. 
7. Application frontend is based on Streamlit


## SETUP INSTRUCTIONS

1. First setup virtual environment:
python -m venv myenv

2. Then activate it:
myenv/Scripts/activate

3. Then install the requirements:
pip install -r requirements.txt

4. Then run the app via TERMINAL:
python main.py

5. In order to run the interface on WEB BROWSER:
streamlit run interface.py