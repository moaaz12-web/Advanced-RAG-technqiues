import os
import json
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

class DocumentProcessor:
    def __init__(self, embeddings):
        self.embeddings = embeddings
    
    def load_doc_and_chunkify(self, file_path):
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=80)
            chunks = text_splitter.split_documents(docs)
            return chunks
        except Exception as e:
            print(f"Error loading and chunkifying document: {e}")
            return []
    
    def load_doc_and_semantically_chunkify(self, file_path):
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            semantic_chunker = SemanticChunker(self.embeddings, breakpoint_threshold_type="percentile")
            semantic_chunks = semantic_chunker.create_documents([d.page_content for d in docs])
            return semantic_chunks
        except Exception as e:
            print(f"Error loading and semantically chunkifying document: {e}")
            return []

    def save_documents_to_json(self, documents_list, filename):
        try:
            directory = "stored_Chunks"
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            serialized_documents = [doc.__dict__ for doc in documents_list]
            with open(os.path.join(directory, filename), 'w') as file:
                json.dump(serialized_documents, file, indent=4)
            print(f"Documents saved to {os.path.join(directory, filename)}")
        except Exception as e:
            print(f"Error saving documents to JSON file: {e}")
    
    def load_documents_from_json_file(self, filename):
        try:
            documents_list = []
            with open(filename, 'r') as file:
                serialized_documents = json.load(file)
                for serialized_document in serialized_documents:
                    page_content = serialized_document.get('page_content', None)
                    if page_content is not None:
                        document = Document(page_content=page_content)
                        documents_list.append(document)
            return documents_list
        except Exception as e:
            print(f"Error loading documents from JSON file: {e}")
            return []



# processor = DocumentProcessor(embeddings)

# # Load and chunkify documents
# chunks = processor.load_doc_and_chunkify("/content/PWS Print.pdf")
# semantic_chunks = processor.load_doc_and_semantically_chunkify("/content/PWS Print.pdf")

# # Save semantic chunks to JSON file
# processor.save_documents_to_json(semantic_chunks, 'documents.json')

# # Load documents from JSON file
# loaded_documents = processor.load_documents_from_json_file(os.path.join("stored_Chunks", 'documents.json'))

# # Print loaded Document objects
# for doc in loaded_documents:
#     print(doc.page_content)
