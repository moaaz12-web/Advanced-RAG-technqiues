from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

class EmbeddingsSelector:
    def __init__(self, option):
        
        if option == "fastembed":
            self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        
        elif option == "huggingface":
            self.embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key="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", model_name="BAAI/bge-base-en-v1.5"
            )
        
        elif option == "huggingface_bge":
            model_name = "BAAI/bge-base-en-v1.5"
            model_kwargs = {"device": "cpu"}
            encode_kwargs = {"normalize_embeddings": True}
            self.embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
            )
        else:
            raise ValueError("Invalid option. Choose 'fastembed', 'huggingface', or 'huggingface_bge'.")
    
    def get_embeddings(self):
        return self.embeddings

# # Example usage:
# option = "huggingface"
# selector = EmbeddingsSelector(option)
# embeddings = selector.get_embeddings()
