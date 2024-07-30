from langchain_community.chat_models import ChatOpenAI
from langchain_groq import ChatGroq

class ChatModelSelector:
    def __init__(self, model_type, **kwargs):
        if model_type == "openai":
            self.chat_model = ChatOpenAI(**kwargs)
        elif model_type == "groq":
            self.chat_model = ChatGroq(**kwargs)
        else:
            raise ValueError("Invalid model_type. Choose 'openai' or 'groq'.")
    
    def get_chat_model(self):
        return self.chat_model

# Example usage for ChatOpenAI:
# openai_api_key = "YOUR_API_KEY"
# llm_openai = ChatModelSelector(model_type='openai', model_name='gpt-3.5-turbo', openai_api_key=openai_api_key).get_chat_model()

# Example usage for ChatGroq:
# groq_api_key = "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"
# llm_groq = ChatModelSelector(model_type='groq', temperature=0.5, model_name="mixtral-8x7b-32768", groq_api_key=groq_api_key).get_chat_model()
