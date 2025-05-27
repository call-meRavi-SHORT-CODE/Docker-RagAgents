from langchain.chat_models import init_chat_model



def get_openai_llm():
    # Uses gpt-4o-mini via the new init_chat_model interface
    return init_chat_model("gpt-4o-mini", model_provider="openai")
