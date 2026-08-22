
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from settings import LangFuseSettings

langfuse_settings = LangFuseSettings()

# Initialize Langfuse client
langfuse = get_client()

langfuse_handler = CallbackHandler()
