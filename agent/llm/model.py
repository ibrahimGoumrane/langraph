from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from ..tool.calcul import tools as calcul_tools
from ..tool.context import retrieve_from_vector_store
load_dotenv()

tmp = init_chat_model(
    "gpt-5-nano",
    model_provider="openai",
)

model = tmp.bind_tools(calcul_tools + [retrieve_from_vector_store])