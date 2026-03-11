from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from ..tool.calcul import tools
load_dotenv()


tmp = init_chat_model(
    "o4-mini",
    model_provider="openai",
)



model = tmp.bind_tools(tools)