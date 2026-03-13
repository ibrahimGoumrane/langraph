from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated, NotRequired
import operator


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: NotRequired[int]