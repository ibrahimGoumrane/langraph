from langchain.messages import AnyMessage
from typing_extensions import Annotated, NotRequired, TypedDict
import operator


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    node_calls: NotRequired[int]
    approval_decision: NotRequired[str]