from typing import Literal
from langgraph.graph import END

from .state.messages import MessagesState
from .tool.calcul import tools_by_name as calcul_tools_by_name
from .tool.context import tools_by_name as context_tools_by_name


def should_continue(state: MessagesState) -> Literal["tool_node", "semantic_tool_node", END]:
    """Route tool calls to arithmetic or semantic tool nodes, otherwise end."""

    messages = state["messages"]
    last_message = messages[-1]

    if not last_message.tool_calls:
        return END

    tool_names = [str(tool_call.get("name", "")) for tool_call in last_message.tool_calls]
    arithmetic_names = set(calcul_tools_by_name.keys())
    semantic_names = set(context_tools_by_name.keys())

    # If every tool call is arithmetic, route to cached arithmetic node.
    if tool_names and all(name in arithmetic_names for name in tool_names):
        return "tool_node"

    # If any semantic retrieval/store tool is present, skip node-level cache.
    if any(name in semantic_names for name in tool_names):
        return "semantic_tool_node"

    # Default to arithmetic node to surface unsupported tools consistently.
    return "tool_node"
