from langchain.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from ..state.messages import MessagesState
from ..tool.calcul import tools_by_name as calcul_tools_by_name
from ..tool.context import tools_by_name as context_tools_by_name


def tool_node(state: MessagesState):
    """Execute arithmetic tools only. This node can be safely cached."""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        tool = calcul_tools_by_name[tool_name]
        observation = tool.invoke(tool_args)
        print(f"Called arithmetic tool {tool_name}({tool_args}) -> {observation}")

        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

    return {"messages": result}


def semantic_tool_node(state: MessagesState, config: RunnableConfig | None = None):
    """Execute retrieval tools without cache, using thread-aware context."""

    thread_id = "default"
    if config:
        thread_id = str(config.get("configurable", {}).get("thread_id", "default"))

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_args["thread_id"] = thread_id

        tool = context_tools_by_name.get(tool_name) or calcul_tools_by_name.get(tool_name)
        if tool is None:
            raise ValueError(f"Unknown tool call: {tool_name}")

        observation = tool.invoke(tool_args)
        print(f"Called semantic tool {tool_name}({tool_args}) -> {observation}")
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

    return {"messages": result}