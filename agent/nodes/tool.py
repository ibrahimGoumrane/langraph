from langchain.messages import ToolMessage
from langgraph.graph import MessagesState

from ..tool.calcul import tools_by_name
from ..cache.main import cache_tool_result, store_tool_result

def tool_node(state : MessagesState) : 
    """Performs the tool call with caching"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Check if result is cached
        is_cached, cached_result = cache_tool_result(tool_name, tool_args)
        
        if is_cached:
            observation = f"{cached_result} (cached)"
            print(f"⚡ Using cached result for {tool_name}({tool_args})")
        else:
            tool = tools_by_name[tool_name]
            observation = tool.invoke(tool_args)
            # Store in cache for future use
            store_tool_result(tool_name, tool_args, observation)
            print(f"📞 Called tool {tool_name}({tool_args}) → {observation}")
        
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}
