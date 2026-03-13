'''
This will be the main entry point for our agent. 
It will define the graph structure, the nodes, and how they connect together.
We'll also set up the retry and cache policies for the tool node here, 
so that we can keep the logic in the tool node itself simple and focused on just calling the tool.
'''
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy, RetryPolicy
from langgraph.cache.sqlite import SqliteCache

from ..memory.saver import memory
from ..node.llm import llm_call
from ..node.tool import approval_node, semantic_tool_node, tool_node
from ..state.messages import MessagesState
from ..utils import after_approval, should_continue


def build_agent():
    agent_builder = StateGraph(MessagesState)

    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node, retry_policy=RetryPolicy(
        max_attempts=4,
        retry_on=(Exception, ValueError),
        # initial_interval=1.0, # Defaults
        # backoff_factor=2.0,
    ), cache_policy=CachePolicy(
        ttl=3600 * 24,
    ))
    agent_builder.add_node("approval_node", approval_node)
    agent_builder.add_node(
        "semantic_tool_node",
        semantic_tool_node,
        retry_policy=RetryPolicy(
            max_attempts=4,
            retry_on=(Exception, ValueError),
        ),
    )

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["approval_node", "tool_node", "semantic_tool_node", END],
    )
    agent_builder.add_conditional_edges(
        "approval_node",
        after_approval,
        ["tool_node", "llm_call"],
    )
    agent_builder.add_edge("tool_node", "llm_call")
    agent_builder.add_edge("semantic_tool_node", "llm_call")

    return agent_builder.compile(checkpointer=memory, cache=SqliteCache(path="nodes_cache.db"))


if __name__ == "__main__":
    agent = build_agent()
    # You can add code here to interact with the agent, such as a CLI loop or a web server endpoint.