from langchain.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from PIL import Image as PILImage

from agent import clear_cache, llm_call, memory, should_continue, tool_node


def build_agent(use_memory: bool):
    agent_builder = StateGraph(MessagesState)

    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END],
    )
    agent_builder.add_edge("tool_node", "llm_call")

    if use_memory:
        return agent_builder.compile(checkpointer=memory)

    return agent_builder.compile()


def render_graph(agent):
    graph_bytes = agent.get_graph(xray=True).draw_mermaid_png()

    with open("graph.png", "wb") as file_obj:
        file_obj.write(graph_bytes)

    print("Graph saved as graph.png")

    # image = PILImage.open(io.BytesIO(graph_bytes))
    # image.show()


def print_state(title: str, state: dict):
    print(f"\n{title}")
    for message in state["messages"]:
        message.pretty_print()


def reset_tool_cache():
    try:
        clear_cache()
        print("Tool cache cleared for this demo.")
    except Exception as exc:
        print(f"Tool cache was not cleared: {exc}")


def run_demo(use_memory: bool, thread_id: str):
    label = "WITH MEMORYSAVER" if use_memory else "WITHOUT CHECKPOINTER"
    print("\n" + "=" * 80)
    print(label)

    if use_memory:
        print(
            "The second invoke reuses the same thread_id, so the graph can see the previous turn while this Python process is still running."
        )
    else:
        print(
            "The second invoke also reuses the same thread_id value, but nothing is restored because there is no checkpointer."
        )

    reset_tool_cache()

    agent = build_agent(use_memory=use_memory)
    config = {"configurable": {"thread_id": thread_id}}

    first_state = agent.invoke(
        {"messages": [HumanMessage(content="Add 3 and 4.")]},
        config=config,
    )
    print_state("First invoke", first_state)

    second_state = agent.invoke(
        {"messages": [HumanMessage(content="Now multiply that result by 2.")]},
        config=config,
    )
    print_state("Second invoke", second_state)


memory_agent = build_agent(use_memory=True)
render_graph(memory_agent)

run_demo(use_memory=True, thread_id="memory_demo_thread")
run_demo(use_memory=False, thread_id="no_memory_demo_thread")

print("\n" + "=" * 80)
print("WHAT THIS MEANS")
print("MemorySaver keeps checkpoints in RAM only.")
print("If this Python process exits, that memory is gone.")
print("A graph compiled without a checkpointer keeps nothing between invokes.")