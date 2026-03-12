import io

from langchain.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from PIL import Image as PILImage

from agent import (
    clear_cache,
    llm_call,
    memory,
    store_conversation_turn,
    should_continue,
    tool_node,
)


def build_agent():
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

    return agent_builder.compile(checkpointer=memory)

def render_graph(agent):
    graph_bytes = agent.get_graph(xray=True).draw_mermaid_png()

    with open("graph.png", "wb") as file_obj:
        file_obj.write(graph_bytes)

    print("Graph saved as graph.png")

    image = PILImage.open(io.BytesIO(graph_bytes))
    image.show()

def main(messages :  list[str] ,thread_id: str):
    agent = build_agent()
    # render_graph(agent)

    for message in messages:
        config = {"configurable": {"thread_id": thread_id}}
        state = agent.invoke(MessagesState(messages=[HumanMessage(content=message)]), config=config)

        assistant_reply = str(state["messages"][-1].content)
        store_conversation_turn(
            thread_id=thread_id,
            user_input=message,
            assistant_output=assistant_reply,
        )

        state["messages"][-1].pretty_print()

if __name__ == "__main__":
    # main([
    #     "Calculate the sum of 5 and 7" , 
    #       "What is the capital of France?" , 
    #       "Add an actual 2 to the actual previous result",
    #       "Then subdivise the last result by 3"
    #     ] , "thread-1")
    # clear_cache()  # Clear cache before running
    main([
        "What the country i asked u before about its capital?" ,
    ] , "thread-1")

