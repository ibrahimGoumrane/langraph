# Build workflow
from langgraph.graph import END, START, MessagesState, StateGraph
from agent import LangGraphLogger, llm_call, tool_node, should_continue
from PIL import Image as PILImage
import io
from langchain.messages import HumanMessage


agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

graph_bytes = agent.get_graph(xray=True ).draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(graph_bytes)
print("Graph saved as graph.png")

image = PILImage.open(io.BytesIO(graph_bytes))
image.show()

# Invoke

messages = [HumanMessage(content="Add 3 and 4.")]
logger = LangGraphLogger()
final_state = logger.invoke_with_logs(agent, {"messages": messages})

for m in final_state["messages"]:
    m.pretty_print()