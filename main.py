from langchain.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from agent import (
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


def make_thread_id(user_name: str) -> str:
    normalized = "_".join(user_name.strip().lower().split())
    return f"user::{normalized or 'guest'}"


def print_help() -> None:
    print("\nCommands:")
    print("  /help              Show commands")
    print("  /name              Change active user name")
    print("  /whoami            Show active user and thread")
    print("  /exit              Quit CLI")


def stream_agent_reply(agent, user_input: str, config: dict) -> dict:
    final_state: dict | None = None

    print("Assistant> ", end="", flush=True)
    for mode, payload in agent.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode=["messages", "values"],
    ):
        if mode == "messages":
            message, metadata = payload
            if metadata.get("langgraph_node") != "llm_call":
                continue
            chunk_text = getattr(message, "content", "")
            if isinstance(chunk_text, str) and chunk_text:
                print(chunk_text, end="", flush=True)
        elif mode == "values" and isinstance(payload, dict):
            final_state = payload

    print()

    if final_state is None:
        raise RuntimeError("Graph stream did not return a final state.")

    return final_state


def run_cli() -> None:
    agent = build_agent()

    current_user = input("Enter your user name: ").strip() or "guest"
    current_thread = make_thread_id(current_user)

    print("\nInteractive Agent CLI started.")
    print("Type /help to see commands.")
    print(f"Active user: {current_user}")
    print(f"Active thread: {current_thread}")

    while True:
        user_input = input(f"\n{current_user}> ").strip()
        if not user_input:
            continue

        if user_input in {"/exit", "exit", "quit"}:
            print("Exiting CLI.")
            break

        if user_input == "/help":
            print_help()
            continue

        if user_input == "/whoami":
            print(f"Active user: {current_user}")
            print(f"Active thread: {current_thread}")
            continue

        if user_input == "/name":
            new_name = input("New user name: ").strip()
            if not new_name:
                print("Name was empty. Keeping current user.")
                continue
            current_user = new_name
            current_thread = make_thread_id(current_user)
            print(f"Switched user to: {current_user}")
            print(f"Switched thread to: {current_thread}")
            continue

        config = {"configurable": {"thread_id": current_thread}}
        try:
            state = stream_agent_reply(agent, user_input, config)
            assistant_reply = str(state["messages"][-1].content)
            store_conversation_turn(
                thread_id=current_thread,
                user_input=user_input,
                assistant_output=assistant_reply,
            )
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    run_cli()

