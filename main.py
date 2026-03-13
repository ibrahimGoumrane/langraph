from langchain.messages import HumanMessage
from langgraph.types import Command

from agent.impl import build_agent, ensure_store_ready, store_conversation_turn


def make_thread_id(user_name: str) -> str:
    normalized = "_".join(user_name.strip().lower().split())
    return f"user::{normalized or 'guest'}"


def print_help() -> None:
    print("\nCommands:")
    print("  /help              Show commands")
    print("  /name              Change active user name")
    print("  /whoami            Show active user and thread")
    print("  /exit              Quit CLI")


def _stream_once(agent, payload, config: dict, *, show_prefix: bool) -> dict | None:
    final_state: dict | None = None

    if show_prefix:
        print("Assistant> ", end="", flush=True)

    for mode, event_payload in agent.stream(
        payload,
        config=config,
        stream_mode=["messages", "values"],
    ):
        if mode == "messages":
            message, metadata = event_payload
            if metadata.get("langgraph_node") != "llm_call":
                continue
            chunk_text = getattr(message, "content", "")
            if isinstance(chunk_text, str) and chunk_text:
                print(chunk_text, end="", flush=True)
        elif mode == "values" and isinstance(event_payload, dict):
            final_state = event_payload

    print()
    return final_state


def _get_interrupts(agent, config: dict):
    snapshot = agent.get_state(config)
    return tuple(getattr(snapshot, "interrupts", ()) or ())


def _get_pending_tool_calls(agent, config: dict) -> list[dict]:
    snapshot = agent.get_state(config)
    values = getattr(snapshot, "values", {}) or {}
    messages = values.get("messages", [])
    if not messages:
        return []
    return list(getattr(messages[-1], "tool_calls", []) or [])


def stream_agent_reply(agent, user_input: str, config: dict) -> dict:
    final_state = _stream_once(
        agent,
        {"messages": [HumanMessage(content=user_input)]},
        config,
        show_prefix=True,
    )

    interrupts = _get_interrupts(agent, config)
    while interrupts:
        print("\nRISKY TOOL DETECTED - Human approval required.")
        pending_tool_calls = _get_pending_tool_calls(agent, config)
        if pending_tool_calls:
            print("Pending tool calls:", pending_tool_calls)

        approve = input("Approve and continue? (y/n): ").strip().lower() == "y"
        print("Resuming...")
        resumed_state = _stream_once(
            agent,
            Command(resume=approve),
            config,
            show_prefix=True,
        )
        if resumed_state is not None:
            final_state = resumed_state
        interrupts = _get_interrupts(agent, config)

    if final_state is None:
        raise RuntimeError("Graph stream did not return a final state.")

    return final_state


def run_cli() -> None:
    agent = build_agent()
    ensure_store_ready()

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

