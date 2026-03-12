from typing import Literal
from langgraph.graph import MessagesState, END


def _message_text_length(message: object) -> int:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return len(content)
    return len(str(content))

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

def truncate_messages(
    messages: list,
    max_chars: int = 6000,
    max_messages: int = 14,
) -> list:
    """Return a bounded recent message window for model input."""

    if not messages:
        return []

    bounded = messages[-max_messages:]
    output = []
    total_chars = 0

    for msg in reversed(bounded):
        msg_len = _message_text_length(msg)
        if output and (total_chars + msg_len) > max_chars:
            break
        output.append(msg)
        total_chars += msg_len

    return list(reversed(output))
    