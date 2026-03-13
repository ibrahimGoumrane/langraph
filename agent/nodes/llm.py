from langchain.messages import HumanMessage, SystemMessage , trim_messages
from langchain_core.runnables import RunnableConfig

from ..llm.model import model
from ..memory.saver import build_retrieval_context


def _latest_user_text(state: dict) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
        if getattr(message, "type", "") == "human":
            return str(getattr(message, "content", ""))
    return ""

def llm_call(state: dict, config: RunnableConfig | None = None):
    """LLM call with retrieval-augmented context."""

    user_query = _latest_user_text(state)
    thread_id = config["configurable"]["thread_id"]

    retrieved_context = build_retrieval_context(thread_id=thread_id, query=user_query, k=3)

    system_content = (
        "You are a helpful assistant. Use available tools for arithmetic when needed. "
        "Use retrieved context when it is relevant to the user question. "
        "If context is missing for factual claims, state uncertainty clearly."
    )
    if retrieved_context:
        system_content += f"\n\nRetrieved context:\n{retrieved_context}"

    # Keep RedisSaver full history in storage, but only send a bounded window to the LLM.
    prompt_messages = trim_messages(
        state.get("messages", []),
        max_tokens=6000,
        strategy="last",
    )
    
    return {
        "messages": [
            model.invoke(
                [
                    SystemMessage(
                        content=system_content
                    )
                ]
                + prompt_messages
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }