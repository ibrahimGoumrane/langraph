from langchain.messages import SystemMessage, trim_messages
from ..llm.model import model
import tiktoken

_enc = tiktoken.encoding_for_model("gpt-4o")

def _count_tokens(messages) -> int:
    return sum(len(_enc.encode(str(getattr(m, "content", "")))) for m in messages)

def llm_call(state: dict):
    """LLM call with retrieval-augmented context."""

    system_content = (
        "You are a helpful assistant. Use available tools for arithmetic when needed. "
        "If the user refers to earlier turns, preferences, or prior facts, call the "
        "retrieve_from_vector_store tool before answering. "
        "If context is missing for factual claims, state uncertainty clearly."
    )
    # Keep RedisSaver full history in storage, but only send a bounded window to the LLM.
    prompt_messages = trim_messages(
        state.get("messages", []),
        max_tokens=6000,
        strategy="last",
        token_counter=_count_tokens,
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