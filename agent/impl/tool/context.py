from langchain.tools import tool

from ..memory.store import build_retrieval_context


@tool
def retrieve_from_vector_store(query: str, thread_id: str, k: int = 3) -> str:
    """Tool to retrieve relevant conversation chunks from the vector store. Returns a formatted string of retrieved context."""
    
    if not query.strip():
        raise ValueError("Query must be a non-empty string.")

    if not thread_id.strip():
        raise ValueError("Thread ID is missing in the context. Cannot retrieve conversation chunks.")
    
    retrieved_context = build_retrieval_context(thread_id=thread_id, query=query, k=k)

    if not retrieved_context:
        return "No relevant conversation chunks found."
    
    return f"Retrieved context:\n{retrieved_context}"


tools = [retrieve_from_vector_store]
tools_by_name = {tool.name: tool for tool in tools}