from langchain.tools import tool
import tenacity

# Define tools
@tool
# We can add retry and caching policies directly to the tool node in the graph, but for demonstration, we'll use tenacity here for the multiply function.
@tenacity.retry(
    stop=tenacity.stop_after_attempt(4),
    retry=tenacity.retry_if_exception_type((Exception, ValueError)),
)
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """

    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    if(b == 0):
        raise ValueError("Cannot divide by zero")
    return a / b


tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}