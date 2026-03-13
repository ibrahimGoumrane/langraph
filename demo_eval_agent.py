from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.messages import HumanMessage

from agent import build_agent, ensure_store_ready, store_conversation_turn


@dataclass
class EvalResult:
    name: str
    passed: bool
    details: str


def _assistant_text(state: dict[str, Any]) -> str:
    return str(state["messages"][-1].content)


def _has_tool_call(state: dict[str, Any], tool_name: str) -> bool:
    for message in state.get("messages", []):
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            continue
        for call in tool_calls:
            if str(call.get("name", "")) == tool_name:
                return True
    return False


def _invoke_and_store(agent, thread_id: str, user_input: str) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
    store_conversation_turn(
        thread_id=thread_id,
        user_input=user_input,
        assistant_output=_assistant_text(state),
    )
    return state


def run_evals() -> list[EvalResult]:
    ensure_store_ready()
    agent = build_agent()

    results: list[EvalResult] = []

    # 1) Basic arithmetic behavior
    math_thread = "eval::math"
    math_state = _invoke_and_store(agent, math_thread, "Add 13 and 29.")
    math_answer = _assistant_text(math_state)
    math_ok = "42" in math_answer
    results.append(
        EvalResult(
            name="Arithmetic correctness",
            passed=math_ok,
            details=f"answer={math_answer!r}",
        )
    )

    # 2) Memory recall in same thread
    memory_thread = "eval::memory"
    _invoke_and_store(
        agent,
        memory_thread,
        "My friend's name is Nizar and he is a student. Please remember this.",
    )
    recall_state = _invoke_and_store(
        agent,
        memory_thread,
        "What is my friend's name and what does he do?",
    )
    recall_answer = _assistant_text(recall_state)
    recall_used_tool = _has_tool_call(recall_state, "retrieve_from_vector_store")
    recall_ok = ("nizar" in recall_answer.lower()) and ("student" in recall_answer.lower())
    results.append(
        EvalResult(
            name="Same-thread memory recall",
            passed=recall_ok,
            details=(
                f"answer={recall_answer!r}; "
                f"retrieve_tool_called={recall_used_tool}"
            ),
        )
    )

    # 3) Thread isolation (fresh thread should not know prior details)
    isolated_thread = "eval::isolation"
    isolation_state = _invoke_and_store(
        agent,
        isolated_thread,
        "What is my friend's name and what does he do?",
    )
    isolation_answer = _assistant_text(isolation_state)
    isolation_bad = ("nizar" in isolation_answer.lower()) and ("student" in isolation_answer.lower())
    results.append(
        EvalResult(
            name="Thread isolation",
            passed=not isolation_bad,
            details=f"answer={isolation_answer!r}",
        )
    )

    return results


def main() -> None:
    results = run_evals()

    print("\n=== Agent Evaluation Report ===")
    passed_count = 0
    for idx, result in enumerate(results, start=1):
        status = "PASS" if result.passed else "FAIL"
        if result.passed:
            passed_count += 1
        print(f"{idx}. [{status}] {result.name}")
        print(f"   {result.details}")

    print(f"\nSummary: {passed_count}/{len(results)} passed")


if __name__ == "__main__":
    main()
