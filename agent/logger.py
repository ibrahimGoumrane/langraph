from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LangGraphLogger:
    """Reusable helper to inspect LangGraph execution in plain console logs."""

    enabled: bool = True
    preview_chars: int = 140

    def _print(self, message: str) -> None:
        if self.enabled:
            print(message)

    def _shorten(self, value: Any) -> str:
        text = str(value).replace("\n", " ").strip()
        if len(text) <= self.preview_chars:
            return text
        return f"{text[: self.preview_chars - 3]}..."

    def _message_summary(self, message: Any) -> str:
        msg_type = getattr(message, "type", type(message).__name__)
        content = getattr(message, "content", "")
        summary = f"{msg_type}: {self._shorten(content)}"

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            names = [call.get("name", "unknown") for call in tool_calls]
            summary += f" | tool_calls={names}"

        return summary

    def log_state(self, state: dict[str, Any], title: str = "State Snapshot") -> None:
        """Print a compact snapshot of current graph state."""

        if not self.enabled:
            return

        messages = state.get("messages", [])
        self._print(f"\n[{title}]")
        self._print(f"messages={len(messages)}")

        if messages:
            self._print(f"last={self._message_summary(messages[-1])}")

        if "llm_calls" in state:
            self._print(f"llm_calls={state['llm_calls']}")

    def invoke_with_logs(
        self,
        graph: Any,
        inputs: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a graph once while logging each streamed state update."""

        self.log_state(inputs, title="Input")

        final_state: dict[str, Any] | None = None

        try:
            stream_iter = graph.stream(inputs, config=config, stream_mode="values")
        except TypeError:
            stream_iter = graph.stream(inputs, config=config)

        for step, chunk in enumerate(stream_iter, start=1):
            if isinstance(chunk, dict):
                self.log_state(chunk, title=f"Step {step}")
                final_state = chunk
            else:
                self._print(f"\n[Step {step}] {self._shorten(chunk)}")

        if final_state is None:
            self._print("\n[Info] Stream did not return full state, falling back to invoke().")
            final_state = graph.invoke(inputs, config=config)
            self.log_state(final_state, title="Final")

        self._print("\n[Run Complete]")
        return final_state
