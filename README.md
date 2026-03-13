# LangGraph Beginner Roadmap (No-Code Build Guide)

## Current Project Status (Verified)

Last checked: 2026-03-13

Completed:

- [x] Custom graph state is wired (`MessagesState`) and used by the graph builder.
- [x] Graph has split tool execution:
  - Cached arithmetic node (`tool_node`).
  - Non-cached semantic retrieval node (`semantic_tool_node`).
- [x] Conditional routing chooses between arithmetic tools, semantic retrieval tools, or `END`.
- [x] Automatic per-thread memory saving is enabled after each assistant reply in CLI flow.
- [x] Retrieval tool is exposed as `retrieve_from_vector_store(query, thread_id, k)`.
- [x] CLI streaming is enabled via `agent.stream(..., stream_mode=["messages", "values"])`.
- [x] Retry policy is configured on tool nodes.

Partially completed:

- [~] Evaluation script exists (`demo_eval_agent.py`) with tests for arithmetic, memory recall, and thread isolation.
- [~] Runtime eval execution is currently blocked in this environment by heavy dependency import startup (`torch`/`transformers`) before tests finish.

Not completed yet:

- [ ] Human approval / interrupt-resume flow before risky actions (for example before `divide`).
- [ ] Explicit recursion/loop safety limit in runtime config.
- [ ] Structured observability/logging report for node-level traces beyond CLI streaming.
- [ ] Regression test automation (repeatable CI-style test run).

Recommended next implementation order:

1. Add interrupt/resume gate before `divide`.
2. Add recursion safety config and one failure-path test.
3. Stabilize local eval runtime environment, then run `demo_eval_agent.py` and record baseline results.

This README is your learning plan for what to build next.
You already have a working base agent loop. The next project should help you learn the most important LangGraph concepts without getting overwhelmed.

## Project to Build

Build a **Math Tutor Agent with Memory and Approval**.

The agent should:

- Answer simple math questions.
- Use tools (`add`, `multiply`, `divide`) when needed.
- Remember conversation context per user/thread.
- Ask for approval before risky actions (for this project: before `divide`).
- Stream progress so you can see graph steps while it runs.

## Why This Is a Good First LangGraph Project

This single project teaches the key LangGraph ideas:

- `State`: what data flows between nodes.
- `Nodes`: units of work (`llm_call`, `tool_node`, etc.).
- `Edges`: how control moves through the graph.
- `Conditional routing`: deciding next step based on model output.
- `Checkpointing`: memory across runs by `thread_id`.
- `Interrupt/Resume`: human-in-the-loop control.
- `Streaming`: inspect execution live.

## Build in 5 Phases

Do these in order. Do not skip ahead.

### Phase 1: Solidify Current Graph (Baseline)

Goal:

- Confirm the graph path is stable: `START -> llm_call -> (tool_node or END)`.

What to look for:

- The model returns tool calls when math is needed.
- Tool results come back as `ToolMessage`.
- Loop returns to `llm_call` after tool execution.
- The run stops when no tool call is produced.

Done when:

- You can ask 3-4 math questions and the graph behaves consistently.

### Phase 2: Add Memory with Checkpointing

Goal:

- Persist conversation state across turns using `thread_id`.

What to look for:

- Same `thread_id` keeps conversation context.
- Different `thread_id` starts fresh.
- Memory behavior is predictable and repeatable.

Done when:

- You run two threads side-by-side and verify they do not leak into each other.

### Phase 3: Add Streaming Visibility

Goal:

- See execution events in real time instead of only final output.

What to look for:

- You can observe node-by-node updates.
- You can tell when tool call is planned, executed, and returned.
- You can trace where failures happen faster than before.

Done when:

- You can explain one full run step-by-step from stream logs.

### Phase 4: Add Human Approval (Interrupt/Resume)

Goal:

- Pause graph before `divide` and ask for explicit approval.

What to look for:

- The run pauses only on protected action.
- Deny path returns safe response.
- Approve path resumes cleanly and completes.

Done when:

- You tested approve and deny behavior at least once each.

### Phase 5: Reliability and Guardrails

Goal:

- Make the graph safe and debuggable.

What to look for:

- Tool errors are handled without crashing the whole run.
- You set a recursion/loop safety limit.
- You have clear logs/traces to debug behavior.

Done when:

- You can trigger a tool failure and still get controlled behavior.

## Learning Checklist (Core Concepts)

Mark each as you truly understand it:

- [ ] Why state schema matters and what should be in state.
- [ ] Difference between node logic vs routing logic.
- [ ] How tool calls appear in messages and how tool results are returned.
- [ ] How checkpointing uses `thread_id`.
- [ ] How and when to use interrupt/resume.
- [ ] How to debug graph execution with streaming/traces.

## Common Beginner Mistakes to Watch For

- Import paths that create circular imports between modules.
- Mixing two different state definitions accidentally.
- Forgetting to pass thread config when testing memory.
- No fallback behavior when a tool raises an exception.
- Running only one happy-path test and assuming architecture is done.

## How to Evaluate If Your Architecture Is Improving

Ask yourself after each phase:

- Is this change isolated to one concern (memory, streaming, approval, etc.)?
- Can I explain the graph flow in one minute?
- If this breaks, can I quickly locate where and why?
- Can I run the same input twice and get predictable behavior?

If most answers are "yes", your architecture is getting stronger.

## Suggested Weekly Practice (Fast Learning)

- Day 1: Baseline graph + routing confidence.
- Day 2: Checkpoint memory + thread tests.
- Day 3: Streaming and debugging habits.
- Day 4: Interrupt/resume approval flow.
- Day 5: Reliability tests and cleanup.

## Next Step for You Right Now

Start with **Phase 2 (Checkpointing)** on your current project and keep everything else unchanged.

Once that works, move to **Phase 3 (Streaming)**.

That order gives the best learning gain with minimal complexity.
