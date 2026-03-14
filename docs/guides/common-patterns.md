# Common Workflow Patterns

Every JSON example below is valid input to `apxm validate`.

## 1. Pipeline

Sequential chain via `Data` edges. Use `{{node_N}}` to reference upstream output. Built-in: `apxm template show pipeline --json`

```json
{
  "name": "pipeline",
  "nodes": [
    {"id": 1, "name": "draft",  "op": "ASK",  "attributes": {"template_str": "Write a short blog post about Rust"}},
    {"id": 2, "name": "review", "op": "THINK", "attributes": {"template_str": "Review this draft for clarity and accuracy: {{node_1}}"}},
    {"id": 3, "name": "refine", "op": "ASK",  "attributes": {"template_str": "Improve the draft based on this review feedback: {{node_2}}"}}
  ],
  "edges": [{"from": 1, "to": 2, "dependency": "Data"}, {"from": 2, "to": 3, "dependency": "Data"}],
  "parameters": [], "metadata": {}
}
```

## 2. Fan-Out / Parallel

Independent tasks run concurrently; `WAIT_ALL` collects all results. Built-in: `apxm template show fan-out --json`
```json
{
  "name": "fan-out",
  "nodes": [
    {"id": 1, "name": "research-a", "op": "ASK", "attributes": {"template_str": "Research topic A"}},
    {"id": 2, "name": "research-b", "op": "ASK", "attributes": {"template_str": "Research topic B"}},
    {"id": 3, "name": "research-c", "op": "ASK", "attributes": {"template_str": "Research topic C"}},
    {"id": 4, "name": "merge", "op": "WAIT_ALL", "attributes": {"tokens": ["{{node_1}}", "{{node_2}}", "{{node_3}}"]}}
  ],
  "edges": [{"from": 1, "to": 4, "dependency": "Data"}, {"from": 2, "to": 4, "dependency": "Data"}, {"from": 3, "to": 4, "dependency": "Data"}],
  "parameters": [], "metadata": {}
}
```

## 3. Map-Reduce

Parallel workers sync via `WAIT_ALL`, then a final step synthesizes. Built-in: `apxm template show map-reduce --json`
```json
{
  "name": "map-reduce",
  "nodes": [
    {"id": 1, "name": "analyze-1",  "op": "ASK",     "attributes": {"template_str": "Analyze aspect 1 of the problem"}},
    {"id": 2, "name": "analyze-2",  "op": "ASK",     "attributes": {"template_str": "Analyze aspect 2 of the problem"}},
    {"id": 3, "name": "analyze-3",  "op": "ASK",     "attributes": {"template_str": "Analyze aspect 3 of the problem"}},
    {"id": 4, "name": "sync",       "op": "WAIT_ALL", "attributes": {"tokens": ["{{node_1}}", "{{node_2}}", "{{node_3}}"]}},
    {"id": 5, "name": "synthesize", "op": "ASK",     "attributes": {"template_str": "Synthesize all analyses into a final report: {{node_4}}"}}
  ],
  "edges": [{"from": 1, "to": 4, "dependency": "Data"}, {"from": 2, "to": 4, "dependency": "Data"}, {"from": 3, "to": 4, "dependency": "Data"}, {"from": 4, "to": 5, "dependency": "Data"}],
  "parameters": [], "metadata": {}
}
```

## 4. Verify (Generate + Fact-Check)

`ASK` produces a claim, `VERIFY` checks it against evidence. Built-in: `apxm template show verify --json`
```json
{
  "name": "verify",
  "nodes": [
    {"id": 1, "name": "generate", "op": "ASK",    "attributes": {"template_str": "State 3 facts about the solar system"}},
    {"id": 2, "name": "check",    "op": "VERIFY", "attributes": {"claim": "{{node_1}}", "evidence": "Common astronomical knowledge"}}
  ],
  "edges": [{"from": 1, "to": 2, "dependency": "Data"}],
  "parameters": [], "metadata": {}
}
```

## 5. Conditional Branching

`BRANCH_ON_VALUE` routes to exactly one path via `Control` edges. Only the taken branch executes. Built-in: `apxm template show conditional --json`
```json
{
  "name": "conditional",
  "nodes": [
    {"id": 1, "name": "classify",       "op": "ASK",             "attributes": {"template_str": "Is this a technical question? Answer only 'yes' or 'no'"}},
    {"id": 2, "name": "branch",         "op": "BRANCH_ON_VALUE", "attributes": {"token": "{{node_1}}", "value": "yes", "label_true": "3", "label_false": "4"}},
    {"id": 3, "name": "technical-path", "op": "ASK",             "attributes": {"template_str": "Give a detailed technical answer"}},
    {"id": 4, "name": "general-path",   "op": "ASK",             "attributes": {"template_str": "Give a friendly general answer"}}
  ],
  "edges": [{"from": 1, "to": 2, "dependency": "Data"}, {"from": 2, "to": 3, "dependency": "Control"}, {"from": 2, "to": 4, "dependency": "Control"}],
  "parameters": [], "metadata": {}
}
```

## 6. Error Handling

`TRY_CATCH` wraps a try subgraph with a catch subgraph. On failure, the `ERR` node fires with a recovery strategy.
```json
{
  "name": "error-handling",
  "nodes": [
    {"id": 1, "name": "try-catch",  "op": "TRY_CATCH", "attributes": {"try_subgraph": "2", "catch_subgraph": "3"}},
    {"id": 2, "name": "risky-call", "op": "ASK",       "attributes": {"template_str": "Translate this document to French"}},
    {"id": 3, "name": "fallback",   "op": "ERR",       "attributes": {"message": "Translation failed", "recovery_template": "Return the original text with a note that translation was unavailable"}}
  ],
  "edges": [{"from": 1, "to": 2, "dependency": "Control"}, {"from": 1, "to": 3, "dependency": "Control"}],
  "parameters": [], "metadata": {}
}
```

## 7. Memory Operations

`QMEM` reads from memory (STM/LTM/Episodic). `UMEM` writes a key-value pair. `FENCE` ensures prior writes are visible before subsequent reads.
```json
{
  "name": "memory-pipeline",
  "nodes": [
    {"id": 1, "name": "recall",  "op": "QMEM",  "attributes": {"query": "user_preferences", "memory_tier": "ltm"}},
    {"id": 2, "name": "respond", "op": "ASK",   "attributes": {"template_str": "Given these preferences: {{node_1}}, recommend a book"}},
    {"id": 3, "name": "store",   "op": "UMEM",  "attributes": {"key": "last_recommendation", "value": "{{node_2}}", "memory_tier": "stm"}},
    {"id": 4, "name": "barrier", "op": "FENCE", "attributes": {}},
    {"id": 5, "name": "confirm", "op": "QMEM",  "attributes": {"query": "last_recommendation", "memory_tier": "stm"}}
  ],
  "edges": [{"from": 1, "to": 2, "dependency": "Data"}, {"from": 2, "to": 3, "dependency": "Data"}, {"from": 3, "to": 4, "dependency": "Data"}, {"from": 4, "to": 5, "dependency": "Data"}],
  "parameters": [], "metadata": {}
}
```

## 8. Multi-Agent Coordination

`GUARD` enforces preconditions (conditions: `> 0.8`, `!= null`, `not_empty`; on_fail: `halt` or `skip`). `CLAIM` atomically takes a task from a shared queue. `UPDATE_GOAL` modifies the agent's goals at runtime (actions: `set`, `remove`, `clear`).
```json
{
  "name": "coordinated-worker",
  "nodes": [
    {"id": 1, "name": "check-ready", "op": "GUARD",       "attributes": {"condition": "not_null", "on_fail": "halt", "error_message": "No input provided"}},
    {"id": 2, "name": "get-task",     "op": "CLAIM",       "attributes": {"queue": "review_tasks", "lease_ms": 30000}},
    {"id": 3, "name": "do-work",      "op": "ASK",         "attributes": {"template_str": "Review the following submission: {{node_2}}"}},
    {"id": 4, "name": "set-goal",     "op": "UPDATE_GOAL", "attributes": {"goal_id": "review_complete", "action": "set", "priority": 1}}
  ],
  "edges": [{"from": 1, "to": 2, "dependency": "Data"}, {"from": 2, "to": 3, "dependency": "Data"}, {"from": 3, "to": 4, "dependency": "Data"}],
  "parameters": [], "metadata": {}
}
```

## Quick Reference

| Pattern | Key Ops | Edge Type | Use Case |
|---------|---------|-----------|----------|
| Pipeline | ASK, THINK | Data | Sequential multi-step processing |
| Fan-out | ASK, WAIT_ALL | Data | Independent parallel tasks |
| Map-reduce | ASK, WAIT_ALL | Data | Parallel analysis + synthesis |
| Verify | ASK, VERIFY | Data | Fact-checking generated content |
| Conditional | BRANCH_ON_VALUE | Data + Control | Routing based on runtime values |
| Error handling | TRY_CATCH, ERR | Control | Recovery from failures |
| Memory | QMEM, UMEM, FENCE | Data | Persistent state across steps |
| Coordination | GUARD, CLAIM, UPDATE_GOAL | Data | Multi-agent work distribution |
