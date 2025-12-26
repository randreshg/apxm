# APXM Examples

Example DSL programs demonstrating APXM capabilities.

## Files

| File | Description |
|------|-------------|
| `hello_world.ais` | Minimal agent with basic reasoning |
| `message_handler.ais` | Event-driven message handling |
| `planner.ais` | Multi-step planning agent |

## Running Examples

```bash
# Compile an example to artifact
cargo run -p apxm-cli --features driver -- compile examples/hello_world.ais

# Run an example (compile + execute)
cargo run -p apxm-cli --features driver -- run examples/hello_world.ais
```

## DSL Quick Reference

### Agent Declaration
```
agent AgentName {
    // flows, handlers, goals, beliefs, memory
}
```

### Flow (Reusable Logic)
```
flow name(arg: type) -> type {
    // statements
}
```

### Event Handler
```
on EventType { field1, field2 } if (condition) => {
    // statements
    return result;
}
```

### Built-in Operations
- `rsn "prompt" -> result` - Reasoning operation
- `think("description", context)` - Thinking/analysis
- `llm("prompt", context)` - LLM inference
- `plan("goal", context)` - Planning
- `verify(condition, message)` - Validation
- `tool(name, args)` - Tool execution

### Goals
```
goals {
    goal_name(priority: NUMBER, description: "...")
}
```

### Beliefs
```
beliefs {
    name: from expression
}
```
