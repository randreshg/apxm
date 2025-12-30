# APXM Examples

Example DSL programs demonstrating APXM capabilities.

## Files

| File | Description |
|------|-------------|
| `hello_world.ais` | Minimal agent with basic reasoning |
| `message_handler.ais` | Event-driven message handling |
| `planner.ais` | Multi-step planning agent |

## Running Examples

### Using Python CLI (Recommended)

```bash
# Build the compiler first
python tools/apxm_cli.py compiler build

# Run an example
python tools/apxm_cli.py compiler run examples/hello_world.ais

# Compile to artifact only
python tools/apxm_cli.py compiler compile examples/hello_world.ais -o examples/hello_world.apxmobj
```

### Manual (After Building)

```bash
# Use the compiled binary directly
./target/release/apxm run examples/hello_world.ais
./target/release/apxm compile examples/hello_world.ais -o output.apxmobj
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
