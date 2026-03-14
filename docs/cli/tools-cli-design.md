# `apxm tools` CLI Design

## Overview

The `apxm tools` CLI provides a parallel interface to the existing `apxm register` command (which manages LLM credentials). While `register` focuses on LLM provider configuration, `tools` manages **tool/capability registration** — binary tools, MCP servers, HTTP endpoints, Python functions, and other executable capabilities.

This document specifies the complete CLI interface, focusing on consistency with APXM's existing patterns and integration with the runtime's `CapabilitySystem`.

---

## Architecture Context

### Existing APXM Tool Infrastructure

1. **CapabilityExecutor** (trait in `apxm-runtime/src/capability/executor.rs`)
   - Async execute method: `execute(&self, args: HashMap<String, Value>) -> CapabilityResult<Value>`
   - Returns metadata for schema validation and introspection

2. **CapabilityRegistry** (in `apxm-runtime/src/capability/registry.rs`)
   - Thread-safe, concurrent registry using DashMap
   - Stores capabilities by name and validates schemas on registration
   - Methods: `register()`, `get()`, `list_names()`, `list_metadata()`, `unregister()`

3. **CapabilitySystem** (in `apxm-runtime/src/capability/mod.rs`)
   - High-level coordinator with timeout, validation, and interceptor support
   - Public API: `register()`, `invoke()`, `list_capabilities()`, `get_metadata()`

4. **Built-in Tools** (in `apxm-tools/src/`)
   - `BashCapability` — shell commands with policy enforcement
   - `ReadCapability` — filesystem read access
   - `WriteCapability` — filesystem write access
   - `SearchWebCapability` — web search with depth control
   - All registered via `register_standard_tools(capability_system, config)`

5. **MCP Integration** (in `agentmate/crates/am-mcp/src/`)
   - `McpClient` — connects to MCP servers via stdio JSON-RPC
   - `McpToolAdapter` — bridges MCP tools to AgentMate `Tool` trait
   - `McpToolAdapter::from_client()` — auto-discovers tools from server

### Credential Store Pattern (Reference)

The `apxm register` command (for LLM credentials) follows this pattern:

```
CredentialStore::open()
  ↓ (~/.apxm/credentials.toml, owned 0600)
  add(name, credential)  ← CLI input
  get(name)              ← Runtime lookup
  list()                 ← Human-readable display
  list_all()             ← Full data for runtime
  remove(name)
  find_by_provider()
  generate_config()      ← For runtime integration
```

The tools system needs a **parallel storage mechanism** for tool definitions.

---

## Tool Storage Design

### File Format: `~/.apxm/tools.toml`

Tools are stored in TOML alongside credentials, with per-tool configuration:

```toml
# ~/.apxm/tools.toml
# Managed by `apxm tools` commands
# File permissions: 0600 (owner read/write only)

[tool.bash]
type = "binary"
enabled = true
command = "bash"
description = "Execute shell commands"
parameters_schema = { type = "object", properties = {...}, required = ["command"] }
return_type = "string"
timeout_secs = 120
# type-specific config
blocked_commands = ["rm -rf", "sudo"]
max_output_bytes = 100000

[tool.read]
type = "binary"
enabled = true
command = "read"
description = "Read file contents"
# ...

[tool.my_search]
type = "mcp"
enabled = true
description = "Search with custom MCP server"
mcp_command = "node /path/to/mcp-search-server.js"
# MCP discovery happens at runtime

[tool.http_api]
type = "http"
enabled = true
description = "Call custom HTTP API"
base_url = "https://api.example.com"
parameters_schema = {...}
return_type = "json"

[tool.custom_python]
type = "python"
enabled = true
description = "Custom Python function"
module_path = "/home/user/tools/custom.py"
function_name = "my_tool"
```

### Data Structures (Rust)

```rust
// crates/apxm-tools/src/tool_store.rs (NEW)

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ToolType {
    #[serde(rename = "binary")]
    Binary { command: String },

    #[serde(rename = "mcp")]
    Mcp { mcp_command: String },

    #[serde(rename = "http")]
    Http { base_url: String, auth_header: Option<String> },

    #[serde(rename = "python")]
    Python { module_path: PathBuf, function_name: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub tool_type: ToolType,
    pub enabled: bool,
    pub description: String,
    pub parameters_schema: serde_json::Value,
    pub return_type: String,
    pub timeout_secs: u64,
    pub tags: Vec<String>,
}

pub struct ToolStore {
    path: PathBuf,
    dir: PathBuf,
}

impl ToolStore {
    pub fn open() -> Result<Self> { /* ~/. apxm/tools.toml */ }
    pub fn add(&self, name: &str, tool: ToolDefinition) -> Result<()> { }
    pub fn get(&self, name: &str) -> Result<Option<ToolDefinition>> { }
    pub fn list(&self) -> Result<Vec<(String, ToolDefinition)>> { }
    pub fn remove(&self, name: &str) -> Result<()> { }
    pub fn enable(&self, name: &str) -> Result<()> { }
    pub fn disable(&self, name: &str) -> Result<()> { }
}
```

---

## CLI Command Specification

### 1. `apxm tools register`

**Purpose:** Register a new tool/capability

**Command Variants:**

#### Binary Tool
```bash
apxm tools register my-tool \
  --type binary \
  --command "mytool" \
  --description "My custom tool" \
  --schema '{"type":"object","properties":{"input":{"type":"string"}},"required":["input"]}' \
  --return-type string \
  --timeout 30 \
  [--tag "category:search" --tag "version:1.0"]
```

#### MCP Server
```bash
apxm tools register my-mcp-search \
  --type mcp \
  --mcp-command "node /path/to/server.js" \
  --description "Search via MCP"
  # Schema auto-discovered at runtime
```

#### HTTP API
```bash
apxm tools register api-tool \
  --type http \
  --base-url "https://api.example.com/tools" \
  --schema '{...}' \
  --return-type json \
  [--auth-header "Authorization: Bearer token"]
```

#### Python Function
```bash
apxm tools register py-func \
  --type python \
  --module "/path/to/tool.py" \
  --function "my_tool" \
  --description "Python-based tool"
```

**Common Flags:**
- `--name <name>` — Required, tool identifier (alphanumeric + dash/underscore)
- `--type <type>` — Required, one of: binary, mcp, http, python
- `--description <text>` — Optional, human-readable description
- `--schema <json>` — Optional, JSON schema for input validation
- `--return-type <type>` — Optional, default "string", one of: string, json, object, array
- `--timeout <secs>` — Optional, default 120
- `--tag <key:value>` — Optional, multiple allowed, for categorization
- `--enabled/--disabled` — Optional, default enabled
- `--json` — Optional, output JSON instead of human format

**Output (Human):**
```
  Tool Registered
  ──────────────────
  ✓ Name                [OK] my-tool
  ✓ Type                [OK] binary
  ✓ Store               [OK] /home/user/.apxm/tools.toml
```

**Output (JSON):**
```json
{
  "status": "ok",
  "tool": {
    "name": "my-tool",
    "type": "binary",
    "enabled": true,
    "description": "My custom tool",
    "path": "/home/user/.apxm/tools.toml"
  }
}
```

---

### 2. `apxm tools list`

**Purpose:** List all registered tools with summaries

**Syntax:**
```bash
apxm tools list [--enabled] [--type <type>] [--json] [--format <format>]
```

**Flags:**
- `--enabled` — Only show enabled tools (default: all)
- `--type <type>` — Filter by type: binary, mcp, http, python (default: all)
- `--format <format>` — Output format: table (default), list, csv
- `--json` — Output structured JSON

**Output (Human, Table):**
```
  Registered Tools
  ────────────────────────────────────────────────────────────────
  Name              Type    Enabled  Return   Timeout  Description
  ────────────────────────────────────────────────────────────────
  bash              binary  yes      string   120s     Shell commands
  read              binary  yes      string   30s      Read files
  my-search         mcp     yes       string  120s     Custom search
  api-tool          http    yes      json     60s      API endpoint
  ────────────────────────────────────────────────────────────────
  Store: /home/user/.apxm/tools.toml
  Total: 4 tools (4 enabled)
```

**Output (JSON):**
```json
{
  "tools": [
    {
      "name": "bash",
      "type": "binary",
      "enabled": true,
      "description": "Shell commands",
      "return_type": "string",
      "timeout_secs": 120,
      "tags": []
    },
    ...
  ],
  "total": 4,
  "enabled_count": 4,
  "store": "/home/user/.apxm/tools.toml"
}
```

---

### 3. `apxm tools show`

**Purpose:** Show detailed information for a specific tool

**Syntax:**
```bash
apxm tools show <name> [--json] [--include-schema]
```

**Flags:**
- `<name>` — Tool name (required)
- `--json` — Output JSON
- `--include-schema` — Include full JSON schema in output
- `--include-metadata` — Include runtime metadata (auto-discovered for MCP)

**Output (Human):**
```
  Tool Details: bash
  ──────────────────────────────────────────
  Name                bash
  Type                binary
  Enabled             yes
  Command             bash
  Description         Execute shell commands

  Parameters
  ──────────────────────────────────────────
  Type                object
  Properties
    - command         string (required) — Shell command to run
    - timeout         integer (optional) — Timeout in seconds

  Execution
  ──────────────────────────────────────────
  Return Type         string
  Timeout             120 seconds
  Max Output          100000 bytes

  Policies
  ──────────────────────────────────────────
  Blocked Commands    rm -rf, sudo, su, mkfs, ...

  Store               /home/user/.apxm/tools.toml
```

**Output (JSON):**
```json
{
  "name": "bash",
  "type": "binary",
  "enabled": true,
  "description": "Execute shell commands",
  "command": "bash",
  "parameters_schema": {
    "type": "object",
    "properties": {
      "command": {
        "type": "string",
        "description": "Shell command to run"
      }
    },
    "required": ["command"]
  },
  "return_type": "string",
  "timeout_secs": 120,
  "config": {
    "max_output_bytes": 100000,
    "blocked_commands": ["rm -rf", "sudo", ...]
  },
  "tags": []
}
```

---

### 4. `apxm tools remove`

**Purpose:** Remove a registered tool

**Syntax:**
```bash
apxm tools remove <name> [--force]
```

**Flags:**
- `<name>` — Tool name (required)
- `--force` — Skip confirmation

**Output:**
```
  Tool Removed
  ──────────────────
  ✓ bash    [OK] removed from /home/user/.apxm/tools.toml
```

---

### 5. `apxm tools test`

**Purpose:** Test a tool by invoking it with parameters

**Syntax:**
```bash
apxm tools test <name> [--param <key:value>] [--input <json>] [--json]
```

**Flags:**
- `<name>` — Tool name (required)
- `--param <key:value>` — Input parameter, multiple allowed
- `--input <json>` — Full JSON input (alternative to --param)
- `--json` — Output JSON
- `--timeout <secs>` — Override timeout for test

**Examples:**
```bash
# Test bash tool
apxm tools test bash --param 'command:echo hello'

# Test with full JSON input
apxm tools test my-search --input '{"query":"rust lang"}'

# Test MCP tool
apxm tools test my-mcp --param 'arg:value'
```

**Output (Success):**
```
  Test Result
  ──────────────────────────────────────
  Tool                bash
  Status              ✓ SUCCESS
  Duration            125 ms
  Output              hello
```

**Output (Failure):**
```
  Test Result
  ──────────────────────────────────────
  Tool                bash
  Status              ✗ FAILED
  Duration            15 ms
  Error               command not found

  Validation Errors   (if input validation failed)
    • "command" is required
```

**Output (JSON):**
```json
{
  "tool": "bash",
  "status": "success",
  "duration_ms": 125,
  "output": "hello",
  "input": {
    "command": "echo hello"
  }
}
```

Error JSON:
```json
{
  "tool": "bash",
  "status": "failed",
  "duration_ms": 15,
  "error": "command not found",
  "validation_errors": [
    {
      "path": "$.command",
      "message": "is required"
    }
  ]
}
```

---

### 6. `apxm tools discover`

**Purpose:** Auto-discover tools from an MCP server without registration

**Syntax:**
```bash
apxm tools discover <mcp-command> [--json] [--register]
```

**Flags:**
- `<mcp-command>` — Command to start MCP server (e.g., `node server.js`)
- `--json` — Output JSON
- `--register` — Automatically register all discovered tools
- `--timeout <secs>` — MCP connection timeout

**Output (Human):**
```
  Discovering MCP Server Tools
  ────────────────────────────────────────────────────────────────
  Command             node /path/to/server.js
  Status              ✓ CONNECTED
  Protocol Version    2024-11-05

  Tools Found
  ────────────────────────────────────────────────────────────────
  Name                Description
  ────────────────────────────────────────────────────────────────
  filesystem          Read/write files in directory
  web_search          Search the web
  database_query      Query SQL database
  ────────────────────────────────────────────────────────────────
  Total: 3 tools

  Next Steps
  ────────────────────────────────────────────────────────────────
  apxm tools discover <cmd> --register  # Auto-register all
  apxm tools register <name> --type mcp --mcp-command <cmd>  # Manual
```

**Output (JSON):**
```json
{
  "command": "node /path/to/server.js",
  "status": "connected",
  "protocol_version": "2024-11-05",
  "tools": [
    {
      "name": "filesystem",
      "description": "Read/write files in directory",
      "input_schema": {
        "type": "object",
        "properties": {...}
      }
    },
    ...
  ],
  "total": 3
}
```

---

### 7. `apxm tools enable/disable`

**Purpose:** Enable or disable a tool without removing it

**Syntax:**
```bash
apxm tools enable <name>
apxm tools disable <name>
```

**Output:**
```
  Tool bash
  ──────────────────
  ✓ [OK] enabled
```

---

### 8. `apxm tools doctor`

**Purpose:** Diagnose tool registration issues

**Syntax:**
```bash
apxm tools doctor [--verbose] [--json]
```

**Output (Human):**
```
  Tools Doctor
  ──────────────────────────────────────────
  ✓ Store file       [OK]  /home/user/.apxm/tools.toml
  ✓ Permissions      [OK]  0600 (secure)
  ✓ Total tools      [OK]  4 registered
  ✓ Enabled tools    [OK]  4 enabled
  ✓ MCP servers      [OK]  1 online
  ✓ Config schema    [OK]  valid

  Issues: None
```

**Output (with issues):**
```
  Tools Doctor
  ──────────────────────────────────────────
  ✓ Store file       [OK]  /home/user/.apxm/tools.toml
  ✗ Permissions      [ERROR] 0644 (world-readable, should be 0600)
  ✓ Total tools      [OK]  4 registered
  ✗ MCP servers      [WARN] 1 server offline: my-search

  Issues (2)
  ──────────────────────────────────────────
  1. Tool store has insecure permissions.
     Fix: chmod 600 /home/user/.apxm/tools.toml

  2. MCP server 'my-search' is not responding.
     Check: node /path/to/server.js
     Last error: connection timeout
```

---

## Integration Points

### 1. Runtime Initialization

When APXM runtime starts, it loads tools from the store:

```rust
// In apxm-driver or apxm-runtime initialization
let tool_store = ToolStore::open()?;
let tool_defs = tool_store.list()?;

for (name, def) in tool_defs {
    if !def.enabled {
        continue;
    }

    match def.tool_type {
        ToolType::Binary { ref command } => {
            // Create BinaryCapability, register with CapabilitySystem
        }
        ToolType::Mcp { ref mcp_command } => {
            // Connect McpClient, discover tools, adapt to Capabilities
        }
        ToolType::Http { ref base_url, .. } => {
            // Create HttpCapability
        }
        ToolType::Python { ref module_path, ref function_name } => {
            // Load Python module via PyO3, create PythonCapability
        }
    }
}
```

### 2. Agent Access

When an LLM agent asks "what tools do I have?", the runtime responds:

```bash
apxm tools list --json
```

Output is fed to the LLM's context or used in constrained generation.

### 3. Config Integration

The runtime config (`~/.apxm/config.toml`) can reference the tool store:

```toml
[tools]
# Load tool definitions from store
from_store = true

# Or override/add tools inline
[[tools.inline]]
name = "custom"
type = "python"
module_path = "/home/user/my_tool.py"
```

### 4. Workflow/Graph Integration

AgentMate graphs can reference tools:

```python
from apxm import Flow

@Flow
def my_workflow():
    result = apxm_call("bash", command="echo hello")
    return result
```

The runtime looks up "bash" in the tool registry and invokes it.

---

## Error Handling

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Tool name already exists` | Duplicate registration | `apxm tools remove <name>` first |
| `Invalid schema` | Malformed JSON schema | Check JSON syntax and schema structure |
| `MCP server not responding` | Server command failed/timeout | Debug with `apxm tools discover <cmd>` |
| `Insecure permissions` | Store file world-readable | `chmod 600 ~/.apxm/tools.toml` |
| `Inside git repo` | Store in version-controlled dir | Move to `~/.apxm/` outside repo |
| `Command not found` | Binary tool command not in PATH | Install tool or provide full path |
| `Validation failed` | Input doesn't match schema | Check input against schema in `apxm tools show` |

---

## Comparison: `apxm register` vs `apxm tools`

| Aspect | register (LLM) | tools (Capabilities) |
|--------|---|---|
| **Purpose** | Manage LLM provider credentials | Manage tool/capability definitions |
| **Storage** | `~/.apxm/credentials.toml` | `~/.apxm/tools.toml` |
| **Main commands** | add, list, remove, test | register, list, show, remove, test, discover |
| **Data types** | API key, provider, model | Tool type, command/endpoint, schema |
| **Registration model** | Interactive or flag-based | Interactive or auto-discovery (MCP) |
| **Runtime use** | Passed to LLM backends | Invoked as capabilities during execution |
| **Validation** | API key test call | Schema validation + test invocation |
| **Introspection** | Limited (masked keys) | Full (schema, metadata) |

---

## Future Extensions

1. **Tool Versioning** — Track and switch between tool versions
2. **Tool Marketplace** — Publish/discover tools from registry
3. **Conditional Registration** — Enable/disable based on platform or config
4. **Tool Aliasing** — Create aliases for tools (`alias search => web_search`)
5. **Tool Chaining** — Compose tools into workflows
6. **Async Tool Queues** — Queue long-running tool invocations
7. **Tool Telemetry** — Metrics on tool usage, latency, errors

---

## Implementation Roadmap

### Phase 1: Core (MVP)
- `tools register` (binary type only)
- `tools list`, `tools show`
- `tools remove`, `tools test`
- ToolStore TOML file format
- Basic error handling

### Phase 2: Discovery & MCP
- `tools discover` command
- MCP server connection and auto-discovery
- MCP tool adapter integration
- Runtime loading of MCP tools

### Phase 3: Advanced Types
- HTTP tool type support
- Python tool type (PyO3 bridge)
- `tools enable/disable`
- `tools doctor` diagnostics

### Phase 4: Integration
- Config file integration
- Workflow/graph tool references
- CLI output improvements (colors, tables)
- Comprehensive tests
