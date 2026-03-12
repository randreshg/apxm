# Registering LLM Backends in APXM

This guide covers how to register and configure LLM backends for use with APXM workflows.

## Table of Contents

- [Overview](#overview)
- [Credential Storage](#credential-storage)
- [Supported Providers](#supported-providers)
- [Registration Commands](#registration-commands)
- [Provider-Specific Examples](#provider-specific-examples)
- [Testing Credentials](#testing-credentials)
- [Configuration Integration](#configuration-integration)
- [Troubleshooting](#troubleshooting)

---

## Overview

APXM uses a secure credential store at `~/.apxm/credentials.toml` to manage LLM provider API keys and configuration. The `apxm register` command provides a simple interface for adding, listing, testing, and removing credentials.

**Key Features:**
- Secure storage with strict file permissions (0600)
- Support for 5+ LLM providers
- Custom headers for enterprise gateways
- Interactive API key entry (hidden input)
- Validation via real API calls

---

## Credential Storage

### Location and Security

Credentials are stored at:
```
~/.apxm/credentials.toml
```

**Security measures:**
- File permissions: `0600` (owner read/write only)
- Directory permissions: `0700` (owner-only access)
- Git protection: refuses to store inside git repositories
- Auto-generated `.gitignore` as safety net
- Permissions validated on every read

### File Structure

```toml
# APXM Credentials - Managed by `apxm register`
# Permissions: 0600 (owner read/write only)
# DO NOT edit manually unless you know what you're doing.

[credentials.my-openai]
provider = "openai"
api_key = "sk-proj-abc123..."
model = "gpt-4o-mini"

[credentials.my-anthropic]
provider = "anthropic"
api_key = "sk-ant-xyz789..."

[credentials.local-ollama]
provider = "ollama"
```

**Note:** While credentials are stored as plaintext, this follows industry-standard practices used by AWS CLI, GitHub CLI, npm, and other developer tools. The strict file permissions prevent unauthorized access on multi-user systems.

---

## Supported Providers

| Provider | Name | Default Model | API Key Required | Base URL |
|----------|------|---------------|------------------|----------|
| OpenAI | `openai` | `gpt-4o-mini` | Yes | `https://api.openai.com/v1` |
| Anthropic | `anthropic` | `claude-opus-4` | Yes | `https://api.anthropic.com/v1` |
| Google | `google` | `gemini-flash-latest` | Yes | `https://generativelanguage.googleapis.com/v1beta` |
| Ollama | `ollama` | `gpt-oss:120b-cloud` | No | `http://localhost:11434` |
| OpenRouter | `openrouter` | (varies) | Yes | `https://openrouter.ai/api` |

**Custom Providers:** Any OpenAI-compatible API endpoint can be registered using `provider = "openai"` with a custom `--base-url`.

---

## Registration Commands

### Add a Credential

```bash
apxm register add <name> --provider <provider> [OPTIONS]
```

**Arguments:**
- `<name>` - Unique identifier for this credential (e.g., "my-openai", "work-claude")
- `--provider` - Provider type (openai, anthropic, google, ollama, openrouter)

**Options:**
- `--api-key <KEY>` - API key (omit for interactive entry)
- `--base-url <URL>` - Custom API endpoint
- `--model <MODEL>` - Default model name
- `--header <KEY=VALUE>` - Additional HTTP headers (repeatable)

**Examples:**

```bash
# Interactive API key entry (recommended)
apxm register add my-openai --provider openai

# Command-line API key
apxm register add my-openai --provider openai --api-key sk-proj-abc123...

# With custom model
apxm register add my-gpt4 --provider openai --api-key sk-... --model gpt-4

# Enterprise gateway with custom headers
apxm register add corp-llm \
  --provider openai \
  --api-key dummy \
  --base-url https://llm-api.company.com/v1 \
  --model gpt-oss-20b \
  --header "Ocp-Apim-Subscription-Key=abc123" \
  --header "user=$USER"
```

### List Credentials

```bash
apxm register list
```

**Output:**
```
Registered Credentials
  my-openai        openai       key=sk-p...xyz
  my-anthropic     anthropic    key=sk-a...123  model=claude-opus-4
  corp-llm         openai       key=dumm...ummy  model=gpt-oss-20b  +2 headers
  local-ollama     ollama       key=<none>

Store: /home/user/.apxm/credentials.toml
```

**Note:** API keys are masked for security. Only the first 4 and last 3 characters are shown.

### Remove a Credential

```bash
apxm register remove <name>
```

**Example:**
```bash
apxm register remove my-openai
```

**Note:** Credentials are immutable. To update a credential, remove and re-add it:
```bash
apxm register remove my-openai
apxm register add my-openai --provider openai --api-key sk-new-key
```

### Test Credentials

```bash
# Test a specific credential
apxm register test <name>

# Test all registered credentials
apxm register test
```

**Example:**
```bash
$ apxm register test my-openai
Testing Credentials
  my-openai        OK (200)
```

**How Testing Works:**

Different providers use different validation methods:

- **OpenAI/OpenRouter:** Tries `GET /v1/models`, falls back to minimal chat completion
- **Anthropic:** Sends minimal message request to `/v1/messages`
- **Google:** Queries `GET /v1/models` with API key
- **Ollama:** Checks local server at `GET /api/tags`

A 400 status is considered success if authentication worked (indicates valid key but bad request).

### Generate config.toml

```bash
apxm register generate-config >> ~/.apxm/config.toml
```

Converts registered credentials to config.toml format (useful for migration or inspection).

---

## Provider-Specific Examples

### OpenAI

**Standard OpenAI API:**
```bash
apxm register add my-openai --provider openai
# Enter API key when prompted
```

**With specific model:**
```bash
apxm register add my-gpt4 \
  --provider openai \
  --api-key sk-proj-... \
  --model gpt-4
```

**OpenAI-compatible endpoint (e.g., Azure OpenAI):**
```bash
apxm register add azure-gpt \
  --provider openai \
  --api-key your-key \
  --base-url https://your-resource.openai.azure.com/openai/deployments/your-deployment \
  --model gpt-4
```

**Models supported:**
- `gpt-4o` - Latest flagship model
- `gpt-4o-mini` - Fast, cost-effective (default)
- `gpt-4-turbo` - Previous generation flagship
- `gpt-4` - Original GPT-4
- `gpt-3.5-turbo` - Legacy model

### Anthropic

**Standard Claude API:**
```bash
apxm register add my-claude --provider anthropic
# Enter API key when prompted
```

**With specific model:**
```bash
apxm register add claude-sonnet \
  --provider anthropic \
  --api-key sk-ant-... \
  --model claude-sonnet-4
```

**Models supported:**
- `claude-opus-4` - Highest capability (default)
- `claude-sonnet-4` - Balanced performance
- `claude-haiku-4` - Fast, efficient
- `claude-3-opus-20240229` - Legacy Opus
- `claude-3-5-sonnet-20241022` - Legacy Sonnet

### Google

**Standard Gemini API:**
```bash
apxm register add my-gemini --provider google
# Enter API key when prompted
```

**With specific model:**
```bash
apxm register add gemini-pro \
  --provider google \
  --api-key AIza... \
  --model gemini-pro
```

**Models supported:**
- `gemini-flash-latest` - Fast, efficient (default)
- `gemini-pro-latest` - Advanced capabilities
- `gemini-pro` - Stable version
- `gemini-flash` - Speed-optimized

### Ollama (Local Models)

**Standard local Ollama:**
```bash
apxm register add local --provider ollama
```

**Custom Ollama server:**
```bash
apxm register add remote-ollama \
  --provider ollama \
  --base-url http://gpu-server:11434 \
  --model llama3.1:70b
```

**Models:** Any model available in your local Ollama installation. Use `ollama list` to see available models.

**Note:** Ollama does not require an API key. The provider connects to your local Ollama server (default: `http://localhost:11434`).

### OpenRouter

**Multi-provider gateway:**
```bash
apxm register add openrouter \
  --provider openrouter \
  --api-key sk-or-... \
  --model anthropic/claude-opus-4
```

**Note:** OpenRouter provides access to multiple model providers through a single API. Specify the model in the format `provider/model-name`.

### Enterprise/On-Premises Gateways

**Custom gateway with authentication headers:**
```bash
apxm register add corp-gateway \
  --provider openai \
  --api-key dummy \
  --base-url https://llm-api.company.com/OnPrem \
  --model GPT-oss-20B \
  --header "Ocp-Apim-Subscription-Key=your-subscription-key" \
  --header "user=$USER"
```

**Custom headers are useful for:**
- Azure APIM subscription keys
- Corporate proxy authentication
- User tracking/attribution
- Custom routing headers

**Note:** Custom headers are added to every request. Values are stored in plaintext in the credentials file.

---

## Testing Credentials

### Why Test?

Testing validates:
1. API key is correct
2. Network connectivity works
3. Provider endpoint is reachable
4. Custom headers are properly configured

### How to Test

**Test a single credential:**
```bash
apxm register test my-openai
```

**Test all credentials:**
```bash
apxm register test
```

**Example output:**
```
Testing Credentials
  my-openai        OK (200)
  my-anthropic     OK (200)
  corp-gateway     OK (200)
  broken-key       ERROR (401) Unauthorized
```

### Interpreting Results

| Status | Meaning |
|--------|---------|
| `OK (200)` | Success - credential is valid |
| `OK (400)` | Success - auth worked, request format issue (still valid) |
| `ERROR (401)` | Unauthorized - invalid API key |
| `ERROR (403)` | Forbidden - valid key but insufficient permissions |
| `ERROR (404)` | Endpoint not found - check base_url |
| `ERROR (...)` | Other HTTP error - check provider status |

### Common Issues

**OpenAI 404 on /v1/models:**
- Expected for some on-premises gateways
- Test falls back to `/chat/completions` automatically
- If you see `OK (200)` or `OK (400)`, your credential is valid

**Anthropic 400 Bad Request:**
- Expected - minimal request is intentionally malformed
- Authentication is checked before request validation
- `OK (400)` means your API key is valid

**Ollama Connection Failed:**
- Check Ollama is running: `ollama list`
- Verify base_url if using remote server
- Ensure port is accessible (default: 11434)

---

## Configuration Integration

### Using Credentials in Workflows

When you register credentials, they become available to APXM workflows. Reference them in your config:

**~/.apxm/config.toml:**
```toml
[chat]
providers = ["my-openai", "my-claude"]
default_model = "gpt-4o-mini"
```

The credential names in the `providers` list must match registered credential names.

### Credential Store as Source of Truth

When using `apxm register`, you don't need `[[llm_backends]]` sections in config.toml. The credential store is the source of truth.

**Before (manual config):**
```toml
[[llm_backends]]
name = "my-openai"
provider = "openai"
api_key = "sk-proj-..."
model = "gpt-4"

[[llm_backends]]
name = "my-claude"
provider = "anthropic"
api_key = "sk-ant-..."
```

**After (using credential store):**
```toml
[chat]
providers = ["my-openai", "my-claude"]
```

Credentials are automatically loaded from `~/.apxm/credentials.toml`.

### Generating config.toml from Credentials

If you want to migrate to manual config or inspect the generated format:

```bash
apxm register generate-config
```

**Output:**
```toml
# Generated by `apxm register generate-config`

[chat]
providers = ["my-openai", "my-claude"]

[[llm_backends]]
name = "my-openai"
provider = "openai"
api_key = "sk-proj-..."
model = "gpt-4o-mini"

[[llm_backends]]
name = "my-claude"
provider = "anthropic"
api_key = "sk-ant-..."
model = "claude-opus-4"
```

You can redirect this to a file:
```bash
apxm register generate-config >> ~/.apxm/config.toml
```

---

## Troubleshooting

### Permission Errors

**Error:** `Credential store at ~/.apxm/credentials.toml has insecure permissions (644)`

**Solution:**
```bash
chmod 600 ~/.apxm/credentials.toml
chmod 700 ~/.apxm
```

### Git Repository Warning

**Error:** `Credential store is inside a git repository at /path/to/repo`

**Solution:**
Move `~/.apxm` outside of any git repository. The credential store must be in a location not tracked by git to prevent accidental credential leaks.

### Credential Already Exists

**Error:** `Credential 'my-openai' already exists`

**Solution:**
Remove the existing credential first:
```bash
apxm register remove my-openai
apxm register add my-openai --provider openai --api-key sk-new-key
```

### Provider Not Found

**Error:** `Unknown provider 'xyz' - cannot validate`

**Solution:**
Use one of the supported providers:
- `openai`
- `anthropic`
- `google`
- `ollama`
- `openrouter`

For custom endpoints, use `provider = "openai"` with `--base-url`.

### API Key Not Working

**Symptoms:**
- `apxm register test` shows `ERROR (401)`
- Workflows fail with authentication errors

**Debugging steps:**

1. **Verify API key is correct:**
   ```bash
   # Re-register with correct key
   apxm register remove my-openai
   apxm register add my-openai --provider openai
   # Enter correct key when prompted
   ```

2. **Test with curl:**
   ```bash
   # OpenAI
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer sk-your-key"

   # Anthropic
   curl https://api.anthropic.com/v1/messages \
     -H "x-api-key: sk-ant-your-key" \
     -H "anthropic-version: 2023-06-01" \
     -H "content-type: application/json" \
     -d '{"model":"claude-3-haiku-20240307","max_tokens":1,"messages":[{"role":"user","content":"hi"}]}'
   ```

3. **Check provider status:**
   - OpenAI: https://status.openai.com/
   - Anthropic: https://status.anthropic.com/
   - Google: https://status.cloud.google.com/

4. **Verify account/billing:**
   - Ensure your API key has active credits
   - Check if your account is in good standing
   - Verify usage limits haven't been exceeded

### Custom Headers Not Working

**Symptoms:**
- On-premises gateway returns 401/403
- Custom authentication fails

**Debugging:**

1. **Check header format:**
   ```bash
   # Correct: KEY=VALUE (no spaces around =)
   --header "Ocp-Apim-Subscription-Key=abc123"

   # Incorrect: KEY = VALUE
   --header "Ocp-Apim-Subscription-Key = abc123"
   ```

2. **Test headers manually:**
   ```bash
   curl https://your-gateway.com/v1/chat/completions \
     -H "Authorization: Bearer dummy" \
     -H "Ocp-Apim-Subscription-Key: abc123" \
     -H "user: $USER" \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-4","messages":[{"role":"user","content":"test"}],"max_tokens":1}'
   ```

3. **Inspect stored headers:**
   ```bash
   cat ~/.apxm/credentials.toml
   ```

   Should show:
   ```toml
   [credentials.corp-gateway.headers]
   Ocp-Apim-Subscription-Key = "abc123"
   user = "youruser"
   ```

### Ollama Connection Issues

**Error:** `Failed to connect to Ollama`

**Solutions:**

1. **Check Ollama is running:**
   ```bash
   ollama list
   # Should show installed models
   ```

2. **Start Ollama if not running:**
   ```bash
   ollama serve
   ```

3. **Verify port accessibility:**
   ```bash
   curl http://localhost:11434/api/tags
   # Should return JSON with model list
   ```

4. **For remote Ollama:**
   ```bash
   # Test remote connection
   curl http://gpu-server:11434/api/tags

   # Register with correct base_url
   apxm register add remote-ollama \
     --provider ollama \
     --base-url http://gpu-server:11434
   ```

### Multiple Credentials for Same Provider

You can register multiple credentials for the same provider:

```bash
apxm register add openai-personal --provider openai --api-key sk-personal-...
apxm register add openai-work --provider openai --api-key sk-work-...
apxm register add gpt4 --provider openai --api-key sk-... --model gpt-4
apxm register add gpt4-mini --provider openai --api-key sk-... --model gpt-4o-mini
```

Use different names and reference the appropriate one in your workflow config.

### Environment Variable Substitution

Custom headers support `env:` prefix for environment variables:

**In config.toml (manual config):**
```toml
[[llm_backends]]
name = "corp"
provider = "openai"
api_key = "env:CORP_API_KEY"
base_url = "https://gateway.company.com"

[llm_backends.extra_headers]
user = "env:USER"
```

**Note:** The credential store stores literal values. Environment variable substitution only works in config.toml with the `env:` prefix. When using `apxm register --header "user=$USER"`, the value is expanded by your shell before being stored.

---

## Summary

**Quick Reference:**

```bash
# Add credentials
apxm register add my-openai --provider openai
apxm register add my-claude --provider anthropic
apxm register add local --provider ollama

# Test credentials
apxm register test

# List credentials
apxm register list

# Remove credential
apxm register remove my-openai

# Use in config
echo 'providers = ["my-openai", "my-claude"]' >> ~/.apxm/config.toml
```

**Best Practices:**

1. Use interactive API key entry (omit `--api-key` flag) to avoid shell history
2. Test credentials after registration
3. Use descriptive names (e.g., "work-gpt4", "personal-claude")
4. Keep credentials outside git repositories
5. Verify file permissions are 0600
6. Use `apxm register test` regularly to catch expired keys
7. For production, consider using environment variables in config.toml

**Security Notes:**

- Credentials are stored as plaintext (industry standard for CLI tools)
- File permissions (0600) prevent unauthorized access
- Never commit `credentials.toml` to version control
- Use separate credentials for different environments (dev/prod)
- Rotate API keys regularly following provider best practices

For more information, see:
- [CLI Reference](cli-reference.md) - Complete CLI documentation
- [Configuration](../README.md) - Config.toml reference
- Provider documentation:
  - [OpenAI API](https://platform.openai.com/docs/api-reference)
  - [Anthropic API](https://docs.anthropic.com/claude/reference/getting-started)
  - [Google AI](https://ai.google.dev/docs)
  - [Ollama](https://ollama.ai/docs)
