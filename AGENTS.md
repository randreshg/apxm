# AGENTS.md

## Scope
Applies to the APXM repository.

## Command Policy (Script-First)
- Always run commands through the repo script when possible: `./apxm ...`
- If a raw `cargo` command is required, activate the APXM environment first:
  - `eval "$(./apxm activate)"`

## Preferred Commands
- Build: `./apxm build`
- Compile + execute graph: `./apxm execute --cargo <file.json> [args...]`
- Run artifact: `./apxm run <file.apxmobj> [args...]`
- Diagnostics: `./apxm doctor`

## Notes
- Run all commands from the repository root (`apxm/`).
- For automation, prefer `--cargo` with `execute` to ensure required features are present.
