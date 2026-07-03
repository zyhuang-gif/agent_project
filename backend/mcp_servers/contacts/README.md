# Contacts MCP Server

P0 exposes:

- `resolve(query)` for contact/group resolution.
- `list_all()` for smoke/debug scripts.

The graph send node does deterministic preflight in backend code and does not expose `contacts_*` tools to the local send agent.
