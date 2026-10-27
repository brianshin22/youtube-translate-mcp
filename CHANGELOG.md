# Changelog

## Unreleased - 2026-10-27

- Accept OAuth bearer tokens and preserve them when calling the API.
- Advertise protected-resource discovery through a standard authentication challenge.
- Validate credentials before exposing tools, not only when a tool calls the backend.
- Use a stateless hosted transport and an explicit origin allowlist.
- Verified a production MCP transcript request, refresh rotation, and immediate revocation with an outside client.
- Keep API keys for account-owned legacy clients. Actual ChatGPT and Claude connection UI tests are still pending.

## 0.2.0 - 2026-10-26

### Added
- Remote MCP streamable HTTP transport.
- Per-request `X-API-Key` authentication with a missing-key 401 response.
- `start_video` and `get_video_status` tools for long-running jobs.
- Timestamped transcript output option.
- Timestamped search results.
- Health endpoint and configurable allowed hosts.
- Docker image supporting the hosted transport.

### Fixed
- Summary tool now creates a summary before retrieving it.
- Search tool uses the API's POST endpoint and correct response shape.
- Subtitle tool creates a missing translation before generating subtitles.
- Tools return API errors instead of silently returning generic failures.
- Transport security accepts the production reverse-proxy hostname.

### Known limitations
- At the 0.2.0 release, OAuth was not available. See the unreleased section for the implemented OAuth flow.
- No public directory submission.
- Long tool calls return a processing message after 75 seconds.
