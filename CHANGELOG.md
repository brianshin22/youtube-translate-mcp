# Changelog

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
- No OAuth. Custom connectors must send an API key header.
- No public directory submission.
- Long tool calls return a processing message after 75 seconds.
