# YouTube Translate MCP

A [Model Context Protocol (MCP)](https://github.com/anthropics/anthropic-cookbook/tree/main/model_composition_protocol) server for accessing the YouTube Translate API, allowing you to obtain transcripts, translations, and summaries of YouTube videos.

## Features

- Get transcripts of YouTube videos
- Translate transcripts to different languages
- Generate subtitles in SRT or VTT format
- Create summaries of video content
- Search for specific content within videos

## Installation

This package requires Python 3.12 or higher:

```bash
pip install youtube-translate-mcp
```

## Usage

To run the server:

```bash
# Using stdio transport (default)
YOUTUBE_TRANSLATE_API_KEY=your_api_key youtube-translate-mcp

# Using SSE transport
YOUTUBE_TRANSLATE_API_KEY=your_api_key youtube-translate-mcp --transport sse
```

## Logging

The server uses Python's built-in logging module with default configuration:
- Log level: INFO
- Format: Timestamp, logger name, level, and message
- Output: Standard output (console)

## Environment Variables

- `YOUTUBE_TRANSLATE_API_KEY`: Required. Your API key for accessing the YouTube Translate API.

## Deployment with Smithery

This package includes a `smithery.yaml` file for easy deployment with [Smithery](https://smithery.anthropic.com). 

To deploy, set the `apiKey` configuration parameter to your YouTube Translate API key.

## License

MIT