# YouTube Translate MCP
[![smithery badge](https://smithery.ai/badge/@brianshin22/youtube-translate-mcp)](https://smithery.ai/server/@brianshin22/youtube-translate-mcp)

A [Model Context Protocol (MCP)](https://github.com/anthropics/anthropic-cookbook/tree/main/model_composition_protocol) server for accessing the YouTube Translate API, allowing you to obtain transcripts, translations, and summaries of YouTube videos.

## Features

- Get transcripts of YouTube videos
- Translate transcripts to different languages
- Generate subtitles in SRT or VTT format
- Create summaries of video content
- Search for specific content within videos

## Installation

### Installing via Smithery

To install youtube-translate-mcp for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@brianshin22/youtube-translate-mcp):

```bash
npx -y @smithery/cli install @brianshin22/youtube-translate-mcp --client claude
```

### Installing Manually

This package requires Python 3.12 or higher:

```bash
# Using uv (recommended)
uv pip install youtube-translate-mcp

# Using pip
pip install youtube-translate-mcp
```

Or install from source:

```bash
# Clone the repository
git clone https://github.com/yourusername/youtube-translate-mcp.git
cd youtube-translate-mcp

# Using uv (recommended)
uv pip install -e .

# Using pip
pip install -e .
```

## Usage

To run the server:

```bash
# Using stdio transport (default)
YOUTUBE_TRANSLATE_API_KEY=your_api_key youtube-translate-mcp

# Using SSE transport
YOUTUBE_TRANSLATE_API_KEY=your_api_key youtube-translate-mcp --transport sse --port 8000
```

## Docker

You can also run the server using Docker:

```bash
# Build the Docker image
docker build -t youtube-translate-mcp .

# Run with stdio transport
docker run -e YOUTUBE_TRANSLATE_API_KEY=your_api_key youtube-translate-mcp

# Run with SSE transport
docker run -p 8000:8000 -e YOUTUBE_TRANSLATE_API_KEY=your_api_key youtube-translate-mcp --transport sse
```

## Environment Variables

- `YOUTUBE_TRANSLATE_API_KEY`: Required. Your API key for accessing the YouTube Translate API.

## Deployment with Smithery

This package includes a `smithery.yaml` file for easy deployment with [Smithery](https://smithery.anthropic.com). 

To deploy, set the `YOUTUBE_TRANSLATE_API_KEY` configuration parameter to your YouTube Translate API key.

## Development

### Prerequisites

- Python 3.12+
- Docker (optional)

### Setup

```bash
# Create and activate a virtual environment using uv (recommended)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies using uv
uv pip install -e .

# Alternatively, with standard tools
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### Testing with Claude Desktop

To test with Claude Desktop (macOS/Windows only), you'll need to add your server to the Claude Desktop configuration file located at `~/Library/Application Support/Claude/claude_desktop_config.json`.

#### Method 1: Local Development

Use this method if you want to test your local development version:

```json
{
    "mcpServers": {
        "youtube-translate": {
            "command": "uv",
            "args": [
                "--directory",
                "/ABSOLUTE/PATH/TO/youtube-translate-mcp",
                "run",
                "-m", "youtube_translate_mcp"
            ],
            "env": {
              "YOUTUBE_TRANSLATE_API_KEY": "YOUR_API_KEY"
            }
        }
    }
}
```

Make sure to replace `/ABSOLUTE/PATH/TO/youtube-translate-mcp` with the actual path to your project directory.

#### Method 2: Docker-based Testing

If you prefer to test using Docker (recommended for more reproducible testing):

```json
{
  "mcpServers": {
    "youtube-translate": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-e",
        "YOUTUBE_TRANSLATE_API_KEY",
        "youtube-translate-mcp"
      ],
      "env": {
        "YOUTUBE_TRANSLATE_API_KEY": "YOUR_API_KEY"
      }
    }
  }
}
```

Replace `YOUR_API_KEY` with your actual YouTube Translate API key.

For more information on using MCP servers with Claude Desktop, see the [MCP documentation](https://modelcontextprotocol.io/quickstart/server).

### Debugging
 - The normal MCP Inspector has a built in timeout for MCP tool calls, which is generally too short for these video processing calls (as of March 13, 2025). Better to use Claude Desktop and look at the MCP logs from Claude at ~/Library/Logs/Claude/mcp-server-{asfasf}.log.
 - Can do tail -f {log-file}.log to follow as you interact with Claude.

## License

MIT
