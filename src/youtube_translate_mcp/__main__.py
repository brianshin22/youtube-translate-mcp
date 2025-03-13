#!/usr/bin/env python3
"""
YouTube Translate MCP Server - Main entry point
"""

import asyncio
from .server import main as server_main

def main():
    """Main entry point for the YouTube Translate MCP server."""
    asyncio.run(server_main())
    return 0

if __name__ == "__main__":
    main()