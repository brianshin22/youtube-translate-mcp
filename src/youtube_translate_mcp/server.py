"""Remote MCP server for the YouTube Translate API.

The server is intentionally thin: API authentication, billing, jobs, and
artifacts belong to the Express API. Each MCP request supplies the caller's
YouTube Translate API key through the X-API-Key header. The environment
variable YOUTUBE_TRANSLATE_API_KEY remains supported for local stdio use.
"""

import argparse
import asyncio
import json
import logging
import os
import re
from contextvars import ContextVar
from typing import Any, Optional

import httpx
import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from mcp.server.transport_security import TransportSecuritySettings

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("youtube-translate-mcp")

API_BASE = os.getenv("YT_TRANSLATE_API_BASE", "https://api.youtubetranslate.com").rstrip("/")
PUBLIC_ORIGIN = os.getenv("PUBLIC_API_ORIGIN", "https://api.youtubetranslate.com").rstrip("/")
REQUEST_HEADER = "X-API-Key"
TOOL_TIMEOUT = float(os.getenv("TOOL_TIMEOUT_SECONDS", "75"))
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL_SECONDS", "2.5"))

_api_key: ContextVar[Optional[str]] = ContextVar("youtube_translate_api_key", default=None)

mcp = MCPServer("youtube-translate")

YOUTUBE_ID_REGEX = (
    r"(?:youtube\.com/(?:[^/\n\s]+/\S+/|(?:v|e(?:mbed)?)/|\S*?[?&]v=)|"
    r"youtu\.be/)([a-zA-Z0-9_-]{11})"
)


class APIError(RuntimeError):
    pass


def extract_youtube_id(url: str) -> str:
    match = re.search(YOUTUBE_ID_REGEX, url)
    return match.group(1) if match else ""


def current_api_key() -> str:
    key = _api_key.get() or os.getenv("YOUTUBE_TRANSLATE_API_KEY", "")
    if not key:
        raise APIError(
            "No YouTube Translate API key was supplied. Add an X-API-Key request "
            "header to the connector, or set YOUTUBE_TRANSLATE_API_KEY for stdio."
        )
    return key


async def api_request(
    method: str,
    path: str,
    *,
    params: Optional[dict[str, Any]] = None,
    json_body: Optional[dict[str, Any]] = None,
    allow_statuses: tuple[int, ...] = (),
) -> Any:
    credential = current_api_key()
    headers = {
        "Accept": "application/json",
        "User-Agent": "YouTubeTranslateMCP/2.1",
    }
    if credential.startswith("Bearer "):
        headers["Authorization"] = credential
    else:
        headers[REQUEST_HEADER] = credential
    async with httpx.AsyncClient(timeout=90.0, follow_redirects=True) as client:
        response = await client.request(
            method,
            f"{API_BASE}{path}",
            headers=headers,
            params=params,
            json=json_body,
        )
    if response.status_code >= 400 and response.status_code not in allow_statuses:
        try:
            body = response.json()
            message = body.get("message") or body.get("error", {}).get("message") or body
        except Exception:
            message = response.text[:500]
        raise APIError(f"API {response.status_code}: {message}")
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        return response.json()
    return response.text


def error_text(error: Exception) -> str:
    raise ToolError(str(error)) from error


async def submit_video(url: str, max_credit_units: Optional[int] = None) -> tuple[str, str]:
    if not extract_youtube_id(url):
        raise APIError("That does not look like a supported YouTube URL.")
    body={"url": url, **({"max_credit_units": max_credit_units} if max_credit_units is not None else {})}
    try:
        result = await api_request("POST", "/api/videos", json_body=body)
    except APIError as error:
        if "API 428" not in str(error):
            raise
        quote = await api_request("POST", "/api/processing/quote", json_body={"operation": "transcribe", "url": url})
        pricing = quote.get("quote", {})
        balance = quote.get("balance", {})
        raise APIError(
            "This video is not cached and needs an explicit spending approval. "
            f"Maximum cost is {pricing.get('maximumCreditUnits', 'unknown')} microcredits "
            f"(about ${pricing.get('maximumCreditUnits', 0) / 1_000_000:.2f}); "
            f"the account has {balance.get('available', 'unknown')} available. "
            "Ask the user to approve that maximum, then call start_video with max_credit_units. "
            "Do not guess or exceed the approved amount."
        ) from error
    video_id = result.get("id")
    if not video_id:
        raise APIError(f"Unexpected API response: {json.dumps(result)[:300]}")
    return video_id, result.get("status", "processing")


async def wait_for_video(video_id: str, deadline: float) -> dict[str, Any]:
    while True:
        result = await api_request("GET", f"/api/videos/{video_id}")
        if result.get("status") in ("completed", "error"):
            return result
        if asyncio.get_running_loop().time() >= deadline:
            return result
        await asyncio.sleep(POLL_INTERVAL)


async def wait_for_lock(
    path: str,
    deadline: float,
    *,
    success_field: str = "status",
) -> dict[str, Any]:
    while True:
        result = await api_request("GET", path)
        data = result.get("data", result)
        status = data.get(success_field) or result.get(success_field)
        if status in ("completed", "failed", "error", "not_found"):
            return data
        if asyncio.get_running_loop().time() >= deadline:
            return data
        await asyncio.sleep(POLL_INTERVAL)


def processing_note(video_id: str, status: str) -> str:
    return (
        f"Video {video_id} is still {status}. Ask the user to retry this request "
        "shortly; processing continues on the server."
    )


async def ensure_video(url: str, deadline: float, max_credit_units: Optional[int] = None) -> tuple[Optional[str], str]:
    video_id, status = await submit_video(url, max_credit_units)
    if status != "completed":
        result = await wait_for_video(video_id, deadline)
        status = result.get("status", "processing")
        if status == "error":
            raise APIError(result.get("message") or "Video processing failed.")
        if status != "completed":
            return video_id, processing_note(video_id, status)
    return video_id, ""


@mcp.tool(name="get_processing_quote", description="Get a processing credit quote and account balance without starting paid work. Obtain explicit spending approval before using a budget.")
async def get_processing_quote(url: str, operation: str = "transcribe", language: str = "en", length: str = "medium") -> str:
    """Quote one operation. No purchase, wallet access, or provider work is performed."""
    try:
        video_id = extract_youtube_id(url)
        if not video_id or operation not in ("transcribe", "translate", "summarize"):
            raise APIError("Use a valid YouTube URL and transcribe, translate, or summarize.")
        result = await api_request("POST", "/api/processing/quote", json_body={
            "operation": operation, "url": url, "video_id": video_id, "language": language, "length": length,
        })
        return json.dumps(result, indent=2)
    except Exception as error:
        return error_text(error)


@mcp.tool(name="start_video", description="Start processing a YouTube video and return its job ID without waiting.")
async def start_video(url: str, max_credit_units: Optional[int] = None) -> str:
    """Queue a YouTube URL for transcript processing."""
    try:
        video_id, status = await submit_video(url, max_credit_units)
        return json.dumps(
            {
                "video_id": video_id,
                "status": status,
                "next": "Call get_video_status and then get_transcript when completed.",
            },
            indent=2,
        )
    except Exception as error:
        return error_text(error)


@mcp.tool(name="get_video_status", description="Check processing status for a YouTube Translate video.")
async def get_video_status(video_id: str) -> str:
    """Get status using either the API job ID or the YouTube video ID."""
    try:
        result = await api_request("GET", f"/api/videos/{video_id}")
        return json.dumps(
            {
                "video_id": result.get("id"),
                "status": result.get("status"),
                "message": result.get("message"),
                "title": result.get("metadata", {}).get("title"),
                "duration": result.get("metadata", {}).get("duration"),
                "language": result.get("metadata", {}).get("language"),
            },
            indent=2,
        )
    except Exception as error:
        return error_text(error)


@mcp.tool(name="get_transcript", description="Use this when the user wants the words spoken in a YouTube video. Returns the transcript, optionally with timestamps. Cached public videos are free. For uncached videos, first call get_processing_quote and get the user's approval, then pass max_credit_units.")
async def get_transcript(url: str, include_timestamps: bool = False, max_credit_units: Optional[int] = None) -> str:
    """Return transcript text, or timestamped segments when requested."""
    try:
        deadline = asyncio.get_running_loop().time() + TOOL_TIMEOUT
        video_id, note = await ensure_video(url, deadline, max_credit_units)
        if not video_id:
            raise APIError("Video submission failed.")
        if note:
            return note
        result = await api_request("GET", f"/api/videos/{video_id}/transcript")
        transcript = result.get("transcript", {})
        text = transcript.get("text", "")
        if not text:
            raise APIError("The API returned an empty transcript.")
        if include_timestamps:
            lines = [
                f"[{float(chunk.get('timestamp', [0, 0])[0]):.1f}s] {chunk.get('text', '')}"
                for chunk in transcript.get("chunks", [])
            ]
            return "\n".join(lines) if lines else text
        return text
    except Exception as error:
        return error_text(error)


@mcp.tool(name="get_translation", description="Use this when the user wants a YouTube transcript in another language. Cached results are free. For uncached work, pass an approved max_credit_units from get_processing_quote.")
async def get_translation(url: str, language: str, max_credit_units: Optional[int] = None) -> str:
    """Translate a transcript. Use a two-letter ISO language code such as es or ko."""
    try:
        deadline = asyncio.get_running_loop().time() + TOOL_TIMEOUT
        video_id, note = await ensure_video(url, deadline, max_credit_units)
        if not video_id:
            raise APIError("Video submission failed.")
        if note:
            return note
        current = await api_request(
            "GET",
            f"/api/videos/{video_id}/translate/{language}/status",
        )
        if current.get("data", {}).get("status") != "completed":
            await api_request(
                "POST", f"/api/videos/{video_id}/translate", json_body={"language": language, **({"max_credit_units": max_credit_units} if max_credit_units is not None else {})}
            )
            status = await wait_for_lock(
                f"/api/videos/{video_id}/translate/{language}/status", deadline
            )
            if status.get("status") != "completed":
                return f"Translation is {status.get('status', 'still processing')}. Retry shortly."
        result = await api_request("GET", f"/api/videos/{video_id}/transcript/{language}")
        data = result.get("data", result)
        text = data.get("text", "")
        if not text:
            raise APIError("The API returned an empty translation.")
        return text
    except Exception as error:
        return error_text(error)


@mcp.tool(name="get_subtitles", description="Use this when the user wants an SRT or VTT subtitle file. Generates the translation first if needed. Pass max_credit_units for uncached work after approval.")
async def get_subtitles(url: str, language: str = "en", format: str = "srt", max_credit_units: Optional[int] = None) -> str:
    """Return subtitle file content. Missing translations are generated automatically."""
    try:
        if format not in ("srt", "vtt"):
            raise APIError("format must be 'srt' or 'vtt'.")
        deadline = asyncio.get_running_loop().time() + TOOL_TIMEOUT
        video_id, note = await ensure_video(url, deadline, max_credit_units)
        if not video_id:
            raise APIError("Video submission failed.")
        if note:
            return note
        translation = await api_request(
            "GET", f"/api/videos/{video_id}/translate/{language}/status"
        )
        if translation.get("data", {}).get("status") != "completed":
            await api_request(
                "POST", f"/api/videos/{video_id}/translate", json_body={"language": language, **({"max_credit_units": max_credit_units} if max_credit_units is not None else {})}
            )
            translated = await wait_for_lock(
                f"/api/videos/{video_id}/translate/{language}/status", deadline
            )
            if translated.get("status") != "completed":
                return f"Translation for {language} is still processing. Retry shortly."
        await api_request(
            "POST",
            f"/api/videos/{video_id}/subtitles",
            json_body={"language": language, "format": format},
        )
        status = await wait_for_lock(
            f"/api/videos/{video_id}/subtitles/status?language={language}&format={format}",
            deadline,
        )
        if status.get("status") != "completed":
            return f"Subtitle generation is {status.get('status', 'still processing')}. Retry shortly."
        content = await api_request(
            "GET",
            f"/api/videos/{video_id}/subtitles",
            params={"language": language, "format": format},
        )
        return str(content)
    except Exception as error:
        return error_text(error)


@mcp.tool(name="get_summary", description="Use this when the user wants a short, medium, or long summary of a YouTube video. Cached results are free. Pass max_credit_units for uncached work after approval.")
async def get_summary(url: str, language: str = "en", length: str = "medium", max_credit_units: Optional[int] = None) -> str:
    """Generate a summary, using an existing translation when needed."""
    try:
        if length not in ("short", "medium", "long"):
            raise APIError("length must be 'short', 'medium', or 'long'.")
        deadline = asyncio.get_running_loop().time() + TOOL_TIMEOUT
        video_id, note = await ensure_video(url, deadline, max_credit_units)
        if not video_id:
            raise APIError("Video submission failed.")
        if note:
            return note
        await api_request(
            "POST",
            f"/api/videos/{video_id}/summarize",
            json_body={"language": language, "length": length, **({"max_credit_units": max_credit_units} if max_credit_units is not None else {})},
        )
        status = await wait_for_lock(
            f"/api/videos/{video_id}/summarize/status?language={language}&length={length}",
            deadline,
        )
        if status.get("status") != "completed":
            return f"Summary is {status.get('status', 'still processing')}. Retry shortly."
        result = await api_request(
            "GET",
            f"/api/videos/{video_id}/summary",
            params={"language": language, "length": length},
        )
        summary = result.get("data", result).get("summary", "")
        if not summary:
            raise APIError("The API returned an empty summary.")
        return summary
    except Exception as error:
        return error_text(error)


@mcp.tool(name="search_video", description="Use this when the user wants to find a topic or quote inside a YouTube video. Returns matching passages with timestamps.")
async def search_video(url: str, query: str, language: str = "en", max_credit_units: Optional[int] = None) -> str:
    """Search with context; each result includes its position in the transcript."""
    try:
        deadline = asyncio.get_running_loop().time() + TOOL_TIMEOUT
        video_id, note = await ensure_video(url, deadline, max_credit_units)
        if not video_id:
            raise APIError("Video submission failed.")
        if note:
            return note
        result = await api_request(
            "POST",
            f"/api/videos/{video_id}/search",
            json_body={"query": query, "language": language, "context_size": 200},
        )
        matches = result.get("data", {}).get("results", [])
        if not matches:
            return f"No matches found for {query!r}."
        return "\n\n".join(
            f"Match {index + 1} at {match.get('start_time_formatted', 'unknown')}"
            f" ({float(match.get('start_time', 0)):.1f}s):\n"
            f"{match.get('context', match.get('text', ''))}"
            for index, match in enumerate(matches)
        )
    except Exception as error:
        return error_text(error)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate every request before discovery or invocation, preserving OAuth."""

    async def dispatch(self, request, call_next):
        if request.url.path == "/health":
            return JSONResponse({"status": "ok"})
        authorization = request.headers.get("Authorization", "")
        key = request.headers.get(REQUEST_HEADER)
        credential = authorization if authorization.startswith("Bearer ") else key
        challenge = {
            "WWW-Authenticate": f'Bearer resource_metadata="{PUBLIC_ORIGIN}/.well-known/oauth-protected-resource/mcp"'
        }
        if not credential:
            return JSONResponse({"error": "Connect your account with OAuth or supply X-API-Key"},
                                status_code=401, headers=challenge)
        headers = {"User-Agent": "YouTubeTranslateMCP/2.1"}
        if authorization.startswith("Bearer "):
            headers["Authorization"] = authorization
        else:
            headers[REQUEST_HEADER] = key
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(f"{API_BASE}/api/account", headers=headers)
            if response.status_code != 200:
                status = response.status_code if response.status_code in (401, 403) else 503
                return JSONResponse({"error": "Account authentication failed"}, status_code=status,
                                    headers=challenge if status == 401 else {})
        except httpx.HTTPError:
            return JSONResponse({"error": "Authentication service unavailable"}, status_code=503)
        token = _api_key.set(credential)
        try:
            return await call_next(request)
        finally:
            _api_key.reset(token)


def build_http_app():
    hosts = [
        host.strip()
        for host in os.getenv(
            "MCP_ALLOWED_HOSTS",
            "api.youtubetranslate.com,127.0.0.1:8000,127.0.0.1:8099,localhost:8000,localhost:8099",
        ).split(",")
        if host.strip()
    ]
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=[PUBLIC_ORIGIN, "https://chatgpt.com", "https://claude.ai"],
    )
    app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        transport_security=security,
    )
    app.add_middleware(APIKeyMiddleware)
    return app


async def main_async(args):
    if args.transport == "stdio":
        await mcp.run_stdio_async()
    elif args.transport == "sse":
        logger.info("Listening for SSE on %s:%s", args.host, args.port)
        await mcp.run_sse_async()
    else:
        logger.info("Listening for streamable HTTP on %s:%s/mcp", args.host, args.port)
        server = uvicorn.Server(
            uvicorn.Config(build_http_app(), host=args.host, port=args.port, log_level="info")
        )
        await server.serve()


def main():
    parser = argparse.ArgumentParser(description="YouTube Translate MCP Server")
    parser.add_argument(
        "--transport",
        default=os.getenv("MCP_TRANSPORT", "stdio"),
        choices=["stdio", "sse", "streamable-http"],
    )
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8000")))
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
