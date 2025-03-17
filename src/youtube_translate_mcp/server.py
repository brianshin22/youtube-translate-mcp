import asyncio
import httpx
import json
import re
import os
import time
import logging
import argparse
from typing import Any, Dict, List, Optional, Tuple
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.resources import TextResource, ResourceTemplate
from mcp.types import Resource as MCPResource, ResourceTemplate as MCPResourceTemplate

# Initialize FastMCP server
mcp_server = FastMCP("youtube-translate")

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('youtube-translate-mcp')

# Global constants
YT_TRANSLATE_API_BASE = "https://api.youtubetranslate.com"
YOUTUBE_TRANSLATE_API_KEY = os.environ.get("YOUTUBE_TRANSLATE_API_KEY", "")

if not YOUTUBE_TRANSLATE_API_KEY:
    logger.warning("YOUTUBE_TRANSLATE_API_KEY environment variable not set!")

# Add a health check endpoint for SSE transport mode
async def add_health_check(app):
    @app.get("/health")
    async def health_check():
        from fastapi import Response
        return Response(content=json.dumps({"status": "ok"}), media_type="application/json")

# Register this function to be called when the FastAPI app is created
try:
    # Try the newer method first
    mcp_server.on_startup(add_health_check)
except AttributeError:
    try:
        # Fall back to the older method
        mcp_server.on_app_init(add_health_check)
    except AttributeError:
        logger.warning("Could not register health check endpoint - MCP version incompatibility")

# Constants
USER_AGENT = "YouTubeTranslateMCP/1.0"

# Regular expression to extract YouTube video ID from various URL formats
YOUTUBE_ID_REGEX = r'(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'

def extract_youtube_id(url: str) -> str:
    """Extract YouTube video ID from a URL.
    
    Args:
        url: YouTube URL
        
    Returns:
        YouTube video ID or empty string if no match
    """
    match = re.search(YOUTUBE_ID_REGEX, url)
    if match:
        return match.group(1)
    return ""

async def make_yt_api_request(endpoint: str, method: str = "GET", params: dict = None, json_data: dict = None) -> dict[str, Any] | str | None:
    """Make a request to the YouTube Translate API with proper error handling."""
    headers = {
        "X-API-Key": YOUTUBE_TRANSLATE_API_KEY,
        "Content-Type": "application/json"
    }
    
    url = f"{YT_TRANSLATE_API_BASE}{endpoint}"
    
    logger.info(f"Making API request: {method} {url}")
    if params:
        logger.info(f"Request params: {params}")
    if json_data:
        logger.info(f"Request data: {json_data}")
    
    async with httpx.AsyncClient() as client:
        try:
            if method.upper() == "GET":
                response = await client.get(url, headers=headers, params=params, timeout=30.0)
            elif method.upper() == "POST":
                response = await client.post(url, headers=headers, params=params, json=json_data, timeout=30.0)
            else:
                logger.error(f"ERROR: Invalid HTTP method: {method}")
                return None
                
            response.raise_for_status()
            
            logger.info(f"API response status: {response.status_code}")
            
            # If the endpoint is for subtitles, directly return the text content
            if "/subtitles" in endpoint:
                return response.text
            
            # For all other endpoints, return the JSON response
            return response.json()
        except Exception as e:
            logger.error(f"API request error: {str(e)}")
            return None

async def process_video(url: str) -> tuple[bool, str, str]:
    """Helper function to submit a video for processing and wait for completion.
    
    This function now tries to optimize API calls by:
    1. Extracting YouTube ID from URL when possible
    2. Checking if video is already processed using YouTube ID directly
    3. Only submitting for processing if needed
    
    Args:
        url: The YouTube video URL
        
    Returns:
        A tuple of (success, video_id, error_message)
    """
    try:
        # Step 1: Try to extract YouTube ID from URL
        youtube_id = extract_youtube_id(url)
        video_id = ""
        
        if youtube_id:
            logger.info(f"Extracted YouTube ID: {youtube_id} from URL: {url}")
            
            # Step 2: Check if video has already been processed using YouTube ID directly
            status_response = await make_yt_api_request(f"/api/videos/{youtube_id}")
            
            if status_response and "status" in status_response:
                video_id = youtube_id
                logger.info(f"Found existing video with YouTube ID: {youtube_id}, status: {status_response.get('status')}")
                
                # If video is already processed or processing, we can use this ID
                if status_response.get("status") == "completed":
                    logger.info(f"Video already processed, using YouTube ID: {youtube_id}")
                    return True, youtube_id, ""
                elif status_response.get("status") == "processing":
                    # Need to wait for processing to complete
                    logger.info(f"Video already processing, waiting for completion: {youtube_id}")
                    # Continue to polling step below with the YouTube ID
                    video_id = youtube_id
                elif status_response.get("status") == "error":
                    error_message = status_response.get("message", "Unknown error occurred")
                    logger.error(f"Error with video: {error_message}")
                    return False, youtube_id, f"Error processing video: {error_message}"
        
        # Step 3: Submit video for processing if needed (if we don't have a video_id yet)
        if not video_id:
            logger.info(f"Submitting video for processing: {url}")
            
            submit_response = await make_yt_api_request("/api/videos", method="POST", json_data={"url": url})
            
            if not submit_response or "id" not in submit_response:
                logger.error("Failed to submit video for processing")
                return False, "", "Failed to submit video for processing."
            
            video_id = submit_response["id"]
            logger.info(f"Video submitted, received ID: {video_id}")
            await asyncio.sleep(1) # wait for 1 second before polling
        
        # Step 4: Poll for video processing status until it's complete
        max_attempts = 10
        attempts = 0
        
        while attempts < max_attempts:
            logger.info(f"Checking video status, attempt {attempts+1}/{max_attempts}")
            
            status_response = await make_yt_api_request(f"/api/videos/{video_id}")
            
            if not status_response:
                logger.error("Failed to retrieve video status")
                return False, video_id, "Failed to retrieve video status."
            
            status = status_response.get("status")
            logger.info(f"Video status: {status}")
                
            if status == "completed":
                logger.info(f"Video processing completed for ID: {video_id}")
                return True, video_id, ""
                
            if status == "error":
                error_message = status_response.get("message", "Unknown error occurred")
                logger.error(f"Error processing video: {error_message}")
                return False, video_id, f"Error processing video: {error_message}"
            
            # Calculate backoff delay
            delay = await calculate_backoff_delay(attempts)
            logger.info(f"Waiting {delay:.1f}s before checking video status again, attempt {attempts+1}/{max_attempts}")
            
            await asyncio.sleep(delay)
            attempts += 1
        
        logger.error("Video processing timeout - too many attempts")
        return False, video_id, "Video processing timed out. Please try again later."
        
    except Exception as e:
        logger.error(f"Exception during video processing: {str(e)}")
        return False, "", f"An error occurred: {str(e)}"

async def calculate_backoff_delay(attempt: int, base_delay: float = 1.0, multiplier: float = 1.5, max_delay: float = 20.0) -> float:
    """Calculate a progressive backoff delay.
    
    Args:
        attempt: The current attempt number (0-based)
        base_delay: The initial delay in seconds
        multiplier: How much to increase the delay each time
        max_delay: Maximum delay in seconds
        
    Returns:
        The delay in seconds for the current attempt
    """
    delay = min(base_delay * (multiplier ** attempt), max_delay)
    return delay

# MCP Tool Functions

@mcp_server.tool(name="get_transcript", description="Get the transcript of a YouTube video")
async def get_transcript(url: str) -> str:
    """Get the transcript of a YouTube video.

    This tool processes a video and retrieves its transcript. It can efficiently
    handle YouTube URLs by extracting the video ID and checking if it's already
    been processed before submitting a new request.

    Args:
        url: The YouTube video URL
        
    Returns:
        The video transcript as text
    """
    logger.info(f"Getting transcript for URL: {url}")
    
    # Process the video to ensure it's ready
    success, video_id, error_message = await process_video(url)
    
    if not success:
        logger.error(f"Failed to process video: {error_message}")
        return f"Error: {error_message}"
    
    # Get the transcript from the API
    transcript_response = await make_yt_api_request(f"/api/videos/{video_id}/transcript")
    
    if not transcript_response:
        error_msg = "Failed to retrieve transcript."
        logger.error(error_msg)
        return f"Error: {error_msg}"
    
    # Check if the response is a string or a JSON object
    if isinstance(transcript_response, str):
        return transcript_response
    elif isinstance(transcript_response, dict) and "transcript" in transcript_response:
        return transcript_response["transcript"]["text"]
    else:
        error_msg = "Unexpected response format from API."
        logger.error(error_msg)
        return f"Error: {error_msg}"

@mcp_server.tool(name="get_translation", description="Get a translated transcript of a YouTube video")
async def get_translation(url: str, language: str) -> str:
    """Get a translated transcript of a YouTube video.

    This tool processes a video and translates its transcript to the specified language.
    It optimizes API calls by first checking if the translation already exists before
    processing the video.

    Args:
        url: The YouTube video URL
        language: Target language code (e.g., "en", "fr", "es")
        
    Returns:
        The translated transcript as text
    """
    logger.info(f"Getting translation for URL: {url} to language: {language}")
    
    # Process the video to ensure it's ready
    success, video_id, error_message = await process_video(url)
    
    if not success:
        logger.error(f"Failed to process video: {error_message}")
        return f"Error: {error_message}"
    
    # First try to get the transcript to make sure it exists
    logger.info(f"Retrieving transcript before requesting translation")
    transcript_response = await make_yt_api_request(f"/api/videos/{video_id}/transcript")
    
    if not transcript_response:
        error_msg = "Failed to retrieve transcript. Cannot translate without a transcript."
        logger.error(error_msg)
        return f"Error: {error_msg}"
    
    # Try to fetch the translated transcript directly first (it might already exist)
    logger.info(f"Checking if translation for language {language} already exists")
    translation_response = await make_yt_api_request(f"/api/videos/{video_id}/transcript/{language}")
    
    # If we got a valid response, return it
    if translation_response and isinstance(translation_response, dict) and "status" in translation_response and translation_response["status"] == "success":
        logger.info(f"Found existing translation for language {language}")
        if "data" in translation_response and "text" in translation_response["data"]:
            return translation_response["data"]["text"]
        elif "data" in translation_response:
            # Return whatever data we have
            return json.dumps(translation_response["data"], indent=2)
    
    # If we don't have a translation yet, request one
    logger.info(f"Requesting new translation for language {language}")
    request_translation = await make_yt_api_request(
        f"/api/videos/{video_id}/translate", 
        method="POST", 
        json_data={"language": language}
    )
    
    if not request_translation:
        error_msg = f"Failed to request translation for language: {language}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
    
    # Check the status of the translation request
    translation_status = None
    max_attempts = 10
    attempts = 0
    
    while attempts < max_attempts:
        logger.info(f"Checking translation status, attempt {attempts+1}/{max_attempts}")
        
        status_response = await make_yt_api_request(
            f"/api/videos/{video_id}/translate/{language}/status"
        )
        
        if not status_response:
            logger.error("Failed to retrieve translation status")
            return "Error: Failed to retrieve translation status."
        
        if isinstance(status_response, dict) and "status" in status_response and "data" in status_response:
            translation_status = status_response["data"].get("status")
            logger.info(f"Translation status: {translation_status}")
            
            if translation_status == "completed":
                logger.info(f"Translation completed for language: {language}")
                break
                
            if translation_status == "error":
                error_message = status_response.get("message", "Unknown error occurred")
                logger.error(f"Error translating: {error_message}")
                return f"Error: Translation failed: {error_message}"
        
        # Calculate backoff delay
        delay = await calculate_backoff_delay(attempts)
        logger.info(f"Waiting {delay:.1f}s before checking translation status again")
        
        await asyncio.sleep(delay)
        attempts += 1
    
    if attempts >= max_attempts and translation_status != "completed":
        logger.error("Translation timed out - too many attempts")
        return "Error: Translation timed out. Please try again later."
    
    # Get the translated transcript
    logger.info(f"Retrieving translated transcript for language: {language}")
    translated_transcript = await make_yt_api_request(f"/api/videos/{video_id}/transcript/{language}")
    
    if not translated_transcript:
        error_msg = f"Failed to retrieve translated transcript for language: {language}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
    
    # Format the response based on what we received
    if isinstance(translated_transcript, str):
        return translated_transcript
    elif isinstance(translated_transcript, dict):
        if "data" in translated_transcript and "text" in translated_transcript["data"]:
            return translated_transcript["data"]["text"]
        elif "text" in translated_transcript:
            return translated_transcript["text"]
        else:
            # Return the complete JSON response if we can't extract specific fields
            return json.dumps(translated_transcript, indent=2)
    else:
        error_msg = "Unexpected response format from API."
        logger.error(error_msg)
        return f"Error: {error_msg}"

@mcp_server.tool(name="get_subtitles", description="Generate subtitle files for a YouTube video")
async def get_subtitles(url: str, language: str = "en", format: str = "srt") -> str:
    """Generate subtitle files for a YouTube video.

    This tool processes a video and generates subtitle files in the specified format and language.
    It first checks if the subtitles already exist before processing the video to optimize
    performance. If the requested language is not available, it automatically requests a 
    translation first.

    Args:
        url: The YouTube video URL
        language: Language code for subtitles (e.g., "en", "fr", "es")
        format: Subtitle format, either "srt" or "vtt" (default: "srt")
        
    Returns:
        The subtitles content as text
    """
    logger.info(f"Getting subtitles for URL: {url}, language: {language}, format: {format}")
    
    # Validate format
    if format not in ["srt", "vtt"]:
        error_msg = f"Invalid format: {format}. Must be 'srt' or 'vtt'."
        logger.error(error_msg)
        return f"Error: {error_msg}"
    
    # Process the video to ensure it's ready
    success, video_id, error_message = await process_video(url)
    
    if not success:
        logger.error(f"Failed to process video: {error_message}")
        return f"Error: {error_message}"
    
    # Get the subtitles from the API
    subtitles_response = await make_yt_api_request(
        f"/api/videos/{video_id}/subtitles",
        params={"language": language, "format": format}
    )
    
    if not subtitles_response:
        error_msg = f"Failed to retrieve subtitles for language: {language}, format: {format}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
    
    # The response could be the subtitles as text or a JSON object with an error
    if isinstance(subtitles_response, dict) and "error" in subtitles_response:
        error_msg = subtitles_response["error"]
        logger.error(f"API error: {error_msg}")
        return f"Error: {error_msg}"
    
    return subtitles_response

@mcp_server.tool(name="get_summary", description="Generate a summary of a YouTube video")
async def get_summary(url: str, language: str = "en", length: str = "medium") -> str:
    """Generate a summary of a YouTube video.

    This tool processes a video and generates a summary of its content in the specified language.
    It properly handles "processing" states by polling until completion rather than failing immediately.
    If the requested language is not available, it automatically requests a translation first.

    Args:
        url: The YouTube video URL
        language: Language code for the summary (e.g., "en", "fr")
        length: Length of the summary ("short", "medium", or "long")
        
    Returns:
        A summary of the video content
    """
    logger.info(f"Getting summary for URL: {url}, language: {language}, length: {length}")
    
    # Validate length
    if length not in ["short", "medium", "long"]:
        error_msg = f"Invalid length: {length}. Must be 'short', 'medium', or 'long'."
        logger.error(error_msg)
        return f"Error: {error_msg}"
    
    # Process the video to ensure it's ready
    success, video_id, error_message = await process_video(url)
    
    if not success:
        logger.error(f"Failed to process video: {error_message}")
        return f"Error: {error_message}"
    
    # Get the summary from the API
    summary_response = await make_yt_api_request(
        f"/api/videos/{video_id}/summary",
        params={"language": language, "length": length}
    )
    
    if not summary_response:
        error_msg = f"Failed to retrieve summary for language: {language}, length: {length}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
    
    # Check if the response is a JSON object with the summary
    if isinstance(summary_response, dict) and "summary" in summary_response:
        return summary_response["summary"]
    elif isinstance(summary_response, dict) and "error" in summary_response:
        error_msg = summary_response["error"]
        logger.error(f"API error: {error_msg}")
        return f"Error: {error_msg}"
    else:
        error_msg = "Unexpected response format from API."
        logger.error(error_msg)
        return f"Error: {error_msg}"

@mcp_server.tool(name="search_video", description="Search for specific content within a YouTube video's transcript")
async def search_video(url: str, query: str) -> str:
    """Search for specific content within a YouTube video's transcript.

    This tool processes a video and searches for specific terms or phrases within its transcript.
    It properly handles "processing" states by polling until completion rather than failing immediately.

    Args:
        url: The YouTube video URL
        query: The search term or phrase to look for
        
    Returns:
        Search results with context from the video transcript
    """
    logger.info(f"Searching video for URL: {url}, query: {query}")
    
    # Process the video to ensure it's ready
    success, video_id, error_message = await process_video(url)
    
    if not success:
        logger.error(f"Failed to process video: {error_message}")
        return f"Error: {error_message}"
    
    # Search the video transcript
    search_response = await make_yt_api_request(
        f"/api/videos/{video_id}/search",
        params={"query": query}
    )
    
    if not search_response:
        error_msg = f"Failed to search video for query: {query}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
    
    # Format the search results
    if isinstance(search_response, dict) and "matches" in search_response:
        matches = search_response["matches"]
        if not matches:
            return f"No matches found for '{query}' in the video."
        
        result = f"Found {len(matches)} matches for '{query}':\n\n"
        for i, match in enumerate(matches, 1):
            timestamp = match.get("timestamp", "Unknown")
            text = match.get("text", "")
            timestamp_str = f"{int(timestamp) // 60}:{int(timestamp) % 60:02d}" if isinstance(timestamp, (int, float)) else timestamp
            result += f"{i}. [{timestamp_str}] {text}\n\n"
        
        return result
    elif isinstance(search_response, dict) and "error" in search_response:
        error_msg = search_response["error"]
        logger.error(f"API error: {error_msg}")
        return f"Error: {error_msg}"
    else:
        error_msg = "Unexpected response format from API."
        logger.error(error_msg)
        return f"Error: {error_msg}"

# Main function to run the server
async def main():
    parser = argparse.ArgumentParser(description="YouTube Translate MCP Server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"], 
                      help="Transport protocol to use (stdio or sse)")
    parser.add_argument("--port", type=int, default=8000, 
                      help="Port to use for SSE transport (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0", 
                      help="Host to bind for SSE transport (default: 0.0.0.0)")
    
    args = parser.parse_args()
    
    logger.info(f"Starting YouTube Translate MCP server with {args.transport} transport")
    
    if args.transport == "stdio":
        await mcp_server.run_stdio_async()
    else:  # SSE
        logger.info(f"Listening on {args.host}:{args.port}")
        await mcp_server.run_sse_async()

if __name__ == "__main__":
    asyncio.run(main()) 