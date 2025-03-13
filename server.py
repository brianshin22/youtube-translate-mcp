import asyncio
import httpx
import json
import re
import os
import time
import logging
import logging.handlers
from typing import Any, Dict, List, Optional, Tuple
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.resources import TextResource, ResourceTemplate
from mcp.types import Resource as MCPResource, ResourceTemplate as MCPResourceTemplate

# Set up logging
log_dir = '/Users/brian/ssd/projects/youtube-translate-mcp/logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'youtube-translate-mcp-server.log')

# Configure logger
logger2 = logging.getLogger('youtube-translate-mcp')
logger2.setLevel(logging.INFO)

# Create file handler
file_handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=10*1024*1024, backupCount=5
)
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('logger2: %(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger2.addHandler(file_handler)
logger2.addHandler(console_handler)

# MCP server setup
try:
    import mcp
except ImportError:
    logger2.error("Failed to import MCP library. Make sure it's installed.")
    raise

# Global constants
YT_TRANSLATE_API_BASE = os.environ.get("YT_TRANSLATE_API_BASE", "https://api.youtubetranslate.com")
YOUTUBE_TRANSLATE_API_KEY = os.environ.get("YOUTUBE_TRANSLATE_API_KEY", "")

if not YOUTUBE_TRANSLATE_API_KEY:
    logger2.warning("YOUTUBE_TRANSLATE_API_KEY environment variable not set!")

# Initialize FastMCP server
mcp = FastMCP("youtube-translate")

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
    
    logger2.info(f"Making API request: {method} {url}")
    if params:
        logger2.info(f"Request params: {params}")
    if json_data:
        logger2.info(f"Request data: {json_data}")
    
    async with httpx.AsyncClient() as client:
        try:
            if method.upper() == "GET":
                response = await client.get(url, headers=headers, params=params, timeout=30.0)
            elif method.upper() == "POST":
                response = await client.post(url, headers=headers, params=params, json=json_data, timeout=30.0)
            else:
                logger2.error(f"Invalid HTTP method: {method}")
                return None
                
            response.raise_for_status()
            
            logger2.info(f"API response status: {response.status_code}")
            
            # Check if the endpoint is for content retrieval (not status check)
            # For certain endpoints like subtitles, the content may be plain text
            # For others like summaries, the content should still be JSON
            if "/subtitles?" in endpoint and not "/status?" in endpoint:
                try:
                    # First try to parse as JSON
                    return response.json()
                except Exception as e:
                    # If JSON parsing fails, the response might be raw text
                    logger2.info(f"Received non-JSON response, treating as raw text: {str(e)}")
                    return response.text
            else:
                # All other endpoints expect JSON responses
                return response.json()
        except Exception as e:
            logger2.error(f"API request error: {str(e)}")
            print(f"API request error: {str(e)}")
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
            logger2.info(f"Extracted YouTube ID: {youtube_id} from URL: {url}")
            
            # Step 2: Check if video has already been processed using YouTube ID directly
            status_response = await make_yt_api_request(f"/api/videos/{youtube_id}")
            
            if status_response and "status" in status_response:
                video_id = youtube_id
                logger2.info(f"Found existing video with YouTube ID: {youtube_id}, status: {status_response.get('status')}")
                
                # If video is already processed or processing, we can use this ID
                if status_response.get("status") == "completed":
                    logger2.info(f"Video already processed, using YouTube ID: {youtube_id}")
                    return True, youtube_id, ""
                elif status_response.get("status") == "processing":
                    # Need to wait for processing to complete
                    logger2.info(f"Video already processing, waiting for completion: {youtube_id}")
                    # Continue to polling step below with the YouTube ID
                    video_id = youtube_id
                elif status_response.get("status") == "error":
                    error_message = status_response.get("message", "Unknown error occurred")
                    logger2.error(f"Error with video: {error_message}")
                    return False, youtube_id, f"Error processing video: {error_message}"
        
        # Step 3: Submit video for processing if needed (if we don't have a video_id yet)
        if not video_id:
            logger2.info(f"Submitting video for processing: {url}")
            
            submit_response = await make_yt_api_request("/api/videos", method="POST", json_data={"url": url})
            
            if not submit_response or "id" not in submit_response:
                logger2.error("Failed to submit video for processing")
                return False, "", "Failed to submit video for processing."
            
            video_id = submit_response["id"]
            logger2.info(f"Video submitted, received ID: {video_id}")
            asyncio.sleep(1) # wait for 1 second before polling
        
        # Step 4: Poll for video processing status until it's complete
        max_attempts = 10
        attempts = 0
        
        while attempts < max_attempts:
            logger2.info(f"Checking video status, attempt {attempts+1}/{max_attempts}")
            
            status_response = await make_yt_api_request(f"/api/videos/{video_id}")
            
            if not status_response:
                logger2.error("Failed to retrieve video status")
                return False, video_id, "Failed to retrieve video status."
            
            status = status_response.get("status")
            logger2.info(f"Video status: {status}")
                
            if status == "completed":
                logger2.info(f"Video processing completed for ID: {video_id}")
                return True, video_id, ""
                
            if status == "error":
                error_message = status_response.get("message", "Unknown error occurred")
                logger2.error(f"Error processing video: {error_message}")
                return False, video_id, f"Error processing video: {error_message}"
            
            # Calculate backoff delay
            delay = await calculate_backoff_delay(attempts)
            logger2.info(f"Waiting {delay:.1f}s before checking video status again, attempt {attempts+1}/{max_attempts}")
            
            await asyncio.sleep(delay)
            attempts += 1
        
        logger2.error("Video processing timeout - too many attempts")
        return False, video_id, "Video processing timed out. Please try again later."
        
    except Exception as e:
        logger2.error(f"Exception during video processing: {str(e)}")
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

@mcp.tool()
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
    try:
        # Extract YouTube ID if possible for direct access
        youtube_id = extract_youtube_id(url)
        video_id = youtube_id if youtube_id else url
        
        logger2.info(f"Attempting to retrieve transcript directly for: {url}")
        
        # First check if video exists and what state it's in
        video_status = await make_yt_api_request(f"/api/videos/{video_id}")
        
        # If video exists but is still processing, poll until complete
        if video_status and video_status.get("status") == "processing":
            logger2.info(f"Video is still processing, waiting for completion")
            
            # Poll for video processing status until it's complete
            max_attempts = 10
            attempts = 0
            
            while attempts < max_attempts:
                # Calculate progressive backoff delay
                delay = await calculate_backoff_delay(attempts)
                logger2.info(f"Waiting {delay:.1f}s before checking video status, attempt {attempts+1}/{max_attempts}")
                await asyncio.sleep(delay)
                
                video_status = await make_yt_api_request(f"/api/videos/{video_id}")
                attempts += 1
                
                if not video_status:
                    logger2.info("Failed to get video status, will retry")
                    continue
                
                status = video_status.get("status")
                logger2.info(f"Video status: {status}")
                
                if status == "completed":
                    logger2.info(f"Video processing completed")
                    break
                    
                if status == "error":
                    logger2.error(f"Error processing video")
                    break
        
        # If video doesn't exist or couldn't be processed, submit it
        if not video_status or video_status.get("status") not in ["completed"]:
            logger2.info(f"Submitting video for processing: {url}")
            success, video_id, error_message = await process_video(url)
            
            if not success:
                return error_message
        
        # Try to get the transcript with polling
        max_attempts = 10
        attempts = 0
        
        while attempts < max_attempts:
            # Calculate progressive backoff delay if not the first attempt
            if attempts > 0:
                delay = await calculate_backoff_delay(attempts - 1)
                logger2.info(f"Waiting {delay:.1f}s before retrying transcript retrieval, attempt {attempts+1}/{max_attempts}")
                await asyncio.sleep(delay)
            
            # Get the transcript
            logger2.info(f"Retrieving transcript for video ID: {video_id}")
            transcript_response = await make_yt_api_request(f"/api/videos/{video_id}/transcript")
            
            # If we have a complete transcript, process it
            if transcript_response:
                if transcript_response.get("metadata").get("status") == "completed":
                    # Return the full transcript as text
                    logger2.info("Successfully retrieved transcript")
                    return transcript_response["transcript"]["text"]
            
            # If we didn't get a proper response, retry
            attempts += 1
            if attempts >= max_attempts:
                return "Failed to retrieve transcript after multiple attempts."
        
        return "Failed to retrieve transcript within the allowed time."
        
    except Exception as e:
        logger2.error(f"Exception during transcript retrieval: {str(e)}")
        return f"An error occurred while retrieving transcript: {str(e)}"

@mcp.tool()
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
    try:
        # Extract YouTube ID if possible for direct access
        youtube_id = extract_youtube_id(url)
        video_id = youtube_id if youtube_id else url
        
        logger2.info(f"Attempting to retrieve translation directly for: {url}, language: {language}")
        
        # First check if video exists and what state it's in
        video_status = await make_yt_api_request(f"/api/videos/{video_id}")
        
        # If video exists but is still processing, poll until complete
        if video_status and video_status.get("status") == "processing":
            logger2.info(f"Video is still processing, waiting for completion")
            
            # Poll for video processing status until it's complete
            max_attempts = 10
            attempts = 0
            
            while attempts < max_attempts:
                # Calculate progressive backoff delay
                delay = await calculate_backoff_delay(attempts)
                logger2.info(f"Waiting {delay:.1f}s before checking video status, attempt {attempts+1}/{max_attempts}")
                await asyncio.sleep(delay)
                
                video_status = await make_yt_api_request(f"/api/videos/{video_id}")
                attempts += 1
                
                if not video_status:
                    logger2.info("Failed to get video status, will retry")
                    continue
                
                status = video_status.get("status")
                logger2.info(f"Video status: {status}")
                
                if status == "completed":
                    logger2.info(f"Video processing completed")
                    break
                    
                if status == "error":
                    logger2.error(f"Error processing video")
                    break
        
        # If video doesn't exist or couldn't be processed, submit it
        if not video_status or video_status.get("status") not in ["completed"]:
            logger2.info(f"Submitting video for processing: {url}")
            success, video_id, error_message = await process_video(url)
            
            if not success:
                logger2.error(f"Video processing failed: {error_message}")
                return error_message
        
        # Check if translation exists
        logger2.info(f"Checking if translation exists for: {video_id}, language: {language}")
        translation_status = await make_yt_api_request(
            f"/api/videos/{video_id}/translate/{language}/status"
        )
        
        # If translation doesn't exist or is not completed, request it
        if not translation_status or translation_status.get("data", {}).get("status") != "completed":
            logger2.info(f"Requesting translation for video ID: {video_id}, language: {language}")
            translate_response = await make_yt_api_request(
                f"/api/videos/{video_id}/translate", 
                method="POST", 
                json_data={"language": language}
            )
            
            if not translate_response:
                logger2.error(f"Failed to request translation to {language}")
                return f"Failed to request translation to {language}."
            
            # Poll for translation status until it's complete
            max_attempts = 15  # Translations can take longer, so we give more attempts
            attempts = 0
            
            while attempts < max_attempts:
                # Calculate progressive backoff delay
                delay = await calculate_backoff_delay(attempts, base_delay=2.0, multiplier=1.5)
                logger2.info(f"Waiting {delay:.1f}s before checking translation status, attempt {attempts+1}/{max_attempts}")
                await asyncio.sleep(delay)
                
                translation_status = await make_yt_api_request(
                    f"/api/videos/{video_id}/translate/{language}/status"
                )
                attempts += 1
                
                if not translation_status:
                    logger2.info("Failed to check translation status, will retry")
                    continue
                
                status = translation_status.get("data", {}).get("status")
                logger2.info(f"Translation status: {status}")
                
                if status == "completed":
                    logger2.info(f"Translation completed")
                    break
                    
                if status == "error":
                    logger2.error("Error during translation")
                    return f"Error during translation."
                
                if status == "processing":
                    logger2.info("Translation is still being processed, continuing to wait")
                
            if attempts >= max_attempts:
                logger2.error("Translation processing timed out")
                return "Translation processing timed out. Try again later."
        else:
            logger2.info(f"Translation already exists")
        
        # Attempt to retrieve the translated transcript with retry logic
        max_attempts = 5
        attempts = 0
        
        while attempts < max_attempts:
            # Calculate progressive backoff delay if not the first attempt
            if attempts > 0:
                delay = await calculate_backoff_delay(attempts - 1)
                logger2.info(f"Waiting {delay:.1f}s before retrying translation retrieval, attempt {attempts+1}/{max_attempts}")
                await asyncio.sleep(delay)
            
            # Get the translated transcript
            logger2.info(f"Retrieving translated transcript for video ID: {video_id}, language: {language}")
            translated_transcript = await make_yt_api_request(
                f"/api/videos/{video_id}/transcript/{language}"
            )
            
            if translated_transcript and "data" in translated_transcript:
                # Format the translation nicely
                logger2.info("Formatting translation response")
                if "text" in translated_transcript["data"]:
                    return translated_transcript["data"]["text"]
                elif "chunks" in translated_transcript["data"]:
                    # Join the text from all chunks
                    chunks_text = [chunk["text"] for chunk in translated_transcript["data"]["chunks"]]
                    return "\n".join(chunks_text)
                else:
                    logger2.error("Translation format not recognized")
                    return "Translation format not recognized."
            
            # If we didn't get a proper response, retry
            attempts += 1
            if attempts >= max_attempts:
                return f"Failed to retrieve {language} translation after multiple attempts."
        
        return f"Failed to retrieve {language} translation within the allowed time."
            
    except Exception as e:
        logger2.error(f"Exception during translation: {str(e)}")
        return f"An error occurred while processing translation: {str(e)}"

@mcp.tool()
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
    try:
        # Validate format parameter
        logger2.info(f"Generating subtitles for: {url}, language: {language}, format: {format}")
        if format.lower() not in ["srt", "vtt"]:
            logger2.error(f"Invalid subtitle format: {format}")
            return "Invalid subtitle format. Please use 'srt' or 'vtt'."
        
        format = format.lower()  # Normalize to lowercase
        
        # Extract YouTube ID if possible for direct access
        youtube_id = extract_youtube_id(url)
        video_id = youtube_id if youtube_id else url
        
        # First check if video exists and what state it's in
        video_status = await make_yt_api_request(f"/api/videos/{video_id}")
        
        # If video exists but is still processing, poll until complete
        if video_status and video_status.get("status") == "processing":
            logger2.info(f"Video is still processing, waiting for completion")
            
            # Poll for video processing status until it's complete
            max_attempts = 10
            attempts = 0
            
            while attempts < max_attempts:
                # Calculate progressive backoff delay
                delay = await calculate_backoff_delay(attempts)
                logger2.info(f"Waiting {delay:.1f}s before checking video status, attempt {attempts+1}/{max_attempts}")
                await asyncio.sleep(delay)
                
                video_status = await make_yt_api_request(f"/api/videos/{video_id}")
                attempts += 1
                
                if not video_status:
                    logger2.info("Failed to get video status, will retry")
                    continue
                
                status = video_status.get("status")
                logger2.info(f"Video status: {status}")
                
                if status == "completed":
                    logger2.info(f"Video processing completed")
                    break
                    
                if status == "error":
                    logger2.error(f"Error processing video")
                    break
        
        # If video doesn't exist or couldn't be processed, submit it
        if not video_status or video_status.get("status") not in ["completed"]:
            logger2.info(f"Submitting video for processing: {url}")
            success, video_id, error_message = await process_video(url)
            
            if not success:
                logger2.error(f"Video processing failed: {error_message}")
                return error_message
        
        # First, check if translation is needed for non-English languages
        translation_needed = False
        if language.lower() != "en":
            # Check if the translation exists
            logger2.info(f"Checking if translation exists for: {video_id}, language: {language}")
            translation_status = await make_yt_api_request(
                f"/api/videos/{video_id}/translate/{language}/status"
            )
            
            # If no translation or it's not completed, we need to request it
            if not translation_status or translation_status.get("data", {}).get("status") != "completed":
                translation_needed = True
                logger2.info(f"Translation needed for {language}, will request it first")
                
                # Request translation
                translation_request = await make_yt_api_request(
                    f"/api/videos/{video_id}/translate", 
                    method="POST", 
                    json_data={"language": language}
                )
                
                if not translation_request:
                    logger2.error(f"Failed to request translation to {language}")
                    return f"Failed to request translation to {language}, which is needed for subtitles."
                
                # Poll for translation completion
                max_attempts = 15  # Translations can take time
                attempts = 0
                
                while attempts < max_attempts:
                    # Calculate progressive backoff delay
                    delay = await calculate_backoff_delay(attempts, base_delay=2.0, multiplier=1.5, max_delay=30.0)
                    logger2.info(f"Waiting {delay:.1f}s before checking translation status, attempt {attempts+1}/{max_attempts}")
                    await asyncio.sleep(delay)
                    
                    translation_status = await make_yt_api_request(
                        f"/api/videos/{video_id}/translate/{language}/status"
                    )
                    attempts += 1
                    
                    if not translation_status:
                        logger2.info("Failed to check translation status, will retry")
                        continue
                    
                    status = translation_status.get("data", {}).get("status")
                    logger2.info(f"Translation status: {status}")
                    
                    if status == "completed":
                        logger2.info(f"Translation to {language} completed")
                        break
                        
                    if status == "error":
                        logger2.error(f"Error translating to {language}")
                        return f"Error translating to {language}, which is needed for subtitles."
                
                if attempts >= max_attempts:
                    logger2.error("Translation processing timed out")
                    return f"Translation to {language} timed out, which is needed for subtitles. Try again later."
                    
            else:
                logger2.info(f"Translation to {language} already exists")
                
        # Check if subtitles already exist
        logger2.info(f"Checking if subtitles exist for: {video_id}, language: {language}, format: {format}")
        subtitle_status = await make_yt_api_request(
            f"/api/videos/{video_id}/subtitles/status?language={language}&format={format}"
        )
        
        # Check if we need to generate subtitles (when they don't exist or are marked as not_found)
        need_to_generate = False
        
        if not subtitle_status:
            logger2.info("No subtitle status returned, will request generation")
            need_to_generate = True
        elif subtitle_status.get("status") == "success" and subtitle_status.get("data", {}).get("status") == "not_found":
            logger2.info("Subtitle status is 'not_found', will request generation")
            need_to_generate = True
        elif subtitle_status.get("data", {}).get("status") != "completed":
            logger2.info(f"Subtitle status is not 'completed': {subtitle_status.get('data', {}).get('status')}, will request generation")
            need_to_generate = True
        
        if need_to_generate:
            logger2.info(f"Requesting subtitle generation for video ID: {video_id}")
            subtitle_request = await make_yt_api_request(
                f"/api/videos/{video_id}/subtitles", 
                method="POST", 
                json_data={"language": language, "format": format}
            )
            
            if not subtitle_request:
                # If we just generated a translation, try again with a delay
                if translation_needed:
                    logger2.info("Translation was just completed, waiting before retrying subtitles request")
                    await asyncio.sleep(3)  # Small delay to let backend systems catch up
                    
                    subtitle_request = await make_yt_api_request(
                        f"/api/videos/{video_id}/subtitles", 
                        method="POST", 
                        json_data={"language": language, "format": format}
                    )
                
                # If still failing after retry
                if not subtitle_request:
                    logger2.error(f"Failed to request {format} subtitles in {language}")
                    return f"Failed to request {format} subtitles in {language}."
            
            # Poll for subtitle generation status until it's complete
            max_attempts = 15  # Subtitles can take time, so we allow more attempts
            attempts = 0
            
            while attempts < max_attempts:
                # Calculate progressive backoff delay
                delay = await calculate_backoff_delay(attempts, base_delay=1.5, multiplier=1.5)
                logger2.info(f"Waiting {delay:.1f}s before checking subtitle status, attempt {attempts+1}/{max_attempts}")
                await asyncio.sleep(delay)
                
                subtitle_status = await make_yt_api_request(
                    f"/api/videos/{video_id}/subtitles/status?language={language}&format={format}"
                )
                attempts += 1
                
                if not subtitle_status:
                    logger2.info("Failed to check subtitle status, will retry")
                    continue
                
                status = subtitle_status.get("data", {}).get("status")
                logger2.info(f"Subtitle status: {status}")
                
                if status == "completed":
                    logger2.info(f"Subtitle generation completed")
                    break
                    
                if status == "error":
                    error_msg = subtitle_status.get("data", {}).get("message", "Unknown error")
                    logger2.error(f"Error generating subtitles: {error_msg}")
                    
                    # If the error is about missing translation, and we haven't already tried to translate
                    if "translation" in error_msg.lower() and not translation_needed:
                        logger2.info("Error suggests translation is needed, will request translation and retry")
                        # Use recursion to restart the process with translation handling
                        return await get_subtitles(url, language, format)
                    
                    return f"Error generating subtitles: {error_msg}"
                    
                if status == "processing":
                    logger2.info(f"Subtitle generation still in progress")
                
            if attempts >= max_attempts:
                logger2.error("Subtitle generation timed out")
                return "Subtitle generation timed out. Try again later."
        else:
            logger2.info(f"Subtitles already exist")
        
        # Attempt to retrieve the subtitles with retry logic
        max_attempts = 5
        attempts = 0
        
        while attempts < max_attempts:
            # Calculate progressive backoff delay if not the first attempt
            if attempts > 0:
                delay = await calculate_backoff_delay(attempts - 1)
                logger2.info(f"Waiting {delay:.1f}s before retrying subtitles retrieval, attempt {attempts+1}/{max_attempts}")
                await asyncio.sleep(delay)
            
            # Get the subtitles
            logger2.info(f"Retrieving subtitles for video ID: {video_id}")
            subtitles_response = await make_yt_api_request(
                f"/api/videos/{video_id}/subtitles?language={language}&format={format}"
            )
            
            # If we get a string response, it's the raw subtitles content
            if subtitles_response and isinstance(subtitles_response, str):
                logger2.info("Successfully retrieved subtitles")
                return subtitles_response
            
            # Check if we got a "not_found" status in the response
            if (subtitles_response and 
                isinstance(subtitles_response, dict) and 
                subtitles_response.get("status") == "success" and 
                subtitles_response.get("data", {}).get("status") == "not_found"):
                
                logger2.info("Subtitle retrieval returned 'not_found', requesting generation")
                
                # Request subtitle generation
                subtitle_request = await make_yt_api_request(
                    f"/api/videos/{video_id}/subtitles", 
                    method="POST", 
                    json_data={"language": language, "format": format}
                )
                
                if not subtitle_request:
                    logger2.error(f"Failed to request {format} subtitles in {language} after 'not_found' status")
                    return f"Failed to request {format} subtitles in {language} after 'not_found' status."
                
                # Poll for subtitle generation by restarting the process
                logger2.info("Restarting subtitle process to handle newly requested generation")
                return await get_subtitles(url, language, format)
            
            # If we get an error about needing translation, and we haven't already tried to translate
            if subtitles_response and isinstance(subtitles_response, dict) and "error" in subtitles_response:
                error_msg = subtitles_response.get("message", "")
                if "translation" in error_msg.lower() and not translation_needed:
                    logger2.info("Error suggests translation is needed, will request translation and retry")
                    # Use recursion to restart the process with translation handling
                    return await get_subtitles(url, language, format)
            
            # If we didn't get a proper response, retry
            attempts += 1
            if attempts >= max_attempts:
                return f"Failed to retrieve {format} subtitles in {language} after multiple attempts."
        
        return f"Failed to retrieve subtitles within the allowed time."
        
    except Exception as e:
        logger2.error(f"Exception during subtitle generation: {str(e)}")
        return f"An error occurred while generating subtitles: {str(e)}"

@mcp.tool()
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
    try:
        # Validate length parameter
        logger2.info(f"Generating summary for: {url}, language: {language}, length: {length}")
        if length.lower() not in ["short", "medium", "long"]:
            logger2.error(f"Invalid summary length: {length}")
            return "Invalid summary length. Please use 'short', 'medium', or 'long'."
        
        length = length.lower()  # Normalize to lowercase
        
        # Extract YouTube ID if possible for direct access
        youtube_id = extract_youtube_id(url)
        video_id = youtube_id if youtube_id else url
        
        # First check if video exists and what state it's in
        video_status = await make_yt_api_request(f"/api/videos/{video_id}")
        
        # If video exists but is still processing, poll until complete
        if video_status and video_status.get("status") == "processing":
            logger2.info(f"Video is still processing, waiting for completion")
            
            # Poll for video processing status until it's complete
            max_attempts = 10
            attempts = 0
            
            while attempts < max_attempts:
                # Calculate progressive backoff delay
                delay = await calculate_backoff_delay(attempts)
                logger2.info(f"Waiting {delay:.1f}s before checking video status, attempt {attempts+1}/{max_attempts}")
                await asyncio.sleep(delay)
                
                video_status = await make_yt_api_request(f"/api/videos/{video_id}")
                attempts += 1
                
                if not video_status:
                    logger2.info("Failed to get video status, will retry")
                    continue
                
                status = video_status.get("status")
                logger2.info(f"Video status: {status}")
                
                if status == "completed":
                    logger2.info(f"Video processing completed")
                    break
                    
                if status == "error":
                    logger2.error(f"Error processing video")
                    break
        
        # If video doesn't exist or couldn't be processed, submit it
        if not video_status or video_status.get("status") not in ["completed"]:
            logger2.info(f"Submitting video for processing: {url}")
            success, video_id, error_message = await process_video(url)
            
            if not success:
                logger2.error(f"Video processing failed: {error_message}")
                return error_message
        
        # First, check if translation is needed for non-English languages
        translation_needed = False
        if language.lower() != "en":
            # Check if the translation exists
            logger2.info(f"Checking if translation exists for: {video_id}, language: {language}")
            translation_status = await make_yt_api_request(
                f"/api/videos/{video_id}/translate/{language}/status"
            )
            
            # If no translation or it's not completed, we need to request it
            if not translation_status or translation_status.get("data", {}).get("status") != "completed":
                translation_needed = True
                logger2.info(f"Translation needed for {language}, will request it first")
                
                # Request translation
                translation_request = await make_yt_api_request(
                    f"/api/videos/{video_id}/translate", 
                    method="POST", 
                    json_data={"language": language}
                )
                
                if not translation_request:
                    logger2.error(f"Failed to request translation to {language}")
                    return f"Failed to request translation to {language}, which is needed for summary."
                
                # Poll for translation completion
                max_attempts = 15  # Translations can take time
                attempts = 0
                
                while attempts < max_attempts:
                    # Calculate progressive backoff delay
                    delay = await calculate_backoff_delay(attempts, base_delay=2.0, multiplier=1.5, max_delay=30.0)
                    logger2.info(f"Waiting {delay:.1f}s before checking translation status, attempt {attempts+1}/{max_attempts}")
                    await asyncio.sleep(delay)
                    
                    translation_status = await make_yt_api_request(
                        f"/api/videos/{video_id}/translate/{language}/status"
                    )
                    attempts += 1
                    
                    if not translation_status:
                        logger2.info("Failed to check translation status, will retry")
                        continue
                    
                    status = translation_status.get("data", {}).get("status")
                    logger2.info(f"Translation status: {status}")
                    
                    if status == "completed":
                        logger2.info(f"Translation to {language} completed")
                        break
                        
                    if status == "error":
                        logger2.error(f"Error translating to {language}")
                        return f"Error translating to {language}, which is needed for summary."
                
                if attempts >= max_attempts:
                    logger2.error("Translation processing timed out")
                    return f"Translation to {language} timed out, which is needed for summary. Try again later."
                    
            else:
                logger2.info(f"Translation to {language} already exists")
        
        # Check if summary already exists
        logger2.info(f"Checking if summary exists for: {video_id}, language: {language}, length: {length}")
        summary_status = await make_yt_api_request(
            f"/api/videos/{video_id}/summarize/status?language={language}&length={length}"
        )
        
        # If summary doesn't exist or is not completed, request it
        if not summary_status or summary_status.get("data", {}).get("status") != "completed":
            logger2.info(f"Requesting summary generation for video ID: {video_id}")
            summary_request = await make_yt_api_request(
                f"/api/videos/{video_id}/summarize", 
                method="POST", 
                json_data={"language": language, "length": length}
            )
            
            if not summary_request:
                # If we just generated a translation, try again with a delay
                if translation_needed:
                    logger2.info("Translation was just completed, waiting before retrying summary request")
                    await asyncio.sleep(3)  # Small delay to let backend systems catch up
                    
                    summary_request = await make_yt_api_request(
                        f"/api/videos/{video_id}/summarize", 
                        method="POST", 
                        json_data={"language": language, "length": length}
                    )
                
                # If still failing after retry
                if not summary_request:
                    logger2.error(f"Failed to request {length} summary in {language}")
                    return f"Failed to request {length} summary in {language}."
            
            # Poll for summary generation status until it's complete
            max_attempts = 15  # Summaries can take longer, especially for longer videos
            attempts = 0
            
            while attempts < max_attempts:
                # Calculate progressive backoff delay - use a longer base delay for summaries
                delay = await calculate_backoff_delay(attempts, base_delay=2.0, multiplier=1.5, max_delay=30.0)
                logger2.info(f"Waiting {delay:.1f}s before checking summary status, attempt {attempts+1}/{max_attempts}")
                await asyncio.sleep(delay)
                
                summary_status = await make_yt_api_request(
                    f"/api/videos/{video_id}/summarize/status?language={language}&length={length}"
                )
                attempts += 1
                
                if not summary_status:
                    logger2.info("Failed to check summary status, will retry")
                    continue
                
                status = summary_status.get("data", {}).get("status")
                logger2.info(f"Summary status: {status}")
                
                if status == "completed":
                    logger2.info(f"Summary generation completed")
                    break
                    
                if status == "error":
                    error_msg = summary_status.get("data", {}).get("message", "Unknown error")
                    logger2.error(f"Error generating summary: {error_msg}")
                    
                    # If the error is about missing translation, and we haven't already tried to translate
                    if "translation" in error_msg.lower() and not translation_needed:
                        logger2.info("Error suggests translation is needed, will request translation and retry")
                        # Use recursion to restart the process with translation handling
                        return await get_summary(url, language, length)
                    
                    return f"Error generating summary: {error_msg}"
                    
                if status == "processing":
                    logger2.info(f"Summary generation still in progress")
                
            if attempts >= max_attempts:
                logger2.error("Summary generation timed out")
                return "Summary generation timed out. Try again later."
        else:
            logger2.info(f"Summary already exists")
        
        # Attempt to retrieve the summary with retry logic
        max_attempts = 5
        attempts = 0
        
        while attempts < max_attempts:
            # Calculate progressive backoff delay if not the first attempt
            if attempts > 0:
                delay = await calculate_backoff_delay(attempts - 1)
                logger2.info(f"Waiting {delay:.1f}s before retrying summary retrieval, attempt {attempts+1}/{max_attempts}")
                await asyncio.sleep(delay)
            
            # Get the summary
            logger2.info(f"Retrieving summary for video ID: {video_id}")
            summary_response = await make_yt_api_request(
                f"/api/videos/{video_id}/summary?language={language}&length={length}"
            )
            
            if summary_response and "status" in summary_response and summary_response["status"] == "success":
                # Return the generated summary from the data field
                logger2.info("Successfully retrieved summary")
                if "data" in summary_response and "summary" in summary_response["data"]:
                    return summary_response["data"]["summary"]
                else:
                    logger2.error("Unexpected response format: missing data.summary field")
                    return "Summary not available due to unexpected response format."
            
            # If we get an error about needing translation, and we haven't already tried to translate
            if summary_response and "error" in summary_response:
                error_msg = summary_response.get("message", "")
                if "translation" in error_msg.lower() and not translation_needed:
                    logger2.info("Error suggests translation is needed, will request translation and retry")
                    # Use recursion to restart the process with translation handling
                    return await get_summary(url, language, length)
            
            # If we didn't get a proper response, retry
            attempts += 1
            if attempts >= max_attempts:
                return f"Failed to retrieve {length} summary in {language} after multiple attempts."
        
        return f"Failed to retrieve summary within the allowed time."
            
    except Exception as e:
        logger2.error(f"Exception during summary generation: {str(e)}")
        return f"An error occurred while generating summary: {str(e)}"

@mcp.tool()
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
    try:
        logger2.info(f"Searching video: {url}, query: {query}")
        if not query or not query.strip():
            logger2.error("Search query cannot be empty")
            return "Search query cannot be empty."
        
        # Extract YouTube ID if possible for direct access
        youtube_id = extract_youtube_id(url)
        video_id = youtube_id if youtube_id else url
        
        # First check if video exists and what state it's in
        video_status = await make_yt_api_request(f"/api/videos/{video_id}")
        
        # If video exists but is still processing, poll until complete
        if video_status and video_status.get("status") == "processing":
            logger2.info(f"Video is still processing, waiting for completion")
            
            # Poll for video processing status until it's complete
            max_attempts = 10
            attempts = 0
            
            while attempts < max_attempts:
                # Calculate progressive backoff delay
                delay = await calculate_backoff_delay(attempts)
                logger2.info(f"Waiting {delay:.1f}s before checking video status, attempt {attempts+1}/{max_attempts}")
                await asyncio.sleep(delay)
                
                video_status = await make_yt_api_request(f"/api/videos/{video_id}")
                attempts += 1
                
                if not video_status:
                    logger2.info("Failed to get video status, will retry")
                    continue
                
                status = video_status.get("status")
                logger2.info(f"Video status: {status}")
                
                if status == "completed":
                    logger2.info(f"Video processing completed")
                    break
                    
                if status == "error":
                    logger2.error(f"Error processing video")
                    break
        
        # If video doesn't exist or couldn't be processed, submit it
        if not video_status or video_status.get("status") not in ["completed"]:
            logger2.info(f"Submitting video for processing: {url}")
            success, video_id, error_message = await process_video(url)
            
            if not success:
                logger2.error(f"Video processing failed: {error_message}")
                return error_message
        else:
            logger2.info(f"Video already processed, proceeding with search")
        
        # Attempt to perform the search with retry logic
        max_attempts = 5
        attempts = 0
        
        while attempts < max_attempts:
            # Calculate progressive backoff delay if not the first attempt
            if attempts > 0:
                delay = await calculate_backoff_delay(attempts - 1)
                logger2.info(f"Waiting {delay:.1f}s before retrying search, attempt {attempts+1}/{max_attempts}")
                await asyncio.sleep(delay)
            
            # Perform the search
            logger2.info(f"Performing search for query: {query} in video ID: {video_id}")
            search_response = await make_yt_api_request(
                f"/api/videos/{video_id}/search", 
                method="POST", 
                json_data={
                    "query": query,
                    "language": "en",  # Default to English
                    "contextSize": 60  # Larger context for better understanding
                }
            )
            
            # If we got a valid search response
            if search_response and "data" in search_response:
                search_data = search_response["data"]
                
                # Format search results
                if "results" not in search_data or not search_data["results"]:
                    logger2.info(f"No matches found for '{query}' in the video")
                    return f"No matches found for '{query}' in the video."
                    
                results = search_data["results"]
                formatted_results = []
                
                logger2.info(f"Found {len(results)} matches for query: {query}")
                for i, result in enumerate(results, 1):
                    context = result.get("context", "No context available")
                    formatted_results.append(f"Match {i}: {context}")
                    
                # Add metadata about the search
                metadata = search_data.get("metadata", {})
                language = metadata.get("language", "unknown")
                matches = metadata.get("matches", 0)
                
                # Create a formatted response
                logger2.info(f"Formatting search results with {matches} matches")
                response = f"Found {matches} match(es) for '{query}' in {language} transcript:\n\n"
                response += "\n\n".join(formatted_results)
                
                return response
            
            # If search is processing, wait and retry
            if search_response and search_response.get("status") == "processing":
                logger2.info(f"Search is still processing, will retry after backoff")
                attempts += 1
                continue
            
            # If we reach here, there's an issue with the search
            attempts += 1
            if attempts >= max_attempts:
                return f"Failed to search for '{query}' in video after multiple attempts."
        
        return f"Failed to search within the allowed time."
        
    except Exception as e:
        logger2.error(f"Exception during search: {str(e)}")
        return f"An error occurred during search: {str(e)}"

# Register empty resources and resource templates for client compatibility
# This should be one of the last things we do, after all tools are defined
mcp.add_resource(
    TextResource(
        uri="empty://resource", 
        name="Empty Resource",
        description="An empty resource placeholder",
        text=""
    )
)

# Register an empty resource template
@mcp.resource("empty://{param}", name="Empty Resource Template", description="An empty resource template placeholder")
def empty_template(param: str) -> str:
    """Return an empty string."""
    return ""

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')