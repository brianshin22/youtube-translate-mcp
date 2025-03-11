# YouTube Translate API Reference

This comprehensive API documentation details how to interact with the YouTube Translate API, including authentication requirements, request formats, and example curl commands for all endpoints.

## Table of Contents

1. [Authentication](#authentication)
2. [Health Check](#health-check)
3. [Video Processing](#video-processing)
4. [Transcription](#transcription)
5. [Translation](#translation)
6. [Summarization](#summarization)
7. [Search](#search)
8. [Subtitles](#subtitles)
9. [Error Handling](#error-handling)

## Authentication

All API requests must include an API key in the `X-API-Key` header (except for the health check endpoint, which is public).

```bash
curl -H "X-API-Key: YOUR_API_KEY" https://api.example.com/api/endpoint
```

### API Key Types

The system uses three types of API keys:

1. **Master API Key**: Format `ut-master-xxxxxxxxxxxxx` (For internal backend-to-backend communication)
2. **User API Key**: Format `ut-user-xxxxxxxxxxxxx` (For end users with subscription plans)
3. **Application API Key**: Format `ut-app-xxxxxxxxxxxxx` (For application integrations)

## Health Check

Check if the API service is running.

**Request**

```
GET /api/health
```

**Example**

```bash
curl http://localhost:3000/api/health
```

**Response**

```json
{
  "status": "ok",
  "timestamp": "2025-03-06T16:06:50.033Z"
}
```

## Video Processing

### Using YouTube IDs Directly

All API endpoints support using a YouTube video ID directly instead of the internal UUID. This allows for more convenient API calls without having to remember the internal UUIDs.

**Example: Get video information using a YouTube ID**

```bash
# Using the YouTube ID directly (dQw4w9WgXcQ is the ID from https://www.youtube.com/watch?v=dQw4w9WgXcQ)
curl -H "X-API-Key: YOUR_API_KEY" \
  http://localhost:3000/api/videos/dQw4w9WgXcQ
```

**Example: Request translation using a YouTube ID**

```bash
curl -X POST \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"language": "fr"}' \
  http://localhost:3000/api/videos/dQw4w9WgXcQ/translate
```

The API will automatically map the YouTube ID to the corresponding internal UUID if the video has been processed before.

### Submit a Video for Processing

Submit a YouTube (or other supported platform) video for processing.

**Request**

```
POST /api/videos
```

**Headers**

- `X-API-Key`: Your API key
- `Content-Type`: application/json

**Request Body Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| url | string | Yes | URL of the video to process |
| language | string | No | Language code for transcription (e.g., "en", "ko"). Default: auto-detect |

**Example**

```bash
curl -X POST \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=UmPNQ302ADU", "language": "ko"}' \
  http://localhost:3000/api/videos
```

**Response**

```json
{
  "id": "ce356194-9c9e-465d-92d4-d00dd371f2a3",
  "status": "processing",
  "message": "Video processing started",
  "url": "https://www.youtube.com/watch?v=UmPNQ302ADU"
}
```

### Get Video Status and Metadata

Retrieve the status and metadata of a processed video.

**Request**

```
GET /api/videos/:id
```

**URL Parameters**

- `id`: The ID of the submitted video. Can be either the internal UUID returned when submitting the video or the YouTube video ID (e.g., "dQw4w9WgXcQ").

**Headers**

- `X-API-Key`: Your API key

**Example**

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  http://localhost:3000/api/videos/ce356194-9c9e-465d-92d4-d00dd371f2a3
```

**Response**

```json
{
  "id": "ce356194-9c9e-465d-92d4-d00dd371f2a3",
  "status": "completed",
  "metadata": {
    "title": "[TEASER🎨] 아이유 X 도경수",
    "duration": 54,
    "thumbnail": "https://i.ytimg.com/vi_webp/UmPNQ302ADU/maxresdefault.webp",
    "uploadDate": "20240510",
    "transcript": {
      "chunks": [
        {"text": "팬클럽 명은 정하셨어요?", "timestamp": [0, 3.74]},
        ...
      ],
      "text": "팬클럽 명은 정하셨어요? 아니요 아직 정하지 못했어요..."
    }
  }
}
```

## Transcription

### Get Video Transcript

Retrieve the transcript for a processed video.

**Request**

```
GET /api/videos/:id/transcript
```

**URL Parameters**

- `id`: The ID of the submitted video. Can be either the internal UUID returned when submitting the video or the YouTube video ID (e.g., "dQw4w9WgXcQ").

**Headers**

- `X-API-Key`: Your API key

**Example**

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  http://localhost:3000/api/videos/ce356194-9c9e-465d-92d4-d00dd371f2a3/transcript
```

**Response**

```json
{
  "id": "ce356194-9c9e-465d-92d4-d00dd371f2a3",
  "status": "completed",
  "metadata": {
    "title": "[TEASER🎨] 아이유 X 도경수",
    "duration": 54,
    "thumbnail": "https://i.ytimg.com/vi_webp/UmPNQ302ADU/maxresdefault.webp",
    "uploadDate": "20240510"
  },
  "transcript": {
    "chunks": [
      {"text": "팬클럽 명은 정하셨어요?", "timestamp": [0, 3.74]},
      ...
    ],
    "text": "팬클럽 명은 정하셨어요? 아니요 아직 정하지 못했어요..."
  }
}
```

### Get Available Transcript Languages

Get a list of available languages for the transcript.

**Request**

```
GET /api/videos/:id/transcript/languages
```

**URL Parameters**

- `id`: The ID of the submitted video. Can be either the internal UUID returned when submitting the video or the YouTube video ID (e.g., "dQw4w9WgXcQ").

**Headers**

- `X-API-Key`: Your API key

**Example**

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  http://localhost:3000/api/videos/ce356194-9c9e-465d-92d4-d00dd371f2a3/transcript/languages
```

**Response**

```json
{
  "status": "success",
  "data": {
    "languages": ["en", "ko"]
  }
}
```

## Translation

### Request Transcript Translation

Request a translation of the transcript to another language.

**Request**

```
POST /api/videos/:id/translate
```

**URL Parameters**

- `id`: The ID of the submitted video. Can be either the internal UUID returned when submitting the video or the YouTube video ID (e.g., "dQw4w9WgXcQ").

**Headers**

- `X-API-Key`: Your API key
- `Content-Type`: application/json

**Request Body Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| language | string | Yes | Target language code (e.g., "en", "fr") |

**Example**

```bash
curl -X POST \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"language": "en"}' \
  http://localhost:3000/api/videos/ce356194-9c9e-465d-92d4-d00dd371f2a3/translate
```

**Response**

```json
{
  "status": "success",
  "data": {
    "videoId": "ce356194-9c9e-465d-92d4-d00dd371f2a3",
    "language": "en",
    "translationId": "dcfc7105-d869-4af5-a5f7-ca1701d26abe",
    "status": "processing"
  }
}
```

### Check Translation Status

Check the status of a translation request.

**Request**

```
GET /api/videos/:id/translate/:language/status
```

**URL Parameters**

- `id`: The ID of the submitted video. Can be either the internal UUID returned when submitting the video or the YouTube video ID (e.g., "dQw4w9WgXcQ").
- `language`: The target language code

**Headers**

- `X-API-Key`: Your API key

**Example**

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  http://localhost:3000/api/videos/ce356194-9c9e-465d-92d4-d00dd371f2a3/translate/en/status
```

**Response**

```json
{
  "status": "success",
  "data": {
    "videoId": "ce356194-9c9e-465d-92d4-d00dd371f2a3",
    "language": "en",
    "status": "completed"
  }
}
```

### Get Translated Transcript

Retrieve the translated transcript.

**Request**

```
GET /api/videos/:id/transcript/:language
```

**URL Parameters**

- `id`: The ID of the submitted video. Can be either the internal UUID returned when submitting the video or the YouTube video ID (e.g., "dQw4w9WgXcQ").
- `language`: The language code of the translation to retrieve

**Headers**

- `X-API-Key`: Your API key

**Example**

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  http://localhost:3000/api/videos/ce356194-9c9e-465d-92d4-d00dd371f2a3/transcript/en
```

**Response**

```json
{
  "status": "success",
  "data": {
    "chunks": [
      {"text": "Have you decided on the fan club name?", "timestamp": [0, 3.74]},
      ...
    ],
    "text": "Have you decided on the fan club name? No, I haven't decided yet..."
  }
}
```

## Summarization

### Request Video Summary

Generate a summary of the video content.

**Request**

```
POST /api/videos/:id/summarize
```

**URL Parameters**

- `id`: The ID of the submitted video. Can be either the internal UUID returned when submitting the video or the YouTube video ID (e.g., "dQw4w9WgXcQ").

**Headers**

- `X-API-Key`: Your API key
- `Content-Type`: application/json

**Request Body Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| language | string | No | Language code for the summary (e.g., "en"). Default: "en" |
| length | string | No | Length of the summary ("short", "medium", "long"). Default: "medium" |

**Notes on Language Handling**

- If you request a summary in a language for which no translation exists, the API will return a 404 error.
- This applies to ALL languages (including English) when they don't match the original video language.
- For example, if the original video is in Korean, you'll need an English translation before requesting an English summary.
- Before requesting a summary in any language other than the original video language, you must first request a translation using the `/api/videos/:id/translate` endpoint.
- The system automatically handles language variants like Chinese ('zh', 'zh-cn', 'zh-tw') and Japanese ('ja', 'jpn').

**Example**

```bash
curl -X POST \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"language": "en", "length": "short"}' \
  http://localhost:3000/api/videos/ce356194-9c9e-465d-92d4-d00dd371f2a3/summarize
```

**Response**

```json
{
  "status": "success",
  "data": {
    "videoId": "ce356194-9c9e-465d-92d4-d00dd371f2a3",
    "language": "en",
    "length": "short",
    "summaryId": "4290e552-a59e-4a80-96f5-bd5d14f2a3f8",
    "status": "processing"
  }
}
```

### Check Summary Status

Check the status of a summary generation request.

**Request**

```
GET /api/videos/:id/summarize/status
```

**URL Parameters**

- `id`: The ID of the submitted video. Can be either the internal UUID returned when submitting the video or the YouTube video ID (e.g., "dQw4w9WgXcQ").

**Query Parameters**

- `language`: Language code for the summary (e.g., "en")
- `length`: Length of the summary ("short", "medium", "long")

**Headers**

- `X-API-Key`: Your API key

**Example**

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  "http://localhost:3000/api/videos/ce356194-9c9e-465d-92d4-d00dd371f2a3/summarize/status?language=en&length=short"
```

**Response**

```json
{
  "status": "success",
  "data": {
    "videoId": "ce356194-9c9e-465d-92d4-d00dd371f2a3",
    "language": "en",
    "length": "short",
    "status": "completed"
  }
}
```

### Get Video Summary

Retrieve the generated summary.

**Request**

```
GET /api/videos/:id/summary
```

**URL Parameters**

- `id`: The ID of the submitted video. Can be either the internal UUID returned when submitting the video or the YouTube video ID (e.g., "dQw4w9WgXcQ").

**Query Parameters**

- `language`: Language code for the summary (e.g., "en")
- `length`: Length of the summary ("short", "medium", "long")

**Headers**

- `X-API-Key`: Your API key

**Example**

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  "http://localhost:3000/api/videos/ce356194-9c9e-465d-92d4-d00dd371f2a3/summary?language=en&length=short"
```

**Response**

```json
{
  "status": "success",
  "data": {
    "summary": "**Video Summary:**\n\n**Title:** [TEASER🎨] 아이유 X 도경수\nCréateur: Inconnu\nDurée: 0:54\n\nCe teaser présente une collaboration entre IU (아이유) et D.O. (도경수) du groupe EXO..."
  }
}
```

## Search

### Search Within Video Content

Search for specific content within the video transcript.

**Request**

```
POST /api/videos/:id/search
```

**URL Parameters**

- `id`: The ID of the submitted video. Can be either the internal UUID returned when submitting the video or the YouTube video ID (e.g., "dQw4w9WgXcQ").

**Headers**

- `X-API-Key`: Your API key
- `Content-Type`: application/json

**Request Body Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | string | Yes | Search term or phrase |
| language | string | No | Language code to search in (e.g., "en", "ko"). Default: "en" |
| contextSize | number | No | Number of characters of context to return around matches. Default: 30 |

**Example**

```bash
curl -X POST \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "fan club", "language": "en", "contextSize": 30}' \
  http://localhost:3000/api/videos/ce356194-9c9e-465d-92d4-d00dd371f2a3/search
```

**Response**

```json
{
  "status": "success",
  "data": {
    "id": "ce356194-9c9e-465d-92d4-d00dd371f2a3",
    "query": "fan club",
    "results": [
      {
        "text": "fan club",
        "context": "Have you decided on the fan club name? No, not yet. I couldn't sleep well yesterday...",
        "position": 24
      }
    ],
    "metadata": {
      "language": "en",
      "context_size": 30,
      "matches": 1
    }
  }
}
```

## Subtitles

### Generate Subtitles

Generate subtitle files for the video.

**Request**

```
POST /api/videos/:id/subtitles
```

**URL Parameters**

- `id`: The ID of the submitted video. Can be either the internal UUID returned when submitting the video or the YouTube video ID (e.g., "dQw4w9WgXcQ").

**Headers**

- `X-API-Key`: Your API key
- `Content-Type`: application/json

**Request Body Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| language | string | No | Language code for subtitles (e.g., "en"). Default: "en" |
| format | string | No | Subtitle format ("srt" or "vtt"). Default: "srt" |

**Example**

```bash
curl -X POST \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"language": "en", "format": "srt"}' \
  http://localhost:3000/api/videos/ce356194-9c9e-465d-92d4-d00dd371f2a3/subtitles
```

**Response**

```json
{
  "status": "success",
  "data": {
    "id": "ce356194-9c9e-465d-92d4-d00dd371f2a3",
    "language": "en",
    "format": "srt",
    "status": "processing"
  }
}
```

### Check Subtitles Status

Check the status of subtitle generation.

**Request**

```
GET /api/videos/:id/subtitles/status
```

**URL Parameters**

- `id`: The ID of the submitted video. Can be either the internal UUID returned when submitting the video or the YouTube video ID (e.g., "dQw4w9WgXcQ").

**Query Parameters**

- `language`: Language code for the subtitles (e.g., "en")
- `format`: Subtitle format ("srt" or "vtt")

**Headers**

- `X-API-Key`: Your API key

**Example**

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  "http://localhost:3000/api/videos/ce356194-9c9e-465d-92d4-d00dd371f2a3/subtitles/status?language=en&format=srt"
```

**Response**

```json
{
  "status": "success",
  "data": {
    "videoId": "ce356194-9c9e-465d-92d4-d00dd371f2a3",
    "language": "en",
    "format": "srt",
    "status": "completed",
    "url": "/api/videos/ce356194-9c9e-465d-92d4-d00dd371f2a3/subtitles?language=en&format=srt"
  }
}
```

### Get Subtitles

Retrieve the generated subtitles.

**Request**

```
GET /api/videos/:id/subtitles
```

**URL Parameters**

- `id`: The ID of the submitted video. Can be either the internal UUID returned when submitting the video or the YouTube video ID (e.g., "dQw4w9WgXcQ").

**Query Parameters**

- `language`: Language code for the subtitles (e.g., "en")
- `format`: Subtitle format ("srt" or "vtt")

**Headers**

- `X-API-Key`: Your API key

**Notes on Language Handling**

- If you request subtitles in a language that doesn't have a translation, the API will return a 404 error.
- This applies to ALL languages (including English) when they don't match the original video language.
- For example, if the original video is in Korean, you'll need an English translation before requesting English subtitles.
- Before requesting subtitles in any language other than the original video language, you must first request a translation using the `/api/videos/:id/translate` endpoint.
- The system automatically handles language variants like Chinese ('zh', 'zh-cn', 'zh-tw') and Japanese ('ja', 'jpn').

**Example**

```bash
curl -H "X-API-Key: YOUR_API_KEY" \
  "http://localhost:3000/api/videos/ce356194-9c9e-465d-92d4-d00dd371f2a3/subtitles?language=en&format=srt"
```

**Response**

The response will be the raw subtitle content with the appropriate content type header:

For SRT format:
```
1
00:00:00,000 --> 00:00:03,740
Have you decided on the fan club name?

2
00:00:03,740 --> 00:00:05,839
No, I haven't decided yet.

...
```

For VTT format:
```
WEBVTT
Language: en
Title: [TEASER🎨] 아이유 X 도경수

00:00:00.000 --> 00:00:03.740
Have you decided on the fan club name?

00:00:03.740 --> 00:00:05.839
No, I haven't decided yet.

...
```

**Content-Type Headers**

- SRT: `text/plain`
- VTT: `text/vtt`

The subtitle files are also provided with a `Content-Disposition` header for downloading with a filename based on the video ID.

**Note on Subtitle Generation Process**

The subtitle generation process includes the following steps:
1. When you request subtitles, the system creates a lock file to indicate processing is in progress
2. If subtitles already exist, they will be returned immediately
3. If subtitles are being generated, the status endpoint will return "processing"
4. Once generation is complete, the lock is updated to "completed" and the subtitles become available

If you experience issues with subtitles remaining in the "processing" state for an extended time, please try the following:
1. Check that the video processing status is "completed" before requesting subtitles
2. For videos with multiple languages, ensure the requested language has been translated first
3. If needed, re-submit the subtitle generation request to clear any stale locks

## Error Handling

The API uses standard HTTP status codes to indicate the success or failure of an API request.

### Common Status Codes

- `200 OK` - The request was successful
- `400 Bad Request` - The request could not be understood or was missing required parameters
- `401 Unauthorized` - Authentication failed or user does not have permissions for the requested operation
- `404 Not Found` - The requested resource could not be found
- `500 Internal Server Error` - An error occurred on the server

### Error Response Format

```json
{
  "status": "error",
  "message": "Error message describing what went wrong"
}
```

### Common Errors

- **Invalid API Key**
  ```json
  {
    "error": {
      "message": "Invalid API key"
    }
  }
  ```

- **Missing Required Parameters**
  ```json
  {
    "error": {
      "message": "Query is required"
    }
  }
  ```

- **Resource Not Found**
  ```json
  {
    "status": "error",
    "message": "Transcript not found"
  }
  ```

- **Processing Error**
  ```json
  {
    "status": "error",
    "message": "Error processing video"
  }
  ``` 
