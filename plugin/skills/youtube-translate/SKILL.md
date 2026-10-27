---
name: youtube-translate
description: Get transcripts, translations, summaries, subtitles, and timestamped search from public YouTube videos. Use when the user provides a YouTube URL and asks what was said, wants another language, wants captions, or wants a summary.
---

# YouTube Translate

Use the YouTube Translate tools for public YouTube videos. Do not claim a transcript exists unless a tool returned it.

## Cached videos

Call `get_transcript` first for a transcript. Cached public videos cost nothing and return immediately. Use `include_timestamps=true` when the user asks for timestamps, quotes, or specific moments.

## Uncached videos

1. Call `get_processing_quote` for the operation.
2. Show the user the maximum microcredit cost and the account balance. One million microcredits equals one US dollar.
3. Ask for explicit approval before spending. Never guess a budget.
4. If approved, call `start_video` with `max_credit_units` equal to the quoted maximum.
5. Poll `get_video_status`, then call `get_transcript`.

If the account has no credits, explain that the user must add prepaid credits at `https://api.youtubetranslate.com/account/payments` and approve a Stripe Link payment. Never ask for card or wallet credentials in chat.

## Translation and subtitles

Use `get_translation` for translated text and `get_subtitles` for SRT or VTT. Use two-letter ISO language codes. Pass `max_credit_units` only after approval. Translation and subtitle formatting use the retained transcript, so they do not download the video again.

## Summaries and search

Use `get_summary` with `short`, `medium`, or `long`. Use `search_video` to find a topic or quote; return its timestamp and surrounding passage.

## Limits

Transcription is limited to 30 minutes of source audio. Translation and summary inputs are limited to 100,000 characters and 2,000 segments. If a tool returns a limit error, explain it instead of retrying blindly.

## Privacy

Do not send private, confidential, or sensitive material to this service. Public-source media and results may be retained in a shared cache.
