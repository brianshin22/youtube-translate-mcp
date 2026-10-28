# Reviewer access and test cases

## Dedicated test account

- Login URL: https://api.youtubetranslate.com/account
- Account identifier: `account:openai-review-2026-10` (dedicated review account shared between directory reviews, not a personal user account).
- Credential: supply the account API key separately in the private reviewer-credentials field. Never include it in a public listing, source repository, prompt, screenshot, or downloadable submission asset.
- Sign-in method: account API key. No email, password, MFA, or magic link is required. A portal that labels its secret field "Password" can carry this account key if the accompanying instructions explain the key-based form.
- Credit: pre-funded for review. Latest check after protocol testing: 960,000 microcredits ($0.96); check again before submission. The amount is a snapshot, not a guaranteed perpetual balance.

## Sign-in and OAuth

1. Open the login URL. Under **Existing user access**, paste the privately supplied key into **Your user API key** and select **Sign in**. There is no username field; the account identifier above is informational.
2. In Claude, add a custom connector pointing to `https://api.youtubetranslate.com/mcp`, and start its OAuth connection.
3. If the consent browser asks you to sign in again, use the same account key on that website. Do not send it through a tool or in a conversation.
4. Select **Allow connection** once. The browser returns to the client's registered callback.
5. The account can now run the test cases without buying credits. Do not create a new empty account during review.

## Eight tool checks

Use the public video **Me at the zoo**: https://www.youtube.com/watch?v=jNQXAC9IVRw. Its English transcript, Spanish translation, and English medium summary have been retained. Normal MCP access is metered; the separate unauthenticated demo endpoint does not make MCP calls free.

| Tool | Prompt or arguments | Expected result |
|---|---|---|
| `get_processing_quote` | Quote transcription of the sample URL. | `cached: true`, maximum 2,000 microcredits; no provider work. |
| `start_video` | Start the sample URL. | Completed cached submission and a `video_id`, rather than a new transcription. |
| `get_video_status` | Check the ID returned above. | Completed, title "Me at the zoo," approximately 20-second duration. |
| `get_transcript` | Get a timestamped transcript of the sample URL. | Spoken text with timestamps covering the short elephant clip. |
| `get_translation` | Translate the sample URL into Spanish (`es`). | Spanish transcript text. |
| `get_subtitles` | Return Spanish SRT subtitles for the sample URL. | Numbered SRT cues with timestamp ranges and Spanish text. |
| `get_summary` | Summarize the sample URL in English at medium length. | A brief summary of the elephant clip, without unrelated claims. |
| `search_video` | Search the sample URL for `elephants`. | Matching passage with a timestamp near the beginning. |

These fixtures exercise short public content only. Speech recognition can mishear words; do not describe generated transcripts as perfectly accurate or verbatim guarantees.

## Negative cases

- `get_transcript` with `https://invalid.example/not-youtube`: actionable tool error, no processing.
- `get_subtitles` with `format: "mp4"`: actionable error allowing only `srt` or `vtt`, not a media download.
- Uncached work without an approved `max_credit_units`: report the budget requirement and quote instead of starting paid work. Do not start uncached work during review without an explicit approved budget.

## Scope and privacy

The connector receives tool arguments and account authorization, not general conversation history. Public-source artifacts can be reused across accounts through a shared cache. Use only the supplied public examples, never confidential or personal material. More: https://api.youtubetranslate.com/privacy.
