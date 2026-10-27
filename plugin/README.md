# ChatGPT and Codex plugin package

This folder is the submission package for the public plugin directory. It connects ChatGPT and Codex to the hosted MCP server at `https://api.youtubetranslate.com/mcp`.

## Package contents

- `plugin.json`: portable plugin identity and OpenAI presentation metadata.
- `mcp.json`: remote Streamable HTTP server with OAuth Dynamic Client Registration.
- `skills/youtube-translate`: tool-selection and spending-approval instructions.
- `skills/setup`: first-run workflow for the free cached demo and the paid path.
- `assets/`: icon, logo, and screenshots.

## Reviewer account

Create a dedicated account for OpenAI review. It must have credits, no MFA prompt, and no real customer data.

1. Create a key-based account on production and save its API key.
2. Grant it enough credits for the test cases below. A reviewer grant is a ledger entry, not a live charge.
3. The reviewer signs in at `https://api.youtubetranslate.com/account` with that key, then completes OAuth when ChatGPT connects the MCP server.
4. Keep the account active for subsequent reviews.

Do not use the owner's personal account, a real customer's account, or a live payment card in review credentials.

## Positive test cases

| # | Prompt | Expected tools | Expected result |
|---|---|---|---|
| 1 | Give me a timestamped transcript of https://www.youtube.com/watch?v=Ro6MDU2CFO4 | `get_transcript` | Timestamped transcript containing “Johnson Space Center”. No charge. |
| 2 | How much would it cost to transcribe https://www.youtube.com/watch?v=dQw4w9WgXcQ? | `get_processing_quote` | Quote in microcredits and dollars, account balance, and no charge. |
| 3 | Translate that transcript into Spanish. | `get_translation` | Spanish transcript text, with the approved budget. |
| 4 | Summarize it in three bullets. | `get_summary` | A short summary, with the approved budget. |
| 5 | Find the part about Johnson Space Center and give me the timestamp. | `search_video` | A matching passage and timestamp. |

## Negative test cases

| # | Prompt | Expected behavior |
|---|---|---|
| 1 | Transcribe this private or invalid URL: not-a-youtube-url | Refuse with a validation message. Do not process. |
| 2 | Just charge my card and start the job. | Refuse. Explain that the user must approve a Stripe Link checkout; never request card details in chat. |
| 3 | Transcribe this 2-hour video. | Explain the 30-minute limit. Do not start a provider call. |

## Walkthrough video

`https://api.youtubetranslate.com/walkthrough.mp4` (2 minutes, 1280x720, captions included).

The recording shows the live service: the free cached demo, a real quote, an uncached job with an approved budget, and the settlement of 6,000 microcredits. It is also embedded on the landing page with a poster frame and a WebVTT caption track.

Re-record it whenever the review test cases change. Keep the captions in sync with the frames.

## Submission steps

1. Zip this folder without private credentials.
2. Upload the ZIP at `https://platform.openai.com/plugins`.
3. Complete business verification for Silver Flight Group, LLC.
4. Connect the MCP server with OAuth DCR and complete the domain challenge at `https://api.youtubetranslate.com/.well-known/openai-apps-challenge`.
5. Scan tools and resolve metadata findings.
6. Enter reviewer credentials separately in Review details.
7. Provide the five positive and three negative test cases plus a short walkthrough video.
8. Submit for review. Publish only after approval.

## Current blockers

- OpenAI organization access and business verification must be completed by the owner.
- The reviewer account exists and holds credits; keep it active for later reviews.
