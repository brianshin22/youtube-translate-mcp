# Directory submissions: YouTube Translate

Status recorded October 28, 2026. This file is the product-specific handoff; the cross-product playbook is maintained at https://github.com/silver-flight-group/skills/blob/main/docs/mcp-directory-submissions.md.

## State by channel

| Channel | Recorded state | What this does not prove |
|---|---|---|
| Official MCP Registry | `io.github.brianshin22/youtube-translate`, v0.2.1, observed active | Approval by a chat-product directory |
| ChatGPT | **Rejected**; OpenAI email read October 28, 2026 | Neither approved nor published; appeal/resubmission offered |
| Claude | Materials prepared in [`claude/`](claude/); no portal submission performed | Claude UI or Inspector testing, review, or publication |
| Third-party directories | Existing entries found for PulseMCP, Glama, mcp.so, LobeHub, and mcpservers.org; listing freshness varies | Usage or new submissions to those directories |
| Smithery | No confirmed listing/submission completed during this launch | The README badge is not evidence of a working listing |

Hosted endpoint: https://api.youtubetranslate.com/mcp.
Docs: https://youtubetranslate.com/docs/.
Privacy: https://api.youtubetranslate.com/privacy.

## OpenAI submission record

Draft/manage URL recorded from the dashboard:

https://platform.openai.com/plugins/manage/plugin_asdk_app_6ae13d56921c81918e89d6b3ad061995?tab=details

The owner completed account onboarding and business verification, uploaded the corrected package, connected the server, supplied reviewer access, and reported submission. Record any future approval/rejection email and publication action separately. Do not mark this as published merely because the owner clicked Submit.

### Manifest iterations

- v0.1.0: initial upload exposed a subtitle over the 30-character limit and a missing support URL. Two skills passed checks, but that did not clear MCP setup or review information.
- v0.2.0: shortened subtitle, added support URL, rewrote description, and embedded five positive/three negative cases, walkthrough URL, commerce disclosure, and release notes.
- v0.2.1: removed remaining cost/approval language from the public listing description. Function and limitations remain; pricing stays in review details and product docs.

Manifest source is [`../plugin/plugin.json`](../plugin/plugin.json). ZIP filenames used during handoff (`...-v2.zip` and `...-v3.zip`) are not the authoritative semantic version; inspect `plugin.json` inside the archive.

### Findings and actual fixes

| Observation | Root cause / fix | Verification |
|---|---|---|
| Subtitle too long | Portal limit is 30 characters; changed to "Transcripts and translations" (28) | Package validation and owner-uploaded dashboard no longer showed that blocker |
| Support URL missing | `supportURL` absent from `extensions.com.openai.interface` | Added public repository Issues URL |
| Pricing advisory | Functional description still mentioned spending and maximum cost | Removed all monetary/approval copy from listing description; requires latest portal scan to confirm clearance |
| Privacy assessment advisory | Existing policy was brief; assessment could not complete | Expanded operator/contact, data categories, providers, assistant boundary, cache, retention, deletion requests, cookies, and international processing; public endpoint returned 200. No claim that the advisory has cleared without a rescan |
| Domain challenge 404 | `OPENAI_APPS_CHALLENGE` not configured | Exact issued token deployed as plain text at the requested endpoint; public response matched byte-for-byte |
| Allow connection appears inert, then "Connection expired" | Helmet's `form-action 'self'` blocked the consent POST's cross-origin 303 callback; first request consumed the handle, second request failed | Reproduced in real Chrome; scoped the consent page to its validated callback origin; fresh browser test reached a controlled callback. General account pages retain self-only form action |
| Eight red tool annotation findings | All eight live tool annotations were null | Added three explicit boolean fields; all eight serialized live tools verified; five Python tests pass |
| Claude requires titles | Tool names existed but human-readable titles did not | Added top-level titles and matching annotation titles; live `tools/list` checked for all eight |
| Reviewer username/password form mismatch | Service signs in with an account key, not a username/password pair | Login instructions explain the dedicated key goes into "Your user API key" on our site, then OAuth consent. No key in public artifacts |

Relevant code changes: API commits `f210066` (Link popup COOP), `b2819e4` (consent CSP and expanded privacy); MCP commits `1cd77bd` (review metadata), `d55b769` (description), `a84f4b8` (annotations), and `885ccaf` (titles and Claude materials).

## Rejection received and policy reassessment

OpenAI's email, subject `ChatGPT Plugin Submission Rejected [asdk_app_6ae13d56921c81918e89d6b3ad061995]`, was sent to the submission contact and read in the owner's forwarded mailbox. It named three findings:

1. Commerce for disallowed offerings, including digital goods/services.
2. Name/description implying affiliation or using potentially trademarked terms.
3. Tool annotation mismatches; the email requested explicit booleans and justifications without naming an offending tool.

The email explicitly permits remediation and resubmission, or appeal by replying. No appeal or resubmission has been sent as part of this reassessment.

### Concrete policy gaps

The current [plugin guidelines](https://developers.openai.com/plugins/plugin-guidelines) prohibit selling digital services/credits through a directory plugin, directly or indirectly, and prohibit links to checkout or pages initiating a purchase. They permit access to an existing paid account and informational entitlement explanations. Our shipped setup and usage skills direct an unfunded user to the account payments page, so the commerce finding is not addressed merely by removing pricing from the public description or deleting a commerce flag. Do not hide this behavior or misdeclare monetization to pass review.

The product name contains YouTube, a third-party brand. The email does not specify the exact offending text or say that changing the name alone guarantees approval. Rebranding and an independence statement may address confusion but do not establish permission to acquire or process third-party content.

All eight tools were verified to emit explicit booleans before submission. The remaining annotation finding therefore may concern semantic correctness or an older scan, not missing keys. Tools charge credits, persist artifacts, and can perform non-reversible provider work; non-destructive flags need reevaluation against OpenAI's explicit treatment of irreversible transactions and overwrites. Do not simply repeat the prior claim that every effect is an additive, non-destructive write. Current guidelines say annotation justifications are no longer required, while the rejection email asks for them; seek clarification or include concise behavior explanations in any appeal.

A further policy risk not cited in the email: unofficial third-party connectors, unauthorized scraping, and circumvention of third-party restrictions. Being able to retrieve public YouTube audio does not establish terms compliance or permission. The residential relay is an engineering mechanism, not evidence of authorization. Assess this before investing in another directory submission or implying that a rename resolves the product's eligibility.

### Distribution and website implications

Do not advertise a public ChatGPT Directory install or assume saying the product name gives ChatGPT its tools. Distinguish existing/custom MCP connections (availability depends on client/workspace) from directory approval. API-capable coding agents can use the first-party REST API directly, but ordinary chat browsing is not an authenticated API runtime. The landing-page claim that its transcript-plus-summary MCP prompt is free is also wrong: free demos use a separate endpoint, not normal authenticated MCP calls.

Recommended near-term focus: reliable REST and hosted MCP access for supported clients, honest setup and pricing, and compliant source acquisition. Pause public-directory approval claims pending an owner decision on a policy-compliant resubmission or appeal. Website copy changes were not performed by this rejection documentation update.

## Tests performed and limits of evidence

- Real owner-authorized $1 Stripe Link payment settled and credited 1,000,000 live microcredits.
- An uncached 20-second transcription settled at 10,000 microcredits, with the unused hold released. This is evidence for that short source, not long podcasts.
- Claude preparation called all eight tools through the hosted MCP JSON-RPC endpoint, using the dedicated reviewer account and cached public fixtures. All returned successful responses. Invalid URLs and unsupported subtitle formats returned errors. Test cost: 14,000 microcredits ($0.014).
- Those protocol tests used an account API key. They were **not** Claude UI, MCP Inspector, or OAuth-authenticated tool tests.
- Browser OAuth regression verified a controlled cross-origin callback after consent. The owner's successful OpenAI connection/submission is separate from that regression.
- Generated speech recognition contains errors (for example, the short elephant fixture includes a misrecognized word). Returning text does not establish perfect accuracy.
- Account and ledger backups were uploaded and downloaded, checksums matched, restored DBs passed integrity checks, and the restored live balance matched production. Daily backups are not zero-data-loss protection.

Latest sanitized tool-test evidence: [`claude/protocol-test-results.json`](claude/protocol-test-results.json).

## Reviewer fixture and account

Use public video `jNQXAC9IVRw` ("Me at the zoo"). English transcript, Spanish translation, and English medium summary were retained for repeatable review. The review account is dedicated test data, not a customer account. Its key must be provided only in a private portal field.

Normal authenticated MCP reads still use credit; free demo endpoints are separate. As of the Claude protocol test, the reviewer balance was 960,000 microcredits. Check its current balance before another review or automated test run rather than assuming this snapshot remains valid.

## Behavior intentionally held steady during review

The owner chose to keep the **30-minute source cap** until plugin review finishes. Do not raise it to four hours as a casual constant change. Longer podcasts also need chunked transcription, accurate timestamp merging, resumable jobs, duration-based budgets, and long-input translation/summary handling. Metadata-only title changes do not implement that feature.

## Claude handoff

The materials in [`claude/`](claude/) include listing copy, icons, OAuth settings, data handling, reviewer instructions, and portal steps. The reference ZIP is not a Claude plugin package. Still required:

- Custom-connector connection and all tool checks in Claude.
- MCP Inspector testing per the current checklist.
- Private reviewer credential entry and balance check.
- Owner review of support contact, disclosures, and policy acknowledgments.
- Portal submission and status tracking.

## Acquisition and analytics lessons

- Existing directory entries were found by searching before attempting duplicate submissions.
- Search Console required the correct signed-in Google account (`authuser` mattered). The domain property existed; the first account's lack of access was not proof that no property existed.
- Sitemap submission and "Indexing requested" were confirmed in the UI, not inferred from HTTP status. IndexNow acceptance is not proof of Google indexing.
- Usage snapshot immediately after launch showed only known owner/reviewer billed activity. Do not count our own checks, OAuth scans, or site crawls as external adoption.
- A newly authorized ChatGPT account with zero jobs is a successful connection, not a paying or active customer.

## Browser automation lessons

The upload handoff was unnecessarily prolonged by repeated clicks, modal/file-picker failures, and attempts to restart services/browser state. Future submissions should follow the computer-use skill: paste real files where supported; if a site only offers an unsupported picker, give the owner the file and exact steps promptly. Do not restart the owner's browser without confirmation. Screenshots must be read at the correct coordinate scale; a reported click does not prove the requested UI transition occurred.
