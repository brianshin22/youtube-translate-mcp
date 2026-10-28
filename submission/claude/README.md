# Claude connector submission

Prepared October 28, 2026. This is a remote MCP connector submission, not a Claude plugin ZIP. No credentials are included in this directory.

## Files

- `listing.json`: copy-ready listing, company, OAuth, prerequisites, and data-handling answers.
- `icon.png` and `logo.png`: listing assets. This connector returns text, not an MCP App UI, so the MCP App carousel requirement does not apply.
- `protocol-test-results.json`: real hosted MCP tests, without credentials or account identifiers.
- `reviewer-instructions.md`: sign-in procedure and eight tool checks. Supply the test key separately in the portal's private reviewer-credentials field.

## Portal steps

Open https://claude.ai/directory/manage in the paid Claude account that will own the listing. Select **Submit new**, then **MCP connector**.

1. **Connection:** `https://api.youtubetranslate.com/mcp`. One stable URL for all users; no per-customer URL pattern.
2. **Tools:** sync all eight tools. Human-readable titles and explicit `readOnlyHint`, `destructiveHint`, and `openWorldHint` booleans are live.
3. **Listing:** copy the public fields from `listing.json`, upload `icon.png`, and select the nearest available categories. Suggested categories are labels, not a claim about the current portal enum. Verify the permanent slug before publication.
4. **Use cases:** research, studying, transcript translation, subtitles, summaries, and in-video search. Explain the account authorization and service-credit requirements. Do not describe normal MCP reads as free.
5. **Company:** Silver Flight Group, LLC, https://youtubetranslate.com, brian@silverflightgroup.com.
6. **Authentication:** OAuth required. PKCE S256 with dynamic client registration. Do not paste an account API key into OAuth client ID/secret fields. The discovery document supplies the endpoints. If the portal requires a pre-registered confidential client, stop: the current server supports public PKCE clients, not client-secret authentication.
7. **Data handling:** our first-party REST API, with public-source retrieval and processing providers disclosed. Do not claim we own YouTube or have a partnership with YouTube. The shared cache is not private account storage. No PHI or sponsored-content use case is offered.
8. **Test & launch:** provide the dedicated test account key privately and the instructions in `reviewer-instructions.md`. Check the test-attestation box only after the required Claude/Inspector checks below.
9. **Compliance:** the owner must read the current seven acknowledgments and confirm they are accurate. We do not pre-accept terms or assert that an unknown acknowledgment is satisfied. The tools do not transfer financial assets or generate images/audio/video; they do consume purchased service credits and use AI for text processing.
10. **Review:** inspect all warnings, save the draft, and submit only after the remaining tests and acknowledgments are complete.

## Completed checks

- Five automated Python tests pass, including tool metadata and budget forwarding.
- Hosted `tools/list` includes titles and three explicit safety booleans on all eight tools.
- All eight tools return successful content using the reviewer account and an already retained public video.
- Invalid YouTube URLs and unsupported subtitle formats return tool errors.
- Public docs and privacy policy are available. OAuth domain verification for OpenAI is separate and does not replace Claude's connection flow.

## Still required before submission

- Add the endpoint as a **custom connector in Claude**, complete OAuth with the reviewer account, and exercise all eight tools from a conversation. Save the results and any screenshots. The production OAuth fix was browser-tested, but that is not a Claude UI test.
- Exercise the endpoint in **MCP Inspector**, as the review checklist requests. Protocol calls in `protocol-test-results.json` are not an Inspector session.
- Confirm the reviewer account still has adequate credit, and enter its key in the portal privately.
- Confirm the company's support address is monitored and the owner approves the privacy disclosures and policy acknowledgments.
- Final portal review and submission have not been performed by this preparation step.

## Official references

- https://claude.com/docs/connectors/building/submission
- https://claude.com/docs/connectors/building/review-criteria
- https://claude.com/docs/connectors/building/authentication
- https://claude.com/docs/connectors/building/testing
