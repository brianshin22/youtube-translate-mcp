---
name: setup-youtube-translate
description: Set up YouTube Translate after installation. Explains the free cached demo, account connection, credits, and the first safe test.
---

# Set up YouTube Translate

1. Test the free public demo first. Open `https://api.youtubetranslate.com/demo` or call the demo endpoint for `https://www.youtube.com/watch?v=Ro6MDU2CFO4`. The demo is free and returns a timestamped transcript. The MCP tools themselves require an account, and a cached MCP read costs 2,000 microcredits.
2. Explain that cached MCP reads and new videos require a YouTube Translate account and prepaid credits.
3. For a cached or uncached MCP request, call `get_processing_quote`. Show the maximum cost in microcredits and dollars, plus the current balance. One million microcredits equals one US dollar.
4. Ask the user to approve the maximum before calling `start_video` with `max_credit_units`.
5. If the balance is zero, direct the user to `https://api.youtubetranslate.com/account/payments`. The user approves a one-time Stripe Link payment. Never request payment credentials in chat.
6. After payment, check the account balance before starting work. Never assume a browser redirect granted credit.

Keep the first interaction small: the free browser demo, then one quoted MCP job only if the user approves it.
