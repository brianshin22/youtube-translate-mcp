---
name: setup-youtube-translate
description: Set up YouTube Translate after installation. Explains the free cached demo, account connection, credits, and the first safe test.
---

# Set up YouTube Translate

1. Test the free cached demo first. Call `get_transcript` on `https://www.youtube.com/watch?v=Ro6MDU2CFO4`. This costs nothing and returns a timestamped transcript.
2. Explain that new or uncached videos require a YouTube Translate account and prepaid credits.
3. For an uncached video, call `get_processing_quote`. Show the maximum cost in microcredits and dollars, plus the current balance. One million microcredits equals one US dollar.
4. Ask the user to approve the maximum before calling `start_video` with `max_credit_units`.
5. If the balance is zero, direct the user to `https://api.youtubetranslate.com/account/payments`. The user approves a one-time Stripe Link payment. Never request payment credentials in chat.
6. After payment, check the account balance before starting work. Never assume a browser redirect granted credit.

Keep the first interaction small: one cached transcript, then one quoted job only if the user approves it.
