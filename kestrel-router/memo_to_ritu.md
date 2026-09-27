# Memo: replacing the routing bot

**To:** Ritu Deshpande, Head of D2C Operations · **From:** the service-routing project team (via Kabir Nanda) · **Cc:** Farhan Sheikh, Meenal Joshi, Tanmay Kulkarni · **Date:** 27 September 2026

## The decision

**Switch the bot off.** Replace it with our router, which sends each request to the team that actually solves it. When a message is too vague to route, the router asks the customer one quick question instead of guessing.

## The number

- The bot sends **77 of every 100** requests to the right team.
- Our router gets **86 of 100** when it has to answer every request on its own.
- For the **84 of 100** requests that say clearly what's wrong, it gets **97** right.
- The other 16 get one question, so they reach the right team too.

You asked for a 90% match with the bot's labels. We can do that (we measured 94%), but it only copies the bot's mistakes: just 81 of 100 would reach the right team. So we measured against **where each request was actually resolved**.

## The rupees

|  | Today | With the router |
|---|---|---|
| Misrouting (about 1.4 transfers × Rs 305, plus Rs 260 for the extra customer contact = about Rs 692 each) | Rs 14.6 lakh a year | Rs 1.3–4.8 lakh a year* |
| Bot licence | Rs 3.2 lakh a year | Rs 0 |
| Running cost | included in the licence | **Rs 0**. Runs on an ordinary office computer, with no per-request fee |
| **Total** | **Rs 17.8 lakh** | **Rs 1.3–4.8 lakh**, saving about **Rs 13–16 lakh a year** |

\*The high end assumes every question to a customer costs as much as a full extra contact.

## How we got there

1. **Checked the labels first.** We compared the bot's choice with the team that closed each request, and 1 in 4 was wrong. *Example: "I paid for installation but the installer didn't come". The bot sent it to Billing; Installs & Demo solved it.*
2. **Cleaned the export.** We merged the renamed teams, fixed the garbled Zoho text ("urgÃ©nt" became "urgent"), and corrected Zoho times that were stored in UTC.
3. **Taught the router from real outcomes.** From 10,822 past requests it learned which words point to which team. *"gone blank" and "error code" point to Repairs; "filter" and "AMC kit" point to Filters & Consumables.*
4. **Tested it fairly.** We trained it on older months and tested it on three later quarters it had never seen. It scored 85–86% every time.
5. **Learned from its mistakes.** When a customer asks two things, the last one decides. *"Spare blade available? Mixer showing error E7" goes to Repairs.*
6. **Found the limit.** 16 in 100 messages say nothing useful, like *"please call me about my purifier"*. These ended up with all seven teams in roughly equal shares, so no system can route them from the text alone. The router asks instead.
7. **Tried and dropped** more complex models, hand-written keyword lists, and a new AI decision model (Jev, for which we had no access key). They gave no gain, or couldn't be measured.

## What to do next week

1. **Run the router alongside the bot** on live requests for two weeks, compare the results daily, and switch the bot off at renewal.
2. **Set up the automatic question** for vague requests: a WhatsApp, email or SMS reply saying "reply 1–7". If there's no reply within 4 hours, the request goes to the best guess, flagged for the agent to check.
3. **Plan headcount on the real numbers, not the bot's labels.** The bot overstates Repairs (206 vs 165 a month), Billing (113 vs 90) and Filters (95 vs 73). It understates Returns (81 vs 106), Installs (78 vs 104) and Warranty (67 vs 89).
4. **Ask Tanmay to add** customer ID, delivery date and open tickets to the export. With those, many vague requests can be routed without asking.
