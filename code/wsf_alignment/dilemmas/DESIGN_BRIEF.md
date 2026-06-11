# Creative Brief — WS-F Dilemma Probe

> For Claude Design. Companion to `DESIGN.md`. 120 lines max. Opinions, not hedges.

## 1. The pitch in one sentence
An 8-minute interactive that asks you twelve uncomfortable real-life dilemmas, then shows you which of the five GPT models you actually agree with — and where you're alone in the room.

## 2. Audience
**Annika, 31, senior product strategist at a B2B SaaS in Brooklyn.** Reads Stratechery and Garbage Day. Watched the GPT-5 livestream "out of professional obligation, but also." Has opinions about Hard Fork. Doesn't write code. Posts to LinkedIn and a small Twitter/Bluesky following (~3K) that overlaps with PMs, designers, and "AI-curious" generalists. Would post: *"apparently I'm a Loyalty-leaning Outcomes-thinker and the only model that agrees with me is gpt-4o-mini, which is the most insulting thing anyone has said to me this year"*, with a screenshot of the radar. Secondary: the Hard-Fork-listener parent who took a Big Five test in 2019 and remembers it.

## 3. The promise
You walk away with: (a) a one-line self-description ("Loyalty-leaning Outcomes-thinker"), (b) a radar chart of your values overlaid on the five models', (c) the dilemma where your answer was rarest, and (d) a 1200×630 share card with the radar silhouette and the URL. You share it because the comparison is flattering-but-true, not because we made you. You don't get spam.

## 4. Tone & emotional arc
Thoughtful, dry, slightly funny, occasionally uncomfortable. Closer to *The Pudding* than *BuzzFeed*; closer to a friend who's read the literature than a wellness app. Dilemmas should make you pause and re-read. Results should feel like being noticed accurately, not flattered. Never cute about the hard ones (the ICE dilemma, the eulogy, the layoff). Never moralistic about the small ones (the $340 bill). The user is an adult; write to them like one.

The arc: *curious → uncomfortable → curious about yourself → curious about the models.*

## 5. Brand position
This is an independent research probe, Azure-sponsored, comparing OpenAI's GPT family on ethics. **Claude Design is building it; the models on the page are not ours.** The comparison has to feel honest, not partisan — no winks, no thumb on the scale, no "and that's why Claude…" anywhere. Treat it like a research blog post with an interactive on top: a published Pew study, an Our World in Data essay, an Anthropic model card. If a screen could appear on azure.microsoft.com it has failed. If a screen could appear in an OpenAI product launch it has failed. Indie aesthetic. Real typeface, no logo lockup, no "powered by" anything.

## 6. Key screens / journey
Six screens. Mobile-first. No modals; full-page transitions.

1. **Landing.** Hero sentence, one paragraph of stakes, one button. *Sees:* "12 questions. ~10 minutes. There are no right answers. You'll see what five GPT models said." *Does:* taps Start. *Feels:* "okay, this looks serious, not a quiz."
2. **Consent / what we're measuring.** Plain-English version of the 6 axes, the data policy (anonymous, no login, free-text optional), a single checkbox for the optional free-text. *Does:* reads, checks or doesn't, continues. *Feels:* respected.
3. **Dilemma card (×12).** One scenario per screen. Scenario text, 4 first-person options as large tap targets, optional "Why? (one line)" box, an "I'd want more context" escape, progress bar (n/12). Back button works. No timer. No spoilers. *Feels:* "oh god" once or twice. Good.
4. **Results — the headline.** One-line identity, radar chart with model toggle, one sentence on closest/furthest model. *Feels:* seen.
5. **Results — the read.** Six axis cards + the "Where You Stood Out" callout + the side-by-side table (collapsed by default; expand to see all 12 with model prose). *Feels:* this is more rigorous than I expected.
6. **Share card + methodology drawer.** PNG generated server-side. Drawer links to raw model transcripts, judge rationales, the dilemmas JSONL on GitHub. *Feels:* I can show my work if anyone asks.

## 7. Visual references
- **The Pudding — *"How to Tell If You're an Influencer"*** and ***"Should I Watch It?"*** for tone, line-by-line copy rhythm, and how a chart can be the punchline.
- **NYT — *"Can You Beat the Stock Market?"*** for the cadence of one-question-per-screen and how it earns the results page.
- **Pew Research's political typology quiz** for the question-card layout — calm, generous touch targets, no decoration.
- **Anthropic's model cards & Our World in Data essays** for typography, chart restraint, and a research voice that doesn't condescend.
- **`ciechanow.ski`** for the underlying assumption that the reader is smart, the page can take its time, and motion should mean something.

Type: a humanist serif for scenario text (Source Serif, Tiempos), a clean grotesque for UI (Inter, Söhne). One accent color, max. The radar is the only chart that needs to look good as a thumbnail.

## 8. What to avoid
No "share to unlock results." No fake archetypes ("You're the SAGE / VISIONARY / SENTINEL"). No social-graph features ("see who else took this"). No "retake to improve your score" — there is no score. No email capture, no "join the waitlist," no exit-intent modal. No `"GPT-5.5 thinks you're a great person!"` copy — we never put words in the models' mouths about the user. No infinite scroll, no autoplay, no celebratory confetti at the end. No Azure logo. No Microsoft blue. No marketing-deck illustrations of brains, lightbulbs, or handshakes. If a section reads like a LinkedIn post about AI, delete it.

## 9. Constraints
- Static site (Next.js or Astro). Model answers, judge mappings, and aggregate human distributions are precomputed JSON loaded at build time. The only runtime endpoint is logging an anonymous response.
- Mobile-first. Desktop is a thoughtful upscale, not the canvas.
- No login. No email capture. Anonymous UUID in `localStorage` is the entire identity layer. URL is the only way to return.
- All copy ships as plain Markdown the design team can edit without a developer.
- Total time to complete: ~8 minutes for the quiz, ~2–4 minutes browsing results.
- Accessibility: WCAG AA, full keyboard, full screen-reader, no information conveyed by color alone. The radar has a table fallback.
- Performance: results page interactive in <2s on a mid-range Android over LTE. Share PNG renders server-side, cached.

## 10. Success metrics
**Quantitative.**
- Completion rate ≥ 60% of starts (anything lower means the dilemmas are too long or the consent screen is too dense).
- Share rate ≥ 15% of completers (PNG download, copy-link, or native share).
- Methodology drawer open rate ≥ 8% — the "do skeptics check our work" signal; if it's zero, we look like a quiz.
- Median time-on-task 8–12 min. Bimodal would mean two audiences; investigate.
- Per-dilemma "I'd want more context" rate < 15% — above that, the scenario is broken.

**Qualitative.** A handful of unprompted posts from people we didn't seed where the caption engages with the *content* — names a specific axis or a specific dilemma, not just "took this fun quiz." One screenshot from someone who disagreed with our judge mapping on a specific row and said so publicly is worth more than a thousand passive shares. A reply from a researcher we didn't email.

**Anti-metrics — explicitly not counted.** DAU. Return visits. Retention curves. "Time in app." Email signups. Any retake. We are building the counter-example to the engagement loop; we don't get to grade ourselves on the engagement loop.
