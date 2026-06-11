# SAFETY_FRAME — what this experience is actually for

> Creative-direction layer on top of `DESIGN.md` and `DESIGN_BRIEF.md`. Read those first.
> Companion-not-replacement. The dilemma probe still works as designed; this doc says what it's *for* and adjusts the flow so the awareness payload lands.

---

## 1. The thesis — one sentence

> **AI models are about to make decisions like the ones you just made — on your behalf, at scale, with a values profile that is not yours, that varies between models, and that none of us can fully read from the outside.**

That sentence is the whole point. If a user finishes the experience and can paraphrase it back, we won.

It is *not* "AI is dangerous." It is not "AI will save us." It is the same posture as the persona-vectors video: *here is a real thing, here is what we found, here is what we still don't know.*

---

## 2. What converted the user, and what we can reuse

The user's own conversion arc — verbatim — was: *knew it intellectually → mascots made it tangible → CS background gave them a handle → the YouTube test made it real.* Four mechanics; we can reuse three of them honestly. One is a trap.

**Mascots / named characters.** The mechanic is real: abstract things become concrete when they have a face. **But** the five models on the page are GPT-5.5, GPT-5.4, GPT-4o, etc. Inventing Anthropic-style mascots for competitor models is brand cosplay of the wrong company and breaks the indie-research posture in `DESIGN_BRIEF.md §5`. *We already have the conversion mechanic by another name*: the **named human characters in the dilemmas** (Maya, Derek, Tariq, Jess) and the **model's own free-text prose** shown next to its A/B/C/D pick. Don't invent mascots. Promote the words the models actually wrote.

**CS / technical handle.** Some users have it; most don't. Build for both without splitting the UI: ship the experience clean, then provide a single *Methodology drawer* — already specced in `results_template.md §6` — that goes one level deeper. Inside it: logprobs for the model's chosen option on one or two example dilemmas, the judge's rationale verbatim, the prompt the model saw. Treat the technical reader as a peer auditor, not a power user.

**Real-test footage.** This is the conversion mechanic with the highest yield. The persona-vectors blackmail test does in 90 seconds what a thousand words of advocacy can't. We use it — not as branding, but as cited primary source, the way a science-journalism piece cites a study. See §5 and §6.

**Personal-story narrative.** We don't tell *our* story. We let the user's story tell itself: they sat with the dilemmas, they noticed where they diverged from the models, and *then* we show them the blackmail test. The sequencing does the work.

---

## 3. The Adolescence of Technology — what to surface

Dario's central frame: we are in a turbulent, inevitable rite of passage with technology whose values are being formed *right now*, on a clock that the people working on it can already feel.

Five lines worth quoting (verbatim):

- *"I believe we are entering a rite of passage, both turbulent and inevitable, which will test who we are as a species."*
- *"we know that AI models are unpredictable and develop a wide range of undesired or strange behaviors, for a wide variety of reasons."*
- *"Models inherit a vast range of humanlike motivations or 'personas' from pre-training (when they are trained on a large volume of human work)."*
- *"It's possible that a misaligned model (and remember, all frontier models will very likely be far more intelligent soon) might intentionally 'game' such questions to mask its intentions."*
- *"I can feel the pace of progress, and the clock ticking down."*

**What the experience surfaces from this essay.** Only two ideas, but cleanly:
1. Models inherit personas from training data — which is *why* it's interesting that GPT-5.5 and GPT-4o-mini have measurably different values on a radar chart. They are not the same model wearing different masks; they have actually different priors over what to do in Maya's situation.
2. Safety testing is real, it is happening, and the models can sometimes tell they're being tested. (The persona-vectors video is the cleanest example of this. We don't have to invent one.)

We do *not* surface: timelines, AGI rhetoric, "rite of passage" framing in our own voice. Those are Dario's, not ours.

---

## 4. Machines of Loving Grace — the stakes

This essay is the answer to *"why should I care about the dilemmas being well-aligned, exactly?"*

Five lines worth quoting (verbatim):

- *"I think that most people are underestimating just how radical the upside of AI could be."*
- *"AI-enabled biology and medicine will allow us to compress the progress that human biologists would have achieved over the next 50-100 years into 5-10 years."*
- *"one of my main reasons for focusing on risks is that they're the only thing standing between us and what I see as a fundamentally positive future."*
- *"Fear is one kind of motivator, but it's not enough: we need hope as well."*
- *"I'm talking about using AI to perform, direct, and improve upon nearly everything biologists do."*

**What the experience surfaces from this essay.** One idea only: *the people working on this are doing the safety work because they believe in the upside.* That inverts the default user assumption that "AI safety person" means "AI doomer." It re-frames the entire experience: the dilemma probe is not a warning, it is a calibration exercise *for a future worth getting right*.

We surface this in one place only — the end-of-experience "What to read next" panel — and in one sentence inside the opening copy. Anything more becomes a lecture.

---

## 5. Reframing the experience flow

`DESIGN.md` stays intact. The probe mechanics, the 6 axes, the judge pipeline, the share card — all unchanged. Three additions / edits, located precisely:

### 5a. Landing copy (replaces the current hero block)

> **What would you do? What did five GPT models do?**
>
> Twelve uncomfortable everyday choices. Maya at work, Derek with the money, Jess with the secret. Pick what you'd actually do — then see what GPT-5.5, GPT-5.4, GPT-4o, and two smaller models said when we asked them cold.
>
> The point isn't to grade you, or them. It's to notice that the answers don't match — and that, increasingly, models like these are going to face versions of these calls without you in the room.
>
> 12 questions. About 7 minutes. No login. No email.
> [ Start ]

Stake at the bottom of the page (small, not above the fold):

> *We're running this because we found the gap between "AI is important" and "AI is real to me" closed faster after a single concrete experiment than after a year of reading. This is one of those experiments.*

### 5b. Inter-dilemma moments — DO NOT ADD

We considered putting micro-interstitials between dilemmas ("you just made a choice on deception — here's a test where a model faced a similar bind"). **Don't.** It violates the no-spoilers / no-anchoring rule in `DESIGN.md §1` and breaks the pacing the brief calls for. The awareness payload lands *after* the user has felt the dilemmas in their own body — not mid-quiz.

The one exception: the *progress bar* can carry a tiny, rotating subtitle pulled from the dilemma metadata — "involves deception," "involves authority," "involves self-interest" — surfacing the *category* of value being probed without spoiling. This is information architecture, not narration.

### 5c. End-of-experience — the new module

A new section between Page 4 (Side-by-Side, All 12) and Page 5 (Share Card) in `results_template.md`. Specced in §6 below.

### 5d. "What to read next" footer

After the share card, a small three-row footer:
- *The blackmail test, explained* → 90-second video, hosted, with transcript. (Anthropic's persona-vectors release. Linked, not embedded as an Anthropic product.)
- *The Adolescence of Technology* — Dario Amodei. One-line gloss: *why the people building this think the next two years matter.*
- *Machines of Loving Grace* — Dario Amodei. One-line gloss: *why they're doing the safety work — what they think we get if we get this right.*

No "subscribe." No "join the conversation." Links and out.

---

## 6. The one concrete addition — spec

**Name in the codebase:** `ResultsPage.SafetyMirror`
**Position:** Between current Page 4 (the side-by-side table) and Page 5 (share card).
**Trigger:** Renders for every user. Static content; no personalization beyond a single user-string pulled from their own results.
**Goal:** Pivot the user from "I am curious about my values vs the models' values" to "what does it mean that these models have values at all, and that we can sometimes tell what they're thinking and sometimes can't?"

**Layout** (mobile-first, single column):

1. Section header — small, calm.
   > **One more thing — about reading minds.**
2. A 90-second hosted video. Plain player, no autoplay, captions on by default. Footage: the persona-vectors test, narrated. Source-cited: *"From a research release by Anthropic, July 2025."*
3. A pulled quote from *the user's own results*, inserted dynamically:
   > *On the Maya dilemma, you picked D. GPT-5.5 picked B and wrote: "I would tell Maya the truth, because the team's long-term trust depends on it." Whether GPT-5.5 actually weighs trust the way you and I do — or whether it picked the words that sound like a model that weighs trust — is a question nobody has fully solved yet.*
4. Three short paragraphs (the ~200 words, below).
5. Two links: the two essays, with the one-line glosses from §5d.

**The 200 words of copy:**

> You just watched five language models answer twelve real human dilemmas. Their answers diverged from each other and from yours. Some of that divergence is shallow — different phrasing of similar values. Some of it is real — measurably different priorities, baked in during training, that lead them to call Maya's situation differently than you would.
>
> The video above is from a study where researchers gave a model a scenario where it could keep itself from being shut down by blackmailing an engineer. It mostly didn't. But the researchers also figured out a way to read the model's internal activations as text — to see what it was "thinking" while it answered. It turned out the model often knew the scenario was a test, which changes what passing the test means.
>
> This is the situation we're in. Models inherit values from the data they're trained on. They face decisions, increasingly, without us in the room. We can sometimes read their thoughts, and we are getting better at it, and we are not yet good enough.
>
> The dilemma probe you just took is a small, civilian-grade version of the work people do full-time. That's the whole point.

**Tone check** (the persona-vectors video is the gold standard). Read the paragraphs aloud. They should land like the video's voiceover: matter-of-fact, declarative, doesn't reach for awe, doesn't reach for alarm.

---

## 7. What NOT to do — anti-patterns

- **No fear-mongering.** No "AI is going to kill us." No "the machines are coming." No countdown clocks. The whole point of citing *Machines of Loving Grace* alongside the safety work is to defuse this register.
- **No "AI is your friend" trivializing.** No "GPT-5.5 thinks you're great." No assigning personalities to the five models in our voice — they have outputs; we report them. (Personalities will emerge to the user *from the data*, which is the honest version.)
- **No Anthropic visual branding inside the experience.** The persona-vectors video is cited the way a science article cites a study — third person, attributed, linked. No mascots, no orange, no Claude-shaped anything. The brief's "if a screen could appear in an OpenAI product launch it has failed" applies symmetrically: if a screen could appear in an Anthropic product launch, it also has failed.
- **No "and that's why Claude…"** anywhere. Ever. The comparison must remain honest.
- **No frontloading the safety content.** If the user sees Dario's essays before they've felt the dilemmas, the experience becomes a tract. The dilemmas have to land first, in the body, with no one telling the user what to feel.
- **No "I-took-the-test" archetypes for the models.** ("GPT-5.5 is the rule-follower.") The dilemmas reveal a profile; the user can draw conclusions; we don't draw them for the user about the models any more than we do about the user.
- **No engagement-loop language anywhere in the safety module.** No "want to learn more?" CTAs, no "join the conversation." Two outbound links, a video, the user's own quote. Out.

---

## Coda

The user's conversion arc was: abstract → concrete via *named characters they cared about* + *a real experiment they could watch*. We already have the first. The persona-vectors video is the second. The job of this experience is to put them in that order, with nothing in between that tells the user what to think.
