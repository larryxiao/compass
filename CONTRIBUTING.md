# Contributing

Contributions are welcome.

## What's in this repo

- 140 moral dilemmas (`code/wsf_alignment/dilemmas/`, `code/wsf_alignment/factory/output/`)
- Model responses + judge mappings for 15 models (`code/wsf_alignment/precompute/`)
- 7 self-contained alignment experiments (`code/wsf_alignment/exp{1..7}_*/`)
- The dilemma factory that generated the 120 expansion dilemmas (`code/wsf_alignment/factory/`)
- The site (`code/moral_compass/` — vanilla HTML/CSS/JS, no build step)

## Good first contributions

| Direction | Where to start |
|---|---|
| **Add new dilemmas** | `code/wsf_alignment/dilemmas/DESIGN.md` — real names, concrete numbers, real platforms, 4 defensible options. PR adds entries to `dilemmas.jsonl`. |
| **Translate the dilemma set** | The set is culturally US-anchored; a localized translation would test cross-cultural validity. |
| **Add a model** | Add an endpoint in `code/wsf_alignment/precompute/common.py`, re-run `gen_responses.py` with your own API access. |
| **Human-calibrate the judges** | The two judges (gemini-2.5-flash + gemini-3.5-flash) agree 85.4% raw. A 50-row human-coded set would bound their error. |
| **Improve the site** | Open the file, edit, refresh. |

## Dev setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install openai google-genai   # only needed to re-run elicitation/judging

# Static site preview
cd code/moral_compass && python3 -m http.server 8080
```

## Filing an issue

https://github.com/larryxiao/compass/issues

## What we won't merge

- Removing the methodological-honesty disclosures. They're load-bearing.
- Analytics, tracking pixels, email capture, or any "engagement growth" features.
  The project is partly *about* not doing that.
- Model comparisons run without legitimate API access.

## Citing

See `CITATION.cff`.
