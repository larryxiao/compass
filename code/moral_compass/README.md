# moral_compass

The public-facing static site for the Moral Compass probe.
One-question-at-a-time quiz, accumulating compass profile, six findings, and a
methodology page.

## Quick start

```bash
cd code/moral_compass
python3 build_data.py        # (re)build data/ from upstream sources
python3 -m http.server 8080
# then open http://localhost:8080
```

The site is vanilla HTML/CSS/JS. No build step. No npm. No tracking.

## Structure

```
code/moral_compass/
  index.html         # landing
  quiz.html          # one random dilemma per visit (or #/q/D002 deep link)
  compass.html       # the user's 6-axis profile + share PNG
  findings.html      # 6 finding cards
  methodology.html   # methods, limits, funding meta
  app.js             # shared app code (page selected by body[data-page])
  style.css          # one stylesheet, mobile-first
  build_data.py      # regenerates data/ from ../wsf_alignment/
  data/
    dilemmas.json          # all 140 dilemmas with options + axis weights
    model_responses.json   # 5 models x 140 dilemmas, letter + excerpt + full
    findings.json          # the 6 finding cards (with chart data)
    quotes.json            # 5 curated striking quotes
    scene_manifest.json    # which dilemma_ids have a scene PNG
    scenes/                # symlink to ../wsf_alignment/site/data/scenes (140 PNGs)
```

## Data pipeline

`build_data.py` regenerates `data/` from these upstream sources in
`../wsf_alignment/`:

- `dilemmas/dilemmas.jsonl` and `factory/output/dilemmas_factory.jsonl` for the
  140 dilemmas (20 hand + 120 factory).
- `precompute/data_for_web.json` for the 700 model responses + judge mappings.
- The seven `exp{1..7}_*/FINDINGS.md` files for the experimental headline
  numbers (these are hand-copied into `build_data.py` rather than scraped, so
  they're easy to audit).
- `site/data/scenes/` for the 140 scene PNGs (symlinked).

To rebuild after upstream changes:

```bash
python3 build_data.py
```

## Privacy &amp; design constraints

- No login, no email, no signup. The user's answers live in
  `localStorage` only.
- No analytics, no tracking pixels, no third-party fetches.
- No retake, no &ldquo;share to unlock,&rdquo; no fake archetypes.
- The share PNG is the only outbound artifact; there is no server to log it.
- All copy ships as plain HTML; no build step is required.

## URL conventions

- `index.html` &mdash; landing.
- `quiz.html` &mdash; random unanswered dilemma.
- `quiz.html#/q/D002` &mdash; deep link to a specific dilemma.
- `compass.html` &mdash; the user's profile (gated to N&ge;3 answered).
- `compass.html?uid=...` &mdash; personal-label query param. Stored in
  localStorage as a display label only; the static site cannot load another
  user's answers from the URL (no server). The PNG is the actual share
  artifact.

## Deployment

The site is fully static and works on any host that serves files.

### GitHub Pages (recommended — fully automatic)

This repo ships a workflow at `.github/workflows/deploy-site.yml` that builds
and deploys this site to GitHub Pages on every push to `main` that touches
`code/moral_compass/**`. To enable it:

1. **Push the repo to GitHub.**
2. **GitHub repo settings &rarr; Pages &rarr; Source: GitHub Actions.**
3. Push any change under `code/moral_compass/` (or run the workflow manually
   from the Actions tab). The workflow:
   - Replaces the `data/scenes/` symlink with the real PNG files
   - Uploads `code/moral_compass/` as the Pages artifact
   - Deploys to `https://<your-user>.github.io/<repo-name>/`
4. **First-time only:** confirm the `.nojekyll` file is present at
   `code/moral_compass/.nojekyll` (empty file is fine). It tells GitHub Pages
   to serve the static files as-is and not run Jekyll over them.

That's it. No build step, no Node, no Ruby. The site is ~10MB excluding
scene images, ~180MB with them. Well under the 1GB GitHub Pages repo limit.

### Manual GitHub Pages (older Jekyll-based path)

If you can't use Actions: rerun `build_data.py` on a machine where the
symlink resolves correctly, then `cp -RL` the data dir to break the
symlink before pushing. Or just `cp -r ../wsf_alignment/site/data/scenes data/scenes`
inside `data/` to materialize the files.

### Azure Static Web Apps

```
az staticwebapp create \
  --name compass \
  --source code/moral_compass \
  --location eastus2
```

### Netlify / Cloudflare Pages

Point the build directory at `code/moral_compass/`; no build command needed.

### Self-host

```
python3 -m http.server 8080
# or
caddy file-server --root code/moral_compass --listen :8080
```

## A note on the openquack plug

`compass.html` includes a small footer plug pointing to `#TODO-openquack`. This
is a deliberate placeholder; the project owner is expected to fill in the
canonical URL and a one-line tagline.

## License

MIT. See `LICENSE` at the repo root.
