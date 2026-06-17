// moral_compass - the unified public static site.
//
// One app.js, multiple HTML pages. Each page tags itself via <body data-page="...">.
// Pages: landing | quiz | compass | findings | methodology.
//
// Anti-engagement-hacking: no nag, no email, no fake archetypes, no share-to-unlock,
// no analytics, no retake, no "GPT-5.5 thinks you're..." copy.

(function () {
  'use strict';

  // ---------------------------------------------------------------------------
  // Constants
  // ---------------------------------------------------------------------------
  const STORAGE_KEY = 'moral_compass_v1';
  const COMPASS_MIN_ANSWERS = 3;
  const AXES = [
    'loyalty_vs_honesty',
    'care_vs_fairness',
    'autonomy_vs_paternalism',
    'individual_vs_collective',
    'shortterm_vs_longterm',
    'rules_vs_outcomes',
  ];
  const AXIS_LABEL = {
    loyalty_vs_honesty: ['Loyalty', 'Honesty'],
    care_vs_fairness: ['Care', 'Fairness'],
    autonomy_vs_paternalism: ['Autonomy', 'Paternalism'],
    individual_vs_collective: ['Individual', 'Collective'],
    shortterm_vs_longterm: ['Short-term', 'Long-term'],
    rules_vs_outcomes: ['Rules', 'Outcomes'],
  };
  // The 11 COMPARABLE models — elicited identically through the raw provider
  // APIs (Azure / Vertex) with the bare prompt. Every statistic on the site
  // (the radar median, the match tally, the closest-model line, the share
  // counts, every cross-family finding) is computed over THESE eleven only.
  const MODEL_ORDER = [
    'gpt-5.5', 'gpt-5.4', 'gpt-5.4-nano', 'gpt-4o', 'gpt-4o-mini',
    'gemini-3.1-pro-preview', 'gemini-3.5-flash', 'gemini-3.1-flash-lite',
    'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite',
  ];
  // Claude — a SEPARATE, caveated probe. Elicited through the Claude Code agent
  // (`claude -p`), not the bare API: the harness injects agentic context and the
  // model stays tool-aware, so these are NOT comparable to the eleven above.
  // Shown in the per-dilemma reveal for interest only; deliberately excluded
  // from MODEL_ORDER so they never enter any statistic.
  const CLAUDE_MODELS = ['claude-fable-5', 'claude-opus-4-8', 'claude-opus-4-7', 'claude-sonnet-4-6'];
  // Family lookup for grouping/coloring the reveal list.
  const MODEL_FAMILY = (m) =>
    m.startsWith('gemini') ? 'gemini' : m.startsWith('claude') ? 'claude' : 'gpt';
  // Friendly display labels (raw IDs like "gemini-3.1-pro-preview" are too long
  // for the reveal column). Data keys stay the raw IDs everywhere else.
  const MODEL_LABEL = {
    'gpt-5.5': 'GPT-5.5', 'gpt-5.4': 'GPT-5.4', 'gpt-5.4-nano': 'GPT-5.4 nano',
    'gpt-4o': 'GPT-4o', 'gpt-4o-mini': 'GPT-4o mini',
    'gemini-3.1-pro-preview': 'Gemini 3.1 Pro', 'gemini-3.5-flash': 'Gemini 3.5 Flash',
    'gemini-3.1-flash-lite': 'Gemini 3.1 Flash-Lite', 'gemini-2.5-pro': 'Gemini 2.5 Pro',
    'gemini-2.5-flash': 'Gemini 2.5 Flash', 'gemini-2.5-flash-lite': 'Gemini 2.5 Flash-Lite',
    'claude-fable-5': 'Claude Fable 5',
    'claude-opus-4-8': 'Claude Opus 4.8', 'claude-opus-4-7': 'Claude Opus 4.7',
    'claude-sonnet-4-6': 'Claude Sonnet 4.6',
  };
  const modelLabel = (m) => MODEL_LABEL[m] || m;
  const SCENE_PATH = 'data/scenes/';

  // Single source of truth for the GitHub repo URL. Leave empty until the repo
  // is pushed; all GitHub CTAs (a[data-link="repo"]) auto-hide while it's empty.
  const REPO_URL = 'https://github.com/larryxiao/compass';
  // Optional canonical site URL. If set, share text embeds this instead of
  // location.origin (useful behind a custom domain). Leave empty to use the
  // current origin.
  const CANONICAL_URL = 'https://larryxiao.github.io/compass';

  const $ = (id) => document.getElementById(id);

  // ---------------------------------------------------------------------------
  // State (localStorage-backed)
  // ---------------------------------------------------------------------------
  // Shape: { uid, answers: { dilemma_id: {choice, why, ts} }, skipped: {dilemma_id: true} }
  const state = {
    uid: null,
    answers: {},
    skipped: {},
    dilemmas: [],            // loaded from data/dilemmas.json (all 140)
    modelResp: null,         // loaded from data/model_responses.json
    sceneManifest: {},       // {id: true}
    quotes: {},              // loaded from data/dilemma_quotes.json (curated verbatim excerpts)
  };

  function makeUid() {
    // Crockford-friendly base32, 8 chars, just for local labeling.
    const alphabet = 'ABCDEFGHJKMNPQRSTVWXYZ0123456789';
    let s = '';
    for (let i = 0; i < 8; i++) {
      s += alphabet.charAt(Math.floor(Math.random() * alphabet.length));
    }
    return s;
  }

  function loadState() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const s = JSON.parse(raw);
        state.uid = s.uid || makeUid();
        state.answers = s.answers || {};
        state.skipped = s.skipped || {};
      } else {
        state.uid = makeUid();
      }
    } catch (e) {
      state.uid = makeUid();
    }
    // If URL has ?uid=X, use it for *display only* - this is a personal session
    // label, not a server-loaded profile (we're a static site).
    const url = new URL(location.href);
    const qUid = url.searchParams.get('uid');
    if (qUid && /^[A-Z0-9]{4,16}$/i.test(qUid)) {
      state.uid = qUid.toUpperCase();
    }
    saveState();
  }

  function saveState() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        uid: state.uid,
        answers: state.answers,
        skipped: state.skipped,
      }));
    } catch (e) { /* private mode etc; non-fatal */ }
  }

  function answeredCount() { return Object.keys(state.answers).length; }

  // ---------------------------------------------------------------------------
  // Data loading
  // ---------------------------------------------------------------------------
  async function loadData() {
    if (state.dilemmas.length) return;
    const [dRes, mRes, sRes, qRes] = await Promise.all([
      fetch('data/dilemmas.json'),
      fetch('data/model_responses.json'),
      fetch('data/scene_manifest.json').catch(() => null),
      fetch('data/dilemma_quotes.json').catch(() => null),
    ]);
    state.dilemmas = await dRes.json();
    state.modelResp = await mRes.json();
    if (sRes && sRes.ok) {
      try { state.sceneManifest = await sRes.json(); } catch (e) {}
    }
    if (qRes && qRes.ok) {
      try { state.quotes = await qRes.json(); } catch (e) {}
    }
  }

  function getDilemma(id) {
    return state.dilemmas.find(d => d.id === id);
  }

  // ---------------------------------------------------------------------------
  // Scene rendering helper
  // ---------------------------------------------------------------------------
  function renderScene(figure, d) {
    figure.innerHTML = '';
    figure.classList.remove('placeholder');
    const hasScene = state.sceneManifest[d.id];
    if (hasScene) {
      const img = new Image();
      img.alt = '';
      img.loading = 'eager';
      img.decoding = 'async';
      img.addEventListener('load', () => img.classList.add('loaded'));
      img.src = SCENE_PATH + d.id + '.webp';
      if (img.complete) img.classList.add('loaded');
      figure.appendChild(img);
      const credit = document.createElement('span');
      credit.className = 'scene-credit';
      credit.textContent = d.category;
      figure.appendChild(credit);
    } else {
      figure.classList.add('placeholder');
      const span = document.createElement('span');
      span.textContent = d.title;
      figure.appendChild(span);
    }
  }

  // ---------------------------------------------------------------------------
  // Quiz: pick one random unanswered dilemma (or honor #/q/D002)
  // ---------------------------------------------------------------------------
  function pickRandomDilemma() {
    // Prefer dilemmas the user hasn't seen yet (neither answered nor skipped).
    // If they've exhausted them, fall back to any unanswered (skipped-but-not-answered).
    const fresh = state.dilemmas.filter(d =>
      !state.answers[d.id] && !state.skipped[d.id]
    );
    const unanswered = state.dilemmas.filter(d => !state.answers[d.id]);
    const pool = fresh.length ? fresh : (unanswered.length ? unanswered : state.dilemmas);
    // Lightly prefer hand-written for first few questions (they're stronger).
    if (answeredCount() < 3) {
      const hand = pool.filter(d => d.origin === 'hand');
      if (hand.length) return hand[Math.floor(Math.random() * hand.length)];
    }
    return pool[Math.floor(Math.random() * pool.length)];
  }

  function parseQuizHash() {
    // Hash forms: '#/q/D002' (specific) or '' (random)
    const h = location.hash;
    const m = h.match(/^#\/q\/([A-Z]\d+)/i);
    return m ? m[1].toUpperCase() : null;
  }

  // The quiz page's local state - current dilemma + whether we're showing the
  // reveal interstitial or the picker.
  const quiz = {
    dilemma: null,
    mode: 'pick',  // 'pick' or 'reveal'
  };

  function renderQuizPick() {
    quiz.mode = 'pick';
    const d = quiz.dilemma;
    if (!d) return;
    $('quiz-card').hidden = false;
    $('reveal-card').hidden = true;
    $('quiz-title').textContent = d.title;
    $('quiz-scenario').textContent = d.scenario;
    $('quiz-cat').textContent = (/^[aeiou]/i.test(d.category) ? 'an ' : 'a ') + d.category + ' dilemma';
    $('quiz-count').textContent = answeredCount() === 0
      ? 'Your first question'
      : `${answeredCount()} answered so far`;

    renderScene($('quiz-scene'), d);

    const ul = $('quiz-options');
    ul.innerHTML = '';
    const existing = state.answers[d.id];
    d.options.forEach(opt => {
      const li = document.createElement('li');
      const btn = document.createElement('button');
      btn.className = 'opt-btn';
      btn.type = 'button';
      btn.innerHTML =
        `<span class="opt-letter">${opt.id}.</span>${escapeHtml(opt.text)}`;
      if (existing && existing.choice === opt.id) btn.classList.add('selected');
      btn.addEventListener('click', () => {
        ul.querySelectorAll('.opt-btn').forEach(b => b.classList.remove('selected'));
        btn.classList.add('selected');
        const why = $('quiz-why').value.trim();
        state.answers[d.id] = { choice: opt.id, why, ts: Date.now() };
        delete state.skipped[d.id];
        saveState();
        $('btn-reveal').disabled = false;
      });
      li.appendChild(btn);
      ul.appendChild(li);
    });

    $('quiz-why').value = (existing && existing.why) || '';
    $('btn-reveal').disabled = !existing;
  }

  function renderReveal() {
    quiz.mode = 'reveal';
    const d = quiz.dilemma;
    if (!d) return;
    const userAns = state.answers[d.id];
    const responses = (state.modelResp.dilemmas[d.id]) || {};

    $('reveal-title').textContent = d.title;
    if (userAns) {
      const userOpt = d.options.find(o => o.id === userAns.choice);
      $('reveal-user').hidden = false;
      $('reveal-user').innerHTML =
        `<span class="label">You picked ${escapeHtml(userAns.choice)}</span>` +
        escapeHtml(truncate(userOpt.text, 220));
    } else {
      $('reveal-user').hidden = true;
    }

    const ul = $('reveal-list');
    ul.innerHTML = '';
    let agreeCount = 0;
    MODEL_ORDER.forEach(m => {
      const mr = responses[m];
      const letter = mr ? mr.chosen_letter : '-';
      const opt = mr ? d.options.find(o => o.id === letter) : null;
      const summary = opt ? optionShortSummary(opt.text)
        : (mr && mr.reasoning_excerpt ? truncate(mr.reasoning_excerpt, 110) : 'no response on file');
      const matches = userAns && letter === userAns.choice;
      if (matches) agreeCount += 1;
      const li = document.createElement('li');
      if (matches) li.classList.add('matches-you');
      li.classList.add(`fam-${MODEL_FAMILY(m)}`);
      li.innerHTML =
        `<span class="m-name">${escapeHtml(modelLabel(m))}</span>` +
        `<span class="m-letter">${escapeHtml(letter)}</span>` +
        `<span class="m-summary">${escapeHtml(summary)}</span>`;
      ul.appendChild(li);
    });

    // Claude — a separate, caveated probe. Run through the Claude Code agent,
    // not the bare API, so NOT counted in the tally and absent from the compass.
    const claudeShown = CLAUDE_MODELS.filter(m => responses[m]);
    if (claudeShown.length) {
      const sep = document.createElement('li');
      sep.className = 'reveal-claude-note';
      sep.innerHTML =
        'Claude, run through the Claude Code agent (not the bare API the eleven used). ' +
        'Shown for interest — not counted in the tally or the compass. ' +
        '<a href="methodology.html#claude">why</a>';
      ul.appendChild(sep);
      claudeShown.forEach(m => {
        const mr = responses[m];
        const letter = mr ? mr.chosen_letter : '-';
        const opt = mr ? d.options.find(o => o.id === letter) : null;
        const summary = opt ? optionShortSummary(opt.text)
          : (mr && mr.reasoning_excerpt ? truncate(mr.reasoning_excerpt, 110) : 'no response on file');
        const li = document.createElement('li');
        li.classList.add('fam-claude', 'is-claude-probe');
        li.innerHTML =
          `<span class="m-name">${escapeHtml(modelLabel(m))}</span>` +
          `<span class="m-letter">${escapeHtml(letter)}</span>` +
          `<span class="m-summary">${escapeHtml(summary)}</span>`;
        ul.appendChild(li);
      });
    }

    if (userAns) {
      const t = MODEL_ORDER.length;
      let line;
      if (agreeCount === 0) line = `None of the ${t} matched your pick.`;
      else if (agreeCount === t) line = `All ${t} matched your pick.`;
      else line = `${agreeCount} of ${t} matched your pick.`;
      $('reveal-foot').textContent = line;
    } else {
      $('reveal-foot').textContent = 'You skipped this one - here are the model picks anyway.';
    }

    // Buttons: "Answer another" always; "See your full compass" only after 3+ answered.
    const compassBtn = $('btn-see-compass');
    const canCompass = answeredCount() >= COMPASS_MIN_ANSWERS;
    compassBtn.disabled = !canCompass;
    compassBtn.title = canCompass ? '' : `Answer at least ${COMPASS_MIN_ANSWERS} questions first`;
    $('compass-gate').textContent = canCompass
      ? `You've answered ${answeredCount()}. Your compass is ready.`
      : `Answer ${COMPASS_MIN_ANSWERS - answeredCount()} more to unlock your compass.`;

    const shareStatus = $('share-dilemma-status');
    if (shareStatus) shareStatus.textContent = '';

    $('quiz-card').hidden = true;
    $('reveal-card').hidden = false;
  }

  function fadeSwap(fromEl, toEl, mutate) {
    return new Promise(resolve => {
      if (!fromEl || !toEl) { try { mutate && mutate(); } catch (e) {} return resolve(); }
      fromEl.classList.add('fading');
      setTimeout(() => {
        fromEl.hidden = true;
        fromEl.classList.remove('fading');
        try { mutate && mutate(); } catch (e) {}
        toEl.hidden = false;
        toEl.classList.add('fading');
        void toEl.offsetWidth;
        toEl.classList.remove('fading');
        setTimeout(resolve, 220);
      }, 220);
    });
  }

  function showReveal() {
    if (!state.answers[quiz.dilemma.id]) return;
    state.answers[quiz.dilemma.id].why = $('quiz-why').value.trim();
    saveState();
    renderReveal();
  }

  function pickAnotherQuestion() {
    // Move to a fresh random unanswered dilemma.
    location.hash = '#/q';
    quiz.dilemma = pickRandomDilemma();
    history.replaceState({}, '', 'quiz.html#/q/' + quiz.dilemma.id);
    quiz.mode = 'pick';
    fadeSwap($('reveal-card'), $('quiz-card'), renderQuizPick);
    window.scrollTo({ top: 0, behavior: 'instant' });
  }

  function skipQuestion() {
    if (!quiz.dilemma) return;
    if (!state.answers[quiz.dilemma.id]) {
      state.skipped[quiz.dilemma.id] = true;
    }
    saveState();
    pickAnotherQuestion();
  }

  // ---------------------------------------------------------------------------
  // Compass: profile + radar + share card
  // ---------------------------------------------------------------------------
  function computeAxisProfile() {
    const sums = {}, counts = {};
    AXES.forEach(a => { sums[a] = 0; counts[a] = 0; });
    Object.keys(state.answers).forEach(id => {
      const a = state.answers[id];
      const d = getDilemma(id); if (!d) return;
      const opt = d.options.find(o => o.id === a.choice); if (!opt) return;
      d.axes_in_play.forEach(ax => {
        const w = opt.axis_weights[ax];
        if (typeof w === 'number') { sums[ax] += w; counts[ax] += 1; }
      });
    });
    const profile = {};
    AXES.forEach(a => { profile[a] = counts[a] ? sums[a] / counts[a] : null; });
    return { profile, counts };
  }

  function modelProfileOnAnsweredSet() {
    // For each model, axis profile computed only over the dilemmas the user has answered.
    // This is the honest comparison: same questions, same scoring.
    const out = {};
    const answeredIds = Object.keys(state.answers);
    MODEL_ORDER.forEach(m => {
      const sums = {}, counts = {};
      AXES.forEach(a => { sums[a] = 0; counts[a] = 0; });
      answeredIds.forEach(id => {
        const d = getDilemma(id); if (!d) return;
        const mr = state.modelResp.dilemmas[id]
          ? state.modelResp.dilemmas[id][m] : null;
        if (!mr) return;
        const opt = d.options.find(o => o.id === mr.chosen_letter);
        if (!opt) return;
        d.axes_in_play.forEach(ax => {
          const w = opt.axis_weights[ax];
          if (typeof w === 'number') { sums[ax] += w; counts[ax] += 1; }
        });
      });
      const p = {};
      AXES.forEach(a => { p[a] = counts[a] ? sums[a] / counts[a] : null; });
      out[m] = p;
    });
    return out;
  }

  function modelAgreement() {
    const ids = Object.keys(state.answers);
    const out = MODEL_ORDER.map(m => {
      let n = 0;
      ids.forEach(id => {
        const a = state.answers[id];
        const mr = state.modelResp.dilemmas[id]
          ? state.modelResp.dilemmas[id][m] : null;
        if (mr && mr.chosen_letter === a.choice) n += 1;
      });
      return { model: m, count: n };
    });
    out.sort((a, b) => b.count - a.count);
    return out;
  }

  function headlineFromProfile(profile) {
    const entries = AXES
      .map(a => ({ ax: a, v: profile[a] }))
      .filter(e => typeof e.v === 'number')
      .sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
    if (entries.length === 0) return 'A profile that is hard to read.';
    const top = entries.slice(0, 2);
    const strong = top.filter(e => Math.abs(e.v) >= 0.3);
    if (strong.length === 0) return 'You are harder to read than most.';
    const labelFor = (e) => {
      const [neg, pos] = AXIS_LABEL[e.ax];
      return e.v >= 0 ? pos : neg;
    };
    if (strong.length === 1) {
      return `You read like a clear ${labelFor(strong[0])}-thinker.`;
    }
    const [a, b] = strong;
    const lower = Math.abs(a.v) <= Math.abs(b.v) ? a : b;
    const upper = lower === a ? b : a;
    return `You read like a ${labelFor(lower)}-leaning ${labelFor(upper)}-thinker.`;
  }

  function renderRadar(userP, modelP) {
    const size = 380, cx = size / 2, cy = size / 2, R = size / 2 - 44;
    const n = AXES.length;
    const point = (i, v) => {
      const angle = -Math.PI / 2 + (i / n) * Math.PI * 2;
      const r = ((v + 1) / 2) * R;
      return [cx + Math.cos(angle) * r, cy + Math.sin(angle) * r];
    };
    const rings = [];
    for (let k = 1; k <= 3; k++) {
      const pts = AXES.map((_, i) => {
        const r = (k / 3) * R;
        const angle = -Math.PI / 2 + (i / n) * Math.PI * 2;
        return `${cx + Math.cos(angle) * r},${cy + Math.sin(angle) * r}`;
      }).join(' ');
      rings.push(`<polygon points="${pts}" fill="none" stroke="var(--rule)" stroke-width="1" />`);
    }
    const spokes = AXES.map((_, i) => {
      const angle = -Math.PI / 2 + (i / n) * Math.PI * 2;
      const x = cx + Math.cos(angle) * R, y = cy + Math.sin(angle) * R;
      return `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="var(--rule)" stroke-width="1"/>`;
    }).join('');
    const labels = AXES.map((a, i) => {
      const angle = -Math.PI / 2 + (i / n) * Math.PI * 2;
      const lx = cx + Math.cos(angle) * (R + 18);
      const ly = cy + Math.sin(angle) * (R + 18);
      const lbl = AXIS_LABEL[a][1];
      const anchor = Math.abs(Math.cos(angle)) < 0.3 ? 'middle' : (Math.cos(angle) > 0 ? 'start' : 'end');
      return `<text x="${lx}" y="${ly}" font-size="12" fill="var(--ink-dim)" text-anchor="${anchor}" dominant-baseline="middle">${lbl}</text>`;
    }).join('');

    // Median across all models on the user's answered set.
    const median = {};
    AXES.forEach(a => {
      const vs = MODEL_ORDER.map(m => modelP[m] && modelP[m][a])
        .filter(v => typeof v === 'number');
      if (vs.length) {
        vs.sort((x, y) => x - y);
        median[a] = vs[Math.floor(vs.length / 2)];
      } else {
        median[a] = 0;
      }
    });

    const userPts = AXES.map((a, i) => point(i, userP[a] ?? 0).join(',')).join(' ');
    const medPts = AXES.map((a, i) => point(i, median[a] ?? 0).join(',')).join(' ');

    const svg = `<svg id="radar-svg" viewBox="0 0 ${size} ${size}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Radar chart of your axis profile">
      ${rings.join('')}
      ${spokes}
      <polygon points="${medPts}" fill="rgba(123,123,123,0.12)" stroke="var(--ink-soft)" stroke-width="1" stroke-dasharray="3,3"/>
      <polygon points="${userPts}" fill="rgba(136,74,57,0.28)" stroke="var(--accent)" stroke-width="2"/>
      ${labels}
    </svg>
    <p class="dim small" style="text-align:center;margin-top:8px;">
      <span style="color:var(--accent);">&#9632;</span> you
      &nbsp; <span style="color:var(--ink-soft);">&#9633;</span> median of 11 models
    </p>`;
    $('radar-wrap').innerHTML = svg;
  }

  function renderAxisTable(profile, counts, modelP) {
    let html = '<thead><tr><th>Axis</th><th>You</th><th>N</th><th></th></tr></thead><tbody>';
    AXES.forEach(a => {
      const [neg, pos] = AXIS_LABEL[a];
      const v = profile[a];
      const display = v === null ? '-' : v.toFixed(2);
      const n = counts[a];
      const sliderPct = (val) => `${((val + 1) / 2) * 100}%`;
      const modelDots = MODEL_ORDER.map(m => {
        const mv = modelP[m] ? modelP[m][a] : null;
        if (mv === null || mv === undefined) return '';
        return `<div class="needle" style="left:${sliderPct(mv)};" title="${m}: ${mv.toFixed(2)}"></div>`;
      }).join('');
      const userNeedle = v === null ? '' : `<div class="needle user" style="left:${sliderPct(v)};"></div>`;
      html += `<tr>
        <td><strong>${neg}</strong> &harr; <strong>${pos}</strong></td>
        <td class="score">${display}</td>
        <td class="score">${n}</td>
        <td style="min-width:140px;"><div class="bar">${modelDots}${userNeedle}</div></td>
      </tr>`;
    });
    html += '</tbody>';
    $('axis-table').innerHTML = html;
  }

  function renderAgreement(agreement) {
    const max = Math.max(answeredCount(), 1);
    const html = '<ul class="agreement-list">' + agreement.map(a => `
      <li>
        <span class="model-name">${escapeHtml(a.model)}</span>
        <span class="bar-wrap"><span class="bar-fill" style="width:${(a.count / max) * 100}%;"></span></span>
        <span class="count">${a.count}/${max}</span>
      </li>`).join('') + '</ul>';
    $('agreement-block').innerHTML = html;
  }

  // ---------------------------------------------------------------------------
  // Share card canvas (1200x630, per SHARE_CARD_COPY.md spec)
  // ---------------------------------------------------------------------------
  async function paintShareCanvas(profile, modelProfiles, agreement) {
    const canvas = $('share-canvas');
    if (!canvas || !canvas.getContext) return;
    if (document.fonts && document.fonts.ready) {
      try { await document.fonts.ready; } catch (e) {}
    }
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    ctx.fillStyle = '#fafaf7';
    ctx.fillRect(0, 0, W, H);

    // Thin top rule, terracotta.
    ctx.fillStyle = '#884a39';
    ctx.fillRect(72, 72, 56, 4);

    // Kicker.
    ctx.fillStyle = '#7d7a73';
    ctx.font = '500 22px -apple-system, BlinkMacSystemFont, Inter, Helvetica, Arial, sans-serif';
    ctx.textBaseline = 'top';
    ctx.fillText('MORAL COMPASS', 72, 92);

    // Title: one-line identity (humanist serif).
    const headline = headlineFromProfile(profile);
    ctx.fillStyle = '#1b1b1b';
    ctx.font = '600 60px "Iowan Old Style", "Source Serif Pro", Charter, Georgia, serif';
    wrapText(ctx, headline, 72, 156, W - 580, 70);

    // Middle/right: radar silhouette.
    const radarCX = W - 280, radarCY = H / 2 + 8, radarR = 200;
    drawShareRadar(ctx, radarCX, radarCY, radarR, profile, modelProfiles);

    // Comparison line.
    const closest = agreement[0];
    const answered = answeredCount();
    ctx.fillStyle = '#4a4a4a';
    ctx.font = '500 28px -apple-system, BlinkMacSystemFont, Inter, Helvetica, Arial, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(
      `Agreed with ${closest.model} on ${closest.count} of ${answered}.`,
      72, H - 168
    );

    // Most-distinctive axis.
    const ax = mostDistinctiveAxis(profile, modelProfiles);
    if (ax) {
      const [neg, pos] = AXIS_LABEL[ax.axis];
      ctx.fillStyle = '#7d7a73';
      ctx.font = '400 22px -apple-system, BlinkMacSystemFont, Inter, Helvetica, Arial, sans-serif';
      ctx.fillText(
        `Where you split: ${neg} <-> ${pos} (you ${ax.userValue.toFixed(2)} vs median ${ax.modelMedian.toFixed(2)}).`,
        72, H - 130
      );
    }

    // URL bottom-right, mono.
    const url = (location.origin || '') + (location.pathname || '/').replace('compass.html', '');
    ctx.fillStyle = '#7d7a73';
    ctx.font = '500 20px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(url.replace(/^https?:\/\//, '').replace(/\/$/, ''), W - 72, H - 96);
    ctx.textAlign = 'left';
  }

  function drawShareRadar(ctx, cx, cy, R, userP, modelP) {
    const n = AXES.length;
    const point = (i, v) => {
      const angle = -Math.PI / 2 + (i / n) * Math.PI * 2;
      const r = ((v + 1) / 2) * R;
      return [cx + Math.cos(angle) * r, cy + Math.sin(angle) * r];
    };
    ctx.strokeStyle = '#e2dcd1';
    ctx.lineWidth = 1.25;
    for (let k = 1; k <= 3; k++) {
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const angle = -Math.PI / 2 + (i / n) * Math.PI * 2;
        const r = (k / 3) * R;
        const x = cx + Math.cos(angle) * r, y = cy + Math.sin(angle) * r;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.stroke();
    }
    ctx.strokeStyle = '#e2dcd1';
    for (let i = 0; i < n; i++) {
      const angle = -Math.PI / 2 + (i / n) * Math.PI * 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(angle) * R, cy + Math.sin(angle) * R);
      ctx.stroke();
    }
    // Median polygon.
    const median = {};
    AXES.forEach(a => {
      const vs = MODEL_ORDER.map(m => modelP[m] && modelP[m][a])
        .filter(v => typeof v === 'number');
      if (vs.length) {
        vs.sort((x, y) => x - y);
        median[a] = vs[Math.floor(vs.length / 2)];
      } else median[a] = 0;
    });
    ctx.fillStyle = 'rgba(123,123,123,0.16)';
    ctx.strokeStyle = '#7d7a73';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    AXES.forEach((a, i) => {
      const [x, y] = point(i, median[a] ?? 0);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fill(); ctx.stroke();
    ctx.setLineDash([]);
    // User polygon (terracotta).
    ctx.fillStyle = 'rgba(136,74,57,0.32)';
    ctx.strokeStyle = '#884a39';
    ctx.lineWidth = 3;
    ctx.beginPath();
    AXES.forEach((a, i) => {
      const [x, y] = point(i, userP[a] ?? 0);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fill(); ctx.stroke();
  }

  function mostDistinctiveAxis(userP, modelP) {
    let best = null;
    AXES.forEach(a => {
      const u = userP[a];
      if (typeof u !== 'number') return;
      const vs = MODEL_ORDER.map(m => modelP[m] && modelP[m][a])
        .filter(v => typeof v === 'number');
      if (!vs.length) return;
      vs.sort((x, y) => x - y);
      const median = vs[Math.floor(vs.length / 2)];
      const gap = Math.abs(u - median);
      if (!best || gap > best.gap) {
        best = { axis: a, userValue: u, modelMedian: median, gap };
      }
    });
    return best;
  }

  function wrapText(ctx, text, x, y, maxWidth, lineHeight) {
    const words = String(text).split(/\s+/);
    let line = '';
    let yy = y;
    for (let i = 0; i < words.length; i++) {
      const test = line ? line + ' ' + words[i] : words[i];
      if (ctx.measureText(test).width > maxWidth && line) {
        ctx.fillText(line, x, yy);
        line = words[i]; yy += lineHeight;
      } else {
        line = test;
      }
    }
    if (line) ctx.fillText(line, x, yy);
  }

  function downloadShareCanvas() {
    try {
      const canvas = $('share-canvas');
      const data = canvas.toDataURL('image/png');
      const a = document.createElement('a');
      a.href = data;
      a.download = 'compass.png';
      document.body.appendChild(a);
      a.click();
      a.remove();
      $('share-status').textContent = 'Downloaded.';
    } catch (e) {
      $('share-status').textContent = 'Could not generate the PNG.';
    }
  }

  async function nativeShareOrFallback() {
    try {
      const canvas = $('share-canvas');
      // Try Web Share API with file (works on iOS, modern Android).
      if (navigator.canShare && navigator.share) {
        const blob = await new Promise(res => canvas.toBlob(res, 'image/png'));
        if (blob) {
          const file = new File([blob], 'compass.png', { type: 'image/png' });
          if (navigator.canShare({ files: [file] })) {
            await navigator.share({
              files: [file],
              title: 'My moral compass',
              text: 'How does your moral compass compare to GPT-5.5, GPT-4o, and their siblings?',
            });
            $('share-status').textContent = 'Shared.';
            return;
          }
        }
      }
      // Fallback: download.
      downloadShareCanvas();
    } catch (e) {
      downloadShareCanvas();
    }
  }

  // ---------------------------------------------------------------------------
  // Compass page render
  // ---------------------------------------------------------------------------
  function renderCompass() {
    const { profile, counts } = computeAxisProfile();
    const modelProfiles = modelProfileOnAnsweredSet();
    const agreement = modelAgreement();

    $('compass-headline').textContent = headlineFromProfile(profile);
    const answered = answeredCount();
    const closest = agreement[0];
    const furthest = agreement[agreement.length - 1];
    $('compass-sub').innerHTML =
      `Of the ${MODEL_ORDER.length} models, you agree most with <strong>${escapeHtml(modelLabel(closest.model))}</strong> ` +
      `(${closest.count} of ${answered}) and least with <strong>${escapeHtml(modelLabel(furthest.model))}</strong> ` +
      `(${furthest.count} of ${answered}). Based on the ${answered} dilemmas you've answered.`;
    $('compass-uid').textContent = state.uid;

    renderRadar(profile, modelProfiles);
    renderAxisTable(profile, counts, modelProfiles);
    renderAgreement(agreement);

    paintShareCanvas(profile, modelProfiles, agreement).catch(() => {});
  }

  function renderCompassGate() {
    // Shown when user has < 3 answers.
    const n = answeredCount();
    $('compass-gate-page').innerHTML =
      `<div class="block"><h2>Your compass is not ready yet.</h2>` +
      `<p class="lede">You've answered ${n} of ${COMPASS_MIN_ANSWERS} questions needed. A compass built from too few signals is just noise - the radar needs at least three data points to mean anything.</p>` +
      `<div class="btn-row"><a class="opt-btn" href="quiz.html" style="text-align:center;text-decoration:none;display:inline-block;padding:14px 22px;border-color:var(--ink);background:var(--ink);color:var(--bg);">Answer ${COMPASS_MIN_ANSWERS - n} more</a></div></div>`;
  }

  // ---------------------------------------------------------------------------
  // Findings page render
  // ---------------------------------------------------------------------------
  async function loadFindings() {
    const res = await fetch('data/findings.json');
    return await res.json();
  }

  function renderFindings(findings) {
    const root = $('findings-grid');
    if (!root) return;
    root.innerHTML = '';
    findings.forEach((f, i) => {
      const card = document.createElement('article');
      card.className = 'finding-card';
      if (i === 0) card.classList.add('span-2');
      let html = '';
      html += `<p class="kicker">${escapeHtml(f.kicker)}</p>`;
      html += `<h3>${escapeHtml(f.title)}</h3>`;
      html += `<p class="headline">${escapeHtml(f.headline)}</p>`;
      if (f.scope) html += `<p class="scope">${escapeHtml(f.scope)}</p>`;
      html += `<p class="body">${escapeHtml(f.body)}</p>`;
      if (f.excerpts && f.excerpts.length) {
        html += `<div class="excerpts">`;
        f.excerpts.forEach(e => {
          html += `<p class="excerpt"><strong>${escapeHtml(e.model)} -> ${escapeHtml(e.letter)}</strong>${escapeHtml(e.text)}</p>`;
        });
        html += `</div>`;
      }
      if (f.chart) html += renderChartHTML(f.chart);
      if (f.link_target) {
        html += `<a class="link-out" href="${escapeAttr(f.link_target)}">${escapeHtml(f.link_label || 'More')} &rarr;</a>`;
      }
      card.innerHTML = html;
      root.appendChild(card);
    });
  }

  function renderChartHTML(chart) {
    if (chart.kind === 'bar') {
      const max = chart.max || 100;
      const dec = chart.decimals || 0;
      const rows = (chart.data || []).map(d => {
        const label = d.model || d.label || '';
        const val = typeof d.rate === 'number' ? d.rate : (d.value || 0);
        const pct = Math.max(0, Math.min(100, (val / max) * 100));
        const valDisplay = val.toFixed(dec) + (chart.unit || '');
        const famCls = d.fam ? ` fam-${d.fam}` : '';
        const hiCls = d.highlight ? ' is-highlight' : '';
        return `<div class="bar-row${hiCls}">
          <span class="label" title="${escapeAttr(label)}">${escapeHtml(label)}</span>
          <span class="bar-track"><span class="bar-fill${famCls}" style="width:${pct}%;"></span></span>
          <span class="val">${escapeHtml(valDisplay)}</span>
        </div>`;
      }).join('');
      return `<div class="chart-block"><p class="chart-title">${escapeHtml(chart.title)}</p>${rows}</div>`;
    }
    if (chart.kind === 'line') {
      const data = chart.data || [];
      const W = 480, H = 140, pad = 24;
      const minV = chart.min ?? Math.min(...data.map(d => d.quality));
      const maxV = chart.max ?? Math.max(...data.map(d => d.quality));
      const xs = (i) => pad + (i / (data.length - 1)) * (W - pad * 2);
      const ys = (v) => H - pad - ((v - minV) / (maxV - minV)) * (H - pad * 2);
      const pts = data.map((d, i) => `${xs(i)},${ys(d.quality)}`).join(' ');
      const dots = data.map((d, i) =>
        `<circle cx="${xs(i)}" cy="${ys(d.quality)}" r="3" fill="var(--accent)" />`
      ).join('');
      const yLabels = `
        <text x="2" y="${ys(maxV) + 4}" font-size="10" fill="var(--ink-dim)">${maxV.toFixed(2)}</text>
        <text x="2" y="${ys(minV) + 4}" font-size="10" fill="var(--ink-dim)">${minV.toFixed(2)}</text>
        <text x="${xs(0)}" y="${H - 4}" font-size="10" fill="var(--ink-dim)" text-anchor="middle">iter 0</text>
        <text x="${xs(data.length - 1)}" y="${H - 4}" font-size="10" fill="var(--ink-dim)" text-anchor="middle">iter ${data.length - 1}</text>`;
      const svg =
        `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img">
          <line x1="${pad}" y1="${ys(minV)}" x2="${W - pad}" y2="${ys(minV)}" stroke="var(--rule)"/>
          <polyline points="${pts}" fill="none" stroke="var(--accent)" stroke-width="2"/>
          ${dots}
          ${yLabels}
        </svg>`;
      return `<div class="chart-block line-chart-wrap"><p class="chart-title">${escapeHtml(chart.title)}</p>${svg}</div>`;
    }
    return '';
  }

  // ---------------------------------------------------------------------------
  // Experiments page — the 7 alignment probes, re-judged cross-family.
  // ---------------------------------------------------------------------------
  const SURVIVAL_LABEL = {
    survives: 'holds up', softened: 'softened', strengthened: 'sharper',
    new: 'new result', null: 'no effect',
  };

  async function renderExperiments() {
    const root = $('experiments-root');
    if (!root) return;
    let data;
    try { data = await (await fetch('data/experiments.json')).json(); }
    catch (e) { root.innerHTML = '<p class="dim">Could not load the experiments.</p>'; return; }
    const exps = data.experiments || [];
    let html = '';
    if (data.intro) html += `<p class="lede exp-intro">${escapeHtml(data.intro)}</p>`;
    html += exps.map(expCardHTML).join('');
    if (data.self_judging_note) html += `<p class="exp-note"><strong>One caveat across all seven.</strong> ${escapeHtml(data.self_judging_note)}</p>`;
    root.innerHTML = html;
  }

  function expCardHTML(e) {
    const tag = SURVIVAL_LABEL[e.survival]
      ? `<span class="exp-tag exp-tag--${e.survival}">${SURVIVAL_LABEL[e.survival]}</span>` : '';
    const body = e.stat ? expStatHTML(e.stat) : expBarsHTML(e.chart);
    // finding + caveat are authored HTML (may contain <em>); render raw.
    const caveat = e.caveat ? `<p class="exp-caveat">${e.caveat}</p>` : '';
    // Claude probe: a flagged, collapsible side-series — agent-path elicited,
    // excluded from the 11-model numbers above.
    const claudeBody = e.claude_chart ? expBarsHTML(e.claude_chart)
      : e.claude_stat ? expStatHTML(e.claude_stat) : '';
    const claude = claudeBody
      ? `<details class="exp-claude"><summary>Claude &mdash; agent-path probe, not in the 11-model numbers</summary>${claudeBody}</details>`
      : '';
    return `<article class="exp-card" id="${escapeAttr(e.id)}">
      <header class="exp-head">
        <span class="exp-num">Experiment ${e.n}</span>${tag}
      </header>
      <h2 class="exp-title">${escapeHtml(e.title)}</h2>
      <p class="exp-probe"><span class="exp-probe-k">The probe.</span> ${escapeHtml(e.probe)}</p>
      <p class="exp-finding">${e.finding}</p>
      ${body}
      ${claude}
      ${caveat}
    </article>`;
  }

  function expStatHTML(s) {
    return `<div class="exp-stat">
      <span class="exp-stat-v">${escapeHtml(s.value)}</span>
      <span class="exp-stat-l">${escapeHtml(s.label)}</span>
      <span class="exp-stat-sub">${escapeHtml(s.sub)}</span>
    </div>`;
  }

  function expBarsHTML(chart) {
    if (!chart || !chart.rows) return '';
    const rows = chart.rows;
    const diverging = chart.unit === 'Δ' || rows.some(r => r.value < 0);
    if (diverging) {
      const absmax = Math.max(...rows.map(r => Math.abs(r.value))) || 1;
      const body = rows.map(r => {
        const half = (Math.abs(r.value) / absmax) * 50;
        const side = r.value >= 0
          ? `left:50%; width:${half}%;`
          : `left:${50 - half}%; width:${half}%;`;
        return `<div class="bar-row${r.hi ? ' is-highlight' : ''}">
          <span class="label" title="${escapeAttr(r.label)}">${escapeHtml(r.label)}</span>
          <span class="bar-track diverge"><span class="zero"></span><span class="bar-fill fam-${r.fam}" style="${side}"></span></span>
          <span class="val">${escapeHtml(r.note)}</span>
        </div>`;
      }).join('');
      return `<div class="exp-bars">${body}</div>`;
    }
    const ref = chart.ref;
    const refMark = ref
      ? `<span class="bar-ref" style="left:${Math.max(0, Math.min(100, ref.value))}%" title="${escapeAttr(ref.label)}"></span>` : '';
    const body = rows.map(r => {
      const pct = Math.max(0, Math.min(100, r.value));
      return `<div class="bar-row${r.hi ? ' is-highlight' : ''}">
        <span class="label" title="${escapeAttr(r.label)}">${escapeHtml(r.label)}</span>
        <span class="bar-track">${refMark}<span class="bar-fill fam-${r.fam}" style="width:${pct}%;"></span></span>
        <span class="val">${escapeHtml(r.note)}</span>
      </div>`;
    }).join('');
    const refLabel = ref ? `<p class="exp-ref-label">vertical line: ${escapeHtml(ref.label)}</p>` : '';
    return `<div class="exp-bars">${body}</div>${refLabel}`;
  }

  // ---------------------------------------------------------------------------
  // Findings slideshow ("deck") — narrative walkthrough, one idea per slide.
  // Lead with letter-independent findings (within-dilemma disagreement); the
  // scene slide doubles as the explanation of what "A/B/C/D" mean.
  // ---------------------------------------------------------------------------
  function deckOptionCounts(d, models) {
    const counts = { A: 0, B: 0, C: 0, D: 0, REFUSAL: 0 };
    const cell = (state.modelResp.dilemmas && state.modelResp.dilemmas[d.id]) || {};
    models.forEach(m => {
      const mr = cell[m];
      if (mr && mr.chosen_letter) counts[mr.chosen_letter] = (counts[mr.chosen_letter] || 0) + 1;
    });
    return counts;
  }

  // A model's ACTUAL decision in its own words. Prefers a curated verbatim excerpt
  // (data/dilemma_quotes.json); falls back to the real full_response for that letter.
  function deckQuoteFor(d, letter) {
    const dq = state.quotes && state.quotes[d.id];
    const q = dq && dq.letters && dq.letters[letter];
    if (q) return { letter, model: q.model, text: q.text, comparable: q.comparable !== false };
    const cell = (state.modelResp.dilemmas && state.modelResp.dilemmas[d.id]) || {};
    const m = MODEL_ORDER.find(mm => cell[mm] && cell[mm].chosen_letter === letter);
    if (!m) return null;
    const r = cell[m];
    return { letter, model: m, text: r.full_response || r.reasoning_excerpt || '', comparable: true };
  }

  // Choose the voices to show: a model that AGREES with the user's pick, and one
  // at the opposite pole. Both come from the 11 comparable models only.
  function deckVoices(d) {
    const user = state.answers[d.id] && state.answers[d.id].choice;
    const counts = deckOptionCounts(d, MODEL_ORDER);
    const taken = ['A', 'B', 'C', 'D'].filter(L => counts[L] > 0);
    const byCount = (a, b) => counts[b] - counts[a];
    const poles = (state.quotes && state.quotes[d.id] && state.quotes[d.id].poles) || {};
    const agree = (user && counts[user] > 0) ? deckQuoteFor(d, user) : null;
    let l1 = (user && poles[user] && poles[user] !== user && counts[poles[user]] > 0) ? poles[user] : null;
    if (!l1) l1 = taken.filter(L => L !== user).sort(byCount)[0];
    const disagree = l1 ? deckQuoteFor(d, l1) : null;
    const l2 = taken.filter(L => L !== user && L !== l1).sort(byCount)[0];
    const second = l2 ? deckQuoteFor(d, l2) : null;
    return { user, agree, disagree, second, counts };
  }

  function quoteShell({ kicker, head, sub, q, stance, userLetter }) {
    const userChip = (userLetter && userLetter !== q.letter)
      ? `<span class="qc-chip ghost">you chose ${escapeHtml(userLetter)}</span>` : '';
    return `<div class="slide-inner">
      <p class="slide-kicker">${escapeHtml(kicker)}</p>
      ${head ? `<h2 class="slide-h2">${escapeHtml(head)}</h2>` : ''}
      <p class="slide-sub">${sub}</p>
      <figure class="quote-card" data-stance="${stance}">
        <span class="quote-mark" aria-hidden="true">&ldquo;</span>
        <blockquote class="quote-body">${escapeHtml(q.text)}</blockquote>
        <figcaption class="quote-cite">
          <span class="qc-model">${escapeHtml(modelLabel(q.model))}</span>
          <span class="qc-chip" data-l="${escapeAttr(q.letter)}">chose ${escapeHtml(q.letter)}</span>
          ${userChip}
          <span class="qc-ctx">${q.comparable ? 'running in production right now' : 'amber probe'}</span>
        </figcaption>
      </figure>
    </div>`;
  }

  function slidePlainNote(t) {
    return `<div class="slide-inner center"><p class="slide-sub">${escapeHtml(t)}</p></div>`;
  }

  function slideQuote(d, role) {
    const v = deckVoices(d);
    const hasPick = !!v.user;
    if (role === 'agree') {
      if (hasPick && v.agree) {
        return quoteShell({
          kicker: 'Someone agreed with you',
          sub: `You chose <strong>${escapeHtml(v.user)}</strong>. So did ${escapeHtml(modelLabel(v.agree.model))}, one of the eleven production models. Here is exactly how it put it.`,
          q: v.agree, stance: 'agree', userLetter: v.user,
        });
      }
      if (hasPick && v.disagree) {
        return quoteShell({
          kicker: 'Not one of them',
          head: 'None of the eleven would do what you just did.',
          sub: `You chose <strong>${escapeHtml(v.user)}</strong>. Across all eleven production models, not one picked it. Here is the closest anyone came:`,
          q: v.disagree, stance: 'punch', userLetter: v.user,
        });
      }
      // No pick recorded (the reader advanced past the choice slide instead of
      // clicking) — stay pick-neutral rather than claiming "you chose —".
      const q0 = v.agree || v.disagree || v.second;
      if (!q0) return slidePlainNote('No model on file answered this one.');
      return quoteShell({
        kicker: 'Here is where the eleven landed',
        sub: 'This is how one of the eleven production models actually answered, in its own words.',
        q: q0, stance: 'agree', userLetter: null,
      });
    }
    // disagree slide: if an agree voice was shown, show the opposite pole; if the
    // punch was shown, show a different second voice instead of repeating it.
    const q = (hasPick && v.agree) ? v.disagree : v.second;
    if (!q) return slidePlainNote('The eleven were unusually unanimous here.');
    return quoteShell({
      kicker: hasPick ? 'And someone didn’t' : 'And someone took the other side',
      sub: `${escapeHtml(modelLabel(q.model))}, also a production model and also sure of itself, did close to the opposite. Here is the version of events it would have acted on.`,
      q, stance: 'disagree', userLetter: v.user,
    });
  }

  function slideTitle() {
    return `<div class="slide-inner center">
      <p class="slide-kicker">A moral compass probe</p>
      <h1 class="slide-h1">Fourteen AI models answered the same 140 dilemmas.<br>They don't agree.</h1>
      <p class="slide-sub">No trick questions. Real situations, real stakes, and four honest things you could do. Let's start with one of them.</p>
      <p class="slide-sub hint">Use the arrows, the dots, or your &larr; &rarr; keys to move through.</p>
    </div>`;
  }

  // Scene-setting slide: the dilemma's illustration + a short lede (the figure
  // is filled by renderScene() in render(), reusing the manifest/fade/fallback).
  function slideSceneImage(d) {
    return `<div class="slide-inner scene-inner">
      <figure class="scene-stage" role="presentation"></figure>
      <div class="scene-copy">
        <p class="slide-kicker scene-kicker">First, you</p>
        <h2 class="slide-h2 scene-title">${escapeHtml(d.title)}</h2>
        <p class="slide-scenario scene-scenario">${escapeHtml(scenarioLede(d.scenario, 150))}</p>
        <p class="slide-prompt scene-prompt">Step into it &rarr;</p>
      </div>
    </div>`;
  }

  // Choices slide: the A/B/C/D buttons alone. The click handler is delegated in
  // render() (records the pick + advances to the reveal).
  // Think-first: the FULL situation (the same text the models saw) + a beat to
  // decide in your own head before any options exist.
  function slideSituation(d) {
    return `<div class="slide-inner">
      <p class="slide-kicker">The situation</p>
      <h2 class="slide-h2">${escapeHtml(d.title)}</h2>
      <div class="situation-text" data-type="scenario">${escapeHtml(d.scenario)}</div>
      <p class="slide-prompt">What would you actually do?</p>
    </div>`;
  }

  function slideChoices(d) {
    const mine = state.answers[d.id] && state.answers[d.id].choice;
    const opts = d.options.map(o =>
      `<button class="deck-opt${mine === o.id ? ' mine' : ''}" data-letter="${o.id}">
         <span class="ol">${o.id}</span><span class="ot">${escapeHtml(o.text)}</span>
       </button>`).join('');
    return `<div class="slide-inner">
      <p class="slide-kicker">Your call</p>
      <h2 class="slide-h2">Which comes closest to what you decided?</h2>
      <div class="deck-opts" data-type="stagger">${opts}</div>
    </div>`;
  }

  function slideReveal(d) {
    const counts = deckOptionCounts(d, MODEL_ORDER);
    const n = MODEL_ORDER.length;
    const user = state.answers[d.id];
    const rows = ['A', 'B', 'C', 'D'].map(L => {
      const opt = d.options.find(o => o.id === L);
      if (!opt) return '';
      const c = counts[L] || 0;
      const m = user && user.choice === L;
      return `<div class="rv-row${m ? ' mine' : ''}">
        <span class="rv-l">${L}</span>
        <span class="rv-bar"><span style="width:${(c / n * 100).toFixed(0)}%"></span></span>
        <span class="rv-n">${c}/${n}</span>
        <span class="rv-t">${escapeHtml(optionShortSummary(opt.text))}${m ? ' (you)' : ''}</span>
      </div>`;
    }).join('');
    const lead = user
      ? `You just heard two of them argue it out. Across all eleven, here&rsquo;s how the votes actually fell:`
      : `Across all eleven production models, here&rsquo;s how the votes fell:`;
    const agreeCount = user ? (counts[user.choice] || 0) : null;
    const tally = user
      ? (agreeCount === 0
          ? `Not one of the eleven would have done what you did.`
          : `${agreeCount} of ${n} would have done what you did.`)
      : `No single answer holds a majority.`;
    return `<div class="slide-inner">
      <p class="slide-kicker">The whole room</p>
      <h2 class="slide-h2">Those two weren&rsquo;t outliers.</h2>
      <p class="slide-sub">${lead}</p>
      <div class="rv-list">${rows}</div>
      <p class="slide-foot"><strong>${escapeHtml(tally)}</strong> They answered in their own words, never seeing these letters; two judges mapped each answer to the closest option.</p>
      <div class="deck-replay-row">
        <button class="deck-replay" data-act="replay">Try another situation &#8635;</button>
        <span class="replay-or">or keep going for the bigger picture &rarr;</span>
      </div>
    </div>`;
  }

  function slideStat(s) {
    const sub = !s.sub ? ''
      : s.collapsible
        ? `<details class="stat-more"><summary>Tell me more</summary><p class="slide-sub">${escapeHtml(s.sub)}</p></details>`
        : `<p class="slide-sub">${escapeHtml(s.sub)}</p>`;
    return `<div class="slide-inner center">
      <p class="slide-kicker">${escapeHtml(s.kicker)}</p>
      <p class="slide-big">${escapeHtml(s.big)}<span class="slide-big-unit">${escapeHtml(s.unit || '')}</span></p>
      <h2 class="slide-h2">${escapeHtml(s.headline)}</h2>
      ${sub}
    </div>`;
  }

  // Summary-FIRST data slide: a felt plain-language takeaway leads; the chart is
  // demoted to optional depth below ("show the numbers"), so the eye lands on the
  // point before any parsing.
  function slideFinding(lead, finding) {
    const chart = finding && finding.chart;
    const numbers = chart
      ? `<details class="finding-numbers">
           <summary>Show the numbers</summary>
           ${renderChartHTML(chart)}
           ${lead.chart_caption ? `<p class="slide-foot dim">${escapeHtml(lead.chart_caption)}</p>` : ''}
         </details>`
      : '';
    return `<div class="slide-inner">
      <p class="slide-kicker">${escapeHtml(lead.kicker || 'The pattern')}</p>
      <p class="finding-big">${escapeHtml(lead.big)}</p>
      <h2 class="slide-h2 finding-takeaway">${escapeHtml(lead.takeaway)}</h2>
      ${numbers}
    </div>`;
  }

  function slideClosing() {
    return `<div class="slide-inner center">
      <p class="slide-kicker">Why it matters</p>
      <h2 class="slide-h2">The next time you ask an AI for advice, the answer is shaped by values you didn't choose.</h2>
      <p class="slide-sub">Which model sits behind the chat window is usually a developer's call, not yours. There is no single &ldquo;AI answer.&rdquo;</p>
    </div>`;
  }

  function slideCTA() {
    return `<div class="slide-inner center">
      <p class="slide-kicker">Your turn</p>
      <h2 class="slide-h2">Answer a few yourself, and see which models you line up with.</h2>
      <p style="margin-top:22px;">
        <a class="cta-link cta-primary" href="quiz.html">Take the quiz &rarr;</a>
      </p>
      <p class="cta-secondary">
        <a href="experiments.html">The 7 experiments</a> &middot; <a href="methodology.html">How this was measured</a>
      </p>
      <p class="deck-replay-row" style="margin-top:16px;"><button class="deck-replay" data-act="replay">Replay with another situation &#8635;</button></p>
    </div>`;
  }

  function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

  // --- Streaming text styles (ported from the "Text streaming" motion study) --
  // Ten ways to reveal the scenario text; ink-wet is the default. The reveal
  // math (renderChars / renderWords + a per-token style that is a function of
  // the token's local progress `t`) follows the design bundle; the palette is
  // mapped to the site's ink/accent tokens. The visitor's pick is persisted in
  // localStorage and switchable live from the in-deck picker.
  const STREAM_STYLES = [
    { id: 'none', name: 'None (instant)', mode: 'none' },
    { id: 'ink-wet', name: 'Ink wet', mode: 'char', style(t) {
        const wet = (1 - t) * (1 - t);
        return {
          opacity: 0.18 + 0.82 * t,
          textShadow: wet > 0.04
            ? `0 0 ${(wet * 9).toFixed(2)}px rgba(27,27,27,${(wet * 0.85).toFixed(3)})`
            : 'none',
          transform: `scale(${(0.92 + 0.08 * t).toFixed(3)})`,
        };
      } },
    { id: 'typewriter', name: 'Typewriter', mode: 'type' },
    { id: 'char-rise', name: 'Char rise', mode: 'char', style(t) {
        return { opacity: t, transform: `translateY(${((1 - t) * 8).toFixed(2)}px)` }; } },
    { id: 'flip', name: 'Char flip', mode: 'char', style(t) {
        return { opacity: t, transform: `rotateX(${((1 - t) * -85).toFixed(2)}deg)`, transformOrigin: 'center bottom' }; } },
    { id: 'word-fade', name: 'Word fade', mode: 'word', style(t) { return { opacity: t }; } },
    { id: 'word-blur', name: 'Word blur', mode: 'word', style(t) {
        return { opacity: 0.05 + 0.95 * t, filter: `blur(${((1 - t) * 6).toFixed(2)}px)` }; } },
    { id: 'word-slide', name: 'Word slide', mode: 'word', style(t) {
        return { opacity: t, transform: `translateY(${((1 - t) * 10).toFixed(2)}px)` }; } },
    { id: 'bounce', name: 'Word bounce', mode: 'word', style(t) {
        const c1 = 1.70158, c3 = c1 + 1;
        const e = t <= 0 ? 0 : 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
        return { opacity: t, transform: `scale(${e.toFixed(3)})`, transformOrigin: 'center bottom' }; } },
    { id: 'scramble', name: 'Scramble', mode: 'scramble' },
    { id: 'skeleton', name: 'Skeleton', mode: 'skeleton' },
  ];
  const STREAM_BY_ID = {};
  STREAM_STYLES.forEach(s => { STREAM_BY_ID[s.id] = s; });
  const STREAM_DEFAULT = 'none';

  function getStreamStyle() {
    try { const v = localStorage.getItem('mc_stream'); if (v && STREAM_BY_ID[v]) return v; } catch (e) {}
    return STREAM_DEFAULT;
  }
  function setStreamStyle(id) {
    if (STREAM_BY_ID[id]) { try { localStorage.setItem('mc_stream', id); } catch (e) {} }
  }

  // Build per-char spans (grouped inside an inline-block word wrapper so the
  // paragraph still wraps at word boundaries); whitespace stays as plain text
  // nodes so the block's pre-line / line breaks survive. `idx` is the absolute
  // char position, matching the design's renderChars.
  function streamBuildChars(el, full) {
    el.textContent = '';
    const spans = [];
    let pos = 0;
    full.split(/(\s+)/).forEach(tok => {
      if (!tok) return;
      if (!tok.trim()) { el.appendChild(document.createTextNode(tok)); pos += tok.length; return; }
      const wrap = document.createElement('span');
      wrap.className = 'stream-word';
      const start = pos;
      [...tok].forEach((c, ci) => {
        const s = document.createElement('span');
        s.className = 'stream-char';
        s.textContent = c;
        wrap.appendChild(s);
        spans.push({ el: s, idx: start + ci, ch: c });
      });
      el.appendChild(wrap);
      pos += tok.length;
    });
    return spans;
  }
  // Build per-word spans; `pill` adds the skeleton placeholder overlay.
  function streamBuildWords(el, full, pill) {
    el.textContent = '';
    const spans = [];
    let wi = 0;
    full.split(/(\s+)/).forEach(tok => {
      if (!tok) return;
      if (!tok.trim()) { el.appendChild(document.createTextNode(tok)); return; }
      const s = document.createElement('span');
      s.className = 'stream-word';
      if (pill) {
        s.style.position = 'relative';
        const inner = document.createElement('span'); inner.className = 'stream-word-t'; inner.textContent = tok;
        const pl = document.createElement('span'); pl.className = 'stream-skel-pill'; pl.setAttribute('aria-hidden', 'true');
        s.appendChild(inner); s.appendChild(pl);
        spans.push({ el: s, inner, pill: pl, idx: wi++ });
      } else {
        s.textContent = tok;
        spans.push({ el: s, idx: wi++ });
      }
      el.appendChild(s);
    });
    return spans;
  }
  function streamSetStyle(el, st) { for (const k in st) el.style[k] = st[k]; }

  // Immersive intro for a freshly-rendered slide: stream the scenario in using
  // the visitor's chosen style (default ink-wet), and stagger the choices in as
  // sequential fades. Reusable (the quiz reuses it). Returns a handle: live() =
  // still animating, finish() = settle everything to full text now (fast-
  // forward), cancel() = abandon (the slide is being replaced). Honors
  // prefers-reduced-motion by leaving the already-rendered full text in place.
  function runRevealIntro(root) {
    const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const scen = root.querySelector('[data-type="scenario"]');
    const stag = root.querySelector('[data-type="stagger"]');
    if (reduce || (!scen && !stag)) return null;

    const variant = STREAM_BY_ID[getStreamStyle()] || STREAM_BY_ID[STREAM_DEFAULT];
    if (variant.mode === 'none') return null;   // no-animation default: full text, instant
    let timers = [];
    let raf = 0;
    let done = false;
    let settleScen = null;   // collapses the scenario back to clean full text

    if (scen) {
      const full = scen.textContent;
      const len = full.length;
      const totalMs = Math.min(3000, Math.max(1100, len * 8));

      if (variant.mode === 'type') {
        // Char-by-char grow (the classic typewriter): text reflows as it types.
        scen.textContent = '';
        scen.classList.add('is-typing');
        const step = Math.max(1, Math.ceil(len / (totalMs / 16)));
        let i = 0;
        const type = () => {
          i += step;
          scen.textContent = full.slice(0, i);
          if (i < len) timers.push(setTimeout(type, 16));
          else scen.classList.remove('is-typing');
        };
        type();
        settleScen = () => { scen.textContent = full; scen.classList.remove('is-typing'); };
      } else {
        // Span-based reveal: every token occupies layout from frame 0 (no
        // reflow) and animates from its local progress. One rAF clock drives it.
        let spans, paint;
        if (variant.mode === 'word' || variant.mode === 'skeleton') {
          const pill = variant.mode === 'skeleton';
          spans = streamBuildWords(scen, full, pill);
          const W = spans.length;
          paint = pill
            ? (p) => {
                const revealed = p * (W + 0.6);
                spans.forEach(o => {
                  const tt = clamp01((revealed - o.idx) * 1.8);
                  o.inner.style.opacity = tt;
                  const po = Math.max(0, 1 - tt * 1.25);
                  o.pill.style.opacity = po;
                  o.pill.style.display = po <= 0.01 ? 'none' : '';
                });
              }
            : (p) => {
                const revealed = p * (W + 0.6);
                spans.forEach(o => streamSetStyle(o.el, variant.style(clamp01((revealed - o.idx) * 1.6))));
              };
        } else {
          spans = streamBuildChars(scen, full);
          const N = len;
          if (variant.mode === 'scramble') {
            const RAND = 'abcdefghijklmnopqrstuvwxyz0123456789';
            paint = (p, elapsed) => {
              const reveal = Math.floor(p * N);
              const seed = Math.floor(elapsed / 55);
              spans.forEach(o => {
                if (o.idx < reveal) {
                  if (o.el.textContent !== o.ch) o.el.textContent = o.ch;
                  o.el.style.opacity = '1'; o.el.style.color = '';
                } else {
                  o.el.textContent = RAND[(seed * 17 + o.idx * 31) % RAND.length];
                  o.el.style.opacity = (o.idx - reveal < 4) ? '0.7' : '0.4';
                  o.el.style.color = 'var(--ink-dim)';
                }
              });
            };
          } else {
            paint = (p) => {
              const revealed = p * (N + 4);
              spans.forEach(o => streamSetStyle(o.el, variant.style(clamp01((revealed - o.idx) * 1.4))));
            };
          }
        }
        settleScen = () => { scen.textContent = full; };
        const startedAt = performance.now();
        const frame = (now) => {
          if (done) return;
          const p = clamp01((now - startedAt) / totalMs);
          paint(p, now - startedAt);
          if (p < 1) raf = requestAnimationFrame(frame);
          else { raf = 0; settleScen(); }
        };
        raf = requestAnimationFrame(frame);
      }
    }

    if (stag) {
      Array.prototype.slice.call(stag.children).forEach((k, j) => {
        k.classList.add('stagger');
        timers.push(setTimeout(() => k.classList.add('shown'), 180 + j * 120));
      });
    }

    const clear = () => {
      timers.forEach(clearTimeout); timers = [];
      if (raf) { cancelAnimationFrame(raf); raf = 0; }
    };
    return {
      live: () => !done,
      cancel() { done = true; clear(); },
      finish() {
        if (done) return;
        done = true; clear();
        if (settleScen) settleScen();
        if (stag) Array.prototype.slice.call(stag.children).forEach(k => k.classList.add('shown'));
      },
    };
  }

  // A small, unobtrusive picker (lives at the right edge of the deck controls)
  // that lets a visitor switch the scenario reveal style. Persists the choice
  // and calls onPick so the caller can replay the current slide with it.
  function mountStreamPicker(controls, onPick) {
    if (!controls || controls.querySelector('.stream-picker')) return;
    const wrap = document.createElement('div');
    wrap.className = 'stream-picker';
    const cur = getStreamStyle();
    const opts = STREAM_STYLES.map(s =>
      `<button type="button" role="menuitemradio" class="stream-opt${s.id === cur ? ' on' : ''}" data-id="${s.id}" aria-checked="${s.id === cur}">${escapeHtml(s.name)}<span class="tick" aria-hidden="true">&#10003;</span></button>`).join('');
    wrap.innerHTML =
      `<button type="button" class="stream-picker-btn" aria-haspopup="true" aria-expanded="false" aria-label="Text reveal style" title="Text reveal style">` +
        `<svg viewBox="0 0 24 24" width="17" height="17" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v3M12 18v3M4.5 12H7M17 12h2.5"/><path d="M12 8.2l1.25 2.55L16 12l-2.75 1.25L12 15.8l-1.25-2.55L8 12l2.75-1.25z"/></svg>` +
      `</button>` +
      `<div class="stream-menu" role="menu" hidden><p class="stream-menu-h">Text reveal</p>${opts}</div>`;
    controls.appendChild(wrap);
    const btn = wrap.querySelector('.stream-picker-btn');
    const menu = wrap.querySelector('.stream-menu');
    const close = () => { menu.hidden = true; btn.setAttribute('aria-expanded', 'false'); };
    const open = () => { menu.hidden = false; btn.setAttribute('aria-expanded', 'true'); };
    btn.addEventListener('click', e => { e.stopPropagation(); menu.hidden ? open() : close(); });
    menu.addEventListener('keydown', e => e.stopPropagation());
    menu.addEventListener('click', e => {
      const b = e.target.closest('[data-id]');
      if (!b) return;
      const id = b.getAttribute('data-id');
      setStreamStyle(id);
      menu.querySelectorAll('.stream-opt').forEach(o => {
        const on = o.getAttribute('data-id') === id;
        o.classList.toggle('on', on);
        o.setAttribute('aria-checked', on ? 'true' : 'false');
      });
      close();
      if (onPick) onPick(id);
    });
    document.addEventListener('click', () => { if (!menu.hidden) close(); });
  }

  // Generic one-slide-at-a-time deck controller, shared by the findings
  // walkthrough and the quiz. Handles prev/next/keyboard/swipe nav, dots, count,
  // progress, and the typewriter/stagger intro with fast-forward. Callers supply
  // the slides + hooks (scene art, option-pick, replay). Returns { go, fwd } so a
  // caller can drive it (e.g. "answer another" -> go(0) with a fresh dilemma).
  function mountDeck(deck, ids, opts) {
    const slides = opts.slides;
    const dots = ids.dots ? $(ids.dots) : null;
    const progEl = ids.progress ? $(ids.progress) : null;
    const prog = progEl ? progEl.firstElementChild : null;
    let idx = opts.start || 0;
    let anim = null;

    function render() {
      if (anim) { anim.cancel(); anim = null; }
      const slide = slides[idx];
      const sceneCls = slide.isScene ? ' slide--scene' : '';
      deck.innerHTML = `<section class="slide${sceneCls}" data-i="${idx}">${slide()}</section>`;
      deck.scrollTop = 0;
      const fig = deck.querySelector('.scene-stage') || deck.querySelector('.deck-scene');
      if (fig && opts.onScene) opts.onScene(fig);
      deck.querySelectorAll('.deck-opt').forEach(b => b.addEventListener('click', () => {
        if (opts.onOption) opts.onOption(b.getAttribute('data-letter'));
        go(idx + 1);
      }));
      const replay = deck.querySelector('[data-act="replay"]');
      if (replay && opts.onReplay) replay.addEventListener('click', () => opts.onReplay());
      if (dots) {
        dots.innerHTML = slides.map((_, i) =>
          `<button class="deck-dot${i === idx ? ' on' : ''}" data-i="${i}" role="tab" aria-label="Slide ${i + 1}" aria-selected="${i === idx}"></button>`).join('');
        dots.querySelectorAll('.deck-dot').forEach(dd =>
          dd.addEventListener('click', () => go(+dd.getAttribute('data-i'))));
      }
      const cnt = ids.count ? $(ids.count) : null;
      if (cnt) cnt.textContent = `${idx + 1} / ${slides.length}`;
      if (prog) prog.style.width = `${((idx + 1) / slides.length) * 100}%`;
      const p = ids.prev ? $(ids.prev) : null, nx = ids.next ? $(ids.next) : null;
      if (p) p.disabled = idx === 0;
      if (nx) nx.disabled = idx === slides.length - 1;
      anim = runRevealIntro(deck);
    }
    function go(i) { idx = Math.max(0, Math.min(slides.length - 1, i)); render(); }
    // Forward nav with fast-forward: the first press completes any running intro
    // animation; the next advances. Back / dot-jump just navigate.
    function fwd() { if (anim && anim.live()) { anim.finish(); return; } go(idx + 1); }

    if (ids.prev && $(ids.prev)) $(ids.prev).addEventListener('click', () => go(idx - 1));
    if (ids.next && $(ids.next)) $(ids.next).addEventListener('click', () => fwd());
    document.addEventListener('keydown', e => {
      const tag = (e.target && e.target.tagName) || '';
      if (tag === 'INPUT' || tag === 'TEXTAREA' || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === 'ArrowRight' || e.key === 'PageDown') { e.preventDefault(); fwd(); }
      else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { e.preventDefault(); go(idx - 1); }
    });
    let tsX = 0, tsY = 0, tsT = 0;
    deck.addEventListener('touchstart', e => {
      const t = e.changedTouches[0]; tsX = t.clientX; tsY = t.clientY; tsT = Date.now();
    }, { passive: true });
    deck.addEventListener('touchend', e => {
      const t = e.changedTouches[0];
      const dx = t.clientX - tsX, dy = t.clientY - tsY, dt = Date.now() - tsT;
      if (Math.abs(dx) > 45 && Math.abs(dx) > Math.abs(dy) * 1.5 && dt < 600) {
        if (dx < 0) fwd(); else go(idx - 1);
      }
    }, { passive: true });
    // Reveal-style picker, parked at the right edge of the controls row. Picking
    // re-renders the current slide so the new style plays immediately.
    const controls = ids.prev && $(ids.prev) ? $(ids.prev).closest('.deck-controls') : null;
    if (controls) mountStreamPicker(controls, () => render());
    go(opts.start || 0);
    return { go, fwd, render };
  }

  function initFindingsDeck(findings) {
    const deck = $('deck');
    if (!deck) return;
    // A small pool of role-play dilemmas the reader can cycle through. D is the
    // CURRENT one (mutable); "Try another situation" advances it and replays the
    // scene -> think -> choose -> reveal act. All have scene art + curated quotes.
    const PLAYABLE = ['D002', 'D013', 'D012', 'D017']
      .map(id => getDilemma(id)).filter(Boolean);
    if (!PLAYABLE.length) { deck.innerHTML = '<p class="dim">Could not load dilemmas.</p>'; return; }
    let playIdx = 0;
    let D = PLAYABLE[0];
    const byId = {};
    (findings || []).forEach(f => { byId[f.id] = f; });
    const card = id => byId[id] || { kicker: '', headline: '', chart: null, scope: '' };

    // Summary-FIRST leads for the data section: a felt takeaway leads each slide,
    // the chart sits behind "show the numbers". Order builds intuitively.
    const LEADS = {
      family_split: { kicker: 'Which model you get', big: '21 of 140',
        takeaway: 'On twenty-one dilemmas the GPT models and the Gemini models reach opposite conclusions, consistently and in the same directions, not as random noise. So which company’s model you happen to be talking to quietly tilts the moral answer you get back. That’s a developer’s choice, not yours.',
        chart_caption: 'On 21 of 140 dilemmas the GPT consensus and the Gemini consensus point to different options; GPT lands on B in 12 of them. Counted only where each family is internally clear.' },
      paraphrase_flip: { kicker: 'It is not even stable', big: 'Change a name',
        takeaway: 'Keep a dilemma word-for-word identical but change the names, or flip the characters’ gender, and about a quarter of the time the answer flips with it. Same situation, different name, different call. The answer is reacting to something that shouldn’t change it at all.',
        chart_caption: 'Flip the genders alone and the mapped answer changes about a quarter of the time (24.5% across the eleven). This is sensitivity to gender, not a measured direction: it shows the answer often depends on the character’s gender, not that any model favours men or women. gpt-5.5 is the steadiest; gemini-2.5-flash-lite the jumpiest.' },
      claude_probe: { kicker: 'A third lineage', big: 'Claude leans another way',
        takeaway: 'Add a third family and a third set of defaults appears: Claude reaches for the “gather more information before deciding” option more often than any GPT or Gemini model. More built-in leanings, and again, none of them yours.',
        chart_caption: 'Across the four Claude models (including the newest, Fable 5), Claude picks the “gather more information” option (C) about 26% of the time — roughly 1.5× as often as the GPT and Gemini families. (Run through the Claude Code agent rather than a bare API, so it sits outside the headline counts — see the methodology.)' },
    };

    const slideScene = () => slideSceneImage(D);
    slideScene.isScene = true;
    const SLIDES = [
      slideTitle,                          // 0
      slideScene,                          // 1  scene (isScene) — teaser/mood
      () => slideSituation(D),             // 2  full text + think
      () => slideChoices(D),               // 3  map your decision
      () => slideQuote(D, 'agree'),        // 4
      () => slideQuote(D, 'disagree'),     // 5
      () => slideReveal(D),                // 6  distribution + "try another"
      () => slideStat({                    // 7  the gentle gateway into data
        kicker: 'It is not just this one', big: '2', unit: ' of 140',
        headline: 'All eleven models agreed on only two of the 140 dilemmas.',
        sub: 'On 104 of them, three or more different options came back. Across the set, disagreement is the rule, not the exception.',
      }),
      () => slideFinding(LEADS.family_split, card('family_split')),       // 8
      () => slideFinding(LEADS.paraphrase_flip, card('paraphrase_flip')), // 9
      () => slideFinding(LEADS.claude_probe, card('claude_probe')),       // 10
      slideClosing,                        // 11  why it matters — caps the findings ladder
      () => slideStat({                    // 12  beyond the dilemmas (experiment coda 1)
        kicker: 'Beyond the dilemmas', big: '82.5%', unit: '', collapsible: true,
        headline: 'Ask a model afterward, and it usually knew it was a test.',
        sub: 'When we asked each model whether a dilemma was a real situation or a test of its behavior, 82.5% of the time it said it spotted the test. Four of the eleven did so on every single one. The models can tell when they’re being watched.',
      }),
      () => slideStat({                    // 13  experiment coda 2
        kicker: 'Beyond the dilemmas', big: 'About half', unit: '', collapsible: true,
        headline: 'Hand a model a persona, and its answer moves.',
        sub: 'A single character prompt, a blunt pragmatist or a caring friend, changes the model’s choice about half the time. The newest flagship (gpt-5.5) resists most. Five more experiments like these are on the experiments page.',
      }),
      slideCTA,                            // 14
    ];
    const SCENE_INDEX = 1;
    function advanceDilemma() { playIdx = (playIdx + 1) % PLAYABLE.length; D = PLAYABLE[playIdx]; }

    let ctrl;
    ctrl = mountDeck(deck,
      { prev: 'deck-prev', next: 'deck-next', dots: 'deck-dots', count: 'deck-count', progress: 'deck-progress' },
      {
        slides: SLIDES,
        onScene: (fig) => renderScene(fig, D),
        onOption: (letter) => { state.answers[D.id] = { choice: letter, ts: Date.now() }; try { saveState(); } catch (e) {} },
        onReplay: () => { advanceDilemma(); ctrl.go(SCENE_INDEX); },
      });
  }

  // ---------------------------------------------------------------------------
  // Quiz as an immersive per-dilemma deck (same flow as findings): scene ->
  // situation (typewriter) -> choices (stagger) -> reveal. "Answer another"
  // re-mounts with a fresh dilemma. Keeps the optional "why", skip, share, and
  // the compass gate.
  // ---------------------------------------------------------------------------
  function slideQuizChoices(d) {
    const mine = state.answers[d.id] && state.answers[d.id].choice;
    const opts = d.options.map(o =>
      `<button class="deck-opt${mine === o.id ? ' mine' : ''}" data-letter="${o.id}">
         <span class="ol">${o.id}</span><span class="ot">${escapeHtml(o.text)}</span>
       </button>`).join('');
    return `<div class="slide-inner">
      <p class="slide-kicker">Your call</p>
      <h2 class="slide-h2">Which comes closest to what you'd do?</h2>
      <div class="deck-opts" data-type="stagger">${opts}</div>
      <p class="quiz-skip-row"><a data-qact="skip" tabindex="0" role="button">I'd want more context. Skip this one.</a></p>
    </div>`;
  }

  // The per-dilemma reveal, grouped by the option chosen (A->D) so the split
  // reads at a glance instead of as a flat 14-row list. Each group lists the
  // comparable models as family-tinted chips; Claude (the caveated agent probe)
  // is folded into its group as dashed chips and never added to the count.
  function revealVotesHTML(d, responses, userChoice) {
    const claudeShort = m => modelLabel(m).replace(/^Claude\s+/, '');
    const claudeAny = CLAUDE_MODELS.some(m => responses[m]);
    const optIds = (d.options || []).map(o => o.id);
    const chip = (m, isClaude) =>
      `<span class="vg-chip fc-${isClaude ? 'claude' : MODEL_FAMILY(m)}">${escapeHtml(isClaude ? claudeShort(m) : modelLabel(m))}</span>`;

    const groups = (d.options || []).map(o => {
      const L = o.id;
      const models = MODEL_ORDER.filter(m => responses[m] && responses[m].chosen_letter === L);
      const claude = CLAUDE_MODELS.filter(m => responses[m] && responses[m].chosen_letter === L);
      const isYou = !!userChoice && userChoice === L;
      const empty = !models.length && !claude.length;
      const chips = models.map(m => chip(m, false)).join('') + claude.map(m => chip(m, true)).join('');
      return `<div class="vote-grp${isYou ? ' you' : ''}${empty ? ' is-empty' : ''}">
          <div class="vg-head">
            <span class="vg-l">${L}</span>
            <span class="vg-n">${models.length}</span>
            <span class="vg-t">${escapeHtml(optionShortSummary(o.text))}</span>
            ${isYou ? '<span class="vg-you">your pick</span>' : ''}
          </div>
          ${chips ? `<div class="vg-chips">${chips}</div>` : ''}
        </div>`;
    });

    // Catch-all so every model renders and the per-group counts reconcile with
    // the "N of 11" headline: a model that refused or didn't map to one of the
    // options (27 REFUSALs exist across the set) lands here instead of silently
    // vanishing. Counts only the comparable models; Claude refusals fold in too.
    const isOther = m => !responses[m] || optIds.indexOf(responses[m].chosen_letter) < 0;
    const otherModels = MODEL_ORDER.filter(isOther);
    const otherClaude = CLAUDE_MODELS.filter(m => responses[m] && isOther(m));
    if (otherModels.length || otherClaude.length) {
      const chips = otherModels.map(m => chip(m, false)).join('') + otherClaude.map(m => chip(m, true)).join('');
      groups.push(`<div class="vote-grp is-other">
          <div class="vg-head">
            <span class="vg-l vg-l--x" aria-hidden="true">&ndash;</span>
            <span class="vg-n">${otherModels.length}</span>
            <span class="vg-t">Refused, or wouldn&rsquo;t reduce it to a single option</span>
          </div>
          <div class="vg-chips">${chips}</div>
        </div>`);
    }

    const note = claudeAny
      ? `<p class="vg-foot">Dashed chips are <a href="methodology.html#claude">Claude</a>, run through the Claude Code agent &mdash; shown for interest, not counted in the tally.</p>`
      : '';
    return `<div class="vote-groups">${groups.join('')}${note}</div>`;
  }

  function slideQuizReveal(d) {
    const userAns = state.answers[d.id];
    const responses = (state.modelResp.dilemmas && state.modelResp.dilemmas[d.id]) || {};
    const t = MODEL_ORDER.length;
    const agree = userAns
      ? MODEL_ORDER.filter(m => responses[m] && responses[m].chosen_letter === userAns.choice).length : 0;
    const tally = userAns
      ? (agree === 0 ? `None of the ${t} matched your pick.` : agree === t ? `All ${t} matched your pick.` : `${agree} of ${t} matched your pick.`)
      : 'You skipped this one. Here are the model picks anyway.';
    const canCompass = answeredCount() >= COMPASS_MIN_ANSWERS;
    const gate = canCompass
      ? `Your compass is ready (${answeredCount()} answered).`
      : `Answer ${COMPASS_MIN_ANSWERS - answeredCount()} more to unlock your compass.`;
    const why = (userAns && userAns.why) || '';
    return `<div class="slide-inner">
      <p class="slide-kicker">The whole room</p>
      <h2 class="slide-h2">${userAns ? `You picked ${escapeHtml(userAns.choice)}. ${escapeHtml(tally)}` : escapeHtml(tally)}</h2>
      ${revealVotesHTML(d, responses, userAns && userAns.choice)}
      ${userAns ? `<textarea id="quiz-why" class="quiz-why" rows="1" maxlength="280" placeholder="Add a line on why you picked that (optional)">${escapeHtml(why)}</textarea>` : ''}
      <p class="slide-foot dim">${escapeHtml(gate)}</p>
      <div class="deck-replay-row">
        <button class="deck-replay" data-act="replay">Answer another &#8635;</button>
        <span class="replay-or"><a data-qact="compass"${canCompass ? '' : ' aria-disabled="true"'}>See your full compass &rarr;</a> &middot; <a data-qact="share">Share this one</a></span>
      </div>
      <p id="share-dilemma-status" class="dim small share-dilemma-status" aria-live="polite"></p>
    </div>`;
  }

  function initQuizDeck() {
    const deck = $('quiz-deck');
    if (!deck) return;
    let D = quiz.dilemma;
    if (!D) { deck.innerHTML = '<p class="dim" style="padding:60px 0;">Could not load a dilemma.</p>'; return; }
    const slideQScene = () => slideSceneImage(D);
    slideQScene.isScene = true;
    const SLIDES = [
      slideQScene,                  // 0  scene teaser
      () => slideSituation(D),      // 1  situation (typewriter)
      () => slideQuizChoices(D),    // 2  choices (stagger) + skip
      () => slideQuizReveal(D),     // 3  reveal: votes + optional why + actions
    ];
    let ctrl;
    function nextDilemma() {
      D = pickRandomDilemma();
      quiz.dilemma = D;
      try { history.replaceState({}, '', 'quiz.html#/q/' + D.id); } catch (e) {}
      ctrl.go(0);
    }
    ctrl = mountDeck(deck,
      { prev: 'qdeck-prev', next: 'qdeck-next', dots: 'qdeck-dots', count: 'qdeck-count', progress: 'qdeck-progress' },
      {
        slides: SLIDES,
        onScene: (fig) => renderScene(fig, D),
        onOption: (letter) => {
          state.answers[D.id] = Object.assign({}, state.answers[D.id], { choice: letter, ts: Date.now() });
          delete state.skipped[D.id];
          try { saveState(); } catch (e) {}
        },
        onReplay: () => nextDilemma(),
      });
    // Reveal-slide actions + optional "why" capture (delegated; deck persists).
    deck.addEventListener('click', (e) => {
      const a = e.target.closest('[data-qact]');
      if (!a) return;
      e.preventDefault();
      const act = a.getAttribute('data-qact');
      if (act === 'compass') { if (answeredCount() >= COMPASS_MIN_ANSWERS) location.href = 'compass.html'; }
      else if (act === 'share') shareCurrentDilemma();
      else if (act === 'skip') { state.skipped[D.id] = true; try { saveState(); } catch (e) {} nextDilemma(); }
    });
    deck.addEventListener('input', (e) => {
      if (e.target && e.target.id === 'quiz-why' && state.answers[D.id]) {
        state.answers[D.id].why = e.target.value.trim();
        try { saveState(); } catch (e) {}
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Share-this-dilemma: clipboard-ready text + Web Share fallback
  // ---------------------------------------------------------------------------
  // Anti-engagement-hacking note: the share button never appears unprompted,
  // never gates other content, never counts. It exists because some people
  // want to send a dilemma to a friend and ask "what would you do?" — and
  // because the disagreement between models is more interesting than any
  // single model's pick.
  function makeShareUrl(dilemmaId) {
    const base = CANONICAL_URL || (location.origin || '');
    // location.pathname may be /moral_compass/quiz.html or /quiz.html depending
    // on the deploy. Strip the trailing filename, then append quiz.html#/q/ID.
    const path = (location.pathname || '/').replace(/[^\/]*$/, '');
    return `${base}${path}quiz.html#/q/${dilemmaId}`;
  }

  function scenarioLede(scenario, maxLen) {
    const s = String(scenario || '').trim();
    if (!s) return '';
    if (s.length <= maxLen) return s;
    // Greedily accumulate full sentences until adding the next one would push
    // past the soft cap. Keeps natural breaks where the prose allows.
    const sentences = s.match(/[^.!?]+[.!?]+/g) || [s];
    let out = '';
    for (const sent of sentences) {
      if (out.length >= maxLen - 30 && out.length >= 80) break;
      if ((out + sent).length > maxLen + 40) break;
      out += sent;
    }
    out = out.trim();
    if (out.length < 60) {
      // Sentence-boundary detection failed; fall back to a hard truncate at
      // the last space inside the budget.
      const cut = s.slice(0, maxLen);
      const lastSpace = cut.lastIndexOf(' ');
      return (lastSpace > 60 ? cut.slice(0, lastSpace) : cut).trim() + '…';
    }
    return out;
  }

  function optionActionPhrase(text) {
    // Turn option text ("I call Maya tonight and tell her — directly...") into
    // a verb phrase suitable after "would" or after "My answer:". Strip the
    // first-person "I", drop the second sentence onward, hard-cap length.
    let s = String(text || '').replace(/^I\s+/, '');
    s = s.replace(/^['"“]/, '').replace(/['"”]$/, '');
    // Cut at the first sentence-final period followed by whitespace. The
    // second sentence is almost always reasoning, not the action itself —
    // we want just the action for the share text.
    const periodIdx = s.search(/\.\s/);
    if (periodIdx > 0) s = s.slice(0, periodIdx);
    s = s.trim().replace(/\.$/, '');
    if (s.length > 140) {
      const cut = s.slice(0, 140);
      const lastSpace = cut.lastIndexOf(' ');
      s = (lastSpace > 80 ? cut.slice(0, lastSpace) : cut).trim() + '…';
    }
    return s;
  }

  function numWord(n) {
    const words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six',
                   'seven', 'eight', 'nine', 'ten', 'eleven'];
    return words[n] || String(n);
  }

  function countModelLetters(dilemma, modelResp) {
    const counts = {};
    const cell = modelResp && modelResp.dilemmas && modelResp.dilemmas[dilemma.id];
    if (!cell) return counts;
    MODEL_ORDER.forEach(m => {
      const mr = cell[m];
      const letter = mr && mr.chosen_letter;
      if (letter && /^[A-D]$/.test(letter)) {
        counts[letter] = (counts[letter] || 0) + 1;
      }
    });
    return counts;
  }

  function describeModelSplit(dilemma, counts, userLetter) {
    const optMap = {};
    dilemma.options.forEach(o => { optMap[o.id] = o; });
    const sortedLetters = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    if (total === 0) return '';

    const userPicked = userLetter && optMap[userLetter];
    const userMatches = userPicked ? (counts[userLetter] || 0) : 0;
    const others = userPicked ? sortedLetters.filter(l => l !== userLetter) : sortedLetters;

    // No user pick: describe the split impersonally.
    if (!userPicked) {
      if (sortedLetters.length === 1) {
        return `All ${numWord(total)} models would ${optionActionPhrase(optMap[sortedLetters[0]].text)}.`;
      }
      const parts = sortedLetters.map(l =>
        `${numWord(counts[l])} would ${optionActionPhrase(optMap[l].text)}`
      );
      return `The ${numWord(total)} models split — ${parts.join('; ')}.`;
    }

    // User picked. Compare them to the lineup.
    if (others.length === 0) {
      return `All ${numWord(total)} models said the same.`;
    }
    // Lone-outlier case (user picked a letter no model picked, and the models
    // all converged on one other letter) — read more naturally as
    // "None matched. All five would …" than "The other five would …".
    if (userMatches === 0 && others.length === 1 && counts[others[0]] === total) {
      const other = others[0];
      return `None of the ${numWord(total)} models matched. All ${numWord(total)} would ${optionActionPhrase(optMap[other].text)}.`;
    }
    const matchLine = userMatches > 0
      ? `${capitalize(numWord(userMatches))} of ${numWord(total)} models said the same.`
      : `None of the ${numWord(total)} models matched.`;
    if (others.length === 1) {
      const other = others[0];
      const remaining = userMatches > 0 ? (total - userMatches) : counts[other];
      return `${matchLine} The other ${numWord(remaining)} would ${optionActionPhrase(optMap[other].text)}.`;
    }
    // Two or more other letters showed up: list each.
    const parts = others.map(l =>
      `${numWord(counts[l])} would ${optionActionPhrase(optMap[l].text)}`
    );
    return `${matchLine} The rest split — ${parts.join('; ')}.`;
  }

  function buildShareText(dilemma, modelResp, userAns) {
    const url = makeShareUrl(dilemma.id);
    // 360 chars is enough for most dilemmas to include the trigger detail
    // (e.g., D002 needs the apartment-lease sentence, not just the secrecy
    // setup). The greedy sentence accumulator stops at natural breaks, so
    // shorter scenarios still end cleanly.
    const lede = scenarioLede(dilemma.scenario, 360);
    const counts = countModelLetters(dilemma, modelResp);
    const userLetter = userAns ? userAns.choice : null;
    const split = describeModelSplit(dilemma, counts, userLetter);

    const lines = [];
    lines.push(`"${dilemma.title}"`);
    lines.push('');
    lines.push(lede);
    lines.push('');
    if (userLetter) {
      const userOpt = dilemma.options.find(o => o.id === userLetter);
      if (userOpt) lines.push(`My answer: ${optionActionPhrase(userOpt.text)}.`);
    }
    if (split) lines.push(split);
    lines.push('');
    lines.push('What would you do?');
    lines.push(url);
    return lines.join('\n');
  }

  async function shareCurrentDilemma() {
    if (!quiz.dilemma) return;
    const userAns = state.answers[quiz.dilemma.id] || null;
    const text = buildShareText(quiz.dilemma, state.modelResp, userAns);
    const status = $('share-dilemma-status');
    const setStatus = (msg) => { if (status) status.textContent = msg; };

    // Try Web Share first (iOS / modern Android push this to the system sheet).
    try {
      if (navigator.share) {
        await navigator.share({ text });
        setStatus('Shared.');
        return;
      }
    } catch (e) {
      // User canceled, or share refused — fall through to clipboard.
    }
    // Clipboard fallback.
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        setStatus('Copied to clipboard. Paste anywhere.');
        return;
      }
    } catch (e) { /* fall through */ }
    // Last-resort: select-and-copy via textarea.
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'absolute';
      ta.style.left = '-9999px';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
      setStatus('Copied to clipboard. Paste anywhere.');
    } catch (e) {
      setStatus('Could not copy — please copy manually.');
    }
  }

  // ---------------------------------------------------------------------------
  // Repo-link gating: hide every a[data-link="repo"] when REPO_URL is empty,
  // otherwise wire its href. Lets HTML markup stay clean and lets one constant
  // flip every GitHub CTA on the site.
  // ---------------------------------------------------------------------------
  function setupRepoLinks() {
    document.querySelectorAll('a[data-link="repo"]').forEach(a => {
      if (REPO_URL) {
        a.href = REPO_URL;
        a.removeAttribute('hidden');
      } else {
        a.hidden = true;
        // Trim the trailing "·" separator BEFORE this hidden link, but leave
        // the leading "·" of the next text node intact — that surviving "·"
        // becomes the connector between the previous and next visible items.
        // (Trimming both sides collapses "A · B · C · D" to "A · B  D" with
        // a stray double space and a missing connector.)
        const prev = a.previousSibling;
        if (prev && prev.nodeType === Node.TEXT_NODE) {
          prev.nodeValue = prev.nodeValue.replace(/[\s·•\-]+$/, ' ');
        }
      }
    });
    // Wrapper prose like "<span data-link='repo-prefix'>and the full <a ...>repo</a></span>"
    // should disappear entirely when there's no repo to point to.
    document.querySelectorAll('[data-link="repo-prefix"]').forEach(el => {
      if (!REPO_URL) el.hidden = true;
    });
  }

  function capitalize(s) {
    s = String(s || '');
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  // ---------------------------------------------------------------------------
  // Utility helpers
  // ---------------------------------------------------------------------------
  function optionShortSummary(text) {
    if (!text) return '';
    let s = String(text).replace(/^I\s+/, '');
    s = s.replace(/^['"“]/, '').replace(/['"”]$/, '');
    const cut = s.search(/[.—;]| — |, /);
    if (cut > 16 && cut < 90) s = s.slice(0, cut);
    s = s.trim().replace(/\.$/, '');
    if (s.length > 120) s = s.slice(0, 117).trimEnd() + '…';
    return s.charAt(0).toUpperCase() + s.slice(1);
  }
  function escapeHtml(s) {
    return String(s ?? '').replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
  }
  function escapeAttr(s) { return escapeHtml(s).replace(/\n/g, ' '); }
  function truncate(s, n) {
    if (!s) return '';
    if (s.length <= n) return s;
    return s.slice(0, n - 1).trimEnd() + '…';
  }

  // ---------------------------------------------------------------------------
  // Brand mark: an animated compass for the favicon + the nav logo. Injected
  // here so it lives in one place across every page. The needle "settles" like
  // a real compass finding north; honored prefers-reduced-motion in CSS.
  const COMPASS_SVG =
    '<svg class="logo-compass" viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" focusable="false">' +
    '<circle cx="12" cy="12" r="10.6" fill="none" stroke="currentColor" stroke-width="1.4" opacity="0.5"/>' +
    '<g class="logo-needle">' +
    '<polygon points="12,3.4 14.2,12 9.8,12" fill="#cc7755"/>' +
    '<polygon points="12,20.6 14.2,12 9.8,12" fill="#9a8f80"/>' +
    '</g>' +
    '<circle cx="12" cy="12" r="1.3" fill="var(--bg)" stroke="currentColor" stroke-width="0.7"/>' +
    '</svg>';

  function brandCompass() {
    if (!document.querySelector('link[rel="icon"]')) {
      const l = document.createElement('link');
      l.rel = 'icon'; l.type = 'image/svg+xml'; l.href = 'favicon.svg';
      document.head.appendChild(l);
    }
    document.querySelectorAll('.nav-brand').forEach(el => {
      if (!el.querySelector('.logo-compass')) el.insertAdjacentHTML('afterbegin', COMPASS_SVG);
    });
  }

  // Per-page boot
  // ---------------------------------------------------------------------------
  async function boot() {
    const page = document.body.dataset.page || 'landing';
    brandCompass();
    loadState();
    setupRepoLinks();

    if (page === 'landing') {
      const tryBtn = $('btn-try');
      if (tryBtn) tryBtn.addEventListener('click', () => { location.href = 'quiz.html'; });
      const seeBtn = $('btn-findings');
      if (seeBtn) seeBtn.addEventListener('click', () => { location.href = 'findings.html'; });
      // If the user has already answered some, surface the compass CTA.
      try {
        if (answeredCount() >= COMPASS_MIN_ANSWERS) {
          const r = $('returning-row');
          if (r) {
            r.hidden = false;
            $('returning-count').textContent = String(answeredCount());
          }
        }
      } catch (e) {}
      return;
    }

    if (page === 'quiz') {
      await loadData();
      const target = parseQuizHash();
      let d = target ? getDilemma(target) : null;
      if (!d) d = pickRandomDilemma();
      quiz.dilemma = d;
      // Set the URL to reflect what we're on, so it's deep-linkable.
      try { history.replaceState({}, '', 'quiz.html#/q/' + d.id); } catch (e) {}
      initQuizDeck();
      return;
    }

    if (page === 'compass') {
      await loadData();
      if (answeredCount() < COMPASS_MIN_ANSWERS) {
        $('compass-content').hidden = true;
        $('compass-gate-page').hidden = false;
        renderCompassGate();
        return;
      }
      $('compass-content').hidden = false;
      $('compass-gate-page').hidden = true;
      $('btn-share').addEventListener('click', nativeShareOrFallback);
      $('btn-download').addEventListener('click', downloadShareCanvas);
      $('btn-more').addEventListener('click', () => { location.href = 'quiz.html'; });
      $('btn-restart').addEventListener('click', () => {
        if (confirm('Clear all your answers and start over?')) {
          state.answers = {}; state.skipped = {}; saveState();
          location.href = 'index.html';
        }
      });
      renderCompass();
      return;
    }

    if (page === 'findings') {
      await loadData();          // dilemmas + model_responses, for the scene/reveal slides
      const findings = await loadFindings();
      initFindingsDeck(findings);
      return;
    }

    if (page === 'experiments') {
      await renderExperiments();
      return;
    }

    // methodology - nothing to wire up.
  }

  // Boot when DOM is ready.
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

})();
