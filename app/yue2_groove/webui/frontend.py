"""Everything that shapes the page outside the handlers.

The ``<head>``: boot scripts that must run before the body paints, the server
data the bundled scripts read, the ``<script src>`` tags for ``static/``, the
tab-strip container query; the ``js=`` snippets Gradio wraps for the chrome
buttons; and the small server-rendered chrome shared by the tabs (score panel,
busy banner).
"""

from __future__ import annotations

import html
import json

from .. import config
from . import runtime

# ───────────────── cover tab (see sheetsage_adapter.py / cover.py) ─────────────────


def score_panel(prefix: str, empty: str, elem_id: str = "", abc: str | None = None) -> str:
    """HTML for one abcjs score panel; static/score.js fills the .bb-score-inner div.

    When *abc* is given the score is rendered straight from that text (a read-only
    panel, e.g. SONG).  Otherwise score.js finds the textarea whose label starts
    with *prefix* — the editable Studio panels keep working that way.
    """
    panel_id = f' id="{html.escape(elem_id, quote=True)}"' if elem_id else ""
    message = html.escape(empty)
    inline = f' data-bb-abc-text="{html.escape(abc, quote=True)}"' if abc is not None else ""
    return (
        f'<div class="bb-score-panel" data-bb-abc="{html.escape(prefix, quote=True)}" '
        f'data-bb-empty="{message}"{inline}{panel_id}>'
        f'<div class="bb-score-inner"><div class="bb-score-empty">{message}</div></div></div>'
    )


# ───────────────── edit tab (see edit_flow.py) ─────────────────

BUSY_HTML = (
    '<div id="bb-busy" role="status" aria-live="polite">'
    "● JOB RUNNING — a second job is refused until it finishes; use CANCEL on the "
    "running tab to stop it"
    "</div>"
)


def busy_banner():
    """Global busy indicator polled by a gr.Timer (no handler signature changes)."""
    return BUSY_HTML if runtime.RUNNING.locked() else ""


SAMPLING_VIEW_TOGGLE_JS = """() => {
  if (window.__bbToggleSamplingView) window.__bbToggleSamplingView();
}"""

THEME_TOGGLE_JS = """() => {
  const r = document.documentElement;
  const on = r.classList.toggle('bb-bright');
  try { localStorage.setItem('bb-theme', on ? 'bright' : 'dark'); } catch (e) {}
  const wrap = document.getElementById('bb-theme-btn');
  const btn = wrap && (wrap.tagName === 'BUTTON' ? wrap : wrap.querySelector('button'));
  if (btn) {
    btn.classList.toggle('bb-bright', on);
    const label = on ? 'Switch to the dark scene' : 'Switch to the bright scene';
    btn.title = label;
    btn.setAttribute('aria-label', label);
  }
  return '';
}"""

RAIL_TOGGLE_JS = """() => {
  const r = document.documentElement;
  const hidden = r.classList.toggle('bb-rail-hidden');
  try { localStorage.setItem('bb-rail', hidden ? 'off' : 'on'); } catch (e) {}
  const wrap = document.getElementById('bb-rail-btn');
  const btn = wrap && (wrap.tagName === 'BUTTON' ? wrap : wrap.querySelector('button'));
  if (btn) {
    btn.classList.toggle('bb-on', !hidden);
    const label = hidden ? 'Show the settings rail' : 'Hide the settings rail';
    btn.title = label;
    btn.setAttribute('aria-label', label);
  }
  return '';
}"""

# ── bundled frontend (yue2_groove/static, served through allowed_paths) ──────
# Every script that is pure logic is a file there: editors and `node --check`
# (tests/test_static_assets.py) see real JavaScript instead of a Python string.
# What stays inline in the <head> is what must run before the body paints or
# carries server data: the theme / rail / view boot scripts, __BB_TIPS__,
# __BB_EXAMPLES__, __BB_SAMPLING_DEFAULTS__ and the layout container query.
SAMPLING_KNOBS_FILE = config.STATIC_DIR / "sampling-knobs.js"


def _static_script(name: str) -> str:
    """A <script src> for a bundled file (the same route abcjs is served from)."""
    return f'<script src="/gradio_api/file={config.STATIC_DIR / name}"></script>'


TIPS = {
    "STYLE": "Genre, instruments, vocal character, language and BPM. The grey text is the repository example; click E to drop it in, or paste your own.",
    "LYRICS": "Words to sing. Use section tags like [Verse] / [Chorus]; line breaks shape the phrasing. Click E to drop in the repository example.",
    "PLAN MODE": "FULL = melody plus a chord plan (editable ABC); MELODY = melody only, no chord symbols, freer arrangement (best for covers); OFF = no symbolic plan at all.",
    "SEED": "-1 = a new random seed each run (the seed used is shown in STATUS and saved with the work). Same seed + same settings reproduces a take; any other number is a different take.",
    "CFG SCALE": "Prompt guidance strength. 0 = default (1.0, or 1.01 for off); higher follows the prompt harder, too high can sound harsh.",
    "OUTPUT ID": "Optional filename-safe name used for the run directory and the request id.",
    "ABC SCORE": "Optional ABC score used as the composition input. Leave empty to let the model plan.",
    "ABC": "Paste an ABC score to validate it, strip chords, or compare before/after edits.",
    "BUDGET PRESET": "Defaults to the upstream protocol caps (6:00). Long song raises both caps for a ~10-minute song (it needs long lyrics); the preview and quick presets only lower the caps to finish faster.",
    "temperature": "Sampling randomness. Lower = safer, more repetitive; higher = more varied but can drift.",
    "top_p": "Nucleus sampling: keep the smallest token set whose probability sums to p. Lower = tighter.",
    "top_k": "Sample only from the k most likely tokens. Smaller = safer, larger = more varied.",
    "repetition_penalty": "Penalises recently used tokens. Above 1 discourages repeats; too high hurts musicality.",
    "penalty_window": "How many recent tokens count for the repetition penalty.",
    "min_tokens": "Do not emit the end token before this many tokens (guarantees a minimum length).",
    "max_tokens": "Hard cap for this phase. Semantic tokens ≈ 25 per second of audio; the song ends when the score ends, usually earlier. Lyrics + score + song share the model's 24 576-token context: a cap that would not fit is trimmed at run time.",
    "RUN DIRECTORY": "A previously saved run that contains latent.npy. Re-decodes it without generating again.",
    "DECODER VAE": "SOURCE = the decoder recorded in the source run; STANDARD = normal listening decoder (YuE2-Vae); LEGACY = benchmark decoder; CUSTOM = the path or HF id below.",
    "CUSTOM VAE PATH / HF ID": "Path or Hugging Face repo id of a custom decoder.",
    "FULL DECODE": "Decode the whole latent at once (faster, more memory). Tiled chunks are safer for long songs.",
    "JSONL REQUESTS": "One JSON request per line: id, style/tags, lyrics, cot, seed, cfg_scale, abc or abc_path, optional abc_sampling / semantic_sampling overrides.",
    "OUTPUT NAME": "Folder name for this batch.",
    "KEEP VOICES": "Which voices survive the chord strip (04 TOOLS checker).",
    "MELODY VOICES": "Which melody voices a cover keeps: both (vocal + instrumental), Vocal only, or Ins only. Applies to STRIP CHORDS and to SEND / GENERATE COVER.",
    "AFTER // EDITED ABC": "The edited score; compared against the original above (05 TOOLS copy of the invariant check).",
    "COMPARE VOICES": "Which voices the invariant check compares.",
    "ALLOW TEMPO CHANGE": "Allow the quarter-note tempo to change without reporting it as a violation.",
    "RUN DIRECTORIES": "One saved run directory per line; each must contain result.json.",
    "VERIFY WEIGHT HASHES": "Re-hash the checkpoint files (about 7 GB, a few seconds) to confirm integrity.",
    "DEVICE": "auto picks CUDA → MPS → CPU.",
    "DTYPE": "bfloat16 is the checkpoint dtype and the default on CUDA/MPS; float32 casts the weights at load (slower, twice the memory). MPS bf16 needs torch >= 2.11 (the 2.10 SDPA defect is fixed there); see overrides/ in the repository.",
    "MODEL ID / LOCAL DIR": "Hugging Face repo id or a local model directory.",
    "DEFAULT VAE": "Decoder for new generations: STANDARD = YuE2-Vae (listening), LEGACY = benchmark decoder, CUSTOM = the path used for CUSTOM VAE below.",
    "CUSTOM VAE": "Used when the default VAE is set to custom.",
    "MODEL REVISION": "Pin a git revision of the model repository (optional).",
    "VAE REVISION": "Pin a revision of the VAE repository (optional).",
    "OFFLINE": "Never touch the network; use only the local Hugging Face cache.",
    "BACKEND": "torch uses CUDA graphs when available and eager elsewhere; vLLM is a CUDA-only fast path; gguf runs yue2.cpp (Q8_0 GGUF) — fits 12 GB cards, a different take for the same seed.",
    "QUANTIZATION": "fp8 shrinks the AR weights on CUDA sm89+; none everywhere else.",
    "OFFLOAD AR WEIGHTS": "Move AR weights to CPU during synthesis to save VRAM (single request only).",
    "MEMORY BUDGET": "CUDA-only memory cap in GiB; also selects VAE chunking (≤12 GiB → 512 frames).",
    "ODE STEPS": "Flow-matching steps for audio synthesis. More = higher quality but slower; 32 is the protocol default.",
    "VAE CORE FRAMES": "Chunk size for tiled VAE decode; auto = 512 at ≤12 GiB budget, otherwise 1024.",
    "WORKS": "Click a work to view and play it; tick the checkbox to select it for deletion. Both are independent.",
    "SORT": "Sort key: TIME = creation time, NAME = work name.",
    "ORDER": "Sort order: DESC = newest / Z→A first, ASC = oldest / A→Z first.",
    "RENAME TO": "New name for the work being viewed; the timestamp prefix is kept.",
    "LIBRARY STATUS": "Result of the last library action.",
    "SOURCE AUDIO": "Reference recording for the cover (wav/mp3/flac). SheetSage2 decodes it to mono 24 kHz; the melody becomes the symbolic condition.",
    "TRANSCRIPTION TASK": "MELODY // VOCAL keeps only the sung line; MELODY // VOCAL+INST keeps vocal and instrumental melodies (chord-free, best for covers); FULL SCORE adds chords for cot=full regeneration.",
    "MAX SECONDS (0 = WHOLE FILE)": "Deliberately crop the transcript to the first N seconds; 0 processes the whole file. Long files take longer and use more GPU memory.",
    "SHEETSAGE2 MODEL / DIR": "Hugging Face id (m-a-p/SheetSage2) or the path of a downloaded snapshot. MERT-v2-FullSong loads automatically as its parent encoder.",
    "BASE MODEL / MERT SNAPSHOT": "Optional local path of the MERT-v2-FullSong snapshot; passed as base_model_path so a fully offline adapter load does not need the Hub cache.",
    "KEEP SHEETSAGE2 WARM": "Keep one SheetSage2 process resident and reuse its loaded model between transcriptions (faster repeats) until UNLOAD SHEETSAGE2 or the idle timeout. Off: every transcription starts a fresh process and frees all memory on exit. Locked while a transcription runs; changes apply from the next request.",
    "COVER ABC": "The transcription, editable. Fix wrong notes/meter before covering; STRIP CHORDS removes harmony for cot=melody, SEND TO GENERATE fills 01 GENERATE and sets the plan mode.",
    "SOURCE WORK": "A saved work with a score.abc (generated plan or an earlier edit); its ABC becomes the frozen baseline.",
    "BASELINE": "Record of the frozen source: hashes plus copies of score.abc/request.json. The original run directory is never modified.",
    "EDITED ABC": "Your edit of the baseline score. It is validated and submitted explicitly — generation never falls back to a fresh plan.",
    "RESULT ABC": "The score actually submitted for the last generation (chord-stripped when the plan mode requires it). The editor above stays untouched.",
    "SAMPLING // FROM 01 GENERATE": "Read-only mirror of 01 GENERATE → ADVANCED // SAMPLING. Both flows share those sliders; change them there.",
    "CONTRACT": "What CHECK INVARIANTS must preserve: EXACT keeps notes, meter and durations (tempo/meter optional); PITCH keeps only the ordered pitch sequence, so rhythm may change; FREE records differences without gating anything.",
    "ALLOW METER CHANGE": "EXACT contract only: treat bar/time-grid differences as permitted instead of a violation.",
    "ALLOW MELODY/RHYTHM CHANGES": "Permit generating even when CHECK INVARIANTS did not pass (e.g. an intentional pitch edit under EXACT). The FREE contract already permits everything, so this override only matters for EXACT/PITCH; it never skips FREEZE BASELINE or the check itself.",
    "CHECK RESULT": "Result of CHECK INVARIANTS under the chosen CONTRACT: EXACT compares sounding notes and the meter grid (tempo/meter permissions optional), PITCH compares only the ordered pitch sequence, FREE lists the differences without gating anything. Chord-only edits pass under EXACT.",
    "COMPARISON": "Baseline vs edited render: a local listening page built from both run directories.",
}


# ── the tab strip follows the width of its column, not the viewport ──────
# Gradio 6 hides overflowing tabs behind a tiny ⋯ menu.  What decides whether
# the seven tabs fit is the main column, which is 100% of the frame on a phone
# or an iPad in portrait but 5/7 of it with the settings rail open (an iPad in
# landscape lost 05..07 that way, a 1440px desktop lost 07).  A container
# query on #bb-main sees exactly that width.  It has to travel in the <head>:
# prefix_css (see BASE_CSS) only knows style / @media / @keyframes /
# @font-face rules and drops an @container block from `css=` outright.  The
# head is mounted after the custom css, so this also wins ties on order.
# Two-up under 660px of column, four-up to 900px (the strip needs ~880px for
# one row), Gradio's own single row above that; the overflow containers turn
# into display:contents so every tab is a wrapped flex item rather than a
# dropdown entry.
LAYOUT_HEAD_CSS = """<style id="bb-layout">
#bb-main { container-type: inline-size; }
@container (width <= 900px) {
  .tabs .tab-wrapper { display: flex !important; flex-wrap: wrap !important;
                       height: auto !important; min-height: 32px; }
  .tabs .tab-container[role="tablist"],
  .tabs .overflow-menu,
  .tabs .overflow-dropdown { display: contents !important; }
  .tabs .overflow-menu > button { display: none !important; }
  .tabs .tab-container[role="tablist"] > button,
  .tabs .overflow-dropdown > button {
    flex: 1 1 22% !important; min-height: 38px; padding: 9px 6px !important;
    font-size: 10.5px !important; letter-spacing: .08em !important;
  }
}
@container (width <= 660px) {
  .tabs .tab-container[role="tablist"] > button,
  .tabs .overflow-dropdown > button { flex-basis: 44% !important; min-height: 40px; padding: 10px 6px !important; }
}
</style>"""

HEAD_HTML = (
    LAYOUT_HEAD_CSS
    + """<meta name="color-scheme" content="dark light">
<script>
(function () {
  try {
    var q = new URLSearchParams(location.search).get('theme');
    var saved = localStorage.getItem('bb-theme');
    var bright = q ? (q === 'bright') : (saved === 'bright');
    if (bright) document.documentElement.classList.add('bb-bright');
  } catch (e) {}
  try {
    // the settings rail starts hidden; "on" is the only value that shows it
    if (localStorage.getItem('bb-rail') !== 'on') {
      document.documentElement.classList.add('bb-rail-hidden');
    }
  } catch (e) {}
  try {
    // iOS zooms the page in when a field takes focus and keeps that zoom after
    // blur.  maximum-scale=1 switches the focus zoom off; iOS has ignored the cap
    // for pinch zoom since iOS 10, so the user can still zoom.  Android honours
    // the cap (no pinch), so only iOS gets it — iPadOS reports a Mac UA with touch.
    var ios = /iP(hone|ad|od)/.test(navigator.userAgent) ||
              (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
    var vp = document.querySelector('meta[name="viewport"]');
    if (ios && vp) {
      vp.setAttribute('content', 'width=device-width, initial-scale=1, maximum-scale=1, shrink-to-fit=no');
    }
  } catch (e) {}
  function sync() {
    var on = document.documentElement.classList.contains('bb-bright');
    var wrap = document.getElementById('bb-theme-btn');
    var btn = wrap && (wrap.tagName === 'BUTTON' ? wrap : wrap.querySelector('button'));
    if (btn) {
      btn.classList.toggle('bb-bright', on);
      var label = on ? 'Switch to the dark scene' : 'Switch to the bright scene';
      btn.title = label;
      btn.setAttribute('aria-label', label);
    }
    var railWrap = document.getElementById('bb-rail-btn');
    var railBtn = railWrap && (railWrap.tagName === 'BUTTON' ? railWrap : railWrap.querySelector('button'));
    if (railBtn) {
      // the rail lives inside STUDIO; view.js disables the toggle in SONG
      if (window.__bbApplyRail) {
        window.__bbApplyRail();
      } else {
        var hidden = document.documentElement.classList.contains('bb-rail-hidden');
        railBtn.classList.toggle('bb-on', !hidden);
        var label = hidden ? 'Show the settings rail' : 'Hide the settings rail';
        railBtn.title = label;
        railBtn.setAttribute('aria-label', label);
      }
    }
  }
  sync();
  [300, 1000, 2500, 5000].forEach(function (t) { setTimeout(sync, t); });
})();
</script>"""
)

HEAD_HTML += (
    "<script>window.__BB_TIPS__ = "
    + json.dumps(TIPS, ensure_ascii=False)
    + ";</script>"
    + _static_script("tips.js")
)

HEAD_HTML += _static_script("abcjs-basic-min.js") + _static_script("score.js")


HEAD_HTML += _static_script("abc-fold.js")


HEAD_HTML += _static_script("persist.js")

HEAD_HTML += (
    "<script>window.__BB_EXAMPLES__ = "
    + json.dumps(
        {"style": runtime.EXAMPLE_STYLE, "lyrics": runtime.EXAMPLE_LYRICS}, ensure_ascii=False
    )
    + ";</script>"
)
HEAD_HTML += _static_script("example.js")
HEAD_HTML += _static_script("library.js")

HEAD_HTML += _static_script("current-work.js")

# ── SONG / STUDIO view switch ──────────────────────────────────────────────
# The two roots are always mounted; the view is a class on <html> so a switch
# never remounts the Studio Blocks (see docs/VIEW_SWITCH_PREFLIGHT.md).  The
# boot script runs in <head> before the body to avoid a flash of both views
# and honours (in order): an explicit --view/env mode, the last stored choice,
# then the SONG default.  `--tab N` is passed through as the studio mode.
VIEW_BOOT_JS = """<script>
(function () {
  var mode = __BB_VIEW_MODE_JSON__;
  window.__BB_VIEW_MODE__ = mode;
  // an explicit --view / --tab must not write through to the remembered choice
  window.__BB_VIEW_FORCED__ = (mode === 'song' || mode === 'studio');
  try {
    var saved = localStorage.getItem('bb-view');
    var view = window.__BB_VIEW_FORCED__ ? mode
             : (saved === 'studio' ? 'studio' : 'song');
    document.documentElement.classList.add('bb-view-' + view);
  } catch (e) {
    document.documentElement.classList.add('bb-view-song');
  }
})();
</script>"""

HEAD_HTML += _static_script("view.js")


def head_html(view_mode: str = "auto") -> str:
    """The page <head> with the resolved initial view baked into the boot script."""
    if view_mode not in ("song", "studio"):
        view_mode = "auto"
    return VIEW_BOOT_JS.replace("__BB_VIEW_MODE_JSON__", json.dumps(view_mode)) + HEAD_HTML


def VIEW_SET_JS(view: str) -> str:
    """Frontend-only handler for the top-chrome SONG / STUDIO buttons."""
    return f"() => {{ if (window.__bbSetView) window.__bbSetView({json.dumps(view)}); }}"


SONG_LISTEN_JS = """() => {
  var wrap = document.getElementById('bb-song-player');
  if (wrap) wrap.scrollIntoView({ behavior: 'smooth', block: 'center' });
  var audio = wrap ? wrap.querySelector('audio') : null;
  if (audio) { try { audio.play(); } catch (e) {} }
}"""

HEAD_HTML += _static_script("song.js")
HEAD_HTML += (
    "<script>window.__BB_SAMPLING_DEFAULTS__ = "
    + json.dumps({"abc": runtime.ABC_DEFAULTS, "sem": runtime.SEM_DEFAULTS})
    + ";</script>"
    + _static_script("sampling-knobs.js")
)
