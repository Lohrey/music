# Contributing

Thanks for working on YUE2 // GROOVE.  This page is the map of the code and the
conventions that keep it readable; the user-facing manual is the README.

## Gates

Every change must pass what CI runs (`.github/workflows/tests.yml`):

```bash
.venv/bin/ruff check . && .venv/bin/ruff format --check .   # lint + formatting
.venv/bin/pyright                                          # types (basic), package only
.venv/bin/python -m compileall -q yue2_groove tests scripts
.venv/bin/python -m pytest -q
```

`ruff format .` fixes formatting; nobody hand-aligns code.  The lint rules and their
documented exceptions live in `[tool.ruff]` in `pyproject.toml` — add an ignore only
with the reason next to it.  With `node` on `PATH`, the test suite also syntax-checks
every bundled and inline script (`tests/test_static_assets.py`).

## Module map

| Module | Owns | Never |
|---|---|---|
| `adapter.py` | every call into `yue2` (lazy imports) | — |
| `config.py` | env / paths / defaults, `static_text()` | import torch or gradio |
| `library.py` | reading run directories by file convention; the details-pane HTML | import `yue2` |
| `workflow.py` | what a work *is*: identity, stage, family, next actions | touch the UI |
| `cover.py`, `edit_flow.py` | the COVER and EDIT request logic | import gradio |
| `sheetsage_adapter.py` | the SheetSage2 subprocess (start, progress, cancel, warm worker) | import transformers |
| `gguf_engine.py` | the yue2.cpp subprocesses (request JSON, log → progress, cancel), GGUF preparation, the BACKEND=auto rule, `install` / `prepare` / `check` | import `yue2` (it reaches the tokenizer and plan objects through `adapter`) or torch at import time |
| `sheetsage_driver.py` | the stdlib-only script run *inside* the SheetSage2 venv | import this package |
| `webui/` | the Gradio app (below) | — |
| `static/` | the bundled frontend: stylesheet + every client script | Python |
| `vendor/` | byte-identical upstream copies (see NOTICE) | be edited |

### `webui/`

```
runtime.py       the one pipeline (load / get / unload, RuntimeSettings), the one job
                 slot (try_start_job / end_job / CANCEL), run_generation + the
                 pending.json durability protocol, run_dir / slug / artifact_files
generate_tab.py  01 GENERATE, PLAN ONLY, ALL MODES, 06 DECODE, 07 BATCH
tools_tab.py     05 TOOLS
cover_tab.py     02 COVER
edit_tab.py      03 EDIT
library_tab.py   04 LIBRARY and the LIBRARY → EDIT / COVER handoffs
song_view.py     the SONG director and the current-work bridge
theme.py         palettes, the Gradio theme, stylesheet assembly
frontend.py      page head, bundled scripts, js= snippets, shared chrome
layout.py        build_ui: one builder per section + the event wiring
cli.py           main
```

Conventions inside the package:

- **Modules import each other as modules** and call `runtime.get_pipe(…)`,
  `library_tab.library_choices(…)`.  This is cycle-safe between the tabs, the owner is
  visible at every call site, and a test patches one place
  (`monkeypatch.setattr(webui.runtime, "get_pipe", …)`).  A name used from another
  module has no leading underscore.
- **State lives in `runtime`** and is read as `runtime.RUNS`, never copied into another
  module.  The CLI sets `runtime.RUNS`; nothing else does.
- **A generator handler** takes the Gradio inputs positionally, packs the settings rail
  into `runtime.RuntimeSettings(...)` and the sliders through `runtime.sampling_pair(...)`,
  then follows one skeleton:

  ```python
  if not runtime.try_start_job():
      yield <busy updates with runtime.BUSY_MESSAGE>; return
  try:
      yield <starting>
      ...
      yield <done>
  except Exception as exc:  # noqa: BLE001 — the UI reports, never crashes
      yield <runtime.failure_text(exc)>
  finally:
      runtime.end_job()
  ```

  Every `yield` between claiming the slot and the `finally` must be inside the `try`.
- **`layout.py`** builds each section in its own function that returns
  `_components(locals())`; the wiring reads `gen.run_btn`, `lib.lib_list`, `rail.device`.
  No section reaches into another at construction time — handoffs are wiring.

## The frontend

- Every pure-logic script is a file in `static/` and is included from `frontend.py`
  with `_static_script("name.js")`.  What stays inline in the `<head>` is only what
  must run before the body paints (theme / rail / view boot) or carries server data
  (`__BB_TIPS__`, `__BB_EXAMPLES__`, `__BB_SAMPLING_DEFAULTS__`).  Data is emitted
  before the script that reads it; `test_static_assets.py` checks the order.
- **Gradio 6 rewrites `css=`** (`prefix_css`): a top-level rule keeps an unscoped copy,
  a rule inside `@media` keeps only the copy prefixed with `.gradio-container… .contain`,
  and `@container` blocks are dropped.  So rules for the outer frame (`html`, `body`,
  `.gradio-container`, `.main.app`) must be top-level (`clamp()` instead of a breakpoint),
  and container queries ride in the `<head>` (`LAYOUT_HEAD_CSS`).
  `tests/test_webui_layout.py` pins this.
- **Gradio 6 morphs `gr.HTML` in place** rather than replacing it: an updated pane keeps
  the same DOM nodes, syncs their attributes to the new HTML (stripping any attribute
  the server HTML does not carry) and recurses.  Client state must therefore hang off
  the element as a JS property, never as a DOM attribute, and listeners are delegated
  from `document` rather than attached per node (`static/library.js` is the model).

## Tests

- Tests are model-free: they patch `webui.runtime.get_pipe` / `run_generation` and run
  the handlers as plain generators.  Put a new handler test next to its tab's tests.
- A regression test names the bug in its docstring and the measurement that found it.
- Browser behaviour (layout at phone / iPad widths, the library player) was verified
  with Playwright against Chromium and WebKit; those scripts are not in the repository,
  but the numbers they produced are in the commit messages of the fixes.
