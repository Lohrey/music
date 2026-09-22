# View switch preflight — SONG / STUDIO

Phase B replaces the single always-on Studio workspace with two views: a
producer-facing **SONG** view and the existing **STUDIO** gear room. The view
switch must not destroy editable state: the rendered score SVG and the audio
player have been reset by remounts before, and the Accordion lazy-mounts its
contents.

Before building the UI we verified how Gradio actually hides a container, with
a real browser (Playwright driving the installed Google Chrome) against a tiny
Blocks app that holds an inline SVG and a playing `<audio>` element.

## Probe A — Gradio `visible=False` / `visible=True`

```python
with gr.Column(elem_id="studio", visible=True) as studio:
    gr.HTML('<svg id="probe-svg" data-render="1">…</svg>')
    gr.Audio(value="tone.wav", interactive=False)
hide.click(lambda: gr.update(visible=False), outputs=[studio])
show.click(lambda: gr.update(visible=True), outputs=[studio])
```

Result:

| observation | before hide | hidden | shown again |
| --- | --- | --- | --- |
| `#studio` present in DOM | yes | **no** | yes |
| `#probe-svg` present | yes | **no** | yes |
| same SVG node as before | — | **no** | **no** |
| `data-mark` survived | `pre` | — | `pre` (new node, re-rendered) |
| `<audio>` present | yes | **no** | yes |
| same audio node | — | **no** | **no** |
| audio playing | yes | **no** | **no** (reset to 0, paused) |

**Conclusion A:** Gradio's `visible` toggle removes the container from the DOM
and remounts it on the way back. SVG and `<audio>` do not survive.

## Probe B — CSS show/hide of always-mounted roots

Both roots stay `visible=True` in Gradio; hiding is a class on `<html>` plus a
`display: none` rule, exactly like the existing theme / settings-rail toggles:

```python
with gr.Column(elem_id="studio", visible=True): …
with gr.Column(elem_id="song", visible=True): …
# css: html.bb-view-song #studio { display: none !important; }
#      html.bb-view-studio #song { display: none !important; }
```

Result:

| observation | before | SONG view | back to STUDIO |
| --- | --- | --- | --- |
| `#studio` present | yes | **yes** | yes |
| `#studio` display | flex | `none` | flex |
| same SVG node | yes | **yes** | **yes** |
| `data-mark` survived | `pre` | `pre` | `pre` |
| same audio node | yes | **yes** | **yes** |
| audio playing | yes | **yes** | **yes** |

**Conclusion B:** CSS show/hide keeps both views mounted. The Studio Blocks,
the Accordion contents, the rendered score and the player all survive a view
switch with no redraw and no playback reset.

## Decision

The view switch is a class on `<html>` (`bb-view-song` / `bb-view-studio`) and
two always-mounted roots (`#bb-song-root`, `#bb-studio-root`). The Studio Blocks
are never destroyed, so the Accordion lazy-mount and the player/SVG state are
not disturbed. No second player implementation is introduced for SONG; the SONG
player is the same kind of Gradio `Audio` player as everywhere else.

The probe lives outside the repo (it is a one-off verification); this document
records the method and the result.
