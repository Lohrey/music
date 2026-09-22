# YuE2 — Exclude / instrumental / vocals research note

**Date:** 2026-09-14  
**Scope:** Upstream YuE / YuE2 (for `yue2_groove` + Pinokio users)  
**Question:** Does YuE2 support Suno-style **exclude** or **instrumental**? How to reduce vocals / humming?

---

## Executive summary

**YuE2 has no Suno-style `exclude` / `instrumental` switch.**

Official `SongRequest` fields are only:

`style`, `lyrics`, `cot`, `seed`, `abc`, `cfg_scale`, `id`

Code named `negative_prefix` is the **internal CFG negative branch** (unconditional path that drops Tags/Lyrics). It is **not** a user-facing “don’t include vocals / humming” list.

Product framing on [map-yue2.github.io](https://map-yue2.github.io/) and the [HF model card](https://huggingface.co/m-a-p/YuE2-3B) is: **lyrics + style → full song with vocals and accompaniment**. There is **no documented exclude API**.

Maintainers closed “generate music without lyrics” as **`not_planned`** ([Issue #112](https://github.com/multimodal-art-projection/YuE/issues/112)).

---

## Upstream evidence (repo / docs / issues)

### 1. YuE1 era (still referenced from README)

[Issue #18 — How to generate just music, no lyrics?](https://github.com/multimodal-art-projection/YuE/issues/18) (closed)

Maintainer guidance:

1. Remove **vocal-related tags** from genre/style (`vocal`, `female voice`, language tags, etc.).
2. Do **not** use a truly empty lyrics file. Use **section labels + several blank lines**:

```text
[verse]



[chorus]



[outro]
```

Reporter confirmed this could yield non-vocal results.

YuE1 pipelines also wrote a separate **instrumental stem** in the output folder — that is a **stem artifact**, not an exclude control.

YuE README still points “instrumental only” readers to issue #18.

### 2. YuE2 (current stack used by groove)

- [HF Discussion #1](https://huggingface.co/m-a-p/YuE2-3B/discussions/1) — “reliably instrumental?”  
  Community recipe (unofficial):
  - **PLAN MODE = `full`**
  - **Do not** put language / singer / vocal character in **style**
  - Lyrics = **empty section tags** (same idea as #18)
  - Writing only `instrumental` or leaving lyrics empty is **unreliable**; coherent singing still appears often

- [Issue #172](https://github.com/multimodal-art-projection/YuE/issues/172) (open as of 2026-09-14) — “always get vocals on top”  
  Multiple users; no new official API; discussion points back to older threads / HF tips.

- Repo **Discussions / Wiki are disabled**. GitHub Pages is demos/benchmarks, **not** exclude documentation.

### 3. Harder citations (second pass)

1. **[Issue #112](https://github.com/multimodal-art-projection/YuE/issues/112)** — “Generate music without lyrics” closed as **`not_planned`** by a maintainer.
2. Official skill doc states there is **no** request field for `negative_prompt` (among other missing controls):  
   [skills/yue2-music/references/generation-and-covers.md](https://github.com/multimodal-art-projection/YuE/blob/main/skills/yue2-music/references/generation-and-covers.md)
3. **Stronger structural workaround than empty lyrics:** edit the ABC — put rests on **`V: Vocal`**, put themes/solos on **`V: Ins`**, regenerate with that `abc` (editing-workflow docs). Still **not** a Suno toggle.

Internal CFG docs (`docs/generation.md` / skills): CFG negative is `instruction_only` or `same_instruction_and_exact_abc` — again, not user exclude text.

### 4. Gap vs Suno-like products

| Capability | Suno-like tools | YuE2 |
|---|---|---|
| Exclude list | Often yes | **No** |
| Instrumental toggle | Often yes | **No** (prompt convention only) |
| Empty lyrics → instrumental | Common | Empty / style-only still often hums or sings |
| Negative CFG as product exclude | User-facing | Internal CFG only; user cannot type “no X” |
| Default separate instrumental stem | Sometimes | YuE1 had stems; **YuE2 default is mixed stereo song** |

---

## Practical recipes inside yue2_groove

### A. Style-only / want instrumental

1. **Style:** instruments, mood, BPM — **avoid** `English`, `female voice`, `vocals`, `singer`.  
   Optional soft hints: `instrumental`, `no vocals`, `guitar and drums only` (**not guaranteed**).
2. **Lyrics:** do **not** leave blank. Use empty sections, e.g.:

```text
[Verse]


[Chorus]


[Bridge]


[Outro]


```

3. **PLAN MODE:** **`full`** (community consensus for YuE2 instrumental attempts).
4. Try several **seeds**. Raising CFG alone is a weak lever; prefer stripping vocal words from style + empty-section lyrics.
5. If still vocal-heavy: **EDIT** the ABC (`V: Vocal` rests / `V: Ins` melody) and regenerate with fixed `abc`.

### B. Have lyrics, hate hummed intros

Upstream has **no** “exclude humming” control. Try:

- Avoid unstable **`[Intro]`** when possible (YuE1 docs already noted intro as less stable; prefer starting at verse/chorus).
- Start lyrics cleanly at **`[Verse]`**; don’t put `humming` / `vocalise` / `scat` in style.
- After generate: **EDIT** ABC and regenerate, or stem-separate in a DAW / UVR — YuE2 usually does **not** ship YuE1-style ready `itrack` stems.

### C. API / UI note

The Python API requires both `style` and `lyrics` strings. An empty string is not “instrumental mode.” Prefer **structured empty sections** over truly empty lyrics.

---

## Productization note (groove UI only)

Upstream will not grow an exclude API from these issues (`not_planned` on #112).  

If groove wants better UX, the realistic path is a **UI shortcut** (e.g. one-click “instrumental template”: strip vocal style tokens + fill empty-section lyrics + set `cot=full`) — **without** changing YuE2 kernels. Reliability remains best-effort.

---

## Bottom line

**Exclude does not exist in YuE2.**  
**Instrumental** is a **community prompt/ABC workaround** (no-vocal style + empty lyric sections + `full` plan, optionally mute `V: Vocal` in ABC), and it remains **unstable**.  

Pain points (vocals when style-only; hummed intros with lyrics) match [#172](https://github.com/multimodal-art-projection/YuE/issues/172) and [HF Discussion #1](https://huggingface.co/m-a-p/YuE2-3B/discussions/1).

### Primary links

| Resource | URL |
|---|---|
| Repo | https://github.com/multimodal-art-projection/YuE |
| Docs / demos | https://map-yue2.github.io/ |
| Model card | https://huggingface.co/m-a-p/YuE2-3B |
| Issue #18 (YuE1 instrumental recipe) | https://github.com/multimodal-art-projection/YuE/issues/18 |
| Issue #112 (`not_planned`) | https://github.com/multimodal-art-projection/YuE/issues/112 |
| Issue #172 (open, YuE2 vocals) | https://github.com/multimodal-art-projection/YuE/issues/172 |
| HF Discussion #1 | https://huggingface.co/m-a-p/YuE2-3B/discussions/1 |
| Skill: no `negative_prompt` | https://github.com/multimodal-art-projection/YuE/blob/main/skills/yue2-music/references/generation-and-covers.md |
