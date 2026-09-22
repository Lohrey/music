"""Bearbone DS v0.2, two scenes: palettes, the Gradio theme and the stylesheet assembly."""

from __future__ import annotations

from gradio.themes import Base, sizes

from .. import config

# ─────────────────────── Bearbone DS v0.2, two scenes ───────────────────────
# dark scene:   warm near-black #0B0A09 ground + ivory #F1ECE2 ink (never pure black/white).
# bright scene: ivory #F1ECE2 ground + warm near-black #16140F ink; dark controls
# (primary buttons, selected chips/checkboxes, sliders) share the #11141C family.
# Shared grammar: 1px strokes, 10px panels, no shadows / gradients / glow, monospace, ops footer.

FONT_STACK = [
    "Berkeley Mono",
    "Sarasa Mono SC",
    "JetBrains Mono",
    "SF Mono",
    "Noto Sans Mono CJK SC",
    "ui-monospace",
    "Menlo",
    "monospace",
]

DARK = {
    "bg": "#0B0A09",
    "bg_panel": "#12110F",
    "bg_input": "#171512",
    "bg_lift": "#1C1916",
    "fg": "#F1ECE2",
    "fg2": "#B5AEA2",
    "fg3": "#7A746A",
    "fg4": "#57524A",
    "stroke": "#2E2B27",
    "stroke2": "#8C8477",
    "primary_fill": "#F1ECE2",
    "primary_hover": "#FBF8F2",
    "primary_text": "#16140F",
    "danger": "#D08A8A",
}

BRIGHT = {
    "bg": "#F1ECE2",
    "bg_panel": "#F7F3EB",
    "bg_input": "#F7F3EB",
    "bg_lift": "#EAE4D8",
    "fg": "#16140F",
    "fg2": "#4A463F",
    "fg3": "#7A746A",
    "fg4": "#A69D8D",
    "stroke": "#C4BBA8",
    "stroke2": "#7E7462",
    "primary_fill": "#11141C",
    "primary_hover": "#2B3244",
    "primary_text": "#F1ECE2",
    "danger": "#B23B3B",
}


def _theme_values(p):
    """Gradio theme variables (snake_case); both scenes share this mapping."""
    return {
        "body_background_fill": p["bg"],
        "body_text_color": p["fg"],
        "body_text_color_subdued": p["fg3"],
        "body_text_size": "13px",
        "background_fill_primary": p["bg"],
        "background_fill_secondary": p["bg_panel"],
        "block_background_fill": p["bg_panel"],
        "block_border_color": "transparent",
        "block_border_width": "0px",
        "block_radius": "10px",
        "block_padding": "12px",
        "block_label_background_fill": "transparent",
        "block_label_border_width": "0px",
        "block_label_text_color": p["fg3"],
        "block_label_text_size": "11px",
        "block_label_text_weight": "500",
        "block_label_padding": "0 0 6px 0",
        "block_label_margin": "0",
        "block_title_text_color": p["fg2"],
        "block_title_text_weight": "500",
        "container_radius": "12px",
        "panel_background_fill": p["bg"],
        "panel_border_color": p["stroke"],
        "panel_border_width": "1px",
        "border_color_primary": p["stroke"],
        "border_color_accent": p["stroke2"],
        "border_color_accent_subdued": p["stroke"],
        "color_accent": p["primary_fill"],
        "color_accent_soft": p["stroke"],
        "link_text_color": p["fg"],
        "link_text_color_hover": p["fg2"],
        "link_text_color_active": p["fg"],
        "link_text_color_visited": p["fg2"],
        "code_background_fill": p["bg_input"],
        "input_background_fill": p["bg_input"],
        "input_background_fill_focus": p["bg_input"],
        "input_background_fill_hover": p["bg_input"],
        "input_border_color": p["stroke"],
        "input_border_color_focus": p["stroke2"],
        "input_border_color_hover": p["stroke2"],
        "input_border_width": "1px",
        "input_radius": "8px",
        "input_placeholder_color": p["fg4"],
        "input_text_size": "13px",
        "input_shadow": "none",
        "input_shadow_focus": "none",
        "button_border_width": "1px",
        "button_large_radius": "9px",
        "button_medium_radius": "9px",
        "button_small_radius": "8px",
        "button_large_text_size": "12px",
        "button_large_text_weight": "600",
        "button_large_padding": "8px 18px",
        "button_small_text_size": "11px",
        "button_small_text_weight": "500",
        "button_small_padding": "4px 10px",
        "button_primary_background_fill": p["primary_fill"],
        "button_primary_background_fill_hover": p["primary_hover"],
        "button_primary_border_color": p["primary_fill"],
        "button_primary_border_color_hover": p["primary_hover"],
        "button_primary_text_color": p["primary_text"],
        "button_primary_text_color_hover": p["primary_text"],
        "button_primary_shadow": "none",
        "button_primary_shadow_hover": "none",
        "button_primary_shadow_active": "none",
        "button_secondary_background_fill": "transparent",
        "button_secondary_background_fill_hover": p["bg_input"],
        "button_secondary_border_color": p["stroke"],
        "button_secondary_border_color_hover": p["stroke2"],
        "button_secondary_text_color": p["fg2"],
        "button_secondary_text_color_hover": p["fg"],
        "button_secondary_shadow": "none",
        "button_secondary_shadow_hover": "none",
        "button_secondary_shadow_active": "none",
        "button_cancel_background_fill": "transparent",
        "button_cancel_background_fill_hover": p["bg_input"],
        "button_cancel_border_color": p["stroke"],
        "button_cancel_border_color_hover": p["stroke2"],
        "button_cancel_text_color": p["fg3"],
        "button_cancel_text_color_hover": p["fg"],
        "button_cancel_shadow": "none",
        "button_cancel_shadow_hover": "none",
        "button_cancel_shadow_active": "none",
        "shadow_drop": "none",
        "shadow_drop_lg": "none",
        "shadow_inset": "none",
        "checkbox_background_color": p["bg_input"],
        "checkbox_background_color_selected": p["primary_fill"],
        "checkbox_background_color_hover": p["bg_lift"],
        "checkbox_border_color": p["fg4"],
        "checkbox_border_color_selected": p["primary_fill"],
        "checkbox_border_width": "1px",
        "checkbox_border_radius": "3px",
        "checkbox_check": p["primary_text"],
        "checkbox_shadow": "none",
        "checkbox_label_background_fill": p["bg_input"],
        "checkbox_label_background_fill_hover": p["bg_lift"],
        "checkbox_label_background_fill_selected": p["primary_fill"]
        if p is BRIGHT
        else p["stroke"],
        "checkbox_label_border_color": p["stroke"],
        "checkbox_label_border_color_selected": p["primary_fill"] if p is BRIGHT else p["stroke2"],
        "checkbox_label_text_color": p["fg2"],
        "checkbox_label_text_color_selected": p["primary_text"] if p is BRIGHT else p["fg"],
        "checkbox_label_shadow": "none",
        "checkbox_label_shadow_hover": "none",
        "checkbox_label_shadow_active": "none",
        "slider_color": p["primary_fill"],
        "loader_color": p["primary_fill"],
        "stat_background_fill": p["bg_input"],
        "table_border_color": p["stroke"],
        "table_even_background_fill": p["bg_panel"],
        "table_odd_background_fill": p["bg"],
        "table_text_color": p["fg2"],
        "table_radius": "8px",
        "error_background_fill": p["bg_input"],
        "error_border_color": p["stroke2"],
        "error_text_color": p["fg"],
        "error_icon_color": p["fg2"],
        "accordion_text_color": p["fg2"],
        "section_header_text_size": "13px",
        "section_header_text_weight": "500",
        "embed_radius": "10px",
        "layout_gap": "10px",
        "form_gap_width": "10px",
    }


def _bb_vars(p):
    """Scene palette → --bb-* variables for the custom CSS."""
    return {
        "--bb-field": p["bg"],
        "--bb-panel": p["bg_panel"],
        "--bb-well": p["bg_input"],
        "--bb-lift": p["bg_lift"],
        "--bb-ink": p["fg"],
        "--bb-ink2": p["fg2"],
        "--bb-ink3": p["fg3"],
        "--bb-ink4": p["fg4"],
        "--bb-line": p["stroke"],
        "--bb-line2": p["stroke2"],
        "--bb-primary-bg": p["primary_fill"],
        "--bb-primary-bg-hover": p["primary_hover"],
        "--bb-primary-fg": p["primary_text"],
        "--bb-danger": p["danger"],
        "--bb-chip-bg": p["stroke"] if p is not BRIGHT else p["primary_fill"],
        "--bb-chip-fg": p["fg"] if p is not BRIGHT else p["primary_text"],
    }


def _palette_css(selector, p):
    """Write the scene palette as --bb-* variables for the custom CSS."""
    scheme = "dark" if p is DARK else "light"
    lines = [f"{selector} {{"]
    for key, value in _bb_vars(p).items():
        lines.append(f"  {key}: {value};")
    lines.append(f"  color-scheme: {scheme};")
    lines.append("}")
    return "\n".join(lines)


def _scene_css(selector, values, palette=None):
    """Override the Gradio theme variables (runtime switch to the bright scene).

    Gradio merges custom CSS per selector, so the --bb-* variables must live in the
    same rule as the theme variables or the two overwrite each other.
    """
    lines = [f"{selector} {{"]
    if palette is not None:
        for key, value in _bb_vars(palette).items():
            lines.append(f"  {key}: {value} !important;")
        lines.append(f"  color-scheme: {'dark' if palette is DARK else 'light'} !important;")
    for key, value in values.items():
        lines.append(f"  --{key.replace('_', '-')}: {value} !important;")
    lines.append("}")
    return "\n".join(lines)


# Gradio adds `.dark` to <body> when the OS is in dark appearance and re-declares its
# theme variables there; an html-level override would lose to that local declaration.
# So the bright scene must be applied on every element that can carry the theme scope.
BASE_CSS = config.static_text("bearbone.css")

# Gradio adds `.dark` to <body> when the OS is in dark appearance and re-declares its
# theme variables there; an html-level override would lose to that local declaration.
# So the bright scene must be applied on every element that can carry the theme scope.
BRIGHT_SELECTOR = (
    "html.bb-bright, html.bb-bright body, html.bb-bright body.dark, "
    "html.bb-bright .dark, html.bb-bright gradio-app, html.bb-bright .gradio-container"
)

BEARBONE_CSS = "\n".join(
    [
        _palette_css(":root", DARK),
        BASE_CSS,
        _scene_css(BRIGHT_SELECTOR, _theme_values(BRIGHT), palette=BRIGHT),
        config.static_text("library.css"),
    ]
)


def bb_theme():
    """Bearbone dark scene at startup; bright is switched at runtime via CSS variables."""
    import inspect as _inspect

    theme = Base(font=FONT_STACK, font_mono=FONT_STACK, radius_size=sizes.radius_sm)
    values = _theme_values(DARK)
    valid = set(_inspect.signature(Base.set).parameters)
    both = {}
    for key, val in values.items():
        both[key] = val
        if f"{key}_dark" in valid:
            both[f"{key}_dark"] = val
    return theme.set(**both)
