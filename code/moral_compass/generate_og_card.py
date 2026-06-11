#!/usr/bin/env python3
"""Generate a static 1200x630 OG card for moral_compass link previews.

Run from this directory; writes og-card.png next to the HTML files.
Matches the site's terracotta-on-cream visual identity.
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

W, H = 1200, 630
BG = (250, 250, 247)        # --bg
INK = (27, 27, 27)          # --ink
INK_SOFT = (74, 74, 74)     # --ink-soft
INK_DIM = (125, 122, 115)   # --ink-dim
ACCENT = (136, 74, 57)      # --accent
ACCENT_BRIGHT = (204, 119, 85)  # --accent-bright
RULE = (226, 220, 209)      # --rule

FONT_BOLD = "/System/Library/Fonts/Helvetica.ttc"
FONT_REG = "/System/Library/Fonts/Helvetica.ttc"
FONT_SERIF = "/System/Library/Fonts/Supplemental/Georgia.ttf"

def load(font_path: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size, index=index)

def text_width(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]

def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Terracotta vertical accent on left edge
    d.rectangle([(0, 0), (12, H)], fill=ACCENT)

    # Top kicker
    kicker_font = load(FONT_BOLD, 22, index=0)
    d.text((72, 70), "MORAL COMPASS", font=kicker_font, fill=ACCENT, spacing=4)

    # Underline below kicker
    d.rectangle([(72, 105), (220, 107)], fill=ACCENT)

    # Main headline (3 lines)
    title_font = load(FONT_BOLD, 58, index=1)  # bold variant
    headline_lines = [
        "15 AI models.",
        "140 moral dilemmas.",
        "They disagree more than you'd think.",
    ]
    y = 150
    for line in headline_lines:
        d.text((72, y), line, font=title_font, fill=INK)
        y += 76

    # Subhead - the surprising stat
    sub_font = load(FONT_REG, 28, index=4)  # regular italic
    d.text(
        (72, y + 18),
        "Same dilemma, swap the names: the answer flips about a quarter of the time.",
        font=sub_font,
        fill=INK_SOFT,
    )

    # Bottom rule
    d.rectangle([(72, H - 84), (W - 72, H - 82)], fill=RULE)

    # Footer left: URL
    foot_font = load(FONT_REG, 22, index=0)
    d.text(
        (72, H - 56),
        "larryxiao.github.io/compass",
        font=foot_font,
        fill=INK_DIM,
    )

    # Footer right: tag
    foot_right = "Independent research probe · MIT"
    fr_w = text_width(d, foot_right, foot_font)
    d.text((W - 72 - fr_w, H - 56), foot_right, font=foot_font, fill=INK_DIM)

    out = Path(__file__).parent / "og-card.png"
    img.save(out, "PNG", optimize=True)
    print(f"wrote {out}  ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
