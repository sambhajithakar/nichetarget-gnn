"""Print-ready vector figure toolkit (no matplotlib needed).

Figures are laid out as SVG in *points* (1 pt = 1/72 in), so a font-size of 7
in the SVG is 7 pt on the printed page when the figure is reproduced at its
design width (90 mm single column = 255.1 pt; 190 mm double column = 538.6 pt).

`render()` writes, with headless Chromium (Playwright):
  * <base>.pdf  vector PDF (fonts embedded by Chromium), page = figure size
  * <base>.tif  LZW-compressed RGB TIFF at the requested dpi at final width
  * <base>.png  optional preview (same pixels as the TIFF) if preview_dir given
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from html import escape

import numpy as np
from PIL import Image

FONT = "Arial, 'Liberation Sans', Helvetica, 'DejaVu Sans', sans-serif"
MM = 72 / 25.4                      # pt per mm
W1, W2 = 90 * MM, 190 * MM          # single / double column widths in pt

# Okabe-Ito
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000",
      "grey": "#999999", "lightgrey": "#C8C8C8"}

MINUS = "−"


def num(v, fmt="{:g}"):
    """Format a number using a true minus sign."""
    s = fmt.format(v)
    return s.replace("-", MINUS) if s.startswith("-") else s


def text(x, y, s, size=7, anchor="start", weight="normal", rotate=None, fill="#222", italic=False, raw=False,
         baseline=None):
    tr = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    st = ' font-style="italic"' if italic else ""
    bl = f' dominant-baseline="{baseline}"' if baseline else ""
    body = s if raw else escape(str(s))
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{fill}"{st}{bl}{tr}>{body}</text>')


def line(x1, y1, x2, y2, stroke="#444", sw=0.5, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{sw}"{d}/>'


def rect(x, y, w, h, fill="none", stroke="none", sw=0.5, rx=0):
    return (f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 0):.2f}" height="{max(h, 0):.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" rx="{rx}"/>')


def circle(cx, cy, r, fill="#000", stroke="none", sw=0.4, opacity=1.0):
    op = f' fill-opacity="{opacity}"' if opacity < 1 else ""
    return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{op}/>'


def marker(shape, cx, cy, s, fill, stroke="white", sw=0.4):
    """Filled marker; s = half-size in pt."""
    if shape == "circle":
        return circle(cx, cy, s, fill, stroke, sw)
    if shape == "square":
        return rect(cx - s * 0.9, cy - s * 0.9, 1.8 * s, 1.8 * s, fill, stroke, sw)
    if shape == "diamond":
        k = s * 1.25
        pts = f"{cx},{cy - k} {cx + k},{cy} {cx},{cy + k} {cx - k},{cy}"
    elif shape == "triangle":
        k = s * 1.2
        pts = f"{cx},{cy - k} {cx + k},{cy + 0.8 * k} {cx - k},{cy + 0.8 * k}"
    else:
        raise ValueError(shape)
    return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def arrow(x1, y1, x2, y2, color="#1b3f80", sw=1.2, head=4.0):
    ang = np.arctan2(y2 - y1, x2 - x1)
    bx, by = x2 - head * np.cos(ang), y2 - head * np.sin(ang)
    px, py = -np.sin(ang) * head * 0.55, np.cos(ang) * head * 0.55
    return (line(x1, y1, bx, by, color, sw) +
            f'<polygon points="{x2:.2f},{y2:.2f} {bx + px:.2f},{by + py:.2f} {bx - px:.2f},{by - py:.2f}" fill="{color}"/>')


def nice_ticks(lo, hi, n=6):
    """Round tick values inside [lo, hi]; step from {1, 2, 2.5, 5} x 10^k giving closest to n ticks."""
    best = None
    for e in range(-6, 7):
        for m in (1, 2, 2.5, 5):
            step = m * 10.0 ** e
            t = np.arange(np.ceil(lo / step - 1e-9), np.floor(hi / step + 1e-9) + 1) * step
            if len(t) < 2:
                continue
            score = (abs(len(t) - n), -len(t))
            if best is None or score < best[0]:
                best = (score, t)
    return [0.0 if abs(v) < 1e-12 else float(np.round(v, 10)) for v in best[1]]


def legend_items(x0, y0, items, size=7, sw=6, col_w=None, ncol=1, row_h=None, kind="rect", shapes=None):
    row_h = row_h or size + 3
    out = []
    for i, (lab, col) in enumerate(items):
        cx = x0 + (i % ncol) * (col_w or 0)
        cy = y0 + (i // ncol) * row_h
        if kind == "rect":
            out.append(rect(cx, cy - sw * 0.85, sw, sw, col))
        else:
            out.append(marker(shapes[i] if shapes else "circle", cx + sw / 2, cy - sw * 0.35, sw / 2, col, "none"))
        out.append(text(cx + sw + 3, cy, lab, size))
    return "\n".join(out)


def cmap(values, anchors):
    """Linear interpolation through a list of hex anchors; values in [0, 1]."""
    a = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in anchors], float)
    v = np.clip(np.asarray(values, float), 0, 1) * (len(a) - 1)
    i = np.minimum(v.astype(int), len(a) - 2)
    f = (v - i)[:, None]
    rgb = a[i] * (1 - f) + a[i + 1] * f
    return ["#%02x%02x%02x" % tuple(int(round(c)) for c in r) for r in rgb]


MAGMA = ["#000004", "#1c1044", "#4f127b", "#812581", "#b5367a", "#e55064", "#fb8761", "#fec287", "#fcfdbf"]


def render(body, W, H, base, dpi=1000, preview_dir=None):
    """Render SVG body (pt units) to <base>.pdf and <base>.tif at `dpi` for width W pt."""
    W, H = float(W), float(H)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.2f}pt" height="{H:.2f}pt" '
           f'viewBox="0 0 {W:.2f} {H:.2f}"><rect width="{W:.2f}" height="{H:.2f}" fill="white"/>{body}</svg>')
    px_w = int(np.ceil(W / 72 * dpi))
    css_w, css_h = W * 96 / 72, H * 96 / 72
    scale = (px_w + 6) / css_w        # small margin: Chromium rounds the device size down
    html = (f'<!doctype html><html><head><meta charset="utf-8"><style>@page {{ size: {W / 72:.4f}in {H / 72:.4f}in; margin: 0 }}'
            f' html,body{{margin:0;padding:0;background:white}} svg{{display:block}}</style></head>'
            f'<body>{svg}</body></html>')
    base = os.path.abspath(base)
    with tempfile.TemporaryDirectory() as td:
        hp = os.path.join(td, "fig.html")
        open(hp, "w", encoding="utf-8").write(html)
        png = os.path.join(td, "fig.png")
        js = os.path.join(td, "r.js")
        cfg = json.dumps({"html": "file://" + hp, "pdf": base + ".pdf", "png": png,
                          "vw": int(np.ceil(css_w)), "vh": int(np.ceil(css_h)), "scale": scale,
                          "cw": css_w, "ch": css_h, "pw": f"{W / 72:.4f}in", "ph": f"{H / 72:.4f}in"})
        open(js, "w").write(f"""
const {{ chromium }} = require('/opt/node22/lib/node_modules/playwright');
const c = {cfg};
(async () => {{
  const b = await chromium.launch({{executablePath: '/opt/pw-browsers/chromium'}});
  const p = await b.newPage({{deviceScaleFactor: c.scale, viewport: {{width: c.vw, height: c.vh}}}});
  await p.goto(c.html);
  await p.screenshot({{path: c.png, clip: {{x: 0, y: 0, width: c.cw, height: c.ch}}}});
  await p.pdf({{path: c.pdf, width: c.pw, height: c.ph, printBackground: true, preferCSSPageSize: true,
               margin: {{top: 0, right: 0, bottom: 0, left: 0}}, pageRanges: '1'}});
  await b.close();
}})();""")
        subprocess.run(["node", js], check=True)
        im = Image.open(png).convert("RGB")
        eff = round(im.size[0] / (W / 72))   # effective dpi at the design width
        assert eff >= dpi, (eff, dpi)
        im.save(base + ".tif", compression="tiff_lzw", dpi=(eff, eff))
        if preview_dir:
            os.makedirs(preview_dir, exist_ok=True)
            im.save(os.path.join(preview_dir, os.path.basename(base) + ".png"), dpi=(eff, eff))
    return im.size
