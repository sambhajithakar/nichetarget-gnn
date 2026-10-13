"""Tiny figure toolkit used where matplotlib is unavailable.

Panels are written as SVG (charts) or rasterised with Pillow (spatial maps
with ~10^5 cells), assembled in one SVG, and rendered to PNG at 300 dpi with
headless Chromium (Playwright).
"""
from __future__ import annotations

import base64
import io
import os
import subprocess
import tempfile
from html import escape

import numpy as np
from PIL import Image, ImageDraw

FONT = "Arial, Helvetica, sans-serif"
# categorical palette (colour-blind-safe, Okabe-Ito based) + greys
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000",
           "#999999", "#8C564B", "#BCBD22", "#17BECF"]


def text(x, y, s, size=11, anchor="start", weight="normal", rotate=None, fill="#222", style=""):
    tr = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{fill}" style="{style}"{tr}>{escape(str(s))}</text>')


def raster_points(coords, colors, width_px, point=1, bg=(255, 255, 255)):
    """Rasterise points (coords in data units) into a PNG data URI; colors: list of hex per point."""
    c = np.asarray(coords, float)
    lo, hi = c.min(0), c.max(0)
    span = hi - lo
    h_px = int(round(width_px * span[1] / span[0]))
    img = Image.new("RGB", (width_px, h_px), bg)
    d = ImageDraw.Draw(img)
    xy = (c - lo) / span[0] * (width_px - 1)
    for (x, y), col in zip(xy, colors):
        d.ellipse([x - point, y - point, x + point, y + point], fill=col)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), h_px


def image(href, x, y, w, h):
    return f'<image href="{href}" x="{x}" y="{y}" width="{w}" height="{h}"/>'


def hbar(x0, y0, w, labels, values, color="#0072B2", vmax=None, label_w=150, fmt="{:.2f}", bar_h=14, gap=5,
         highlight=None, xlabel=None, ref_line=None):
    """Horizontal bar chart; returns (svg, height)."""
    vmax = vmax or max(max(values), 1e-9)
    out = []
    pw = w - label_w - 40
    for i, (lab, v) in enumerate(zip(labels, values)):
        y = y0 + i * (bar_h + gap)
        col = highlight.get(lab, color) if highlight else color
        out.append(text(x0 + label_w - 6, y + bar_h - 3, lab, 10, "end"))
        out.append(f'<rect x="{x0 + label_w}" y="{y}" width="{max(v, 0) / vmax * pw:.1f}" height="{bar_h}" fill="{col}"/>')
        out.append(text(x0 + label_w + max(v, 0) / vmax * pw + 4, y + bar_h - 3, fmt.format(v), 9))
    hgt = len(labels) * (bar_h + gap)
    if ref_line is not None:
        xr = x0 + label_w + ref_line / vmax * pw
        out.append(f'<line x1="{xr:.1f}" y1="{y0 - 3}" x2="{xr:.1f}" y2="{y0 + hgt}" stroke="#555" stroke-dasharray="3,3"/>')
    if xlabel:
        out.append(text(x0 + label_w + pw / 2, y0 + hgt + 14, xlabel, 10, "middle", fill="#444"))
        hgt += 18
    return "\n".join(out), hgt


def scatter(x0, y0, w, h, xs, ys, color="#0072B2", labels=None, xlabel="", ylabel="", r=2.2, diag=False,
            label_idx=(), zero_lines=True, title=None, opacity=0.6):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    pad = 38
    xmin, xmax = np.nanmin(xs), np.nanmax(xs)
    ymin, ymax = np.nanmin(ys), np.nanmax(ys)
    if diag:
        xmin = ymin = min(xmin, ymin)
        xmax = ymax = max(xmax, ymax)
    dx, dy = (xmax - xmin) * 0.06 + 1e-12, (ymax - ymin) * 0.06 + 1e-12
    xmin, xmax, ymin, ymax = xmin - dx, xmax + dx, ymin - dy, ymax + dy
    X = lambda v: x0 + pad + (v - xmin) / (xmax - xmin) * (w - pad - 8)  # noqa: E731
    Y = lambda v: y0 + h - pad + 8 - (v - ymin) / (ymax - ymin) * (h - pad - 8)  # noqa: E731
    out = [f'<rect x="{x0 + pad}" y="{y0 + 8}" width="{w - pad - 8}" height="{h - pad}" fill="none" stroke="#888"/>']
    if zero_lines:
        if xmin < 0 < xmax:
            out.append(f'<line x1="{X(0):.1f}" y1="{y0 + 8}" x2="{X(0):.1f}" y2="{y0 + h - pad + 8}" stroke="#ccc"/>')
        if ymin < 0 < ymax:
            out.append(f'<line x1="{x0 + pad}" y1="{Y(0):.1f}" x2="{x0 + w - 8}" y2="{Y(0):.1f}" stroke="#ccc"/>')
    cols = color if isinstance(color, (list, np.ndarray)) else [color] * len(xs)
    for a, b, c in zip(xs, ys, cols):
        if np.isfinite(a) and np.isfinite(b):
            out.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="{r}" fill="{c}" fill-opacity="{opacity}"/>')
    for i in label_idx:
        out.append(text(X(xs[i]) + 4, Y(ys[i]) - 3, labels[i], 9, weight="bold"))
    for v in np.linspace(xmin + dx, xmax - dx, 4):
        out.append(text(X(v), y0 + h - pad + 21, f"{v:.2g}", 9, "middle", fill="#555"))
    for v in np.linspace(ymin + dy, ymax - dy, 4):
        out.append(text(x0 + pad - 4, Y(v) + 3, f"{v:.2g}", 9, "end", fill="#555"))
    out.append(text(x0 + pad + (w - pad) / 2, y0 + h + 4, xlabel, 10, "middle"))
    out.append(text(x0 + 10, y0 + 8 + (h - pad) / 2, ylabel, 10, "middle", rotate=-90))
    if title:
        out.append(text(x0 + pad, y0 + 2, title, 11, weight="bold"))
    return "\n".join(out)


def legend(x0, y0, items, size=10, sw=10, col_w=None, ncol=1):
    out = []
    for i, (lab, col) in enumerate(items):
        cx = x0 + (i % ncol) * (col_w or 0)
        cy = y0 + (i // ncol) * (sw + 5)
        out.append(f'<rect x="{cx}" y="{cy}" width="{sw}" height="{sw}" fill="{col}"/>')
        out.append(text(cx + sw + 4, cy + sw - 1, lab, size))
    return "\n".join(out)


def render(svg_body, width, height, png_path, dpi=300, svg_path=None):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}"><rect width="{width}" height="{height}" fill="white"/>{svg_body}</svg>')
    if svg_path:
        open(svg_path, "w").write(svg)
    scale = dpi / 96.0
    with tempfile.TemporaryDirectory() as td:
        sp = os.path.join(td, "fig.svg")
        open(sp, "w").write(svg)
        js = os.path.join(td, "r.js")
        open(js, "w").write(f"""
const {{ chromium }} = require('/opt/node22/lib/node_modules/playwright');
(async () => {{
  const b = await chromium.launch({{executablePath: '/opt/pw-browsers/chromium'}});
  const p = await b.newPage({{deviceScaleFactor: {scale}, viewport: {{width: {int(width)}, height: {int(height)}}}}});
  await p.goto('file://{sp}');
  await p.screenshot({{path: '{png_path}', clip: {{x: 0, y: 0, width: {int(width)}, height: {int(height)}}}}});
  await b.close();
}})();""")
        subprocess.run(["node", js], check=True)
    im = Image.open(png_path)
    im.save(png_path, dpi=(dpi, dpi))
