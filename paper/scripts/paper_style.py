"""
One place for the look of the figures in the paper.

Ticks point inwards and appear on all four sides, the fonts are sans serif to match the rest of the
figure text, and numbers written into labels by hand go through mathtext so that a minus sign is a
minus sign rather than a hyphen.
"""
from pathlib import Path

import matplotlib as mpl
import numpy as np


def use_paper_style():
    mpl.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "axes.grid": False, "axes.linewidth": 0.8,
        # Figures are saved without bbox="tight": the saved canvas is the text width, so a point in
        # the figure is a point on the page. Anything placed outside the canvas is clipped.
        "font.family": "sans-serif",
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": True,
    })


# The width of the text block of the JCAP class in inches. Every figure is included at \textwidth
# and drawn at this width, so a point in the figure is a point on the page.
WIDTH = 440.0 / 72.27

# The package name, set in monospace to match \texttt{binaryfog} in the text. mathtext's \mathtt
# picks DejaVu Sans Mono, which is the monospace companion of the sans face the figures use.
BF = r"$\mathtt{binaryfog}$"


def frequency_axis(axis, labels=(2, 5, 20)):
    """A log frequency axis in nHz: a major tick at each decade, minor ticks between them, and the
    minor ticks at the frequencies in labels labelled as well.
    """
    import matplotlib.ticker as mt
    axis.set_major_locator(mt.LogLocator(base=10, numticks=12))
    axis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_locator(mt.LogLocator(base=10, subs=np.arange(2, 10), numticks=12))
    keep = {float(v) for v in labels}
    axis.set_minor_formatter(mt.FuncFormatter(
        lambda v, _: f"{v:g}" if any(abs(v - k) < 1e-6 * k for k in keep) else ""))
    axis.set_tick_params(which="minor", labelsize=mpl.rcParams["xtick.labelsize"])


# The logo (black holes above the name) at 600 px width, and at 300 px for the small logos in a panel corner.
LOGO = Path(__file__).resolve().parents[1] / "inputs/logo/binaryfog_logo_600.png"
LOGO_SMALL = LOGO.with_name("binaryfog_logo_300.png")


def _png_size(path):
    """Width and height in pixels, read from the header of a PNG file."""
    with open(path, "rb") as fh:
        head = fh.read(24)
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


_W, _H = _png_size(LOGO)
LOGO_ASPECT = _W / _H           # width / height of the logo, the same for both files


def add_logo(fig, rect, alpha=1.0, path=None):
    """The binaryfog logo (black holes with the name below) in the figure-fraction
    rectangle rect = (left, bottom, width, height), aspect kept, on an axes of its own above everything.
    interpolation="none" embeds the image at its own resolution in a PDF instead of resampling it at
    the figure dpi, which makes it pixelated."""
    import matplotlib.image as mpimg
    ax = fig.add_axes(rect, zorder=10)
    ax.imshow(mpimg.imread(path or LOGO), alpha=alpha, interpolation="none")
    ax.set_axis_off()
    ax.set_anchor("C")
    ax.set_label("logo")
    return ax


def logo_in_corner(fig, ax, corner="lower right", size=0.32, margin=0.12, inset=0.03, white=245, uniform=None, path=None):
    """The logo in a corner of the panel ax, as large as possible up to size times the panel height,
    with a margin of margin times its own height kept free of anything drawn.
    The figure is rendered without the logo, and the logo is shrunk until the rectangle it needs,
    margin included, holds no pixel darker than white in any channel. With uniform set, the rectangle
    may instead hold any single colour, such as a filled region, as long as no channel varies by more
    than uniform across it (the logo has a transparent background)."""
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    H_px, W_px = img.shape[:2]
    inv = fig.transFigure.inverted()
    box = ax.get_window_extent()
    W, H = fig.get_size_inches()
    for k in range(40):
        h = size * (0.95 ** k) * box.height / (H * fig.dpi)            # figure fraction
        w = h * LOGO_ASPECT * H / W
        pad_y, pad_x = margin * h, margin * h * H / W
        ins_x, ins_y = inset * box.width / (W * fig.dpi), inset * box.height / (H * fig.dpi)
        b = box.transformed(inv)
        x0 = b.x1 - ins_x - w if "right" in corner else b.x0 + ins_x
        y0 = b.y0 + ins_y if "lower" in corner else b.y1 - ins_y - h
        # pixel rectangle of logo plus margin; rows counted from the top
        c0, c1 = int((x0 - pad_x) * W_px), int((x0 + w + pad_x) * W_px)
        r0, r1 = int((1 - (y0 + h + pad_y)) * H_px), int((1 - (y0 - pad_y)) * H_px)
        # keep the panel's own frame and its inward ticks out of the test
        tick = int(np.ceil(max(mpl.rcParams["xtick.major.size"], mpl.rcParams["ytick.major.size"])
                           * fig.dpi / 72)) + 2
        a0, a1 = int(b.x0 * W_px) + tick, int(b.x1 * W_px) - tick
        t0, t1 = int((1 - b.y1) * H_px) + tick, int((1 - b.y0) * H_px) - tick
        patch = img[max(r0, t0):min(r1, t1), max(c0, a0):min(c1, a1)]
        if uniform is not None:
            flat = patch.size and np.all(patch.max(axis=(0, 1)).astype(int) - patch.min(axis=(0, 1)) <= uniform)
            if flat:
                return add_logo(fig, (x0, y0, w, h), path=path)
        elif patch.size and patch.min() >= white:
            return add_logo(fig, (x0, y0, w, h), path=path)
    raise RuntimeError("no free corner for the logo")


def logo_below_legend(fig, leg, gap=0.01, scale=0.6):
    """The logo directly below a legend, centred under it, scale times the legend width (corner
    plot)."""
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    bb = leg.get_window_extent().transformed(inv)
    W, H = fig.get_size_inches()
    w = scale * bb.width
    h = w / LOGO_ASPECT * W / H
    return add_logo(fig, (bb.x0 + (bb.width - w) / 2, bb.y0 - gap - h, w, h))