"""
Unified plotting style for Octopus NDSS 2027 submission.

Usage:
    from plot_style import apply_style, COLORS, save_fig, FIG_SINGLE, FIG_DOUBLE

All figures will use Times New Roman, Type 42 fonts, consistent colors,
and appropriate sizes for NDSS double-column format.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# ─── Register Times New Roman fonts from project directory ───────────────────
_FONT_DIR = Path(__file__).parent / "fonts"
_TNR_REGISTERED = False

def _register_tnr():
    global _TNR_REGISTERED
    if _TNR_REGISTERED:
        return
    for ttf in _FONT_DIR.glob("*.ttf"):
        try:
            fm.fontManager.addfont(str(ttf))
        except Exception:
            pass
    _TNR_REGISTERED = True

# ─── NDSS column dimensions (inches) ─────────────────────────────────────────
FIG_SINGLE = (3.5, 2.6)   # Single column width
FIG_DOUBLE = (7.0, 3.0)   # Double column width
FIG_SQUARE = (3.5, 3.2)   # Square (radar, heatmap)

# ─── Color palette (Nature-style, colorblind-safe, consistent across all figures) ──
COLORS = {
    "octopus":  "#E64B35",  # Nature Red - our method (stands out)
    "cusum":    "#4DBBD5",  # Cyan - baseline
    "exp3":     "#00A087",  # Teal - baseline
    "ucb":      "#3C5488",  # Navy - baseline
    "random":   "#B09C85",  # Warm gray - baseline
    "ablation": "#F39B7F",  # Salmon - ablation variants
    "ci_fill":  "#E64B35",  # CI shading (alpha applied separately)
}

# Ordered list for consistent legend ordering
COLOR_ORDER = ["octopus", "cusum", "exp3", "ucb", "random"]

# ─── Line styles for B&W friendliness ─────────────────────────────────────────
LINESTYLES = {
    "octopus": "-",
    "cusum":   "--",
    "exp3":    "-.",
    "ucb":     ":",
    "random":  (0, (3, 1, 1, 1)),
}

MARKERS = {
    "octopus": "o",
    "cusum":   "s",
    "exp3":    "^",
    "ucb":     "D",
    "random":  "v",
}


def apply_style():
    """Apply unified matplotlib style. Call once at script start."""
    _register_tnr()
    plt.rcParams.update({
        # Fonts
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        # Type 42 fonts (no Type 3)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Figure
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Axes
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        # Legend
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "legend.borderpad": 0.3,
        "legend.handlelength": 1.5,
        # Grid (when enabled)
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })


def save_fig(fig, name: str, output_dir: str = None, formats=("pdf",)):
    """Save figure with consistent settings.

    Args:
        fig: matplotlib figure
        name: filename without extension
        output_dir: directory path (default: ./figures/)
        formats: tuple of formats to save (default: pdf only)
    """
    if output_dir is None:
        output_dir = Path("figures")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt)
    plt.close(fig)
    return output_dir / f"{name}.{formats[0]}"


def get_color(method: str) -> str:
    """Get color for a method name (case-insensitive fuzzy match)."""
    key = method.lower().replace("-", "").replace("_", "").replace(" ", "")
    for k, v in COLORS.items():
        if k in key or key in k:
            return v
    return "#333333"


def get_linestyle(method: str) -> str:
    """Get linestyle for a method name."""
    key = method.lower().replace("-", "").replace("_", "").replace(" ", "")
    for k, v in LINESTYLES.items():
        if k in key or key in k:
            return v
    return "-"


def method_style(method: str) -> dict:
    """Get full style dict for plt.plot() kwargs."""
    return {
        "color": get_color(method),
        "linestyle": get_linestyle(method),
        "marker": MARKERS.get(method.lower(), None),
        "markevery": 50,
    }
