# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Plot PipeLink vs public sm90 attention kernels from the committed result records.

uv run campaigns/attention/sm90-baseline/plot_placement.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
REPO = HERE.parents[2]

CELLS = (
    ("W1-32k", "Dense causal", "Qwen3-8B"),
    ("W2a-32k", "Sliding window", "Mistral-7B"),
    ("W2b-32k", "Sliding window", "Gemma3-27B"),
    ("W3a-32k", "Window + softcap", "Gemma2-27B"),
    ("W3b-8k", "Softcap (8K)", "Grok-1"),
    ("W4-32k", "Chunked", "Llama 4 Scout"),
    ("W5-32k", "Block-sparse", "Phi-3-small"),
    ("W6-prefix-32k", "MSA, prefix", "MiniMax-M3"),
    ("W6-uniform-32k", "MSA, uniform", "MiniMax-M3"),
)
NAMES = {
    "fa3": "FA3", "fa4": "FA4", "flashinfer-fa3": "FlashInfer", "flex-flash": "Flex (FA4)",
    "flex-triton": "Flex (Triton)", "vllm-m3-triton": "vLLM Triton", "phi3-small-triton": "Phi-3 Triton",
    "flash-msa": "flash-msa", "fa3-varlen-chunks": "FA3 varlen",
}
THEMES = {
    "light": {"fg": "#1f2328", "muted": "#59636e", "grid": "#d1d9e0", "ours": "#4f46e5",
              "best": "#8c959f", "flex": "#d1d9e0"},
    "dark": {"fg": "#e6edf3", "muted": "#9198a1", "grid": "#3d444d", "ours": "#818cf8",
             "best": "#6e7681", "flex": "#3d444d"},
}


def load() -> list[dict]:
    placement = {cell["id"]: cell for cell in json.loads((RESULTS / "sm90-placement.json").read_text())["cells"]}
    public = {cell["id"]: cell for cell in json.loads((RESULTS / "sm90-public-definitive.json").read_text())["cells"]}
    rows = []
    for cell_id, kind, model in CELLS:
        ok = [r for r in public[cell_id]["results"] if r.get("status") == "ok" and r.get("useful_tflops")]
        best = max((r for r in ok if r["impl"] != "flex-triton"), key=lambda r: r["useful_tflops"])
        flex = next(r for r in ok if r["impl"] == "flex-triton")
        fa4 = next((r for r in ok if r["impl"] == "fa4"), None)
        rows.append({"kind": kind, "model": model, "ours": placement[cell_id]["linked"]["useful_tflops"],
                     "best": best["useful_tflops"], "best_name": NAMES.get(best["impl"], best["impl"]),
                     "flex": flex["useful_tflops"],
                     # MSA: FA4's general block-sparse path is the other exact implementation.
                     "fa4": fa4["useful_tflops"] if kind.startswith("MSA") and fa4 and fa4 is not best else None})
    return rows


def plot(rows: list[dict], theme: str, output: Path) -> None:
    colors = THEMES[theme]
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11, "svg.fonttype": "none",
                         "text.color": colors["fg"], "axes.labelcolor": colors["muted"],
                         "xtick.color": colors["fg"], "ytick.color": colors["muted"]})
    fig, ax = plt.subplots(figsize=(12, 4.6))
    width, xs = 0.27, range(len(rows))
    bars = (("ours", "PipeLink", -width), ("best", "Fastest public kernel", 0.0), ("flex", "FlexAttention (Triton)", width))
    for key, label, offset in bars:
        ax.bar([x + offset for x in xs], [row[key] for row in rows], width * 0.92,
               color=colors[key], label=label, zorder=3)
    for x, row in zip(xs, rows):
        ax.text(x - width, row["ours"] + 8, f"{row['ours']:.0f}", ha="center", va="bottom",
                fontsize=9, color=colors["ours"], fontweight="bold")
        ax.text(x, row["best"] + 8, row["best_name"], ha="center", va="bottom", fontsize=8,
                color=colors["muted"], rotation=90)
        if row["fa4"] is not None:
            ax.text(x + width, row["flex"] + 8, f"FA4 {row['fa4']:.0f}", ha="center", va="bottom",
                    fontsize=8, color=colors["muted"], rotation=90)
    ax.set_xticks(list(xs), [f"{row['kind']}\n{row['model']}" for row in rows], fontsize=9.5)
    ax.set_ylabel("useful TFLOP/s")
    ax.set_ylim(0, 700)
    ax.grid(axis="y", color=colors["grid"], linewidth=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(colors["grid"])
    ax.tick_params(axis="both", length=0)
    ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=10)
    ax.set_title("Hopper attention: one PipeLink core package, six linked variants", loc="left",
                 fontsize=13, fontweight="bold", color=colors["fg"])
    fig.text(0.0, -0.02, "H200, bf16, head dim 128, 32K tokens unless noted, SM clock locked at 1300 MHz. "
             "Useful FLOPs count only visible query-key pairs.", fontsize=8.5, color=colors["muted"])
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=REPO / "docs" / "assets")
    args = parser.parse_args()
    rows = load()
    for theme in THEMES:
        plot(rows, theme, args.out_dir / f"sm90-attention-{theme}.svg")
    for row in rows:
        print(f"{row['kind']:<18} ours {row['ours']:6.1f}  best {row['best']:6.1f} ({row['best_name']})  flex {row['flex']:6.1f}")


if __name__ == "__main__":
    main()
