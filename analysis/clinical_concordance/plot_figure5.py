#!/usr/bin/env python3
"""Render Fig. 5 exclusively from figure5_source_data.csv."""
from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def method_colors(methods: list[str]) -> dict[str, Any]:
    # Colors do not encode results. Unknown methods receive a stable tab10 color.
    preferred = {
        "HANAMI": "#2C8C99",
        "TriMoGCL": "#E28B62",
        "TriNet": "#92B1B6",
        "N2V-MLP": "#CAD0BD",
        "RF": "#E1A6AD",
    }
    fallback = plt.get_cmap("tab10")
    return {
        method: preferred.get(method, fallback(index % 10))
        for index, method in enumerate(methods)
    }


def panel_title(value: str) -> str:
    return "\n".join(textwrap.wrap(str(value), width=34, break_long_words=False))


def draw_panel(ax, rows: pd.DataFrame, colors: dict[str, Any]) -> None:
    rows = rows.sort_values("bar_order")
    methods = rows["method"].astype(str).tolist()
    values = rows["rank_percentile"].to_numpy(float)
    x = np.arange(len(rows))
    bars = ax.bar(x, values, width=0.68, color=[colors[method] for method in methods])
    x_label_size = 7 if len(methods) > 2 else 8
    ax.set_xticks(x, methods, fontsize=x_label_size)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.set_ylabel("Rank percentile (%)", fontsize=8)
    ax.set_title(panel_title(rows.iloc[0]["panel_title"]), fontsize=9, fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    maximum = max(values) if len(values) else 1.0
    significance = str(rows.iloc[0].get("significance", ""))
    has_bracket = bool(significance and significance.lower() != "nan" and significance != "ns")
    upper = maximum * (1.35 if has_bracket else 1.20)
    if upper <= 0:
        upper = 1.0
    ax.set_ylim(0, upper)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + upper * 0.018,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )

    if has_bracket:
        baseline = str(rows.iloc[0]["comparison_method"])
        reference = str(rows.iloc[0]["reference_method"])
        if baseline not in methods or reference not in methods:
            raise ValueError(
                f"Significance bracket requests {baseline} vs {reference}, absent from panel"
            )
        left, right = methods.index(baseline), methods.index(reference)
        y = maximum * 1.16
        tick = upper * 0.025
        ax.plot([left, left, right, right], [y - tick, y, y, y - tick], color="black", lw=0.7)
        ax.text((left + right) / 2, y + upper * 0.012, significance, ha="center", fontsize=11)


def plot(config_path: Path, output_pdf: Path | None = None, output_png: Path | None = None) -> None:
    config = load_config(config_path)
    result_dir = resolve_path(config["paths"]["results_dir"])
    source_path = result_dir / "figure5_source_data.csv"
    if not source_path.is_file():
        raise FileNotFoundError(f"Run analyze_clinical_concordance.py first: {source_path}")
    source = pd.read_csv(source_path, low_memory=False)
    required = {
        "panel",
        "panel_type",
        "panel_title",
        "method",
        "rank_percentile",
        "bar_order",
        "comparison_method",
        "reference_method",
        "significance",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Figure source lacks columns: {sorted(missing)}")

    panel_order = source["panel"].astype(str).drop_duplicates().tolist()
    columns = int(config.get("figure", {}).get("columns", 4))
    rows = math.ceil(len(panel_order) / columns)
    figure_config = config.get("figure", {})
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            float(figure_config.get("width_inches", 13.5)),
            float(figure_config.get("height_inches", 6.6)),
        ),
        squeeze=False,
    )
    colors = method_colors(source.method.astype(str).drop_duplicates().tolist())
    for axis, panel in zip(axes.flat, panel_order):
        draw_panel(axis, source[source.panel.astype(str).eq(panel)], colors)
        axis.text(
            -0.16,
            1.12,
            panel,
            transform=axis.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )
    for axis in list(axes.flat)[len(panel_order) :]:
        axis.set_visible(False)
    figure.subplots_adjust(
        left=0.055, right=0.99, top=0.92, bottom=0.08, wspace=0.28, hspace=0.36
    )

    output_pdf = output_pdf or result_dir / "figure5.pdf"
    output_png = output_png or result_dir / "figure5.png"
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight")
    figure.savefig(
        output_png,
        dpi=int(figure_config.get("dpi", 300)),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    print(f"Wrote {output_pdf} and {output_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR / "config.yaml")
    parser.add_argument("--output-pdf", type=Path)
    parser.add_argument("--output-png", type=Path)
    args = parser.parse_args()
    plot(
        args.config.resolve(),
        args.output_pdf.resolve() if args.output_pdf else None,
        args.output_png.resolve() if args.output_png else None,
    )


if __name__ == "__main__":
    main()
