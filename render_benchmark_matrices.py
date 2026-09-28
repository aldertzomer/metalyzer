"""Render the benchmark source confusion matrices as SVG tables.

Run from the repository root after updating benchmark_classified.tsv or
benchmark_true_labels.tsv:

    python render_benchmark_matrices.py
"""

from __future__ import annotations

import csv
import argparse
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRUE_LABELS = ROOT / "benchmark_true_labels.tsv"
ASSET_DIR = ROOT / "assets"

LABELS = [
    "cat",
    "cattle",
    "chicken",
    "dog",
    "environment",
    "goat",
    "human",
    "laboratory",
    "other_animal",
    "pig",
    "sheep",
    "turkey",
    "unknown",
    "wastewater",
    "water",
    "waterbird",
    "wildbird",
]

SHORT_LABELS = {
    "chicken": "Chk",
    "environment": "Env",
    "laboratory": "Lab",
    "other_animal": "Other",
    "turkey": "Turk",
    "unknown": "Unk",
    "wastewater": "WW",
    "waterbird": "Wbird",
    "wildbird": "Wildbird",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def load_matrix(classified: Path) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    true_rows = read_tsv(TRUE_LABELS)
    classified_rows = read_tsv(classified)
    true_by_id = {row["run_accession"]: row["final_species"] for row in true_rows}
    predicted_by_id = {row["run_accession"]: row["best_hit"] for row in classified_rows}

    if set(true_by_id) != set(predicted_by_id):
        raise ValueError("Benchmark and classified files do not contain the same accessions.")

    observed = set(true_by_id.values()) | set(predicted_by_id.values())
    unknown_labels = observed - set(LABELS)
    if unknown_labels:
        raise ValueError(f"Labels missing from LABELS: {sorted(unknown_labels)}")

    row_totals = {label: 0 for label in LABELS}
    matrix = {true_label: {predicted_label: 0 for predicted_label in LABELS} for true_label in LABELS}
    for accession, true_label in true_by_id.items():
        predicted_label = predicted_by_id[accession]
        row_totals[true_label] += 1
        matrix[true_label][predicted_label] += 1
    return row_totals, matrix


def colour(value: int, total: int) -> tuple[str, str]:
    """Return a blue heatmap fill and a contrasting text colour."""
    fraction = value / total if total else 0
    red = round(247 - 186 * fraction)
    green = round(251 - 173 * fraction)
    blue = round(255 - 83 * fraction)
    text = "#ffffff" if fraction >= 0.55 else "#172033"
    return f"rgb({red}, {green}, {blue})", text


def render_svg(
    path: Path,
    title: str,
    row_totals: dict[str, int],
    matrix: dict[str, dict[str, int]],
    percentages: bool,
) -> None:
    left_width = 148
    total_width = 42
    cell_width = 62
    cell_height = 32
    matrix_x = left_width + total_width
    header_height = 128
    footer_height = 52
    width = matrix_x + len(LABELS) * cell_width + 18
    height = header_height + len(LABELS) * cell_height + footer_height
    cells: list[str] = []

    for column, label in enumerate(LABELS):
        x = matrix_x + column * cell_width + cell_width / 2
        displayed = SHORT_LABELS.get(label, label)
        cells.append(
            f'<text x="{x}" y="{header_height - 12}" transform="rotate(-48 {x} {header_height - 12})" '
            f'class="header">{escape(displayed)}</text>'
        )

    for row, true_label in enumerate(LABELS):
        y = header_height + row * cell_height
        total = row_totals[true_label]
        cells.append(f'<rect x="0" y="{y}" width="{left_width}" height="{cell_height}" class="row-label"/>')
        cells.append(f'<rect x="{left_width}" y="{y}" width="{total_width}" height="{cell_height}" class="total"/>')
        cells.append(f'<text x="8" y="{y + 21}" class="row-name">{escape(true_label)}</text>')
        cells.append(f'<text x="{left_width + total_width / 2}" y="{y + 21}" class="cell">{total}</text>')

        for column, predicted_label in enumerate(LABELS):
            value = matrix[true_label][predicted_label]
            fill, text_colour = colour(value, total)
            x = matrix_x + column * cell_width
            displayed = f"{value / total * 100:.1f}%" if percentages else str(value)
            cells.append(
                f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" fill="{fill}" class="matrix-cell"/>'
            )
            cells.append(
                f'<text x="{x + cell_width / 2}" y="{y + 21}" fill="{text_colour}" class="cell">{displayed}</text>'
            )

    legend = (
        "Columns: Chk=chicken · Env=environment · Lab=laboratory · Other=other_animal · "
        "Turk=turkey · Unk=unknown · WW=wastewater · Wbird=waterbird · Wildbird=wildbird"
    )
    percentage_note = "Values are row percentages; each true-source row sums to 100% before rounding."
    suffix = percentage_note if percentages else "Values are absolute counts."
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">
  <title id="title">{escape(title)}</title>
  <desc id="description">Rows are true sources; columns are predicted sources. {escape(suffix)}</desc>
  <style>
    .title {{ font: 700 21px Arial, sans-serif; fill: #172033; }}
    .subtitle, .legend {{ font: 12px Arial, sans-serif; fill: #4b5563; }}
    .header {{ font: 600 11px Arial, sans-serif; fill: #172033; text-anchor: end; }}
    .row-name {{ font: 600 12px Arial, sans-serif; fill: #172033; }}
    .cell {{ font: 11px Arial, sans-serif; text-anchor: middle; }}
    .row-label {{ fill: #f8fafc; stroke: #cbd5e1; }}
    .total {{ fill: #eef2f7; stroke: #cbd5e1; }}
    .matrix-cell {{ stroke: #cbd5e1; }}
  </style>
  <rect width="100%" height="100%" fill="white"/>
  <text x="0" y="27" class="title">{escape(title)}</text>
  <text x="0" y="48" class="subtitle">Rows: true source · Columns: predicted source · n: true records</text>
  <text x="8" y="{header_height - 8}" class="header" transform="rotate(-48 8 {header_height - 8})">True source</text>
  <text x="{left_width + total_width / 2}" y="{header_height - 8}" class="header" transform="rotate(-48 {left_width + total_width / 2} {header_height - 8})">n</text>
  {''.join(cells)}
  <text x="0" y="{height - 29}" class="legend">{escape(legend)}</text>
  <text x="0" y="{height - 10}" class="legend">{escape(suffix)}</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classified", type=Path, default=ROOT / "benchmark_classified.tsv")
    parser.add_argument("--prefix", default="benchmark", help="Output filename prefix under assets/")
    parser.add_argument("--title", default="Benchmark", help="Title prefix in the SVG")
    args = parser.parse_args()
    classified = args.classified if args.classified.is_absolute() else ROOT / args.classified
    row_totals, matrix = load_matrix(classified)
    ASSET_DIR.mkdir(exist_ok=True)
    render_svg(
        ASSET_DIR / f"{args.prefix}-confusion-absolute.svg",
        f"{args.title} source confusion matrix — absolute counts",
        row_totals,
        matrix,
        percentages=False,
    )
    render_svg(
        ASSET_DIR / f"{args.prefix}-confusion-percent.svg",
        f"{args.title} source confusion matrix — row percentages",
        row_totals,
        matrix,
        percentages=True,
    )


if __name__ == "__main__":
    main()
