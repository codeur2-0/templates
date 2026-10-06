"""Figures du résumé : ce qu'un tableau de ROUGE ne montre pas.

Quatre figures, choisies parce que chacune répond à une question qu'un chiffre moyen laisse ouverte :

* ``rouge_par_strategie`` — le ROUGE-1/2/L de chaque stratégie mesurée sur les mêmes lignes, avec le
  résumé vide à zéro : la comparaison des modèles et de leurs références sur un seul graphique ;
* ``couverture_par_type`` — la couverture des faits par type : un résumé qui couvre les symptômes et
  oublie les durées est visible immédiatement, alors qu'une moyenne de couverture le cache ;
* ``compression_couverture`` — le nuage longueur × couverture, où l'on voit si couvrir plus demande
  d'écrire plus : c'est l'arbitrage réel du produit, et il est mesuré ;
* ``couts_sortie`` — la part de résumés qui butent sur leur budget et la distribution des longueurs :
  un décodeur qui s'arrête au lieu de conclure se voit ici et nulle part ailleurs.

Chaque fonction écrit un PNG et n'échoue jamais le pipeline : une figure impossible (table vide) est
silencieusement ignorée, jamais remplacée par un graphique trompeur.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

#: Palette used by every figure of the project (identical from one run to the next).
STRATEGY_COLOURS: tuple[str, ...] = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#7f7f7f",
)


def plot_strategy_rouge(frame: pd.DataFrame, path: str | Path) -> Path | None:
    """Bar chart of ROUGE-1/2/L per strategy.

    Args:
        frame: Table produced by ``per_strategy_frame`` (``strategy``, ``rouge1_f``…).
        path: Destination PNG.

    Returns:
        The written path, or ``None`` when the table carries no data.
    """
    if frame.empty or "rouge1_f" not in frame.columns:
        logger.warning("Figure ignorée : aucune métrique de stratégie disponible")
        return None
    metrics = [name for name in ("rouge1_f", "rouge2_f", "rouge_l_f") if name in frame.columns]
    if not metrics:
        return None
    strategies = [str(value) for value in frame["strategy"]]
    positions = np.arange(len(strategies))
    width = 0.8 / len(metrics)
    figure, axis = plt.subplots(figsize=(7.5, 4.0))
    for index, metric in enumerate(metrics):
        values = [float(value) for value in frame[metric]]
        axis.bar(
            positions + index * width,
            values,
            width=width,
            label=metric.upper().replace("_F", ""),
            color=STRATEGY_COLOURS[index % len(STRATEGY_COLOURS)],
        )
    axis.set_xticks(positions + width * (len(metrics) - 1) / 2)
    axis.set_xticklabels(strategies, rotation=15, ha="right")
    axis.set_ylabel("F1 vs meilleure référence")
    axis.set_title("ROUGE par stratégie (mêmes lignes de test)")
    axis.set_ylim(0.0, max(0.1, float(frame[metrics].to_numpy().max()) * 1.25))
    axis.grid(axis="y", alpha=0.3)
    axis.legend(frameon=False)
    route = _save(figure, path)
    return route


def plot_coverage_by_type(frame: pd.DataFrame, path: str | Path) -> Path | None:
    """Bar chart of the salient-fact coverage, by fact type.

    Args:
        frame: Fidelity table produced by ``coverage_frame`` (``covered_<type>`` columns).
        path: Destination PNG.

    Returns:
        The written path, or ``None`` when no per-type coverage is available.
    """
    columns = [column for column in frame.columns if column.startswith("covered_")]
    if frame.empty or not columns:
        logger.warning("Figure ignorée : aucune couverture par type disponible")
        return None
    values = [float(frame[column].mean()) for column in columns]
    labels = [column.removeprefix("covered_") for column in columns]
    order = np.argsort(values)[::-1]
    figure, axis = plt.subplots(figsize=(7.5, 4.0))
    axis.bar(
        [labels[index] for index in order], [values[index] for index in order], color="#1f77b4"
    )
    axis.set_ylabel("Faits saillants couverts")
    axis.set_title("Couverture des faits par type")
    axis.set_ylim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.3)
    for position, index in enumerate(order):
        axis.text(position, values[index] + 0.02, f"{values[index]:.2f}", ha="center", fontsize=8)
    route = _save(figure, path)
    return route


def plot_compression_vs_coverage(frame: pd.DataFrame, path: str | Path) -> Path | None:
    """Scatter plot of summary length against fact coverage.

    Args:
        frame: Fidelity table with ``n_words`` and ``fact_coverage`` columns.
        path: Destination PNG.

    Returns:
        The written path, or ``None`` when the table is empty.
    """
    if frame.empty or "n_words" not in frame.columns or "fact_coverage" not in frame.columns:
        logger.warning("Figure ignorée : longueurs ou couverture absentes")
        return None
    figure, axis = plt.subplots(figsize=(7.5, 4.0))
    axis.scatter(
        [float(value) for value in frame["n_words"]],
        [float(value) for value in frame["fact_coverage"]],
        s=18,
        alpha=0.7,
        color="#2ca02c",
        label="documents évalués",
    )
    if len(frame) >= 2:
        words = frame["n_words"].to_numpy(dtype="float64")
        coverage = frame["fact_coverage"].to_numpy(dtype="float64")
        if float(words.std()) > 0.0 and float(coverage.std()) > 0.0:
            slope, intercept = np.polyfit(words, coverage, 1)
            grid = np.linspace(words.min(), words.max(), 50)
            axis.plot(
                grid,
                slope * grid + intercept,
                color="#d62728",
                linewidth=1.2,
                label="tendance linéaire",
            )
            axis.legend(frameon=False)
    axis.set_xlabel("Longueur du résumé (mots)")
    axis.set_ylabel("Couverture des faits saillants")
    axis.set_title("Ce que coûte la couverture")
    axis.grid(alpha=0.3)
    route = _save(figure, path)
    return route


def plot_output_costs(frame: pd.DataFrame, path: str | Path) -> Path | None:
    """Histogram of summary lengths, with the share of summaries that hit their budget.

    Args:
        frame: Prediction table (``n_tokens``, ``hit_max_length``).
        path: Destination PNG.

    Returns:
        The written path, or ``None`` when the table is empty.
    """
    if frame.empty or "n_tokens" not in frame.columns:
        logger.warning("Figure ignorée : aucune longueur de résumé disponible")
        return None
    lengths = [float(value) for value in frame["n_tokens"]]
    truncated = float(frame["hit_max_length"].mean()) if "hit_max_length" in frame.columns else 0.0
    figure, axis = plt.subplots(figsize=(7.5, 4.0))
    axis.hist(lengths, bins=min(20, max(5, len(set(lengths)))), color="#9467bd", alpha=0.85)
    axis.set_xlabel("Tokens du résumé produit")
    axis.set_ylabel("Documents")
    axis.set_title(f"Longueurs produites — {truncated:.0%} des résumés atteignent le budget")
    axis.grid(axis="y", alpha=0.3)
    route = _save(figure, path)
    return route


def plot_segment_metrics(frame: pd.DataFrame, path: str | Path) -> Path | None:
    """Grouped bars of ROUGE-1 per segment dimension.

    Args:
        frame: Segment table produced by ``segment_frame`` (``segment``, ``value``, ``rouge1_f``).
        path: Destination PNG.

    Returns:
        The written path, or ``None`` when the table is empty.
    """
    if frame.empty or "rouge1_f" not in frame.columns:
        logger.warning("Figure ignorée : aucune métrique de segment disponible")
        return None
    dimensions = list(dict.fromkeys(str(value) for value in frame["segment"]))
    figure, axes = plt.subplots(
        1, len(dimensions), figsize=(4.0 * len(dimensions), 3.8), squeeze=False
    )
    for axis, dimension in zip(axes[0], dimensions, strict=True):
        subset = frame.loc[frame["segment"] == dimension].sort_values("rouge1_f", ascending=False)
        axis.bar(
            [str(value) for value in subset["value"]],
            [float(value) for value in subset["rouge1_f"]],
            color="#ff7f0e",
        )
        axis.set_title(dimension)
        axis.set_ylim(0.0, max(0.1, float(subset["rouge1_f"].max()) * 1.25))
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.3)
    figure.suptitle("ROUGE-1 selon les segments du corpus")
    figure.tight_layout()
    route = _save(figure, path)
    return route


def plot_all(
    *,
    per_strategy: pd.DataFrame,
    fidelity: pd.DataFrame,
    predictions: pd.DataFrame,
    segments: pd.DataFrame,
    figures_dir: str | Path,
) -> Mapping[str, Path]:
    """Write every figure of an evaluation, tolerating the ones that cannot be drawn.

    Args:
        per_strategy: Strategy comparison table.
        fidelity: Per-document fidelity table.
        predictions: Prediction table.
        segments: Segment table.
        figures_dir: Destination directory.

    Returns:
        Mapping ``figure name -> written path`` (only the figures that were produced).
    """
    directory = Path(figures_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    jobs: tuple[tuple[str, object], ...] = (
        ("rouge_par_strategie", lambda target: plot_strategy_rouge(per_strategy, target)),
        ("couverture_par_type", lambda target: plot_coverage_by_type(fidelity, target)),
        ("compression_couverture", lambda target: plot_compression_vs_coverage(fidelity, target)),
        ("couts_sortie", lambda target: plot_output_costs(predictions, target)),
        ("segments", lambda target: plot_segment_metrics(segments, target)),
    )
    for name, job in jobs:
        route = job(directory / f"{name}.png")
        if route is not None:
            written[name] = route
    logger.info("{} figure(s) écrite(s) dans {}", len(written), directory)
    return written


def _save(figure: plt.Figure, path: str | Path) -> Path:
    """Save a figure as PNG and close it.

    Args:
        figure: Matplotlib figure.
        path: Destination path.

    Returns:
        The written path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=110)
    plt.close(figure)
    return target


def figure_names() -> Sequence[str]:
    """Names of the figures this module can produce (used by the report and the README).

    Returns:
        The figure names, in production order.
    """
    return (
        "rouge_par_strategie",
        "couverture_par_type",
        "compression_couverture",
        "couts_sortie",
        "segments",
    )


__all__ = [
    "STRATEGY_COLOURS",
    "figure_names",
    "plot_all",
    "plot_compression_vs_coverage",
    "plot_coverage_by_type",
    "plot_output_costs",
    "plot_segment_metrics",
    "plot_strategy_rouge",
]
