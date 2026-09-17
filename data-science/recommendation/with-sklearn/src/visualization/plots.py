"""Figures used by the evaluation report and the notebooks of a **ranking** project.

Every function returns the path of the written PNG: figures are artefacts, not side effects. The
rendering backend is forced to ``Agg`` so the code works headless (CI, containers, SSH sessions).

The figure set answers the questions a recommender reviewer actually asks, in order:

1. is the published ranking any good, and against what? (``metrics_bar``, ``ceiling_position``)
2. does the model beat the free popularity engine? (``baselines_comparison``)
3. how many slots should be published? (``quality_vs_cutoff``)
4. does the engine destroy the catalogue? (``coverage_vs_cutoff``, ``catalog_concentration``)
5. is it fair to new users and to unavailable items? (``segment_breakdown``, ``stock_and_margin``)
6. is the quality stable over time? (``backtest_stability``)
7. where exactly does it misrank? (``worst_misrankings``)
8. what does it actually lean on? (``feature_importance``)

Figures never raise: a missing table returns ``None`` and :meth:`RankingPlots.save_all` skips it,
so a partial evaluation still produces a readable report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logging import get_logger

logger = get_logger(__name__)

sns.set_theme(style="whitegrid", context="notebook")

DEFAULT_DPI = 140

#: Number of worst misrankings displayed by :meth:`RankingPlots.worst_misrankings`.
TOP_ERRORS = 12

#: Number of features displayed by :meth:`RankingPlots.feature_importance`.
TOP_FEATURES = 15

#: Constant colour coding, so a reader never has to re-learn which bar is which scorer.
COLOR_MODEL = "#1f4e79"
COLOR_POPULARITY = "#7f7f7f"
COLOR_RANDOM = "#bfbfbf"
COLOR_INTENT = "#9bbb59"
COLOR_CEILING = "#c0504d"

#: French labels of the three error families, kept identical across figures and reports.
LABEL_FORGET = "oubli"
LABEL_FALSE_PROMISE = "fausse_promesse"


def _save(fig: Any, path: str | Path, *, close: bool = True) -> Path:
    """Write a figure to disk.

    Args:
        fig: Matplotlib figure.
        path: Destination file.
        close: Whether to close the figure afterwards (keeps memory flat in notebooks).

    Returns:
        The written path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=DEFAULT_DPI, bbox_inches="tight")
    if close:
        plt.close(fig)
    return destination


def _safe_float(value: Any, default: float = float("nan")) -> float:
    """Coerce ``value`` to ``float`` without raising.

    Args:
        value: Value to coerce (may be ``None``, a NumPy scalar, ...).
        default: Value returned when the coercion fails.

    Returns:
        The coerced float.
    """
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return default if np.isnan(converted) else converted


def _extras(result: Any) -> dict[str, Any]:
    """Return the ``extras`` mapping of a result, or an empty mapping.

    Args:
        result: Evaluation result.

    Returns:
        The extras mapping.
    """
    extras = getattr(result, "extras", None)
    return dict(extras) if isinstance(extras, dict) else {}


def _top_k(result: Any) -> int:
    """Return the published cutoff used to label the figures.

    Args:
        result: Evaluation result.

    Returns:
        The cutoff.
    """
    return int(_extras(result).get("top_k", 10))


def _metric(result: Any, name: str, default: float = float("nan")) -> float:
    """Read one metric from a result without raising.

    Args:
        result: Evaluation result.
        name: Metric name.
        default: Value returned when the metric is absent.

    Returns:
        The metric value.
    """
    metrics = getattr(result, "metrics", None)
    if not isinstance(metrics, dict):
        return default
    return _safe_float(metrics.get(name), default)


class RankingPlots:
    """Figure factory for a ranking (recommendation) evaluation."""

    def __init__(self, figures_dir: str | Path, *, palette: str = "viridis") -> None:
        """Store the output directory.

        Args:
            figures_dir: Directory receiving the PNG files.
            palette: Seaborn/matplotlib palette for density and heat maps.
        """
        self.figures_dir = Path(figures_dir)
        self.palette = palette
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ quality ---------
    def metrics_bar(self, result: Any, *, name: str = "metrics_bar.png") -> Path | None:
        """Draw the five ranking metrics of the published cutoff against their thresholds.

        Les cinq métriques ne racontent pas la même chose et ne doivent pas être lues séparément :
        un rappel élevé avec une précision faible décrit un moteur généreux mais bruyant,
        l'inverse décrit un moteur prudent qui abandonne des clients. Les afficher ensemble, avec
        le seuil de chaque objectif en pointillé, permet de voir d'un coup lequel des deux
        déséquilibres on a.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when no metric is available.
        """
        cutoff = _top_k(result)
        wanted = (
            (f"ndcg_at_{cutoff}", f"NDCG@{cutoff}", 0.45),
            (f"precision_at_{cutoff}", f"Precision@{cutoff}", 0.22),
            (f"recall_at_{cutoff}", f"Recall@{cutoff}", 0.55),
            (f"map_at_{cutoff}", f"MAP@{cutoff}", None),
            (f"hit_rate_at_{cutoff}", f"Hit-rate@{cutoff}", None),
        )
        labels, values, thresholds = [], [], []
        for key, label, threshold in wanted:
            value = _metric(result, key)
            if np.isnan(value):
                continue
            labels.append(label)
            values.append(value)
            thresholds.append(threshold)
        if not labels:
            logger.warning("metrics_bar ignoré : aucune métrique de classement disponible")
            return None

        fig, axis = plt.subplots(figsize=(9.0, 4.6))
        colours = [
            COLOR_MODEL if threshold is None or value >= threshold else COLOR_CEILING
            for value, threshold in zip(values, thresholds, strict=False)
        ]
        bars = axis.bar(labels, values, color=colours, edgecolor="white")
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        for index, threshold in enumerate(thresholds):
            if threshold is None:
                continue
            axis.hlines(
                threshold,
                index - 0.38,
                index + 0.38,
                colors="black",
                linestyles="--",
                linewidth=1.4,
            )
            axis.text(
                index + 0.40,
                threshold,
                f"seuil {threshold:.2f}",
                va="center",
                fontsize=8,
                color="black",
            )
        axis.set_ylim(0, max(1.02, max(values) * 1.18))
        axis.set_ylabel("Valeur")
        axis.set_title(f"Qualité du classement publié (top-{cutoff})")
        axis.tick_params(axis="x", labelsize=9)
        fig.tight_layout()
        return _save(fig, self.figures_dir / name)

    def ceiling_position(self, result: Any, *, name: str = "ceiling_position.png") -> Path | None:
        """Place the model on the scale running from random guessing to the reachable ceiling.

        C'est la figure la plus honnête du jeu : un NDCG de 0,58 ne veut rien dire isolément, mais
        « 58 % du chemin entre le tri par popularité et le meilleur classement connaissable » se
        comprend immédiatement. Le plafond est publié par le générateur de données, qui connaît la
        probabilité de pertinence avant bruit ; la distance qui reste est du bruit irréductible,
        pas une marge de progression.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when no reference is published.
        """
        extras = _extras(result)
        reference = extras.get("generation_reference") or {}
        cutoff = _top_k(result)
        # Les références sont publiées par le générateur sous ces trois clés, au niveau racine des
        # métadonnées de génération, pour la coupure `top_k_reference` (celle du produit).
        ceiling = _safe_float(reference.get("ndcg_ceiling_oracle"))
        popularity = _safe_float(reference.get("ndcg_baseline_popularity"))
        random_value = _safe_float(reference.get("ndcg_baseline_random"))
        model = _metric(result, f"ndcg_at_{cutoff}")
        if np.isnan(ceiling) or np.isnan(model):
            logger.warning("ceiling_position ignoré : plafond ou NDCG modèle indisponible")
            return None
        reference_k = _safe_float(reference.get("top_k_reference"))
        if not np.isnan(reference_k) and int(reference_k) != cutoff:
            logger.warning(
                "Les références publiées portent sur un top-{} alors que l'évaluation publie un "
                "top-{} : la comparaison reste indicative",
                int(reference_k),
                cutoff,
            )

        fig, axis = plt.subplots(figsize=(9.6, 3.2))
        axis.hlines(0.5, 0, ceiling * 1.12, colors="#d9d9d9", linewidth=6, zorder=1)
        markers = [
            ("aléatoire", random_value, COLOR_RANDOM),
            ("popularité", popularity, COLOR_POPULARITY),
            ("modèle", model, COLOR_MODEL),
            ("plafond oracle", ceiling, COLOR_CEILING),
        ]
        for index, (label, value, colour) in enumerate(markers):
            if np.isnan(value):
                continue
            axis.scatter([value], [0.5], s=190, color=colour, zorder=3, edgecolor="white")
            offset = 0.30 if index % 2 == 0 else 0.70
            axis.annotate(
                f"{label}\n{value:.3f}",
                xy=(value, 0.5),
                xytext=(value, offset),
                ha="center",
                fontsize=9,
                arrowprops={"arrowstyle": "-", "color": colour, "linewidth": 1.1},
                color=colour,
                fontweight="bold" if label == "modèle" else "normal",
            )
        share = _safe_float(extras.get("ceiling_share"))
        title = f"Position du modèle sur l'échelle de qualité (NDCG@{cutoff})"
        if not np.isnan(share):
            title += f" — {share * 100:.0f} % du plafond atteignable"
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.set_xlabel(f"NDCG@{cutoff}")
        axis.set_xlim(0, ceiling * 1.12)
        fig.tight_layout()
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ baselines -------
    def baselines_comparison(
        self, result: Any, *, name: str = "baselines_comparison.png"
    ) -> Path | None:
        """Compare the model with the three references it must beat.

        La comparaison à la popularité est celle qui décide : trier par audience est gratuit,
        explicable et robuste, donc un modèle qui ne fait pas mieux ne vaut pas son coût de
        maintenance. L'intention récente mesure l'exploitation pure — recommander ce que
        l'utilisateur a déjà consulté — et le tirage aléatoire borne la difficulté du jeu de
        candidats.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the baseline table is missing.
        """
        table = getattr(result, "baselines", None)
        if not isinstance(table, pd.DataFrame) or table.empty:
            logger.warning("baselines_comparison ignoré : aucune table de références")
            return None
        metric_columns = [column for column in ("ndcg", "precision", "recall") if column in table]
        if not metric_columns:
            return None

        colours = {
            "modele": COLOR_MODEL,
            "popularite": COLOR_POPULARITY,
            "aleatoire": COLOR_RANDOM,
            "intention_recente": COLOR_INTENT,
        }
        fig, axis = plt.subplots(figsize=(9.8, 4.8))
        positions = np.arange(len(metric_columns))
        width = 0.8 / max(len(table), 1)
        for index, row in enumerate(table.itertuples()):
            values = [_safe_float(getattr(row, column)) for column in metric_columns]
            offset = (index - (len(table) - 1) / 2) * width
            bars = axis.bar(
                positions + offset,
                values,
                width=width,
                label=str(row.scorer),
                color=colours.get(str(row.scorer)),
                edgecolor="white",
            )
            for bar, value in zip(bars, values, strict=True):
                if np.isnan(value):
                    continue
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.008,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.6,
                )
        gain = _safe_float(getattr(table.iloc[-1], "gain_relatif_pct", np.nan))
        title = f"Modèle contre références (top-{_top_k(result)})"
        if not np.isnan(gain):
            title += f" — gain NDCG sur la popularité : {gain:+.1f} %"
        axis.set_title(title)
        axis.set_xticks(positions)
        axis.set_xticklabels([column.upper() for column in metric_columns])
        axis.set_ylabel("Valeur")
        axis.set_ylim(0, 1.05)
        axis.legend(fontsize=8, ncol=2, loc="upper right")
        fig.tight_layout()
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ cutoff ----------
    def quality_vs_cutoff(self, result: Any, *, name: str = "quality_vs_cutoff.png") -> Path | None:
        """Trace precision, recall and NDCG as a function of the number of published slots.

        Cette courbe est l'outil de décision du K : la précision chute mécaniquement quand on
        publie plus, le rappel monte, et le NDCG — qui tient compte de la position — passe
        généralement par un optimum. Le K retenu par la configuration est marqué d'une ligne
        verticale, ce qui rend visible le coût d'un K trop grand ou trop petit.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the curves are missing.
        """
        curves = getattr(result, "curves", None)
        if not isinstance(curves, dict) or "cutoff" not in curves:
            logger.warning("quality_vs_cutoff ignoré : courbes indisponibles")
            return None
        cutoffs = list(curves["cutoff"])
        series = [
            ("precision", "Précision", COLOR_MODEL),
            ("recall", "Rappel", COLOR_INTENT),
            ("ndcg", "NDCG", COLOR_CEILING),
        ]
        available = [(key, label, colour) for key, label, colour in series if key in curves]
        if not available:
            return None

        fig, axis = plt.subplots(figsize=(9.4, 4.6))
        for key, label, colour in available:
            axis.plot(cutoffs, curves[key], marker="o", markersize=4, label=label, color=colour)
        published = _top_k(result)
        if published in cutoffs:
            axis.axvline(published, color="black", linestyle="--", linewidth=1.2)
            axis.text(
                published + 0.2,
                axis.get_ylim()[1] * 0.94,
                f"K publié = {published}",
                fontsize=8.5,
                color="black",
            )
        axis.set_xlabel("Nombre d'emplacements publiés (K)")
        axis.set_ylabel("Valeur")
        axis.set_title("Qualité du classement en fonction de la coupure K")
        axis.set_ylim(0, 1.02)
        axis.legend(fontsize=9)
        fig.tight_layout()
        return _save(fig, self.figures_dir / name)

    def coverage_vs_cutoff(
        self, result: Any, *, name: str = "coverage_vs_cutoff.png"
    ) -> Path | None:
        """Trace catalogue coverage and concentration as a function of the published cutoff.

        Publier davantage d'articles diversifie mécaniquement l'exposition, mais au prix de la
        précision. Cette figure montre le taux de change : combien de couverture catalogue on
        achète par emplacement supplémentaire, et si l'on est déjà dans la zone où la
        concentration (Herfindahl) ne baisse plus.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the curves are missing.
        """
        curves = getattr(result, "curves", None)
        if not isinstance(curves, dict) or "cutoff" not in curves:
            return None
        cutoffs = list(curves["cutoff"])
        if "coverage" not in curves or "herfindahl" not in curves:
            return None

        fig, left = plt.subplots(figsize=(9.4, 4.6))
        left.plot(cutoffs, curves["coverage"], marker="o", markersize=4, color=COLOR_MODEL)
        left.set_xlabel("Nombre d'emplacements publiés (K)")
        left.set_ylabel("Couverture catalogue", color=COLOR_MODEL)
        left.tick_params(axis="y", labelcolor=COLOR_MODEL)
        left.set_ylim(0, 1.02)
        coverage_min = _safe_float(_extras(result).get("coverage_min", np.nan))
        if not np.isnan(coverage_min):
            left.axhline(coverage_min, color=COLOR_MODEL, linestyle=":", linewidth=1.2)

        right = left.twinx()
        right.plot(cutoffs, curves["herfindahl"], marker="s", markersize=4, color=COLOR_CEILING)
        right.set_ylabel("Concentration (Herfindahl)", color=COLOR_CEILING)
        right.tick_params(axis="y", labelcolor=COLOR_CEILING)
        right.grid(False)

        published = _top_k(result)
        if published in cutoffs:
            left.axvline(published, color="black", linestyle="--", linewidth=1.2)
        left.set_title("Diversité du catalogue en fonction de la coupure K")
        fig.tight_layout()
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ segments --------
    def segment_breakdown(self, result: Any, *, name: str = "segment_breakdown.png") -> Path | None:
        """Draw the NDCG of every business segment next to the global average.

        La moyenne globale est un résumé trompeur : elle peut masquer un moteur excellent pour les
        clients fidèles et médiocre pour les nouveaux. Chaque axe disponible (activité de
        l'utilisateur, catégorie, disponibilité, promotion, bande de marge) est dessiné en
        barres, avec la moyenne globale en pointillé et un marquage rouge des segments sous 90 %
        de cette moyenne.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the segment table is missing.
        """
        table = getattr(result, "per_segment", None)
        if not isinstance(table, pd.DataFrame) or table.empty or "ndcg" not in table:
            logger.warning("segment_breakdown ignoré : aucune table de segments")
            return None

        axes = [axis for axis in table["axe"].unique() if axis]
        if not axes:
            return None
        fig, gridspec = plt.subplots(len(axes), 1, figsize=(9.6, 2.5 * len(axes)), squeeze=False)
        global_mean = float(table["ndcg"].mean())
        for row_index, axis_name in enumerate(axes):
            axis = gridspec[row_index][0]
            subset = table[table["axe"] == axis_name].copy()
            subset = subset.sort_values("ndcg", ascending=True)
            colours = [
                COLOR_CEILING if _safe_float(value, 1.0) < 0.90 else COLOR_MODEL
                for value in subset.get("ratio_ndcg_global", pd.Series([1.0] * len(subset)))
            ]
            bars = axis.barh(subset["segment"].astype(str), subset["ndcg"], color=colours)
            axis.axvline(global_mean, color="black", linestyle="--", linewidth=1.1)
            for bar, value in zip(bars, subset["ndcg"], strict=True):
                axis.text(
                    value + 0.006,
                    bar.get_y() + bar.get_height() / 2,
                    f"{_safe_float(value):.3f}",
                    va="center",
                    fontsize=8,
                )
            axis.set_xlim(0, max(1.02, float(subset["ndcg"].max()) * 1.16))
            axis.set_title(axis_name.replace("_", " ").capitalize(), fontsize=10, loc="left")
            axis.tick_params(labelsize=8.5)
        fig.suptitle(
            f"NDCG@{_top_k(result)} par segment (pointillé = moyenne globale {global_mean:.3f})",
            fontsize=11.5,
            y=1.005,
        )
        fig.tight_layout()
        return _save(fig, self.figures_dir / name)

    def stock_and_margin(self, result: Any, *, name: str = "stock_and_margin.png") -> Path | None:
        """Audit what is actually published: availability of the slots and their margin profile.

        Deux gaspillages distincts se lisent ici. Publier un article en rupture coûte un
        emplacement rare et une déception client ; publier uniquement des articles à forte marge
        dégrade l'expérience. La figure croise la part d'emplacements perdus en rupture, la
        répartition des marges publiées et l'arbitrage chiffré entre classement par pertinence et
        classement pondéré par la marge.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when no published recommendation is available.
        """
        published = getattr(result, "predictions", None)
        if not isinstance(published, pd.DataFrame) or published.empty:
            logger.warning("stock_and_margin ignoré : aucune recommandation publiée")
            return None
        extras = _extras(result)
        trade_off = extras.get("margin_trade_off") or {}

        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))

        if "en_stock" in published.columns:
            counts = published["en_stock"].astype(bool).value_counts()
            labels = ["en stock" if bool(key) else "rupture" for key in counts.index]
            axes[0].bar(
                labels,
                counts.to_numpy(dtype="float64"),
                color=[COLOR_MODEL, COLOR_CEILING],
                edgecolor="white",
            )
            share = _safe_float(extras.get("out_of_stock_published_share"))
            title = "Disponibilité des emplacements publiés"
            if not np.isnan(share):
                title += f"\n{share * 100:.1f} % perdus en rupture"
            axes[0].set_title(title, fontsize=10)
            axes[0].set_ylabel("Emplacements")
            for index, value in enumerate(counts.to_numpy(dtype="float64")):
                axes[0].text(index, value, f"{int(value)}", ha="center", va="bottom", fontsize=8.5)
        else:
            axes[0].axis("off")

        if "marge_pct" in published.columns:
            margins = pd.to_numeric(published["marge_pct"], errors="coerce").dropna()
            if not margins.empty:
                sns.histplot(margins, bins=22, kde=True, ax=axes[1], color=COLOR_MODEL)
                axes[1].axvline(
                    float(margins.mean()), color=COLOR_CEILING, linestyle="--", linewidth=1.3
                )
                axes[1].set_title(
                    f"Marge des articles publiés (moyenne {margins.mean() * 100:.1f} %)",
                    fontsize=10,
                )
                axes[1].set_xlabel("Marge brute")
            else:
                axes[1].axis("off")
        else:
            axes[1].axis("off")

        if trade_off:
            labels = ["par pertinence", "pondéré marge"]
            ndcgs = [
                _safe_float(trade_off.get("ndcg_pertinence")),
                _safe_float(trade_off.get("ndcg_pondere_marge")),
            ]
            margins = [
                _safe_float(trade_off.get("marge_moyenne_publiee")),
                _safe_float(trade_off.get("marge_moyenne_ponderee")),
            ]
            positions = np.arange(len(labels))
            axes[2].bar(positions - 0.19, ndcgs, width=0.38, color=COLOR_MODEL, label="NDCG")
            twin = axes[2].twinx()
            twin.bar(positions + 0.19, margins, width=0.38, color=COLOR_INTENT, label="Marge")
            twin.grid(False)
            axes[2].set_xticks(positions)
            axes[2].set_xticklabels(labels, fontsize=8.5)
            axes[2].set_ylim(0, 1.02)
            axes[2].set_ylabel("NDCG", color=COLOR_MODEL)
            twin.set_ylabel("Marge moyenne publiée", color=COLOR_INTENT)
            axes[2].set_title("Arbitrage pertinence contre marge", fontsize=10)
            handles_left, labels_left = axes[2].get_legend_handles_labels()
            handles_right, labels_right = twin.get_legend_handles_labels()
            axes[2].legend(
                handles_left + handles_right,
                labels_left + labels_right,
                fontsize=8,
                loc="upper right",
            )
        else:
            axes[2].axis("off")

        fig.suptitle("Audit de ce qui est réellement publié", fontsize=12)
        fig.tight_layout()
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ stability -------
    def backtest_stability(
        self, result: Any, *, name: str = "backtest_stability.png"
    ) -> Path | None:
        """Draw the NDCG of each chronological fold with its mean and dispersion band.

        Un moteur de recommandation s'évalue sur une période longue, et la saisonnalité du
        catalogue est forte : les pics d'activité changent à la fois les intentions et la
        disponibilité. Mesurer la dispersion du NDCG entre plis chronologiques dit si le moteur
        tient en régime normal comme en pic, ou s'il faudra le recalibrer à chaque saison.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the backtest table is missing.
        """
        table = getattr(result, "backtest", None)
        if not isinstance(table, pd.DataFrame) or table.empty or "ndcg" not in table:
            logger.warning("backtest_stability ignoré : aucun backtest disponible")
            return None
        values = table["ndcg"].to_numpy(dtype="float64")
        dispersion = table.attrs.get("dispersion", {}) if hasattr(table, "attrs") else {}
        mean = _safe_float(dispersion.get("ndcg_moyen"), float(np.nanmean(values)))
        std = _safe_float(dispersion.get("ndcg_ecart_type"), float(np.nanstd(values)))

        fig, axis = plt.subplots(figsize=(9.4, 4.4))
        labels = [
            f"{row.pli}\n{str(getattr(row, 'periode_debut', ''))[:10]}"
            for row in table.itertuples()
        ]
        axis.bar(labels, values, color=COLOR_MODEL, edgecolor="white")
        axis.axhline(mean, color=COLOR_CEILING, linestyle="--", linewidth=1.4)
        axis.axhspan(mean - std, mean + std, color=COLOR_CEILING, alpha=0.10)
        axis.text(
            len(labels) - 0.4,
            mean,
            f" moyenne {mean:.3f}",
            va="center",
            ha="right",
            fontsize=8.5,
            color=COLOR_CEILING,
        )
        for index, value in enumerate(values):
            axis.text(index, value + 0.008, f"{value:.3f}", ha="center", fontsize=8.5)
        amplitude = _safe_float(
            dispersion.get("amplitude"), float(np.nanmax(values) - np.nanmin(values))
        )
        axis.set_title(f"Stabilité temporelle du NDCG@{_top_k(result)} — amplitude {amplitude:.3f}")
        axis.set_ylabel(f"NDCG@{_top_k(result)}")
        axis.set_xlabel("Pli chronologique")
        axis.set_ylim(0, max(1.02, float(np.nanmax(values)) * 1.18))
        fig.tight_layout()
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ errors ----------
    def worst_misrankings(self, result: Any, *, name: str = "worst_misrankings.png") -> Path | None:
        """Show where the engine misranks: buried relevant items and published noise.

        Les deux familles d'erreurs n'ont pas la même cause. Un oubli (article pertinent relégué
        au-delà de la coupure) signale un déficit de lecture de l'intention ; une fausse promesse
        (article non pertinent publié) signale un excès de confiance du score. Répartir les erreurs
        les plus coûteuses par rang et par famille montre immédiatement laquelle domine, et donc
        quel levier actionner en premier.

        Args:
            result: Evaluation result.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the error table is missing.
        """
        table = getattr(result, "errors", None)
        if not isinstance(table, pd.DataFrame) or table.empty:
            logger.warning("worst_misrankings ignoré : aucune erreur de classement")
            return None
        displayed = table.head(TOP_ERRORS).copy()
        if "cout_ndcg" not in displayed.columns:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})

        labels = [
            f"{getattr(row, 'type_erreur', '?')} · rang {getattr(row, 'rang', '?')}"
            for row in displayed.itertuples()
        ]
        colours = [
            COLOR_CEILING if str(kind) == LABEL_FALSE_PROMISE else COLOR_MODEL
            for kind in displayed.get("type_erreur", pd.Series([LABEL_FORGET] * len(displayed)))
        ]
        bars = axes[0].barh(
            labels[::-1], displayed["cout_ndcg"].to_numpy()[::-1], color=colours[::-1]
        )
        for bar, value in zip(bars, displayed["cout_ndcg"].to_numpy()[::-1], strict=True):
            axes[0].text(
                value + 0.004,
                bar.get_y() + bar.get_height() / 2,
                f"{_safe_float(value):.3f}",
                va="center",
                fontsize=8,
            )
        axes[0].set_title(
            f"Les {len(displayed)} mauvais classements les plus coûteux", fontsize=10.5
        )
        axes[0].set_xlabel("Coût NDCG (contribution perdue ou gaspillée)")
        axes[0].tick_params(labelsize=8)

        if "type_erreur" in displayed.columns:
            counts = displayed["type_erreur"].value_counts()
            axes[1].bar(
                list(counts.index),
                counts.to_numpy(dtype="float64"),
                color=[
                    COLOR_CEILING if str(key) == LABEL_FALSE_PROMISE else COLOR_MODEL
                    for key in counts.index
                ],
                edgecolor="white",
            )
            for index, value in enumerate(counts.to_numpy(dtype="float64")):
                axes[1].text(index, value + 0.08, f"{int(value)}", ha="center", fontsize=9)
            axes[1].set_title("Répartition par famille d'erreur", fontsize=10.5)
            axes[1].set_ylabel("Occurrences")
            axes[1].set_ylim(0, max(counts.to_numpy(dtype="float64")) * 1.22)
        else:
            axes[1].axis("off")

        fig.suptitle("Analyse des erreurs de classement", fontsize=12)
        fig.tight_layout()
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ importance ------
    def feature_importance(
        self, result: Any, *, name: str = "feature_importance.png", top: int = TOP_FEATURES
    ) -> Path | None:
        """Draw the grouped permutation importance measured on the ranking quality.

        L'importance native d'un arbre mesure la réduction d'impureté, sans rapport avec la
        qualité d'un classement. Celle-ci est obtenue en mélangeant chaque colonne et en
        recalculant le NDCG utilisateur par utilisateur : elle dit de quoi le moteur dépend
        réellement. Un moteur dont le NDCG s'effondre quand on mélange l'audience de l'article est
        un moteur de popularité habillé en modèle.

        Args:
            result: Evaluation result.
            name: Output file name.
            top: Number of features displayed.

        Returns:
            The written path, or ``None`` when no importance is available.
        """
        table = getattr(result, "feature_importance", None)
        if not isinstance(table, pd.DataFrame) or table.empty:
            logger.warning("feature_importance ignoré : aucune importance calculée")
            return None
        columns = [column for column in ("feature", "importance") if column in table.columns]
        if len(columns) < 2:
            return None
        displayed = table.head(int(top)).iloc[::-1]

        fig, axis = plt.subplots(figsize=(9.2, max(3.6, 0.32 * len(displayed) + 1.6)))
        colours = [
            COLOR_CEILING if _safe_float(value) < 0 else COLOR_MODEL
            for value in displayed["importance"]
        ]
        bars = axis.barh(displayed["feature"].astype(str), displayed["importance"], color=colours)
        for bar, value in zip(bars, displayed["importance"], strict=True):
            axis.text(
                value + (0.0006 if _safe_float(value) >= 0 else -0.0006),
                bar.get_y() + bar.get_height() / 2,
                f"{_safe_float(value):.4f}",
                va="center",
                ha="left" if _safe_float(value) >= 0 else "right",
                fontsize=8,
            )
        axis.axvline(0, color="black", linewidth=0.9)
        method = str(table.get("method", pd.Series(["permutation_ndcg"])).iloc[0])
        axis.set_title(f"Importance par permutation ({method}) — chute de NDCG@{_top_k(result)}")
        axis.set_xlabel("Perte de NDCG quand la colonne est mélangée")
        axis.tick_params(labelsize=8.5)
        fig.tight_layout()
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ all -------------
    def save_all(self, result: Any, *, thresholds: pd.DataFrame | None = None) -> dict[str, Path]:
        """Render every available figure of a ranking evaluation.

        Args:
            result: Evaluation result.
            thresholds: Accepted for API symmetry with the classification plots; a ranking model
                publishes a top-K rather than a decision threshold, so the argument is ignored.

        Returns:
            Mapping of figure name to written path (missing figures are skipped).
        """
        del thresholds  # aucune notion de seuil de décision en classement
        produced: dict[str, Path] = {}
        renderers = (
            ("metrics_bar", self.metrics_bar),
            ("ceiling_position", self.ceiling_position),
            ("baselines_comparison", self.baselines_comparison),
            ("quality_vs_cutoff", self.quality_vs_cutoff),
            ("coverage_vs_cutoff", self.coverage_vs_cutoff),
            ("segment_breakdown", self.segment_breakdown),
            ("stock_and_margin", self.stock_and_margin),
            ("backtest_stability", self.backtest_stability),
            ("worst_misrankings", self.worst_misrankings),
            ("feature_importance", self.feature_importance),
        )
        for name, renderer in renderers:
            try:
                path = renderer(result)
            except Exception as error:
                logger.warning("Figure '{}' non générée : {}", name, error)
                continue
            if path is not None:
                produced[name] = path
        logger.info("{} figures de classement générées dans {}", len(produced), self.figures_dir)
        return produced


__all__ = [
    "COLOR_CEILING",
    "COLOR_INTENT",
    "COLOR_MODEL",
    "COLOR_POPULARITY",
    "COLOR_RANDOM",
    "RankingPlots",
    "TOP_ERRORS",
    "TOP_FEATURES",
]
