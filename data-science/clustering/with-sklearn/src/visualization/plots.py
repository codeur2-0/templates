"""Figures used by the evaluation report and the notebooks.

Every function returns the path of the written PNG: figures are artefacts, not side effects.
The rendering backend is forced to ``Agg`` so that the code works headless (CI, containers).

The figure set is built for an **unsupervised segmentation**: it answers the four questions a
reviewer asks about a clustering — is the structure real (silhouette, k-selection), are the groups
usable (sizes, profiles), are the assignments reliable (distances, confidence), and do the groups
separate anything the business cares about (external validity)?
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

#: Part de population en dessous de laquelle un groupe n'est pas exploitable par le CRM.
MIN_CLUSTER_SHARE = 0.03

#: Nombre de features affichées dans la carte de chaleur des profils.
HEATMAP_FEATURES = 12

#: Palette qualitative des clusters (jusqu'à 10 groupes lisibles).
CLUSTER_PALETTE = "tab10"

#: Couleurs du nuance de confiance (élevée, moyenne, faible, indéterminée).
CONFIDENCE_COLORS = ("#0a9396", "#ee9b00", "#ae2012", "#94a3b8")


def _save(fig: Any, path: str | Path, *, close: bool = True) -> Path:
    """Write a figure to disk.

    Args:
        fig: Matplotlib figure.
        path: Destination file.
        close: Close the figure after writing (avoids memory leaks in loops).

    Returns:
        The written path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=DEFAULT_DPI, bbox_inches="tight")
    if close:
        plt.close(fig)
    logger.debug("Figure written: {}", destination)
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
        return float(value)
    except (TypeError, ValueError):
        return default


def _frame(result: Any) -> pd.DataFrame:
    """Row-level diagnostic frame of an evaluation result (possibly empty)."""
    predictions = getattr(result, "predictions", None)
    return predictions if isinstance(predictions, pd.DataFrame) else pd.DataFrame()


def _profiles(result: Any) -> pd.DataFrame:
    """Per-cluster profile table of an evaluation result (possibly empty)."""
    per_segment = getattr(result, "per_segment", None)
    return per_segment if isinstance(per_segment, pd.DataFrame) else pd.DataFrame()


def _cluster_label(row: pd.Series) -> str:
    """Readable bar label: cluster id, plus its dominant latent profile when known."""
    label = f"{int(_safe_float(row.get('cluster'), 0))}"
    dominant = row.get("dominant_latent")
    if dominant is not None and not pd.isna(dominant):
        label = f"{label} — {dominant}"
    return label


class ClusteringPlots:
    """Figure factory for a clustering evaluation."""

    def __init__(self, figures_dir: str | Path, *, palette: str = CLUSTER_PALETTE) -> None:
        """Store the output directory.

        Args:
            figures_dir: Directory receiving the PNG files.
            palette: Qualitative palette used to colour the clusters.
        """
        self.figures_dir = Path(figures_dir)
        self.palette = palette
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ structure -------
    def cluster_map(self, result: Any, *, name: str = "cluster_map.png") -> Path | None:
        """Project the customers on their first two principal components, coloured by cluster.

        A PCA map is not a proof of separation (the projection can hide it), but it makes the
        overlap visible: groups that interpenetrate on the map are exactly the ones whose members
        have a low individual silhouette.

        Args:
            result: :class:`~src.evaluation.evaluator.EvaluationResult`.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the projection is unavailable.
        """
        curves = dict(getattr(result, "curves", {}) or {})
        projection = dict(curves.get("pca_map", {}) or {})
        x = np.asarray(projection.get("x", []), dtype="float64")
        y = np.asarray(projection.get("y", []), dtype="float64")
        clusters = np.asarray(projection.get("cluster", []), dtype="int64")
        if x.size == 0 or y.size == 0 or clusters.size == 0:
            logger.debug("cluster_map: projection PCA indisponible")
            return None
        variance = list(projection.get("explained_variance", []) or [])
        first = f"PC1 ({_safe_float(variance[0], 0.0):.1%})" if variance else "PC1"
        second = f"PC2 ({_safe_float(variance[1], 0.0):.1%})" if len(variance) > 1 else "PC2"
        frame = pd.DataFrame({"x": x, "y": y, "cluster": clusters.astype(str)})
        fig, axis = plt.subplots(figsize=(8.6, 6.0))
        sns.scatterplot(
            data=frame,
            x="x",
            y="y",
            hue="cluster",
            palette=self.palette,
            alpha=0.62,
            edgecolor="none",
            s=26,
            ax=axis,
            legend="brief",
        )
        axis.set_xlabel(first)
        axis.set_ylabel(second)
        axis.set_title("Carte des clients (ACP 2D) — les groupes se chevauchent-ils ?")
        return _save(fig, self.figures_dir / name)

    def k_selection(self, result: Any, *, name: str = "k_selection.png") -> Path | None:
        """Plot the k-selection curve: silhouette, Davies-Bouldin and inertia elbow.

        The three criteria rarely agree, and that is the point of the figure: choosing k is a
        trade-off between statistical separation and operational readability (a CRM cannot run
        twelve campaigns).

        Args:
            result: :class:`~src.evaluation.evaluator.EvaluationResult`.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the curve is unavailable.
        """
        curves = dict(getattr(result, "curves", {}) or {})
        rows = list(curves.get("k_selection", []) or [])
        if not rows:
            logger.debug("k_selection: aucune courbe disponible")
            return None
        table = pd.DataFrame(rows)
        if table.empty or "k" not in table.columns:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))

        left = axes[0]
        left.plot(table["k"], table["silhouette"], marker="o", color="#005f73", label="silhouette")
        left.set_xlabel("nombre de groupes (k)")
        left.set_ylabel("silhouette moyenne", color="#005f73")
        left.tick_params(axis="y", labelcolor="#005f73")
        right = left.twinx()
        right.plot(
            table["k"],
            table["davies_bouldin"],
            marker="s",
            color="#ae2012",
            label="Davies-Bouldin",
        )
        right.set_ylabel("Davies-Bouldin (plus bas = mieux)", color="#ae2012")
        right.tick_params(axis="y", labelcolor="#ae2012")
        right.grid(False)
        valid = table["silhouette"].notna()
        best = int(table.loc[table["silhouette"].idxmax(), "k"]) if bool(valid.any()) else -1
        if best > 0:
            left.axvline(best, color="#94d2bd", linestyle="--", linewidth=1.4)
            left.annotate(
                f"pic de silhouette : k={best}",
                (best, float(table["silhouette"].max())),
                textcoords="offset points",
                xytext=(8, -4),
                fontsize=9,
            )
        left.set_title("Choix du nombre de groupes")

        inertia = axes[1]
        inertia.plot(table["k"], table["inertia"], marker="o", color="#bb3e03")
        inertia.set_xlabel("nombre de groupes (k)")
        inertia.set_ylabel("inertie intra-cluster")
        inertia.set_title("Coude d'inertie — la décroissance ralentit-elle ?")
        if "min_cluster_share" in table.columns:
            twin = inertia.twinx()
            twin.plot(
                table["k"],
                table["min_cluster_share"] * 100.0,
                marker="^",
                color="#6a4c93",
                linewidth=1.2,
            )
            twin.axhline(MIN_CLUSTER_SHARE * 100.0, color="#6a4c93", linestyle=":", linewidth=1.0)
            twin.set_ylabel("taille du plus petit groupe (%)", color="#6a4c93")
            twin.tick_params(axis="y", labelcolor="#6a4c93")
            twin.grid(False)
        fig.tight_layout()
        return _save(fig, self.figures_dir / name)

    def silhouette_distribution(
        self, result: Any, *, name: str = "silhouette_distribution.png"
    ) -> Path | None:
        """Histogram of the per-customer silhouette, with the mean and the zero line.

        A negative silhouette means the customer is closer to another group than to its own: those
        rows are the segmentation's borderline cases, and their share is a far better indicator
        than the mean alone.

        Args:
            result: :class:`~src.evaluation.evaluator.EvaluationResult`.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the data is unavailable.
        """
        frame = _frame(result)
        if frame.empty or "silhouette" not in frame.columns:
            logger.debug("silhouette_distribution: colonne absente")
            return None
        values = pd.to_numeric(frame["silhouette"], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            return None
        fig, axis = plt.subplots(figsize=(8.4, 4.6))
        axis.hist(values, bins=40, color="#0a9396", edgecolor="white")
        mean = float(np.mean(values))
        axis.axvline(
            mean, color="#ae2012", linestyle="--", linewidth=1.6, label=f"moyenne = {mean:.3f}"
        )
        axis.axvline(0.0, color="#495057", linestyle=":", linewidth=1.2, label="0 (frontière)")
        negative = float(np.mean(values < 0))
        axis.set_title(
            f"Silhouette individuelle — {negative:.1%} des clients plus proches d'un autre groupe"
        )
        axis.set_xlabel("silhouette")
        axis.set_ylabel("nombre de clients")
        axis.legend(fontsize=9)
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ lisibilité ------
    def cluster_sizes(self, result: Any, *, name: str = "cluster_sizes.png") -> Path | None:
        """Bar chart of cluster sizes, with the degeneracy threshold.

        Args:
            result: :class:`~src.evaluation.evaluator.EvaluationResult`.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the profile table is unavailable.
        """
        profiles = _profiles(result)
        if profiles.empty or "size" not in profiles.columns:
            logger.debug("cluster_sizes: table de profils absente")
            return None
        table = profiles.copy()
        table["label"] = [_cluster_label(row) for _, row in table.iterrows()]
        shares = table["share"].to_numpy(dtype="float64") * 100.0
        colors = ["#ae2012" if value < MIN_CLUSTER_SHARE * 100.0 else "#005f73" for value in shares]
        fig, axis = plt.subplots(figsize=(8.6, 0.62 * len(table) + 1.8))
        bars = axis.barh(table["label"][::-1], shares[::-1], color=colors[::-1])
        axis.bar_label(bars, fmt="%.1f %%", padding=3, fontsize=9)
        axis.axvline(MIN_CLUSTER_SHARE * 100.0, color="#6a4c93", linestyle="--", linewidth=1.2)
        axis.annotate(
            f"seuil d'exploitabilité ({MIN_CLUSTER_SHARE:.0%})",
            (MIN_CLUSTER_SHARE * 100.0, max(len(table) - 0.4, 0.0)),
            textcoords="offset points",
            xytext=(6, 0),
            fontsize=9,
            color="#6a4c93",
        )
        axis.set_xlabel("part de la base clients (%)")
        axis.set_title("Taille des segments — les micro-groupes sont-ils actionnables ?")
        axis.set_xlim(0, max(float(shares.max()) * 1.25, 1.0))
        return _save(fig, self.figures_dir / name)

    def profile_heatmap(self, result: Any, *, name: str = "profile_heatmap.png") -> Path | None:
        """Heatmap of the centroid coordinates (standardised space) for the strongest drivers.

        Because the preprocessing standardises the features, a centroid coordinate **is** an
        effect size: ``+1.5`` means "1.5 standard deviations above the average customer". This is
        what makes a cluster readable without opening a spreadsheet.

        Args:
            result: :class:`~src.evaluation.evaluator.EvaluationResult`.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the centroid table is unavailable.
        """
        importance = getattr(result, "feature_importance", None)
        if not isinstance(importance, pd.DataFrame) or importance.empty:
            logger.debug("profile_heatmap: table de centroïdes absente")
            return None
        table = importance.copy()
        if not {"cluster", "feature", "centroid_z"}.issubset(table.columns):
            return None
        pivot = table.pivot_table(index="feature", columns="cluster", values="centroid_z")
        strength = table.groupby("feature")["centroid_z"].apply(
            lambda series: float(series.abs().max())
        )
        keep = strength.sort_values(ascending=False).head(HEATMAP_FEATURES).index.tolist()
        pivot = pivot.loc[[column for column in keep if column in pivot.index]]
        if pivot.empty:
            return None
        width = 0.9 * len(pivot.columns) + 6.0
        height = 0.42 * len(pivot) + 2.0
        fig, axis = plt.subplots(figsize=(width, height))
        sns.heatmap(
            pivot,
            cmap="RdBu_r",
            center=0.0,
            annot=True,
            fmt=".1f",
            linewidths=0.4,
            cbar_kws={"label": "écart-type par rapport au client moyen"},
            ax=axis,
        )
        axis.set_title("Profil des segments — ce qui distingue chaque groupe (espace standardisé)")
        axis.set_xlabel("segment")
        axis.set_ylabel("")
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ validité --------
    def churn_by_cluster(self, result: Any, *, name: str = "churn_by_cluster.png") -> Path | None:
        """Churn rate observed per cluster (external validity).

        The churn column is never a feature: it is generated *after* the snapshot and used only to
        check that the segmentation separates something the business cares about. A segmentation
        whose clusters all churn at the same rate is statistically neat and commercially useless.

        Args:
            result: :class:`~src.evaluation.evaluator.EvaluationResult`.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the outcome column is unavailable.
        """
        profiles = _profiles(result)
        if profiles.empty or "churn_rate" not in profiles.columns:
            logger.debug("churn_by_cluster: aucun outcome externe dans les profils")
            return None
        table = profiles.dropna(subset=["churn_rate"]).copy()
        if table.empty:
            return None
        table["churn_pct"] = table["churn_rate"].to_numpy(dtype="float64") * 100.0
        table = table.sort_values("churn_pct", ascending=False)
        median = float(table["churn_pct"].median())
        colors = ["#ae2012" if value >= median else "#0a9396" for value in table["churn_pct"]]
        labels = [f"segment {int(value)}" for value in table["cluster"]]
        fig, axis = plt.subplots(figsize=(8.4, 4.6))
        bars = axis.bar(labels, table["churn_pct"], color=colors)
        axis.bar_label(bars, fmt="%.1f %%", padding=3, fontsize=9)
        spread = float(table["churn_pct"].max() - table["churn_pct"].min())
        axis.set_title(f"Churn à 90 jours par segment — écart de {spread:.1f} points")
        axis.set_ylabel("clients sans commande à 90 jours (%)")
        axis.set_xlabel("segment")
        return _save(fig, self.figures_dir / name)

    def confidence_breakdown(
        self, result: Any, *, name: str = "confidence_breakdown.png"
    ) -> Path | None:
        """Distribution of the assignment confidence and of the distance to the centre.

        Args:
            result: :class:`~src.evaluation.evaluator.EvaluationResult`.
            name: Output file name.

        Returns:
            The written path, or ``None`` when the data is unavailable.
        """
        frame = _frame(result)
        if frame.empty or "confidence" not in frame.columns:
            logger.debug("confidence_breakdown: colonne absente")
            return None
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
        known = {"élevée", "moyenne", "faible", "indéterminée"}
        order = [
            level
            for level in ("élevée", "moyenne", "faible", "indéterminée")
            if level in set(frame["confidence"]) and level in known
        ]
        counts = frame["confidence"].value_counts().reindex(order).fillna(0)
        axes[0].bar(
            counts.index,
            counts.to_numpy(dtype="float64"),
            color=list(CONFIDENCE_COLORS)[: len(counts)],
        )
        for index, value in enumerate(counts.to_numpy(dtype="float64")):
            axes[0].annotate(
                f"{value / max(len(frame), 1):.1%}",
                (index, value),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=9,
            )
        axes[0].set_title("Confiance de l'affectation")
        axes[0].set_ylabel("nombre de clients")
        axes[0].set_xlabel("niveau de confiance")

        distances = pd.to_numeric(frame.get("distance_to_centroid"), errors="coerce").dropna()
        if len(distances):
            sns.histplot(distances, bins=36, color="#005f73", ax=axes[1], kde=False)
            axes[1].axvline(
                float(distances.median()),
                color="#ae2012",
                linestyle="--",
                label=f"médiane = {distances.median():.2f}",
            )
            axes[1].legend(fontsize=9)
        axes[1].set_title("Distance au centre du segment (espace standardisé)")
        axes[1].set_xlabel("distance euclidienne")
        axes[1].set_ylabel("nombre de clients")
        fig.tight_layout()
        return _save(fig, self.figures_dir / name)

    # ------------------------------------------------------------------ orchestration ---
    def save_all(self, result: Any, *, thresholds: pd.DataFrame | None = None) -> dict[str, Path]:
        """Render every available figure and return their paths.

        Args:
            result: :class:`~src.evaluation.evaluator.EvaluationResult`.
            thresholds: Accepted for API symmetry with the classification figures; a clustering
                has no decision threshold, so the argument is ignored.

        Returns:
            Mapping of figure name to written path (unrenderable figures are absent).
        """
        del thresholds  # aucune notion de seuil de décision en clustering
        candidates = (
            ("cluster_map", self.cluster_map),
            ("k_selection", self.k_selection),
            ("silhouette_distribution", self.silhouette_distribution),
            ("cluster_sizes", self.cluster_sizes),
            ("profile_heatmap", self.profile_heatmap),
            ("churn_by_cluster", self.churn_by_cluster),
            ("confidence_breakdown", self.confidence_breakdown),
        )
        written: dict[str, Path] = {}
        for figure_name, renderer in candidates:
            try:
                path = renderer(result)
            except (ValueError, KeyError, TypeError) as error:  # données insuffisantes
                logger.warning("Figure '{}' non générée : {}", figure_name, error)
                continue
            if path is not None:
                written[figure_name] = path
        logger.info("Figures générées : {}", sorted(written))
        return written


__all__ = ["ClusteringPlots"]
