"""Evaluation of a ranking model: per-user top-K quality, baselines, coverage and fairness.

Évaluer un moteur de recommandation n'est pas évaluer un classifieur. Trois différences
structurelles gouvernent tout ce module :

1. **L'unité d'évaluation est l'utilisateur, pas la ligne.** Une précision calculée sur toutes
   les lignes mêle un utilisateur très actif à un nouveau venu et ne décrit aucun des deux.
   Chaque métrique est donc calculée par utilisateur puis moyennée, ce qui donne le même poids
   à tout le monde — et révèle les segments que la moyenne globale masque.

2. **Un score absolu ne veut rien dire.** Un NDCG@10 de 0,42 est excellent ou médiocre selon le
   nombre de candidats, la prévalence et le bruit de la cible. Ce module calcule donc toujours
   trois références : le tirage aléatoire, le tri par popularité (le moteur historique) et, quand
   les métadonnées de génération sont disponibles, le **plafond atteignable** publié par le
   générateur. La place du modèle dans cette fourchette est la seule lecture honnête.

3. **La pertinence n'est pas le seul objectif.** Un moteur peut améliorer son NDCG en se
   refermant sur dix best-sellers : il détruit alors le catalogue à moyen terme. La couverture
   catalogue, la concentration du top-K (indice de Herfindahl), la part d'articles disponibles
   publiés et le rappel sur les utilisateurs froids sont donc mesurés au même titre que la
   pertinence, et chacun fait l'objet d'un verdict.

Aucun nom de colonne n'est codé en dur : la colonne de regroupement, la colonne d'audience qui
sert de baseline popularité, la colonne de stock et la colonne de marge viennent de la
configuration (bloc ``recommendation`` de ``conf/config.yaml``), ce qui permet de réutiliser cet
évaluateur sur un autre catalogue sans toucher au code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.training.losses_metrics import MetricCalculator, MetricInputs
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

#: Nombre de recommandations publiées par utilisateur.
TOP_K = 10

#: Coupures évaluées : publier 5, 10 ou 20 articles ne mesure pas la même qualité.
DEFAULT_CUTOFFS: tuple[int, ...] = (5, 10, 20)

#: Cardinal maximal d'un axe de segmentation avant regroupement des modalités rares.
MAX_SEGMENT_CARDINALITY = 10

#: Nombre de mauvais classements conservés pour l'analyse d'erreurs.
TOP_K_ERRORS = 25

#: Nom de la métrique de référence du tri par popularité.
POPULARITY_BASELINE = "popularite"

#: Nom de la baseline d'exploitation pure : ne recommander que ce que l'utilisateur a déjà vu.
INTENT_BASELINE = "intention_recente"

#: Nom de la baseline aléatoire.
RANDOM_BASELINE = "aleatoire"

#: Fichier de métadonnées de génération, source du plafond atteignable.
GENERATION_METADATA = "generation_metadata.json"


def _safe_mean(values: Sequence[float]) -> float:
    """Return the mean of a sequence, or NaN when it holds no usable value.

    Args:
        values: Values to average.

    Returns:
        The mean, or ``float('nan')``.
    """
    array = np.asarray([value for value in values if value is not None], dtype="float64")
    array = array[~np.isnan(array)]
    return float(array.mean()) if array.size else float("nan")


def _discounts(size: int) -> np.ndarray:
    """Return the NDCG positional discounts for a list of ``size`` slots.

    Args:
        size: Number of ranked slots.

    Returns:
        The discount vector ``1 / log2(rank + 1)``.
    """
    if size <= 0:
        return np.zeros(0, dtype="float64")
    return 1.0 / np.log2(np.arange(2, size + 2, dtype="float64"))


def _dcg(gains: np.ndarray) -> float:
    """Return the discounted cumulative gain of an ordered gain vector.

    Args:
        gains: Relevance gains, best ranked first.

    Returns:
        The DCG value.
    """
    gains = np.asarray(gains, dtype="float64")
    if gains.size == 0:
        return 0.0
    return float((gains * _discounts(gains.size)).sum())


def _ndcg(truth: np.ndarray, order: np.ndarray, top_k: int) -> float:
    """Return the NDCG@K of one ranked list.

    Args:
        truth: Relevance of every candidate.
        order: Candidate positions, best score first.
        top_k: Number of published slots.

    Returns:
        The NDCG@K value, or 0 when the list holds nothing relevant.
    """
    truth = np.asarray(truth, dtype="float64")
    ideal = _dcg(np.sort(truth)[::-1][:top_k])
    return _dcg(truth[order][:top_k]) / ideal if ideal > 0 else 0.0


def _precision(truth: np.ndarray, order: np.ndarray, top_k: int) -> float:
    """Return the share of published slots that are relevant.

    Args:
        truth: Relevance of every candidate.
        order: Candidate positions, best score first.
        top_k: Number of published slots.

    Returns:
        Precision@K.
    """
    selected = np.asarray(truth, dtype="float64")[order][:top_k]
    return float(selected.sum() / selected.size) if selected.size else 0.0


def _recall(truth: np.ndarray, order: np.ndarray, top_k: int) -> float:
    """Return the share of available relevance captured by the published list.

    Args:
        truth: Relevance of every candidate.
        order: Candidate positions, best score first.
        top_k: Number of published slots.

    Returns:
        Recall@K, or NaN when the user has no relevant candidate.
    """
    truth = np.asarray(truth, dtype="float64")
    total = float(truth.sum())
    if total <= 0:
        return float("nan")
    return float(truth[order][:top_k].sum() / total)


def _average_precision(truth: np.ndarray, order: np.ndarray, top_k: int) -> float:
    """Return the average precision at K of one ranked list.

    Args:
        truth: Relevance of every candidate.
        order: Candidate positions, best score first.
        top_k: Number of published slots.

    Returns:
        AP@K, or 0 when nothing relevant exists.
    """
    selected = np.asarray(truth, dtype="float64")[order][:top_k]
    if selected.sum() <= 0:
        return 0.0
    precisions = np.cumsum(selected) / np.arange(1, selected.size + 1, dtype="float64")
    return float((precisions * selected).sum() / min(selected.sum(), top_k))


def _hit(truth: np.ndarray, order: np.ndarray, top_k: int) -> float:
    """Return 1 when at least one relevant item is published.

    Args:
        truth: Relevance of every candidate.
        order: Candidate positions, best score first.
        top_k: Number of published slots.

    Returns:
        1.0 or 0.0.
    """
    return 1.0 if float(np.asarray(truth, dtype="float64")[order][:top_k].sum()) > 0 else 0.0


def _resolve_time_column(frame: pd.DataFrame, declared: str | None) -> str | None:
    """Return the time axis of a frame: the declared column, else the first datetime column.

    Un backtest chronologique sans axe temporel découpe des lignes au hasard et prétend mesurer une
    stabilité temporelle. Plutôt que de coder en dur un nom de colonne, on accepte celui de la
    configuration et, à défaut, on découvre la première colonne de type date du cadre — la même
    stratégie que l'évaluateur de prévision.

    Args:
        frame: Frame to inspect.
        declared: Column name declared in the configuration (may be ``None``).

    Returns:
        The resolved column name, or ``None`` when the frame carries no date at all.
    """
    if declared and declared in frame.columns:
        return str(declared)
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            return str(column)
    return None


@dataclass(slots=True)
class RankingSettings:
    """Everything the ranking evaluation needs, resolved from the configuration.

    Attributes:
        top_k: Number of recommendations published per user.
        cutoffs: Cutoffs at which the metrics are recomputed (quality depends on K).
        group_column: Column identifying the user — the evaluation unit.
        item_column: Column identifying the catalogue item.
        popularity_column: Column used as the popularity baseline score.
        intent_column: Column used as the exploitation-only baseline score.
        stock_column: Column holding available units (zero means unpublishable).
        margin_column: Column holding the gross margin, for the value-versus-relevance trade-off.
        activity_column: Column counting past orders, used to define the cold-start segment.
        cold_user_max_orders: Order count at or below which a user is considered cold.
        coverage_min: Minimum share of the catalogue that must appear in at least one top-K.
        backtest_spread_max: Maximum acceptable NDCG amplitude between chronological folds. It must
            sit above the sampling noise floor of a fold, otherwise the criterion measures chance
            rather than instability.
        time_column: Session date, used by the temporal backtest.
        random_seed: Seed of the random baseline (reproducible comparison).
    """

    top_k: int = TOP_K
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS
    group_column: str = "user_id"
    item_column: str = "item_id"
    popularity_column: str = "item_views_7d"
    intent_column: str = "user_item_views_30d"
    stock_column: str = "item_stock_units"
    margin_column: str = "item_margin_pct"
    activity_column: str = "user_orders_12m"
    cold_user_max_orders: int = 1
    coverage_min: float = 0.25
    backtest_spread_max: float = 0.15
    time_column: str | None = None
    random_seed: int = 42

    @classmethod
    def resolve(
        cls, config: Mapping[str, Any] | None, *, paths: ProjectPaths | None = None
    ) -> RankingSettings:
        """Build the settings from the root configuration, falling back to the data node.

        Args:
            config: Root configuration mapping (may be ``None`` in tests).
            paths: Project layout, accepted for symmetry with the other task evaluators; a
                ranking evaluation reads nothing from disk while resolving its settings.

        Returns:
            The resolved settings.
        """
        node = dict((config or {}).get("recommendation") or {})
        data_node = dict((config or {}).get("data") or {})
        defaults = cls()

        group_column = str(node.get("group_column") or data_node.get("group_column") or "user_id")
        item_column = str(node.get("item_column") or data_node.get("id_column") or "item_id")
        time_column = node.get("time_column") or data_node.get("time_column")

        cutoffs = node.get("cutoffs") or list(defaults.cutoffs)
        top_k = int(node.get("top_k", defaults.top_k))
        if top_k not in {int(value) for value in cutoffs}:
            cutoffs = [*sorted({*cutoffs, top_k})]

        seed_value = (config or {}).get("seed", defaults.random_seed)
        logger.debug(
            "Ranking settings | group={} item={} top_k={} cutoffs={} coverage_min={}",
            group_column,
            item_column,
            top_k,
            tuple(cutoffs),
            float(node.get("coverage_min", defaults.coverage_min)),
        )
        return cls(
            top_k=top_k,
            cutoffs=tuple(int(value) for value in cutoffs),
            group_column=group_column,
            item_column=item_column,
            popularity_column=str(
                node.get("popularity_column") or defaults.popularity_column
            ),
            intent_column=str(node.get("intent_column") or defaults.intent_column),
            stock_column=str(node.get("stock_column") or defaults.stock_column),
            margin_column=str(node.get("margin_column") or defaults.margin_column),
            activity_column=str(node.get("activity_column") or defaults.activity_column),
            cold_user_max_orders=int(
                node.get("cold_user_max_orders", defaults.cold_user_max_orders)
            ),
            coverage_min=float(node.get("coverage_min", defaults.coverage_min)),
            backtest_spread_max=float(
                node.get("backtest_spread_max", defaults.backtest_spread_max)
            ),
            time_column=str(time_column) if time_column else None,
            random_seed=int(seed_value),
        )


@dataclass(slots=True)
class EvaluationResult:
    """Every diagnostic produced by one ranking evaluation.

    La forme est volontairement identique à celle des autres tâches du dépôt (mêmes noms de
    champs), si bien que le rapporteur, les figures et les notebooks fonctionnent de la même
    façon d'une famille à l'autre. Le champ ``per_horizon`` porte ici les métriques **par
    coupure K** — l'analogue direct d'un horizon de prévision : publier 5 ou 20 articles ne
    mesure pas la même qualité.

    Attributes:
        task: Task identifier (``ranking``).
        split: Split name the evaluation ran on.
        primary_metric: Name of the decision metric.
        metrics: Metric name to value, computed per user then averaged.
        predictions: Published top-K per user, with score, rank and business flags.
        labels: Human readable names of the computed metrics.
        per_segment: Metrics broken down by user activity, category, stock and promotion.
        per_horizon: Metrics recomputed at every cutoff K.
        baselines: Model versus random, popularity and recent-intent scorers.
        backtest: Temporal folds, with the dispersion of the primary metric.
        intervals: Unused by ranking (kept empty for shape compatibility).
        errors: Worst misrankings — relevant items buried, irrelevant items published.
        curves: Metric-versus-cutoff and coverage-versus-cutoff curves.
        feature_importance: Permutation importance of the scoring model.
        extras: Verdicts, coverage, concentration, ceiling and generation references.
        n_samples: Number of evaluated candidate rows.
    """

    task: str
    split: str = "test"
    primary_metric: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    labels: list[str] = field(default_factory=list)
    per_segment: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_horizon: pd.DataFrame = field(default_factory=pd.DataFrame)
    baselines: pd.DataFrame = field(default_factory=pd.DataFrame)
    backtest: pd.DataFrame = field(default_factory=pd.DataFrame)
    intervals: pd.DataFrame = field(default_factory=pd.DataFrame)
    errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    curves: dict[str, Any] = field(default_factory=dict)
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    extras: dict[str, Any] = field(default_factory=dict)
    n_samples: int = 0

    @property
    def primary_value(self) -> float:
        """Return the value of the decision metric."""
        return float(self.metrics.get(self.primary_metric, float("nan")))

    @property
    def error_rate(self) -> float:
        """Return the share of published slots that are not relevant.

        Un moteur de classement ne produit pas d'erreur binaire : ce qui se compte, ce sont les
        emplacements publiés à tort. ``1 - Precision@K`` est cette mesure, et c'est elle que les
        outils communs du dépôt affichent sous le nom de taux d'erreur.
        """
        precision = self.metrics.get(f"precision_at_{self._k_label}", float("nan"))
        return float(1.0 - precision) if not np.isnan(precision) else float("nan")

    @property
    def coverage(self) -> float:
        """Return the share of the catalogue exposed by at least one published top-K."""
        return float(self.extras.get("catalog_coverage", float("nan")))

    @property
    def bias(self) -> float:
        """Return the popularity bias: share of published slots taken by the top decile of views.

        Un moteur sans biais de popularité publierait environ 10 % de ses emplacements pour le
        premier décile d'audience. Au-delà de 40 %, la longue traîne n'existe plus.
        """
        return float(self.extras.get("popularity_bias_share", float("nan")))

    @property
    def interval_coverage(self) -> float:
        """Return NaN: a ranking model publishes no prediction interval."""
        return float("nan")

    @property
    def _k_label(self) -> int:
        """Return the published cutoff used to label the metrics."""
        return int(self.extras.get("top_k", TOP_K))

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a JSON-friendly mapping.

        Returns:
            The serialised evaluation.
        """
        def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
            return [] if frame is None or frame.empty else frame.to_dict(orient="records")

        return {
            "task": self.task,
            "split": self.split,
            "primary_metric": self.primary_metric,
            "metrics": {key: float(value) for key, value in self.metrics.items()},
            "n_samples": int(self.n_samples),
            "labels": list(self.labels),
            "per_segment": _records(self.per_segment),
            "per_cutoff": _records(self.per_horizon),
            "baselines": _records(self.baselines),
            "backtest": _records(self.backtest),
            "errors": _records(self.errors),
            "extras": dict(self.extras),
        }


class Evaluator:
    """Score a fitted ranking model and produce every recommendation diagnostic.

    L'évaluateur ne se contente pas de calculer des métriques : il répond aux quatre questions
    qu'un comité de produit pose réellement — le moteur est-il meilleur que ce qu'on a déjà
    (baselines), l'est-il pour tout le monde (segments et utilisateurs froids), que coûte-t-il au
    catalogue (couverture et concentration), et que rapporte-t-il (marge contre pertinence).
    """

    def __init__(
        self,
        model: BaseModel,
        *,
        metrics_config: Mapping[str, Any] | None = None,
        paths: ProjectPaths | None = None,
        task: str | None = None,
        top_k_errors: int = TOP_K_ERRORS,
        context_columns: Sequence[str] | None = None,
        config: Mapping[str, Any] | None = None,
        settings: RankingSettings | None = None,
    ) -> None:
        """Inject the model, the metric policy and the ranking settings.

        Args:
            model: Fitted model implementing :class:`BaseModel`.
            metrics_config: ``metrics`` configuration node (primary / secondary / task).
            paths: Project layout, used to persist metrics and read the generation metadata.
            task: Task override (defaults to ``model.task``).
            top_k_errors: Number of worst misrankings kept for the error analysis.
            context_columns: Columns copied from the enriched context into the diagnostic frame.
                Defaults to the identifiers declared in the configuration.
            config: Root configuration mapping, used to resolve :class:`RankingSettings`.
            settings: Pre-resolved settings (takes precedence over ``config``).
        """
        self.model = model
        self.paths = paths or ProjectPaths.from_root()
        self.config: dict[str, Any] = dict(metrics_config or {})
        self.task = task or self.config.get("task") or model.task
        self.top_k_errors = int(top_k_errors)
        self.settings = settings or RankingSettings.resolve(config, paths=self.paths)
        self.context_columns: tuple[str, ...] = tuple(
            str(column)
            for column in (context_columns or (self.settings.group_column, self.settings.item_column))
        )
        self.primary_metric = str(self.config.get("primary", "ndcg_at_k"))
        self.metric_names = [
            self.primary_metric,
            *[str(name) for name in self.config.get("secondary", []) or []],
        ]

    @classmethod
    def from_config(
        cls, model: BaseModel, config: Mapping[str, Any], paths: ProjectPaths | None = None
    ) -> Evaluator:
        """Build an evaluator from a root configuration mapping.

        Args:
            model: Fitted model.
            config: Root configuration.
            paths: Optional project layout.

        Returns:
            The configured evaluator.
        """
        metrics_node = dict(config.get("metrics") or {})
        data_node = dict(config.get("data") or {})
        settings = RankingSettings.resolve(config, paths=paths)
        # Aucun nom de colonne n'est codé en dur : identifiants, groupe et date viennent de la
        # configuration, si bien que l'évaluateur suit un changement de schéma sans édition.
        context_columns = [
            str(data_node[key])
            for key in ("id_column", "group_column", "time_column")
            if data_node.get(key)
        ]
        for extra_column in (settings.item_column, settings.time_column):
            if extra_column and extra_column not in context_columns:
                context_columns.append(str(extra_column))
        return cls(
            model,
            metrics_config=metrics_node,
            paths=paths,
            task=metrics_node.get("task"),
            context_columns=tuple(context_columns) or None,
            config=config,
            settings=settings,
        )

    # ------------------------------------------------------------------ scoring ---------
    def _score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return a ranking score for every row, higher meaning more relevant.

        Un modèle de classement peut être un classifieur (la probabilité de pertinence est le
        score naturel) ou un régresseur (la valeur prédite joue le même rôle). Les deux sont
        acceptés, ce qui permet de comparer un arbre probabiliste et un modèle de factorisation
        sans changer de code.

        Args:
            X: Preprocessed features.

        Returns:
            The score vector.

        Raises:
            RuntimeError: When the model is not fitted.
        """
        self.model.check_is_fitted()
        if bool(getattr(self.model, "supports_proba", False)):
            try:
                probabilities = np.asarray(self.model.predict_proba(X), dtype="float64")
                if probabilities.ndim == 2:
                    return probabilities[:, -1].ravel()
                return probabilities.ravel()
            except Exception as exc:  # noqa: BLE001 - un repli vaut mieux qu'un échec d'évaluation
                logger.warning("predict_proba unavailable ({}); falling back to predict", exc)
        return np.asarray(self.model.predict(X), dtype="float64").ravel()

    def _diagnostic_frame(
        self,
        y: pd.Series | Sequence[Any],
        score: np.ndarray,
        context: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Assemble the frame every analysis works on: one row per scored candidate.

        Args:
            y: Observed relevance.
            score: Model score, aligned with ``y``.
            context: Enriched split rows carrying the identifiers and segment columns.

        Returns:
            A frame with truth, score, group, item and every available segment column.

        Raises:
            ValueError: When the group column is missing — per-user metrics are impossible
                without it, and a global average would be a contresens.
        """
        truth = np.asarray(pd.Series(y).to_numpy(), dtype="float64").ravel()
        frame = pd.DataFrame({"y_true": truth, "y_pred": np.asarray(score, dtype="float64")})
        if context is None:
            msg = (
                "Ranking evaluation needs the evaluation context: the user column is what makes "
                "per-user metrics possible, and a global average would mix heavy and new users."
            )
            raise ValueError(msg)

        available = context.reset_index(drop=True)
        if len(available) != len(frame):
            msg = (
                f"Context rows ({len(available)}) and scored rows ({len(frame)}) do not match; "
                "the evaluation context must be aligned with the feature matrix."
            )
            raise ValueError(msg)

        wanted = [
            column
            for column in (
                *self.context_columns,
                self.settings.popularity_column,
                self.settings.intent_column,
                self.settings.stock_column,
                self.settings.margin_column,
                self.settings.activity_column,
            )
            if column in available.columns
        ]
        # The session date drives the chronological backtest. It comes from the configuration when
        # declared; otherwise the first datetime column of the context is used, which is how the
        # forecasting evaluator finds its own time axis. Discovering it beats hard-coding a name,
        # and beats silently dropping the time axis — a backtest on unsorted rows would claim to be
        # chronological while measuring nothing of the sort.
        time_column = _resolve_time_column(available, self.settings.time_column)
        if time_column and time_column not in wanted:
            wanted.append(time_column)
        for column in dict.fromkeys(wanted):
            frame[column] = available[column].to_numpy()

        group_column = self.settings.group_column
        if group_column not in frame.columns:
            msg = (
                f"Group column '{group_column}' is missing from the evaluation context; "
                "declare it in `data.group_column` or `recommendation.group_column`."
            )
            raise ValueError(msg)
        return frame

    # ------------------------------------------------------------------ metrics ---------
    def ranking_metrics(
        self, frame: pd.DataFrame, score_column: str, *, top_k: int | None = None
    ) -> dict[str, float]:
        """Compute the ranking metrics per user, then average them.

        Args:
            frame: Diagnostic frame.
            score_column: Column holding the score to rank by.
            top_k: Published cutoff (defaults to the configured one).

        Returns:
            A mapping with NDCG@K, Precision@K, Recall@K, MAP@K and hit-rate@K. Users with no
            relevant candidate are skipped for the recall, which is undefined for them.
        """
        cutoff = int(top_k or self.settings.top_k)
        buckets = self._per_user_values(frame, score_column, cutoff=cutoff)
        return {
            f"ndcg_at_{cutoff}": _safe_mean(buckets["ndcg"]),
            f"precision_at_{cutoff}": _safe_mean(buckets["precision"]),
            f"recall_at_{cutoff}": _safe_mean(buckets["recall"]),
            f"map_at_{cutoff}": _safe_mean(buckets["map"]),
            f"hit_rate_at_{cutoff}": _safe_mean(buckets["hit"]),
            "users_evaluated": float(len(buckets["ndcg"])),
        }

    def _per_user_values(
        self, frame: pd.DataFrame, score_column: str, *, cutoff: int
    ) -> dict[str, list[float]]:
        """Compute every ranking metric once per user and return the raw per-user values.

        Retourner les valeurs individuelles plutôt que leur seule moyenne est ce qui permet de
        mesurer la **dispersion** d'une métrique : sans l'écart-type entre utilisateurs, aucune
        différence entre deux segments ou deux périodes ne peut être distinguée du bruit.

        Args:
            frame: Diagnostic frame.
            score_column: Column to rank by.
            cutoff: Number of published slots.

        Returns:
            A mapping of metric family to the list of per-user values. Users holding no relevant
            candidate are skipped: no ranking can satisfy them, and including them would reward a
            model that publishes little.
        """
        buckets: dict[str, list[float]] = {
            "ndcg": [],
            "precision": [],
            "recall": [],
            "map": [],
            "hit": [],
        }
        for _, group in frame.groupby(self.settings.group_column, observed=True, sort=False):
            truth = group["y_true"].to_numpy(dtype="float64")
            if truth.sum() <= 0:
                continue
            scores = group[score_column].to_numpy(dtype="float64")
            order = np.argsort(-scores, kind="stable")
            buckets["ndcg"].append(_ndcg(truth, order, cutoff))
            buckets["precision"].append(_precision(truth, order, cutoff))
            buckets["recall"].append(_recall(truth, order, cutoff))
            buckets["map"].append(_average_precision(truth, order, cutoff))
            buckets["hit"].append(_hit(truth, order, cutoff))
        return buckets

    def _configured_metrics(self, frame: pd.DataFrame) -> dict[str, float]:
        """Return the metrics declared in the configuration, computed via the shared calculator.

        Passer par :class:`MetricCalculator` garantit que les chiffres du rapport sont calculés
        exactement comme ceux des tests et des notebooks : une seule implémentation, une seule
        vérité.

        Args:
            frame: Diagnostic frame.

        Returns:
            A mapping of configured metric name to value.
        """
        cutoff = self.settings.top_k
        known = {
            "ndcg_at_k": "ndcg",
            "precision_at_k": "precision",
            "recall_at_k": "recall",
            "map_at_k": "map",
            "hit_rate_at_k": "hit",
        }
        names = [name for name in self.metric_names if name]
        requested = [name for name in names if name in known]
        if not requested:
            requested = ["ndcg_at_k"]
        calculator = MetricCalculator(
            task=str(self.task), metrics=requested, extra={"top_k": cutoff}
        )
        values = calculator.evaluate(
            MetricInputs(
                y_true=frame["y_true"].to_numpy(),
                y_pred=frame["y_pred"].to_numpy(),
                groups=frame[self.settings.group_column].to_numpy(),
                extra={"top_k": cutoff},
            )
        )
        return {str(key): float(value) for key, value in values.items()}

    def per_cutoff(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Recompute every metric at each published cutoff.

        Publier 5 ou 20 articles ne mesure pas la même qualité : la précision chute quand K
        grandit, le rappel monte. Cette table montre où se situe le compromis et justifie le K
        retenu par la configuration au lieu de le laisser implicite.

        Args:
            frame: Diagnostic frame.

        Returns:
            A frame with one row per cutoff.
        """
        rows: list[dict[str, Any]] = []
        for cutoff in self.settings.cutoffs:
            values = self.ranking_metrics(frame, "y_pred", top_k=int(cutoff))
            coverage, concentration = self.catalog_exposure(frame, "y_pred", top_k=int(cutoff))
            rows.append(
                {
                    "top_k": int(cutoff),
                    "ndcg": round(values[f"ndcg_at_{cutoff}"], 4),
                    "precision": round(values[f"precision_at_{cutoff}"], 4),
                    "recall": round(values[f"recall_at_{cutoff}"], 4),
                    "map": round(values[f"map_at_{cutoff}"], 4),
                    "hit_rate": round(values[f"hit_rate_at_{cutoff}"], 4),
                    "catalog_coverage": round(coverage, 4),
                    "herfindahl": round(concentration, 4),
                    "is_configured": bool(int(cutoff) == self.settings.top_k),
                }
            )
        return pd.DataFrame(rows)

    def catalog_exposure(
        self, frame: pd.DataFrame, score_column: str, *, top_k: int | None = None
    ) -> tuple[float, float]:
        """Measure how much of the catalogue the published lists expose, and how concentrated.

        La couverture compte les références distinctes publiées au moins une fois ; l'indice de
        Herfindahl mesure la concentration des emplacements (1,0 = un seul article occupe tout,
        0 = exposition parfaitement uniforme). Les deux ensemble disent si un moteur diversifie
        ou s'il se referme sur une poignée de best-sellers.

        Args:
            frame: Diagnostic frame.
            score_column: Column to rank by.
            top_k: Published cutoff.

        Returns:
            The coverage share and the Herfindahl index.
        """
        cutoff = int(top_k or self.settings.top_k)
        item_column = self.settings.item_column
        if item_column not in frame.columns:
            return float("nan"), float("nan")
        published: list[Any] = []
        for _, group in frame.groupby(self.settings.group_column, observed=True, sort=False):
            order = np.argsort(-group[score_column].to_numpy(dtype="float64"), kind="stable")[
                :cutoff
            ]
            published.extend(group[item_column].to_numpy()[order].tolist())
        if not published:
            return float("nan"), float("nan")
        counts = pd.Series(published).value_counts()
        catalogue_size = max(int(frame[item_column].nunique()), 1)
        shares = counts.to_numpy(dtype="float64") / float(counts.sum())
        return float(counts.size / catalogue_size), float((shares**2).sum())

    # ------------------------------------------------------------------ baselines -------
    def compare_to_baselines(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Rank the model against three references a recommender must beat.

        Trois références, trois questions distinctes. Le tirage aléatoire mesure la difficulté
        intrinsèque du jeu de candidats. Le tri par popularité est le moteur historique : le battre
        est la condition minimale pour justifier un modèle, car il est gratuit, explicable et
        robuste. La recommandation de l'intention récente (ce que l'utilisateur a déjà consulté)
        est l'exploitation pure : elle capture le retour des visiteurs sans rien découvrir. Un
        modèle utile doit dominer les trois, et l'écart à la popularité est le chiffre à citer en
        comité.

        Args:
            frame: Diagnostic frame.

        Returns:
            A frame with one row per scorer and the configured metrics plus coverage.
        """
        scorers: dict[str, str] = {RANDOM_BASELINE: "_score_random"}
        popularity = self.settings.popularity_column
        intent = self.settings.intent_column
        if popularity in frame.columns:
            scorers[POPULARITY_BASELINE] = popularity
        if intent in frame.columns:
            scorers[INTENT_BASELINE] = intent
        scorers["modele"] = "y_pred"

        working = frame.copy()
        rng = np.random.default_rng(self.settings.random_seed)
        working["_score_random"] = rng.random(len(working))
        if popularity not in working.columns:
            working[popularity] = np.zeros(len(working))

        rows: list[dict[str, Any]] = []
        for label, column in scorers.items():
            values = self.ranking_metrics(working, column)
            coverage, concentration = self.catalog_exposure(working, column)
            rows.append(
                {
                    "scorer": label,
                    "ndcg": round(values[f"ndcg_at_{self.settings.top_k}"], 4),
                    "precision": round(values[f"precision_at_{self.settings.top_k}"], 4),
                    "recall": round(values[f"recall_at_{self.settings.top_k}"], 4),
                    "map": round(values[f"map_at_{self.settings.top_k}"], 4),
                    "hit_rate": round(values[f"hit_rate_at_{self.settings.top_k}"], 4),
                    "catalog_coverage": round(coverage, 4),
                    "herfindahl": round(concentration, 4),
                }
            )
        table = pd.DataFrame(rows)
        reference = table.set_index("scorer")
        if POPULARITY_BASELINE in reference.index and "modele" in reference.index:
            popularity_ndcg = float(reference.loc[POPULARITY_BASELINE, "ndcg"])
            model_ndcg = float(reference.loc["modele", "ndcg"])
            gain = model_ndcg - popularity_ndcg
            relative = gain / popularity_ndcg * 100.0 if popularity_ndcg > 0 else float("nan")
            table["gain_vs_popularite"] = round(gain, 4)
            table["gain_relatif_pct"] = round(relative, 2)
        else:
            table["gain_vs_popularite"] = float("nan")
            table["gain_relatif_pct"] = float("nan")
        logger.info(
            "Baselines NDCG@{} | {}",
            self.settings.top_k,
            " | ".join(f"{row.scorer}={row.ndcg:.3f}" for row in table.itertuples()),
        )
        return table

    # ------------------------------------------------------------------ segments --------
    def _segment_axes(self, frame: pd.DataFrame) -> dict[str, pd.Series]:
        """Derive every segmentation axis available in the diagnostic frame.

        Args:
            frame: Diagnostic frame.

        Returns:
            A mapping of axis label to a categorical Series aligned with the frame.
        """
        axes: dict[str, pd.Series] = {}
        activity = self.settings.activity_column
        if activity in frame.columns:
            counts = pd.to_numeric(frame[activity], errors="coerce").fillna(0)
            axes["segment_utilisateur"] = pd.Series(
                np.select(
                    [
                        counts <= self.settings.cold_user_max_orders,
                        counts <= max(self.settings.cold_user_max_orders * 5, 5),
                    ],
                    ["froid", "tiepide"],
                    default="chaud",
                ),
                index=frame.index,
            )
        stock = self.settings.stock_column
        if stock in frame.columns:
            units = pd.to_numeric(frame[stock], errors="coerce").fillna(0)
            axes["disponibilite"] = pd.Series(
                np.where(units > 0, "en_stock", "rupture"), index=frame.index
            )
        margin = self.settings.margin_column
        if margin in frame.columns:
            values = pd.to_numeric(frame[margin], errors="coerce")
            axes["bande_marge"] = pd.Series(
                pd.cut(values, bins=[0.0, 0.25, 0.40, 1.0], labels=["faible", "moyenne", "haute"]),
                index=frame.index,
            ).astype("object")
        for column in frame.columns:
            if column.endswith("_category") and column not in axes:
                axes["categorie"] = frame[column]
                break
        if "item_promoted" in frame.columns:
            axes["promotion"] = pd.Series(
                np.where(frame["item_promoted"].astype(bool), "mis_en_avant", "organique"),
                index=frame.index,
            )
        return axes

    def _per_segment(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Break the ranking quality down by every available business axis.

        Args:
            frame: Diagnostic frame.

        Returns:
            A long frame with one row per axis value.
        """
        cutoff = self.settings.top_k
        rows: list[dict[str, Any]] = []
        for axis, labels in self._segment_axes(frame).items():
            working = frame.copy()
            working["_segment"] = labels
            for value, group in working.groupby("_segment", observed=True, sort=True):
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    continue
                values = self.ranking_metrics(group, "y_pred", top_k=cutoff)
                rows.append(
                    {
                        "axe": axis,
                        "segment": str(value),
                        "n_lignes": int(len(group)),
                        "n_utilisateurs": int(values["users_evaluated"]),
                        "ndcg": round(values[f"ndcg_at_{cutoff}"], 4),
                        "precision": round(values[f"precision_at_{cutoff}"], 4),
                        "recall": round(values[f"recall_at_{cutoff}"], 4),
                        "hit_rate": round(values[f"hit_rate_at_{cutoff}"], 4),
                    }
                )
        table = pd.DataFrame(rows)
        if table.empty:
            return table
        global_ndcg = float(table.loc[table["axe"] == "segment_utilisateur", "ndcg"].mean())
        if np.isnan(global_ndcg):
            global_ndcg = float(table["ndcg"].mean())
        table["ratio_ndcg_global"] = (table["ndcg"] / global_ndcg).round(3) if global_ndcg else np.nan
        table["verdict"] = np.where(
            table["ratio_ndcg_global"].fillna(1.0) < 0.90, "à surveiller", "aligné"
        )
        return table.sort_values(["axe", "segment"]).reset_index(drop=True)

    def cold_start_ratio(self, frame: pd.DataFrame, per_segment: pd.DataFrame) -> float:
        """Return the cold-user recall divided by the global recall.

        Le démarrage à froid est le test le plus sévère d'un moteur de recommandation : un
        utilisateur sans historique ne peut être servi que par les attributs du catalogue. Un
        ratio proche de 1 signifie que le modèle ne dépend pas de l'historique ; un ratio bas
        signifie qu'il exclut de fait les nouveaux venus, ce qui est un problème d'acquisition
        avant d'être un problème de précision.

        Args:
            frame: Diagnostic frame.
            per_segment: Segment table produced by :meth:`_per_segment`.

        Returns:
            The ratio, or NaN when it cannot be measured.
        """
        cutoff = self.settings.top_k
        if per_segment.empty:
            return float("nan")
        selection = per_segment[per_segment["axe"] == "segment_utilisateur"]
        cold = selection[selection["segment"] == "froid"]
        if cold.empty:
            return float("nan")
        global_values = self.ranking_metrics(frame, "y_pred", top_k=cutoff)
        global_recall = global_values[f"recall_at_{cutoff}"]
        cold_recall = float(cold["recall"].iloc[0])
        if not global_recall or np.isnan(global_recall):
            return float("nan")
        return round(cold_recall / global_recall, 4)

    # ------------------------------------------------------------------ publication -----
    def published_topk(self, frame: pd.DataFrame, *, top_k: int | None = None) -> pd.DataFrame:
        """Materialise what would actually be shown to each user.

        Cette table est la seule que le produit consomme : les autres sont des diagnostics. Elle
        porte pour chaque emplacement le rang, le score, la pertinence observée, la disponibilité
        et la marge, ce qui permet d'auditer ligne à ligne ce que le moteur publie — et de compter
        les emplacements perdus en rupture de stock.

        Args:
            frame: Diagnostic frame.
            top_k: Number of published slots.

        Returns:
            The published recommendations, best user first.
        """
        cutoff = int(top_k or self.settings.top_k)
        item_column = self.settings.item_column
        stock_column = self.settings.stock_column
        margin_column = self.settings.margin_column
        records: list[dict[str, Any]] = []
        for group_key, group in frame.groupby(self.settings.group_column, observed=True, sort=False):
            order = np.argsort(-group["y_pred"].to_numpy(dtype="float64"), kind="stable")[:cutoff]
            selected = group.iloc[order]
            for rank, (_, row) in enumerate(selected.iterrows(), start=1):
                stock = row.get(stock_column, np.nan) if stock_column in selected.columns else np.nan
                records.append(
                    {
                        self.settings.group_column: group_key,
                        "rank": rank,
                        item_column: row.get(item_column, np.nan),
                        "score": round(float(row["y_pred"]), 6),
                        "pertinent": bool(row["y_true"] > 0),
                        "en_stock": bool(pd.notna(stock) and float(stock) > 0),
                        "stock_restant": float(stock) if pd.notna(stock) else float("nan"),
                        "marge_pct": float(row[margin_column])
                        if margin_column in selected.columns and pd.notna(row[margin_column])
                        else float("nan"),
                        "score_popularite": float(row[self.settings.popularity_column])
                        if self.settings.popularity_column in selected.columns
                        else float("nan"),
                    }
                )
        return pd.DataFrame(records)

    def margin_trade_off(self, frame: pd.DataFrame, published: pd.DataFrame) -> dict[str, float]:
        """Compare relevance-only ranking against margin-weighted ranking.

        Classer par pertinence maximise l'expérience ; classer par pertinence pondérée par la
        marge maximise le revenu. Les deux ne coïncident pas, et l'écart chiffré est ce qui permet
        de trancher en comité plutôt que par intuition. Le module publie les deux NDCG et la marge
        moyenne attendue de chaque politique.

        Args:
            frame: Diagnostic frame.
            published: Published recommendations of the relevance-only policy.

        Returns:
            A mapping with both NDCG values and both expected margins.
        """
        margin_column = self.settings.margin_column
        if margin_column not in frame.columns:
            return {
                "ndcg_pertinence": float("nan"),
                "ndcg_pondere_marge": float("nan"),
                "marge_moyenne_publiee": float("nan"),
                "marge_moyenne_ponderee": float("nan"),
            }
        working = frame.copy()
        margins = pd.to_numeric(working[margin_column], errors="coerce").fillna(0.0).to_numpy()
        working["_score_marge"] = working["y_pred"].to_numpy(dtype="float64") * (1.0 + margins)
        relevance = self.ranking_metrics(working, "y_pred")
        weighted = self.ranking_metrics(working, "_score_marge")
        cutoff = self.settings.top_k

        def _mean_margin(score_column: str) -> float:
            values: list[float] = []
            for _, group in working.groupby(self.settings.group_column, observed=True, sort=False):
                order = np.argsort(-group[score_column].to_numpy(dtype="float64"), kind="stable")[
                    :cutoff
                ]
                values.extend(
                    pd.to_numeric(group[margin_column], errors="coerce")
                    .fillna(0.0)
                    .to_numpy()[order]
                    .tolist()
                )
            return _safe_mean(values)

        published_margin = (
            _safe_mean(published["marge_pct"].tolist()) if "marge_pct" in published.columns else np.nan
        )
        return {
            "ndcg_pertinence": round(relevance[f"ndcg_at_{cutoff}"], 4),
            "ndcg_pondere_marge": round(weighted[f"ndcg_at_{cutoff}"], 4),
            "marge_moyenne_publiee": round(float(published_margin), 4)
            if not np.isnan(published_margin)
            else round(_mean_margin("y_pred"), 4),
            "marge_moyenne_ponderee": round(_mean_margin("_score_marge"), 4),
        }

    # ------------------------------------------------------------------ errors ----------
    def worst_misrankings(self, frame: pd.DataFrame, *, top_k: int | None = None) -> pd.DataFrame:
        """Isolate the misrankings that cost the most, and say why each one hurts.

        Deux familles d'erreurs, deux causes différentes. Un article pertinent relégué au-delà de
        la coupure est un **oubli** : le moteur n'a pas su lire l'intention. Un article non
        pertinent publié dans le top-K est une **fausse promesse** : il occupe un emplacement
        rare. Les classer par perte de NDCG permet de traiter d'abord ce qui coûte vraiment, au
        lieu de corriger l'erreur la plus visible.

        Args:
            frame: Diagnostic frame.
            top_k: Published cutoff.

        Returns:
            The worst misrankings, most costly first.
        """
        cutoff = int(top_k or self.settings.top_k)
        item_column = self.settings.item_column
        rows: list[dict[str, Any]] = []
        for group_key, group in frame.groupby(self.settings.group_column, observed=True, sort=False):
            truth = group["y_true"].to_numpy(dtype="float64")
            if truth.sum() <= 0:
                continue
            scores = group["y_pred"].to_numpy(dtype="float64")
            order = np.argsort(-scores, kind="stable")
            ideal = _dcg(np.sort(truth)[::-1][:cutoff])
            if ideal <= 0:
                continue
            for position, index in enumerate(order, start=1):
                row = group.iloc[int(index)]
                relevant = bool(truth[int(index)] > 0)
                published = position <= cutoff
                if relevant == published:
                    continue  # bien classé : ni oubli ni fausse promesse
                if relevant and not published:
                    kind = "oubli"
                    # Coût réel : la contribution NDCG perdue en ne publiant pas cet emplacement.
                    cost = float(truth[int(index)] / np.log2(position + 1))
                else:
                    kind = "fausse_promesse"
                    cost = float(1.0 / np.log2(position + 1))
                rows.append(
                    {
                        self.settings.group_column: group_key,
                        item_column: row.get(item_column, np.nan),
                        "type_erreur": kind,
                        "rang": int(position),
                        "score": round(float(scores[int(index)]), 6),
                        "pertinence": float(truth[int(index)]),
                        "cout_ndcg": round(cost, 4),
                        "en_stock": bool(
                            pd.notna(row.get(self.settings.stock_column, np.nan))
                            and float(row.get(self.settings.stock_column, 0) or 0) > 0
                        ),
                        "vues_7j": float(row.get(self.settings.popularity_column, np.nan))
                        if self.settings.popularity_column in group.columns
                        else float("nan"),
                        "consultations_30j": float(row.get(self.settings.intent_column, np.nan))
                        if self.settings.intent_column in group.columns
                        else float("nan"),
                        "note_article": float(row.get("item_rating", np.nan))
                        if "item_rating" in group.columns
                        else float("nan"),
                    }
                )
        if not rows:
            return pd.DataFrame()
        table = pd.DataFrame(rows).sort_values("cout_ndcg", ascending=False)
        return table.head(self.top_k_errors).reset_index(drop=True)

    def curves(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Trace how quality and diversity evolve with the published cutoff.

        Args:
            frame: Diagnostic frame.

        Returns:
            A mapping of curve name to ``{"cutoff": [...], "value": [...]}``.
        """
        maximum = max(*self.settings.cutoffs, self.settings.top_k)
        grid = list(range(1, maximum + 1))
        series: dict[str, list[float]] = {
            "ndcg": [],
            "precision": [],
            "recall": [],
            "coverage": [],
            "herfindahl": [],
            "part_rupture": [],
        }
        for cutoff in grid:
            values = self.ranking_metrics(frame, "y_pred", top_k=cutoff)
            coverage, concentration = self.catalog_exposure(frame, "y_pred", top_k=cutoff)
            published = self.published_topk(frame, top_k=cutoff)
            out_of_stock = (
                float((~published["en_stock"].astype(bool)).mean())
                if not published.empty and "en_stock" in published.columns
                else float("nan")
            )
            series["ndcg"].append(round(values[f"ndcg_at_{cutoff}"], 4))
            series["precision"].append(round(values[f"precision_at_{cutoff}"], 4))
            series["recall"].append(round(values[f"recall_at_{cutoff}"], 4))
            series["coverage"].append(round(coverage, 4))
            series["herfindahl"].append(round(concentration, 4))
            series["part_rupture"].append(round(out_of_stock, 4))
        return {"cutoff": grid, **series}

    # ------------------------------------------------------------------ backtest --------
    def backtest(
        self,
        X: pd.DataFrame | None = None,
        y: pd.Series | Sequence[Any] | None = None,
        context: pd.DataFrame | None = None,
        *,
        folds: int = 3,
        refit: bool = False,
        model_factory: Any = None,
        score: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Measure the stability of the ranking quality across successive periods.

        Un NDCG moyen sur une année cache les saisons : un moteur peut très bien se comporter en
        période creuse et s'effondrer pendant les pics, quand le catalogue et les intentions
        changent le plus. Découper la période évaluée en plis chronologiques successifs et
        recalculer les métriques sur chacun révèle cette dispersion. Avec ``refit``, chaque pli
        est entraîné uniquement sur ce qui le précède, ce qui mesure en plus la sensibilité du
        modèle à la fraîcheur des données.

        Args:
            X: Preprocessed features, required when ``refit``.
            y: Observed relevance, required when ``refit``.
            context: Enriched split rows carrying the identifiers and the date.
            folds: Number of chronological folds.
            refit: Whether to retrain on the past of each fold instead of reusing the model.
            model_factory: Callable returning an unfitted model, used when ``refit``.
            score: Pre-computed scores, reused when the caller already scored the split.

        Returns:
            A frame with one row per fold, plus the dispersion of the primary metric.
        """
        if context is None:
            logger.warning("Backtest skipped: no evaluation context available")
            return pd.DataFrame()
        if score is None:
            score = self._score(X) if X is not None else np.asarray(y, dtype="float64")
        frame = self._diagnostic_frame(y if y is not None else [], score, context)
        time_column = _resolve_time_column(frame, self.settings.time_column)
        if time_column and time_column in frame.columns:
            frame = frame.sort_values(time_column, kind="stable").reset_index(drop=True)
        folds = max(int(folds), 2)
        if len(frame) < folds * 30:
            logger.warning(
                "Backtest skipped: {} rows are not enough for {} chronological folds",
                len(frame),
                folds,
            )
            return pd.DataFrame()

        boundaries = np.linspace(0, len(frame), folds + 1, dtype=int)
        cutoff = self.settings.top_k
        rows: list[dict[str, Any]] = []
        for index in range(folds):
            start, stop = int(boundaries[index]), int(boundaries[index + 1])
            if stop <= start:
                continue
            fold = frame.iloc[start:stop]
            if refit and X is not None and y is not None and model_factory is not None:
                fold = self._refit_fold(X, y, fold, start, stop, model_factory)
            values = self.ranking_metrics(fold, "y_pred", top_k=cutoff)
            coverage, concentration = self.catalog_exposure(fold, "y_pred", top_k=cutoff)
            fold_ndcg = np.asarray(
                self._per_user_values(fold, "y_pred", cutoff=cutoff)["ndcg"], dtype="float64"
            )
            # Erreur-type de la moyenne du pli : c'est elle qui dit si un écart entre plis est une
            # instabilité réelle ou le simple bruit d'un effectif de quelques dizaines
            # d'utilisateurs. Publier l'amplitude sans cette référence serait publier un chiffre
            # ininterprétable.
            standard_error = (
                float(fold_ndcg.std(ddof=1) / np.sqrt(fold_ndcg.size))
                if fold_ndcg.size > 1
                else float("nan")
            )
            period = (
                (str(fold[time_column].min())[:10], str(fold[time_column].max())[:10])
                if time_column and time_column in fold.columns
                else (f"pli_{index + 1}", f"pli_{index + 1}")
            )
            rows.append(
                {
                    "pli": index + 1,
                    "periode_debut": period[0],
                    "periode_fin": period[1],
                    "n_lignes": int(len(fold)),
                    "n_utilisateurs": int(values["users_evaluated"]),
                    "prevalence": round(float(fold["y_true"].mean()), 4),
                    "ndcg": round(values[f"ndcg_at_{cutoff}"], 4),
                    "precision": round(values[f"precision_at_{cutoff}"], 4),
                    "recall": round(values[f"recall_at_{cutoff}"], 4),
                    "catalog_coverage": round(coverage, 4),
                    "herfindahl": round(concentration, 4),
                    "ndcg_erreur_type": round(standard_error, 4)
                    if standard_error == standard_error
                    else float("nan"),
                }
            )
        table = pd.DataFrame(rows)
        if table.empty:
            return table
        values = table["ndcg"].to_numpy(dtype="float64")
        errors = pd.to_numeric(table.get("ndcg_erreur_type"), errors="coerce").dropna()
        table.attrs["dispersion"] = {
            "ndcg_moyen": round(float(values.mean()), 4),
            # Erreur-type moyenne d'un pli : le plancher de bruit sous lequel une amplitude
            # inter-plis ne mesure rien d'autre que le hasard.
            "ndcg_erreur_type_pli": round(float(errors.mean()), 4) if not errors.empty else float("nan"),
            "ndcg_ecart_type": round(float(values.std(ddof=1)) if len(values) > 1 else 0.0, 4),
            "ndcg_min": round(float(values.min()), 4),
            "ndcg_max": round(float(values.max()), 4),
            "amplitude": round(float(values.max() - values.min()), 4),
        }
        logger.info(
            "Backtest | NDCG@{} moyen {:.4f} (écart-type {:.4f}, amplitude {:.4f})",
            cutoff,
            table.attrs["dispersion"]["ndcg_moyen"],
            table.attrs["dispersion"]["ndcg_ecart_type"],
            table.attrs["dispersion"]["amplitude"],
        )
        return table

    def _refit_fold(
        self,
        X: pd.DataFrame,
        y: pd.Series | Sequence[Any],
        fold: pd.DataFrame,
        start: int,
        stop: int,
        model_factory: Any,
    ) -> pd.DataFrame:
        """Retrain on everything preceding a fold and rescore that fold.

        Args:
            X: Preprocessed features, in the same order as the unsorted diagnostic frame.
            y: Observed relevance.
            fold: Chronologically ordered fold rows (index maps back to ``X``).
            start: First row of the fold in chronological order.
            stop: Row after the last one of the fold.
            model_factory: Callable returning an unfitted model exposing ``fit`` / ``predict``.

        Returns:
            The fold with a refreshed ``y_pred`` column.
        """
        positions = fold.index.to_numpy()
        train_positions = positions[:start] if start else np.array([], dtype=int)
        if len(train_positions) < 50:
            logger.warning(
                "Fold ending at row {} has fewer than 50 training rows; reusing the fitted model",
                stop,
            )
            return fold
        try:
            candidate = model_factory()
            fitted = candidate.fit(X.iloc[train_positions], np.asarray(y)[train_positions])
            refreshed = fold.copy()
            if bool(getattr(fitted, "supports_proba", False)):
                probabilities = np.asarray(
                    fitted.predict_proba(X.iloc[positions]), dtype="float64"
                )
                refreshed["y_pred"] = (
                    probabilities[:, -1].ravel() if probabilities.ndim == 2 else probabilities.ravel()
                )
            else:
                refreshed["y_pred"] = np.asarray(
                    fitted.predict(X.iloc[positions]), dtype="float64"
                ).ravel()
            return refreshed
        except Exception as exc:  # noqa: BLE001 - un pli en échec ne doit pas tuer l'évaluation
            logger.warning("Refit backtest failed on fold ending at {}: {}", stop, exc)
            return fold

    # ------------------------------------------------------------------ importance ------
    def feature_importance(
        self,
        X: pd.DataFrame,
        *,
        y: pd.Series | Sequence[Any] | None = None,
        groups: Sequence[Any] | None = None,
        max_features: int = 20,
        repeats: int = 3,
        sample_size: int = 4000,
    ) -> pd.DataFrame:
        """Rank the features by how much the ranking quality collapses without them.

        L'importance native d'un ensemble d'arbres mesure la réduction d'impureté, qui n'a aucun
        rapport avec la qualité d'un classement. Ce module calcule donc une **importance par
        permutation groupée** : chaque colonne est mélangée, le NDCG@K est recalculé utilisateur
        par utilisateur, et l'écart au NDCG intact mesure la dépendance réelle du moteur à cette
        colonne. Un moteur dont le NDCG s'effondre quand on mélange la popularité de l'article
        est, en pratique, un moteur de popularité habillé en modèle.

        Args:
            X: Preprocessed features.
            y: Observed relevance (required by the permutation).
            groups: User identifier per row, required to score per user.
            max_features: Maximum number of features reported.
            repeats: Permutation repeats averaged per feature.
            sample_size: Row budget; larger inputs are sampled without replacement.

        Returns:
            A frame ``feature | importance | method`` sorted by descending importance. An empty
            frame is returned when the permutation cannot run.
        """
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
        if y is None or groups is None:
            logger.warning("Feature importance skipped: relevance or user groups are missing")
            return pd.DataFrame(columns=["feature", "importance", "method"])
        truth = np.asarray(pd.Series(y).to_numpy(), dtype="float64").ravel()
        keys = np.asarray(pd.Series(groups).to_numpy()).ravel()
        if len(truth) != len(frame) or len(keys) != len(frame):
            logger.warning("Feature importance skipped: inputs are not aligned")
            return pd.DataFrame(columns=["feature", "importance", "method"])

        if len(frame) > sample_size:
            rng = np.random.default_rng(self.settings.random_seed)
            keep = rng.choice(len(frame), size=sample_size, replace=False)
            frame = frame.iloc[keep].reset_index(drop=True)
            truth = truth[keep]
            keys = keys[keep]

        cutoff = self.settings.top_k

        def _ndcg_of(features: pd.DataFrame) -> float:
            scores = self._score(features)
            buckets: list[float] = []
            work = pd.DataFrame({"g": keys, "t": truth, "s": scores})
            for _, group in work.groupby("g", observed=True, sort=False):
                if group["t"].sum() <= 0:
                    continue
                order = np.argsort(-group["s"].to_numpy(dtype="float64"), kind="stable")
                buckets.append(_ndcg(group["t"].to_numpy(dtype="float64"), order, cutoff))
            return _safe_mean(buckets)

        try:
            reference = _ndcg_of(frame)
        except Exception as exc:  # noqa: BLE001 - l'importance est un diagnostic, pas un bloquant
            logger.warning("Feature importance skipped: baseline scoring failed ({})", exc)
            return pd.DataFrame(columns=["feature", "importance", "method"])

        rng = np.random.default_rng(self.settings.random_seed)
        rows: list[dict[str, Any]] = []
        for column in frame.columns:
            drops: list[float] = []
            for _ in range(max(int(repeats), 1)):
                shuffled = frame.copy()
                shuffled[column] = rng.permutation(shuffled[column].to_numpy())
                try:
                    drops.append(reference - _ndcg_of(shuffled))
                except Exception:  # noqa: BLE001 - une colonne récalcitrante ne bloque pas le reste
                    continue
            if not drops:
                continue
            rows.append(
                {
                    "feature": str(column),
                    "importance": round(float(np.mean(drops)), 6),
                    "method": "permutation_ndcg",
                }
            )
        if not rows:
            return pd.DataFrame(columns=["feature", "importance", "method"])
        table = pd.DataFrame(rows).sort_values("importance", ascending=False)
        return table.head(int(max_features)).reset_index(drop=True)

    # ------------------------------------------------------------------ ceiling ---------
    def generation_reference(self) -> dict[str, Any]:
        """Read the reachable ceiling published by the data generator, when available.

        Le générateur connaît la probabilité de pertinence *avant* bruit : il peut donc classer
        parfaitement tout ce qui est connaissable. Ce classement oracle définit le **plafond
        atteignable** — aucun modèle réel ne peut le dépasser, puisque le reste est du bruit
        irréductible. Le lire dans les métadonnées de génération permet de situer le modèle dans
        une fourchette honnête (aléatoire, popularité, plafond) au lieu de commenter un chiffre
        absolu dénué de référence.

        Returns:
            A mapping with the published baselines and ceiling, empty when the metadata file is
            absent (données réelles, par exemple).
        """
        candidates = (
            self.paths.raw_dir / GENERATION_METADATA,
            self.paths.data_dir / GENERATION_METADATA,
        )
        for path in candidates:
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Generation metadata unreadable at {}: {}", path, exc)
                return {}
            # The generator publishes its references at the root of the metadata: the oracle
            # ceiling (ranking by the pre-noise relevance probability), the two trivial baselines
            # and the headroom between them. Everything numeric is kept, so a generator that
            # publishes more references needs no change here.
            reference: dict[str, Any] = {
                str(key): float(value)
                for key, value in payload.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            if not any(key.startswith("ndcg_") for key in reference):
                logger.warning(
                    "Aucune référence NDCG publiée dans {} : le plafond ne sera pas affiché",
                    path.name,
                )
                return {}
            reference["source"] = str(path)
            logger.info(
                "Plafond atteignable lu dans {} | NDCG oracle {} (popularité {}, aléatoire {})",
                path.name,
                reference.get("ndcg_ceiling_oracle", "n/d"),
                reference.get("ndcg_baseline_popularity", "n/d"),
                reference.get("ndcg_baseline_random", "n/d"),
            )
            return reference
        return {}

    # ------------------------------------------------------------------ verdicts --------
    def _verdicts(
        self,
        metrics: Mapping[str, float],
        baselines: pd.DataFrame,
        coverage: float,
        cold_ratio: float,
        out_of_stock_share: float,
        spread: float = float("nan"),
    ) -> dict[str, Any]:
        """Turn every measured quantity into an explicit pass/fail verdict.

        Un rapport qui liste des chiffres sans dire s'ils sont bons ou mauvais laisse la décision
        au lecteur. Chaque objectif du projet est donc confronté à son seuil, avec le chiffre
        mesuré, l'écart et un verdict lisible — y compris les objectifs non pertinencielles
        (couverture, démarrage à froid, disponibilité), qui sont précisément ceux qu'on oublie de
        surveiller.

        Args:
            metrics: Configured metric values.
            baselines: Baseline comparison table.
            coverage: Share of the catalogue exposed.
            cold_ratio: Cold-user recall divided by global recall.
            out_of_stock_share: Share of published slots pointing at unavailable items.
            spread: NDCG amplitude between chronological folds.

        Returns:
            A mapping with the per-objective verdicts and an overall status.
        """
        cutoff = self.settings.top_k
        ndcg = float(metrics.get(self.primary_metric, float("nan")))
        precision = float(metrics.get(f"precision_at_{cutoff}", float("nan")))
        recall = float(metrics.get(f"recall_at_{cutoff}", float("nan")))
        gain = float("nan")
        if not baselines.empty and POPULARITY_BASELINE in set(baselines["scorer"]):
            reference = baselines.set_index("scorer")
            popularity_ndcg = float(reference.loc[POPULARITY_BASELINE, "ndcg"])
            model_ndcg = float(reference.loc["modele", "ndcg"]) if "modele" in reference.index else ndcg
            gain = (
                (model_ndcg - popularity_ndcg) / popularity_ndcg * 100.0
                if popularity_ndcg > 0
                else float("nan")
            )

        checks = [
            (
                f"NDCG@{cutoff} au-dessus du seuil",
                ndcg,
                0.45,
                ">=",
                "Le classement publié est suffisamment proche de l'ordre idéal.",
            ),
            (
                f"Precision@{cutoff} au-dessus du seuil",
                precision,
                0.22,
                ">=",
                "Au moins un emplacement sur cinq publié est pertinent.",
            ),
            (
                f"Recall@{cutoff} au-dessus du seuil",
                recall,
                0.55,
                ">=",
                "La majorité de la pertinence disponible est capturée.",
            ),
            (
                "Gain NDCG sur la popularite",
                gain,
                25.0,
                ">=",
                "Le modèle apporte plus que le tri par audience, qui est gratuit.",
            ),
            (
                "Couverture catalogue",
                coverage,
                float(self.settings.coverage_min),
                ">=",
                "Le moteur n'abandonne pas la longue traîne du catalogue.",
            ),
            (
                "Rappel utilisateurs froids / global",
                cold_ratio,
                0.60,
                ">=",
                "Les nouveaux venus ne sont pas servis au rabais.",
            ),
            (
                "Emplacements perdus en rupture",
                out_of_stock_share,
                0.15,
                "<=",
                "Publier un article indisponible coûte une place et une déception.",
            ),
            (
                "Amplitude du NDCG entre périodes",
                spread,
                float(self.settings.backtest_spread_max),
                "<=",
                "Une qualité qui dépend de la période impose un recalibrage saisonnier.",
            ),
        ]
        verdicts: list[dict[str, Any]] = []
        for label, value, threshold, operator, rationale in checks:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                verdicts.append(
                    {
                        "critere": label,
                        "mesure": float("nan"),
                        "seuil": float(threshold),
                        "operateur": operator,
                        "verdict": "non_mesurable",
                        "lecture": rationale,
                    }
                )
                continue
            passed = value >= threshold if operator == ">=" else value <= threshold
            verdicts.append(
                {
                    "critere": label,
                    "mesure": round(float(value), 4),
                    "seuil": float(threshold),
                    "operateur": operator,
                    "verdict": "respecte" if passed else "non_respecte",
                    "lecture": rationale,
                }
            )
        measurable = [item for item in verdicts if item["verdict"] != "non_mesurable"]
        respected = sum(1 for item in measurable if item["verdict"] == "respecte")
        status = (
            "conforme"
            if measurable and respected == len(measurable)
            else ("partiel" if respected else "non_conforme")
        )
        return {
            "objectifs": verdicts,
            "statut": status,
            "respectes": respected,
            "mesurables": len(measurable),
        }

    # ------------------------------------------------------------------ evaluation ------
    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | Sequence[Any],
        *,
        split: str = "test",
        context: pd.DataFrame | None = None,
    ) -> EvaluationResult:
        """Run the complete ranking evaluation on one split.

        Args:
            X: Preprocessed features, one row per (user, candidate) pair.
            y: Observed relevance for each candidate.
            split: Split name, used for labelling and logging.
            context: Enriched split rows carrying the user and item identifiers plus the segment
                columns. Required: per-user metrics are meaningless without the user column.

        Returns:
            The full :class:`EvaluationResult`.

        Raises:
            RuntimeError: When the model is not fitted.
            ValueError: When the context is missing or misaligned.
        """
        self.model.check_is_fitted()
        features = self.model.align_features(X)
        score = self._score(features)
        frame = self._diagnostic_frame(y, score, context)
        cutoff = self.settings.top_k
        logger.info(
            "Evaluation ranking | split={} lignes={} utilisateurs={} prevalence={:.4f}",
            split,
            len(frame),
            frame[self.settings.group_column].nunique(),
            float(frame["y_true"].mean()),
        )

        metrics = self._configured_metrics(frame)
        detailed = self.ranking_metrics(frame, "y_pred", top_k=cutoff)
        for key, value in detailed.items():
            metrics.setdefault(key, value)

        per_cutoff = self.per_cutoff(frame)
        baselines = self.compare_to_baselines(frame)
        per_segment = self._per_segment(frame)
        cold_ratio = self.cold_start_ratio(frame, per_segment)
        published = self.published_topk(frame)
        margin = self.margin_trade_off(frame, published)
        coverage, concentration = self.catalog_exposure(frame, "y_pred")
        errors = self.worst_misrankings(frame)
        curves = self.curves(frame)
        importance = self.feature_importance(
            features,
            y=frame["y_true"].to_numpy(),
            groups=frame[self.settings.group_column].to_numpy(),
        )
        reference = self.generation_reference()
        temporal = self.backtest(context=context, score=score, y=frame["y_true"].to_numpy())

        out_of_stock_share = (
            float((~published["en_stock"].astype(bool)).mean())
            if not published.empty and "en_stock" in published.columns
            else float("nan")
        )
        popularity_bias = self._popularity_bias(frame, published)
        verdicts = self._verdicts(
            metrics,
            baselines,
            coverage,
            cold_ratio,
            out_of_stock_share,
            spread=float(dict(temporal.attrs.get("dispersion", {})).get("amplitude", float("nan"))),
        )

        extras: dict[str, Any] = {
            "top_k": cutoff,
            "cutoffs": list(self.settings.cutoffs),
            "catalog_coverage": round(float(coverage), 4) if coverage == coverage else float("nan"),
            "herfindahl_top_k": round(float(concentration), 4)
            if concentration == concentration
            else float("nan"),
            "out_of_stock_published_share": round(out_of_stock_share, 4)
            if out_of_stock_share == out_of_stock_share
            else float("nan"),
            "popularity_bias_share": round(popularity_bias, 4)
            if popularity_bias == popularity_bias
            else float("nan"),
            "cold_start_recall_ratio": cold_ratio,
            "users_evaluated": int(detailed["users_evaluated"]),
            "users_total": int(frame[self.settings.group_column].nunique()),
            "prevalence": round(float(frame["y_true"].mean()), 4),
            "candidates_per_user": round(
                len(frame) / max(int(frame[self.settings.group_column].nunique()), 1), 2
            ),
            "margin_trade_off": margin,
            "verdicts": verdicts,
            "generation_reference": {
                key: value
                for key, value in reference.items()
                if not isinstance(value, str) or key == "source"
            },
            "ceiling_share": self._ceiling_share(metrics.get(self.primary_metric), reference),
            "dispersion_backtest": dict(temporal.attrs.get("dispersion", {})),
            "model": self.model.summary(),
        }

        result = EvaluationResult(
            task=str(self.task),
            split=split,
            primary_metric=self.primary_metric,
            metrics={key: float(value) for key, value in metrics.items()},
            predictions=published,
            labels=[
                f"NDCG@{cutoff}",
                f"Precision@{cutoff}",
                f"Recall@{cutoff}",
                f"MAP@{cutoff}",
                f"Hit-rate@{cutoff}",
                "Couverture catalogue",
                "Concentration (Herfindahl)",
            ],
            per_segment=per_segment,
            per_horizon=per_cutoff,
            baselines=baselines,
            backtest=temporal,
            errors=errors,
            curves=curves,
            feature_importance=importance,
            extras=extras,
            n_samples=int(len(frame)),
        )
        logger.info(
            "Résultat | {}={} couverture={:.1%} ratio froid={:.2f} statut={}",
            self.primary_metric,
            round(result.primary_value, 4),
            result.coverage if result.coverage == result.coverage else float("nan"),
            cold_ratio if cold_ratio == cold_ratio else float("nan"),
            verdicts["statut"],
        )
        return result

    def _popularity_bias(self, frame: pd.DataFrame, published: pd.DataFrame) -> float:
        """Return the share of published slots taken by the top decile of item popularity.

        Args:
            frame: Diagnostic frame.
            published: Published recommendations.

        Returns:
            The share, or NaN when popularity is unavailable.
        """
        popularity_column = self.settings.popularity_column
        if popularity_column not in frame.columns or published.empty:
            return float("nan")
        popularity = pd.to_numeric(frame[popularity_column], errors="coerce")
        if popularity.notna().sum() == 0:
            return float("nan")
        threshold = float(popularity.quantile(0.90))
        hot_items = set(frame.loc[popularity >= threshold, self.settings.item_column].unique())
        item_column = self.settings.item_column
        if item_column not in published.columns or not hot_items:
            return float("nan")
        return float(published[item_column].isin(hot_items).mean())

    @staticmethod
    def _ceiling_share(value: Any, reference: Mapping[str, Any]) -> float:
        """Return the share of the reachable ceiling captured by the model.

        Args:
            value: Measured primary metric.
            reference: Generation metadata, holding the oracle ceiling.

        Returns:
            The ratio, or NaN when no ceiling is published.
        """
        if value is None or not reference:
            return float("nan")
        ceiling_keys = [key for key in reference if "oracle" in key and key.startswith("ndcg")]
        if not ceiling_keys:
            return float("nan")
        # `ndcg_ceiling_oracle` est la clé publiée par le générateur de cette famille ; le tri
        # alphabétique sert uniquement de repli déterministe si une autre clé oracle apparaît.
        ceiling_key = (
            "ndcg_ceiling_oracle"
            if "ndcg_ceiling_oracle" in reference
            else sorted(ceiling_keys)[0]
        )
        ceiling = float(reference[ceiling_key])
        measured = float(value)
        return round(measured / ceiling, 4) if ceiling > 0 else float("nan")

    # ------------------------------------------------------------------ persistence -----
    def save_metrics(self, result: EvaluationResult, path: Path | str | None = None) -> Path:
        """Persist the evaluation to ``metrics/evaluation_metrics.json``.

        Args:
            result: The evaluation to persist.
            path: Destination override.

        Returns:
            The written path.
        """
        target = Path(path) if path else self.paths.metrics_dir / "evaluation_metrics.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = result.to_dict()
        payload["model"] = self.model.summary()
        payload["settings"] = {
            "top_k": self.settings.top_k,
            "cutoffs": list(self.settings.cutoffs),
            "group_column": self.settings.group_column,
            "item_column": self.settings.item_column,
            "coverage_min": self.settings.coverage_min,
        }
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        logger.info("Métriques d'évaluation écrites dans {}", target)
        return target
