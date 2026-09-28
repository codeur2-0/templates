"""Synthetic data generator for the recommendation (ranking) family.

Ce module ne génère **pas** une table d'interactions brutes : il génère un *journal de candidats
scorés*, c'est-à-dire la table sur laquelle un modèle de classement est réellement entraîné en
production. Chaque ligne est un couple ``(utilisateur, article candidat)`` observé lors d'une
session, décrit uniquement par l'information disponible **à cet instant**, avec la pertinence
constatée ensuite comme cible.

Pourquoi ce format plutôt qu'une matrice utilisateur-article :

* la modalité tabulaire du dépôt (splits chronologiques, preprocessing ajusté sur le train
  uniquement, contrats Pandera, artefacts rechargables) s'applique sans adaptation ;
* c'est le format réel d'un ``learning to rank`` industriel : un moteur recall sélectionne des
  candidats, un modèle de classement les ordonne, et le journal de bord conserve les deux ;
* l'échantillonnage des négatifs — le point délicat de l'évaluation d'un recommender — devient
  explicite et paramétrable au lieu d'être implicite dans une matrice creuse.

La pertinence est produite par un **modèle latent** : un logit combinant l'affinité catégorie,
l'écart de prix, l'intention récente, le signal collaboratif, la qualité de la fiche, un léger
effet de popularité et un bruit gaussien, puis un tirage de Bernoulli. Deux conséquences assumées :

1. aucun modèle ne peut atteindre un NDCG@10 de 1,0 — le plafond atteignable est mesuré à la
   génération et publié dans les métadonnées, pour ne pas poursuivre un bruit résiduel ;
2. les signaux sont hiérarchisés comme dans un vrai catalogue (l'affinité individuelle domine,
   la popularité n'apporte qu'un peu), ce qui rend la comparaison avec la baseline popularité
   instructive plutôt que triviale.

Les identifiants ``user_id`` et ``item_id`` sont produits mais ne doivent jamais devenir des
features : un modèle qui mémorise un identifiant ne sait rien dire d'un utilisateur ou d'un
article jamais vu. Le schéma Pandera les déclare ``identifier`` / ``group`` et la configuration
les place dans les colonnes ignorées.

Exemples :
    $ python -c "from src.data.generators import SyntheticDataGenerator as G; print(G().generate().shape)"
    $ python scripts/generate_data.py data.n_samples=6000
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.utils.io import write_json, write_table_multiple
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

#: Nom de fichier par défaut du jeu généré (surchargé par ``dataset_name``).
DEFAULT_DATASET_NAME = "ecommerce_candidate_impressions"

#: Catégories du catalogue, avec leur part et leurs propriétés moyennes.
CATEGORIES: tuple[str, ...] = (
    "mode",
    "maison",
    "high_tech",
    "sport",
    "beaute",
    "alimentaire",
    "jouet",
    "jardin",
)

#: Part de chaque catégorie dans le catalogue (somme = 1).
CATEGORY_SHARE: tuple[float, ...] = (0.19, 0.15, 0.12, 0.11, 0.10, 0.13, 0.10, 0.10)

#: Prix médian par catégorie (EUR) — les distributions sont log-normales autour de cette valeur.
CATEGORY_PRICE_MEDIAN: tuple[float, ...] = (45.0, 60.0, 220.0, 70.0, 28.0, 18.0, 32.0, 55.0)

#: Marge brute moyenne par catégorie (%) — la high_tech marge peu, la beauté marge fort.
CATEGORY_MARGIN_MEAN: tuple[float, ...] = (42.0, 38.0, 18.0, 35.0, 52.0, 22.0, 40.0, 36.0)

#: Gammes de prix préférées des utilisateurs.
PRICE_BANDS: tuple[str, ...] = ("entree", "milieu", "premium")

#: Canaux d'achat.
CHANNELS: tuple[str, ...] = ("web", "mobile", "app")

#: Régions de livraison.
REGIONS: tuple[str, ...] = ("nord", "sud", "est", "ouest", "idf")

#: Nombre de catégories du catalogue (utile pour les vecteurs d'affinité).
N_CATEGORIES: int = len(CATEGORIES)


def _sigmoid(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Return the logistic function, computed without overflow.

    Args:
        values: Logits.

    Returns:
        Probabilities in ``(0, 1)``.
    """
    return 0.5 * (1.0 + np.tanh(np.asarray(values, dtype="float64") / 2.0))


def _dcg(gains: npt.NDArray[np.float64]) -> float:
    """Return the discounted cumulative gain of an ordered gain vector.

    Args:
        gains: Relevance gains, best ranked first.

    Returns:
        The DCG value.
    """
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2, dtype="float64"))
    return float((gains * discounts).sum())


def ndcg_at_k_from_groups(
    frame: pd.DataFrame,
    *,
    group_column: str,
    truth_column: str,
    score_column: str,
    top_k: int,
) -> float:
    """Return the mean NDCG@K over groups that hold at least one relevant row.

    Fonction de référence utilisée à la génération pour publier le plafond atteignable et la
    valeur des baselines dans les métadonnées. Elle est volontairement indépendante du code
    d'évaluation du projet : deux implémentations qui se rejoignent sur les mêmes chiffres
    valident le calcul.

    Args:
        frame: Rows to score, one row per (group, candidate).
        group_column: Column holding the group key (the user).
        truth_column: Binary relevance column.
        score_column: Score column, higher is better.
        top_k: Number of recommendations retained per group.

    Returns:
        The mean NDCG@K, or ``nan`` when no group has a relevant row.
    """
    scores: list[float] = []
    for _, group in frame.groupby(group_column, observed=True):
        truth = group[truth_column].to_numpy(dtype="float64")
        if truth.sum() <= 0:
            continue
        ranking = group[score_column].to_numpy(dtype="float64")
        order = np.argsort(-ranking, kind="stable")
        ideal = float(_dcg(np.sort(truth)[::-1][:top_k]))
        scores.append(_dcg(truth[order][:top_k]) / ideal if ideal > 0 else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


class SyntheticDataGenerator:
    """Generate a reproducible e-commerce candidate-scoring log.

    Le générateur construit d'abord un **catalogue** et une **base utilisateurs** cohérents
    (prix, marge, note, ancienneté d'un côté ; historique, appétence, gamme de prix de l'autre),
    puis tire pour chaque utilisateur un jeu de candidats à une date de session donnée, calcule
    les signaux croisés disponibles à cet instant, et en déduit la pertinence par un modèle
    latent bruité.

    Tout est déterministe à graine fixée : la même graine produit exactement les mêmes lignes,
    les mêmes identifiants et les mêmes métriques de référence.

    Attributes:
        n_samples: Requested number of rows (rounded to a multiple of ``candidates_per_user``).
        seed: Random seed driving every draw.
        dataset_name: Base name of the written files.
        output_dir: Directory receiving the files.
        formats: Formats to write.
    """

    #: Options acceptées depuis la configuration ``data`` (filtrées par ``_generator_options``).
    SUPPORTED_OPTIONS: ClassVar[tuple[str, ...]] = (
        "n_users",
        "catalog_size",
        "candidates_per_user",
        "positive_rate",
        "window_days",
        "cold_user_share",
        "out_of_stock_rate",
        "never_viewed_share",
        "noise_scale",
        "popularity_weight",
        "appeal_weight",
        "start",
    )

    #: Nombre maximal de passes de réparation des listes de candidats. Une seule passe suffit en
    #: pratique (le tirage de remplacement est sans remise) ; la borne garantit la terminaison.
    MAX_DEDUPLICATION_PASSES: ClassVar[int] = 4

    #: Colonnes produites, dans l'ordre du schéma brut (``RawDataSchema`` est strict).
    COLUMNS: ClassVar[tuple[str, ...]] = (
        "sample_id",
        "user_id",
        "item_id",
        "session_date",
        "user_tenure_days",
        "user_orders_12m",
        "user_spend_eur_12m",
        "user_sessions_30d",
        "user_distinct_categories_12m",
        "user_avg_basket_eur",
        "user_typical_price_eur",
        "user_return_rate",
        "user_price_band_pref",
        "user_channel",
        "user_region",
        "user_is_member",
        "item_category",
        "item_price_eur",
        "item_rating_avg",
        "item_reviews_count",
        "item_stock_units",
        "item_age_days",
        "item_margin_pct",
        "item_views_7d",
        "item_conversion_rate_30d",
        "item_is_promoted",
        "user_category_affinity",
        "user_brand_affinity",
        "price_gap_pct",
        "days_since_last_view",
        "user_item_views_30d",
        "similar_users_buy_rate",
        "relevance",
    )

    def __init__(
        self,
        *,
        n_samples: int = 36000,
        seed: int = 42,
        dataset_name: str | None = None,
        output_dir: Path | str | None = None,
        formats: Sequence[str] = ("parquet", "csv"),
        n_users: int | None = None,
        catalog_size: int = 900,
        candidates_per_user: int = 30,
        positive_rate: float = 0.14,
        window_days: int = 420,
        cold_user_share: float = 0.29,
        out_of_stock_rate: float = 0.07,
        never_viewed_share: float = 0.62,
        noise_scale: float = 0.95,
        popularity_weight: float = 0.25,
        appeal_weight: float = 1.30,
        start: str = "2024-09-02",
    ) -> None:
        """Store the generation parameters.

        Args:
            n_samples: Requested row count. The generator rounds it to a whole number of
                candidate lists so that every user is scored on the same number of candidates —
                a requirement for Precision@K and Recall@K to be comparable across users.
            seed: Random seed (fully determines the output).
            dataset_name: Base file name (defaults to :data:`DEFAULT_DATASET_NAME`).
            output_dir: Directory receiving the files (defaults to ``data/raw``).
            formats: Formats to write (``parquet`` and/or ``csv``).
            n_users: Advisory number of users. The effective value is always derived from
                ``n_samples`` and ``candidates_per_user``, because the row count is the contract
                checked by the tests and the generation pipeline; a conflicting value is ignored
                with a warning.
            catalog_size: Number of distinct items in the catalogue.
            candidates_per_user: Candidates scored per user and per session. Thirty candidates
                for a top-10 means the model really has to choose: with twelve candidates a
                top-10 is nearly the whole list, NDCG@10 saturates and even a random scorer looks
                decent, which destroys the comparison with the baselines.
            positive_rate: Target share of relevant candidates. The logit intercept is calibrated
                by bisection so that the generated frame matches this rate.
            window_days: Length of the observation window in days (~14 months by default, so that
                a yearly seasonality and a year-end peak are both visible).
            cold_user_share: Share of users with at most one order in the last 12 months.
            out_of_stock_rate: Share of candidates with zero stock (never publishable).
            never_viewed_share: Share of candidates the user has never viewed — the engine mostly
                explores rather than exploits.
            noise_scale: Standard deviation of the latent logit noise. It sets the reachable
                ceiling: the oracle NDCG@10 published in the metadata is computed with it.
            popularity_weight: Weight of the *observed* popularity term (7-day views) in the
                latent logit. Kept small on purpose: a strong weight would let the popularity
                baseline win on its own, and the comparison would teach nothing.
            appeal_weight: Weight of the latent item appeal. This is what makes the popularity
                baseline strong yet beatable: appeal drives both the view count and the relevance,
                so ranking by views is informative, but only as a noisy proxy of a factor the model
                can reconstruct from views, rating and conversion together.
            start: First session date of the window.
        """
        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self.dataset_name = str(dataset_name or DEFAULT_DATASET_NAME)
        self.output_dir = Path(output_dir) if output_dir is not None else Path("data/raw")
        self.formats: tuple[str, ...] = tuple(formats)
        self.catalog_size = int(catalog_size)
        self.candidates_per_user = max(2, int(candidates_per_user))
        # `data.n_samples` est le contrat du projet : c'est lui que vérifient les tests et le
        # pipeline de génération. Le nombre d'utilisateurs en **découle**, et non l'inverse — un
        # `n_users` explicite qui imposerait un autre nombre de lignes rendrait ce contrat
        # impossible à tenir. Il est donc ignoré quand il entre en conflit, avec un avertissement
        # qui dit exactement ce qui a été retenu et pourquoi.
        derived_users = max(4, round(self.n_samples / self.candidates_per_user))
        if n_users is not None and int(n_users) != derived_users:
            logger.warning(
                "n_users={} ignoré : data.n_samples={} avec {} candidats par utilisateur impose "
                "{} utilisateurs. Le nombre de lignes est le contrat vérifié par les tests, le "
                "nombre d'utilisateurs en découle.",
                int(n_users),
                self.n_samples,
                self.candidates_per_user,
                derived_users,
            )
        self.n_users = derived_users
        self.positive_rate = float(positive_rate)
        self.window_days = int(window_days)
        self.cold_user_share = float(cold_user_share)
        self.out_of_stock_rate = float(out_of_stock_rate)
        self.never_viewed_share = float(never_viewed_share)
        self.noise_scale = float(noise_scale)
        self.popularity_weight = float(popularity_weight)
        self.appeal_weight = float(appeal_weight)
        self.start = pd.Timestamp(start)

        #: Probabilité latente du dernier ``generate()`` : sert uniquement à publier le plafond
        #: atteignable dans les métadonnées. Elle n'est **jamais** écrite dans le jeu de données,
        #: ce qui serait une fuite — le modèle doit apprendre l'affinité, pas la recopier.
        self._latent_probability: npt.NDArray[np.float64] | None = None
        self._latent_frame: pd.DataFrame | None = None
        self._catalogue: pd.DataFrame | None = None
        self._users: pd.DataFrame | None = None

    @property
    def row_count(self) -> int:
        """Return the exact number of rows produced for the current parameters."""
        return self.n_users * self.candidates_per_user

    # ----------------------------------------------------------------------------------
    # Catalogue et base utilisateurs
    # ----------------------------------------------------------------------------------
    def _build_catalogue(self, rng: np.random.Generator) -> pd.DataFrame:
        """Build the item catalogue with coherent, category-driven attributes.

        L'ordre des tirages suit les dépendances réelles : la catégorie fixe le prix médian et la
        marge, l'**attrait latent** fixe l'audience, la note et le taux de conversion, l'ancienneté
        fixe le volume d'avis. Rien n'est tiré indépendamment de ce qu'il est censé expliquer.

        Args:
            rng: Seeded random generator.

        Returns:
            A frame with one row per catalogue item.
        """
        category_index = rng.choice(
            N_CATEGORIES, size=self.catalog_size, p=np.asarray(CATEGORY_SHARE)
        )
        medians = np.asarray(CATEGORY_PRICE_MEDIAN, dtype="float64")[category_index]
        margins = np.asarray(CATEGORY_MARGIN_MEAN, dtype="float64")[category_index]

        # Prix log-normal autour de la médiane de catégorie : queue droite réaliste.
        price = medians * np.exp(rng.normal(0.0, 0.45, size=self.catalog_size))
        price = np.clip(price, 1.0, 2500.0)

        # Ancienneté : exponentielle — beaucoup de nouveautés, une longue traîne d'anciens.
        age = np.minimum(rng.exponential(300.0, size=self.catalog_size), 2200.0).astype(np.int64)

        # Marge : normale par catégorie, bornée.
        margin = np.clip(rng.normal(margins, 7.5), 0.0, 70.0)

        # Attrait latent : un seul facteur explique qu'une référence soit à la fois très vue, bien
        # notée, convertisseuse et largement pertinente. Sans lui, la popularité ne serait qu'un
        # bruit indépendant de la cible et la baseline popularité tomberait au niveau du tirage
        # aléatoire — ce qui est faux dans un vrai catalogue, où les best-sellers le sont parce
        # qu'ils plaisent au plus grand nombre. L'attrait reste partiellement observable (audience,
        # note et conversion en sont des proxies bruités), donc un modèle qui combine plusieurs
        # proxies bat le tri par popularité sans le recopier.
        appeal = rng.beta(2.2, 2.0, size=self.catalog_size)

        # Note moyenne : cohérente avec l'attrait, mais bruitée — la note n'est pas l'attrait.
        rating = np.clip(
            rng.beta(9.0, 2.4, size=self.catalog_size) * 2.2 + 1.6 + 1.35 * (appeal - 0.5),
            1.0,
            5.0,
        )

        # Avis : loi puissance, fonction de l'ancienneté et de la note (preuve sociale).
        base_reviews = rng.pareto(1.35, size=self.catalog_size) + 1.0
        reviews = base_reviews * (0.35 + age / 900.0) * (0.55 + (rating - 1.6) / 3.4)
        reviews = np.clip(reviews, 0.0, 6000.0).astype(np.int64)

        # Audience 7 jours : loi puissance modulée par l'attrait — quelques best-sellers, une
        # longue traîne invisible, et un lien réel entre audience et pertinence moyenne.
        popularity = rng.pareto(1.15, size=self.catalog_size) + 1.0
        views = popularity * (0.22 + 2.60 * appeal) * (18.0 + 0.06 * age)
        views = np.clip(views, 0.0, 40000.0).astype(np.int64)

        # Taux de conversion : bêta, corrélé à l'attrait, à la note et à la preuve sociale.
        conversion = rng.beta(2.2, 45.0, size=self.catalog_size)
        conversion = (
            conversion
            * (0.35 + 1.25 * appeal)
            * (0.70 + (rating - 1.6) / 4.0)
            * (0.75 + np.log1p(reviews) / 11.0)
        )
        conversion = np.clip(conversion, 0.0, 0.6)

        promoted = rng.random(self.catalog_size) < 0.18

        return pd.DataFrame(
            {
                "item_id": [f"I-{index:05d}" for index in range(self.catalog_size)],
                "category_index": category_index,
                "item_category": np.asarray(CATEGORIES)[category_index],
                "item_price_eur": np.round(price, 2),
                "item_rating_avg": np.round(rating, 2),
                "item_reviews_count": reviews,
                "item_age_days": age,
                "item_margin_pct": np.round(margin, 1),
                "item_views_7d": views,
                "item_conversion_rate_30d": np.round(conversion, 4),
                "item_is_promoted": promoted,
                # Affinité "marque" latente de l'article : sert à produire le signal croisé.
                "brand_appeal": np.round(rng.beta(2.0, 5.0, size=self.catalog_size), 4),
                # Attrait latent : jamais écrit dans le jeu de données, seulement utilisé par le
                # logit de pertinence. L'exposer serait une fuite.
                "item_appeal": np.round(appeal, 4),
            }
        )

    def _build_users(self, rng: np.random.Generator) -> pd.DataFrame:
        """Build the user base, including an explicit cold-start segment.

        Args:
            rng: Seeded random generator.

        Returns:
            A frame with one row per user, plus their latent taste vector.
        """
        n_users = self.n_users

        # Ancienneté : loi puissance tronquée — beaucoup de comptes récents.
        tenure = np.clip(rng.pareto(1.6, size=n_users) * 110.0, 0.0, 3000.0).astype(np.int64)

        # Commandes 12 mois : le segment froid est imposé explicitement, pas laissé au hasard,
        # sinon sa taille dépendrait de la graine et le critère de réussite deviendrait instable.
        orders = rng.poisson(4.5, size=n_users)
        cold_mask = rng.random(n_users) < self.cold_user_share
        orders[cold_mask] = rng.integers(0, 2, size=int(cold_mask.sum()))
        warm_boost = rng.poisson(3, size=n_users)
        orders = np.where(cold_mask, orders, orders + warm_boost)
        orders = np.clip(orders, 0, 120).astype(np.int64)

        # Dépense : log-normale pilotée par le nombre de commandes.
        basket = np.exp(rng.normal(np.log(65.0), 0.5, size=n_users))
        basket = np.clip(basket, 8.0, 900.0)
        spend = basket * np.maximum(orders, 0) * np.exp(rng.normal(0.0, 0.18, size=n_users))
        spend = np.clip(spend, 0.0, 12000.0)

        # Prix habituel : proche du panier moyen, moins dispersé.
        typical_price = np.clip(basket * np.exp(rng.normal(0.0, 0.22, size=n_users)), 5.0, 900.0)

        sessions = np.clip(rng.poisson(9.0, size=n_users), 0, 200).astype(np.int64)
        distinct_categories = np.clip(
            rng.binomial(N_CATEGORIES, np.clip(0.10 + orders / 60.0, 0.05, 0.85)), 0, N_CATEGORIES
        ).astype(np.int64)
        return_rate = np.clip(rng.beta(1.1, 13.0, size=n_users), 0.0, 1.0)

        # Gamme de prix préférée : cohérente avec le prix habituel, pas tirée au hasard.
        band_index = np.digitize(typical_price, [35.0, 110.0])
        price_band = np.asarray(PRICE_BANDS)[band_index]

        channel = rng.choice(np.asarray(CHANNELS), size=n_users, p=np.asarray([0.22, 0.33, 0.45]))
        region = rng.choice(
            np.asarray(REGIONS), size=n_users, p=np.asarray([0.17, 0.17, 0.16, 0.16, 0.34])
        )
        # L'adhésion au programme de fidélité suit l'activité : corrélée, pas indépendante.
        member_probability = np.clip(0.10 + orders / 26.0, 0.05, 0.92)
        is_member = rng.random(n_users) < member_probability

        # Goût latent : vecteur d'affinité par catégorie, piqué (Dirichlet concentré) pour que
        # chaque utilisateur ait quelques catégories fortes et beaucoup de catégories indifférentes.
        taste = rng.dirichlet(np.full(N_CATEGORIES, 0.55), size=n_users)
        # Appétence à la nouveauté et sensibilité au prix : deux traits qui modulent la pertinence.
        novelty_appetite = np.clip(rng.beta(2.0, 3.0, size=n_users), 0.0, 1.0)
        price_sensitivity = np.clip(rng.beta(2.5, 2.5, size=n_users), 0.0, 1.0)
        brand_loyalty = np.clip(rng.beta(2.0, 4.0, size=n_users), 0.0, 1.0)

        return pd.DataFrame(
            {
                "user_id": [f"U-{index:04d}" for index in range(n_users)],
                "user_tenure_days": tenure,
                "user_orders_12m": orders,
                "user_spend_eur_12m": np.round(spend, 2),
                "user_sessions_30d": sessions,
                "user_distinct_categories_12m": distinct_categories,
                "user_avg_basket_eur": np.round(basket, 2),
                "user_typical_price_eur": np.round(typical_price, 2),
                "user_return_rate": np.round(return_rate, 3),
                "user_price_band_pref": price_band,
                "user_channel": channel,
                "user_region": region,
                "user_is_member": is_member,
                "taste": list(taste),
                "novelty_appetite": np.round(novelty_appetite, 4),
                "price_sensitivity": np.round(price_sensitivity, 4),
                "brand_loyalty": np.round(brand_loyalty, 4),
                "is_cold": cold_mask,
            }
        )

    # ----------------------------------------------------------------------------------
    # Sessions et candidats
    # ----------------------------------------------------------------------------------
    def _session_dates(self, rng: np.random.Generator) -> npt.NDArray[np.int64]:
        """Draw one session date per user, with weekly seasonality and a year-end peak.

        Chaque utilisateur apparaît à une seule date de session. Ce n'est pas une simplification
        de confort : le split chronologique répartit donc des **utilisateurs** entiers entre train
        et test, ce qui mesure en plus la capacité du moteur à classer pour quelqu'un qu'il n'a
        jamais vu. Protocole plus sévère qu'un split par interaction, et plus proche de la
        question « que vaut ce modèle sur un nouveau visiteur ? ».

        Args:
            rng: Seeded random generator.

        Returns:
            Day offsets from ``start``, one per user.
        """
        offsets = np.arange(self.window_days, dtype="float64")
        dates = self.start + pd.to_timedelta(offsets, unit="D")
        weekend = np.where(dates.dayofweek.to_numpy() >= 5, 1.35, 1.0)
        month = dates.month.to_numpy(dtype="float64")
        # Pic de fin d'année (novembre-décembre) et creux d'été, comme un catalogue réel.
        seasonal = 1.0 + 0.55 * np.isin(month, (11, 12)).astype("float64")
        seasonal = seasonal * np.where(np.isin(month, (7, 8)), 0.82, 1.0)
        weights = weekend * seasonal
        weights = weights / weights.sum()
        return rng.choice(offsets.astype(np.int64), size=self.n_users, p=weights)

    def _sample_candidates(
        self, users: pd.DataFrame, catalogue: pd.DataFrame, rng: np.random.Generator
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Draw the candidate list of every user.

        Le mélange reproduit ce que produit une vraie étape de recall : une part personnalisée
        (catégories préférées de l'utilisateur), une part populaire (le pool best-sellers que tout
        moteur ressort) et une part d'exploration aléatoire. Sans la part populaire, la baseline
        popularité serait artificiellement faible et la comparaison perdrait son sens.

        Args:
            users: User base.
            catalogue: Item catalogue.
            rng: Seeded random generator.

        Returns:
            Three aligned arrays of positional indices: user, item, and the sampled slot.
        """
        n_users = self.n_users
        per_user = self.candidates_per_user
        taste = np.stack(users["taste"].to_numpy())
        popularity = catalogue["item_views_7d"].to_numpy(dtype="float64") + 1.0
        popularity_probability = popularity / popularity.sum()
        category_index = catalogue["category_index"].to_numpy()
        items_by_category = [
            np.flatnonzero(category_index == index) for index in range(N_CATEGORIES)
        ]

        # Répartition du mélange : 45 % affinité, 35 % popularité, 20 % exploration.
        slots = np.tile(np.arange(per_user), n_users)
        affinity_slots = slots < round(per_user * 0.45)
        popularity_slots = (slots >= round(per_user * 0.45)) & (slots < round(per_user * 0.80))

        user_index = np.repeat(np.arange(n_users), per_user)
        item_index = np.zeros(user_index.size, dtype="int64")

        # Candidats populaires : tirés une fois pour toutes, indépendants de l'utilisateur.
        popular_draws = rng.choice(
            self.catalog_size, size=int(popularity_slots.sum()), p=popularity_probability
        )
        item_index[popularity_slots] = popular_draws

        # Candidats d'exploration : uniformes sur le catalogue.
        explore_slots = ~(affinity_slots | popularity_slots)
        item_index[explore_slots] = rng.integers(0, self.catalog_size, size=int(explore_slots.sum()))

        # Candidats d'affinité : catégorie tirée selon le goût de l'utilisateur, puis article
        # tiré dans cette catégorie en pondérant par l'audience (un moteur recall ne remonte pas
        # l'article le plus obscur d'une catégorie aimée).
        affinity_positions = np.flatnonzero(affinity_slots)
        for position in affinity_positions:
            user = int(user_index[position])
            weights = taste[user]
            if weights.sum() <= 0:
                weights = np.full(N_CATEGORIES, 1.0 / N_CATEGORIES)
            else:
                weights = weights / weights.sum()
            category = int(rng.choice(N_CATEGORIES, p=weights))
            pool = items_by_category[category]
            if pool.size == 0:  # pragma: no cover - garde-fou, impossible avec un catalogue fourni
                item_index[position] = int(rng.integers(0, self.catalog_size))
                continue
            pool_weights = popularity[pool]
            item_index[position] = int(pool[rng.choice(pool.size, p=pool_weights / pool_weights.sum())])

        item_index = self._deduplicate_per_user(user_index, item_index, rng)
        return user_index, item_index, slots

    def _deduplicate_per_user(
        self,
        user_index: npt.NDArray[np.int64],
        item_index: npt.NDArray[np.int64],
        rng: np.random.Generator,
    ) -> npt.NDArray[np.int64]:
        """Return item indices with no repeated article inside a single user list.

        Les trois strates de candidats (affinité, popularité, exploration) sont tirées
        indépendamment : sur un catalogue de quelques centaines de références, deux strates
        retombent régulièrement sur le même article. Sans réparation, la même liste contiendrait
        deux fois la même référence — le moteur publierait un doublon, et surtout la cible
        deviendrait **contradictoire** : deux lignes aux entrées identiques mais tirées
        séparément, donc parfois l'une pertinente et l'autre non. Le modèle devrait alors prédire
        la moyenne de deux étiquettes incompatibles, ce qui abaisserait le plafond atteignable
        sans qu'aucune feature puisse l'expliquer.

        La réparation conserve la première occurrence de chaque article (la strate qui l'a produit
        garde son sens) et retire les suivantes parmi les références absentes de la liste. Un seul
        passage suffit : le tirage de remplacement est sans remise et hors des articles déjà
        choisis. La boucle est bornée pour rester déterministe quoi qu'il arrive.

        Args:
            user_index: Positional user index of every row.
            item_index: Positional item index of every row, to repair in place.
            rng: Seeded random generator.

        Returns:
            The repaired item index array (same shape, same order).
        """
        repaired = item_index.copy()
        catalogue_indices = np.arange(self.catalog_size)
        collisions = 0
        for user in range(int(user_index.max()) + 1 if user_index.size else 0):
            positions = np.flatnonzero(user_index == user)
            if positions.size < 2:
                continue
            for _ in range(self.MAX_DEDUPLICATION_PASSES):
                chosen = repaired[positions]
                codes, inverse = np.unique(chosen, return_inverse=True)
                if codes.size == chosen.size:
                    break
                first_seen: dict[int, int] = {}
                duplicates: list[int] = []
                for offset, code in enumerate(inverse.reshape(-1).tolist()):
                    if code in first_seen:
                        duplicates.append(offset)
                    else:
                        first_seen[code] = offset
                collisions += len(duplicates)
                available = np.setdiff1d(catalogue_indices, codes, assume_unique=True)
                replacements = (
                    rng.choice(available, size=len(duplicates), replace=False)
                    if available.size >= len(duplicates)
                    else rng.integers(0, self.catalog_size, size=len(duplicates))
                )
                repaired[positions[np.asarray(duplicates, dtype="int64")]] = replacements
        if collisions:
            logger.debug(
                "Candidate lists repaired: {} duplicated slots redrawn without replacement",
                collisions,
            )
        return repaired

    # ----------------------------------------------------------------------------------
    # Signaux croisés et état de l'article à la date de session
    # ----------------------------------------------------------------------------------
    def _cross_signals(
        self,
        users: pd.DataFrame,
        catalogue: pd.DataFrame,
        user_index: npt.NDArray[np.int64],
        item_index: npt.NDArray[np.int64],
        rng: np.random.Generator,
    ) -> dict[str, npt.NDArray[Any]]:
        """Compute the user-item signals available at scoring time.

        Args:
            users: User base.
            catalogue: Item catalogue.
            user_index: Positional user index of every row.
            item_index: Positional item index of every row.
            rng: Seeded random generator.

        Returns:
            A mapping of signal name to aligned array.
        """
        n_rows = user_index.size
        taste = np.stack(users["taste"].to_numpy())
        orders = users["user_orders_12m"].to_numpy(dtype="float64")
        distinct = users["user_distinct_categories_12m"].to_numpy(dtype="int64")
        brand_loyalty = users["brand_loyalty"].to_numpy(dtype="float64")
        typical_price = users["user_typical_price_eur"].to_numpy(dtype="float64")

        category_index = catalogue["category_index"].to_numpy()[item_index]
        item_price = catalogue["item_price_eur"].to_numpy(dtype="float64")[item_index]
        brand_appeal = catalogue["brand_appeal"].to_numpy(dtype="float64")[item_index]
        conversion = catalogue["item_conversion_rate_30d"].to_numpy(dtype="float64")[item_index]

        # Affinité catégorie : part du goût de l'utilisateur pour la catégorie de l'article,
        # mise à zéro pour les catégories qu'il n'a jamais achetées. C'est ce zéro qui rend la
        # distribution bimodale et le signal réellement discriminant.
        affinity = np.asarray(
            [taste[user][category] for user, category in zip(user_index, category_index, strict=True)],
            dtype="float64",
        )
        activity_factor = np.clip(0.55 + orders[user_index] / 12.0, 0.55, 2.4)
        affinity = affinity * activity_factor

        # Les `distinct_categories` premières catégories du goût sont considérées comme déjà
        # achetées ; les autres valent zéro (l'utilisateur n'y a jamais mis les pieds).
        rank_of_category = np.asarray(
            [
                int(np.flatnonzero(np.argsort(-taste[user]) == category)[0])
                for user, category in zip(user_index, category_index, strict=True)
            ],
            dtype="int64",
        )
        never_bought = rank_of_category >= np.maximum(distinct[user_index], 1)
        affinity[never_bought] = 0.0
        affinity = np.clip(affinity * 2.1, 0.0, 1.0)

        brand_affinity = np.clip(
            brand_loyalty[user_index] * (0.35 + brand_appeal) * np.exp(rng.normal(0.0, 0.28, n_rows)),
            0.0,
            1.0,
        )

        price_gap = 100.0 * (item_price - typical_price[user_index]) / np.maximum(
            typical_price[user_index], 1.0
        )
        price_gap = np.clip(price_gap, -95.0, 400.0)

        # Consultation passée de cet article : absente pour la majorité des candidats, ce qui
        # place le moteur en exploration plutôt qu'en exploitation.
        never_viewed = rng.random(n_rows) < self.never_viewed_share
        days_since = np.where(
            never_viewed,
            np.nan,
            np.clip(rng.exponential(26.0, size=n_rows), 0.0, 420.0),
        )
        item_views = np.where(
            never_viewed, 0.0, np.clip(rng.poisson(0.9, size=n_rows), 0.0, 40.0)
        )
        # Un article consulté récemment l'est généralement plusieurs fois : les deux signaux
        # doivent être corrélés, sinon le modèle apprend une relation qui n'existe nulle part.
        recent = (~never_viewed) & (days_since < 14.0)
        item_views = np.where(recent, item_views + rng.poisson(1.6, size=n_rows), item_views)
        item_views = np.clip(item_views, 0.0, 40.0)

        # Signal collaboratif : taux d'achat des utilisateurs proches, fonction de la conversion
        # de la fiche et de l'affinité de catégorie — pas un bruit indépendant.
        similar = (
            conversion
            * (0.45 + 1.15 * affinity)
            * np.exp(rng.normal(0.0, 0.32, n_rows))
        )
        similar = np.clip(similar, 0.0, 0.5)

        return {
            "user_category_affinity": np.round(affinity, 4),
            "user_brand_affinity": np.round(brand_affinity, 4),
            "price_gap_pct": np.round(price_gap, 1),
            "days_since_last_view": np.round(days_since, 0),
            "user_item_views_30d": item_views.astype(np.int64),
            "similar_users_buy_rate": np.round(similar, 4),
        }

    def _item_state_at_session(
        self,
        catalogue: pd.DataFrame,
        item_index: npt.NDArray[np.int64],
        offsets: npt.NDArray[np.int64],
        rng: np.random.Generator,
    ) -> dict[str, npt.NDArray[Any]]:
        """Return the time-dependent state of each candidate item at its session date.

        L'audience et le stock d'un article dépendent de la date : sans cela, un même article
        aurait exactement les mêmes statistiques dans le train et dans le test, et le modèle
        apprendrait une constante au lieu d'une dynamique. Toutes les fenêtres restent
        **antérieures** à la session, ce qui est la condition d'absence de fuite.

        Args:
            catalogue: Item catalogue.
            item_index: Positional item index of every row.
            offsets: Day offset of the session, aligned with ``item_index``.
            rng: Seeded random generator.

        Returns:
            A mapping with the dynamic item columns.
        """
        n_rows = item_index.size
        base_views = catalogue["item_views_7d"].to_numpy(dtype="float64")[item_index]
        age = catalogue["item_age_days"].to_numpy(dtype="float64")[item_index]

        dates = self.start + pd.to_timedelta(offsets, unit="D")
        weekend = np.where(dates.dayofweek.to_numpy() >= 5, 1.22, 1.0)
        month = dates.month.to_numpy(dtype="float64")
        seasonal = 1.0 + 0.45 * np.isin(month, (11, 12)).astype("float64")
        # Un article récent monte, un article ancien décroît lentement : effet de nouveauté.
        novelty = np.exp(-(offsets - (self.window_days - age)) / 260.0)
        novelty = np.clip(novelty, 0.35, 1.9)
        views = base_views * weekend * seasonal * novelty * np.exp(rng.normal(0.0, 0.25, n_rows))
        views = np.clip(np.round(views), 0.0, 40000.0).astype(np.int64)

        # Stock : plus un article est demandé, plus il part en rupture — corrélation réaliste
        # qui rend le filtre métier coûteux en pertinence, et donc intéressant à chiffrer.
        stock_level = rng.integers(0, 900, size=n_rows).astype(np.int64)
        shortage_probability = self.out_of_stock_rate * (0.55 + np.log1p(views) / 7.5)
        out_of_stock = rng.random(n_rows) < np.clip(shortage_probability, 0.0, 0.6)
        stock = np.where(out_of_stock, 0, np.maximum(stock_level, 1))

        # Promotions : calendrier chargé en fin d'année, comme un vrai plan d'animation.
        base_promoted = catalogue["item_is_promoted"].to_numpy()[item_index]
        promo_boost = np.isin(month, (6, 11, 12))
        promoted = base_promoted | (rng.random(n_rows) < np.where(promo_boost, 0.22, 0.05))

        conversion = catalogue["item_conversion_rate_30d"].to_numpy(dtype="float64")[item_index]
        conversion = np.clip(conversion * weekend * np.exp(rng.normal(0.0, 0.10, n_rows)), 0.0, 0.6)

        return {
            "item_views_7d": views,
            "item_stock_units": stock,
            "item_is_promoted": promoted,
            "item_conversion_rate_30d": np.round(conversion, 4),
        }

    # ----------------------------------------------------------------------------------
    # Pertinence latente
    # ----------------------------------------------------------------------------------
    def _latent_logit(
        self,
        frame: pd.DataFrame,
        users: pd.DataFrame,
        user_index: npt.NDArray[np.int64],
        item_index: npt.NDArray[np.int64],
        rng: np.random.Generator,
    ) -> npt.NDArray[np.float64]:
        """Return the uncalibrated latent logit of every candidate.

        Le logit combine les signaux dans un ordre voulu : l'affinité individuelle domine,
        l'intention récente et le signal collaboratif viennent ensuite, la qualité de la fiche
        pèse peu, et la popularité n'apporte qu'une contribution marginale — sinon la baseline
        popularité serait imbattable et l'exercice perdrait son intérêt. Le bruit gaussien fixe
        le plafond atteignable : il est délibérément non observable dans le jeu de données.

        Args:
            frame: Rows with every feature already computed.
            users: User base (latent traits).
            user_index: Positional user index of every row.
            item_index: Positional item index of every row.
            rng: Seeded random generator.

        Returns:
            The latent logit, intercept excluded.
        """
        affinity = frame["user_category_affinity"].to_numpy(dtype="float64")
        brand = frame["user_brand_affinity"].to_numpy(dtype="float64")
        gap = frame["price_gap_pct"].to_numpy(dtype="float64") / 100.0
        intent = np.log1p(frame["user_item_views_30d"].to_numpy(dtype="float64"))
        collaborative = frame["similar_users_buy_rate"].to_numpy(dtype="float64")
        rating = frame["item_rating_avg"].to_numpy(dtype="float64")
        views = frame["item_views_7d"].to_numpy(dtype="float64")
        catalogue = self._catalogue
        if catalogue is None:
            # Le facteur latent d'attrait est une propriété du catalogue : il n'existe qu'après
            # `_build_catalogue`. Échouer ici, avec ce message, vaut mieux qu'un None silencieux
            # plus bas dans le calcul du logit.
            msg = "The latent logit needs the catalogue; build it before scoring candidates."
            raise RuntimeError(msg)
        appeal = catalogue["item_appeal"].to_numpy(dtype="float64")[item_index]
        promoted = frame["item_is_promoted"].to_numpy(dtype="float64")
        conversion = frame["item_conversion_rate_30d"].to_numpy(dtype="float64")
        stock = frame["item_stock_units"].to_numpy(dtype="float64")
        age = frame["item_age_days"].to_numpy(dtype="float64")

        price_sensitivity = users["price_sensitivity"].to_numpy(dtype="float64")[user_index]
        novelty_appetite = users["novelty_appetite"].to_numpy(dtype="float64")[user_index]
        is_cold = users["is_cold"].to_numpy(dtype="float64")[user_index]

        # Pénalité de prix asymétrique : un candidat plus cher que les habitudes coûte nettement
        # plus qu'un candidat moins cher. Être très en dessous n'est pas un avantage — un article
        # à 5 EUR pour un client habitué à 200 EUR n'est pas plus pertinent, il est hors sujet.
        price_penalty = price_sensitivity * (
            1.55 * np.maximum(gap, 0.0) + 0.45 * np.maximum(-gap - 0.25, 0.0)
        )

        dates = frame["session_date"]
        weekend = np.where(dates.dt.dayofweek.to_numpy() >= 5, 0.22, 0.0)
        month = dates.dt.month.to_numpy(dtype="float64")
        season = 0.30 * np.isin(month, (11, 12)).astype("float64") - 0.18 * np.isin(
            month, (7, 8)
        ).astype("float64")

        logit = (
            2.55 * affinity
            + self.appeal_weight * appeal
            + 1.15 * brand
            - price_penalty
            + 0.85 * intent
            + 7.50 * collaborative
            + 0.42 * (rating - 4.0)
            + self.popularity_weight * np.log1p(views) / 7.0
            + 0.30 * promoted
            + 1.10 * conversion * 8.0
            - 0.55 * is_cold
            - 0.55 * (stock <= 0).astype("float64")
            + 0.55 * novelty_appetite * (age < 45.0).astype("float64")
            + weekend
            + season
            + rng.normal(0.0, self.noise_scale, size=frame.shape[0])
        )
        return np.asarray(logit, dtype="float64")

    def _calibrate_intercept(
        self, logit: npt.NDArray[np.float64], target_rate: float
    ) -> float:
        """Return the intercept making the mean relevance probability match ``target_rate``.

        Résolu par dichotomie sur la probabilité moyenne : la prévalence configurée est donc
        respectée quelle que soit la graine, ce qui stabilise les seuils de réussite du projet.

        Args:
            logit: Latent logit without intercept.
            target_rate: Desired share of relevant rows.

        Returns:
            The calibrated intercept.
        """
        target = float(np.clip(target_rate, 1e-4, 1.0 - 1e-4))
        low, high = -25.0, 25.0
        for _ in range(80):
            middle = 0.5 * (low + high)
            rate = float(_sigmoid(logit + middle).mean())
            if rate < target:
                low = middle
            else:
                high = middle
        return float(0.5 * (low + high))

    # ----------------------------------------------------------------------------------
    # Génération
    # ----------------------------------------------------------------------------------
    def generate(self) -> pd.DataFrame:
        """Generate the candidate-scoring log.

        Returns:
            A frame with :data:`COLUMNS` as columns and ``n_users * candidates_per_user`` rows.
        """
        rng = np.random.default_rng(self.seed)
        catalogue = self._build_catalogue(rng)
        users = self._build_users(rng)
        self._catalogue = catalogue
        self._users = users

        offsets = self._session_dates(rng)
        user_index, item_index, _slots = self._sample_candidates(users, catalogue, rng)

        session_date = self.start + pd.to_timedelta(offsets[user_index], unit="D")
        user_columns = {
            column: users[column].to_numpy()[user_index]
            for column in (
                "user_id",
                "user_tenure_days",
                "user_orders_12m",
                "user_spend_eur_12m",
                "user_sessions_30d",
                "user_distinct_categories_12m",
                "user_avg_basket_eur",
                "user_typical_price_eur",
                "user_return_rate",
                "user_price_band_pref",
                "user_channel",
                "user_region",
                "user_is_member",
            )
        }
        item_columns = {
            column: catalogue[column].to_numpy()[item_index]
            for column in (
                "item_id",
                "item_category",
                "item_price_eur",
                "item_rating_avg",
                "item_reviews_count",
                "item_age_days",
                "item_margin_pct",
            )
        }

        frame = pd.DataFrame(
            {
                "sample_id": [f"RC-{index:06d}" for index in range(user_index.size)],
                "user_id": user_columns["user_id"],
                "item_id": item_columns["item_id"],
                "session_date": session_date.astype("datetime64[ns]"),
                **{key: value for key, value in user_columns.items() if key != "user_id"},
                **{key: value for key, value in item_columns.items() if key != "item_id"},
            }
        )

        dynamic = self._item_state_at_session(catalogue, item_index, offsets[user_index], rng)
        for column, values in dynamic.items():
            frame[column] = values
        cross = self._cross_signals(users, catalogue, user_index, item_index, rng)
        for column, values in cross.items():
            frame[column] = values

        logit = self._latent_logit(frame, users, user_index, item_index, rng)
        intercept = self._calibrate_intercept(logit, self.positive_rate)
        probability = _sigmoid(logit + intercept)
        frame["relevance"] = (rng.random(frame.shape[0]) < probability).astype(np.int8)

        frame = frame[list(self.COLUMNS)].reset_index(drop=True)

        # La probabilité latente est conservée hors du jeu de données : elle sert à publier le
        # plafond atteignable, jamais à entraîner. L'écrire dans la table serait une fuite.
        self._latent_probability = probability
        self._latent_frame = pd.DataFrame(
            {
                "user_id": frame["user_id"].to_numpy(),
                "relevance": frame["relevance"].to_numpy(dtype="float64"),
                "latent_probability": probability,
                "popularity_score": frame["item_views_7d"].to_numpy(dtype="float64"),
            }
        )

        logger.info(
            "Candidate log generated | users={} items={} rows={} positive_rate={:.3f} "
            "cold_users={:.1%} out_of_stock={:.1%}",
            self.n_users,
            self.catalog_size,
            len(frame),
            float(frame["relevance"].mean()),
            float(users["is_cold"].mean()),
            float((frame["item_stock_units"] == 0).mean()),
        )
        return frame

    # ----------------------------------------------------------------------------------
    # Export et API publique
    # ----------------------------------------------------------------------------------
    def export(self, frame: pd.DataFrame | None = None) -> dict[str, Path]:
        """Write the dataset in every configured format.

        Args:
            frame: Data to write; generated when ``None``.

        Returns:
            Mapping of format to written path.
        """
        data = frame if frame is not None else self.generate()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        base = self.output_dir / self.dataset_name
        written = write_table_multiple(data, base, formats=self.formats)
        logger.info("Dataset exported: {}", {fmt: str(path) for fmt, path in written.items()})
        return written

    def run(self) -> dict[str, Path]:
        """Generate, export and persist the generation metadata.

        Returns:
            Mapping of format to written path.
        """
        frame = self.generate()
        written = self.export(frame)
        payload = self.metadata(frame)
        payload["files"] = {fmt: str(path) for fmt, path in written.items()}
        write_json(self.output_dir / "generation_metadata.json", payload)
        return written

    def sample(self, n_samples: int, *, with_target: bool = False) -> pd.DataFrame:
        """Draw a small sample, typically used to demo inference.

        Args:
            n_samples: Number of rows.
            with_target: Keep the relevance column, as a backtest would, or drop it, as a real
                scoring request would (the engine does not know the future when it ranks).

        Returns:
            The sampled frame.
        """
        frame = self.generate()
        sample_frame = frame.head(max(int(n_samples), 1)).copy()
        if not with_target and "relevance" in sample_frame.columns:
            sample_frame = sample_frame.drop(columns=["relevance"])
        return sample_frame

    # ------------------------------------------------------------------ metadata --------
    def metadata(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Build the traceability payload of a generation run.

        Le point important est le trio de NDCG@10 : le **plafond** (score latent parfait,
        inaccessible puisqu'il contient le bruit), la **baseline popularité** (ce que fait le
        moteur historique) et l'**aléatoire**. Un modèle livré doit se situer nettement au-dessus
        de la popularité et nettement sous le plafond ; hors de cette fourchette, soit il
        n'apporte rien, soit il triche.

        Args:
            frame: Generated data.

        Returns:
            A JSON-serialisable mapping.
        """
        top_k = 10
        latent = self._latent_frame
        if latent is None:
            latent = pd.DataFrame(
                {
                    "user_id": frame["user_id"].to_numpy(),
                    "relevance": frame["relevance"].to_numpy(dtype="float64"),
                    "latent_probability": np.full(len(frame), np.nan),
                    "popularity_score": frame["item_views_7d"].to_numpy(dtype="float64"),
                }
            )
        reference_rng = np.random.default_rng(self.seed + 1)
        random_frame = latent.assign(random_score=reference_rng.random(len(latent)))

        oracle = ndcg_at_k_from_groups(
            latent, group_column="user_id", truth_column="relevance",
            score_column="latent_probability", top_k=top_k,
        )
        popularity = ndcg_at_k_from_groups(
            latent, group_column="user_id", truth_column="relevance",
            score_column="popularity_score", top_k=top_k,
        )
        random = ndcg_at_k_from_groups(
            random_frame, group_column="user_id", truth_column="relevance",
            score_column="random_score", top_k=top_k,
        )

        cold_users = frame.loc[frame["user_orders_12m"] <= 1, "user_id"].nunique()
        category_counts = frame["item_category"].value_counts(normalize=True).head(5)
        payload: dict[str, Any] = {
            "generator": "SyntheticDataGenerator(recommendation)",
            "dataset_name": self.dataset_name,
            "seed": self.seed,
            "n_rows": int(len(frame)),
            "n_columns": int(frame.shape[1]),
            "n_users": int(frame["user_id"].nunique()),
            "n_items": int(frame["item_id"].nunique()),
            "catalog_size": self.catalog_size,
            "candidates_per_user": self.candidates_per_user,
            "window_days": self.window_days,
            "date_min": str(pd.Timestamp(frame["session_date"].min()).date()),
            "date_max": str(pd.Timestamp(frame["session_date"].max()).date()),
            "target": "relevance",
            "positive_rate_requested": self.positive_rate,
            "positive_rate_actual": round(float(frame["relevance"].mean()), 4),
            "cold_user_share": round(float(cold_users / max(frame["user_id"].nunique(), 1)), 4),
            "out_of_stock_share": round(float((frame["item_stock_units"] == 0).mean()), 4),
            "never_viewed_share": round(float(frame["days_since_last_view"].isna().mean()), 4),
            "zero_intent_share": round(float((frame["user_item_views_30d"] == 0).mean()), 4),
            "promoted_share": round(float(frame["item_is_promoted"].mean()), 4),
            "catalog_coverage": round(
                float(frame["item_id"].nunique() / max(self.catalog_size, 1)), 4
            ),
            "top_k_reference": top_k,
            "ndcg_ceiling_oracle": None if np.isnan(oracle) else round(float(oracle), 4),
            "ndcg_baseline_popularity": None if np.isnan(popularity) else round(float(popularity), 4),
            "ndcg_baseline_random": None if np.isnan(random) else round(float(random), 4),
            "ndcg_headroom": None
            if (np.isnan(oracle) or np.isnan(popularity))
            else round(float(oracle - popularity), 4),
            "category_distribution_top5": {
                str(key): round(float(value), 4) for key, value in category_counts.items()
            },
            "noise_scale": self.noise_scale,
            "popularity_weight": self.popularity_weight,
            "options": {
                key: getattr(self, key)
                for key in self.SUPPORTED_OPTIONS
                if key != "start" and hasattr(self, key)
            },
            "notes": (
                "La probabilité latente n'est pas écrite dans le jeu : elle sert uniquement à "
                "publier le plafond de NDCG@10 atteignable. Un modèle qui l'approcherait de trop "
                "près signale une fuite, pas une performance."
            ),
        }
        return payload

    def describe_target(self, frame: pd.DataFrame | None = None) -> pd.DataFrame:
        """Describe the target globally and per decision-relevant segment.

        Args:
            frame: Data to describe; generated when ``None``.

        Returns:
            A frame with one row per segment: share, relevance rate and candidate counts.
        """
        data = frame if frame is not None else self.generate()
        rows: list[dict[str, Any]] = [
            {
                "segment": "ensemble",
                "rows": int(len(data)),
                "relevance_rate": round(float(data["relevance"].mean()), 4),
                "users": int(data["user_id"].nunique()),
                "items": int(data["item_id"].nunique()),
            }
        ]
        segments: list[tuple[str, pd.Series]] = [
            ("utilisateur froid (<= 1 commande)", data["user_orders_12m"] <= 1),
            ("utilisateur tiède (2 à 8)", data["user_orders_12m"].between(2, 8)),
            ("utilisateur chaud (> 8)", data["user_orders_12m"] > 8),
            ("candidat en rupture de stock", data["item_stock_units"] == 0),
            ("candidat jamais consulté", data["user_item_views_30d"] == 0),
            ("candidat consulté récemment", data["user_item_views_30d"] > 0),
            ("affinité catégorie nulle", data["user_category_affinity"] == 0.0),
            ("affinité catégorie forte (> 0,3)", data["user_category_affinity"] > 0.3),
            ("article populaire (top décile d'audience)",
             data["item_views_7d"] >= data["item_views_7d"].quantile(0.9)),
            ("longue traîne (dernier quartile d'audience)",
             data["item_views_7d"] <= data["item_views_7d"].quantile(0.25)),
            ("article en promotion", data["item_is_promoted"].astype(bool)),
            ("membre du programme de fidélité", data["user_is_member"].astype(bool)),
        ]
        for label, mask in segments:
            subset = data.loc[mask]
            rows.append(
                {
                    "segment": label,
                    "rows": int(len(subset)),
                    "relevance_rate": round(float(subset["relevance"].mean()), 4)
                    if len(subset)
                    else float("nan"),
                    "users": int(subset["user_id"].nunique()),
                    "items": int(subset["item_id"].nunique()),
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        """Return a compact summary of the generation parameters and their measured effect.

        La table ne se contente pas de rappeler les réglages : elle publie ce qu'ils ont produit
        (prévalence réelle, part d'utilisateurs froids, couverture catalogue) et les trois NDCG@10
        de référence — plafond, popularité, aléatoire. C'est ce qui permet de lire un score de
        modèle sans le comparer à rien.

        Returns:
            A two-column frame (parameter, value).
        """
        frame = self.generate()
        payload = self.metadata(frame)
        keys = (
            "seed",
            "n_rows",
            "n_users",
            "n_items",
            "candidates_per_user",
            "date_min",
            "date_max",
            "positive_rate_requested",
            "positive_rate_actual",
            "cold_user_share",
            "out_of_stock_share",
            "never_viewed_share",
            "zero_intent_share",
            "catalog_coverage",
            "top_k_reference",
            "ndcg_ceiling_oracle",
            "ndcg_baseline_popularity",
            "ndcg_baseline_random",
            "ndcg_headroom",
        )
        return pd.DataFrame([{"parametre": key, "valeur": payload.get(key)} for key in keys])


    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], paths: ProjectPaths | None = None
    ) -> SyntheticDataGenerator:
        """Build a generator from the ``data`` configuration node.

        Args:
            config: Root configuration mapping (or the ``data`` node alone).
            paths: Project layout (defaults to :meth:`ProjectPaths.from_root`).

        Returns:
            The configured generator.
        """
        # Même forme que les autres familles : `dict(...)` borne l'inférence mypy.
        node = dict(config.get("data", config) or {})
        layout = paths or ProjectPaths.from_root()
        options = {
            key: node[key] for key in cls.SUPPORTED_OPTIONS if key in node and node[key] is not None
        }
        return cls(
            n_samples=int(node.get("n_samples", 36000)),
            seed=int(config.get("seed", node.get("seed", 42))),
            dataset_name=str(node.get("dataset_name", DEFAULT_DATASET_NAME)),
            output_dir=layout.raw_dir,
            formats=tuple(node.get("formats", ("parquet", "csv"))),
            start=str(node.get("start", "2024-09-02")),
            **options,
        )


__all__ = [
    "CATEGORIES",
    "CHANNELS",
    "DEFAULT_DATASET_NAME",
    "PRICE_BANDS",
    "REGIONS",
    "SyntheticDataGenerator",
    "ndcg_at_k_from_groups",
]
