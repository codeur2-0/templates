"""Contrats de données — Journal de candidats scorés d'un catalogue e-commerce.

Ce module est la **documentation exécutable** du jeu de données. Trois schémas Pandera
(``DataFrameModel``) encadrent le cycle de vie :

``RawDataSchema``        appliqué juste après le chargement des données brutes : types, bornes,
                         valeurs autorisées, unicité de la clé, absence de nuls non déclarés.
``ProcessedDataSchema``  appliqué sur la matrice livrée au modèle : uniquement numérique,
                         aucune valeur manquante, au moins une feature.
``InferenceDataSchema``  appliqué sur toute requête de prédiction : colonnes optionnelles et
                         nulls tolérés, car un payload externe est incomplet par nature.

Pourquoi des schémas plutôt que des ``assert`` dispersés ?

* l'erreur est **explicite** (colonne, check, valeurs en échec) et localisée à l'entrée,
* le contrat est **versionné** avec le code et testé (``tests/test_data_schemas.py``),
* la coercition (``coerce=True``) absorbe les différences de types entre Parquet et CSV,
* le schéma sert de **documentation** : ``describe_schema()`` produit une table lisible.

Références :
    * https://pandera.readthedocs.io/en/stable/schema_models.html
    * https://pandera.readthedocs.io/en/stable/reference/generated/pandera.api.pandas.model.DataFrameModel.html
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:  # pandera >= 0.26 expose un sous-module dédié à pandas
    import pandera.pandas as pa
except ModuleNotFoundError:  # pragma: no cover - pandera < 0.26
    import pandera as pa  # type: ignore[no-redef]

from pandera.typing import DataFrame, Series  # noqa: E402  (import après le fallback pandera)

from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

#: Nom logique du jeu de données (utilisé dans les messages d'erreur et les rapports).
DATASET_NAME = "ecommerce_candidate_impressions"

#: Colonnes attendues et leur type logique — source de vérité pour la documentation.
EXPECTED_DTYPES: dict[str, str] = {"sample_id": "str", "user_id": "str", "item_id": "str", "session_date": "datetime", "user_tenure_days": "int", "user_orders_12m": "int", "user_spend_eur_12m": "float", "user_sessions_30d": "int", "user_distinct_categories_12m": "int", "user_avg_basket_eur": "float", "user_typical_price_eur": "float", "user_return_rate": "float", "user_price_band_pref": "category", "user_channel": "category", "user_region": "str", "user_is_member": "bool", "item_category": "category", "item_price_eur": "float", "item_rating_avg": "float", "item_reviews_count": "int", "item_stock_units": "int", "item_age_days": "int", "item_margin_pct": "float", "item_views_7d": "int", "item_conversion_rate_30d": "float", "item_is_promoted": "bool", "user_category_affinity": "float", "user_brand_affinity": "float", "price_gap_pct": "float", "days_since_last_view": "int", "user_item_views_30d": "int", "similar_users_buy_rate": "float", "relevance": "int"}

#: Colonnes utilisées comme features par le modèle.
FEATURE_COLUMNS: list[str] = ['user_tenure_days', 'user_orders_12m', 'user_spend_eur_12m', 'user_sessions_30d', 'user_distinct_categories_12m', 'user_avg_basket_eur', 'user_typical_price_eur', 'user_return_rate', 'user_price_band_pref', 'user_channel', 'user_region', 'user_is_member', 'item_category', 'item_price_eur', 'item_rating_avg', 'item_reviews_count', 'item_stock_units', 'item_age_days', 'item_margin_pct', 'item_views_7d', 'item_conversion_rate_30d', 'item_is_promoted', 'user_category_affinity', 'user_brand_affinity', 'price_gap_pct', 'days_since_last_view', 'user_item_views_30d', 'similar_users_buy_rate']

#: Colonne cible.
TARGET_COLUMN: str = "relevance"

#: Clé métier (unique).
ID_COLUMN: str = "sample_id"

#: Colonne portant l'unité de publication d'une liste de candidats (l'utilisateur, la session).
GROUP_COLUMN: str | None = 'user_id'

#: Colonne identifiant l'article recommandé, déclarée dans le bloc `recommendation` de la
#: configuration racine : aucun nom n'est codé en dur dans le contrat.
ITEM_COLUMN: str = "item_id"

#: Colonnes non modélisables (identifiants, horodatage, métadonnées).
NON_FEATURE_COLUMNS: list[str] = ['sample_id', 'user_id', 'item_id', 'session_date', 'relevance']


class RawDataSchema(pa.DataFrameModel):
    """Contrat des **données brutes** (``data/raw/ecommerce_candidate_impressions.parquet``).

    Chaque déclaration ci-dessous correspond à une colonne du schéma documenté dans
    ``data/README.md``. ``coerce=True`` convertit les types compatibles (utile quand le
    dataset est lu depuis un CSV plutôt que depuis Parquet) ; ``strict=True`` refuse toute
    colonne non déclarée, ce qui détecte immédiatement une évolution silencieuse de la source.
    """

    sample_id: Series[str] = pa.Field(nullable=False, str_matches='^RC-[0-9]{6}$', description="Identifiant unique du couple (utilisateur, article candidat)")
    user_id: Series[str] = pa.Field(nullable=False, str_matches='^U-[0-9]{4}$', description="Identifiant utilisateur — clé de regroupement des métriques de classement, jamais une feature")
    item_id: Series[str] = pa.Field(nullable=False, str_matches='^I-[0-9]{5}$', description="Identifiant article du catalogue — jamais une feature, pour les mêmes raisons")
    session_date: Series[pa.DateTime] = pa.Field(nullable=False, description="Date de la session au cours de laquelle le candidat a été scoré — clé du split chronologique")
    user_tenure_days: Series[int] = pa.Field(nullable=False, ge=0, le=3000, description="Ancienneté du compte utilisateur en jours")
    user_orders_12m: Series[int] = pa.Field(nullable=False, ge=0, le=120, description="Nombre de commandes de l'utilisateur sur les 12 derniers mois — pilote le segment froid / tiède / chaud")
    user_spend_eur_12m: Series[float] = pa.Field(nullable=False, ge=0.0, le=12000.0, description="Dépense cumulée de l'utilisateur sur 12 mois")
    user_sessions_30d: Series[int] = pa.Field(nullable=False, ge=0, le=200, description="Sessions ouvertes sur les 30 derniers jours — intensité d'usage récente")
    user_distinct_categories_12m: Series[int] = pa.Field(nullable=False, ge=0, le=12, description="Nombre de catégories distinctes achetées sur 12 mois — amplitude d'exploration")
    user_avg_basket_eur: Series[float] = pa.Field(nullable=False, ge=0.0, le=900.0, description="Panier moyen de l'utilisateur")
    user_typical_price_eur: Series[float] = pa.Field(nullable=False, ge=0.0, le=900.0, description="Prix médian des articles achetés par l'utilisateur — référence pour mesurer l'écart de prix d'un candidat")
    user_return_rate: Series[float] = pa.Field(nullable=False, ge=0.0, le=1.0, description="Part des commandes retournées sur 12 mois — un retour fréquent signale une pertinence mal calibrée")
    user_price_band_pref: Series[str] = pa.Field(nullable=False, isin=['entree', 'milieu', 'premium'], description="Gamme de prix préférée de l'utilisateur")
    user_channel: Series[str] = pa.Field(nullable=False, isin=['web', 'mobile', 'app'], description="Canal principal d'achat (le classement mobile privilégie les fiches courtes)")
    user_region: Series[str] = pa.Field(nullable=False, isin=['nord', 'sud', 'est', 'ouest', 'idf'], description="Région de livraison — effet sur la disponibilité et les délais")
    user_is_member: Series[int] = pa.Field(nullable=False, description="Adhésion au programme de fidélité")
    item_category: Series[str] = pa.Field(nullable=False, isin=['mode', 'maison', 'high_tech', 'sport', 'beaute', 'alimentaire', 'jouet', 'jardin'], description="Catégorie catalogue de l'article")
    item_price_eur: Series[float] = pa.Field(nullable=False, ge=1.0, le=2500.0, description="Prix de vente de l'article")
    item_rating_avg: Series[float] = pa.Field(nullable=False, ge=1.0, le=5.0, description="Note moyenne de l'article")
    item_reviews_count: Series[int] = pa.Field(nullable=False, ge=0, le=6000, description="Nombre d'avis publiés — preuve sociale, mais aussi proxy de l'âge de l'article")
    item_stock_units: Series[int] = pa.Field(nullable=False, ge=0, le=900, description="Stock disponible au moment du scoring — un stock nul interdit la publication")
    item_age_days: Series[int] = pa.Field(nullable=False, ge=0, le=2200, description="Ancienneté de l'article au catalogue")
    item_margin_pct: Series[float] = pa.Field(nullable=False, ge=0.0, le=70.0, description="Marge brute de l'article — permet d'arbitrer explicitement pertinence contre valeur")
    item_views_7d: Series[int] = pa.Field(nullable=False, ge=0, le=40000, description="Vues de l'article sur les 7 jours **précédant** la session — fenêtre strictement antérieure, donc sans fuite")
    item_conversion_rate_30d: Series[float] = pa.Field(nullable=False, ge=0.0, le=0.6, description="Taux de conversion de la fiche sur 30 jours glissants antérieurs")
    item_is_promoted: Series[int] = pa.Field(nullable=False, description="Article poussé par une opération marketing au moment de la session")
    user_category_affinity: Series[float] = pa.Field(nullable=False, ge=0.0, le=1.0, description="Part des dépenses de l'utilisateur dans la catégorie de l'article — le signal croisé le plus fort")
    user_brand_affinity: Series[float] = pa.Field(nullable=False, ge=0.0, le=1.0, description="Affinité de l'utilisateur à la marque de l'article (historique d'achats et de consultations)")
    price_gap_pct: Series[float] = pa.Field(nullable=False, ge=-95.0, le=400.0, description="Écart relatif entre le prix du candidat et le prix habituel de l'utilisateur — un écart fort fait chuter la pertinence")
    days_since_last_view: Series[int] = pa.Field(nullable=True, ge=0, le=420, description="Jours écoulés depuis la dernière consultation de cet article par cet utilisateur")
    user_item_views_30d: Series[int] = pa.Field(nullable=False, ge=0, le=40, description="Consultations de ce candidat par cet utilisateur sur 30 jours glissants antérieurs")
    similar_users_buy_rate: Series[float] = pa.Field(nullable=False, ge=0.0, le=0.5, description="Taux d'achat de ce candidat par les utilisateurs les plus proches (signal collaboratif précalculé sur fenêtre antérieure)")
    relevance: Series[int] = pa.Field(nullable=False, isin=[0, 1], description="Pertinence observée : 1 si l'utilisateur a ajouté au panier ou acheté, 0 sinon")

    class Config:
        """Politique de validation des données brutes."""

        name = "RawDataSchema"
        coerce = True
        strict = True
        ordered = False
        unique = ["sample_id"]

    @pa.dataframe_check
    @classmethod
    def candidate_list_has_no_duplicate_item(cls, frame: DataFrame) -> bool:
        """Un article ne peut pas occuper deux emplacements de la même liste de candidats.

        Une liste qui répète la même référence ferait publier deux fois le même article, et
        rendrait surtout la cible **contradictoire** : deux lignes aux entrées strictement
        identiques, dont les étiquettes ont été tirées séparément, peuvent porter l'une 1 et
        l'autre 0. Le modèle devrait alors prédire la moyenne de deux réponses incompatibles,
        ce qui abaisse le plafond atteignable sans qu'aucune feature ne puisse l'expliquer —
        un bruit qu'aucune évaluation ne sait diagnostiquer.

        Le contrôle porte sur le couple (unité de publication, article) : le même article peut
        légitimement revenir dans deux sessions différentes, jamais deux fois dans la même liste.
        """
        if GROUP_COLUMN is None or GROUP_COLUMN not in frame.columns or frame.empty:
            return True
        if ITEM_COLUMN not in frame.columns:
            return True
        return not bool(frame.duplicated(subset=[GROUP_COLUMN, ITEM_COLUMN]).any())

    @pa.dataframe_check
    @classmethod
    def target_has_multiple_classes(cls, frame: DataFrame) -> bool:
        """Une donnée d'entraînement doit contenir au moins deux classes.

        Un dataset mono-classe produirait un modèle trivialement parfait et des métriques
        indéfinies (ROC AUC, log-loss). Mieux vaut échouer ici que découvrir le problème
        après un entraînement.
        """
        if TARGET_COLUMN is None or TARGET_COLUMN not in frame.columns or frame.empty:
            return True
        return int(frame[TARGET_COLUMN].nunique()) >= 2


class ProcessedDataSchema(pa.DataFrameModel):
    """Contrat de la **matrice livrée au modèle** (sortie du preprocessing).

    Après imputation, encodage et scaling, la matrice doit être :

    * entièrement numérique (les encodeurs ont fait leur travail),
    * sans aucune valeur manquante (sinon la plupart des frameworks plantent ou dégradent
      silencieusement la qualité),
    * non vide et dotée d'au moins une feature.

    ``strict=False`` est volontaire : le nombre et le nom des colonnes dépendent de
    l'encodage (one-hot) et sont donc dynamiques.
    """

    class Config:
        """Politique de validation des données transformées."""

        name = "ProcessedDataSchema"
        coerce = True
        strict = False

    @pa.dataframe_check
    @classmethod
    def has_rows(cls, frame: DataFrame) -> bool:
        """Refuser une matrice vide (aucun apprentissage possible)."""
        return len(frame) > 0

    @pa.dataframe_check
    @classmethod
    def has_columns(cls, frame: DataFrame) -> bool:
        """Refuser une matrice sans feature."""
        return frame.shape[1] > 0

    @pa.dataframe_check
    @classmethod
    def all_numeric(cls, frame: DataFrame) -> bool:
        """Toutes les colonnes doivent être numériques après encodage."""
        return all(pd.api.types.is_numeric_dtype(frame[column]) for column in frame.columns)

    @pa.dataframe_check
    @classmethod
    def no_missing_value(cls, frame: DataFrame) -> bool:
        """Aucune valeur manquante ne doit survivre au preprocessing."""
        return bool(frame.notna().all().all())

    @pa.dataframe_check
    @classmethod
    def no_infinite_value(cls, frame: DataFrame) -> bool:
        """Aucun infini (souvent produit par une division par zéro dans une feature ratio)."""
        numeric = frame.select_dtypes(include=[np.number])
        return bool(np.isfinite(numeric.to_numpy(dtype="float64")).all()) if not numeric.empty else True


class InferenceDataSchema(pa.DataFrameModel):
    """Contrat des **données soumises pour prédiction**.

    Différences assumées avec :class:`RawDataSchema` :

    * la cible n'est pas requise (on ne la connaît pas encore),
    * les colonnes sont optionnelles et les nulls tolérés : le preprocessing imputera,
    * ``strict=False`` : un payload peut contenir des colonnes supplémentaires, elles seront
      ignorées par le sélecteur de features.

    C'est précisément ce qui rend le service robuste : une requête partielle est corrigée par
    le pipeline, une requête incohérente est refusée avec un message explicite.
    """

    sample_id: Series[str] = pa.Field(nullable=True, required=False, str_matches='^RC-[0-9]{6}$', description="Identifiant unique du couple (utilisateur, article candidat)")
    user_id: Series[str] = pa.Field(nullable=True, required=False, str_matches='^U-[0-9]{4}$', description="Identifiant utilisateur — clé de regroupement des métriques de classement, jamais une feature")
    item_id: Series[str] = pa.Field(nullable=True, required=False, str_matches='^I-[0-9]{5}$', description="Identifiant article du catalogue — jamais une feature, pour les mêmes raisons")
    session_date: Series[pa.DateTime] = pa.Field(nullable=False, required=False, description="Date de la session au cours de laquelle le candidat a été scoré — clé du split chronologique")
    user_tenure_days: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=3000, description="Ancienneté du compte utilisateur en jours")
    user_orders_12m: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=120, description="Nombre de commandes de l'utilisateur sur les 12 derniers mois — pilote le segment froid / tiède / chaud")
    user_spend_eur_12m: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=12000.0, description="Dépense cumulée de l'utilisateur sur 12 mois")
    user_sessions_30d: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=200, description="Sessions ouvertes sur les 30 derniers jours — intensité d'usage récente")
    user_distinct_categories_12m: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=12, description="Nombre de catégories distinctes achetées sur 12 mois — amplitude d'exploration")
    user_avg_basket_eur: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=900.0, description="Panier moyen de l'utilisateur")
    user_typical_price_eur: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=900.0, description="Prix médian des articles achetés par l'utilisateur — référence pour mesurer l'écart de prix d'un candidat")
    user_return_rate: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=1.0, description="Part des commandes retournées sur 12 mois — un retour fréquent signale une pertinence mal calibrée")
    user_price_band_pref: Series[str] = pa.Field(nullable=False, required=False, isin=['entree', 'milieu', 'premium'], description="Gamme de prix préférée de l'utilisateur")
    user_channel: Series[str] = pa.Field(nullable=False, required=False, isin=['web', 'mobile', 'app'], description="Canal principal d'achat (le classement mobile privilégie les fiches courtes)")
    user_region: Series[str] = pa.Field(nullable=True, required=False, isin=['nord', 'sud', 'est', 'ouest', 'idf'], description="Région de livraison — effet sur la disponibilité et les délais")
    user_is_member: Series[int] = pa.Field(nullable=False, required=False, description="Adhésion au programme de fidélité")
    item_category: Series[str] = pa.Field(nullable=False, required=False, isin=['mode', 'maison', 'high_tech', 'sport', 'beaute', 'alimentaire', 'jouet', 'jardin'], description="Catégorie catalogue de l'article")
    item_price_eur: Series[float] = pa.Field(nullable=True, required=False, ge=1.0, le=2500.0, description="Prix de vente de l'article")
    item_rating_avg: Series[float] = pa.Field(nullable=True, required=False, ge=1.0, le=5.0, description="Note moyenne de l'article")
    item_reviews_count: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=6000, description="Nombre d'avis publiés — preuve sociale, mais aussi proxy de l'âge de l'article")
    item_stock_units: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=900, description="Stock disponible au moment du scoring — un stock nul interdit la publication")
    item_age_days: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=2200, description="Ancienneté de l'article au catalogue")
    item_margin_pct: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=70.0, description="Marge brute de l'article — permet d'arbitrer explicitement pertinence contre valeur")
    item_views_7d: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=40000, description="Vues de l'article sur les 7 jours **précédant** la session — fenêtre strictement antérieure, donc sans fuite")
    item_conversion_rate_30d: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=0.6, description="Taux de conversion de la fiche sur 30 jours glissants antérieurs")
    item_is_promoted: Series[int] = pa.Field(nullable=False, required=False, description="Article poussé par une opération marketing au moment de la session")
    user_category_affinity: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=1.0, description="Part des dépenses de l'utilisateur dans la catégorie de l'article — le signal croisé le plus fort")
    user_brand_affinity: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=1.0, description="Affinité de l'utilisateur à la marque de l'article (historique d'achats et de consultations)")
    price_gap_pct: Series[float] = pa.Field(nullable=True, required=False, ge=-95.0, le=400.0, description="Écart relatif entre le prix du candidat et le prix habituel de l'utilisateur — un écart fort fait chuter la pertinence")
    days_since_last_view: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=420, description="Jours écoulés depuis la dernière consultation de cet article par cet utilisateur")
    user_item_views_30d: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=40, description="Consultations de ce candidat par cet utilisateur sur 30 jours glissants antérieurs")
    similar_users_buy_rate: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=0.5, description="Taux d'achat de ce candidat par les utilisateurs les plus proches (signal collaboratif précalculé sur fenêtre antérieure)")

    class Config:
        """Politique de validation des payloads d'inférence."""

        name = "InferenceDataSchema"
        coerce = True
        strict = False


#: Registres utilisés par les loaders, les tests et la documentation.
SCHEMAS: dict[str, type[pa.DataFrameModel]] = {
    "raw": RawDataSchema,
    "processed": ProcessedDataSchema,
    "inference": InferenceDataSchema,
}


def validate_frame(
    frame: pd.DataFrame,
    schema: type[pa.DataFrameModel] | str = RawDataSchema,
    *,
    lazy: bool = False,
    context: str | None = None,
) -> pd.DataFrame:
    """Validate a DataFrame against one of the project contracts.

    Args:
        frame: DataFrame to validate.
        schema: Schema class or registry key (``raw``, ``processed``, ``inference``).
        lazy: Collect every error before raising (useful when debugging a new source).
        context: Label used in logs and error messages.

    Returns:
        The validated (and coerced) DataFrame.

    Raises:
        KeyError: When the schema key is unknown.
        pandera.errors.SchemaError: When the contract is violated.
    """
    resolved = SCHEMAS[schema] if isinstance(schema, str) else schema
    if isinstance(schema, str) and schema not in SCHEMAS:
        msg = f"Unknown schema key '{schema}'. Allowed: {sorted(SCHEMAS)}"
        raise KeyError(msg)

    label = context or resolved.__name__
    logger.debug("Validating {} rows against {}", len(frame), label)
    try:
        validated = resolved.validate(frame, lazy=lazy)
    except pa.errors.SchemaError as exc:
        logger.error("Validation '{}' failed: {}", label, exc.failure_cases.to_dict("records") if hasattr(exc, "failure_cases") else str(exc))
        raise
    logger.info("Validation '{}' succeeded | rows={} cols={}", label, len(validated), validated.shape[1])
    return validated


def describe_schema(schema: type[pa.DataFrameModel] | str = RawDataSchema) -> pd.DataFrame:
    """Return a documentation table of a schema.

    Args:
        schema: Schema class or registry key.

    Returns:
        A DataFrame with one row per column (dtype, nullable, unique, checks).
    """
    resolved = SCHEMAS[schema] if isinstance(schema, str) else schema
    container = resolved.to_schema()
    rows: list[dict[str, Any]] = []
    for name, column in container.columns.items():
        rows.append(
            {
                "column": name,
                "dtype": str(getattr(column, "dtype", "")),
                "nullable": bool(getattr(column, "nullable", False)),
                "required": bool(getattr(column, "required", True)),
                "unique": bool(getattr(column, "unique", False)),
                "checks": "; ".join(str(check) for check in getattr(column, "checks", []) or []),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def schema_to_markdown(schema: type[pa.DataFrameModel] | str = RawDataSchema) -> str:
    """Render a schema as a Markdown table (used in reports and notebooks).

    Args:
        schema: Schema class or registry key.

    Returns:
        The Markdown table.
    """
    description = describe_schema(schema)
    resolved = SCHEMAS[schema] if isinstance(schema, str) else schema
    lines = [f"### `{resolved.__name__}`", "", "| Colonne | Type | Nullable | Requis | Unique | Checks |", "| --- | --- | --- | --- | --- | --- |"]
    for name, row in description.iterrows():
        lines.append(
            f"| `{name}` | {row['dtype']} | {row['nullable']} | {row['required']} | {row['unique']} | {row['checks'] or '-'} |"
        )
    return "\n".join(lines)


def validation_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Produce a compact data-quality snapshot of a frame (for reports and notebooks).

    Args:
        frame: DataFrame to profile.

    Returns:
        Mapping with shape, dtypes, missing values and duplicate statistics.
    """
    missing = frame.isna().sum()
    n_cells = int(frame.size)
    return {
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_cells": int(missing.sum()),
        # Proportion de *cellules* manquantes (et non moyenne des taux par colonne) :
        # la valeur reste donc toujours dans [0, 1].
        "missing_rate": float(int(missing.sum()) / n_cells) if n_cells else 0.0,
        "columns_with_missing": {str(column): int(value) for column, value in missing[missing > 0].items()},
        "duplicate_ids": int(frame["sample_id"].duplicated().sum()),
        "memory_kb": float(frame.memory_usage(deep=True).sum() / 1024),
    }
