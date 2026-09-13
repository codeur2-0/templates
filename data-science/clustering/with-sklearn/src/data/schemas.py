"""Contrats de données — Base clients e-commerce et comportement d'achat sur 12 mois.

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
DATASET_NAME = "retail_customer_base"

#: Colonnes attendues et leur type logique — source de vérité pour la documentation.
EXPECTED_DTYPES: dict[str, str] = {"customer_id": "str", "signup_date": "datetime", "last_order_date": "datetime", "tenure_months": "int", "recency_days": "int", "orders_12m": "int", "revenue_12m_eur": "float", "avg_basket_eur": "float", "distinct_categories_12m": "int", "discount_share": "float", "return_rate": "float", "support_tickets_12m": "int", "newsletter_opens_12m": "int", "web_sessions_12m": "int", "mobile_share": "float", "loyalty_tier": "category", "acquisition_channel": "category", "region": "category", "nps_score": "int", "opt_in_marketing": "bool", "latent_segment": "category", "churned_next_90d": "bool"}

#: Colonnes utilisées comme features par le modèle.
FEATURE_COLUMNS: list[str] = ['tenure_months', 'recency_days', 'orders_12m', 'revenue_12m_eur', 'avg_basket_eur', 'distinct_categories_12m', 'discount_share', 'return_rate', 'support_tickets_12m', 'newsletter_opens_12m', 'web_sessions_12m', 'mobile_share', 'loyalty_tier', 'acquisition_channel', 'region', 'nps_score', 'opt_in_marketing']

#: Pas de cible : apprentissage non supervisé.
TARGET_COLUMN: str | None = None

#: Clé métier (unique).
ID_COLUMN: str = "customer_id"

#: Colonnes non modélisables (identifiants, horodatage, métadonnées).
NON_FEATURE_COLUMNS: list[str] = ['customer_id', 'signup_date', 'last_order_date', 'latent_segment', 'churned_next_90d']


class RawDataSchema(pa.DataFrameModel):
    """Contrat des **données brutes** (``data/raw/retail_customer_base.parquet``).

    Chaque déclaration ci-dessous correspond à une colonne du schéma documenté dans
    ``data/README.md``. ``coerce=True`` convertit les types compatibles (utile quand le
    dataset est lu depuis un CSV plutôt que depuis Parquet) ; ``strict=True`` refuse toute
    colonne non déclarée, ce qui détecte immédiatement une évolution silencieuse de la source.
    """

    customer_id: Series[str] = pa.Field(nullable=False, str_matches='^CUS-[0-9]{6}$', description="Identifiant unique du client")
    signup_date: Series[pa.DateTime] = pa.Field(nullable=False, description="Date d'inscription (première création de compte)")
    last_order_date: Series[pa.DateTime] = pa.Field(nullable=False, description="Date de la dernière commande")
    tenure_months: Series[int] = pa.Field(nullable=False, ge=0, le=62, description="Ancienneté du compte en mois")
    recency_days: Series[int] = pa.Field(nullable=False, ge=1, le=730, description="Nombre de jours depuis la dernière commande")
    orders_12m: Series[int] = pa.Field(nullable=False, ge=0, le=60, description="Nombre de commandes sur 12 mois glissants")
    revenue_12m_eur: Series[float] = pa.Field(nullable=False, ge=0.0, le=15000.0, description="Chiffre d'affaires réalisé sur 12 mois glissants")
    avg_basket_eur: Series[float] = pa.Field(nullable=True, ge=0.0, le=950.0, description="Panier moyen (chiffre d'affaires / nombre de commandes)")
    distinct_categories_12m: Series[int] = pa.Field(nullable=False, ge=0, le=12, description="Nombre de catégories de produits achetées (largeur du catalogue)")
    discount_share: Series[float] = pa.Field(nullable=False, ge=0.0, le=1.0, description="Part du chiffre d'affaires réalisée avec une remise")
    return_rate: Series[float] = pa.Field(nullable=False, ge=0.0, le=1.0, description="Taux de retour (articles retournés / articles commandés)")
    support_tickets_12m: Series[int] = pa.Field(nullable=False, ge=0, le=15, description="Nombre de contacts au service client (réclamation, SAV, question livraison)")
    newsletter_opens_12m: Series[int] = pa.Field(nullable=False, ge=0, le=120, description="Nombre d'ouvertures de newsletter sur 12 mois")
    web_sessions_12m: Series[int] = pa.Field(nullable=False, ge=0, le=400, description="Nombre de sessions web ou application sur 12 mois")
    mobile_share: Series[float] = pa.Field(nullable=False, ge=0.0, le=1.0, description="Part des sessions réalisées sur mobile")
    loyalty_tier: Series[str] = pa.Field(nullable=False, isin=['none', 'silver', 'gold', 'platinum'], description="Palier du programme de fidélité")
    acquisition_channel: Series[str] = pa.Field(nullable=False, isin=['organic', 'paid_search', 'social', 'marketplace', 'referral', 'email'], description="Canal d'acquisition du client")
    region: Series[str] = pa.Field(nullable=False, isin=['ile_de_france', 'nord', 'ouest', 'sud_ouest', 'sud_est', 'est'], description="Région de livraison principale")
    nps_score: Series[int] = pa.Field(nullable=True, ge=0, le=10, description="Note de recommandation déclarée (Net Promoter Score individuel)")
    opt_in_marketing: Series[int] = pa.Field(nullable=False, isin=[0, 1], description="Consentement aux communications marketing (1 = oui)")
    latent_segment: Series[str] = pa.Field(nullable=False, isin=['vip_fidele', 'chasseur_promo', 'acheteur_occasionnel', 'dormeur', 'nouveau_curieux', 'client_insatisfait'], description="Profil latent injecté par le générateur — diagnostic pédagogique uniquement")
    churned_next_90d: Series[int] = pa.Field(nullable=False, isin=[0, 1], description="Aucune commande dans les 90 jours suivant la date de référence (validité externe)")

    class Config:
        """Politique de validation des données brutes."""

        name = "RawDataSchema"
        coerce = True
        strict = True
        ordered = False
        unique = ["customer_id"]



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

    customer_id: Series[str] = pa.Field(nullable=True, required=False, str_matches='^CUS-[0-9]{6}$', description="Identifiant unique du client")
    signup_date: Series[pa.DateTime] = pa.Field(nullable=False, required=False, description="Date d'inscription (première création de compte)")
    last_order_date: Series[pa.DateTime] = pa.Field(nullable=False, required=False, description="Date de la dernière commande")
    tenure_months: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=62, description="Ancienneté du compte en mois")
    recency_days: Series[int] = pa.Field(nullable=True, required=False, ge=1, le=730, description="Nombre de jours depuis la dernière commande")
    orders_12m: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=60, description="Nombre de commandes sur 12 mois glissants")
    revenue_12m_eur: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=15000.0, description="Chiffre d'affaires réalisé sur 12 mois glissants")
    avg_basket_eur: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=950.0, description="Panier moyen (chiffre d'affaires / nombre de commandes)")
    distinct_categories_12m: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=12, description="Nombre de catégories de produits achetées (largeur du catalogue)")
    discount_share: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=1.0, description="Part du chiffre d'affaires réalisée avec une remise")
    return_rate: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=1.0, description="Taux de retour (articles retournés / articles commandés)")
    support_tickets_12m: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=15, description="Nombre de contacts au service client (réclamation, SAV, question livraison)")
    newsletter_opens_12m: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=120, description="Nombre d'ouvertures de newsletter sur 12 mois")
    web_sessions_12m: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=400, description="Nombre de sessions web ou application sur 12 mois")
    mobile_share: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=1.0, description="Part des sessions réalisées sur mobile")
    loyalty_tier: Series[str] = pa.Field(nullable=False, required=False, isin=['none', 'silver', 'gold', 'platinum'], description="Palier du programme de fidélité")
    acquisition_channel: Series[str] = pa.Field(nullable=False, required=False, isin=['organic', 'paid_search', 'social', 'marketplace', 'referral', 'email'], description="Canal d'acquisition du client")
    region: Series[str] = pa.Field(nullable=False, required=False, isin=['ile_de_france', 'nord', 'ouest', 'sud_ouest', 'sud_est', 'est'], description="Région de livraison principale")
    nps_score: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=10, description="Note de recommandation déclarée (Net Promoter Score individuel)")
    opt_in_marketing: Series[int] = pa.Field(nullable=False, required=False, isin=[0, 1], description="Consentement aux communications marketing (1 = oui)")
    latent_segment: Series[str] = pa.Field(nullable=False, required=False, isin=['vip_fidele', 'chasseur_promo', 'acheteur_occasionnel', 'dormeur', 'nouveau_curieux', 'client_insatisfait'], description="Profil latent injecté par le générateur — diagnostic pédagogique uniquement")
    churned_next_90d: Series[int] = pa.Field(nullable=False, required=False, isin=[0, 1], description="Aucune commande dans les 90 jours suivant la date de référence (validité externe)")

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
        "duplicate_ids": int(frame["customer_id"].duplicated().sum()),
        "memory_kb": float(frame.memory_usage(deep=True).sum() / 1024),
    }
