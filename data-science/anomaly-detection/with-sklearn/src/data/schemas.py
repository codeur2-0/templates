"""Contrats de données — Transactions de paiement et fraude observée.

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
DATASET_NAME = "payment_transactions"

#: Colonnes attendues et leur type logique — source de vérité pour la documentation.
EXPECTED_DTYPES: dict[str, str] = {"transaction_id": "str", "occurred_at": "datetime", "amount_eur": "float", "merchant_category": "category", "channel": "category", "shopper_country": "category", "billing_country": "category", "card_age_months": "int", "account_tenure_months": "int", "transactions_24h": "int", "distinct_merchants_24h": "int", "distinct_countries_7d": "int", "failed_attempts_1h": "int", "device_age_days": "float", "session_duration_sec": "float", "billing_shipping_distance_km": "float", "amount_to_customer_avg_ratio": "float", "is_night": "bool", "three_ds_authenticated": "bool", "previous_chargebacks_12m": "int", "is_fraud": "int", "fraud_scheme": "str"}

#: Colonnes utilisées comme features par le modèle.
FEATURE_COLUMNS: list[str] = ['amount_eur', 'merchant_category', 'channel', 'shopper_country', 'billing_country', 'card_age_months', 'account_tenure_months', 'transactions_24h', 'distinct_merchants_24h', 'distinct_countries_7d', 'failed_attempts_1h', 'device_age_days', 'session_duration_sec', 'billing_shipping_distance_km', 'amount_to_customer_avg_ratio', 'is_night', 'three_ds_authenticated', 'previous_chargebacks_12m']

#: Pas de cible : apprentissage non supervisé.
TARGET_COLUMN: str | None = None

#: Clé métier (unique).
ID_COLUMN: str = "transaction_id"


#: Colonnes non modélisables (identifiants, horodatage, métadonnées).
NON_FEATURE_COLUMNS: list[str] = ['transaction_id', 'occurred_at', 'is_fraud', 'fraud_scheme']


class RawDataSchema(pa.DataFrameModel):
    """Contrat des **données brutes** (``data/raw/payment_transactions.parquet``).

    Chaque déclaration ci-dessous correspond à une colonne du schéma documenté dans
    ``data/README.md``. ``coerce=True`` convertit les types compatibles (utile quand le
    dataset est lu depuis un CSV plutôt que depuis Parquet) ; ``strict=True`` refuse toute
    colonne non déclarée, ce qui détecte immédiatement une évolution silencieuse de la source.
    """

    transaction_id: Series[str] = pa.Field(nullable=False, str_matches='^TXN-[0-9]{7}$', description="Identifiant unique de la transaction")
    occurred_at: Series[pa.DateTime] = pa.Field(nullable=False, description="Horodatage de la tentative de paiement")
    amount_eur: Series[float] = pa.Field(nullable=False, ge=0.5, le=12000.0, description="Montant de la transaction")
    merchant_category: Series[str] = pa.Field(nullable=False, isin=['grocery', 'electronics', 'travel', 'gaming', 'jewelry', 'utilities', 'fashion', 'restaurant'], description="Catégorie du marchand (MCC agrégé)")
    channel: Series[str] = pa.Field(nullable=False, isin=['web', 'mobile_app', 'in_store', 'phone'], description="Canal de saisie du paiement")
    shopper_country: Series[str] = pa.Field(nullable=False, isin=['FR', 'BE', 'DE', 'ES', 'IT', 'LU', 'PT', 'other'], description="Pays de la session d'achat")
    billing_country: Series[str] = pa.Field(nullable=False, isin=['FR', 'BE', 'DE', 'ES', 'IT', 'LU', 'PT', 'other'], description="Pays de facturation de la carte")
    card_age_months: Series[int] = pa.Field(nullable=False, ge=0, le=120, description="Ancienneté de la carte utilisée")
    account_tenure_months: Series[int] = pa.Field(nullable=False, ge=0, le=108, description="Ancienneté du compte client chez le PSP")
    transactions_24h: Series[int] = pa.Field(nullable=False, ge=1, le=60, description="Nombre de tentatives de paiement sur la carte en 24 h")
    distinct_merchants_24h: Series[int] = pa.Field(nullable=False, ge=1, le=40, description="Marchands distincts contactés par la carte en 24 h")
    distinct_countries_7d: Series[int] = pa.Field(nullable=False, ge=1, le=9, description="Pays distincts observés sur la carte en 7 jours")
    failed_attempts_1h: Series[int] = pa.Field(nullable=False, ge=0, le=25, description="Tentatives refusées dans l'heure précédente")
    device_age_days: Series[float] = pa.Field(nullable=True, ge=0, le=1080, description="Âge de l'empreinte d'appareil (première fois vue) — flottant car la colonne est nullable")
    session_duration_sec: Series[float] = pa.Field(nullable=True, ge=2.0, le=3600.0, description="Durée de la session avant paiement")
    billing_shipping_distance_km: Series[float] = pa.Field(nullable=True, ge=0.0, le=14000.0, description="Distance entre adresses de facturation et de livraison")
    amount_to_customer_avg_ratio: Series[float] = pa.Field(nullable=False, ge=0.02, le=60.0, description="Rapport du montant au panier moyen historique du porteur")
    is_night: Series[int] = pa.Field(nullable=False, isin=[0, 1], description="Transaction entre 1 h et 5 h (heure locale)")
    three_ds_authenticated: Series[int] = pa.Field(nullable=False, isin=[0, 1], description="Authentification forte 3-D Secure aboutie")
    previous_chargebacks_12m: Series[int] = pa.Field(nullable=False, ge=0, le=6, description="Chargebacks confirmés sur la carte dans les 12 derniers mois")
    is_fraud: Series[int] = pa.Field(nullable=False, isin=[0, 1], description="Fraude confirmée par chargeback (étiquette de diagnostic, JAMAIS une feature)")
    fraud_scheme: Series[str] = pa.Field(nullable=True, isin=['card_not_present', 'account_takeover', 'synthetic_identity', 'friendly_fraud'], description="Mode opératoire de la fraude confirmée (diagnostic pédagogique et explication)")

    class Config:
        """Politique de validation des données brutes."""

        name = "RawDataSchema"
        coerce = True
        strict = True
        ordered = False
        unique = ["transaction_id"]



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

    transaction_id: Series[str] = pa.Field(nullable=True, required=False, str_matches='^TXN-[0-9]{7}$', description="Identifiant unique de la transaction")
    occurred_at: Series[pa.DateTime] = pa.Field(nullable=False, required=False, description="Horodatage de la tentative de paiement")
    amount_eur: Series[float] = pa.Field(nullable=True, required=False, ge=0.5, le=12000.0, description="Montant de la transaction")
    merchant_category: Series[str] = pa.Field(nullable=False, required=False, isin=['grocery', 'electronics', 'travel', 'gaming', 'jewelry', 'utilities', 'fashion', 'restaurant'], description="Catégorie du marchand (MCC agrégé)")
    channel: Series[str] = pa.Field(nullable=False, required=False, isin=['web', 'mobile_app', 'in_store', 'phone'], description="Canal de saisie du paiement")
    shopper_country: Series[str] = pa.Field(nullable=False, required=False, isin=['FR', 'BE', 'DE', 'ES', 'IT', 'LU', 'PT', 'other'], description="Pays de la session d'achat")
    billing_country: Series[str] = pa.Field(nullable=False, required=False, isin=['FR', 'BE', 'DE', 'ES', 'IT', 'LU', 'PT', 'other'], description="Pays de facturation de la carte")
    card_age_months: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=120, description="Ancienneté de la carte utilisée")
    account_tenure_months: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=108, description="Ancienneté du compte client chez le PSP")
    transactions_24h: Series[int] = pa.Field(nullable=True, required=False, ge=1, le=60, description="Nombre de tentatives de paiement sur la carte en 24 h")
    distinct_merchants_24h: Series[int] = pa.Field(nullable=True, required=False, ge=1, le=40, description="Marchands distincts contactés par la carte en 24 h")
    distinct_countries_7d: Series[int] = pa.Field(nullable=True, required=False, ge=1, le=9, description="Pays distincts observés sur la carte en 7 jours")
    failed_attempts_1h: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=25, description="Tentatives refusées dans l'heure précédente")
    device_age_days: Series[float] = pa.Field(nullable=True, required=False, ge=0, le=1080, description="Âge de l'empreinte d'appareil (première fois vue) — flottant car la colonne est nullable")
    session_duration_sec: Series[float] = pa.Field(nullable=True, required=False, ge=2.0, le=3600.0, description="Durée de la session avant paiement")
    billing_shipping_distance_km: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=14000.0, description="Distance entre adresses de facturation et de livraison")
    amount_to_customer_avg_ratio: Series[float] = pa.Field(nullable=True, required=False, ge=0.02, le=60.0, description="Rapport du montant au panier moyen historique du porteur")
    is_night: Series[int] = pa.Field(nullable=False, required=False, isin=[0, 1], description="Transaction entre 1 h et 5 h (heure locale)")
    three_ds_authenticated: Series[int] = pa.Field(nullable=False, required=False, isin=[0, 1], description="Authentification forte 3-D Secure aboutie")
    previous_chargebacks_12m: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=6, description="Chargebacks confirmés sur la carte dans les 12 derniers mois")
    is_fraud: Series[int] = pa.Field(nullable=True, required=False, isin=[0, 1], description="Fraude confirmée par chargeback (étiquette de diagnostic, JAMAIS une feature)")
    fraud_scheme: Series[str] = pa.Field(nullable=True, required=False, isin=['card_not_present', 'account_takeover', 'synthetic_identity', 'friendly_fraud'], description="Mode opératoire de la fraude confirmée (diagnostic pédagogique et explication)")

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
        "duplicate_ids": int(frame["transaction_id"].duplicated().sum()),
        "memory_kb": float(frame.memory_usage(deep=True).sum() / 1024),
    }
