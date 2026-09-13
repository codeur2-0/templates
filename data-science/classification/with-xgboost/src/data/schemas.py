"""Contrats de données — Abonnés télécom et attrition (churn).

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
DATASET_NAME = "telecom_churn"

#: Colonnes attendues et leur type logique — source de vérité pour la documentation.
EXPECTED_DTYPES: dict[str, str] = {"customer_id": "str", "signup_date": "datetime", "tenure_months": "int", "contract_type": "category", "internet_service": "category", "payment_method": "category", "region": "category", "monthly_charges": "float", "total_charges": "float", "support_tickets_6m": "int", "avg_monthly_data_gb": "float", "num_products": "int", "has_promotion": "bool", "satisfaction_score": "float", "churned": "int"}

#: Colonnes utilisées comme features par le modèle.
FEATURE_COLUMNS: list[str] = ['tenure_months', 'contract_type', 'internet_service', 'payment_method', 'region', 'monthly_charges', 'total_charges', 'support_tickets_6m', 'avg_monthly_data_gb', 'num_products', 'has_promotion', 'satisfaction_score']

#: Colonne cible.
TARGET_COLUMN: str = "churned"

#: Clé métier (unique).
ID_COLUMN: str = "customer_id"

#: Colonnes non modélisables (identifiants, horodatage, métadonnées).
NON_FEATURE_COLUMNS: list[str] = ['customer_id', 'signup_date', 'churned']


class RawDataSchema(pa.DataFrameModel):
    """Contrat des **données brutes** (``data/raw/telecom_churn.parquet``).

    Chaque déclaration ci-dessous correspond à une colonne du schéma documenté dans
    ``data/README.md``. ``coerce=True`` convertit les types compatibles (utile quand le
    dataset est lu depuis un CSV plutôt que depuis Parquet) ; ``strict=True`` refuse toute
    colonne non déclarée, ce qui détecte immédiatement une évolution silencieuse de la source.
    """

    customer_id: Series[str] = pa.Field(nullable=False, str_matches='^CUS-[0-9]{5}$', description="Identifiant unique de l'abonné")
    signup_date: Series[pa.DateTime] = pa.Field(nullable=False, description="Date de souscription de l'abonnement")
    tenure_months: Series[int] = pa.Field(nullable=False, ge=0, le=72, description="Ancienneté de l'abonné")
    contract_type: Series[str] = pa.Field(nullable=False, isin=['month_to_month', 'one_year', 'two_year'], description="Type de contrat souscrit")
    internet_service: Series[str] = pa.Field(nullable=False, isin=['fiber', 'dsl', 'none'], description="Service internet associé à l'abonnement")
    payment_method: Series[str] = pa.Field(nullable=False, isin=['electronic_check', 'mailed_check', 'bank_transfer', 'credit_card'], description="Moyen de paiement utilisé")
    region: Series[str] = pa.Field(nullable=False, isin=['north', 'south', 'east', 'west'], description="Région commerciale de rattachement")
    monthly_charges: Series[float] = pa.Field(nullable=False, ge=18.0, le=480.0, description="Montant mensuel facturé")
    total_charges: Series[float] = pa.Field(nullable=False, ge=18.0, le=20000.0, description="Cumul facturé depuis la souscription")
    support_tickets_6m: Series[int] = pa.Field(nullable=False, ge=0, le=14, description="Nombre de tickets support ouverts sur 6 mois")
    avg_monthly_data_gb: Series[float] = pa.Field(nullable=True, ge=0.0, le=1000.0, description="Consommation data mensuelle moyenne")
    num_products: Series[int] = pa.Field(nullable=False, ge=1, le=5, description="Nombre de services/produits souscrits (bundle)")
    has_promotion: Series[int] = pa.Field(nullable=False, isin=[0, 1], description="Bénéficie d'une promotion active (1 = oui)")
    satisfaction_score: Series[float] = pa.Field(nullable=True, ge=1.0, le=10.0, description="Score de satisfaction déclaré (enquête CSAT)")
    churned: Series[int] = pa.Field(nullable=False, isin=[0, 1], description="Attrition observée dans les 30 jours (1 = parti)")

    class Config:
        """Politique de validation des données brutes."""

        name = "RawDataSchema"
        coerce = True
        strict = True
        ordered = False
        unique = ["customer_id"]

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

    customer_id: Series[str] = pa.Field(nullable=True, required=False, str_matches='^CUS-[0-9]{5}$', description="Identifiant unique de l'abonné")
    signup_date: Series[pa.DateTime] = pa.Field(nullable=False, required=False, description="Date de souscription de l'abonnement")
    tenure_months: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=72, description="Ancienneté de l'abonné")
    contract_type: Series[str] = pa.Field(nullable=False, required=False, isin=['month_to_month', 'one_year', 'two_year'], description="Type de contrat souscrit")
    internet_service: Series[str] = pa.Field(nullable=False, required=False, isin=['fiber', 'dsl', 'none'], description="Service internet associé à l'abonnement")
    payment_method: Series[str] = pa.Field(nullable=False, required=False, isin=['electronic_check', 'mailed_check', 'bank_transfer', 'credit_card'], description="Moyen de paiement utilisé")
    region: Series[str] = pa.Field(nullable=False, required=False, isin=['north', 'south', 'east', 'west'], description="Région commerciale de rattachement")
    monthly_charges: Series[float] = pa.Field(nullable=True, required=False, ge=18.0, le=480.0, description="Montant mensuel facturé")
    total_charges: Series[float] = pa.Field(nullable=True, required=False, ge=18.0, le=20000.0, description="Cumul facturé depuis la souscription")
    support_tickets_6m: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=14, description="Nombre de tickets support ouverts sur 6 mois")
    avg_monthly_data_gb: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=1000.0, description="Consommation data mensuelle moyenne")
    num_products: Series[int] = pa.Field(nullable=True, required=False, ge=1, le=5, description="Nombre de services/produits souscrits (bundle)")
    has_promotion: Series[int] = pa.Field(nullable=False, required=False, isin=[0, 1], description="Bénéficie d'une promotion active (1 = oui)")
    satisfaction_score: Series[float] = pa.Field(nullable=True, required=False, ge=1.0, le=10.0, description="Score de satisfaction déclaré (enquête CSAT)")

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
