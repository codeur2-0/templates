"""Contrats de données — Biens immobiliers résidentiels et prix de vente.

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
DATASET_NAME = "real_estate_prices"

#: Colonnes attendues et leur type logique — source de vérité pour la documentation.
EXPECTED_DTYPES: dict[str, str] = {"property_id": "str", "listing_date": "datetime", "surface_m2": "float", "rooms": "int", "floor_level": "int", "has_elevator": "bool", "has_outdoor_space": "bool", "building_year": "int", "energy_rating": "category", "district": "category", "transport_walk_min": "int", "condo_fees_eur": "float", "property_tax_eur": "float", "condition_score": "float", "recent_sales_1km": "int", "price_eur": "float"}

#: Colonnes utilisées comme features par le modèle.
FEATURE_COLUMNS: list[str] = ['surface_m2', 'rooms', 'floor_level', 'has_elevator', 'has_outdoor_space', 'building_year', 'energy_rating', 'district', 'transport_walk_min', 'condo_fees_eur', 'property_tax_eur', 'condition_score', 'recent_sales_1km']

#: Colonne cible.
TARGET_COLUMN: str = "price_eur"

#: Clé métier (unique).
ID_COLUMN: str = "property_id"

#: Colonnes non modélisables (identifiants, horodatage, métadonnées).
NON_FEATURE_COLUMNS: list[str] = ['property_id', 'listing_date', 'price_eur']


class RawDataSchema(pa.DataFrameModel):
    """Contrat des **données brutes** (``data/raw/real_estate_prices.parquet``).

    Chaque déclaration ci-dessous correspond à une colonne du schéma documenté dans
    ``data/README.md``. ``coerce=True`` convertit les types compatibles (utile quand le
    dataset est lu depuis un CSV plutôt que depuis Parquet) ; ``strict=True`` refuse toute
    colonne non déclarée, ce qui détecte immédiatement une évolution silencieuse de la source.
    """

    property_id: Series[str] = pa.Field(nullable=False, str_matches='^PRP-[0-9]{6}$', description="Identifiant unique du bien")
    listing_date: Series[pa.DateTime] = pa.Field(nullable=False, description="Date de mise en annonce du bien")
    surface_m2: Series[float] = pa.Field(nullable=False, ge=12.0, le=320.0, description="Surface habitable")
    rooms: Series[int] = pa.Field(nullable=False, ge=1, le=8, description="Nombre de pièces principales")
    floor_level: Series[int] = pa.Field(nullable=False, ge=0, le=12, description="Étage du bien (0 = rez-de-chaussée)")
    has_elevator: Series[int] = pa.Field(nullable=False, isin=[0, 1], description="Présence d'un ascenseur dans l'immeuble (1 = oui)")
    has_outdoor_space: Series[int] = pa.Field(nullable=False, isin=[0, 1], description="Balcon, terrasse ou jardin (1 = oui)")
    building_year: Series[int] = pa.Field(nullable=False, ge=1850, le=2025, description="Année de construction de l'immeuble")
    energy_rating: Series[str] = pa.Field(nullable=False, isin=['A', 'B', 'C', 'D', 'E', 'F', 'G'], description="Classe énergétique du diagnostic de performance énergétique (DPE)")
    district: Series[str] = pa.Field(nullable=False, isin=['centre_historique', 'bords_de_loire', 'quartier_universitaire', 'quartier_affaires', 'peripherie_nord', 'peripherie_sud'], description="Quartier (segment géographique) de rattachement du bien")
    transport_walk_min: Series[int] = pa.Field(nullable=False, ge=1, le=45, description="Temps d'accès piéton au transport structurant le plus proche")
    condo_fees_eur: Series[float] = pa.Field(nullable=True, ge=0.0, le=900.0, description="Charges de copropriété mensuelles")
    property_tax_eur: Series[float] = pa.Field(nullable=False, ge=120.0, le=6500.0, description="Taxe foncière annuelle")
    condition_score: Series[float] = pa.Field(nullable=True, ge=1.0, le=10.0, description="État déclaré par le vendeur (1 = à rénover entièrement, 10 = haut de gamme rénové)")
    recent_sales_1km: Series[int] = pa.Field(nullable=False, ge=0, le=140, description="Nombre de ventes comparables enregistrées à moins d'un kilomètre sur 12 mois")
    price_eur: Series[float] = pa.Field(nullable=False, ge=35000.0, le=2500000.0, description="Prix de vente net vendeur observé")

    class Config:
        """Politique de validation des données brutes."""

        name = "RawDataSchema"
        coerce = True
        strict = True
        ordered = False
        unique = ["property_id"]

    @pa.dataframe_check
    @classmethod
    def target_is_not_constant(cls, frame: DataFrame) -> bool:
        """Une cible constante rend tout modèle inutilisable (variance nulle)."""
        if TARGET_COLUMN is None or TARGET_COLUMN not in frame.columns or frame.empty:
            return True
        return float(frame[TARGET_COLUMN].std() or 0.0) > 0.0


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

    property_id: Series[str] = pa.Field(nullable=True, required=False, str_matches='^PRP-[0-9]{6}$', description="Identifiant unique du bien")
    listing_date: Series[pa.DateTime] = pa.Field(nullable=False, required=False, description="Date de mise en annonce du bien")
    surface_m2: Series[float] = pa.Field(nullable=True, required=False, ge=12.0, le=320.0, description="Surface habitable")
    rooms: Series[int] = pa.Field(nullable=True, required=False, ge=1, le=8, description="Nombre de pièces principales")
    floor_level: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=12, description="Étage du bien (0 = rez-de-chaussée)")
    has_elevator: Series[int] = pa.Field(nullable=False, required=False, isin=[0, 1], description="Présence d'un ascenseur dans l'immeuble (1 = oui)")
    has_outdoor_space: Series[int] = pa.Field(nullable=False, required=False, isin=[0, 1], description="Balcon, terrasse ou jardin (1 = oui)")
    building_year: Series[int] = pa.Field(nullable=True, required=False, ge=1850, le=2025, description="Année de construction de l'immeuble")
    energy_rating: Series[str] = pa.Field(nullable=False, required=False, isin=['A', 'B', 'C', 'D', 'E', 'F', 'G'], description="Classe énergétique du diagnostic de performance énergétique (DPE)")
    district: Series[str] = pa.Field(nullable=False, required=False, isin=['centre_historique', 'bords_de_loire', 'quartier_universitaire', 'quartier_affaires', 'peripherie_nord', 'peripherie_sud'], description="Quartier (segment géographique) de rattachement du bien")
    transport_walk_min: Series[int] = pa.Field(nullable=True, required=False, ge=1, le=45, description="Temps d'accès piéton au transport structurant le plus proche")
    condo_fees_eur: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=900.0, description="Charges de copropriété mensuelles")
    property_tax_eur: Series[float] = pa.Field(nullable=True, required=False, ge=120.0, le=6500.0, description="Taxe foncière annuelle")
    condition_score: Series[float] = pa.Field(nullable=True, required=False, ge=1.0, le=10.0, description="État déclaré par le vendeur (1 = à rénover entièrement, 10 = haut de gamme rénové)")
    recent_sales_1km: Series[int] = pa.Field(nullable=True, required=False, ge=0, le=140, description="Nombre de ventes comparables enregistrées à moins d'un kilomètre sur 12 mois")

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
        "duplicate_ids": int(frame["property_id"].duplicated().sum()),
        "memory_kb": float(frame.memory_usage(deep=True).sum() / 1024),
    }
