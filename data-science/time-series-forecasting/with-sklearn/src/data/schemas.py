"""Contrats de données — Consommation électrique régionale et prévision multi-horizons.

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
DATASET_NAME = "regional_electricity_load"

#: Colonnes attendues et leur type logique — source de vérité pour la documentation.
EXPECTED_DTYPES: dict[str, str] = {"sample_id": "str", "origin_date": "datetime", "horizon_days": "int", "target_date": "datetime", "target_month": "str", "event_type": "str", "is_extreme_event": "bool", "load_mw": "float", "load_last_observed": "float", "load_lag_1d": "float", "load_lag_2d": "float", "load_lag_7d": "float", "load_lag_14d": "float", "load_lag_28d": "float", "load_rolling_mean_7d": "float", "load_rolling_std_7d": "float", "load_rolling_min_7d": "float", "load_rolling_mean_28d": "float", "load_seasonal_naive": "float", "temperature_forecast_c": "float", "temperature_forecast_prev_c": "float", "temperature_anomaly_c": "float", "temperature_lag_1d": "float", "temperature_rolling_mean_7d": "float", "hdd_target": "float", "cdd_target": "float", "target_weekday": "str", "target_is_weekend": "bool", "target_is_holiday": "bool", "target_school_holiday": "bool", "target_doy_sin": "float", "target_doy_cos": "float"}

#: Colonnes utilisées comme features par le modèle.
FEATURE_COLUMNS: list[str] = ['horizon_days', 'load_last_observed', 'load_lag_1d', 'load_lag_2d', 'load_lag_7d', 'load_lag_14d', 'load_lag_28d', 'load_rolling_mean_7d', 'load_rolling_std_7d', 'load_rolling_min_7d', 'load_rolling_mean_28d', 'load_seasonal_naive', 'temperature_forecast_c', 'temperature_forecast_prev_c', 'temperature_anomaly_c', 'temperature_lag_1d', 'temperature_rolling_mean_7d', 'hdd_target', 'cdd_target', 'target_weekday', 'target_is_weekend', 'target_is_holiday', 'target_school_holiday', 'target_doy_sin', 'target_doy_cos']

#: Colonne cible.
TARGET_COLUMN: str = "load_mw"

#: Clé métier (unique).
ID_COLUMN: str = "sample_id"


#: Colonnes non modélisables (identifiants, horodatage, métadonnées).
NON_FEATURE_COLUMNS: list[str] = ['sample_id', 'origin_date', 'target_date', 'target_month', 'event_type', 'is_extreme_event', 'load_mw']


class RawDataSchema(pa.DataFrameModel):
    """Contrat des **données brutes** (``data/raw/regional_electricity_load.parquet``).

    Chaque déclaration ci-dessous correspond à une colonne du schéma documenté dans
    ``data/README.md``. ``coerce=True`` convertit les types compatibles (utile quand le
    dataset est lu depuis un CSV plutôt que depuis Parquet) ; ``strict=True`` refuse toute
    colonne non déclarée, ce qui détecte immédiatement une évolution silencieuse de la source.
    """

    sample_id: Series[str] = pa.Field(nullable=False, str_matches='^FC-[0-9]{5}-H[0-9]{1,2}$', description="Identifiant unique du couple (origine, horizon)")
    origin_date: Series[pa.DateTime] = pa.Field(nullable=False, description="Date de l'origine : dernier jour dont la consommation est connue au moment de la prévision")
    horizon_days: Series[int] = pa.Field(nullable=False, isin=[1, 2, 3, 7], description="Horizon visé en jours (1, 2, 3 ou 7) — le modèle apprend la dégradation avec l'horizon")
    target_date: Series[pa.DateTime] = pa.Field(nullable=False, description="Date prévue (origine + horizon) — sert à l'analyse d'erreur, jamais au modèle")
    target_month: Series[str] = pa.Field(nullable=False, isin=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], description="Mois du jour cible, pour la ventilation saisonnière des erreurs")
    event_type: Series[str] = pa.Field(nullable=False, isin=['none', 'cold_snap', 'heatwave', 'industrial_shutdown'], description="Nature de l'épisode touchant le jour cible (aucun, vague de froid, canicule, arrêt industriel)")
    is_extreme_event: Series[int] = pa.Field(nullable=False, description="Vrai si le jour cible appartient à un épisode extrême — cible prioritaire de l'analyse d'erreur")
    load_mw: Series[float] = pa.Field(nullable=False, ge=200.0, le=8000.0, description="Consommation électrique régionale observée le jour cible")
    load_last_observed: Series[float] = pa.Field(nullable=False, ge=150.0, le=8500.0, description="Dernière consommation connue : le jour de l'origine lui-même (ancre de persistance)")
    load_lag_1d: Series[float] = pa.Field(nullable=False, ge=150.0, le=8500.0, description="Consommation un jour avant l'origine")
    load_lag_2d: Series[float] = pa.Field(nullable=False, ge=150.0, le=8500.0, description="Consommation deux jours avant l'origine")
    load_lag_7d: Series[float] = pa.Field(nullable=False, ge=150.0, le=8500.0, description="Consommation sept jours avant l'origine (même jour de la semaine que l'origine)")
    load_lag_14d: Series[float] = pa.Field(nullable=False, ge=150.0, le=8500.0, description="Consommation quatorze jours avant l'origine")
    load_lag_28d: Series[float] = pa.Field(nullable=False, ge=150.0, le=8500.0, description="Consommation vingt-huit jours avant l'origine (niveau de référence mensuel)")
    load_rolling_mean_7d: Series[float] = pa.Field(nullable=False, ge=150.0, le=8500.0, description="Moyenne glissante 7 jours de la consommation, terminée à l'origine")
    load_rolling_std_7d: Series[float] = pa.Field(nullable=False, ge=0.0, le=2000.0, description="Écart-type glissant 7 jours : volatilité récente du réseau")
    load_rolling_min_7d: Series[float] = pa.Field(nullable=False, ge=150.0, le=8500.0, description="Minimum glissant 7 jours : plancher récent (détecte un creux de vacances)")
    load_rolling_mean_28d: Series[float] = pa.Field(nullable=False, ge=150.0, le=8500.0, description="Moyenne glissante 28 jours : niveau de fond mensuel")
    load_seasonal_naive: Series[float] = pa.Field(nullable=False, ge=150.0, le=8500.0, description="Ancrage saisonnier : consommation du même jour de la semaine, une semaine avant le jour cible (toujours ≤ origine, donc licite)")
    temperature_forecast_c: Series[float] = pa.Field(nullable=False, ge=-25.0, le=48.0, description="Prévision de température moyenne pour le jour cible, disponible à l'origine — entachée d'une erreur croissante avec l'horizon")
    temperature_forecast_prev_c: Series[float] = pa.Field(nullable=True, ge=-25.0, le=48.0, description="Prévision de température de la veille du jour cible (observation à J+1) — permet au modèle de reconstruire l'inertie thermique du bâti")
    temperature_anomaly_c: Series[float] = pa.Field(nullable=False, ge=-25.0, le=25.0, description="Écart de la prévision de température à la normale saisonnière du jour cible")
    temperature_lag_1d: Series[float] = pa.Field(nullable=True, ge=-25.0, le=48.0, description="Température observée à l'origine")
    temperature_rolling_mean_7d: Series[float] = pa.Field(nullable=True, ge=-25.0, le=48.0, description="Moyenne glissante 7 jours de la température observée, terminée à l'origine")
    hdd_target: Series[float] = pa.Field(nullable=False, ge=0.0, le=40.0, description="Degrés-jours de chauffe du jour cible : max(0, 18 °C - température prévue)")
    cdd_target: Series[float] = pa.Field(nullable=False, ge=0.0, le=25.0, description="Degrés-jours de froid du jour cible : max(0, température prévue - 24 °C)")
    target_weekday: Series[str] = pa.Field(nullable=False, isin=['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'], description="Jour de la semaine du jour cible (connu par avance)")
    target_is_weekend: Series[int] = pa.Field(nullable=False, description="Vrai si le jour cible tombe un samedi ou un dimanche")
    target_is_holiday: Series[int] = pa.Field(nullable=False, description="Vrai si le jour cible est un jour férié national français")
    target_school_holiday: Series[int] = pa.Field(nullable=False, description="Vrai si le jour cible tombe pendant les vacances scolaires (effet résidentiel marqué)")
    target_doy_sin: Series[float] = pa.Field(nullable=False, ge=-1.0, le=1.0, description="Composante sinus du jour de l'année cible (encodage cyclique de la saisonnalité annuelle)")
    target_doy_cos: Series[float] = pa.Field(nullable=False, ge=-1.0, le=1.0, description="Composante cosinus du jour de l'année cible")

    class Config:
        """Politique de validation des données brutes."""

        name = "RawDataSchema"
        coerce = True
        strict = True
        ordered = False
        unique = ["sample_id"]

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

    sample_id: Series[str] = pa.Field(nullable=True, required=False, str_matches='^FC-[0-9]{5}-H[0-9]{1,2}$', description="Identifiant unique du couple (origine, horizon)")
    origin_date: Series[pa.DateTime] = pa.Field(nullable=False, required=False, description="Date de l'origine : dernier jour dont la consommation est connue au moment de la prévision")
    horizon_days: Series[int] = pa.Field(nullable=True, required=False, isin=[1, 2, 3, 7], description="Horizon visé en jours (1, 2, 3 ou 7) — le modèle apprend la dégradation avec l'horizon")
    target_date: Series[pa.DateTime] = pa.Field(nullable=False, required=False, description="Date prévue (origine + horizon) — sert à l'analyse d'erreur, jamais au modèle")
    target_month: Series[str] = pa.Field(nullable=True, required=False, isin=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], description="Mois du jour cible, pour la ventilation saisonnière des erreurs")
    event_type: Series[str] = pa.Field(nullable=True, required=False, isin=['none', 'cold_snap', 'heatwave', 'industrial_shutdown'], description="Nature de l'épisode touchant le jour cible (aucun, vague de froid, canicule, arrêt industriel)")
    is_extreme_event: Series[int] = pa.Field(nullable=False, required=False, description="Vrai si le jour cible appartient à un épisode extrême — cible prioritaire de l'analyse d'erreur")
    load_last_observed: Series[float] = pa.Field(nullable=True, required=False, ge=150.0, le=8500.0, description="Dernière consommation connue : le jour de l'origine lui-même (ancre de persistance)")
    load_lag_1d: Series[float] = pa.Field(nullable=True, required=False, ge=150.0, le=8500.0, description="Consommation un jour avant l'origine")
    load_lag_2d: Series[float] = pa.Field(nullable=True, required=False, ge=150.0, le=8500.0, description="Consommation deux jours avant l'origine")
    load_lag_7d: Series[float] = pa.Field(nullable=True, required=False, ge=150.0, le=8500.0, description="Consommation sept jours avant l'origine (même jour de la semaine que l'origine)")
    load_lag_14d: Series[float] = pa.Field(nullable=True, required=False, ge=150.0, le=8500.0, description="Consommation quatorze jours avant l'origine")
    load_lag_28d: Series[float] = pa.Field(nullable=True, required=False, ge=150.0, le=8500.0, description="Consommation vingt-huit jours avant l'origine (niveau de référence mensuel)")
    load_rolling_mean_7d: Series[float] = pa.Field(nullable=True, required=False, ge=150.0, le=8500.0, description="Moyenne glissante 7 jours de la consommation, terminée à l'origine")
    load_rolling_std_7d: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=2000.0, description="Écart-type glissant 7 jours : volatilité récente du réseau")
    load_rolling_min_7d: Series[float] = pa.Field(nullable=True, required=False, ge=150.0, le=8500.0, description="Minimum glissant 7 jours : plancher récent (détecte un creux de vacances)")
    load_rolling_mean_28d: Series[float] = pa.Field(nullable=True, required=False, ge=150.0, le=8500.0, description="Moyenne glissante 28 jours : niveau de fond mensuel")
    load_seasonal_naive: Series[float] = pa.Field(nullable=True, required=False, ge=150.0, le=8500.0, description="Ancrage saisonnier : consommation du même jour de la semaine, une semaine avant le jour cible (toujours ≤ origine, donc licite)")
    temperature_forecast_c: Series[float] = pa.Field(nullable=True, required=False, ge=-25.0, le=48.0, description="Prévision de température moyenne pour le jour cible, disponible à l'origine — entachée d'une erreur croissante avec l'horizon")
    temperature_forecast_prev_c: Series[float] = pa.Field(nullable=True, required=False, ge=-25.0, le=48.0, description="Prévision de température de la veille du jour cible (observation à J+1) — permet au modèle de reconstruire l'inertie thermique du bâti")
    temperature_anomaly_c: Series[float] = pa.Field(nullable=True, required=False, ge=-25.0, le=25.0, description="Écart de la prévision de température à la normale saisonnière du jour cible")
    temperature_lag_1d: Series[float] = pa.Field(nullable=True, required=False, ge=-25.0, le=48.0, description="Température observée à l'origine")
    temperature_rolling_mean_7d: Series[float] = pa.Field(nullable=True, required=False, ge=-25.0, le=48.0, description="Moyenne glissante 7 jours de la température observée, terminée à l'origine")
    hdd_target: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=40.0, description="Degrés-jours de chauffe du jour cible : max(0, 18 °C - température prévue)")
    cdd_target: Series[float] = pa.Field(nullable=True, required=False, ge=0.0, le=25.0, description="Degrés-jours de froid du jour cible : max(0, température prévue - 24 °C)")
    target_weekday: Series[str] = pa.Field(nullable=True, required=False, isin=['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'], description="Jour de la semaine du jour cible (connu par avance)")
    target_is_weekend: Series[int] = pa.Field(nullable=False, required=False, description="Vrai si le jour cible tombe un samedi ou un dimanche")
    target_is_holiday: Series[int] = pa.Field(nullable=False, required=False, description="Vrai si le jour cible est un jour férié national français")
    target_school_holiday: Series[int] = pa.Field(nullable=False, required=False, description="Vrai si le jour cible tombe pendant les vacances scolaires (effet résidentiel marqué)")
    target_doy_sin: Series[float] = pa.Field(nullable=True, required=False, ge=-1.0, le=1.0, description="Composante sinus du jour de l'année cible (encodage cyclique de la saisonnalité annuelle)")
    target_doy_cos: Series[float] = pa.Field(nullable=True, required=False, ge=-1.0, le=1.0, description="Composante cosinus du jour de l'année cible")

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
