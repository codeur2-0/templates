"""Tests des contrats de données Pandera.

Ces tests vérifient les deux propriétés qui donnent sa valeur à un contrat de données :

1. il **accepte** les données conformes (sinon il bloque la production pour rien),
2. il **refuse** les données corrompues avec un message exploitable (sinon il ne sert à rien).

Les tests sont volontairement génériques : ils découvrent les colonnes et leurs checks depuis
``RawDataSchema`` plutôt que de les coder en dur, ce qui les rend réutilisables quel que soit le
jeu de données du projet.
"""

from __future__ import annotations

from io import StringIO
from typing import Any

import numpy as np
import pandas as pd
import pytest

try:  # pandera >= 0.26
    import pandera.pandas as pa
except ModuleNotFoundError:  # pragma: no cover - pandera < 0.26
    import pandera as pa  # type: ignore[no-redef]

from src.data.schemas import (
    InferenceDataSchema,
    ProcessedDataSchema,
    RawDataSchema,
    describe_schema,
    schema_to_markdown,
    validate_frame,
    validation_report,
)

#: Pandera lève ``SchemaError`` (un check en échec) ou ``SchemaErrors`` (plusieurs erreurs de
#: conteneur, par exemple une colonne non déclarée). Les deux sont des violations du contrat.
SchemaViolation = (pa.errors.SchemaError, pa.errors.SchemaErrors)


def _failure_count(error: BaseException) -> int:
    """Count the individual violations carried by a pandera error.

    Args:
        error: The raised pandera error (``SchemaError`` or ``SchemaErrors``).

    Returns:
        The number of failure cases (at least 1).
    """
    cases = getattr(error, "failure_cases", None)
    if cases is not None and len(cases):
        return int(len(cases))
    sub_errors = list(getattr(error, "schema_errors", []) or [])
    return max(len(sub_errors), 1)


def _column_with_check(check_name: str) -> tuple[str, Any] | None:
    """Find the first raw column declaring a given check.

    Args:
        check_name: Name of the pandera check (``ge``, ``isin``, ``str_matches``, ...).

    Returns:
        The ``(column_name, check)`` pair, or ``None`` when no column declares it.
    """
    for name, column in RawDataSchema.to_schema().columns.items():
        for check in column.checks or []:
            if getattr(check, "name", "") == check_name:
                return name, check
    return None


def _categorical_column() -> str | None:
    """Find the first column constrained by an ``isin`` check."""
    found = _column_with_check("isin")
    return None if found is None else found[0]


def _bounded_numeric_column() -> str | None:
    """Find the first column constrained by a ``ge``/``greater_than_or_equal_to`` check."""
    for check_name in ("ge", "greater_than_or_equal_to", "in_range"):
        found = _column_with_check(check_name)
        if found is not None:
            return found[0]
    return None


class TestRawSchema:
    """Contrat des données brutes."""

    def test_accepts_generated_dataset(self, raw_dataset: pd.DataFrame) -> None:
        """Le dataset généré doit passer le contrat sans erreur."""
        validated = RawDataSchema.validate(raw_dataset)
        assert len(validated) == len(raw_dataset)
        assert list(validated.columns) == list(raw_dataset.columns)

    def test_declares_every_column(self, raw_dataset: pd.DataFrame) -> None:
        """Le schéma documente exactement les colonnes produites (ni plus, ni moins)."""
        assert set(RawDataSchema.to_schema().columns) == set(raw_dataset.columns)

    def test_validate_frame_helper_matches_direct_call(self, raw_dataset: pd.DataFrame) -> None:
        """Le helper ``validate_frame`` applique bien le schéma demandé."""
        direct = RawDataSchema.validate(raw_dataset)
        helper = validate_frame(raw_dataset, "raw")
        pd.testing.assert_frame_equal(direct, helper)

    def test_rejects_out_of_range_value(self, raw_dataset: pd.DataFrame) -> None:
        """Une valeur hors bornes doit être refusée (le contrat porte les bornes métier)."""
        column = _bounded_numeric_column()
        if column is None:
            pytest.skip("aucune colonne numérique bornée dans ce schéma")
        corrupted = raw_dataset.copy()
        corrupted.loc[corrupted.index[0], column] = 10_000_000.0
        with pytest.raises(SchemaViolation) as error:
            RawDataSchema.validate(corrupted)
        assert column in str(error.value)

    def test_rejects_unknown_column(self, raw_dataset: pd.DataFrame) -> None:
        """Une colonne non déclarée doit être refusée (strict=True)."""
        corrupted = raw_dataset.copy()
        corrupted["colonne_inconnue"] = 1
        with pytest.raises(SchemaViolation):
            RawDataSchema.validate(corrupted)

    def test_rejects_missing_column(self, raw_dataset: pd.DataFrame) -> None:
        """Une colonne absente doit être refusée."""
        corrupted = raw_dataset.drop(columns=[raw_dataset.columns[-1]])
        with pytest.raises(SchemaViolation):
            RawDataSchema.validate(corrupted)

    def test_rejects_invalid_category(self, raw_dataset: pd.DataFrame) -> None:
        """Une catégorie hors liste doit être refusée."""
        column = _categorical_column()
        if column is None:
            pytest.skip("aucune colonne catégorielle contrainte par isin")
        corrupted = raw_dataset.copy()
        corrupted.loc[corrupted.index[0], column] = "valeur_inexistante"
        with pytest.raises(SchemaViolation) as error:
            RawDataSchema.validate(corrupted)
        assert column in str(error.value)

    def test_rejects_null_identifier(self, raw_dataset: pd.DataFrame, app_config: Any) -> None:
        """Un identifiant nul doit être refusé quand une clé est déclarée."""
        identifier = app_config.data.id_column
        if not identifier:
            pytest.skip("aucune colonne identifiant déclarée")
        corrupted = raw_dataset.copy()
        corrupted.loc[corrupted.index[0], identifier] = None
        with pytest.raises(SchemaViolation):
            RawDataSchema.validate(corrupted)

    def test_rejects_duplicated_identifier(
        self, raw_dataset: pd.DataFrame, app_config: Any
    ) -> None:
        """Une clé dupliquée doit être refusée (integrité du jeu de données)."""
        identifier = app_config.data.id_column
        if not identifier:
            pytest.skip("aucune colonne identifiant déclarée")
        corrupted = raw_dataset.copy()
        corrupted.loc[corrupted.index[1], identifier] = corrupted.loc[
            corrupted.index[0], identifier
        ]
        with pytest.raises(SchemaViolation):
            RawDataSchema.validate(corrupted)

    def test_lazy_validation_collects_every_error(self, raw_dataset: pd.DataFrame) -> None:
        """En mode lazy, toutes les erreurs sont remontées d'un coup (debug plus rapide)."""
        column = _bounded_numeric_column()
        if column is None:
            pytest.skip("aucune colonne numérique bornée dans ce schéma")
        corrupted = raw_dataset.copy()
        corrupted.loc[corrupted.index[0], column] = -1_000_000.0
        corrupted.loc[corrupted.index[1], column] = 1_000_000.0
        with pytest.raises(SchemaViolation) as error:
            RawDataSchema.validate(corrupted, lazy=True)
        failure_cases = error.value.failure_cases
        assert len(failure_cases) >= 2

    def test_coercion_from_csv_types(self, raw_dataset: pd.DataFrame) -> None:
        """Un aller-retour CSV (types plus lâches) doit rester valide grâce à coerce=True."""
        through_csv = pd.read_csv(StringIO(raw_dataset.to_csv(index=False)))
        validated = RawDataSchema.validate(through_csv)
        assert len(validated) == len(raw_dataset)


class TestProcessedSchema:
    """Contrat de la matrice livrée au modèle."""

    def test_accepts_model_matrix(self, matrices: dict[str, Any]) -> None:
        """La matrice produite par le preprocessing respecte le contrat."""
        validated = ProcessedDataSchema.validate(matrices["X_train"])
        assert not validated.isna().any().any()

    def test_rejects_missing_values(self, matrices: dict[str, Any]) -> None:
        """Une valeur manquante doit être refusée : le preprocessing aurait failli."""
        corrupted = matrices["X_train"].copy()
        corrupted.iloc[0, 0] = np.nan
        with pytest.raises(SchemaViolation):
            ProcessedDataSchema.validate(corrupted)

    def test_rejects_non_numeric_column(self, matrices: dict[str, Any]) -> None:
        """Une colonne non numérique doit être refusée (encodage manquant)."""
        corrupted = matrices["X_train"].copy()
        corrupted["texte"] = "non numérique"
        with pytest.raises(SchemaViolation):
            ProcessedDataSchema.validate(corrupted)

    def test_rejects_infinite_values(self, matrices: dict[str, Any]) -> None:
        """Un infini (division par zéro dans une feature ratio) doit être refusé."""
        corrupted = matrices["X_train"].copy()
        corrupted.iloc[0, 0] = np.inf
        with pytest.raises(SchemaViolation):
            ProcessedDataSchema.validate(corrupted)

    def test_rejects_empty_matrix(self) -> None:
        """Une matrice vide doit être refusée."""
        with pytest.raises(SchemaViolation):
            ProcessedDataSchema.validate(pd.DataFrame({"a": pd.Series([], dtype="float64")}))


class TestInferenceSchema:
    """Contrat des payloads de prédiction."""

    def test_accepts_payload_without_target(
        self, raw_dataset: pd.DataFrame, app_config: Any
    ) -> None:
        """Un payload d'inférence ne contient pas la cible : le contrat doit l'accepter."""
        payload = raw_dataset.drop(columns=[app_config.data.target]).head(10)
        validated = InferenceDataSchema.validate(payload)
        assert len(validated) == 10

    def test_tolerates_missing_values(self, raw_dataset: pd.DataFrame, app_config: Any) -> None:
        """Les valeurs manquantes sont tolérées : le preprocessing imputera."""
        payload = raw_dataset.drop(columns=[app_config.data.target]).head(20).copy()
        payload.iloc[0, 0] = None
        validated = InferenceDataSchema.validate(payload)
        assert len(validated) == 20

    def test_tolerates_extra_columns(self, raw_dataset: pd.DataFrame, app_config: Any) -> None:
        """Un payload peut contenir des colonnes supplémentaires (elles seront ignorées)."""
        payload = raw_dataset.drop(columns=[app_config.data.target]).head(5).copy()
        payload["canal_inutile"] = "web"
        assert len(InferenceDataSchema.validate(payload)) == 5

    def test_rejects_invalid_type(self, raw_dataset: pd.DataFrame, app_config: Any) -> None:
        """Un type incohérent reste refusé : la tolérance ne signifie pas l'absence de contrat."""
        column = _bounded_numeric_column()
        if column is None:
            pytest.skip("aucune colonne numérique bornée dans ce schéma")
        payload = raw_dataset.drop(columns=[app_config.data.target]).head(5).copy()
        payload[column] = "pas-un-nombre"
        with pytest.raises(SchemaViolation):
            InferenceDataSchema.validate(payload)


class TestSchemaDocumentation:
    """Le schéma sert aussi de documentation exécutable."""

    def test_describe_schema_lists_every_column(self) -> None:
        """``describe_schema`` produit une ligne par colonne déclarée."""
        description = describe_schema("raw")
        assert set(description.index) == set(RawDataSchema.to_schema().columns)
        assert {"dtype", "nullable", "required", "checks"} <= set(description.columns)

    def test_markdown_rendering_is_a_table(self) -> None:
        """Le rendu Markdown est un tableau exploitable dans un rapport."""
        markdown = schema_to_markdown("raw")
        assert markdown.startswith("### `RawDataSchema`")
        assert markdown.count("|") > 10

    def test_validation_report_profiles_the_frame(self, raw_dataset: pd.DataFrame) -> None:
        """Le rapport de validation expose shape, dtypes et manquants."""
        report = validation_report(raw_dataset)
        assert report["n_rows"] == len(raw_dataset)
        assert report["n_columns"] == raw_dataset.shape[1]
        assert 0.0 <= report["missing_rate"] <= 1.0
        assert len(report["dtypes"]) == raw_dataset.shape[1]
