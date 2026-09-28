"""Tests de la couche d'accès aux données (chargement, validation, split).

Le split est le point le plus critique d'un projet supervisé : une fuite entre train et test
rend toutes les métriques optimistes et donc toutes les décisions fausses. Ces tests
vérifient donc trois choses :

* le chargement applique bien le contrat Pandera (et propose une action quand le fichier manque),
* les tailles de split correspondent à la configuration,
* aucune ligne n'apparaît dans deux splits, et la stratification est respectée.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

try:  # pandera >= 0.26
    import pandera.pandas as pa
except ModuleNotFoundError:  # pragma: no cover - pandera < 0.26
    import pandera as pa  # type: ignore[no-redef]

from src.data.loaders import (
    DatasetSplitter,
    InferenceDataLoader,
    ProcessedDataLoader,
    RawDataLoader,
    assert_no_overlap,
    feature_target_split,
    save_split,
    target_distribution,
)
from src.data.schemas import InferenceDataSchema, ProcessedDataSchema, RawDataSchema
from src.utils.paths import ProjectPaths

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


def _nullable_inference_column(frame: pd.DataFrame, app_config: Any) -> str | None:
    """Find a numeric column the inference contract accepts as null.

    Le test reste ainsi générique : il ne dépend d'aucun nom de colonne propre au cas métier.

    Args:
        frame: Raw dataset.
        app_config: Application configuration.

    Returns:
        The first nullable column present in the payload, or ``None``.
    """
    excluded = {app_config.data.target, app_config.data.id_column}
    for name, column in InferenceDataSchema.to_schema().columns.items():
        if name in excluded or name not in frame.columns:
            continue
        if bool(getattr(column, "nullable", False)) and pd.api.types.is_numeric_dtype(frame[name]):
            return name
    return None


@pytest.fixture
def raw_project(
    tmp_path: Path, raw_dataset: pd.DataFrame, app_config: Any
) -> tuple[ProjectPaths, str]:
    """Create an isolated project layout holding the tiny raw dataset."""
    paths = ProjectPaths.from_root(tmp_path).ensure()
    raw_dataset.to_parquet(paths.data_file(app_config.data.dataset_name), index=False)
    return paths, app_config.data.dataset_name


@pytest.fixture
def raw_loader(raw_project: tuple[ProjectPaths, str]) -> RawDataLoader:
    """Return a validating raw loader pointed at the temporary project."""
    paths, dataset_name = raw_project
    return RawDataLoader(paths, dataset_name=dataset_name, formats=("parquet", "csv"))


class TestRawDataLoader:
    """Chargement et validation des données brutes."""

    def test_reads_and_validates(
        self, raw_loader: RawDataLoader, raw_dataset: pd.DataFrame
    ) -> None:
        """Le loader lit le Parquet et le retourne validé."""
        frame = raw_loader.load()
        assert len(frame) == len(raw_dataset)
        assert list(frame.columns) == list(raw_dataset.columns)
        RawDataSchema.validate(frame)  # ne doit pas lever

    def test_missing_file_gives_an_actionable_hint(self, tmp_path: Path, app_config: Any) -> None:
        """Un dataset absent doit produire un message qui dit quoi faire (DX)."""
        loader = RawDataLoader(
            ProjectPaths.from_root(tmp_path).ensure(),
            dataset_name=app_config.data.dataset_name,
        )
        with pytest.raises(FileNotFoundError) as error:
            loader.load()
        assert "generate_data" in str(error.value)

    def test_limit_caps_the_rows(self, raw_project: tuple[ProjectPaths, str]) -> None:
        """``limit`` permet d'itérer vite sur un échantillon."""
        paths, dataset_name = raw_project
        loader = RawDataLoader(paths, dataset_name=dataset_name, limit=40)
        assert len(loader.load()) == 40

    def test_csv_fallback(
        self, raw_project: tuple[ProjectPaths, str], raw_dataset: pd.DataFrame
    ) -> None:
        """En l'absence de Parquet, le CSV est lu puis coercé par le schéma."""
        paths, dataset_name = raw_project
        (paths.raw_dir / f"{dataset_name}.parquet").unlink()
        raw_dataset.to_csv(paths.raw_dir / f"{dataset_name}.csv", index=False)
        loader = RawDataLoader(paths, dataset_name=dataset_name)
        frame = loader.load()
        assert len(frame) == len(raw_dataset)

    def test_extra_column_is_rejected(
        self, raw_project: tuple[ProjectPaths, str], raw_dataset: pd.DataFrame, app_config: Any
    ) -> None:
        """Une colonne non déclarée doit casser le chargement (dérive de la source)."""
        paths, dataset_name = raw_project
        corrupted = raw_dataset.copy()
        corrupted["colonne_surprise"] = 0
        corrupted.to_parquet(paths.data_file(dataset_name), index=False)
        loader = RawDataLoader(paths, dataset_name=dataset_name)
        with pytest.raises(SchemaViolation):
            loader.load()

    def test_validation_can_be_disabled(
        self, raw_project: tuple[ProjectPaths, str], raw_dataset: pd.DataFrame, app_config: Any
    ) -> None:
        """``validate=False`` permet d'inspecter des données cassées sans exception."""
        paths, dataset_name = raw_project
        corrupted = raw_dataset.copy()
        corrupted["colonne_surprise"] = 0
        corrupted.to_parquet(paths.data_file(dataset_name), index=False)
        loader = RawDataLoader(paths, dataset_name=dataset_name, validate=False)
        assert "colonne_surprise" in loader.load().columns

    def test_lazy_validation_reports_every_failure(
        self, raw_project: tuple[ProjectPaths, str], raw_dataset: pd.DataFrame, app_config: Any
    ) -> None:
        """En mode lazy, toutes les violations sont listées d'un coup."""
        paths, dataset_name = raw_project
        numeric = [
            column
            for column, check_list in (
                (name, getattr(col, "checks", []) or [])
                for name, col in RawDataSchema.to_schema().columns.items()
            )
            if any(
                getattr(check, "name", "") in {"ge", "greater_than_or_equal_to"}
                for check in check_list
            )
        ]
        if not numeric:
            pytest.skip("aucune colonne numérique bornée")
        corrupted = raw_dataset.copy()
        corrupted.loc[corrupted.index[:5], numeric[0]] = 10_000_000
        corrupted.to_parquet(paths.data_file(dataset_name), index=False)
        loader = RawDataLoader(paths, dataset_name=dataset_name, lazy_validation=True)
        with pytest.raises(SchemaViolation) as error:
            loader.load()
        assert _failure_count(error.value) >= 5
        assert app_config.data.validation.lazy in {True, False}


class TestProcessedDataLoader:
    """Chargement des matrices déjà transformées."""

    def test_round_trip(
        self, raw_project: tuple[ProjectPaths, str], matrices: dict[str, Any]
    ) -> None:
        """Une matrice sauvegardée puis rechargée reste conforme au contrat."""
        paths, _ = raw_project
        written = save_split(matrices["X_train"], paths, "features_train")
        loader = ProcessedDataLoader(paths, dataset_name="features_train")
        frame = loader.load()
        assert written.suffix == ".parquet"
        assert list(frame.columns) == list(matrices["X_train"].columns)
        ProcessedDataSchema.validate(frame)

    def test_rejects_non_numeric_matrix(self, raw_project: tuple[ProjectPaths, str]) -> None:
        """Une matrice contenant du texte doit être refusée."""
        paths, _ = raw_project
        save_split(pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]}), paths, "bad_features")
        loader = ProcessedDataLoader(paths, dataset_name="bad_features")
        with pytest.raises(SchemaViolation):
            loader.load()


def _without_target(frame: pd.DataFrame, target: str | None) -> pd.DataFrame:
    """Retire la colonne cible d'un jeu brut.

    En tâche non supervisée (clustering), ``target`` vaut ``None`` et le jeu est renvoyé tel quel :
    un payload d'inférence ne contient de toute façon jamais de cible.

    Args:
        frame: Jeu brut généré.
        target: Nom de la colonne cible, ou ``None``.

    Returns:
        Le jeu sans sa colonne cible.
    """
    if target is not None and target in frame.columns:
        return frame.drop(columns=[target])
    return frame

class TestInferenceDataLoader:
    """Chargement des payloads de prédiction."""

    def test_load_from_frame_accepts_missing_values(
        self, raw_dataset: pd.DataFrame, app_config: Any, tmp_path: Path
    ) -> None:
        """Un payload partiel est accepté : le preprocessing imputera."""
        nullable = _nullable_inference_column(raw_dataset, app_config)
        loader = InferenceDataLoader(ProjectPaths.from_root(tmp_path), dataset_name="inference")
        payload = _without_target(raw_dataset, app_config.data.target).head(15).copy()
        if nullable is not None:
            payload[nullable] = None
        validated = loader.load_from_frame(payload)
        assert len(validated) == 15

    def test_load_from_json_file(
        self, raw_dataset: pd.DataFrame, app_config: Any, tmp_path: Path
    ) -> None:
        """Un fichier JSON (format d'échange API) est lu puis validé."""
        loader = InferenceDataLoader(ProjectPaths.from_root(tmp_path), dataset_name="inference")
        payload = _without_target(raw_dataset, app_config.data.target).head(5)
        path = tmp_path / "payload.json"
        payload.to_json(path, orient="records")
        assert len(loader.load_from(path)) == 5

    def test_missing_file_is_explicit(self, tmp_path: Path) -> None:
        """Un fichier absent doit lever une erreur explicite, pas un KeyError obscur."""
        loader = InferenceDataLoader(ProjectPaths.from_root(tmp_path), dataset_name="inference")
        with pytest.raises(FileNotFoundError):
            loader.load_from(tmp_path / "introuvable.parquet")

    def test_load_without_source_is_refused(self, tmp_path: Path) -> None:
        """``load()`` sans source configurée explique comment fournir les données."""
        loader = InferenceDataLoader(ProjectPaths.from_root(tmp_path), dataset_name="inference")
        with pytest.raises(FileNotFoundError, match="load_from"):
            loader.load()


class TestDatasetSplitter:
    """Politique de split train / validation / test."""

    def test_sizes_match_configuration(
        self, split_frames: Any, raw_dataset: pd.DataFrame, app_config: Any
    ) -> None:
        """Les tailles observées respectent la configuration (à l'arrondi près)."""
        sizes = split_frames.sizes
        assert sum(sizes.values()) == len(raw_dataset)
        split = app_config.train.split
        expected_test = len(raw_dataset) * split.test_size
        assert abs(sizes["test"] - expected_test) <= 2
        if split.val_size > 0:
            expected_val = len(raw_dataset) * split.val_size
            assert abs(sizes["val"] - expected_val) <= 3
        else:
            assert split_frames.val is None

    def test_stratification_preserves_class_balance(
        self, split_frames: Any, app_config: Any
    ) -> None:
        """La stratification conserve la proportion de la classe positive (± 5 points)."""
        target = app_config.data.target
        if target is None or not app_config.train.split.stratify:
            pytest.skip("tâche non supervisée ou stratification désactivée")
        global_rate = float(split_frames.train[target].mean())
        test_rate = float(split_frames.test[target].mean())
        assert abs(global_rate - test_rate) <= 0.05

    def test_no_row_overlap(self, split_frames: Any, app_config: Any) -> None:
        """Aucune ligne ne doit apparaître dans deux splits (fuite de données)."""
        frames = [split_frames.train, split_frames.val, split_frames.test]
        assert_no_overlap(
            *[frame for frame in frames if frame is not None], key=app_config.data.id_column
        )

    def test_split_is_reproducible(self, raw_dataset: pd.DataFrame, app_config: Any) -> None:
        """Même graine ⇒ même split (reproductibilité exigée)."""
        first = DatasetSplitter.from_config(app_config.model_dump(), seed=11).split(
            raw_dataset, target=app_config.data.target
        )
        second = DatasetSplitter.from_config(app_config.model_dump(), seed=11).split(
            raw_dataset, target=app_config.data.target
        )
        pd.testing.assert_frame_equal(first.train, second.train)

    def test_seed_effects_match_the_split_strategy(
        self, raw_dataset: pd.DataFrame, app_config: Any
    ) -> None:
        """La graine pilote le tirage aléatoire — et ne doit rien changer à un split chronologique.

        Les deux comportements sont des contrats, pas des détails d'implémentation :

        * un split **aléatoire** doit dépendre de la graine, sinon le paramètre est décoratif et
          deux exécutions d'un notebook ne sont pas reproductibles de la façon annoncée ;
        * un split **temporel** doit en être indépendant, parce que la frontière suit l'ordre des
          horodatages : la déplacer au hasard mélangerait le passé et le futur et casserait
          l'antériorité de l'information, ce qui est la fuite la plus coûteuse en prévision.

        Le test lit la stratégie réellement configurée au lieu de supposer l'une des deux.
        """
        splitter = DatasetSplitter.from_config(app_config.model_dump(), seed=1)
        first = splitter.split(raw_dataset)
        second = DatasetSplitter.from_config(app_config.model_dump(), seed=2).split(raw_dataset)
        if splitter.time_based and splitter.time_column:
            pd.testing.assert_frame_equal(first.train, second.train)
            pd.testing.assert_frame_equal(first.test, second.test)
        else:
            assert not first.train.equals(second.train)

    def test_incoherent_sizes_are_rejected(self) -> None:
        """Des tailles incohérentes doivent être refusées à la construction."""
        with pytest.raises(ValueError, match="test_size"):
            DatasetSplitter(test_size=0.8)
        with pytest.raises(ValueError, match="val_size"):
            DatasetSplitter(val_size=-0.2)
        with pytest.raises(ValueError, match=r"below 0\.9"):
            DatasetSplitter(test_size=0.45, val_size=0.48)

    def test_temporal_split_requires_time_column(self) -> None:
        """Un split chronologique sans colonne temporelle est une erreur de configuration."""
        with pytest.raises(ValueError, match="time_column"):
            DatasetSplitter(time_based=True)

    def test_temporal_split_is_chronological(
        self, raw_dataset: pd.DataFrame, app_config: Any
    ) -> None:
        """En split temporel, le passé entraîne et le futur est évalué."""
        time_column = app_config.data.time_column
        if not time_column:
            pytest.skip("ce projet n'a pas de colonne temporelle")
        splitter = DatasetSplitter(
            test_size=0.2, val_size=0.15, time_based=True, time_column=time_column
        )
        splits = splitter.split(raw_dataset)
        assert splits.train[time_column].max() <= splits.test[time_column].min()
        if splits.val is not None:
            assert splits.train[time_column].max() <= splits.val[time_column].min()
            assert splits.val[time_column].max() <= splits.test[time_column].min()


class TestFeatureTargetHelpers:
    """Helpers de séparation features / cible."""

    def test_target_and_drops_are_excluded(self, split_frames: Any, app_config: Any) -> None:
        """La cible et les colonnes non modélisables ne doivent jamais devenir des features."""
        target = app_config.data.target
        features, labels = feature_target_split(
            split_frames.train, target, app_config.data.drop_columns
        )
        assert target not in features.columns
        for column in app_config.data.drop_columns:
            assert column not in features.columns
        if target is None:
            assert labels is None  # tâche non supervisée : aucune cible à extraire
        else:
            assert labels is not None and len(labels) == len(features)

    def test_unsupervised_split_has_no_target(self, split_frames: Any) -> None:
        """Sans cible, ``feature_target_split`` renvoie ``None`` pour y (tâches non supervisées)."""
        features, labels = feature_target_split(split_frames.train, None, [])
        assert labels is None
        assert features.shape[1] == split_frames.train.shape[1]

    def test_target_distribution_is_normalised(self, split_frames: Any, app_config: Any) -> None:
        """La distribution de la cible somme à 1 (utilisée dans les rapports)."""
        target = app_config.data.target
        if target is None:
            pytest.skip("tâche non supervisée : aucune distribution de cible à normaliser")
        distribution = target_distribution(split_frames.train[target])
        assert distribution
        assert sum(distribution.values()) == pytest.approx(1.0)
        assert all(0.0 <= value <= 1.0 for value in distribution.values())

    def test_target_distribution_is_empty_without_labels(self) -> None:
        """Sans cible, la distribution est vide plutôt qu'erronée."""
        assert target_distribution(None) == {}

    def test_overlap_guard_detects_leakage(self, app_config: Any) -> None:
        """Le garde-fou de fuite doit effectivement lever quand une clé se répète."""
        key = app_config.data.id_column or "id"
        left = pd.DataFrame({key: ["a", "b"]})
        right = pd.DataFrame({key: ["b", "c"]})
        with pytest.raises(AssertionError, match="leakage"):
            assert_no_overlap(left, right, key=key)
        assert_no_overlap(left, pd.DataFrame({key: ["c", "d"]}), key=key)
