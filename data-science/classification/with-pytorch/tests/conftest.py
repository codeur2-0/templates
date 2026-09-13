"""Pytest fixtures for the telecom-churn-pytorch project.

Trois règles de conception, appliquées partout dans cette suite :

* **rapidité** — chaque fixture construit un *petit* jeu de données (300 lignes) plutôt que le
  dataset complet : la suite doit rester utilisable à chaque commit ;
* **isolation** — les artefacts sont écrits dans les ``tmp_path`` de pytest, jamais dans le projet ;
* **fidélité** — la configuration utilisée est la *vraie* configuration Hydra (avec des
  overrides), donc les tests exercent exactement le même chemin de code que la production.

Les fixtures de session (``raw_dataset``, ``split_frames``, ``prepared``, ``fitted_model``) sont
coûteuses : elles ne sont calculées qu'une fois pour toute la suite.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

#: Racine du projet, ajoutée à ``sys.path`` pour que ``import src...`` fonctionne.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generators import SyntheticDataGenerator  # noqa: E402
from src.data.loaders import (  # noqa: E402
    DatasetSplitter,
    SplitFrames,
    feature_target_split,
)
from src.features.build_features import (  # noqa: E402
    FeatureBuilder,
    select_feature_columns,
    split_by_dtype,
)
from src.models import build_model  # noqa: E402
from src.models.base import BaseModel  # noqa: E402
from src.preprocessing.pipelines import PreprocessingPipeline  # noqa: E402
from src.schemas.config import AppConfig, validate_config  # noqa: E402
from src.utils.paths import ProjectPaths  # noqa: E402

#: Nombre de lignes des fixtures : assez pour être significatif, assez petit pour être rapide.
TINY_ROWS = 300
#: Graine dédiée aux tests (différente de la graine de production).
TEST_SEED = 7


@pytest.fixture(scope="session")
def project_paths() -> ProjectPaths:
    """Return the real project layout (paths only, nothing is written)."""
    return ProjectPaths.from_root(PROJECT_ROOT)


@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    """Compose the real Hydra configuration with test-friendly overrides.

    Returns:
        A validated :class:`AppConfig`.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # GlobalHydra est un singleton process-wide : on le réinitialise pour permettre
    # plusieurs compositions (tests + notebooks) sans erreur "already initialized".
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "conf"), version_base=None):
        raw = compose(
            config_name="config",
            overrides=[
                "mode=train",
                f"data.n_samples={TINY_ROWS}",
                f"seed={TEST_SEED}",
                "log_level=WARNING",
                "data.validation.strict=true",
                "++train.epochs=2",
                "++train.batch_size=32",
                "train.callbacks.progress_bar=false",
            ],
        )
    return validate_config(raw)


@pytest.fixture(scope="session")
def raw_dataset(app_config: AppConfig) -> pd.DataFrame:
    """Generate a tiny raw dataset (session scoped: generated once)."""
    generator = SyntheticDataGenerator(n_samples=TINY_ROWS, seed=TEST_SEED)
    return generator.generate()


@pytest.fixture(scope="session")
def raw_dataset_path(raw_dataset: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Persist the tiny dataset in a temporary directory and return its Parquet path."""
    directory = tmp_path_factory.mktemp("raw-data")
    path = directory / "telecom_churn.parquet"
    raw_dataset.to_parquet(path, index=False)
    return path


@pytest.fixture(scope="session")
def split_frames(raw_dataset: pd.DataFrame, app_config: AppConfig) -> SplitFrames:
    """Split the tiny dataset exactly like the training pipeline does."""
    splitter = DatasetSplitter.from_config(app_config.model_dump(), seed=TEST_SEED)
    return splitter.split(raw_dataset, target=app_config.data.target)


@pytest.fixture(scope="session")
def feature_builder(split_frames: SplitFrames, app_config: AppConfig) -> FeatureBuilder:
    """Fit the declarative feature builder on the training split only (no leakage)."""
    builder = FeatureBuilder.from_config(app_config.model_dump(), target=app_config.data.target)
    if builder.recipes:
        builder.fit(split_frames.train)
    return builder


@pytest.fixture(scope="session")
def enriched_splits(
    feature_builder: FeatureBuilder, split_frames: SplitFrames
) -> dict[str, pd.DataFrame]:
    """Enrich the three splits with the derived features."""
    return {
        "train": feature_builder.transform(split_frames.train),
        "val": None if split_frames.val is None else feature_builder.transform(split_frames.val),
        "test": feature_builder.transform(split_frames.test),
    }


@pytest.fixture(scope="session")
def prepared(enriched_splits: dict[str, pd.DataFrame], app_config: AppConfig) -> dict[str, Any]:
    """Reproduce :class:`TrainPipeline` preprocessing on the tiny dataset.

    Returns:
        Mapping with the fitted ``pipeline`` and the ``X_*`` / ``y_*`` matrices.
    """
    target = app_config.data.target
    drop_columns = list(app_config.data.drop_columns)
    train_frame = enriched_splits["train"]

    feature_columns = select_feature_columns(train_frame, drop_columns=drop_columns, target=target)
    numeric_columns, categorical_columns = split_by_dtype(train_frame, feature_columns)
    explicit = app_config.preprocessing.model_dump().get("columns") or {}
    numeric_columns = list(explicit.get("numeric") or numeric_columns)
    categorical_columns = list(explicit.get("categorical") or categorical_columns)

    pipeline = PreprocessingPipeline(
        numeric_features=numeric_columns,
        categorical_features=categorical_columns,
        config=app_config.preprocessing.model_dump(),
        target=target,
    )

    X_train_frame, y_train = feature_target_split(train_frame, target, drop_columns)
    X_train = pipeline.fit_transform(X_train_frame, y_train)

    def project(frame: pd.DataFrame | None) -> tuple[pd.DataFrame | None, pd.Series | None]:
        if frame is None:
            return None, None
        _, labels = feature_target_split(frame, target, drop_columns)
        return pipeline.transform(frame.loc[:, X_train_frame.columns]), labels

    X_val, y_val = project(enriched_splits["val"])
    X_test, y_test = project(enriched_splits["test"])

    return {
        "pipeline": pipeline,
        "numeric_features": numeric_columns,
        "categorical_features": categorical_columns,
        "feature_frame": X_train_frame,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": list(pipeline.feature_names_out),
    }


@pytest.fixture(scope="session")
def preprocessing(prepared: dict[str, Any]) -> PreprocessingPipeline:
    """Return the fitted preprocessing pipeline."""
    return prepared["pipeline"]


@pytest.fixture(scope="session")
def matrices(prepared: dict[str, Any]) -> dict[str, Any]:
    """Return the preprocessed matrices handed to the model."""
    return prepared


@pytest.fixture(scope="session")
def model(app_config: AppConfig, prepared: dict[str, Any]) -> BaseModel:
    """Build the model from the configuration (not fitted yet)."""
    return build_model(app_config, feature_names=prepared["feature_names"])


@pytest.fixture(scope="session")
def fitted_model(model: BaseModel, prepared: dict[str, Any]) -> BaseModel:
    """Return a model fitted on the tiny training split (session scoped: fitted once)."""
    model.fit(
        prepared["X_train"],
        prepared["y_train"],
        X_val=prepared["X_val"],
        y_val=prepared["y_val"],
        callbacks=[],
    )
    return model


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a seeded NumPy generator for stochastic assertions."""
    return np.random.default_rng(TEST_SEED)


@pytest.fixture(autouse=True)
def _quiet_logging() -> Iterator[None]:
    """Keep the test output readable by silencing loguru during tests."""
    from loguru import logger

    logger.remove()
    yield
    logger.remove()
