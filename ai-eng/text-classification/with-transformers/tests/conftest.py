"""Fixtures partagées de la suite de tests de classification de texte.

Trois règles, les mêmes que dans les projets tabulaires :

* **rapidité** — les fixtures génèrent un *petit* corpus (300 tickets au lieu des 1 200 de la
  configuration de référence, soit 180 tickets d'entraînement) : la suite doit rester exécutable à
  chaque commit, et elle l'est même en ajoutant une pile à entraînement lent ;
* **isolation** — rien n'est écrit dans le projet : les pipelines des tests écrivent dans le
  ``tmp_path`` de pytest ;
* **fidélité** — la configuration est composée par Hydra, avec des overrides, donc les tests
  exercent exactement le même chemin de code que ``python -m src.main``.

Le corpus vient du générateur de la famille (``src.data.generators``), jamais d'une fixture écrite
à la main : un test qui valide un contrat sur des données inventées par le test valide le test, pas
le générateur.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

#: Racine du projet, ajoutée à ``sys.path`` pour que ``import src...`` fonctionne.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generator_base import GeneratedCorpus  # noqa: E402
from src.data.loaders import TextLabelLoader  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.contract import BaseTextClassifier  # noqa: E402
from src.schemas.config import AppConfig, validate_config  # noqa: E402
from src.training.trainer import ClassificationOutcome, ClassificationTrainer  # noqa: E402
from src.utils.config_access import node  # noqa: E402
from src.utils.paths import ProjectPaths  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from hydra import compose, initialize_config_dir  # noqa: F401

#: Tickets visés par les fixtures : six classes x trois styles, cinquante répétitions.
TINY_DOCUMENTS = 300
#: Graine dédiée aux tests, différente de la graine de production (42).
TEST_SEED = 7


@pytest.fixture(scope="session")
def project_paths() -> ProjectPaths:
    """Return the real project layout (read-only: nothing is written there)."""
    return ProjectPaths.from_root(PROJECT_ROOT)


@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    """Compose the real Hydra configuration with test-friendly overrides.

    Returns:
        The validated :class:`~src.schemas.config.AppConfig`.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # ``GlobalHydra`` est un singleton : on le réinitialise pour permettre plusieurs compositions
    # (tests et notebooks cohabitent dans le même processus).
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "conf"), version_base=None):
        raw = compose(
            config_name="config",
            overrides=[
                "mode=train",
                f"data.n_samples={TINY_DOCUMENTS}",
                f"seed={TEST_SEED}",
                "log_level=WARNING",
                "data.validation.strict=true",
                "train.epochs=1",
                "train.early_stopping.enabled=false",
                "train.callbacks.progress_bar=false",
            ],
        )
    return validate_config(raw)


@pytest.fixture(scope="session")
def corpus(app_config: AppConfig) -> GeneratedCorpus:
    """Generate the tiny labelled corpus used by every test."""
    from src.data.generators import SyntheticTicketGenerator

    generator = SyntheticTicketGenerator.from_config(app_config.data)
    return generator.generate()


@pytest.fixture(scope="session")
def documents(corpus: GeneratedCorpus) -> pd.DataFrame:
    """Return the generated tickets."""
    return corpus.documents


@pytest.fixture(scope="session")
def train_split(documents: pd.DataFrame) -> pd.DataFrame:
    """Return the training split of the tiny corpus."""
    return documents[documents["split"] == "train"].reset_index(drop=True)


@pytest.fixture(scope="session")
def test_split(documents: pd.DataFrame) -> pd.DataFrame:
    """Return the test split of the tiny corpus."""
    return documents[documents["split"] == "test"].reset_index(drop=True)


@pytest.fixture(scope="session")
def loader(tmp_path_factory: pytest.TempPathFactory) -> TextLabelLoader:
    """Return a loader wired to a throw-away project layout."""
    root = tmp_path_factory.mktemp("corpus")
    return TextLabelLoader(ProjectPaths.from_root(root))


@pytest.fixture(scope="session")
def persisted_corpus(
    corpus: GeneratedCorpus, tmp_path_factory: pytest.TempPathFactory
) -> TextLabelLoader:
    """Persist the tiny corpus on disk and return the loader that reads it back."""
    root = tmp_path_factory.mktemp("persisted")
    loader = TextLabelLoader(ProjectPaths.from_root(root))
    loader.paths.ensure()
    loader.save_documents(corpus.documents)
    loader.save_metadata(corpus.metadata)
    return loader


@pytest.fixture(scope="session")
def fitted_model(app_config: AppConfig, train_split: pd.DataFrame) -> BaseTextClassifier:
    """Fit the configured model on the training split (once per session)."""
    model = build_model(app_config.model_dump())
    model.fit(train_split)
    return model


@pytest.fixture(scope="session")
def predictions_frame(
    app_config: AppConfig, fitted_model: BaseTextClassifier, test_split: pd.DataFrame
) -> pd.DataFrame:
    """Classify the test split, exactly as ``mode=predict`` does.

    Returns:
        The prediction table (identity, prediction, confidence, per-label probabilities, latency).
    """
    from src.inference.predictor import TextClassificationPredictor

    predictor = TextClassificationPredictor(
        fitted_model,
        config=app_config.model_dump(),
        text_column=str(node(app_config, "model").get("text_column", "text")),
    )
    return predictor.predict(test_split)


@pytest.fixture(scope="session")
def training_outcome(app_config: AppConfig, documents: pd.DataFrame) -> ClassificationOutcome:
    """Run the trainer on the tiny corpus, as the training pipeline does."""
    trainer = ClassificationTrainer(
        build_model(app_config.model_dump()),
        config=app_config.train.model_dump(),
        text_column=str(node(app_config, "model").get("text_column", "text")),
        target_column=str(app_config.data.target or "label"),
        metric_names=app_config.metrics.all_metrics,
        primary_metric=app_config.metrics.primary,
    )
    return trainer.run(documents)


@pytest.fixture
def tiny_paths(tmp_path: Path) -> ProjectPaths:
    """Return an isolated project layout, created on demand."""
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    return paths


@pytest.fixture(scope="session")
def schema_frames(corpus: GeneratedCorpus) -> dict[str, Any]:
    """Return the generated tables, keyed by contract name."""
    return {"tickets": corpus.documents}


@pytest.fixture(scope="session", autouse=True)
def _quiet_logs() -> Iterator[None]:
    """Keep the test output readable: the pipelines log a lot at INFO level."""
    from src.utils.logging import get_logger

    get_logger(__name__)
    yield
