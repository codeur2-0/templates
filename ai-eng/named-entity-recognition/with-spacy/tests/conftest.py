"""Fixtures partagées de la suite de tests d'extraction d'entités.

Trois règles, les mêmes que dans les autres familles de texte :

* **rapidité** — les fixtures génèrent un *petit* corpus (120 messages au lieu des 1 200 de la
  configuration de référence, soit 72 messages d'entraînement) et **une** époque : la suite reste
  exécutable à chaque commit, même si la stack entraîne un réseau ;
* **isolation** — rien n'est écrit dans le projet : les pipelines des tests écrivent dans le
  ``tmp_path`` de pytest ;
* **fidélité** — la configuration est composée par Hydra, avec des overrides, donc les tests
  exercent le même chemin de code que ``python -m src.main``.

Le corpus vient du générateur de la famille (``src.data.generators``), jamais d'une fixture écrite à
la main : un test qui valide un contrat sur des données inventées par le test valide le test, pas le
générateur. Les **annotations** sont la seconde table du corpus (``corpus.queries``) : c'est la
supervision du projet, et chaque test qui en a besoin la reçoit par une fixture nommée.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

#: Racine du projet, ajoutée à ``sys.path`` pour que ``import src...`` fonctionne.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generator_base import GeneratedCorpus  # noqa: E402
from src.data.loaders import EntityCorpusLoader  # noqa: E402
from src.inference.predictor import EntityPredictor  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.contract import BaseEntityTagger  # noqa: E402
from src.schemas.config import AppConfig, validate_config  # noqa: E402
from src.training.trainer import EntityTrainer, EntityTrainingOutcome  # noqa: E402
from src.utils.config_access import node  # noqa: E402
from src.utils.paths import ProjectPaths  # noqa: E402

#: Messages visés par les fixtures : un dixième du corpus de référence, sept minutes gagnées.
TINY_DOCUMENTS = 120
#: Graine dédiée aux tests, différente de la graine de production (42).
TEST_SEED = 13


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

    # ``GlobalHydra`` est un singleton : on le réinitialise pour permettre plusieurs compositions.
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "conf"), version_base=None):
        raw = compose(
            config_name="config",
            overrides=[
                "mode=train",
                f"data.n_samples={TINY_DOCUMENTS}",
                # Deux nœuds, deux rôles : ``seed`` pilote le modèle (``model.random_state``),
                # ``data.seed`` le générateur de corpus. Les tests fixent les deux.
                f"seed={TEST_SEED}",
                f"data.seed={TEST_SEED}",
                "log_level=WARNING",
                "data.validation.strict=true",
                "train.epochs=1",
                "train.early_stopping.enabled=false",
                "train.callbacks.logging_every=100",
                "train.callbacks.progress_bar=false",
            ],
        )
    return validate_config(raw)


@pytest.fixture(scope="session")
def corpus(app_config: AppConfig) -> GeneratedCorpus:
    """Generate the tiny annotated corpus used by every test."""
    from src.data.generators import SyntheticEntityCorpusGenerator

    generator = SyntheticEntityCorpusGenerator.from_config(app_config.data)
    return generator.generate()


@pytest.fixture(scope="session")
def documents(corpus: GeneratedCorpus) -> pd.DataFrame:
    """Return the generated messages."""
    return corpus.documents


@pytest.fixture(scope="session")
def spans(corpus: GeneratedCorpus) -> pd.DataFrame:
    """Return the generated annotations (the supervision table)."""
    return corpus.queries


@pytest.fixture(scope="session")
def train_split(documents: pd.DataFrame, spans: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the training messages and their annotations."""
    identifiers = set(documents.loc[documents["split"] == "train", "msg_id"])
    return (
        documents[documents["msg_id"].isin(identifiers)].reset_index(drop=True),
        spans[spans["msg_id"].isin(identifiers)].reset_index(drop=True),
    )


@pytest.fixture(scope="session")
def test_split(documents: pd.DataFrame, spans: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the test messages and their annotations (read once, by the evaluation tests)."""
    identifiers = set(documents.loc[documents["split"] == "test", "msg_id"])
    return (
        documents[documents["msg_id"].isin(identifiers)].reset_index(drop=True),
        spans[spans["msg_id"].isin(identifiers)].reset_index(drop=True),
    )


@pytest.fixture(scope="session")
def loader(tmp_path_factory: pytest.TempPathFactory) -> EntityCorpusLoader:
    """Return a loader wired to a throw-away project layout."""
    root = tmp_path_factory.mktemp("corpus")
    return EntityCorpusLoader(ProjectPaths.from_root(root))


@pytest.fixture(scope="session")
def persisted_corpus(
    corpus: GeneratedCorpus, tmp_path_factory: pytest.TempPathFactory
) -> EntityCorpusLoader:
    """Persist the tiny corpus on disk and return the loader that reads it back."""
    root = tmp_path_factory.mktemp("persisted")
    loader = EntityCorpusLoader(ProjectPaths.from_root(root))
    loader.paths.ensure()
    loader.save_documents(corpus.documents)
    loader.save_annotations(corpus.queries)
    loader.save_metadata(corpus.metadata)
    return loader


@pytest.fixture(scope="session")
def fitted_model(
    app_config: AppConfig, train_split: tuple[pd.DataFrame, pd.DataFrame]
) -> BaseEntityTagger:
    """Fit the configured extractor on the training split (once per session)."""
    documents, spans = train_split
    model = build_model(app_config.model_dump())
    model.fit(documents, spans)
    return model


@pytest.fixture(scope="session")
def predictions(
    fitted_model: BaseEntityTagger,
    app_config: AppConfig,
    test_split: tuple[pd.DataFrame, pd.DataFrame],
) -> pd.DataFrame:
    """Extract the entities of the test split, exactly as ``mode=predict`` does.

    Returns:
        The mention table (offsets, type, surface, provenance, confidence, latency, verdict).
    """
    documents, spans = test_split
    predictor = EntityPredictor(
        fitted_model,
        config=app_config.model_dump(),
        text_column=str(node(app_config, "model").get("text_column", "text")),
        id_column=str(app_config.data.id_column or "msg_id"),
    )
    return predictor.predict(documents, spans=spans)


@pytest.fixture(scope="session")
def training_outcome(
    app_config: AppConfig, documents: pd.DataFrame, spans: pd.DataFrame
) -> EntityTrainingOutcome:
    """Run the trainer on the tiny corpus, as the training pipeline does."""
    trainer = EntityTrainer(
        build_model(app_config.model_dump()),
        config=app_config.train.model_dump(),
        text_column=str(node(app_config, "model").get("text_column", "text")),
        target_column=str(node(app_config, "model").get("target", "label")),
        id_column=str(app_config.data.id_column or "msg_id"),
        split_column=str(app_config.data.group_column or "split"),
        metric_names=app_config.metrics.all_metrics,
        primary_metric=app_config.metrics.primary,
        min_primary_metric=app_config.metrics.min_primary,
    )
    return trainer.run(documents, spans)


@pytest.fixture
def tiny_paths(tmp_path: Path) -> ProjectPaths:
    """Return an isolated project layout, created on demand.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        The layout, with its directories created.
    """
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    return paths


@pytest.fixture(scope="session")
def schema_frames(corpus: GeneratedCorpus) -> dict[str, Any]:
    """Return the generated tables, keyed by contract name."""
    return {"messages": corpus.documents, "spans": corpus.queries}
