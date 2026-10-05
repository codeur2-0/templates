"""Fixtures partagées de la suite de tests texte.

Trois règles, les mêmes que dans les projets tabulaires :

* **rapidité** — les fixtures génèrent un *petit* corpus (64 documents) au lieu des 128 de la
  configuration de référence : la suite doit rester exécutable à chaque commit ;
* **isolation** — rien n'est écrit dans le projet : les pipelines de test écrivent dans le
  ``tmp_path`` de pytest ;
* **fidélité** — la configuration est composée par Hydra, avec des overrides, donc les tests
  exercent exactement le même chemin de code que ``python -m src.main``.

Le corpus est généré par le générateur de la famille (``src.data.generators``), pas par une
fixture écrite à la main : un test qui valide le contrat sur des données inventées par le test
valide le test, pas le générateur.
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
from src.data.loaders import TextCorpusLoader  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.base import BaseModel  # noqa: E402
from src.schemas.config import AppConfig, validate_config  # noqa: E402
from src.utils.paths import ProjectPaths  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from hydra import compose, initialize_config_dir  # noqa: F401

#: Nombre de documents visé par les fixtures : huit thèmes x huit périmètres, une édition.
TINY_DOCUMENTS = 64
#: Graine dédiée aux tests, différente de la graine de production (42).
TEST_SEED = 7


@pytest.fixture(scope="session")
def project_paths() -> ProjectPaths:
    """Return the real project layout (read-only: nothing is written)."""
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
                "model.llm.provider=extractive",
                "train.epochs=1",
                "train.callbacks.progress_bar=false",
            ],
        )
    return validate_config(raw)


@pytest.fixture(scope="session")
def corpus(app_config: AppConfig) -> GeneratedCorpus:
    """Generate the tiny synthetic corpus used by every test."""
    from src.data.generators import SyntheticCorpusGenerator

    generator = SyntheticCorpusGenerator.from_config(app_config.data)
    return generator.generate()


@pytest.fixture(scope="session")
def documents(corpus: GeneratedCorpus) -> pd.DataFrame:
    """Return the generated documents."""
    return corpus.documents


@pytest.fixture(scope="session")
def queries(corpus: GeneratedCorpus) -> pd.DataFrame:
    """Return the generated annotated questions."""
    return corpus.queries


@pytest.fixture(scope="session")
def loader(tmp_path_factory: pytest.TempPathFactory) -> TextCorpusLoader:
    """Return a loader wired to a throw-away project layout."""
    root = tmp_path_factory.mktemp("corpus")
    return TextCorpusLoader(ProjectPaths.from_root(root))


@pytest.fixture(scope="session")
def persisted_corpus(
    corpus: GeneratedCorpus, tmp_path_factory: pytest.TempPathFactory
) -> TextCorpusLoader:
    """Persist the tiny corpus on disk and return the loader that reads it back."""
    root = tmp_path_factory.mktemp("persisted")
    loader = TextCorpusLoader(ProjectPaths.from_root(root))
    loader.paths.ensure()
    loader.save_documents(corpus.documents)
    loader.save_queries(corpus.queries)
    loader.save_metadata(corpus.metadata)
    return loader


@pytest.fixture(scope="session")
def fitted_model(app_config: AppConfig, corpus: GeneratedCorpus) -> BaseModel:
    """Fit the configured model on the tiny corpus (once per session)."""
    model = build_model(app_config.model_dump())
    model.fit(corpus.documents, corpus.queries[corpus.queries["split"] == "calibration"])
    return model


@pytest.fixture
def tiny_paths(tmp_path: Path) -> ProjectPaths:
    """Return an isolated project layout, created on demand."""
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure()
    return paths


@pytest.fixture(scope="session")
def schema_frames(corpus: GeneratedCorpus) -> dict[str, Any]:
    """Return the generated tables, keyed by contract name."""
    return {"documents": corpus.documents, "queries": corpus.queries}


@pytest.fixture(scope="session", autouse=True)
def _quiet_logs() -> Iterator[None]:
    """Keep the test output readable: the pipelines log a lot at INFO level."""
    from src.utils.logging import get_logger

    get_logger(__name__)
    yield
