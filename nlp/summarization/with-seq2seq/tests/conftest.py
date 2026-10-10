"""Fixtures partagées de la suite de tests de résumé automatique.

Trois règles, les mêmes que dans les autres familles de texte :

* **rapidité** — les fixtures génèrent un *petit* corpus (120 comptes-rendus au lieu des 600 de la
  configuration de référence) et un modèle minuscule (une couche, 48 unités, vocabulaire de 320
  pièces) : la suite reste exécutable à chaque commit, même si la stack entraîne un réseau ;
* **isolation** — rien n'est écrit dans le projet : les pipelines des tests écrivent dans le
  ``tmp_path`` de pytest ;
* **fidélité** — la configuration est composée par Hydra, avec des overrides, donc les tests
  exercent le même chemin de code que ``python -m src.main``.

Le corpus vient du générateur de la famille (``src.data.generators``), jamais d'une fixture écrite à
la main : un test qui valide un contrat sur des données inventées par le test valide le test, pas le
générateur. La **table des faits** est la seconde table du corpus — c'est elle qui porte la fidélité
du projet — et chaque test qui en a besoin la reçoit par une fixture nommée.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

#: Racine du projet, ajoutée à ``sys.path`` pour que ``import src...`` fonctionne.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generator_base import GeneratedCorpus  # noqa: E402
from src.data.generators import GeneratedSummaryCorpus  # noqa: E402
from src.data.loaders import SummaryCorpusLoader  # noqa: E402
from src.evaluation.evaluator import SummaryEvaluation, SummaryEvaluator  # noqa: E402
from src.inference.predictor import SummaryPredictor  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.contract import BaseTextGenerator  # noqa: E402
from src.models.lead import LeadSummarizer  # noqa: E402
from src.models.textrank import TextRankSummarizer  # noqa: E402
from src.schemas.config import AppConfig, validate_config  # noqa: E402
from src.training.trainer import SummaryTrainer, SummaryTrainingOutcome  # noqa: E402
from src.utils.config_access import node  # noqa: E402
from src.utils.paths import ProjectPaths  # noqa: E402

#: Comptes-rendus visés par les fixtures : un cinquième du corpus de référence.
TINY_DOCUMENTS = 120
#: Graine dédiée aux tests, différente de la graine de production (42).
TEST_SEED = 13
#: Architecture de test : minuscule, mais de *même nature* que celle du projet.
TINY_OVERRIDES = [
    "model.params.layers=1",
    "model.params.units=48",
    "+model.params.vocab_size=320",
    "+model.params.min_frequency=1",
    "model.params.pretraining.epochs=1",
    "model.params.max_output_tokens=60",
]


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
                "train.batch_size=16",
                "train.early_stopping.enabled=false",
                "train.callbacks.logging_every=100",
                "train.callbacks.progress_bar=false",
                *TINY_OVERRIDES,
            ],
        )
    return validate_config(raw)


@pytest.fixture(scope="session")
def corpus(app_config: AppConfig) -> GeneratedSummaryCorpus:
    """Generate the tiny corpus (documents, references and facts) used by every test."""
    from src.data.generators import SyntheticSummaryCorpusGenerator

    generator = SyntheticSummaryCorpusGenerator.from_config(app_config.data)
    return generator.generate()


@pytest.fixture(scope="session")
def documents(corpus: GeneratedCorpus) -> pd.DataFrame:
    """Return the generated documents."""
    return corpus.documents


@pytest.fixture(scope="session")
def references(corpus: GeneratedCorpus) -> pd.DataFrame:
    """Return the generated reference summaries (the supervision of the project)."""
    return corpus.queries


@pytest.fixture(scope="session")
def facts(corpus: GeneratedSummaryCorpus) -> pd.DataFrame:
    """Return the annotated facts (fidelity ground truth)."""
    return corpus.facts


def _split(
    documents: pd.DataFrame, references: pd.DataFrame, facts: pd.DataFrame, name: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Restrict the three tables to one split."""
    identifiers = set(documents.loc[documents["split"] == name, "doc_id"])
    return (
        documents[documents["doc_id"].isin(identifiers)].reset_index(drop=True),
        references[references["doc_id"].isin(identifiers)].reset_index(drop=True),
        facts[facts["doc_id"].isin(identifiers)].reset_index(drop=True),
    )


@pytest.fixture(scope="session")
def train_split(
    documents: pd.DataFrame, references: pd.DataFrame, facts: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the training documents, their references and their facts."""
    return _split(documents, references, facts, "train")


@pytest.fixture(scope="session")
def val_split(
    documents: pd.DataFrame, references: pd.DataFrame, facts: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the validation documents, their references and their facts."""
    return _split(documents, references, facts, "val")


@pytest.fixture(scope="session")
def test_split(
    documents: pd.DataFrame, references: pd.DataFrame, facts: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the test documents, their references and their facts (read once, by evaluation)."""
    return _split(documents, references, facts, "test")


@pytest.fixture(scope="session")
def loader(tmp_path_factory: pytest.TempPathFactory) -> SummaryCorpusLoader:
    """Return a loader wired to a throw-away project layout."""
    root = tmp_path_factory.mktemp("corpus")
    return SummaryCorpusLoader(ProjectPaths.from_root(root))


@pytest.fixture(scope="session")
def persisted_corpus(
    corpus: GeneratedSummaryCorpus, tmp_path_factory: pytest.TempPathFactory
) -> SummaryCorpusLoader:
    """Persist the tiny corpus on disk and return the loader that reads it back."""
    root = tmp_path_factory.mktemp("persisted")
    loader = SummaryCorpusLoader(ProjectPaths.from_root(root))
    loader.paths.ensure()
    loader.save_documents(corpus.documents)
    loader.save_references(corpus.queries)
    loader.save_facts(corpus.facts)
    loader.save_metadata(corpus.metadata)
    return loader


@pytest.fixture(scope="session")
def lead_model(
    app_config: AppConfig, train_split: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
) -> LeadSummarizer:
    """Fit the extractive floor on the training split (once per session)."""
    train_documents, train_references, _ = train_split
    model = LeadSummarizer(
        compression=float(node(app_config, "model").get("params", {}).get("compression", 0.45)),
        max_output_tokens=int(
            node(app_config, "model").get("params", {}).get("max_output_tokens", 90)
        ),
        seed=TEST_SEED,
    )
    model.fit(train_documents, train_references)
    return model


@pytest.fixture(scope="session")
def textrank_model(
    app_config: AppConfig, train_split: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
) -> TextRankSummarizer:
    """Fit the extractive baseline on the training split (once per session)."""
    train_documents, train_references, _ = train_split
    model = TextRankSummarizer(
        compression=float(node(app_config, "model").get("params", {}).get("compression", 0.45)),
        max_output_tokens=int(
            node(app_config, "model").get("params", {}).get("max_output_tokens", 90)
        ),
        seed=TEST_SEED,
    )
    model.fit(train_documents, train_references)
    return model


@pytest.fixture(scope="session")
def fitted_model(
    app_config: AppConfig, train_split: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
) -> BaseTextGenerator:
    """Fit the configured architecture on the training split (once per session)."""
    train_documents, train_references, _ = train_split
    model = build_model(app_config.model_dump())
    model.fit(train_documents, train_references)
    return model


@pytest.fixture(scope="session")
def evaluator(
    lead_model: LeadSummarizer,
    app_config: AppConfig,
    project_paths: ProjectPaths,
) -> SummaryEvaluator:
    """Return the evaluator of the extractive floor, wired to the real configuration.

    La stratégie mesurée est la **baseline extractive** : elle ne coûte rien à ajuster, donc une
    fixture de session peut la garder en cache, alors que le chemin exercé (génération, ROUGE,
    fidélité, ventilation par segment, verdict publié) est celui de la production. Un test qui veut
    comparer plusieurs stratégies construit son propre évaluateur, avec l'argument ``baselines``.
    """
    model_node = node(app_config, "model")
    return SummaryEvaluator(
        lead_model,
        config=app_config.model_dump(),
        metrics_config=app_config.metrics.model_dump(),
        paths=project_paths,
        text_column=str(model_node.get("text_column", "text")),
        id_column=str(app_config.data.id_column or "doc_id"),
    )


@pytest.fixture(scope="session")
def predictor(
    lead_model: LeadSummarizer,
    app_config: AppConfig,
    persisted_corpus: SummaryCorpusLoader,
) -> SummaryPredictor:
    """Build the inference predictor of the extractive floor, wired to the persisted corpus.

    Le corpus est **sur le disque** (fixture ``persisted_corpus``) : l'inférence retrouve donc les
    résumés de référence et publie des métriques de fidélité, au lieu de les laisser vides comme
    elle le ferait sur un fichier externe.
    """
    return SummaryPredictor(
        lead_model,
        paths=persisted_corpus.paths,
        config=app_config.model_dump(),
        text_column=str(node(app_config, "model").get("text_column", "text")),
        id_column=str(node(app_config, "data").get("id_column", "doc_id")),
    )


@pytest.fixture(scope="session")
def predictions(
    predictor: SummaryPredictor,
    test_split: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> pd.DataFrame:
    """Summarise the test split with the extractive floor, as ``mode=predict`` does."""
    documents, _, _ = test_split
    return predictor.predict(documents)


@pytest.fixture
def published_predictions(
    predictor: SummaryPredictor,
    predictions: pd.DataFrame,
    tiny_paths: ProjectPaths,
) -> pd.DataFrame:
    """Return the prediction table **as published on disk**.

    C'est le contrat de ``predictions.csv`` — la table passe par ``predictor.save``, donc par le
    schéma de publication, et c'est celle-là que le validateur Pandera doit accepter.
    """
    path = predictor.save(predictions, path=tiny_paths.reports_dir / "predictions.csv")
    return pd.read_csv(path, keep_default_na=False)


@pytest.fixture(scope="session")
def evaluation(
    evaluator: SummaryEvaluator,
    test_split: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> SummaryEvaluation:
    """Measure the extractive floor on the test split, as ``mode=evaluate`` does.

    L'évaluation passe par :meth:`~src.evaluation.evaluator.SummaryEvaluator.evaluate`, et non par
    ``score_model`` : c'est la seule façon d'obtenir le verdict, la ventilation par segment, les
    références publiées et la trame d'erreurs — c'est-à-dire ce que le rapport publie.
    """
    documents, references_frame, facts_frame = test_split
    return evaluator.evaluate(documents, references_frame, facts_frame)


@pytest.fixture(scope="session")
def training_outcome(
    lead_model: LeadSummarizer,
    app_config: AppConfig,
    tmp_path_factory: pytest.TempPathFactory,
    documents: pd.DataFrame,
    references: pd.DataFrame,
    facts: pd.DataFrame,
) -> SummaryTrainingOutcome:
    """Run the trainer on the extractive floor, as ``mode=train`` does.

    Le modèle entraîné est la baseline extractive : elle ne coûte rien, donc la fixture reste
    rapide, alors que le chemin exercé (découpage, comparaison sur la validation, verdict,
    persistance) est celui du projet. Le corpus **entier** lui est donné : c'est le *trainer* qui
    découpe train et validation, comme en production — lui donner directement le train laisserait
    ``validation_metrics`` vide, et une fixture qui ne mesure rien ne prouve rien.
    """
    root = tmp_path_factory.mktemp("training")
    paths = ProjectPaths.from_root(root)
    paths.ensure()
    trainer = SummaryTrainer(
        lead_model,
        config=app_config.train.model_dump(),
        paths=paths,
        algorithm="lead",
        text_column=str(node(app_config, "model").get("text_column", "text")),
        id_column=str(app_config.data.id_column or "doc_id"),
        compare=("lead",),
        comparison_documents=8,
        baseline=lead_model,
    )
    return trainer.run(documents, references, facts)


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
