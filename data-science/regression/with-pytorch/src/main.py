"""Point d'entrée principal du projet.

Estimation du prix de vente immobilier (AVM) avec PyTorch.

Ce module est la seule porte d'entrée applicative du projet. Il :

1. compose la configuration avec **Hydra** (``conf/``),
2. la **valide et la type** avec Pydantic (:func:`src.schemas.config.validate_config`),
3. fixe la graine aléatoire (reproductibilité),
4. instancie le pipeline demandé et l'exécute.

Modes disponibles (``mode=...``) :
    * ``generate-data`` — génère le jeu de données synthétique dans data/raw
    * ``train`` — entraîne, évalue et sauvegarde les artefacts
    * ``evaluate`` — ré-évalue un modèle existant et régénère rapports/figures
    * ``predict`` — score de nouvelles données (fichier ou échantillon)
    * ``all`` — enchaîne les quatre étapes ci-dessus

Exemples :
    $ python -m src.main mode=train
    $ python -m src.main mode=train ++train.epochs=10 data.n_samples=2000
    $ python -m src.main mode=predict predict.n_samples=10
    $ python -m src.main --multirun model.params.hidden_layers=50,100,200
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Permet `python src/main.py` en plus de `python -m src.main` : dans le premier cas, le
# répertoire courant d'import est `src/` et le package `src` n'est pas importable.
if __package__ in (None, ""):  # pragma: no cover - dépend du mode de lancement
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.pipelines import (
    DataGenerationPipeline,
    EvaluationPipeline,
    InferencePipeline,
    PipelineResult,
    TrainPipeline,
)
from src.schemas.config import AppConfig, validate_config
from src.utils.logging import get_logger, setup_logging
from src.utils.utils import set_seed

logger = get_logger(__name__)

#: Mapping mode -> pipeline. ``all`` est traité séparément (enchaînement).
PIPELINES: dict[str, type[Any]] = {
    "generate-data": DataGenerationPipeline,
    "train": TrainPipeline,
    "evaluate": EvaluationPipeline,
    "predict": InferencePipeline,
}

#: Séquence exécutée par le mode ``all``.
ALL_MODES: tuple[str, ...] = ("generate-data", "train", "evaluate", "predict")


def run_mode(config: AppConfig, mode: str) -> PipelineResult:
    """Instantiate and run a single pipeline.

    Args:
        config: Validated application configuration.
        mode: One of :data:`PIPELINES` keys.

    Returns:
        The structured result of the pipeline.

    Raises:
        ValueError: When the mode is unknown.
    """
    if mode not in PIPELINES:
        msg = f"Unknown mode '{mode}'. Allowed: {sorted([*PIPELINES, 'all'])}"
        raise ValueError(msg)

    pipeline_cls = PIPELINES[mode]
    logger.info("Running mode '{}' with pipeline {}", mode, pipeline_cls.__name__)
    pipeline = pipeline_cls(config)
    return pipeline.run()


def run_modes(config: AppConfig, modes: tuple[str, ...] | list[str]) -> dict[str, PipelineResult]:
    """Run several modes sequentially, stopping at the first failure.

    Args:
        config: Validated application configuration.
        modes: Ordered modes to execute.

    Returns:
        Mapping of mode to its :class:`PipelineResult`.
    """
    results: dict[str, PipelineResult] = {}
    for mode in modes:
        result = run_mode(config, mode)
        results[mode] = result
        if not result.succeeded:
            logger.error("Pipeline '{}' failed, stopping the sequence", mode)
            break
    return results


def summarize(results: dict[str, PipelineResult]) -> None:
    """Log a human readable summary of the executed pipelines.

    Args:
        results: Mapping of mode to pipeline result.
    """
    logger.info("-" * 78)
    for mode, result in results.items():
        metrics = ", ".join(f"{key}={value:.5f}" for key, value in list(result.metrics.items())[:6])
        logger.info(
            "{:<14} {:<8} {:>7.2f}s  {} artefacts  {}",
            mode,
            result.status,
            result.duration_seconds,
            len(result.artifacts),
            metrics or "-",
        )
    logger.info("-" * 78)


def main_flow(cfg: DictConfig | dict[str, Any]) -> dict[str, PipelineResult]:
    """Validate the configuration then execute the requested mode(s).

    This function is deliberately independent from Hydra decorators: it can be called from
    the tests, from a notebook or from ``scripts/*`` with a plain dictionary.

    Args:
        cfg: Raw configuration (OmegaConf or mapping).

    Returns:
        Mapping of mode to pipeline result.
    """
    config = validate_config(cfg)
    setup_logging(level=config.log_level, log_file=config.log_file)
    seeded = set_seed(config.seed)
    logger.debug("Random generators seeded: {}", seeded)
    logger.info(
        "Project '{}' ({} / {} / {}) | mode={} | seed={}",
        config.project.title,
        config.project.domain,
        config.project.problem,
        config.project.stack,
        config.mode,
        config.seed,
    )

    modes = ALL_MODES if config.mode == "all" else (config.mode,)
    results = run_modes(config, modes)
    summarize(results)
    return results


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point.

    Args:
        cfg: Composed configuration.

    Raises:
        SystemExit: With a non-zero code when at least one pipeline failed.
    """
    logger.debug("Composed configuration:\n{}", OmegaConf.to_yaml(cfg))
    results = main_flow(cfg)
    failed = [mode for mode, result in results.items() if not result.succeeded]
    if failed:
        raise SystemExit(f"Pipeline(s) failed: {failed}")


if __name__ == "__main__":  # pragma: no cover
    main()
