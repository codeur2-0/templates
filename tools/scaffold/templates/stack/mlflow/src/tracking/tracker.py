"""MLflow experiment tracker — local, explicit, reproducible.

Every ``mode=train`` opens **one MLflow run** and records, explicitly (no autolog, so that each
logged value can be read in this file):

* the resolved configuration, flattened (``model.params.n_estimators=300``), plus the seed;
* the training and validation metrics computed by the :class:`~src.training.trainer.Trainer`;
* the artefacts written by the pipeline (model card, metrics, preprocessing, resolved config);
* the fitted scikit-learn estimator as an ``mlflow.sklearn`` model, with an input signature
  inferred on a few preprocessed rows — so ``mlflow.pyfunc.load_model`` can reload it anywhere.

``mode=evaluate`` reopens the **same run** (its id is kept in
``artifacts/metrics/mlflow_run.json``) and adds the test metrics prefixed with ``test_`` and the
evaluation report. Training and held-out evaluation therefore read side by side in ``mlflow ui``.

The backend is a local SQLite database under ``artifacts/mlruns/`` (ignored by git): no server,
no credential, no data leaving the workstation. Pointing ``tracking.tracking_uri`` at a shared
server is a configuration change, not a code change.

The logged model is the estimator **alone**: it consumes the preprocessed matrix. The fitted
preprocessing is logged next to it as an artefact, because the published inference path
(:mod:`src.inference.predictor`) applies it before calling the model.
"""

from __future__ import annotations

import os
import tempfile
from importlib import metadata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import read_json, write_json
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

# MLflow 3 imprime au premier import une note destinée aux assistants de code : inutile ici.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import skops.io  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402

#: Fichier qui relie l'évaluation au run d'entraînement.
RUN_FILE = "mlflow_run.json"
#: MLflow refuse plus de 100 paramètres par appel groupé.
_PARAMS_PER_BATCH = 100
#: Longueur maximale d'une valeur de paramètre conservée (au-delà, elle est tronquée).
_MAX_PARAM_LENGTH = 500
#: Dépendances d'exécution du modèle journalisé, épinglées à la version installée. MLflow sait
#: les deviner en inspectant les modules chargés, mais il embarque alors tout ce que le processus
#: a importé (un venv partagé y ajoute torch) : une liste explicite est plus courte et exacte.
_MODEL_REQUIREMENTS = ("scikit-learn", "numpy", "pandas", "scipy", "skops")
#: Nombre de lignes conservées comme exemple d'entrée du modèle.
_EXAMPLE_ROWS = 5


def sklearn_trusted_types(estimator: Any) -> list[str]:
    """Return the skops types to trust for an estimator — scikit-learn internals only.

    MLflow 3 serialises scikit-learn models with **skops**, which refuses to reload any type it
    does not know (unlike pickle, which executes whatever the file contains). A random forest
    stores its trees as ``sklearn.tree._tree.Tree``, a type skops does not trust by default. We
    trust the types of the ``sklearn`` package and nothing else: a foreign type left in the list
    makes MLflow fail loudly instead of being trusted blindly.

    Args:
        estimator: Fitted scikit-learn estimator.

    Returns:
        The untrusted type names that belong to scikit-learn.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "estimator.skops"
        skops.io.dump(estimator, path)
        untrusted = skops.io.get_untrusted_types(file=path)
    return [name for name in untrusted if name.startswith("sklearn.")]


def pinned_requirements(packages: Iterable[str] = _MODEL_REQUIREMENTS) -> list[str]:
    """Return ``package==version`` for each installed package.

    Args:
        packages: Distribution names.

    Returns:
        The pinned requirements (packages that are not installed are skipped).
    """
    pinned: list[str] = []
    for package in packages:
        try:
            pinned.append(f"{package}=={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            continue
    return pinned


def flatten(node: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested configuration into ``dotted.key -> string`` parameters.

    Args:
        node: Configuration mapping.
        prefix: Prefix of the current level.

    Returns:
        One string value per leaf (lists are kept whole, truncated when too long).
    """
    flat: dict[str, str] = {}
    for key, value in node.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten(value, name))
        else:
            flat[name] = str(value)[:_MAX_PARAM_LENGTH]
    return flat


class MlflowTracker:
    """Record training and evaluation runs in a local MLflow backend."""

    enabled = True

    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str,
        artifact_root: Path,
        run_file: Path,
        log_model: bool = True,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        """Configure the tracker (nothing is written before the first run).

        Args:
            tracking_uri: MLflow backend store URI (``sqlite:///...`` by default).
            experiment_name: Experiment receiving the runs.
            artifact_root: Directory receiving the run artefacts.
            run_file: JSON file keeping the id of the last training run.
            log_model: Whether the fitted estimator is logged as an MLflow model.
            tags: Tags set on every run (stack, family, task).
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.artifact_root = Path(artifact_root)
        self.run_file = Path(run_file)
        self.log_model = bool(log_model)
        self.tags = dict(tags or {})

    # ------------------------------------------------------------------ runs ------------
    def _experiment_id(self) -> str:
        """Return the experiment id, creating the experiment on first use."""
        mlflow.set_tracking_uri(self.tracking_uri)
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is not None:
            return str(experiment.experiment_id)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        return str(
            mlflow.create_experiment(
                self.experiment_name, artifact_location=self.artifact_root.resolve().as_uri()
            )
        )

    def log_training(
        self,
        *,
        config: Mapping[str, Any],
        metrics: Mapping[str, float],
        model: Any,
        artifacts: Iterable[str | Path],
        sample: pd.DataFrame | None = None,
    ) -> None:
        """Open a run and record parameters, metrics, artefacts and the model.

        Args:
            config: Resolved configuration.
            metrics: Training / validation metrics.
            model: Fitted :class:`~src.models.base.BaseModel`.
            artifacts: Files produced by the training pipeline.
            sample: A few preprocessed rows (signature and input example).
        """
        experiment_id = self._experiment_id()
        algorithm = str(getattr(model, "algorithm", "model"))
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"train-{algorithm}") as run:
            mlflow.set_tags({**self.tags, "algorithm": algorithm, "pipeline": "train"})
            params = sorted(flatten(config).items())
            for start in range(0, len(params), _PARAMS_PER_BATCH):
                mlflow.log_params(dict(params[start : start + _PARAMS_PER_BATCH]))
            mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
            for path in artifacts:
                if Path(path).is_file():
                    mlflow.log_artifact(str(path), artifact_path="pipeline")
            estimator = getattr(model, "estimator_", None)
            if self.log_model and estimator is not None:
                example = None if sample is None else sample.head(_EXAMPLE_ROWS)
                signature = (
                    None if example is None else infer_signature(example, estimator.predict(example))
                )
                mlflow.sklearn.log_model(
                    estimator,
                    name="model",
                    signature=signature,
                    input_example=example,
                    skops_trusted_types=sklearn_trusted_types(estimator),
                    pip_requirements=pinned_requirements(),
                )
            run_id = run.info.run_id
        write_json(
            self.run_file,
            {"run_id": run_id, "experiment": self.experiment_name, "tracking_uri": self.tracking_uri},
        )
        logger.info("MLflow | run {} enregistré dans l'expérience '{}'", run_id, self.experiment_name)

    def log_evaluation(
        self, *, metrics: Mapping[str, float], artifacts: Iterable[str | Path]
    ) -> None:
        """Attach the test metrics and reports to the training run.

        Args:
            metrics: Test metrics (logged with a ``test_`` prefix).
            artifacts: Files produced by the evaluation.
        """
        if not self.run_file.is_file():
            logger.warning("MLflow | aucun run d'entraînement connu ({}) : évaluation non suivie", self.run_file)
            return
        run_id = str(read_json(self.run_file)["run_id"])
        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({f"test_{key}": float(value) for key, value in metrics.items()})
            for path in artifacts:
                if Path(path).is_file():
                    mlflow.log_artifact(str(path), artifact_path="evaluation")
        logger.info("MLflow | métriques de test ajoutées au run {}", run_id)


#: Interface attendue par les pipelines : ici, le traceur MLflow lui-même.
Tracker = MlflowTracker


def build_tracker(config: Mapping[str, Any], paths: ProjectPaths) -> MlflowTracker:
    """Build the MLflow tracker from the ``tracking`` block of ``conf/config.yaml``.

    Args:
        config: Resolved configuration (``model_dump()`` of the application config).
        paths: Project layout.

    Returns:
        The configured tracker.
    """
    node = dict(config.get("tracking") or {})
    store = paths.artifacts_dir / "mlruns"
    store.mkdir(parents=True, exist_ok=True)
    tracking_uri = str(node.get("tracking_uri") or f"sqlite:///{(store / 'mlflow.db').as_posix()}")
    project = dict(config.get("project") or {})
    experiment = str(node.get("experiment_name") or project.get("name") or "experiment")
    tags = {
        "stack": str(project.get("stack", "mlflow")),
        "task": str(dict(config.get("metrics") or {}).get("task", "")),
        "seed": str(config.get("seed", "")),
    }
    return MlflowTracker(
        tracking_uri=tracking_uri,
        experiment_name=experiment,
        artifact_root=store / "artifacts",
        run_file=paths.metrics_dir / RUN_FILE,
        log_model=bool(node.get("log_model", True)),
        tags=tags,
    )
