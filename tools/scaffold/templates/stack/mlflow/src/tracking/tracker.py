"""MLflow experiment tracker — local, explicit, reproducible.

Every ``mode=train`` opens **one MLflow run** and records, explicitly (no autolog, so that each
logged value can be read in this file):

* the resolved configuration, flattened (``model.params.n_estimators=300``), plus the seed;
* the training and validation metrics computed by the :class:`~src.training.trainer.Trainer`
  (``train_*`` and ``val_*``: an unprefixed ``roc_auc`` would be read as the model's score when it
  is only its fit on the training split);
* the artefacts written by the pipeline (model card, metrics, preprocessing, resolved config);
* the fitted scikit-learn estimator as an ``mlflow.sklearn`` model, with an input signature
  inferred on a few preprocessed rows — so ``mlflow.pyfunc.load_model`` can reload it anywhere.

``mode=evaluate`` reopens the **same run** (its id is kept in
``artifacts/metrics/mlflow_run.json``) and adds the test metrics prefixed with ``test_`` and the
evaluation report. Training and held-out evaluation therefore read side by side in ``mlflow ui``.

The backend is a local SQLite database under ``artifacts/mlruns/`` (ignored by git): no server,
no credential, no data leaving the workstation. A shared server is declared with
``tracking.tracking_uri``: the experiment is then created **without** a local artefact location, so
that the server stores the artefacts where every user can read them. Credentials never belong in
the URI: MLflow reads ``MLFLOW_TRACKING_USERNAME`` / ``MLFLOW_TRACKING_PASSWORD`` /
``MLFLOW_TRACKING_TOKEN`` from the environment, and any user info left in the URI is redacted
before being logged or written to disk.

The artefacts on disk remain the source of truth: a tracking failure during ``mode=evaluate`` is
reported as a warning, never as a failed evaluation.

The logged model is the estimator **alone**: it consumes the preprocessed matrix. The fitted
preprocessing is logged next to it as an artefact, because the published inference path
(:mod:`src.inference.predictor`) applies it before calling the model.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping
from importlib import metadata
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

from src.utils.io import read_json, write_json
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

# MLflow 3 imprime au premier import une note destinée aux assistants de code : inutile ici.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

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
#: Préfixes déjà qualifiés ; toute autre métrique du `Trainer` est mesurée sur le train.
_METRIC_PREFIXES = ("train_", "val_", "test_")
#: Blocs de configuration jamais journalisés comme paramètres (URI de suivi, éventuels secrets).
_UNLOGGED_PREFIXES = ("tracking.",)
#: Schémas d'URI dont le stockage d'artefacts doit rester local au projet.
_LOCAL_SCHEMES = ("file", "sqlite")


def redact_uri(uri: str) -> str:
    """Remove the user info (``user:token@``) of a URI before it is logged or written.

    Args:
        uri: Tracking URI.

    Returns:
        The same URI without credentials.
    """
    parts = urlsplit(uri)
    if not parts.netloc or "@" not in parts.netloc:
        return uri
    return urlunsplit(parts._replace(netloc=parts.netloc.rsplit("@", 1)[1]))


def is_local_backend(uri: str) -> bool:
    """Tell whether a tracking URI designates a backend stored on this workstation.

    Args:
        uri: Tracking URI.

    Returns:
        ``True`` for a file path, ``file:`` or ``sqlite:`` URI.
    """
    scheme = urlsplit(uri).scheme.lower()
    if not scheme:
        # Sans schéma : un chemin, sauf les mots-clés de backends distants (`databricks`...).
        return not uri.lower().startswith("databricks")
    # Un chemin Windows (`C:/...`) se lit comme un schéma d'une lettre : c'est un chemin local.
    return scheme in _LOCAL_SCHEMES or len(scheme) == 1


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
    import skops.io

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

    The ``tracking`` block is left out: it describes where runs are stored (and may carry a
    server address), not how the model was trained.

    Args:
        node: Configuration mapping.
        prefix: Prefix of the current level.

    Returns:
        One string value per leaf (lists are kept whole, truncated when too long).
    """
    flat: dict[str, str] = {}
    for key, value in node.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if name.startswith(_UNLOGGED_PREFIXES) or f"{name}.".startswith(_UNLOGGED_PREFIXES):
            continue
        if isinstance(value, Mapping):
            flat.update(flatten(value, name))
        else:
            flat[name] = str(value)[:_MAX_PARAM_LENGTH]
    return flat


def qualify_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Prefix the unqualified training metrics with ``train_``.

    Args:
        metrics: Metrics of the training pipeline (``roc_auc``, ``val_roc_auc``...).

    Returns:
        The metrics, each one qualified by the split it was measured on.
    """
    return {
        (key if key.startswith(_METRIC_PREFIXES) else f"train_{key}"): float(value)
        for key, value in metrics.items()
    }


class MlflowTracker:
    """Record training and evaluation runs in an MLflow backend (local by default)."""

    enabled = True

    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str,
        artifact_root: Path,
        run_file: Path,
        log_model: bool = True,
        log_input_example: bool = True,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        """Configure the tracker (nothing is written before the first run).

        Args:
            tracking_uri: MLflow backend store URI (``sqlite:///...`` by default).
            experiment_name: Experiment receiving the runs.
            artifact_root: Directory receiving the run artefacts of a **local** backend.
            run_file: JSON file keeping the id of the last training run.
            log_model: Whether the fitted estimator is logged as an MLflow model.
            log_input_example: Whether a few input rows are stored with the model (disable it
                on real personal data: the example lives next to the model).
            tags: Tags set on every run (stack, family, task).
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.artifact_root = Path(artifact_root)
        self.run_file = Path(run_file)
        self.log_model = bool(log_model)
        self.log_input_example = bool(log_input_example)
        self.tags = dict(tags or {})

    # ------------------------------------------------------------------ runs ------------
    def _experiment_id(self) -> str:
        """Return the experiment id, creating (or restoring) the experiment on first use."""
        import mlflow

        mlflow.set_tracking_uri(self.tracking_uri)
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is not None:
            if experiment.lifecycle_stage == "deleted":
                # Supprimée dans l'UI, l'expérience garde son nom : `start_run` y échouerait.
                mlflow.tracking.MlflowClient().restore_experiment(experiment.experiment_id)
                logger.warning("MLflow | expérience '{}' restaurée", self.experiment_name)
            return str(experiment.experiment_id)
        if not is_local_backend(self.tracking_uri):
            # Serveur partagé : il choisit lui-même où stocker les artefacts. Un chemin
            # `file:///C:/...` du poste qui crée l'expérience serait illisible pour les autres.
            return str(mlflow.create_experiment(self.experiment_name))
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
        import mlflow
        import mlflow.sklearn
        from mlflow.models import infer_signature

        # Le lien vers le run précédent est retiré AVANT tout appel à MLflow : si ce run-ci
        # échoue (base verrouillée, serveur injoignable, interruption), l'évaluation suivante ne
        # doit pas rattacher le nouveau modèle à l'ancien run.
        self.run_file.unlink(missing_ok=True)
        experiment_id = self._experiment_id()
        algorithm = str(getattr(model, "algorithm", "model"))
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"train-{algorithm}") as run:
            mlflow.set_tags({**self.tags, "algorithm": algorithm, "pipeline": "train"})
            params = sorted(flatten(config).items())
            for start in range(0, len(params), _PARAMS_PER_BATCH):
                mlflow.log_params(dict(params[start : start + _PARAMS_PER_BATCH]))
            mlflow.log_metrics(qualify_metrics(metrics))
            for path in artifacts:
                if Path(path).is_file():
                    mlflow.log_artifact(str(path), artifact_path="pipeline")
            estimator = getattr(model, "estimator_", None)
            if self.log_model and estimator is not None:
                rows = None if sample is None else sample.head(_EXAMPLE_ROWS)
                signature = None if rows is None else infer_signature(rows, estimator.predict(rows))
                mlflow.sklearn.log_model(
                    estimator,
                    name="model",
                    signature=signature,
                    input_example=rows if self.log_input_example else None,
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
                    skops_trusted_types=sklearn_trusted_types(estimator),
                    pip_requirements=pinned_requirements(),
                )
            run_id = run.info.run_id
        write_json(
            self.run_file,
            {
                "run_id": run_id,
                "experiment": self.experiment_name,
                "tracking_uri": redact_uri(self.tracking_uri),
            },
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
            logger.warning(
                "MLflow | aucun run d'entraînement connu ({}) : évaluation non suivie",
                self.run_file,
            )
            return
        link = dict(read_json(self.run_file))
        if str(link.get("tracking_uri")) != redact_uri(self.tracking_uri):
            logger.warning(
                "MLflow | le run {} vit dans un autre backend ({}) : évaluation non suivie",
                link.get("run_id"),
                link.get("tracking_uri"),
            )
            return
        run_id = str(link["run_id"])
        import mlflow
        from mlflow.exceptions import MlflowException

        mlflow.set_tracking_uri(self.tracking_uri)
        try:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics({f"test_{key}": float(value) for key, value in metrics.items()})
                for path in artifacts:
                    if Path(path).is_file():
                        mlflow.log_artifact(str(path), artifact_path="evaluation")
        except MlflowException as error:
            # Les rapports et métriques sont déjà sur disque : le suivi ne fait pas échouer
            # l'évaluation, il le signale.
            logger.warning("MLflow | évaluation non rattachée au run {} : {}", run_id, error)
            return
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
    configured_uri = node.get("tracking_uri")
    if not configured_uri:
        store.mkdir(parents=True, exist_ok=True)
    tracking_uri = str(configured_uri or f"sqlite:///{(store / 'mlflow.db').as_posix()}")
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
        log_input_example=bool(node.get("log_input_example", True)),
        tags=tags,
    )
