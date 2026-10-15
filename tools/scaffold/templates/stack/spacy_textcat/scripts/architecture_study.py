"""Étude de sélection des architectures spaCy, rejouable : celle qui a choisi le modèle servi.

Le protocole est celui du projet, et seulement celui-là : la configuration Hydra du projet (taille
de lot, budget d'époques), les défauts du registre pour chaque architecture (``params={}``, la
comparaison loyale du notebook 04), le split d'entraînement pour apprendre et le split de
**validation** pour mesurer — jamais le test. La seule chose que l'étude ajoute est la répétition
sur plusieurs graines : la dispersion entre graines est l'étalon qui décide si un écart de F1 est un
résultat ou un tirage.

Exemples :
    $ python scripts/architecture_study.py
    $ python scripts/architecture_study.py --seeds 42 0 1 7 19 --epochs 12
    $ python scripts/architecture_study.py --batch-size 16 --algorithms textcat_bow

La sortie est un tableau (moyenne et écart-type de la F1 macro de validation par architecture et
par époque) affiché et écrit dans ``outputs/architecture_study.csv``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Racine du projet (le dossier qui contient `src/` et `conf/`).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

from src.data.loaders import TextLabelLoader  # noqa: E402
from src.models import available_algorithms, build_model  # noqa: E402
from src.schemas.config import validate_config  # noqa: E402
from src.training.metrics import classification_metrics  # noqa: E402
from src.utils.paths import ProjectPaths  # noqa: E402

#: Graines de l'étude publiée dans le manifeste.
DEFAULT_SEEDS: tuple[int, ...] = (42, 0, 1, 7, 19)


@dataclass
class _Context:
    """Minimal callback context (the attributes the classifier writes and reads)."""

    epoch: int = 0
    logs: dict[str, float] = field(default_factory=dict)
    stopped_early: bool = False


class _ValidationScorer:
    """Callback that scores the validation split at the end of every epoch."""

    def __init__(self, model: Any, texts: list[str], truth: list[str]) -> None:
        self.model = model
        self.texts = texts
        self.truth = truth
        self.scores: list[float] = []

    def on_epoch_end(self, context: _Context) -> None:
        """Record the validation macro F1 of the current epoch."""
        del context
        # Pendant `fit`, le contrat refuse encore `predict_proba` (le modèle n'est déclaré ajusté
        # qu'à la fin) : le score d'époque passe par le calcul interne, sur le pipeline en cours.
        probabilities = self.model._predict_proba(self.texts)
        labels = list(self.model.labels)
        predicted = [labels[int(index)] for index in probabilities.argmax(axis=1)]
        self.scores.append(
            float(classification_metrics(self.truth, predicted, labels=labels)["macro_f1"])
        )


def run(argv: Sequence[str] | None = None) -> pd.DataFrame:
    """Run the study and return the per-epoch summary.

    Args:
        argv: Command line arguments (defaults to ``sys.argv``).

    Returns:
        One row per (algorithm, epoch): mean and standard deviation of the validation macro F1.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=None, help="Défaut : train.epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Défaut : train.batch_size")
    parser.add_argument("--algorithms", nargs="+", default=None, help="Défaut : toute la stack")
    args = parser.parse_args(list(argv) if argv is not None else None)

    overrides = ["mode=train", "log_level=WARNING"]
    if args.epochs is not None:
        overrides.append(f"train.epochs={args.epochs}")
    if args.batch_size is not None:
        overrides.append(f"train.batch_size={args.batch_size}")
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "conf"), version_base=None):
        config = validate_config(compose(config_name="config", overrides=overrides))

    paths = ProjectPaths.from_root(PROJECT_ROOT)
    loader = TextLabelLoader(paths, formats=config.data.formats)
    documents = loader.load_documents()
    train = loader.split("train", frame=documents)
    validation = loader.split("val", frame=documents)
    target = str(config.data.target or "label")
    texts = validation["text"].astype(str).tolist()
    truth = validation[target].astype(str).tolist()

    rows: list[dict[str, Any]] = []
    for algorithm in args.algorithms or available_algorithms():
        for seed in args.seeds:
            model = build_model(
                {**config.model_dump(), "seed": seed}, algorithm=algorithm, params={}
            )
            model.random_state = seed
            scorer = _ValidationScorer(model, texts, truth)
            model.fit(train, target=target, callbacks=[scorer], context=_Context())
            for epoch, score in enumerate(scorer.scores, start=1):
                rows.append({"algorithm": algorithm, "seed": seed, "epoch": epoch, "macro_f1": score})
            print(f"{algorithm} graine={seed} : {[round(score, 4) for score in scorer.scores]}")

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["algorithm", "epoch"])["macro_f1"]
        .agg(["mean", "std"])
        .reset_index()
        .round(4)
    )
    output = PROJECT_ROOT / "outputs" / "architecture_study.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(summary.to_string(index=False))
    print(f"\nÉtude écrite dans {output.relative_to(PROJECT_ROOT)}")
    return summary


if __name__ == "__main__":
    run()
