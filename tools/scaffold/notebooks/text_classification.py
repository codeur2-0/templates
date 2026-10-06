"""Notebooks pédagogiques des projets de **classification de texte** (famille `text_classification`).

Un projet de classification de texte ne s'explore pas comme une recherche documentaire : il n'y a
pas de rappel à régler, mais une **frontière de décision** à lire, une **classe minoritaire** à
protéger et une **confiance** dont il faut savoir quoi faire. Les six notebooks gardent la
progression commune aux autres modalités, avec les questions propres à la famille :

* ``01_eda.ipynb`` — cartographier le corpus (classes, styles, canaux, longueurs) et, surtout,
  mesurer ce qu'une **règle de surface** expliquerait : niveau de la classe majoritaire, niveau de
  la règle à mots-clés, gain des distracteurs déclarés. C'est la carte qui dit ce qu'une F1 macro
  vaudra plus tard ;
* ``02_validation.ipynb`` — les contrats Pandera du corpus et des prédictions, et surtout ce qui
  se passe quand on les casse : une règle qu'on n'a jamais vue échouer n'est pas une règle ;
* ``03_preprocessing.ipynb`` — normalisation, tokenisation, mots vides, puis **mesure** du prix du
  vocabulaire : ce que ``min_df`` retire au vecteur et ce qu'il coûte aux textes inédits ;
* ``04_model_exploration.ipynb`` — les trois algorithmes de la stack comparés sur le split de
  validation, l'effet du rééquilibrage sur la classe minoritaire, la courbe de calibration, puis un
  **seuil d'automatisation** arbitré sur la calibration et lu sur le test ;
* ``05_training.ipynb`` — le pipeline d'entraînement réel dans un bac à sable, ses cinq artefacts,
  le rechargement du modèle et la preuve de déterminisme ;
* ``06_error_analysis.ipynb`` — le verdict contractuel, la ventilation par style (ce que le modèle
  perd sur les paraphrases), la classe sacrifiée, les erreurs les plus confiantes expliquées terme
  à terme, et des recommandations chiffrées.

Les helpers de rendu (``_md``, ``_insight``, ``_objectives``) sont partagés avec les notebooks
tabulaire et texte : une classification de tickets parle la même langue typographique qu'une
régression, sans en recopier le contenu.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tools.scaffold.notebooks.tabular import _insight, _md, _objectives
from tools.scaffold.utils_notebooks import NotebookNode, code_cell, write_notebook

if TYPE_CHECKING:  # pragma: no cover
    from tools.scaffold.utils_notebooks import NotebookContext

__all__ = ["build_all"]

#: Taille du corpus réduit des notebooks (tickets) : l'exécution complète reste rapide tout en
#: laissant une dizaine de lignes d'entraînement par couple (classe, style).
NB_DOCUMENTS = 320

#: Seuils d'automatisation comparés par le notebook 04, en confiance annoncée.
THRESHOLD_GRID = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

#: Exactitude visée sur les tickets routés automatiquement (la cible de service).
TARGET_AUTOMATION_ACCURACY = 0.95


def _labels(context: NotebookContext) -> tuple[str, ...]:
    """Return the target labels declared by the manifest.

    Args:
        context: Notebook context.

    Returns:
        The label list of the target column, empty when the manifest declares none.
    """
    for column in context.spec.data.columns:
        if str(column.role) == "target":
            declared = dict(column.checks or {}).get("isin") or []
            return tuple(str(value) for value in declared)
    return ()


def _column_values(context: NotebookContext, name: str) -> tuple[str, ...]:
    """Return the declared values of one categorical column of the manifest.

    Args:
        context: Notebook context.
        name: Column name.

    Returns:
        The values declared by the column contract, empty when the column is absent.
    """
    for column in context.spec.data.columns:
        if str(column.name) == name:
            declared = dict(column.checks or {}).get("isin") or []
            return tuple(str(value) for value in declared)
    return ()


def _tokens(context: NotebookContext) -> dict[str, str]:
    """Build the substitution table of the classification notebooks.

    Args:
        context: Notebook context.

    Returns:
        Mapping of ``__TOKEN__`` to replacement text.
    """
    spec = context.spec
    artifacts = dict(spec.train.get("artifacts", {}))
    return {
        "__TITLE__": spec.title,
        "__PROJECT__": spec.project_slug,
        "__DATASET__": spec.data.dataset_name,
        "__SEED__": str(spec.data.seed),
        "__NB_DOCUMENTS__": str(NB_DOCUMENTS),
        "__PRIMARY__": str(spec.metrics.primary),
        "__MIN_PRIMARY__": str(spec.metrics.min_primary),
        "__TASK__": str(spec.metrics.task),
        "__TARGET__": str(spec.data.target or "label"),
        "__REPORT_NAME__": str(artifacts.get("report_file", "evaluation_report.md")),
        "__MODEL_FILE__": str(artifacts.get("model_file", "classifier.joblib")),
        "__LABELS__": repr(list(_labels(context))),
        "__STYLES__": repr(list(_column_values(context, "style"))),
        "__THRESHOLDS__": repr(list(THRESHOLD_GRID)),
        "__MEASURED_RESULTS__": repr(
            [(str(item.label), str(item.value)) for item in spec.business.measured_results]
        ),
        "__TARGET_AUTOMATION__": str(TARGET_AUTOMATION_ACCURACY),
    }


def _text(source: str, context: NotebookContext) -> NotebookNode:
    """Render a code cell with the classification tokens substituted.

    Args:
        source: Code template (jetons ``__XXX__``).
        context: Notebook context.

    Returns:
        The rendered code cell.
    """
    rendered = source
    for token, value in _tokens(context).items():
        rendered = rendered.replace(token, value)
    return code_cell(rendered)


def _header(
    context: NotebookContext, number: str, title: str, objectives: list[str]
) -> list[NotebookNode]:
    """Build the title and the objectives cells of a notebook.

    Args:
        context: Notebook context.
        number: Notebook number (``01``, ``02``, …).
        title: Notebook title.
        objectives: Pedagogical objectives (French prose).

    Returns:
        The two opening cells.
    """
    return [
        _md(f"# {number} — {title}\n\n**Projet** : {context.spec.title}\n"),
        _objectives(context, objectives),
    ]


SETUP = """
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

# --- Racine du projet ---------------------------------------------------------------------------
# Le notebook s'exécute depuis `notebooks/` : on remonte d'un cran pour pouvoir importer `src`.
PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

from src.schemas.config import validate_config  # noqa: E402
from src.utils.config_access import node  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.paths import ProjectPaths  # noqa: E402

# --- Réglages d'affichage -----------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25})
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 170)
# Le projet journalise au niveau INFO ; un notebook noierait ses tableaux sous ces lignes.
setup_logging(level="WARNING")

# --- Configuration : exactement celle de `python -m src.main` -----------------------------------
# Les notebooks travaillent sur un corpus réduit (__NB_DOCUMENTS__ tickets) quand `data/raw` est
# vide : l'exécution reste rapide. Si le corpus de référence a été généré par `make data`, il est
# utilisé tel quel — les valeurs absolues d'un notebook ne sont pas celles de la référence.
NB_DOCUMENTS = __NB_DOCUMENTS__

GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=str(PROJECT_ROOT / "conf"), version_base=None):
    CONFIG = validate_config(
        compose(
            config_name="config",
            overrides=[
                "mode=train",
                f"data.n_samples={NB_DOCUMENTS}",
                "seed=__SEED__",
                "log_level=WARNING",
                "train.callbacks.progress_bar=false",
            ],
        )
    )

PATHS = ProjectPaths.from_root(PROJECT_ROOT)
# Les notebooks lisent le corpus du projet et écrivent leurs artefacts dans `outputs/notebooks` :
# ils ne doivent jamais écraser ceux produits par `make train` / `make evaluate`.
NB_PATHS = ProjectPaths(
    root=PROJECT_ROOT,
    data_dir=PATHS.data_dir,
    artifacts_dir=PROJECT_ROOT / "outputs" / "notebooks" / "artifacts",
).ensure()

MODEL_NODE = dict(node(CONFIG, "model"))
TEXT_COLUMN = str(MODEL_NODE.get("text_column", "text"))
TARGET_COLUMN = str(CONFIG.data.target or "__TARGET__")
LABELS = __LABELS__
STYLES = __STYLES__

print(f"Projet            : {CONFIG.project.name}")
print(f"Tâche             : {CONFIG.metrics.task}")
print(f"Métrique primaire : {CONFIG.metrics.primary} (seuil contractuel : __MIN_PRIMARY__)")
print(f"Algorithme        : {CONFIG.model.algorithm} ({CONFIG.model.name})")
print(f"Classes           : {len(LABELS)} — {', '.join(LABELS)}")
print(f"Sorties notebook  : {NB_PATHS.artifacts_dir.relative_to(PROJECT_ROOT)}")
"""

LOAD_CORPUS = """
from src.data.generators import SyntheticTicketGenerator
from src.data.loaders import TextLabelLoader

LOADER = TextLabelLoader(
    PATHS,
    formats=CONFIG.data.formats,
    validation_enabled=CONFIG.data.validation.raw,
    lazy_validation=CONFIG.data.validation.lazy,
)
if LOADER.documents_path.exists():
    DOCUMENTS = LOADER.load_documents()
    METADATA = LOADER.load_metadata()
    print(f"Corpus lu depuis data/raw : {len(DOCUMENTS)} tickets, graine {METADATA.get('seed')}")
else:
    # Un clone frais n'a pas encore de données : le notebook reste exécutable. Le corpus réduit est
    # persisté pour que les pipelines (train, evaluate) trouvent les mêmes fichiers que le
    # notebook ; `make data` régénère le corpus de référence.
    bundle = SyntheticTicketGenerator.from_config(CONFIG.data, seed=CONFIG.seed).generate()
    DOCUMENTS, METADATA = bundle.documents, bundle.metadata
    LOADER.save_documents(DOCUMENTS)
    LOADER.save_metadata(METADATA)
    print(f"data/raw absent : corpus de {len(DOCUMENTS)} tickets généré et persisté")

TRAIN = LOADER.split("train", frame=DOCUMENTS)
VALIDATION = LOADER.split("val", frame=DOCUMENTS)
CALIBRATION = LOADER.split("calibration", frame=DOCUMENTS)
TEST = LOADER.split("test", frame=DOCUMENTS)
# La calibration sert à **régler** (seuil d'automatisation), la validation à **choisir**
# (algorithme, rééquilibrage), et le test à **juger** — une seule fois, à la fin.
print("splits :", {"train": len(TRAIN), "val": len(VALIDATION),
                   "calibration": len(CALIBRATION), "test": len(TEST)})

LABEL_COUNTS = DOCUMENTS[TARGET_COLUMN].value_counts()
MAJORITY_LABEL = str(LABEL_COUNTS.index[0])
MINORITY_LABEL = str(LABEL_COUNTS.index[-1])
print(f"classe majoritaire : {MAJORITY_LABEL} ({LABEL_COUNTS.iloc[0] / len(DOCUMENTS):.1%}) "
      f"| classe minoritaire : {MINORITY_LABEL} ({LABEL_COUNTS.iloc[-1] / len(DOCUMENTS):.1%})")
"""


# ---------------------------------------------------------------------------------------------
# 01 — exploration du corpus
# ---------------------------------------------------------------------------------------------
def build_01_eda(context: NotebookContext, destination: Path) -> Path:
    """Build ``01_eda.ipynb``: cartographie du corpus et mesure des raccourcis.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    cells: list[NotebookNode] = [
        *_header(
            context,
            "01",
            "Exploration du corpus — ce qu'une règle de surface expliquerait déjà",
            [
                "Décrire un corpus multiclasse : effectifs par classe, par style, par canal, "
                "longueurs, dates.",
                "Mesurer le **niveau à battre** : classe majoritaire, tirage stratifié, et la "
                "règle à mots-clés publiée avec le corpus.",
                "Chiffrer les **distracteurs déclarés** (priorité et canal) au lieu d'affirmer "
                "qu'ils sont inutiles.",
                "Vérifier la propriété qui rend le score crédible : les formulations du test ne "
                "sont jamais vues à l'entraînement.",
            ],
        ),
        _md(
            "## 1. Mise en place\n\n"
            "Le notebook charge **la configuration du projet**, pas une copie : la même que "
            "`python -m src.main`. Les valeurs affichées plus bas sont donc celles du run réel, à "
            "la taille du corpus près (les notebooks travaillent sur un corpus réduit quand "
            "`data/raw` est vide, pour rester exécutables en quelques secondes)."
        ),
        _text(SETUP, context),
        _text(LOAD_CORPUS, context),
        _insight(
            [
                "Le corpus, ses métadonnées et ses splits viennent du **même générateur** que "
                "celui de `make data` : le notebook ne fabrique pas un jeu d'exemple à part.",
                "Les quatre splits sont **écrits dans le corpus** : deux exécutions voient les "
                "mêmes lignes de test, donc les chiffres publiés sont comparables.",
                "Le corpus réduit des notebooks est persisté dans `data/raw`, comme celui de "
                "`make data` : les pipelines lancés ensuite lisent exactement ce que le notebook "
                "a lu.",
            ],
            title="Ce que la cellule précédente vient d'établir",
        ),
        _md(
            "## 2. Effectifs : les classes, les styles, les canaux\n\n"
            "Trois tableaux, trois questions différentes. Les **classes** sont la cible du modèle. "
            "Les **styles** disent *comment* le ticket est écrit : c'est le découpage qui "
            "distinguera plus tard « lire un mot » de « comprendre une demande ». Les **canaux** "
            "et la **priorité** sont des distracteurs déclarés : ils sont tirés indépendamment du "
            "libellé, et la section 5 le chiffre."
        ),
        _text(
            """
summary = pd.DataFrame(
    {
        "tickets": LABEL_COUNTS,
        "part": (LABEL_COUNTS / len(DOCUMENTS)).round(4),
        "split train": TRAIN[TARGET_COLUMN].value_counts(),
        "split test": TEST[TARGET_COLUMN].value_counts(),
    }
).rename_axis("classe")
display(summary)

shares = pd.DataFrame(
    {
        "style": (DOCUMENTS["style"].value_counts() / len(DOCUMENTS)).round(4),
        "source": (DOCUMENTS["source"].value_counts() / len(DOCUMENTS)).round(4),
        "priority": (DOCUMENTS["priority"].value_counts() / len(DOCUMENTS)).round(4),
    }
)
display(shares)
""",
            context,
        ),
        _insight(
            [
                "Aucune classe n'écrase les autres, mais la plus fréquente pèse plusieurs fois la "
                "plus rare : une exactitude globale récompenserait un modèle qui ignorerait la "
                "classe difficile — d'où la F1 **macro** comme métrique principale.",
                "Les trois styles et les quatre canaux sont dans les proportions annoncées par le "
                "manifeste : le corpus réduit respecte les mêmes parts que le corpus de "
                "référence.",
                "Les styles sont répartis dans les deux splits : le test contient donc bien des "
                "tickets canoniques, des paraphrases et des tickets bruités.",
            ]
        ),
        _md(
            "## 3. Croisement des styles et des classes et longueurs\n\n"
            "Un segment n'est lisible que s'il est rempli. On vérifie donc que chaque style couvre "
            "chaque classe — et on regarde la longueur des tickets, parce qu'un classifier "
            "lexical peut apprendre « les tickets longs sont des remboursements » sans rien "
            "comprendre à la demande."
        ),
        _text(
            """
cross = pd.crosstab(DOCUMENTS["style"], DOCUMENTS[TARGET_COLUMN], margins=True)
display(cross)

lengths = (
    DOCUMENTS.groupby([TARGET_COLUMN, "style"], observed=True)["n_tokens"]
    .agg(["size", "mean", "median", "max"])
    .round(2)
)
display(lengths.head(12))
print("longueur moyenne :", round(float(DOCUMENTS["n_tokens"].mean()), 2),
      "| médiane :", float(DOCUMENTS["n_tokens"].median()))
print("part des tickets sous 20 tokens :",
      f"{float((DOCUMENTS['n_tokens'] < 20).mean()):.1%}")
""",
            context,
        ),
        _text(
            """
from src.visualization.plots import plot_lengths

FIGURE = plot_lengths(DOCUMENTS, NB_PATHS.figures_dir / "01_lengths.png")
print(f"figure écrite : {FIGURE.relative_to(PROJECT_ROOT)}")
""",
            context,
        ),
        _insight(
            [
                "Chaque couple (classe, style) existe, donc chaque segment de la future ventilation "
                "aura de quoi mesurer quelque chose.",
                "Les longueurs se chevauchent d'une classe à l'autre : la longueur seule ne peut "
                "pas trancher — c'est ce que le test de raccourci vérifie numériquement plus bas.",
                "Les tickets bruités sont mécaniquement plus longs (politesse et signature) : leur "
                "exactitude devra être lue à côté de celle des tickets canoniques, jamais seule.",
            ]
        ),
        _md(
            "## 4. Le niveau à battre\n\n"
            "Un score n'a de sens que situé. On mesure ici les trois références du projet : la "
            "classe majoritaire (le plancher absolu), le tirage stratifié (ce que le hasard bien "
            "fait obtient) et la **règle à mots-clés** publiée avec le corpus — celle qu'un "
            "gestionnaire de support écrirait en dix lignes dans un tableur."
        ),
        _text(
            """
from src.data.generators import class_keywords, keyword_baseline_accuracy
from src.training.metrics import majority_baseline, stratified_baseline

truth = DOCUMENTS[TARGET_COLUMN].astype(str)
baselines = {
    "classe majoritaire": majority_baseline(truth),
    "tirage stratifie": stratified_baseline(truth, seed=CONFIG.seed),
}
baseline_table = pd.DataFrame(baselines).T
display(baseline_table.round(4))

rule_accuracy = keyword_baseline_accuracy(DOCUMENTS)
keywords = class_keywords()
rule_table = pd.DataFrame(
    {
        "classe": list(keywords),
        "mots-clés": [", ".join(terms) if terms else "(aucun : la classe « autre »)"
                      for terms in keywords.values()],
    }
).set_index("classe")
display(rule_table)
print(f"règle à mots-clés : exactitude {rule_accuracy:.4f} "
      f"(classe majoritaire : {baselines['classe majoritaire']['accuracy']:.4f})")
""",
            context,
        ),
        _insight(
            [
                "Le plancher est bas : la classe majoritaire vaut une fraction des tickets, et la "
                "F1 macro de ce plancher est catastrophique — c'est la métrique qui protège les "
                "classes rares.",
                "La règle à mots-clés **bat largement** la classe majoritaire : un projet qui ne la "
                "dépasse pas n'aurait rien apporté à l'équipe support, et c'est écrit dans le "
                "rapport.",
                "La classe « autre » n'a aucun mot-clé par construction : aucune règle de surface "
                "ne peut la prédire, et c'est là que la F1 macro se joue.",
            ]
        ),
        _md(
            "## 5. Distracteurs déclarés : le test de raccourci\n\n"
            "La priorité déclarée par le client et le canal d'arrivée sont tentants : ils sont "
            "disponibles avant la lecture du texte. S'ils suffisaient à prédire le libellé, le "
            "projet serait un exercice de tableur. On mesure donc le gain de la meilleure règle à "
            "un seuil construite sur chacun d'eux, et on le compare au niveau de la classe "
            "majoritaire — pas à zéro."
        ),
        _text(
            """
from src.features.build_features import TEXT_FEATURES, shortcut_scores

candidates = [*TEXT_FEATURES, "source", "priority"]
scores = shortcut_scores(DOCUMENTS)
shortcut_table = (
    pd.DataFrame({"gain": [scores[name] for name in candidates]}, index=candidates)
    .assign(marge_vs_plancher=lambda frame: (frame["gain"] - scores["majority_baseline"]).round(4))
    .sort_values("gain", ascending=False)
    .round(4)
)
display(shortcut_table)

print(f"niveau de la classe majoritaire : {scores['majority_baseline']:.4f}")
print(f"meilleur gain de surface        : {scores['shortcut_margin']:.4f}")
assert scores["source"] < scores["majority_baseline"] + 0.05
assert scores["priority"] < scores["majority_baseline"] + 0.05
print("contrat vérifié : ni le canal ni la priorité ne prédisent le libellé")
""",
            context,
        ),
        _insight(
            [
                "Le canal et la priorité ne font pas mieux qu'une règle constante : ce sont bien "
                "des distracteurs, et le test de raccourci le vérifie à chaque exécution du "
                "notebook.",
                "Les traits de forme (longueur, politesse, chiffres) restent sous le niveau du "
                "plancher + 0,05 : mémoriser la forme des tickets ne suffit pas à les router.",
                "Ce tableau est le même que celui que le rapport d'évaluation publie : le notebook "
                "et la CI regardent le même instrument.",
            ]
        ),
        _md(
            "## 6. La formulation du test n'est jamais vue à l'entraînement\n\n"
            "Dernière vérification, et la plus importante : chaque pool de formulations "
            "(classe et style) est coupé en deux, quelques tournures étant réservées aux splits "
            "d'évaluation. Sans cette réserve, un modèle lexical se contenterait de recopier des "
            "gabarits mémorisés et la F1 publiée serait un score de recopie."
        ),
        _text(
            """
import re

from src.data.generators import PRODUCTS
from src.preprocessing.transformers import normalise_text

ORDER = re.compile(r"cmd-\\d{5}")

def wording(text: str) -> str:
    \"\"\"Rend la formulation d'un ticket : ses valeurs tirées deviennent des jetons neutres.\"\"\"
    lowered = normalise_text(str(text)).lower()
    for product in PRODUCTS:
        lowered = lowered.replace(product.lower(), "{produit}")
    return ORDER.sub("{commande}", lowered).strip()

rows = []
for (label, style), group in DOCUMENTS.groupby([TARGET_COLUMN, "style"], observed=True):
    learned = {wording(text) for text in group.loc[group["split"] == "train", "text"]}
    evaluated = {wording(text) for text in group.loc[group["split"] != "train", "text"]}
    rows.append(
        {
            "classe": label,
            "style": style,
            "formulations vues (train)": len(learned),
            "formulations évaluées": len(evaluated),
            "intersection": len(learned & evaluated),
        }
    )
holdout = pd.DataFrame(rows)
display(holdout.head(9))
assert (holdout["intersection"] == 0).all(), "une formulation du test a été vue à l'entraînement"
print(f"réserve déclarée par le manifeste : {METADATA.get('phrasing_holdout')} formulations "
      "par couple (classe, style)")
print("contrat vérifié : aucune formulation d'évaluation n'apparaît dans le train")
""",
            context,
        ),
        _text(
            """
example = TEST.loc[TEST["style"] == "paraphrase", "text"].iloc[0]
print("exemple de ticket du test (paraphrase) :")
print(" ", example)
print()
print("cinq tickets canoniques du train, pour comparer les tournures :")
for text in TRAIN.loc[TRAIN["style"] == "canonique", "text"].head(5):
    print(" -", text)
""",
            context,
        ),
        _insight(
            [
                "Les formulations du test sont disjointes de celles du train : un score élevé ne "
                "peut pas venir d'un gabarit mémorisé.",
                "La réserve est un paramètre du corpus (`data.corpus.phrasing_holdout`), pas une "
                "constante cachée dans le générateur.",
                "C'est cette propriété qui rend la comparaison avec la variante `transformers` "
                "honnête : les deux modèles sont jugés sur les mêmes tournures inédites.",
            ]
        ),
    ]
    return _write(destination, "01_eda", cells)


# ---------------------------------------------------------------------------------------------
# 02 — contrats de données
# ---------------------------------------------------------------------------------------------
def build_02_validation(context: NotebookContext, destination: Path) -> Path:
    """Build ``02_validation.ipynb``: contrats Pandera du corpus et des prédictions.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    cells: list[NotebookNode] = [
        *_header(
            context,
            "02",
            "Contrats de données — ce que le projet refuse d'ingérer",
            [
                "Lire les contrats Pandera du corpus et de la table de prédictions.",
                "Casser chaque règle volontairement et lire le message d'erreur produit.",
                "Comprendre le mode *lazy* : collecter toutes les violations en un seul passage.",
                "Relier chaque colonne du contrat à l'usage qui en est fait plus loin.",
            ],
        ),
        _md("## 1. Mise en place"),
        _text(SETUP, context),
        _text(LOAD_CORPUS, context),
        _md(
            "## 2. Le contrat du corpus\n\n"
            "Le corpus est décrit une seule fois, dans `src/data/schemas.py`, et ce contrat sert à "
            "la génération, au chargement et à l'écriture. Un schéma qui ne sert qu'à documenter "
            "ne protège personne : ici, tout ce qui entre dans le projet passe par lui."
        ),
        _text(
            """
from src.data.schemas import (
    LABELS as DECLARED_LABELS,
    probability_columns,
    validate_predictions,
    validate_tickets,
)

validated = validate_tickets(DOCUMENTS)
print(f"corpus validé : {len(validated)} lignes, {len(validated.columns)} colonnes")
print("colonnes :", ", ".join(str(name) for name in validated.columns))
print("classes déclarées :", DECLARED_LABELS)
print("colonnes de probabilités :", probability_columns())
""",
            context,
        ),
        _text(
            """
from pandera.errors import SchemaError, SchemaErrors

def expect_failure(label: str, frame: pd.DataFrame, validator) -> None:
    \"\"\"Montre le refus du contrat : une règle qu'on n'a jamais vue échouer n'est pas une règle.\"\"\"
    try:
        validator(frame)
    except (SchemaError, SchemaErrors) as error:
        first = str(error).strip().splitlines()[0]
        print(f"[refusé] {label}\\n         {first[:150]}")
    else:
        raise AssertionError(f"{label} aurait dû être refusé par le contrat")

sabotaged = DOCUMENTS.copy()
sabotaged.loc[sabotaged.index[1], "doc_id"] = sabotaged.loc[sabotaged.index[0], "doc_id"]
expect_failure("identifiant dupliqué", sabotaged, validate_tickets)

sabotaged = DOCUMENTS.copy()
sabotaged.loc[sabotaged.index[0], TARGET_COLUMN] = "reclamation_inconnue"
expect_failure("classe hors nomenclature", sabotaged, validate_tickets)

sabotaged = DOCUMENTS.copy()
sabotaged.loc[sabotaged.index[0], "text"] = "trop court"
expect_failure("texte tronqué", sabotaged, validate_tickets)

sabotaged = DOCUMENTS.copy()
sabotaged.loc[sabotaged.index[0], "n_tokens"] = -3
expect_failure("longueur négative", sabotaged, validate_tickets)
""",
            context,
        ),
        _insight(
            [
                "Chaque refus nomme la colonne et la règle : le message s'adresse à la personne qui "
                "devra corriger la donnée, pas au développeur du schéma.",
                "L'identifiant dupliqué est refusé parce que toute la reproductibilité du projet "
                "repose sur `doc_id` (jointures entre corpus, prédictions et rapport).",
                "Une classe inconnue est refusée **avant** l'entraînement : c'est le genre d'erreur "
                "qui produirait sinon un modèle silencieusement amputé d'une catégorie.",
            ]
        ),
        _md(
            "## 3. Le contrat des prédictions\n\n"
            "La table de prédictions est ce qui sort du projet : elle est jointe au rapport et "
            "relue par l'équipe support. Son contrat est donc aussi strict que celui du corpus — "
            "avec une différence de taille : `expected_label` est **nullable**, parce qu'à "
            "l'inférence la vérité terrain n'existe pas."
        ),
        _text(
            """
from src.models import build_model

model = build_model(CONFIG)
model.fit(TRAIN, target=TARGET_COLUMN)
scored = model.predict_frame(VALIDATION, text_column=TEXT_COLUMN)
scored.insert(0, "doc_id", VALIDATION["doc_id"].to_numpy())
scored.insert(1, "text", VALIDATION[TEXT_COLUMN].astype(str).to_numpy())
scored["expected_label"] = VALIDATION[TARGET_COLUMN].astype(str).to_numpy()
scored["correct"] = (scored["prediction"] == scored["expected_label"]).astype(int)
scored["latency_ms"] = 0.0

predictions = validate_predictions(scored)
print(f"prédictions validées : {len(predictions)} lignes")
display(predictions[["doc_id", "expected_label", "prediction", "confidence", "correct"]].head())
print("probabilités d'une ligne :",
      predictions.loc[0, list(probability_columns())].round(4).to_dict())
""",
            context,
        ),
        _text(
            """
broken = scored.copy()
broken.loc[broken.index[0], probability_columns()[0]] = 0.99
expect_failure("probabilités qui ne somment plus à 1", broken, validate_predictions)

broken = scored.copy()
broken.loc[broken.index[0], "confidence"] = 1.4
expect_failure("confiance hors de [0, 1]", broken, validate_predictions)

broken = scored.copy()
broken.loc[broken.index[0], "prediction"] = "colis_perdu"
expect_failure("libellé inconnu dans la prédiction", broken, validate_predictions)

broken = scored.drop(columns=["latency_ms"])
expect_failure("colonne obligatoire absente", broken, validate_predictions)
""",
            context,
        ),
        _insight(
            [
                "Le contrôle de somme des probabilités est fait **au niveau du schéma** : une table "
                "de prédictions qui n'est plus une distribution est refusée, pas arrondie en "
                "silence.",
                "Le contrat est strict sur les colonnes (`strict = True`) : ajouter une colonne "
                "implicite dans un notebook ne passera pas la porte du rapport.",
                "Le contrôle « latence ≥ 0 » empêche de publier une mesure de performance "
                "négative, qui est toujours le signe d'une horloge mal instrumentée.",
            ]
        ),
        _md(
            "## 4. Validation paresseuse : toutes les erreurs d'un seul passage\n\n"
            "En mode `lazy`, Pandera collecte **toutes** les violations avant de lever. C'est ce "
            "qu'on veut sur un rapport de qualité de données : corriger dix lignes en un passage "
            "coûte moins cher que dix allers-retours."
        ),
        _text(
            """
lazy_frame = DOCUMENTS.copy()
lazy_frame.loc[lazy_frame.index[0], TARGET_COLUMN] = "hors_liste"
lazy_frame.loc[lazy_frame.index[1], "n_tokens"] = -1
lazy_frame.loc[lazy_frame.index[2], "text"] = "court"

try:
    validate_tickets(lazy_frame, lazy=True)
except SchemaErrors as errors:
    failures = errors.failure_cases
    print(f"{len(failures)} violations collectées d'un coup")
    display(failures[["column", "check", "failure_case"]].head(8))
""",
            context,
        ),
        _text(
            """
validation_node = dict(node(CONFIG, "data").get("validation", {}))
print("politique de validation du projet :", validation_node)
print("rappel : `raw` couvre le chargement du corpus, `processed` les tables intermédiaires,")
print("         `inference` les textes arrivant en production, `strict`/`lazy` règlent la sévérité.")
""",
            context,
        ),
        _insight(
            [
                "Le mode paresseux sert aux rapports qualité ; le mode strict sert à la CI, où l'on "
                "veut échouer vite.",
                "Les quatre interrupteurs (`raw`, `processed`, `inference`, `strict`/`lazy`) sont "
                "dans la configuration Hydra : aucune validation ne s'active par surprise.",
                "Un contrat qu'on peut désactiver est un contrat dont il faut lire la "
                "configuration — c'est exactement ce que fait la cellule précédente.",
            ]
        ),
    ]
    return _write(destination, "02_validation", cells)


# ---------------------------------------------------------------------------------------------
# 03 — prétraitement et traits
# ---------------------------------------------------------------------------------------------
def build_03_preprocessing(context: NotebookContext, destination: Path) -> Path:
    """Build ``03_preprocessing.ipynb``: tokenisation, traits de surface, prix du vocabulaire.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    cells: list[NotebookNode] = [
        *_header(
            context,
            "03",
            "Prétraitement et traits de surface — mesurer le prix du vocabulaire",
            [
                "Suivre une phrase de ticket à travers normalisation, tokenisation et mots vides.",
                "Décrire les textes avec des traits de surface, par classe.",
                "Mesurer ce que `min_df` retire au vecteur et ce qu'il coûte aux textes inédits.",
                "Vérifier qu'aucun trait de surface ne suffit à prédire le libellé.",
            ],
        ),
        _md("## 1. Mise en place"),
        _text(SETUP, context),
        _text(LOAD_CORPUS, context),
        _md(
            "## 2. D'une phrase brute à une suite de tokens\n\n"
            "La normalisation est un choix, pas une évidence : elle décide de ce que le modèle "
            "pourra confondre. On la lit sur un exemple **bruité**, parce que c'est là qu'elle "
            "travaille : politesse, majuscules et référence de commande."
        ),
        _text(
            """
from src.preprocessing.transformers import (
    TextPreprocessingPipeline,
    normalise_text,
    tokenize,
)

example = DOCUMENTS.loc[DOCUMENTS["style"] == "bruite", "text"].iloc[0]
print("brut      :", str(example)[:180])
print("normalisé :", normalise_text(str(example))[:180])
print("tokens    :", tokenize(example)[:14], f"... ({len(tokenize(example))} tokens)")

preprocessing = TextPreprocessingPipeline.from_config(node(CONFIG, "preprocessing"))
print("politique de découpage :", dict(node(CONFIG, "preprocessing")).get("chunking", {}))
print("tokenisation du pipeline :", preprocessing.transform([str(example)])[0][:14], "...")

stop_words = preprocessing.stop_words
kept_before = tokenize(example)
kept_after = stop_words.transform(kept_before)
print(f"avant filtre : {len(kept_before)} tokens | après filtre : {len(kept_after)}")
print("mots vides retirés :", [token for token in kept_before if token not in kept_after][:10])
""",
            context,
        ),
        _insight(
            [
                "La normalisation met en minuscules et déplie les espaces : « CMD-12345 » devient "
                "un token stable, que la référence soit écrite en majuscules ou non.",
                "Les mots vides sont retirés **par le même code** que celui de l'entraînement : le "
                "notebook ne réimplémente rien.",
                "Une phrase de politesse produit des tokens sans information : ils seront présents "
                "dans toutes les classes — donc inutilisables comme signal, mais capables de noyer "
                "les termes utiles si le vecteur les sur-pondère.",
            ]
        ),
        _md(
            "## 3. Traits de surface, par classe\n\n"
            "Avant de vectoriser, on décrit. Ces traits ne servent pas de features au modèle "
            "principal : ils servent à **comprendre** ce que le corpus contient, et à vérifier "
            "qu'il ne contient pas de raccourci."
        ),
        _text(
            """
from src.features.build_features import build_text_features, feature_summary

enriched = build_text_features(TRAIN)
surface = feature_summary(enriched)
display(surface.round(3))
""",
            context,
        ),
        _text(
            """
profiles = surface.drop(index="all")
figure, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].barh(profiles.index, profiles["n_words"], color="#4c72b0")
axes[0].set_title("Longueur moyenne par classe (tokens)")
axes[0].set_xlabel("tokens")
axes[1].barh(profiles.index, profiles["has_signature"], color="#55a868")
axes[1].set_title("Part des tickets signés")
axes[1].set_xlabel("part")
figure.tight_layout()
plt.show()
""",
            context,
        ),
        _insight(
            [
                "Le tableau par classe est la première lecture d'un corpus de texte : il dit si les "
                "classes se ressemblent en surface.",
                "Les écarts de longueur sont modestes — c'est voulu : la forme ne doit pas être un "
                "signal, sinon le projet mesurerait la verbosité des clients.",
                "La ligne `all` (toute la population) sert de référence : un segment qui s'en écarte "
                "fortement est un segment qu'il faudra regarder en particulier.",
            ]
        ),
        _md(
            "## 4. Mesurer le prix d'un vocabulaire\n\n"
            "`min_df` jette les termes trop rares. Trop petit, le vecteur mémorise des coquilles ; "
            "trop grand, un texte qui utilise une tournure inédite n'a plus rien à dire au modèle. "
            "On **mesure** les deux effets sur le corpus du projet plutôt que de recopier une "
            "valeur par défaut."
        ),
        _text(
            """
from src.preprocessing.transformers import LexicalVectorizer

lexical_config = dict(dict(node(CONFIG, "model").get("params", {})).get("lexical", {}))
train_texts = TRAIN[TEXT_COLUMN].astype(str).tolist()
test_texts = TEST[TEXT_COLUMN].astype(str).tolist()
test_tokens = [len(tokenize(text)) for text in test_texts]

rows = []
for min_df in (1, 2, 3, 5):
    vectorizer = LexicalVectorizer.from_config({**lexical_config, "min_df": min_df})
    vectorizer.fit(train_texts)
    matrix = np.asarray(vectorizer.transform(test_texts))
    known = np.asarray((matrix != 0).sum(axis=1)).ravel()
    rows.append(
        {
            "min_df": min_df,
            "vocabulaire": vectorizer.vocabulary_size,
            "termes connus par ticket": round(float(known.mean()), 2),
            "tickets muets": int((known == 0).sum()),
            "part des tickets muets": round(float((known == 0).mean()), 4),
        }
    )
grid = pd.DataFrame(rows).set_index("min_df")
display(grid)
print("rappel : le vecteur indexe des unigrammes **et** des bigrammes — "
      "« termes connus par ticket » se lit en variation, pas comme un nombre de mots.")
""",
            context,
        ),
        _insight(
            [
                "Le vocabulaire fond quand `min_df` monte : les tournures rares disparaissent du "
                "vecteur, et la couverture des tokens du test baisse avec elles.",
                "Un ticket **muet** (aucun terme connu) est le cas extrême : il est parfaitement "
                "lisible pour un humain et totalement invisible pour un modèle lexical.",
                "La valeur de `min_df` retenue par le projet est celle du manifeste : elle est "
                "visible, comparable et rediscutable sur ces deux colonnes.",
            ]
        ),
        _md(
            "## 5. Le test de raccourci, trait par trait\n\n"
            "Même instrument que le notebook 01, mais vu par trait : quel trait unique suffirait à "
            "router les tickets ? La réponse doit rester « aucun, tant s'en faut »."
        ),
        _text(
            """
from src.features.build_features import (
    TEXT_FEATURES,
    group_gain,
    shortcut_scores,
    stump_gain,
)

candidates = [*TEXT_FEATURES, "source", "priority"]
scores = shortcut_scores(DOCUMENTS)

def gain(name: str) -> float:
    \"\"\"Gain de la meilleure règle à un seuil sur un trait (0,5 : sépare deux classes d'un coup).\"\"\"
    if name in TEXT_FEATURES:
        return stump_gain(enriched, name, label_column=TARGET_COLUMN)
    return group_gain(DOCUMENTS, name, label_column=TARGET_COLUMN)

gains = pd.DataFrame(
    {"trait": candidates, "gain": [gain(name) for name in candidates]}
).set_index("trait").sort_values("gain", ascending=False)
gains["marge_vs_plancher"] = (gains["gain"] - scores["majority_baseline"]).round(4)
display(gains.round(4))

best = gains.index[0]
print(f"meilleur trait : {best} (gain {gains.loc[best, 'gain']:.4f}) — "
      f"plancher : {scores['majority_baseline']:.4f}")
""",
            context,
        ),
        _insight(
            [
                "Un gain de 0,5 signifie qu'un seul seuil sur ce trait sépare parfaitement deux "
                "classes : aucun trait du corpus n'en approche.",
                "Les traits de forme restent sous le plancher majoritaire + 0,05 : la forme ne route "
                "pas les tickets.",
                "Le jour où un trait dépasserait ce niveau, le rapport le signalerait comme "
                "raccourci et une revue de modèle serait obligatoire avant toute mise en "
                "production.",
            ]
        ),
    ]
    return _write(destination, "03_preprocessing", cells)


# ---------------------------------------------------------------------------------------------
# 04 — exploration du modèle
# ---------------------------------------------------------------------------------------------
def build_04_model_exploration(context: NotebookContext, destination: Path) -> Path:
    """Build ``04_model_exploration.ipynb``: comparaison des algorithmes et seuil d'automatisation.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    cells: list[NotebookNode] = [
        *_header(
            context,
            "04",
            "Choix du modèle — trois algorithmes, un seuil d'automatisation",
            [
                "Comparer les algorithmes de la stack sur le split de validation, à corpus égal.",
                "Chiffrer l'effet du rééquilibrage sur la classe minoritaire.",
                "Lire la courbe de calibration : ce que « confiance 0,9 » annonce réellement.",
                "Arbitrer un seuil d'automatisation sur la calibration, puis le lire sur le test.",
            ],
        ),
        _md("## 1. Mise en place"),
        _text(SETUP, context),
        _text(LOAD_CORPUS, context),
        _md(
            "## 2. Les algorithmes de la stack\n\n"
            "La stack en déclare trois : régression logistique sur TF-IDF, bayésien naïf "
            "complémentaire, centroïde de classe. Un seul est **servi** (celui du manifeste) ; les "
            "autres servent de témoins, et c'est leur écart qui donne un sens au choix."
        ),
        _text(
            """
from src.models import ALGORITHMS, available_algorithms, describe_algorithm

algorithms = available_algorithms(CONFIG.metrics.task)
print("algorithmes disponibles :", algorithms)
documentation = pd.DataFrame([describe_algorithm(name) for name in algorithms])
display(documentation.set_index("name")[["display_name", "rationale", "defaults"]])
print("algorithmes déclarés par le registre de la stack :", sorted(ALGORITHMS))
""",
            context,
        ),
        _md(
            "## 3. Comparaison sur le split de validation\n\n"
            "Une seule règle : on **choisit** sur `val`, on ne juge jamais sur `test`. Les modèles "
            "sont montés avec les **défauts du registre** (`params={}`), sinon chaque algorithme "
            "recevrait les hyper-paramètres d'un autre — la comparaison doit être loyale. Le "
            "tableau mélange qualité (F1 macro, exactitude, kappa), calibration (ECE), coût "
            "(latence par ticket) et une lecture métier : le rappel de la classe minoritaire."
        ),
        _text(
            """
import time

from src.models import build_model
from src.training.metrics import classification_metrics, per_class_frame

validation_truth = VALIDATION[TARGET_COLUMN].astype(str)
rows = []
fitted = {}
for name in algorithms:
    candidate = build_model(CONFIG, algorithm=name, params={})
    started = time.perf_counter()
    candidate.fit(TRAIN, target=TARGET_COLUMN)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    probabilities = candidate.predict_proba(VALIDATION[TEXT_COLUMN].astype(str).tolist())
    inference_seconds = time.perf_counter() - started
    predicted = np.asarray(candidate.labels)[probabilities.argmax(axis=1)]
    scores = classification_metrics(
        validation_truth, predicted, confidences=probabilities.max(axis=1),
        labels=list(candidate.labels),
    )
    per_class = per_class_frame(
        validation_truth, pd.Series(predicted), labels=list(candidate.labels)
    ).set_index("class")
    fitted[name] = candidate
    rows.append(
        {
            "algorithme": name,
            "F1 macro": round(scores["macro_f1"], 4),
            "exactitude": round(scores["accuracy"], 4),
            "kappa": round(scores["cohen_kappa"], 4),
            "ECE": round(scores["expected_calibration_error"], 4),
            f"rappel {MINORITY_LABEL}": round(float(per_class.loc[MINORITY_LABEL, "recall"]), 4),
            "fit (s)": round(fit_seconds, 3),
            "ms / ticket": round(1000.0 * inference_seconds / max(1, len(VALIDATION)), 3),
        }
    )
comparison = pd.DataFrame(rows).set_index("algorithme")
display(comparison)

reference = str(CONFIG.model.algorithm)
print(f"algorithme servi par le manifeste : {reference}")
for name in comparison.index:
    delta = comparison.loc[name, "F1 macro"] - comparison.loc[reference, "F1 macro"]
    print(f"  {name:16s} F1 macro {comparison.loc[name, 'F1 macro']:.4f} ({delta:+.4f} "
          f"contre le modèle servi)")
""",
            context,
        ),
        _insight(
            [
                "Les trois algorithmes vivent derrière le même contrat : le pipeline ne change pas "
                "quand l'algorithme change — c'est ce qui rend la comparaison possible.",
                "Le bayésien naïf suppose les termes indépendants et paie cette hypothèse sur les "
                "classes qui se ressemblent ; le centroïde n'apprend aucun poids et sert de "
                "plancher.",
                "Le temps d'ajustement se lit en secondes et la latence en millisecondes par "
                "ticket : à ces échelles, le choix se fait sur la qualité, pas sur la vitesse.",
            ]
        ),
        _md(
            "## 4. Rééquilibrage : ce que la classe minoritaire coûte\n\n"
            "Le manifeste utilise `class_weight='balanced'`. On mesure ce que ce choix apporte — et "
            "ce qu'il coûte — en comparant les deux réglages sur le même split, à algorithme "
            "égal."
        ),
        _text(
            """
from src.models import build_model
from src.training.metrics import classification_metrics, per_class_frame

weighting_rows = {}
for label, class_weight in (("balanced", "balanced"), ("sans rééquilibrage", None)):
    candidate = build_model(
        CONFIG, algorithm=reference, params={"estimator": {"class_weight": class_weight}}
    )
    candidate.fit(TRAIN, target=TARGET_COLUMN)
    probabilities = candidate.predict_proba(VALIDATION[TEXT_COLUMN].astype(str).tolist())
    predicted = np.asarray(candidate.labels)[probabilities.argmax(axis=1)]
    scores = classification_metrics(
        validation_truth, predicted, confidences=probabilities.max(axis=1),
        labels=list(candidate.labels),
    )
    per_class = per_class_frame(
        validation_truth, pd.Series(predicted), labels=list(candidate.labels)
    ).set_index("class")
    weighting_rows[label] = {
        "F1 macro": round(scores["macro_f1"], 4),
        "exactitude": round(scores["accuracy"], 4),
        f"rappel {MINORITY_LABEL}": round(float(per_class.loc[MINORITY_LABEL, "recall"]), 4),
        f"rappel {MAJORITY_LABEL}": round(float(per_class.loc[MAJORITY_LABEL, "recall"]), 4),
    }
weighting = pd.DataFrame(weighting_rows).T
display(weighting)
delta = (
    weighting.loc["balanced", f"rappel {MINORITY_LABEL}"]
    - weighting.loc["sans rééquilibrage", f"rappel {MINORITY_LABEL}"]
)
print(f"écart de rappel sur la classe {MINORITY_LABEL} : {delta:+.4f}")
""",
            context,
        ),
        _insight(
            [
                "Le rééquilibrage déplace du rappel de la classe majoritaire vers la classe rare : "
                "c'est exactement ce que demande une métrique macro.",
                "Sur ce corpus, l'écart est mesuré et **publié** ; sur un corpus où la classe rare "
                "serait plus fréquente, le réglage deviendrait moins utile — la question se "
                "repose à chaque réentraînement.",
                "Le même réglage se lit sur les probabilités : un modèle rééquilibré annonce des "
                "confiances plus basses, ce qui change le seuil d'automatisation de la section 6.",
            ]
        ),
        _md(
            "## 5. Calibration : ce qu'une confiance annonce\n\n"
            "Un modèle peut être exact et mal calibré. C'est exactement ce qui décide si l'on peut "
            "automatiser un routage : une confiance de 0,9 qui ne vaut que 0,6 envoie les mauvais "
            "tickets en production. La courbe est tracée sur la **calibration**, jamais sur le "
            "test."
        ),
        _text(
            """
from src.visualization.plots import plot_calibration

served = fitted[reference]
calibration_scores = served.predict_proba(CALIBRATION[TEXT_COLUMN].astype(str).tolist())
calibration_prediction = np.asarray(served.labels)[calibration_scores.argmax(axis=1)]
calibration_frame = pd.DataFrame(
    {
        "confidence": calibration_scores.max(axis=1),
        "correct": (
            calibration_prediction == CALIBRATION[TARGET_COLUMN].astype(str).to_numpy()
        ).astype(int),
    }
)
CALIBRATION_FIGURE = plot_calibration(
    calibration_frame, NB_PATHS.figures_dir / "04_calibration.png"
)
print(f"figure écrite : {CALIBRATION_FIGURE.relative_to(PROJECT_ROOT)}")
print("confiance moyenne :", round(float(calibration_frame["confidence"].mean()), 4),
      "| exactitude observée :", round(float(calibration_frame["correct"].mean()), 4))
""",
            context,
        ),
        _insight(
            [
                "Si la confiance moyenne dépasse l'exactitude observée, le modèle est "
                "**sur-confiant** : il faut alors des seuils plus hauts pour obtenir la même "
                "garantie — et l'inverse est une bonne nouvelle.",
                "La courbe est tracée sur le split de calibration, le seuil y sera arbitré : le "
                "test ne sert à aucun réglage.",
                "La calibration se dégrade quand le corpus est bruité ou quand une classe est rare : "
                "deux propriétés que la ventilation par segment du notebook 06 retrouvera.",
            ]
        ),
        _md(
            "## 6. Arbitrer un seuil d'automatisation\n\n"
            "Règle de service : un ticket est routé automatiquement si la confiance du modèle "
            "atteint le seuil ; en dessous, il part en relecture humaine. On cherche le plus petit "
            "seuil qui garantit l'exactitude cible sur la calibration, puis on lit le résultat sur "
            "le test — une seule fois."
        ),
        _text(
            """
thresholds = __THRESHOLDS__
target_accuracy = __TARGET_AUTOMATION__
rows = []
for threshold in thresholds:
    automatic = calibration_frame[calibration_frame["confidence"] >= threshold]
    accuracy = float(automatic["correct"].mean()) if len(automatic) else float("nan")
    rows.append(
        {
            "seuil": threshold,
            "tickets routés": len(automatic),
            "couverture": round(len(automatic) / len(calibration_frame), 4),
            "exactitude des routés": round(accuracy, 4) if len(automatic) else np.nan,
            "tickets en relecture": int(len(calibration_frame) - len(automatic)),
        }
    )
sweep = pd.DataFrame(rows).set_index("seuil")
display(sweep)

# Deux contraintes, pas une : l'exactitude cible **et** une couverture minimale. Un seuil qui
# n'automatise rien est parfait et inutile : on exige de router au moins un ticket sur cinq.
MIN_COVERAGE = 0.2
eligible = sweep[(sweep["exactitude des routés"] >= target_accuracy)
                 & (sweep["couverture"] >= MIN_COVERAGE)]
if len(eligible):
    CHOSEN_THRESHOLD = float(eligible.index.min())
    reason = "plus petit seuil qui satisfait les deux contraintes"
else:
    candidates = sweep[sweep["couverture"] >= MIN_COVERAGE]
    CHOSEN_THRESHOLD = float(candidates["exactitude des routés"].idxmax())
    reason = "aucun seuil n'atteint la cible : meilleur seuil qui route encore des tickets"
print(f"règle de service : exactitude cible {target_accuracy:.2f}, couverture minimale "
      f"{MIN_COVERAGE:.0%}")
print(f"seuil retenu sur la calibration : {CHOSEN_THRESHOLD:.2f} ({reason})")
""",
            context,
        ),
        _text(
            """
calibration_selection = calibration_frame["confidence"] >= CHOSEN_THRESHOLD
print(f"calibration — routage automatique : {int(calibration_selection.sum())}"
      f"/{len(calibration_selection)} tickets "
      f"({float(calibration_selection.mean()):.1%})")

test_probabilities = served.predict_proba(TEST[TEXT_COLUMN].astype(str).tolist())
test_predicted = np.asarray(served.labels)[test_probabilities.argmax(axis=1)]
test_confidence = test_probabilities.max(axis=1)
test_truth = TEST[TARGET_COLUMN].astype(str).to_numpy()

mask = test_confidence >= CHOSEN_THRESHOLD
automatic_accuracy = float((test_predicted[mask] == test_truth[mask]).mean()) if mask.any() else float("nan")
global_accuracy = float((test_predicted == test_truth).mean())
print(f"test — routage automatique : {int(mask.sum())}/{len(mask)} tickets "
      f"({float(mask.mean()):.1%}), exactitude {automatic_accuracy:.4f}")
print(f"test — routage complet     : exactitude {global_accuracy:.4f}")
print(f"test — relecture humaine   : {int((~mask).sum())} tickets, dont "
      f"{int((test_predicted[~mask] != test_truth[~mask]).sum())} erreurs interceptées")
""",
            context,
        ),
        _insight(
            [
                "Le seuil est un arbitrage **métier** : automatiser une partie des tickets à très "
                "haute exactitude vaut mieux que d'automatiser tout le stock à l'exactitude "
                "moyenne — mais un seuil qui ne route rien est un aveu d'échec, d'où la couverture "
                "minimale.",
                "Le seuil est choisi sur la calibration et lu sur le test : c'est la seule manière "
                "d'annoncer une couverture sans se mentir.",
                "Ce réglage ne remplace pas la métrique contractuelle : la F1 macro reste mesurée "
                "sur **tous** les tickets, seuil ou pas.",
            ]
        ),
    ]
    return _write(destination, "04_model_exploration", cells)


# ---------------------------------------------------------------------------------------------
# 05 — entraînement et artefacts
# ---------------------------------------------------------------------------------------------
def build_05_training(context: NotebookContext, destination: Path) -> Path:
    """Build ``05_training.ipynb``: pipeline d'entraînement, artefacts et déterminisme.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    cells: list[NotebookNode] = [
        *_header(
            context,
            "05",
            "Entraînement — le pipeline réel, ses artefacts et sa reproductibilité",
            [
                "Exécuter le pipeline d'entraînement dans un bac à sable qui n'écrase rien.",
                "Lire les cinq artefacts produits : modèle, prétraitement, métriques, fiche, "
                "configuration résolue.",
                "Recharger le modèle depuis son artefact et vérifier qu'il prédit **exactement** "
                "comme un modèle ajusté sur les mêmes données.",
                "Prouver le déterminisme : deux ajustements à graine fixée donnent le même chiffre.",
            ],
        ),
        _md("## 1. Mise en place"),
        _text(SETUP, context),
        _text(LOAD_CORPUS, context),
        _md(
            "## 2. Le pipeline d'entraînement, dans un bac à sable\n\n"
            "Le notebook appelle **le pipeline du projet**, pas une réimplémentation : ce qu'on lit "
            "ici est exactement ce que fait `python -m src.main mode=train`. Les artefacts partent "
            "dans `outputs/notebooks/artifacts`, donc ceux de `make train` ne sont jamais écrasés."
        ),
        _text(
            """
from src.pipelines import DataGenerationPipeline, TrainPipeline

data_result = DataGenerationPipeline(CONFIG, paths=NB_PATHS).run()
print("generate-data :", "succès" if data_result.succeeded else "échec",
      f"en {data_result.duration_seconds:.2f}s")
for message in data_result.messages:
    print("  -", message)

train_result = TrainPipeline(CONFIG, paths=NB_PATHS).run()
print()
print("train :", "succès" if train_result.succeeded else "échec",
      f"en {train_result.duration_seconds:.2f}s")
for message in train_result.messages:
    print("  -", message)
assert train_result.succeeded, train_result.messages
outcome = train_result.payload
""",
            context,
        ),
        _text(
            """
artifacts = {
    "modèle": NB_PATHS.models_dir / "__MODEL_FILE__",
    "prétraitement": NB_PATHS.models_dir / "preprocessing.joblib",
    "métriques": NB_PATHS.metrics_dir / "training_metrics.json",
    "fiche de modèle": NB_PATHS.models_dir / "model_card.json",
    "configuration résolue": NB_PATHS.models_dir / "resolved_config.json",
}
for name, path in artifacts.items():
    size = path.stat().st_size if path.exists() else 0
    print(f"{name:24s} {'écrit' if path.exists() else 'ABSENT':6s} "
          f"{path.relative_to(PROJECT_ROOT)} ({size} octets)")
assert all(path.exists() for path in artifacts.values())
""",
            context,
        ),
        _insight(
            [
                "Cinq artefacts, cinq questions : le modèle (comment prédire), le prétraitement "
                "(comment préparer le texte), les métriques (ce que ça vaut), la fiche (comment "
                "c'est fait) et la configuration résolue (avec quels réglages exactement).",
                "La configuration résolue est le seul moyen de rejouer un run : la configuration "
                "Hydra est faite de couches, et une couche qui change change tout.",
                "`DataGenerationPipeline` a d'abord persisté le corpus : l'entraînement ne peut pas "
                "lire un `data/raw` fantôme.",
            ]
        ),
        _md(
            "## 3. Lire les métriques et la fiche du run\n\n"
            "Les métriques d'entraînement distinguent ce que le modèle a **vu** (`train_*`) de ce "
            "qu'il n'a pas vu (`val_*`). L'écart entre les deux est la première mesure de "
            "généralisation disponible."
        ),
        _text(
            """
import json

training = json.loads((NB_PATHS.metrics_dir / "training_metrics.json").read_text(encoding="utf-8"))
display(pd.Series(training["metrics"]).to_frame("valeur").round(4))

gap = float(training["metrics"]["train_accuracy"] - training["metrics"]["val_accuracy"])
print(f"tickets d'entraînement : {training['n_train_documents']} | "
      f"tickets de validation : {training['n_val_documents']}")
print(f"écart exactitude train - validation : {gap:+.4f}")
print(f"métrique principale du run : val_{CONFIG.metrics.primary}="
      f"{float(training['metrics'][f'val_{CONFIG.metrics.primary}']):.4f} "
      f"(seuil contractuel : {CONFIG.metrics.min_primary})")
fit = outcome.fit_result
print(f"ajustement : {fit.n_samples} tickets, {fit.n_features} termes, "
      f"{fit.duration_seconds:.3f}s")
print("hyper-paramètres effectifs (registre + manifeste) :",
      {"lexical": fit.params.get("lexical"), "estimator": fit.params.get("estimator")})
""",
            context,
        ),
        _text(
            """
card = json.loads((NB_PATHS.models_dir / "model_card.json").read_text(encoding="utf-8"))
display(pd.Series({key: str(value) for key, value in card.items()}).to_frame("fiche de modèle"))
""",
            context,
        ),
        _insight(
            [
                "Un écart train / validation nul n'est pas une bonne nouvelle : c'est le signe d'un "
                "problème trop simple ou d'une fuite. Ici l'écart est faible parce que la tâche est "
                "bien posée, et il est **publié**.",
                "La fiche de modèle contient les paramètres effectifs, la liste des classes, la "
                "taille du vocabulaire et les notes de limitation : c'est elle qui rend le run "
                "rejouable par quelqu'un d'autre.",
                "Les métriques sont écrites en JSON : un portail de CI peut les comparer d'un run à "
                "l'autre sans lire un notebook.",
            ]
        ),
        _md(
            "## 4. Recharger le modèle et vérifier l'identité\n\n"
            "Un artefact qu'on ne recharge pas est un artefact dont on ne sait rien. On vérifie que "
            "le modèle **rechargé** produit les mêmes probabilités qu'un modèle ajusté dans la "
            "session — c'est le test qui garantit que l'inférence ne dépend pas du hasard d'un "
            "processus."
        ),
        _text(
            """
from src.models import build_model, load_model

model_path = NB_PATHS.models_dir / "__MODEL_FILE__"
reloaded = load_model(model_path, config=node(CONFIG, "model"))
print("modèle rechargé :", reloaded.summary())

memory_model = build_model(CONFIG)
memory_model.fit(TRAIN, target=TARGET_COLUMN)

texts = TEST[TEXT_COLUMN].astype(str).tolist()
same = np.allclose(memory_model.predict_proba(texts), reloaded.predict_proba(texts), atol=1e-9)
print("probabilités identiques entre modèle ajusté et artefact rechargé :", same)
assert same, "l'artefact rechargé ne reproduit pas le modèle ajusté"

display(
    pd.DataFrame(
        {
            "texte": [text[:64] + "…" for text in texts[:3]],
            "prédit (mémoire)": memory_model.predict(texts[:3]),
            "prédit (artefact)": reloaded.predict(texts[:3]),
        }
    )
)
""",
            context,
        ),
        _md(
            "## 5. Déterminisme : deux runs, le même chiffre\n\n"
            "Le dépôt impose des graines partout. La preuve la plus simple : refaire le run et "
            "comparer la métrique principale **exactement** — puis la comparer à celle qu'a "
            "enregistrée le pipeline."
        ),
        _text(
            """
from src.training.metrics import classification_metrics

validation_truth = VALIDATION[TARGET_COLUMN].astype(str)
second_model = build_model(CONFIG)
second_model.fit(TRAIN, target=TARGET_COLUMN)

first_scores = memory_model.predict_proba(VALIDATION[TEXT_COLUMN].astype(str).tolist())
second_scores = second_model.predict_proba(VALIDATION[TEXT_COLUMN].astype(str).tolist())
first_metrics = classification_metrics(
    validation_truth,
    np.asarray(memory_model.labels)[first_scores.argmax(axis=1)],
    labels=list(memory_model.labels),
)
second_metrics = classification_metrics(
    validation_truth,
    np.asarray(second_model.labels)[second_scores.argmax(axis=1)],
    labels=list(second_model.labels),
)
pipeline_metric = float(training["metrics"][f"val_{CONFIG.metrics.primary}"])
print(f"premier ajustement  : {CONFIG.metrics.primary}={first_metrics[CONFIG.metrics.primary]:.10f}")
print(f"second ajustement   : {CONFIG.metrics.primary}={second_metrics[CONFIG.metrics.primary]:.10f}")
print(f"pipeline enregistré : {CONFIG.metrics.primary}={pipeline_metric:.10f}")
assert first_metrics[CONFIG.metrics.primary] == second_metrics[CONFIG.metrics.primary]
assert abs(first_metrics[CONFIG.metrics.primary] - pipeline_metric) < 1e-12
print("déterminisme vérifié : même graine, même corpus, même chiffre")
""",
            context,
        ),
        _insight(
            [
                "Le déterminisme n'est pas un luxe pédagogique : c'est ce qui permet de comparer "
                "deux variantes de stack sur un écart de F1 et non sur du bruit d'exécution.",
                "Le modèle rechargé sert dans le pipeline d'évaluation et dans l'inférence : les "
                "trois chemins partagent le même objet, et donc les mêmes probabilités.",
                "Aucun accès réseau n'est nécessaire à aucune de ces étapes : tout est appris sur "
                "le corpus synthétique, et le vocabulaire est appris sur le train.",
            ]
        ),
    ]
    return _write(destination, "05_training", cells)


# ---------------------------------------------------------------------------------------------
# 06 — analyse d'erreurs
# ---------------------------------------------------------------------------------------------
def build_06_error_analysis(context: NotebookContext, destination: Path) -> Path:
    """Build ``06_error_analysis.ipynb``: verdict, ventilation et erreurs expliquées.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    cells: list[NotebookNode] = [
        *_header(
            context,
            "06",
            "Analyse d'erreurs — le verdict, la ventilation et les erreurs expliquées",
            [
                "Lire le verdict contractuel et le situer face aux références triviales.",
                "Ventiler les métriques par style : chiffrer ce que les paraphrases coûtent.",
                "Identifier la classe sacrifiée et lire la matrice de confusion.",
                "Expliquer les erreurs les plus confiantes terme à terme, et conclure par des "
                "recommandations chiffrées.",
            ],
        ),
        _md("## 1. Mise en place"),
        _text(SETUP, context),
        _text(LOAD_CORPUS, context),
        _md(
            "## 2. Le pipeline d'évaluation\n\n"
            "Il recharge les artefacts produits par l'entraînement et mesure le split de **test**, "
            "une seule fois. Si le notebook est ouvert sur un projet fraîchement cloné, il entraîne "
            "d'abord : un notebook doit rester exécutable de bout en bout."
        ),
        _text(
            """
from src.pipelines import EvaluationPipeline, TrainPipeline

model_file = NB_PATHS.models_dir / "__MODEL_FILE__"
if not model_file.exists():
    print("artefact absent : entraînement sur le corpus réduit…")
    TrainPipeline(CONFIG, paths=NB_PATHS).run()

evaluation_result = EvaluationPipeline(CONFIG, paths=NB_PATHS).run()
assert evaluation_result.succeeded, evaluation_result.messages
evaluated = evaluation_result.payload
for message in evaluation_result.messages:
    print("  -", message)
""",
            context,
        ),
        _md(
            "## 3. Verdict et niveau à battre\n\n"
            "La métrique principale est comparée à son seuil contractuel **et** aux références "
            "triviales. Un verdict « conforme » qui ne bat pas la classe majoritaire ne vaudrait "
            "rien ; c'est pourquoi les deux lectures sont côte à côte."
        ),
        _text(
            """
primary = CONFIG.metrics.primary
print(f"verdict : {evaluated.verdict}")
print(f"{primary} = {evaluated.metrics[primary]:.4f} "
      f"(seuil {evaluated.threshold}, {evaluated.n_documents} tickets de test)")

baselines = pd.DataFrame(evaluated.baselines).T
baselines["marge du modèle"] = (evaluated.metrics[primary] - baselines[primary]).round(4)
display(baselines.round(4))

headline = pd.DataFrame(
    {
        "indicateur": [
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "balanced_accuracy",
            "cohen_kappa",
            "mean_confidence",
            "expected_calibration_error",
        ]
    }
).set_index("indicateur")
headline["valeur"] = [round(float(evaluated.metrics[name]), 4) for name in headline.index]
display(headline)
""",
            context,
        ),
        _insight(
            [
                "Le verdict se lit sur la F1 macro : sur six classes dont une minoritaire, c'est la "
                "seule moyenne qui ne récompense pas l'abandon de la classe difficile.",
                "La marge contre la classe majoritaire et contre le tirage stratifié est l'argument "
                "du projet : tout le reste n'est que de l'habillage.",
                "La confiance moyenne se lit **contre** l'exactitude : si elle est plus basse, le "
                "modèle se sous-estime, ce qui est préférable à l'inverse pour un seuil "
                "d'automatisation.",
            ]
        ),
        _md(
            "## 4. Ventilation par style : ce que les paraphrases coûtent\n\n"
            "C'est la lecture qui distingue « lire un mot » de « comprendre une demande ». Si "
            "l'écart entre le segment canonique et le segment paraphrase est nul, soit la tâche est "
            "trop facile, soit le corpus a été écrit avec les mêmes mots partout."
        ),
        _text(
            """
segments = evaluated.segments.copy()
display(segments.round(4))

style_rows = segments[segments["segment"].str.startswith("style=")].set_index("segment")
display(style_rows[["n_documents", primary]].round(4))
if {"style=canonique", "style=paraphrase"} <= set(style_rows.index):
    canon = float(style_rows.loc["style=canonique", primary])
    para = float(style_rows.loc["style=paraphrase", primary])
    print(f"écart canonique - paraphrase sur {primary} : {canon - para:+.4f} "
          f"({canon:.4f} contre {para:.4f})")
    print("taille des segments :", style_rows["n_documents"].to_dict())
else:
    print("segments de style absents du corpus courant")

# Les chiffres du corpus **complet**, extraits du manifeste : le corpus réduit des notebooks ne
# fait que quelques dizaines de tickets de test, et un segment trop petit peut inverser un écart.
REFERENCE_RESULTS = __MEASURED_RESULTS__
display(
    pd.DataFrame(REFERENCE_RESULTS, columns=["mesure de référence (corpus complet)", "valeur"])
    .set_index("mesure de référence (corpus complet)")
)
""",
            context,
        ),
        _text(
            """
figure, axis = plt.subplots(figsize=(7, 3.5))
style_rows[primary].plot.barh(ax=axis, color="#4c72b0")
axis.axvline(float(evaluated.metrics[primary]), color="#c44e52", linestyle="--",
             label=f"moyenne ({evaluated.metrics[primary]:.3f})")
axis.set_xlabel(primary)
axis.set_title("Par style rédactionnel — la moyenne cache l'écart")
axis.legend()
figure.tight_layout()
plt.show()
""",
            context,
        ),
        _insight(
            [
                "Sur le corpus complet, le segment canonique est nettement au-dessus du segment "
                "paraphrase : le modèle lit un vocabulaire et paie l'absence de ces mots — la "
                "limite **assumée** de l'approche lexicale, et ce qu'une variante `transformers` "
                "doit venir chercher.",
                "Un écart de signe **inverse** n'est pas une bonne nouvelle : sur un petit segment, "
                "quelques tickets suffisent à retourner une moyenne. On lit donc toujours l'écart "
                "avec la taille des segments, jamais seul.",
                "Le segment bruité se situe entre les deux : la politesse et la signature ne portent "
                "pas d'information, mais elles diluent le signal — sa F1 se lit avec la longueur du "
                "segment.",
            ]
        ),
        _md(
            "## 5. La classe sacrifiée\n\n"
            "Une F1 macro est une moyenne : elle cache nécessairement une classe qui souffre. On la "
            "cherche dans le tableau par classe, puis dans la matrice de confusion."
        ),
        _text(
            """
per_class = evaluated.per_class.sort_values("f1").reset_index(drop=True)
display(per_class.round(4))

sacrificed = str(per_class.loc[0, "class"])
print(f"classe la plus fragile : {sacrificed} (F1 {per_class.loc[0, 'f1']:.4f}, "
      f"{int(per_class.loc[0, 'support'])} tickets)")
print("vers quoi la classe fragile se trompe-t-elle ?")
display(evaluated.confusion.loc[sacrificed].sort_values(ascending=False).to_frame("prédictions"))
""",
            context,
        ),
        _text(
            """
from src.visualization.plots import plot_confusion, plot_per_class

print("figure de confusion :",
      plot_confusion(evaluated.confusion, NB_PATHS.figures_dir / "06_confusion.png").name)
print("figure par classe   :",
      plot_per_class(evaluated.per_class, NB_PATHS.figures_dir / "06_f1_per_class.png").name)
""",
            context,
        ),
        _insight(
            [
                "La classe la plus fragile est presque toujours la plus rare **et** la plus ambiguë "
                "pour l'approche retenue : c'est elle qui décide de la F1 macro.",
                "La matrice de confusion nomme la paire de classes à travailler : sans elle, on sait "
                "qu'on se trompe, pas où.",
                "Les deux figures sont produites par les mêmes fonctions que le rapport : le "
                "notebook ne fabrique pas une visualisation parallèle.",
            ]
        ),
        _md(
            "## 6. Les erreurs les plus confiantes, expliquées\n\n"
            "Les erreurs les plus confiantes sont celles qui coûtent le plus cher en production : "
            "un ticket routé avec une forte confiance vers le mauvais service repart pour un tour. "
            "Elles sont archivées par le pipeline, et le modèle linéaire sait dire **quels termes** "
            "l'ont emporté."
        ),
        _text(
            """
from src.models import load_model

reloaded = load_model(NB_PATHS.models_dir / "__MODEL_FILE__", config=node(CONFIG, "model"))
errors = evaluated.top_errors
text_by_id = dict(zip(evaluated.predictions["doc_id"], evaluated.predictions["text"], strict=True))

for position, row in enumerate(errors.head(5).itertuples(), start=1):
    text = text_by_id.get(str(row.doc_id), "")
    print(f"--- erreur {position} : attendu « {row.expected_label} », "
          f"prédit « {row.prediction} » (confiance {row.confidence:.3f})")
    print(f"    {text[:200]}")
    terms = reloaded.explain([text], k=4)[0]
    print("    termes décisifs :",
          ", ".join(f"{term} ({weight:+.2f})" for term, weight in terms))
    print()

print(f"erreurs archivées : {len(errors)} (les plus confiantes du split de test)")
""",
            context,
        ),
        _insight(
            [
                "Savoir expliquer une erreur est ce qui permet de la corriger : ajouter une tournure "
                "au corpus, revoir une règle de service, ou accepter la limite et envoyer le ticket "
                "en relecture.",
                "Un modèle linéaire s'explique par ses termes : c'est un argument de déploiement "
                "dans une équipe support, pas un détail technique.",
                "Les erreurs confiantes sont minoritaires : le modèle se trompe surtout là où il "
                "hésite — ce qui est précisément ce que le seuil d'automatisation exploite.",
            ]
        ),
        _md(
            "## 7. Le rapport d'évaluation\n\n"
            "Le même pipeline écrit un rapport Markdown et ses tables CSV : verdict, métriques, "
            "références, ventilation par classe et par segment, erreurs, figures, lecture honnête "
            "et reproductibilité. On en relit ici la section « Lecture honnête »."
        ),
        _text(
            """
report_path = NB_PATHS.reports_dir / "__REPORT_NAME__"
report_text = report_path.read_text(encoding="utf-8")
start = report_text.find("## 9.")
print(report_text[start:start + 1100] if start >= 0 else report_text[-1100:])
""",
            context,
        ),
        _md(
            "## 8. Recommandations\n\n"
            "Ce que ce notebook permet d'écrire noir sur blanc pour la suite du projet."
        ),
        _text(
            """
recommendations = [
    f"Router automatiquement au-delà du seuil arbitré au notebook 04 : {CONFIG.metrics.primary} "
    "reste la métrique contractuelle, mesurée sur l'ensemble du test.",
    f"Surveiller le rappel de la classe « {MINORITY_LABEL} » à chaque réentraînement : c'est la "
    "première classe qui souffre quand la distribution des tickets glisse.",
    "Rejouer la ventilation par style après chaque changement de corpus : un écart "
    "canonique / paraphrase qui se réduit est le signe d'un corpus qui s'appauvrit.",
    "Comparer avec la variante `transformers` sur le même corpus : l'écart de F1 sur le segment "
    "paraphrase est exactement ce qu'un modèle contextualisé apporte.",
    "Relire les erreurs les plus confiantes avant toute mise en production : c'est là que se "
    "cachent les tournures ambiguës, pas dans les erreurs de faible confiance.",
]
for position, line in enumerate(recommendations, start=1):
    print(f"{position}. {line}")
""",
            context,
        ),
        _insight(
            [
                "Le notebook et le rapport racontent la même histoire : verdict, ventilation, "
                "erreurs, recommandations — avec des chiffres, jamais des adjectifs.",
                "La conclusion honnête de ce projet est un couple : « excellent sur les tickets qui "
                "emploient le vocabulaire de leur classe, mesuré sur les paraphrases ».",
                "Toute la suite du dépôt (variante `transformers`, fusion des deux stacks) se juge "
                "sur le même protocole : corpus identique, splits identiques, test lu une seule "
                "fois.",
            ]
        ),
    ]
    return _write(destination, "06_error_analysis", cells)


def _write(destination: Path, stem: str, cells: list[NotebookNode]) -> Path:
    """Write one notebook of the family.

    Args:
        destination: ``notebooks/`` directory.
        stem: File name without extension.
        cells: Ordered cells.

    Returns:
        The written path.
    """
    return write_notebook(destination / f"{stem}.ipynb", cells)


#: Builders of the six notebooks, in reading order.
BUILDERS = (
    build_01_eda,
    build_02_validation,
    build_03_preprocessing,
    build_04_model_exploration,
    build_05_training,
    build_06_error_analysis,
)


def build_all(context: NotebookContext, destination: Path) -> list[Path]:
    """Build the six notebooks of a text classification project.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory of the project.

    Returns:
        The written notebook paths, in order.
    """
    return [builder(context, destination) for builder in BUILDERS]
