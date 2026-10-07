"""Notebooks pédagogiques des projets de **résumé automatique** (``nlp/summarization``).

Un projet de résumé ne s'explore pas comme une classification : il n'y a pas de classes à équilibrer
mais un **corpus**, des **résumés de référence** à lire ligne à ligne, et une question que le ROUGE
ne pose pas — *ce qui a été écrit est-il vrai ?* Les six notebooks suivent la progression commune,
avec les questions propres à la génération de texte :

* ``01_eda.ipynb`` — cartographier les comptes-rendus **et** leurs résumés : longueurs, compression
  réellement demandée, faits annotés par type, part saillante, distracteurs déclarés. La dernière
  cellule mesure ce que la référence couvre d'elle-même : le plafond de la tâche, pas une opinion ;
* ``02_validation.ipynb`` — les contrats Pandera des **trois** tables (documents, résumés de
  référence, faits) puis leurs liens **croisés** — un résumé orphelin, un document sans référence, un
  fait hors des phrases de son document — et la table de prédictions publiée. Chaque règle est cassée
  volontairement pour lire le message d'échec ;
* ``03_preprocessing.ipynb`` — découpage en phrases, tokénisation, et **alignement des faits** sur le
  texte (surface au caractère, index de phrase) ; puis la mesure du budget de longueur et de ce
  qu'une recopie de phrase retrouve déjà, c'est-à-dire la part de réécriture que le corpus impose ;
* ``04_model_exploration.ipynb`` — les trois stratégies de la stack (``lead``, ``textrank``,
  ``transformer_tiny``) mesurées sur **les mêmes lignes** : planchers triviaux, baselines
  extractives, modèle appris court, ROUGE et couverture des faits côte à côte ;
* ``05_training.ipynb`` — les pipelines **réels** (``generate-data``, ``train``) dans un bac à
  sable : vocabulaire appris, pré-entraînement par débruitage, affinage supervisé, artefacts,
  rechargement à l'identique et preuve de déterminisme ;
* ``06_error_analysis.ipynb`` — le verdict contractuel, la fidélité par type de fait, la ventilation
  par segment, les erreurs relues une par une (fait oublié, valeur inventée, phrase qui bute sur sa
  borne), les figures du rapport et des recommandations attachées à un chiffre.

Les helpers de rendu (``_md``, ``_code``, ``_insight``, ``_objectives``) sont partagés avec le
squelette tabulaire : un notebook de résumé et un notebook de classification parlent la même langue
typographique.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tools.scaffold.notebooks.tabular import _insight, _md, _objectives
from tools.scaffold.utils_notebooks import NotebookNode, code_cell, write_notebook

if TYPE_CHECKING:  # pragma: no cover
    from tools.scaffold.utils_notebooks import NotebookContext

__all__ = ["build_all"]

#: Taille du corpus réduit des notebooks (comptes-rendus) : l'exécution complète reste rapide.
NB_DOCUMENTS = 200

#: Budget d'époques du modèle de démonstration : un notebook n'entraîne pas pendant une heure.
NB_EPOCHS = 8

#: Époques de pré-entraînement par débruitage (l'étape qui stabilise un modèle initialisé au hasard).
NB_PRETRAIN_EPOCHS = 1

#: Architecture réduite des notebooks : même code, modèle plus petit, exécution de quelques secondes.
NB_LAYERS = 1
NB_UNITS = 96
NB_VOCAB = 800
NB_MIN_FREQUENCY = 2

#: Planchers triviaux mesurés par le notebook 04 (résumé vide, recopie intégrale).
TRIVIAL_NAMES = ("resume_vide", "recopie_integrale")


def _tokens(context: NotebookContext) -> dict[str, str]:
    """Build the substitution table of the summarization notebooks.

    Args:
        context: Notebook context.

    Returns:
        Mapping of ``__TOKEN__`` to replacement text.
    """
    spec = context.spec
    artifacts = dict(spec.train.get("artifacts", {}))
    params = dict(spec.model.params)
    return {
        "__TITLE__": spec.title,
        "__PROJECT__": spec.project_slug,
        "__DATASET__": spec.data.dataset_name,
        "__SEED__": str(spec.data.seed),
        "__NB_DOCUMENTS__": str(NB_DOCUMENTS),
        "__NB_EPOCHS__": str(NB_EPOCHS),
        "__NB_PRETRAIN_EPOCHS__": str(NB_PRETRAIN_EPOCHS),
        "__NB_LAYERS__": str(NB_LAYERS),
        "__NB_UNITS__": str(NB_UNITS),
        "__NB_VOCAB__": str(NB_VOCAB),
        "__NB_MIN_FREQUENCY__": str(NB_MIN_FREQUENCY),
        "__PRIMARY__": str(spec.metrics.primary),
        "__MIN_PRIMARY__": str(spec.metrics.min_primary),
        "__TASK__": str(spec.metrics.task),
        "__BASELINE__": str(spec.metrics.baseline or "lead"),
        "__COMPRESSION__": str(params.get("compression", 0.45)),
        "__MAX_OUTPUT_TOKENS__": str(params.get("max_output_tokens", 90)),
        "__MIN_OUTPUT_TOKENS__": str(params.get("min_output_tokens", 12)),
        "__MODEL_FILE__": str(artifacts.get("model_file", "summarizer.pt")),
        "__PREDICTIONS_FILE__": str(artifacts.get("predictions_file", "predictions.csv")),
        "__REPORT_NAME__": str(artifacts.get("report_file", "evaluation_report.md")),
    }


def _text(source: str, context: NotebookContext) -> NotebookNode:
    """Render a code cell with the summarization tokens substituted.

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
# Les notebooks travaillent sur un corpus réduit (__NB_DOCUMENTS__ comptes-rendus), un budget
# d'époques réduit et une architecture réduite : l'exécution reste de quelques secondes, et les
# valeurs absolues d'un notebook ne sont **pas** celles de la référence publiée par le README.
NB_DOCUMENTS = __NB_DOCUMENTS__
NB_EPOCHS = __NB_EPOCHS__

GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=str(PROJECT_ROOT / "conf"), version_base=None):
    CONFIG = validate_config(
        compose(
            config_name="config",
            overrides=[
                "mode=train",
                f"data.n_samples={NB_DOCUMENTS}",
                f"train.epochs={NB_EPOCHS}",
                "model.params.layers=__NB_LAYERS__",
                "model.params.units=__NB_UNITS__",
                "model.params.pretraining.epochs=__NB_PRETRAIN_EPOCHS__",
                "+model.params.vocab_size=__NB_VOCAB__",
                "+model.params.min_frequency=__NB_MIN_FREQUENCY__",
                "train.callbacks.progress_bar=false",
                "train.callbacks.logging_every=100",
                "train.early_stopping.enabled=false",
                "seed=__SEED__",
                "log_level=WARNING",
            ],
        )
    )

PATHS = ProjectPaths.from_root(PROJECT_ROOT)
# Bac à sable : les notebooks génèrent **leur** corpus (réduit) et écrivent **leurs** artefacts sous
# `outputs/notebooks/`. `data/raw`, `artifacts/` et donc les chiffres publiés par le README restent
# ceux de `make data` / `make train` : aucun notebook ne peut les modifier par accident.
NB_ROOT = PROJECT_ROOT / "outputs" / "notebooks"
NB_PATHS = ProjectPaths(
    root=PROJECT_ROOT,
    data_dir=NB_ROOT / "data",
    artifacts_dir=NB_ROOT / "artifacts",
).ensure()

MODEL_NODE = dict(node(CONFIG, "model"))
TEXT_COLUMN = str(MODEL_NODE.get("text_column", "text"))
ID_COLUMN = str(MODEL_NODE.get("id_column", "doc_id"))
COMPRESSION = float(MODEL_NODE.get("params", {}).get("compression", __COMPRESSION__))
MAX_OUTPUT_TOKENS = int(MODEL_NODE.get("params", {}).get("max_output_tokens", __MAX_OUTPUT_TOKENS__))
STRATEGIES = list(MODEL_NODE.get("strategies_to_compare", []) or ["__BASELINE__"])

print(f"Projet            : {CONFIG.project.name}")
print(f"Tâche             : {CONFIG.metrics.task}")
print(f"Métrique primaire : {CONFIG.metrics.primary} (seuil contractuel : __MIN_PRIMARY__)")
print(f"Algorithme servi  : {CONFIG.model.algorithm} ({CONFIG.model.name})")
print(f"Stratégies comparées : {', '.join(STRATEGIES)}")
print(f"Budget de longueur: compression {COMPRESSION:.2f}, plafond {MAX_OUTPUT_TOKENS} tokens")
print(f"Sorties notebook  : {NB_PATHS.artifacts_dir.relative_to(PROJECT_ROOT)}")
"""

LOAD_CORPUS = """
from src.data.generators import SyntheticSummaryCorpusGenerator
from src.data.loaders import SummaryCorpusLoader

LOADER = SummaryCorpusLoader(
    NB_PATHS,
    dataset_name=str(CONFIG.data.dataset_name),
    formats=CONFIG.data.formats,
    validation_enabled=CONFIG.data.validation.raw,
    lazy_validation=CONFIG.data.validation.lazy,
)
if LOADER.documents_path.exists():
    DOCUMENTS, REFERENCES, FACTS = LOADER.load_corpus()
    METADATA = LOADER.load_metadata()
    print(f"Corpus du notebook lu : {len(DOCUMENTS)} comptes-rendus, "
          f"{len(REFERENCES)} résumés de référence, {len(FACTS)} faits annotés")
else:
    # Un clone frais n'a pas encore de corpus réduit : le notebook le génère, de façon déterministe,
    # dans son bac à sable. `make data` produit le corpus de référence (600 comptes-rendus) dans
    # `data/raw` — les deux ne se mélangent jamais.
    bundle = SyntheticSummaryCorpusGenerator.from_config(CONFIG.data).generate()
    DOCUMENTS, REFERENCES, FACTS = bundle.documents, bundle.queries, bundle.facts
    METADATA = bundle.metadata
    LOADER.save_documents(DOCUMENTS)
    LOADER.save_references(REFERENCES)
    LOADER.save_facts(FACTS)
    LOADER.save_metadata(METADATA)
    print(f"Corpus réduit généré dans outputs/notebooks/data : "
          f"{len(DOCUMENTS)} comptes-rendus")

TRAIN_DOCUMENTS, TRAIN_REFERENCES, TRAIN_FACTS = LOADER.load_split("train")
VAL_DOCUMENTS, VAL_REFERENCES, VAL_FACTS = LOADER.load_split("val")
TEST_DOCUMENTS, TEST_REFERENCES, TEST_FACTS = LOADER.load_split("test")
SPLIT_SIZES = LOADER.split_sizes()
# Le test ne sert **jamais** à régler quoi que ce soit : les réglages se lisent sur la validation,
# le verdict sur le test, une seule fois.
GOLDEN = dict(zip(REFERENCES["doc_id"], REFERENCES["summary"], strict=False))
print("splits :", SPLIT_SIZES)
print("compression moyenne demandée :", round(float(REFERENCES["compression"].mean()), 4))
"""


def _family_tokens(context: NotebookContext) -> tuple[str, str, str]:
    """Return the three names the notebooks print most often.

    Args:
        context: Notebook context.

    Returns:
        ``(dataset, baseline, model file)``.
    """
    spec = context.spec
    return (
        spec.data.dataset_name,
        str(spec.metrics.baseline or "lead"),
        str(dict(spec.train.get("artifacts", {})).get("model_file", "summarizer.pt")),
    )


# ---------------------------------------------------------------------------------------------
# 01 — exploration du corpus, des résumés de référence et des faits
# ---------------------------------------------------------------------------------------------
def build_01_eda(context: NotebookContext, destination: Path) -> Path:
    """Build ``01_eda.ipynb``: cartographie du corpus, des résumés et des faits.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    cells = [
        *_header(
            context,
            "01",
            "Exploration du corpus, des résumés et des faits",
            [
                "Décrire un corpus de comptes-rendus **et** les résumés qu'on attend de lui : "
                "longueurs, compression, vocabulaire, période.",
                "Lire la table des faits annotés : sept types, une part saillante, un décor qui "
                "n'est jamais résumé.",
                "Mesurer ce que le résumé de référence couvre **de lui-même** : le plafond de la "
                "tâche, avant tout modèle.",
                "Comprendre pourquoi un ROUGE ne se lit jamais seul : il ne dit rien de la vérité "
                "d'une phrase, la couverture des faits si.",
            ],
        ),
        _md(
            "## 0. Environnement\n\n"
            "Toute la configuration vient de **Hydra** (`conf/`) : aucune valeur métier n'est codée "
            "en dur dans ce notebook. Si le corpus du notebook manque, le générateur synthétique le "
            "prend le relais (voir `make data` pour le corpus de référence)."
        ),
        _text(SETUP, context),
        _text(LOAD_CORPUS, context),
        _md(
            "## 1. Les comptes-rendus\n\n"
            "Un compte-rendu d'intervention raconte une visite : un équipement, un symptôme, une "
            "cause, des actions, parfois une pièce, une durée et un statut. Les quatre lectures "
            "ci-dessous suffisent à détecter les deux pathologies d'un corpus de résumé : une "
            "catégorie qui écrase toutes les autres, et des documents si courts que la compression "
            "n'a plus rien à couper."
        ),
        _text(
            """
DOCUMENTS.head(3)[
    ["doc_id", "intervention_type", "urgency", "site", "n_sentences", "n_tokens", "split"]
]
""",
            context,
        ),
        _text(
            """
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
DOCUMENTS["intervention_type"].value_counts().plot.bar(
    ax=axes[0], title="Comptes-rendus par type d'intervention", rot=30
)
DOCUMENTS["urgency"].value_counts().plot.bar(ax=axes[1], title="Par urgence", rot=0)
DOCUMENTS["n_tokens"].plot.hist(ax=axes[2], bins=30, title="Longueur des documents (tokens)")
plt.tight_layout()
plt.show()

print("période couverte        :", DOCUMENTS["published_at"].min().date(),
      "->", DOCUMENTS["published_at"].max().date())
print("identifiants dupliqués  :", int(DOCUMENTS["doc_id"].duplicated().sum()))
print("phrases par document    :", round(float(DOCUMENTS["n_sentences"].mean()), 1),
      f"(min {int(DOCUMENTS['n_sentences'].min())}, max {int(DOCUMENTS['n_sentences'].max())})")
print("tokens par document     :", round(float(DOCUMENTS["n_tokens"].mean()), 1))
print("sites distincts         :", DOCUMENTS["site"].nunique())
""",
            context,
        ),
        _insight(
            [
                "Un type d'intervention qui concentre la majorité des documents oriente tout ce "
                "que le modèle apprendra : la ventilation par type est un prérequis à la lecture "
                "des scores, et le notebook 06 ventile les erreurs par segment.",
                "La longueur des documents fixe la difficulté réelle : plus un compte-rendu est "
                "long, plus il contient de faits non saillants qu'un résumé doit **laisser "
                "tomber**.",
                "Des identifiants dupliqués casseraient toutes les jointures entre documents, "
                "résumés et faits : le contrat les refuse (notebook 02) et cette cellule les "
                "compte.",
            ]
        ),
        _md(
            "## 2. Les résumés de référence\n\n"
            "Un résumé de référence n'est pas « le document en plus court » : il est **écrit** à "
            "partir des faits saillants, avec ses propres formulations. La compression qu'il "
            "atteint est la cible réelle de la tâche — celle que le budget de longueur du modèle "
            "doit respecter."
        ),
        _text(
            """
from src.features.build_features import SentenceFeatureBuilder

profile = SentenceFeatureBuilder.profile(DOCUMENTS, REFERENCES)
display(pd.Series(profile, name="valeur").to_frame().round(4))

display(REFERENCES.head(3)[["doc_id", "n_sentences", "n_tokens", "n_salient_facts", "compression"]])
""",
            context,
        ),
        _text(
            """
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
REFERENCES["n_tokens"].plot.hist(ax=axes[0], bins=30, title="Longueur des résumés (tokens)")
REFERENCES["compression"].plot.hist(ax=axes[1], bins=30, title="Compression des résumés")
REFERENCES["n_salient_facts"].plot.hist(ax=axes[2], bins=12, title="Faits saillants par résumé")
plt.tight_layout()
plt.show()

print("compression moyenne     :", round(float(REFERENCES["compression"].mean()), 4))
print("compression maximale    :", round(float(REFERENCES["compression"].max()), 4))
print("part au plafond du budget (>= 0,599) :",
      round(float((REFERENCES["compression"] >= 0.599).mean()), 3))
print("phrases par résumé      :", round(float(REFERENCES["n_sentences"].mean()), 1),
      f"(min {int(REFERENCES['n_sentences'].min())}, max {int(REFERENCES['n_sentences'].max())})")
""",
            context,
        ),
        _text(
            """
# La question qui décide de tout : la référence **recopie-t-elle** le document, ou le réécrit-elle ?
# Si elle recopiait, un extracteur de phrases gagnerait ; elle réécrit, donc il faut générer.
example_id = str(DOCUMENTS.iloc[0]["doc_id"])
document = DOCUMENTS.loc[DOCUMENTS["doc_id"] == example_id, "text"].iloc[0]
reference = REFERENCES.loc[REFERENCES["doc_id"] == example_id, "summary"].iloc[0]
print(f"--- document {example_id} ({len(document.split())} mots) ---")
print(document)
print(f"\\n--- résumé de référence ({len(reference.split())} mots) ---")
print(reference)
print("\\nphrases du document présentes telles quelles dans le résumé :",
      int(sum(1 for sentence in reference.split(". ") if sentence.strip().rstrip(".") + "." in document)))
""",
            context,
        ),
        _insight(
            [
                "La compression demandée est le premier chiffre à connaître : un budget de longueur "
                "fixé au-dessus d'elle produit des résumés plus longs que la référence, donc un "
                "rappel de ROUGE plafonné.",
                "Très peu de références touchent le plafond du budget : le corpus n'est pas un "
                "exercice de troncature, c'est un exercice de sélection **et** de reformulation.",
                "Les résumés de référence ne recopient pas les phrases du document : c'est la part "
                "de réécriture du corpus, et c'est elle qui empêche un extracteur d'atteindre 1,0.",
            ]
        ),
        _md(
            "## 3. Les faits annotés\n\n"
            "Chaque phrase qui porte un fait rapportable a écrit une ligne dans la table des faits : "
            "son type, sa valeur, sa **surface** exacte dans le texte et l'index de sa phrase. Tous "
            "les faits ne sont pas saillants : le résumé de référence n'en rapporte qu'une partie, "
            "et c'est cette partie-là qui est mesurée."
        ),
        _text(
            """
from src.data.schemas import fact_columns, salient_fact_counts

display(pd.Series(salient_fact_counts(FACTS), name="faits saillants").to_frame())
print("faits annotés          :", len(FACTS), f"({len(FACTS) / len(DOCUMENTS):.2f} par document)")
print("faits saillants        :", int(FACTS["salient"].sum()),
      f"({FACTS['salient'].mean():.2f} par fait annoté)")
print("colonnes de fidélité publiées :", fact_columns())
""",
            context,
        ),
        _text(
            """
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
FACTS["fact_type"].value_counts().plot.bar(
    ax=axes[0], title="Faits annotés par type", rot=30
)
(
    FACTS[FACTS["salient"] == 1]["fact_type"].value_counts()
    / FACTS["fact_type"].value_counts()
).plot.bar(ax=axes[1], title="Part saillante par type", rot=30)
plt.tight_layout()
plt.show()

example_facts = FACTS[FACTS["doc_id"] == example_id]
display(example_facts[["fact_type", "value", "surface", "salient", "sentence_index"]])
""",
            context,
        ),
        _md(
            "## 4. Le décor est un distracteur déclaré\n\n"
            "Numéros de ticket, de téléphone ou de parking, horaires de présence, formules "
            "administratives : ces valeurs sont **dans** les documents et ne figurent **jamais** "
            "dans les résumés. Elles servent à piéger deux comportements : la recopie intégrale et "
            "l'invention. Le rapport publie les valeurs non supportées, donc un modèle qui en "
            "produit une est détecté (notebook 06)."
        ),
        _text(
            """
from src.data.vocabulary import DISTRACTORS, MAX_COMPRESSION, SUMMARY_SENTENCE_RANGE

corpus_text = " ".join(DOCUMENTS["text"])
reference_text = " ".join(REFERENCES["summary"])
rows = [
    {
        "distracteur": value,
        "dans les documents": int(corpus_text.count(value)),
        "dans les résumés": int(reference_text.count(value)),
    }
    for value in DISTRACTORS
]
display(pd.DataFrame(rows))
print("types de faits      :", sorted(FACTS["fact_type"].unique()))
print("phrases par résumé  :", SUMMARY_SENTENCE_RANGE, "(contrat)")
print("compression maximale:", MAX_COMPRESSION, "(plafond du générateur de corpus)")
""",
            context,
        ),
        _md(
            "## 5. Ce que la référence couvre d'elle-même\n\n"
            "Avant de mesurer un modèle, on mesure la **vérité** : le résumé de référence "
            "contient-il les faits saillants qu'il est censé rapporter ? Si la réponse n'était pas "
            "« tous », aucune couverture de modèle ne serait interprétable — l'annotation et la "
            "référence se contrediraient. C'est le plafond de la tâche, et il se vérifie."
        ),
        _text(
            """
from src.training.metrics import coverage_scores

per_document = []
for row in REFERENCES.itertuples(index=False):
    facts = FACTS[FACTS["doc_id"] == str(row.doc_id)]
    score = coverage_scores(str(row.summary), facts)
    per_document.append(
        {
            "doc_id": str(row.doc_id),
            "coverage": score["fact_coverage"],
            "unsupported": score["n_unsupported"],
        }
    )
CEILING = pd.DataFrame(per_document)
print("couverture de la référence : min", round(float(CEILING["coverage"].min()), 4),
      "| moyenne", round(float(CEILING["coverage"].mean()), 4))
print("valeurs non supportées     :", int(CEILING["unsupported"].sum()))
print("documents parfaitement couverts :",
      int((CEILING["coverage"] == 1.0).sum()), "/", len(CEILING))
""",
            context,
        ),
        _text(
            """
# Le plancher symétrique : un résumé vide a une couverture nulle et un ROUGE de 0,0 — la métrique
# est *définie* sur un système muet, ce qui est la condition pour publier un verdict.
from src.evaluation.rouge import rouge_scores
from src.training.metrics import trivial_floor

empty = rouge_scores(GOLDEN[str(REFERENCES.iloc[0]["doc_id"])], "")
copy_rouge = rouge_scores(
    GOLDEN[str(REFERENCES.iloc[0]["doc_id"])],
    DOCUMENTS.loc[DOCUMENTS["doc_id"] == REFERENCES.iloc[0]["doc_id"], "text"].iloc[0],
)
display(pd.DataFrame(
    [trivial_floor(), {"rouge1_f": empty.rouge1.f1, "rouge_l_f": empty.rouge_l.f1},
     {"rouge1_f": copy_rouge.rouge1.f1, "rouge_l_f": copy_rouge.rouge_l.f1}],
    index=["résumé vide (plancher)", "résumé vide (ROUGE)", "recopie intégrale (exemple)"],
).round(4))
""",
            context,
        ),
        _insight(
            [
                "Un plafond mesuré avant tout modèle est ce qui rend un score interprétable : ici "
                "la référence couvre exactement les faits saillants qu'elle annonce.",
                "Un ROUGE élevé sur une recopie intégrale serait un mauvais signe : il signifierait "
                "que le corpus ne demande pas de réécriture — ce n'est pas ce corpus.",
                "La couverture et le ROUGE mesurent deux choses différentes, et le projet publie "
                "les deux : le premier dit si le résumé est **vrai**, le second s'il ressemble à la "
                "référence.",
            ]
        ),
    ]
    return write_notebook(destination / "01_eda.ipynb", cells)


# ---------------------------------------------------------------------------------------------
# 02 — les contrats des trois tables, et leurs liens croisés
# ---------------------------------------------------------------------------------------------
def build_02_validation(context: NotebookContext, destination: Path) -> Path:
    """Build ``02_validation.ipynb``: contrats Pandera et cohérences croisées.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    cells = [
        *_header(
            context,
            "02",
            "Validation des données et contrats",
            [
                "Lire les contrats du projet : documents, résumés de référence, faits et table de "
                "prédictions publiée.",
                "Comprendre pourquoi **trois** tables ne suffisent pas : ce qu'un contrat par table "
                "ne peut pas voir, et qui pourtant fausse un score.",
                "Casser chaque règle volontairement pour lire le message d'échec qu'on lira en "
                "production.",
                "Distinguer une violation de table (``SchemaError``) d'une incohérence croisée "
                "(``ValueError``) : les deux se corrigent différemment.",
            ],
        ),
        _md(
            "## 0. Environnement et corpus\n\n"
            "Le notebook lit son propre corpus réduit (`outputs/notebooks/data`) ; s'il manque, il "
            "le génère et le persiste pour que les pipelines du notebook travaillent sur les mêmes "
            "fichiers que lui."
        ),
        _text(SETUP, context),
        _text(LOAD_CORPUS, context),
        _md(
            "## 1. Les quatre contrats\n\n"
            "Un contrat de données est du code exécutable : il nomme chaque colonne, son type et ses "
            "bornes. C'est la documentation la plus fiable du jeu de données, puisqu'elle est vérifiée "
            "à chaque exécution du pipeline."
        ),
        _text(
            """
from src.data.schemas import (
    DocumentsSchema,
    FactsSchema,
    PredictedSummariesSchema,
    ReferenceSummariesSchema,
    validate_documents,
    validate_facts,
    validate_references,
)

for schema in (DocumentsSchema, ReferenceSummariesSchema, FactsSchema, PredictedSummariesSchema):
    print(f"{schema.__name__:26s} -> {', '.join(schema.to_schema().columns)}")
""",
            context,
        ),
        _md("### 1.1 Le corpus respecte ses contrats"),
        _text(
            """
validated_documents = validate_documents(DOCUMENTS)
validated_references = validate_references(REFERENCES)
validated_facts = validate_facts(FACTS)
print(f"{len(validated_documents)} documents, {len(validated_references)} résumés, "
      f"{len(validated_facts)} faits : les trois tables passent leurs contrats")

# Une propriété que le contrat par table **ne peut pas** vérifier : la surface d'un fait est bien
# l'extrait du document à la phrase annoncée. Elle est vérifiée ici, une fois, sur tout le corpus.
texts = dict(zip(validated_documents["doc_id"], validated_documents["text"], strict=False))
surfaces_ok = 0
for row in validated_facts.itertuples(index=False):
    if str(row.surface) in texts[str(row.doc_id)]:
        surfaces_ok += 1
print(f"surfaces retrouvées dans leur document : {surfaces_ok} / {len(validated_facts)}")
assert surfaces_ok == len(validated_facts), "une surface de fait n'est pas dans son document"
""",
            context,
        ),
        _insight(
            [
                "Le contrat des faits vérifie la **forme** (type connu, saillance binaire, index "
                "positif) ; la cellule ci-dessus vérifie la **substance** (la surface est bien dans "
                "le texte). Les deux sont nécessaires : un fait bien formé et faux reste faux.",
                "La table des prédictions est stricte sur ses colonnes : c'est le contrat publié, "
                "donc un ajout de colonne dans un rapport est un changement de contrat, pas un "
                "détail.",
                "Les résumés de référence sont validés sur leurs bornes (2 à 8 phrases) : un résumé "
                "d'une phrase serait un titre, pas un résumé, et fausserait la compression cible.",
            ]
        ),
        _md(
            "## 2. Les liens entre les tables\n\n"
            "Ce qu'aucun contrat par table ne peut voir : un résumé de référence qui parle d'un "
            "document inconnu, un document que personne n'a résumé, un fait qui pointe au-delà des "
            "phrases de son document. Ces trois erreurs sont celles qui faussent un score sans jamais "
            "lever d'erreur — elles sont donc vérifiées par :func:`validate_corpus`."
        ),
        _text(
            """
from src.data.schemas import validate_corpus

validate_corpus(DOCUMENTS, REFERENCES, FACTS)
print("les trois tables sont cohérentes entre elles :",
      f"{len(DOCUMENTS)} documents, {len(REFERENCES)} références, {len(FACTS)} faits")
""",
            context,
        ),
        _md(
            "## 3. Casser volontairement les contrats\n\n"
            "Chaque corruption ci-dessous correspond à un incident réel. Le message d'erreur est "
            "affiché tel quel — c'est lui qu'on lira en production, et c'est pour cela qu'il nomme le "
            "coupable."
        ),
        _text(
            """
from typing import Any


def show_failure(label: str, call: Any) -> None:
    \"\"\"Exécuter un validateur et afficher la première ligne de son erreur.\"\"\"
    try:
        call()
    except Exception as error:  # on veut le message du contrat, quel qu'il soit
        print(f"[refusé] {label} : {str(error).splitlines()[0]}")
    else:
        print(f"[accepté] {label} : aucune erreur (le contrat est trop laxiste)")


# 1. un identifiant de document hors format
bad_id = DOCUMENTS.head(1).copy()
bad_id.loc[bad_id.index[0], "doc_id"] = "CR-1"
show_failure("identifiant hors format", lambda: validate_documents(bad_id))

# 2. une colonne manquante : la table n'est plus celle du contrat
show_failure("colonne manquante", lambda: validate_documents(DOCUMENTS.drop(columns=["text"])))

# 3. un type de fait inconnu
unknown_type = FACTS.head(1).copy()
unknown_type.loc[unknown_type.index[0], "fact_type"] = "humeur"
show_failure("type de fait inconnu", lambda: validate_facts(unknown_type))

# 4. une saillance qui n'est ni 0 ni 1
bad_salient = FACTS.head(1).copy()
bad_salient.loc[bad_salient.index[0], "salient"] = 2
show_failure("saillance hors {0, 1}", lambda: validate_facts(bad_salient))
""",
            context,
        ),
        _text(
            """
# 5. un résumé de référence orphelin : il parle d'un document qui n'existe pas
orphan = REFERENCES.copy()
orphan.loc[orphan.index[0], "doc_id"] = "CR-9999"
show_failure("résumé orphelin", lambda: validate_corpus(DOCUMENTS, orphan, FACTS))

# 6. un document que personne n'a résumé : il ne pourra jamais être mesuré
missing = REFERENCES.iloc[1:].copy()
show_failure("document sans référence", lambda: validate_corpus(DOCUMENTS, missing, FACTS))

# 7. un fait au-delà des phrases de son document
sizes = DOCUMENTS.set_index("doc_id")["n_sentences"]
target = str(sizes.idxmin())
outside = FACTS.copy()
outside.loc[outside["doc_id"] == target, "sentence_index"] = int(sizes.min())
show_failure("fait hors des phrases de son document",
             lambda: validate_corpus(DOCUMENTS, REFERENCES, outside))
""",
            context,
        ),
        _insight(
            [
                "Les quatre premières corruptions lèvent un ``SchemaError`` (la table casse son "
                "propre contrat) ; les trois suivantes un ``ValueError`` (les tables se "
                "contredisent). Deux familles d'erreurs, deux corrections différentes.",
                "Un document sans référence est aussi grave qu'un résumé orphelin : dans les deux cas "
                "une jointure silencieuse ferait disparaître des lignes — et un score calculé sur "
                "moins de lignes n'est plus comparable.",
                "Un fait dont l'index dépasse les phrases du document est typique d'une annotation "
                "décalée après un retraitement du texte : le contrat le refuse au lieu de le "
                "rattacher à la dernière phrase.",
            ]
        ),
        _md(
            "## 4. Le contrat de la table publiée\n\n"
            "Le pipeline d'inférence écrit une table de prédictions : identifiant, stratégie, résumé "
            "produit, longueur, ROUGE quand la référence est connue, couverture des faits et latence. "
            "C'est un **artefact publié** : son contrat est vérifié avant écriture (notebook 06)."
        ),
        _text(
            """
from src.inference.predictor import SummaryPredictor
from src.models.lead import LeadSummarizer

lead = LeadSummarizer(compression=COMPRESSION, max_output_tokens=MAX_OUTPUT_TOKENS, seed=__SEED__)
lead.fit(TRAIN_DOCUMENTS, TRAIN_REFERENCES)
pre_predictor = SummaryPredictor(
    lead, paths=NB_PATHS, config=CONFIG.model_dump(),
    text_column=TEXT_COLUMN, id_column=ID_COLUMN,
)
RAW_PREDICTIONS = pre_predictor.predict(TEST_DOCUMENTS)
# `save` écrit la table **publiée** (le contrat), puis on la relit comme le ferait un lecteur du
# rapport : ce qui est validé ici est exactement ce qui est sur le disque.
PUBLISHED = pd.read_csv(pre_predictor.save(RAW_PREDICTIONS), keep_default_na=False)
print("colonnes de la table publiée :", list(PUBLISHED.columns))
display(PUBLISHED.head(2).round(4))
""",
            context,
        ),
        _text(
            """
from src.data.schemas import validate_predictions

validated_predictions = validate_predictions(PUBLISHED)
print(f"{len(validated_predictions)} prédictions validées")

bad_prediction = PUBLISHED.copy()
bad_prediction.loc[bad_prediction.index[0], "doc_id"] = "pas-un-document"
show_failure("prédiction d'un document inconnu", lambda: validate_predictions(bad_prediction))

impossible_compression = PUBLISHED.copy()
impossible_compression.loc[impossible_compression.index[0], "compression"] = 1.4
show_failure("compression supérieure à 1 (résumé plus long que le document)",
             lambda: validate_predictions(impossible_compression))

extra_column = PUBLISHED.copy()
extra_column["commentaire"] = "une colonne ajoutée à la main"
show_failure("colonne non déclarée dans la table publiée",
             lambda: validate_predictions(extra_column))
""",
            context,
        ),
        _insight(
            [
                "La compression est bornée à 1,0 dans le contrat publié : un résumé plus long que son "
                "document n'est pas un résumé, et cette borne a déjà attrapé un décodeur qui "
                "recrachait ses propres jetons spéciaux.",
                "La table publiée est stricte : une colonne ajoutée par un analyste casse le "
                "contrat, donc elle est refusée **avant** d'être écrite — pas après, dans un "
                "dashboards qui lit des colonnes par position.",
                "Un résumé vide est autorisé par le contrat (c'est la baseline ``resume_vide``) : "
                "c'est la métrique qui doit être *définie* sur un système muet, pas le contrat qui "
                "doit l'interdire.",
            ]
        ),
    ]
    return write_notebook(destination / "02_validation.ipynb", cells)


# ---------------------------------------------------------------------------------------------
# 03 — découpage, alignement des faits et budget de longueur
# ---------------------------------------------------------------------------------------------
def build_03_preprocessing(context: NotebookContext, destination: Path) -> Path:
    """Build ``03_preprocessing.ipynb``: phrases, faits alignés et budget de longueur.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    cells = [
        *_header(
            context,
            "03",
            "Découpage, alignement des faits et budget de longueur",
            [
                "Découper un compte-rendu en phrases et en tokens, et vérifier que rien ne se "
                "perd — les faits annotés sont repérés au caractère.",
                "Calculer le budget de longueur qu'un modèle respectera, et mesurer la compression "
                "réellement demandée par le corpus.",
                "Mesurer ce qu'un **extracteur de phrases** retrouverait déjà : le plafond de la "
                "recopie, avant toute réécriture.",
                "Vérifier si la centralité d'une phrase suffit à prédire sa saillance — la réponse "
                "conditionne le choix d'un modèle extractif ou génératif.",
            ],
        ),
        _md(
            "## 0. Environnement et corpus\n\n"
            "Le notebook lit son propre corpus réduit (`outputs/notebooks/data`) ; s'il manque, il "
            "le génère."
        ),
        _text(SETUP, context),
        _text(LOAD_CORPUS, context),
        _md(
            "## 1. Découper un compte-rendu\n\n"
            "Un fait annoté est repéré par une **surface exacte** dans le texte et par l'index de sa "
            "phrase. Toute étape de prétraitement qui déplace le texte (normalisation, retrait de "
            "doublons) déplace donc aussi les annotations : c'est la contrainte qui décide de qui a "
            "le droit de réécrire le corpus."
        ),
        _text(
            """
from src.preprocessing.transformers import normalise_text, split_sentences, tokenize

example_id = str(DOCUMENTS.iloc[0]["doc_id"])
example_text = DOCUMENTS.loc[DOCUMENTS["doc_id"] == example_id, "text"].iloc[0]
sentences = split_sentences(example_text)
sentence_table = pd.DataFrame(
    {
        "phrase": range(len(sentences)),
        "tokens": [len(tokenize(sentence)) for sentence in sentences],
        "caractères": [len(sentence) for sentence in sentences],
        "début": [example_text.find(sentence) for sentence in sentences],
        "extrait": [sentence[:70] for sentence in sentences],
    }
)
display(sentence_table)

declared_sentences = int(
    DOCUMENTS.loc[DOCUMENTS["doc_id"] == example_id, "n_sentences"].iloc[0]
)
print("phrases reconstituées :", len(sentences), "| déclarées au corpus :", declared_sentences)
print("tokénisation sans perte  :", len(tokenize(example_text)), "tokens")
print("normalisation (exemple)  :", normalise_text("Le  Câble   s'est DÉTENDU !") )
""",
            context,
        ),
        _insight(
            [
                "Le nombre de phrases reconstituées doit égaler la valeur déclarée par le corpus : "
                "un écart signalerait un découpage différent de celui qui a produit les "
                "annotations.",
                "L'offset de chaque phrase est publié (`début`) : c'est lui qui permet de rattacher "
                "une surface de fait à sa phrase, et donc de vérifier l'annotation au lieu de la "
                "supposer.",
                "La tokénisation est sans perte par construction : elle sert à **mesurer**, pas à "
                "réécrire le corpus. Une normalisation destructive casserait tous les offsets.",
            ]
        ),
        _md("### 1.1 Les faits sont alignés sur le texte\n\n"
            "Trois vérifications, sur tout le corpus : la surface est dans le document, la valeur "
            "est portée par la phrase annoncée, et l'index de phrase existe."
        ),
        _text(
            """
texts = dict(zip(DOCUMENTS["doc_id"], DOCUMENTS["text"], strict=False))
out_of_bounds = 0
surface_missing = 0
sentence_mismatch = 0
for row in FACTS.itertuples(index=False):
    document = texts[str(row.doc_id)]
    if str(row.surface) not in document:
        surface_missing += 1
        continue
    document_sentences = split_sentences(document)
    if int(row.sentence_index) >= len(document_sentences):
        out_of_bounds += 1
        continue
    if str(row.value).split()[0].casefold() not in document_sentences[int(row.sentence_index)].casefold():
        sentence_mismatch += 1

print(f"faits hors bornes            : {out_of_bounds}")
print(f"surfaces absentes du document: {surface_missing}")
print(f"valeurs absentes de la phrase annoncée : {sentence_mismatch}")
example_facts = FACTS[FACTS["doc_id"] == example_id]
display(example_facts[["fact_type", "value", "surface", "sentence_index", "salient"]])
""",
            context,
        ),
        _insight(
            [
                "Un fait mal aligné ne casse aucun contrat de table : il fausse silencieusement la "
                "couverture, parce qu'un modèle ne pourra jamais retrouver une surface qui n'est "
                "pas dans le texte.",
                "Les faits non saillants sont conservés dans la table : ils ne sont pas mesurés (le "
                "résumé n'a pas à les rapporter) mais ils rendent visible ce qu'un modèle qui "
                "recopie tout gagnerait — rien, en couverture.",
                "La phrase annoncée est vérifiée séparément de la surface : une valeur peut être "
                "dans le document **et** dans la mauvaise phrase si l'annotation a été décalée.",
            ]
        ),
        _md(
            "## 2. Le budget de longueur\n\n"
            "Le modèle ne décide pas librement de sa longueur : il reçoit un budget, proportionnel au "
            "document et borné par un plancher et un plafond. Le notebook recalcule ce budget comme "
            "le fera le modèle, puis le compare aux résumés de référence."
        ),
        _text(
            """
from src.data.vocabulary import MAX_COMPRESSION
from src.training.metrics import length_stats

# Reconstitution explicite de la formule du contrat : arrondi de `compression x tokens`, borné par
# `min_output_tokens` puis par `max_output_tokens`. La recalculer ici est le but de la cellule : un
# budget qu'on ne sait pas recalculer est un budget qu'on ne sait pas expliquer.
MIN_OUTPUT_TOKENS = int(node(CONFIG, "model").get("params", {}).get("min_output_tokens", 12))
computed = (
    (DOCUMENTS["n_tokens"] * COMPRESSION).round().clip(MIN_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS)
)
budget_table = pd.DataFrame({"document": DOCUMENTS["n_tokens"], "budget": computed})
budget_table["référence"] = REFERENCES["n_tokens"].to_numpy()
display(budget_table.describe().round(2))

print("budget moyen                 :", round(float(computed.mean()), 1), "tokens")
print("longueur moyenne des résumés :", round(float(REFERENCES["n_tokens"].mean()), 1), "tokens")
print("résumés au-dessus du budget  :",
      int((REFERENCES["n_tokens"] > computed).sum()), "/", len(REFERENCES))
print("part des résumés qui atteignent le plafond du budget  :",
      round(float((REFERENCES["compression"] >= MAX_COMPRESSION - 0.001).mean()), 3))
""",
            context,
        ),
        _text(
            """
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(budget_table["budget"], bins=30, alpha=0.6, label="budget du modèle")
axes[0].hist(budget_table["référence"], bins=30, alpha=0.6, label="résumé de référence")
axes[0].set_title("Budget et longueur des références (tokens)")
axes[0].legend()
axes[1].scatter(budget_table["document"], budget_table["référence"], s=8, alpha=0.5)
axes[1].set_xlabel("tokens du document")
axes[1].set_ylabel("tokens du résumé de référence")
axes[1].set_title("Compression observée")
plt.tight_layout()
plt.show()

# La table des références ne porte pas les colonnes d'une **prédiction** : on les construit ici, avec
# le budget recalculé plus haut, plutôt que d'ajouter de fausses colonnes au corpus.
reference_lengths = pd.DataFrame(
    {
        "compression": REFERENCES["compression"].to_numpy(),
        "n_sentences": REFERENCES["n_sentences"].to_numpy(),
        "hit_max_length": (REFERENCES["n_tokens"].to_numpy() >= computed.to_numpy()).astype(int),
    }
)
print("longueurs de référence, mesurées avec les métriques du projet :")
display(pd.Series(length_stats(reference_lengths), name="référence").round(4))
""",
            context,
        ),
        _insight(
            [
                "Un budget calculé **avant** l'entraînement évite le réglage a posteriori : le "
                "modèle apprend à tenir dans une longueur, il ne la découvre pas à l'inférence.",
                "Si la plupart des résumés de référence dépassaient le budget, le rappel de ROUGE "
                "serait plafonné par construction — c'est le premier chiffre à regarder avant "
                "d'accuser le modèle.",
                "Le plancher et le plafond du budget ne sont pas cosmétiques : sans plancher, un "
                "document court produit un résumé de trois mots ; sans plafond, un document long "
                "produit un décodeur qui n'en finit pas.",
            ]
        ),
        _md(
            "## 3. Ce qu'un extracteur de phrases retrouverait déjà\n\n"
            "Avant de générer, on mesure la **recopie** : pour chaque phrase du résumé de référence, "
            "quelle est la meilleure phrase du document ? Ce plafond d'extraction dit la part de la "
            "tâche qui n'est qu'une sélection, et celle qui exige une reformulation."
        ),
        _text(
            """
from src.evaluation.rouge import rouge_scores

SAMPLE = VAL_DOCUMENTS.head(40)
golden = dict(zip(VAL_REFERENCES["doc_id"], VAL_REFERENCES["summary"], strict=False))
oracle_scores = []
for row in SAMPLE.itertuples(index=False):
    document_sentences = split_sentences(str(row.text))
    reference_sentences = split_sentences(golden[str(row.doc_id)])
    for reference_sentence in reference_sentences:
        best = max(
            (rouge_scores(reference_sentence, candidate).rouge1.f1
             for candidate in document_sentences),
            default=0.0,
        )
        oracle_scores.append(best)

oracle = pd.Series(oracle_scores, name="meilleure phrase du document (ROUGE-1)")
print("plafond d'extraction (recopier la meilleure phrase par phrase de référence) :")
print(f"  ROUGE-1 moyen : {oracle.mean():.4f}")
print(f"  phrases du résumé parfaitement retrouvables (ROUGE-1 = 1,0) : "
      f"{float((oracle >= 0.999).mean()):.1%}")
print(f"  phrases sans aucune correspondance correcte (ROUGE-1 < 0,5) : "
      f"{float((oracle < 0.5).mean()):.1%}")
""",
            context,
        ),
        _insight(
            [
                "Un extracteur de phrases est plafonné par ce chiffre : au-delà, il faut réécrire. "
                "C'est le même argument que la comparaison des stratégies, mais mesuré sur la "
                "référence au lieu d'un modèle.",
                "Les phrases sans correspondance correcte sont celles qui portent une **synthèse** "
                "(ou une reformulation) : elles justifient un modèle génératif, à condition que la "
                "génération reste fidèle.",
                "Mesurer le plafond sur la validation, jamais sur le test : le test ne sert qu'une "
                "fois, pour le verdict du notebook 06.",
            ]
        ),
        _md(
            "## 4. La centralité suffit-elle ?\n\n"
            "Une heuristique classique prend les phrases **les plus centrales** (proches des "
            "autres). Le projet construit ce graphe, donc la question se mesure : les phrases "
            "centrales portent-elles les faits saillants ?"
        ),
        _text(
            """
from src.features.build_features import SentenceFeatureBuilder

builder = SentenceFeatureBuilder.from_config(CONFIG.preprocessing).build(example_id, example_text)
print(builder.summary())
central = builder.most_central_sentences(3)
for rank, (index, score) in enumerate(central, start=1):
    print(f"  {rank}. phrase {index} (centralité {score:.4f}) — {builder.sentences[index][:80]}")
    print("     termes saillants :", [term for term, _ in builder.top_terms(index, 5)])
""",
            context,
        ),
        _text(
            """
hits = []
for row in VAL_DOCUMENTS.head(40).itertuples(index=False):
    features = SentenceFeatureBuilder.from_config(CONFIG.preprocessing).build(
        str(row.doc_id), str(row.text)
    )
    central_indices = [index for index, _ in features.most_central_sentences(3)]
    salient = FACTS[(FACTS["doc_id"] == str(row.doc_id)) & (FACTS["salient"] == 1)]
    if salient.empty:
        continue
    hits.append(float(salient["sentence_index"].isin(central_indices).mean()))

centrality_hit = pd.Series(hits, name="faits saillants dans les 3 phrases les plus centrales")
print(f"documents mesurés : {len(centrality_hit)}")
print(f"part des faits saillants portés par une phrase centrale : {centrality_hit.mean():.1%}")
print(f"documents où toutes les phrases centrales portent un fait : "
      f"{float((centrality_hit > 0).mean()):.1%}")
""",
            context,
        ),
        _insight(
            [
                "Si la centralité retrouvait la majorité des faits saillants, une baseline "
                "extractive suffirait — et le projet publierait cette baseline comme référence, ce "
                "qu'il fait (``textrank``).",
                "La centralité est lexicale : elle ne distingue pas « la pièce CRR-255 » de « la "
                "presse PR-6 ». Un fait rare pèse moins qu'un mot fréquent, donc il passe "
                "sous le seuil de sélection.",
                "C'est exactement l'écart de mesure entre ROUGE (ressemblance aux mots de la "
                "référence) et couverture (présence des faits annotés) : les deux sont publiés.",
            ]
        ),
    ]
    return write_notebook(destination / "03_preprocessing.ipynb", cells)


# ---------------------------------------------------------------------------------------------
# 04 — les trois stratégies mesurées sur les mêmes lignes
# ---------------------------------------------------------------------------------------------
def build_04_model_exploration(context: NotebookContext, destination: Path) -> Path:
    """Build ``04_model_exploration.ipynb``: planchers, baselines et modèle appris.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    cells = [
        *_header(
            context,
            "04",
            "Exploration des stratégies : du plancher trivial au modèle appris",
            [
                "Mesurer les planchers triviaux : résumé vide, recopie intégrale, premières "
                "phrases. Un score n'a de sens que comparé à ce que le hasard et la paresse "
                "obtiennent.",
                "Comparer les deux baselines extractives de la stack sur **les mêmes lignes** de "
                "validation.",
                "Entraîner un encodeur-décodeur minuscule sur le corpus réduit et le mesurer avec "
                "exactement les mêmes métriques.",
                "Lire ce que le modèle regarde (parts d'attention par phrase) et ce qu'il ne peut "
                "pas savoir (taux de jetons inconnus du vocabulaire).",
            ],
        ),
        _md(
            "## 0. Environnement et corpus\n\n"
            "Le corpus réduit des notebooks (`__NB_DOCUMENTS__` comptes-rendus) et une architecture "
            "réduite gardent l'exécution de ce notebook en quelques secondes. Les valeurs absolues "
            "ne sont **pas** celles de la référence publiée par le README."
        ),
        _text(SETUP, context),
        _text(LOAD_CORPUS, context),
        _md(
            "## 1. Les planchers triviaux\n\n"
            "Trois systèmes que personne n'appellerait « un modèle », et qu'aucun modèle n'a le "
            "droit de ne pas battre : un résumé vide, la recopie intégrale du document, et les "
            "premières phrases. Le troisième est publié comme **référence** dans le rapport "
            "d'évaluation."
        ),
        _text(
            """
from src.evaluation.evaluator import SummaryEvaluator
from src.evaluation.rouge import corpus_rouge
from src.models import build_model
from src.models.lead import LeadSummarizer
from src.training.metrics import trivial_floor

val_references = VAL_REFERENCES["summary"].tolist()
empty_rouge = corpus_rouge(val_references, [""] * len(val_references))
copy_rouge = corpus_rouge(val_references, VAL_DOCUMENTS["text"].tolist())

# L'évaluateur est le même objet que celui du pipeline `evaluate` : scorer une stratégie ici, c'est
# appliquer exactement les métriques publiées (ROUGE **et** fidélité), sur les mêmes lignes.
SCORER = SummaryEvaluator(
    LeadSummarizer(compression=COMPRESSION, max_output_tokens=MAX_OUTPUT_TOKENS, seed=__SEED__),
    config=CONFIG.model_dump(),
    metrics_config=CONFIG.metrics.model_dump(),
    paths=NB_PATHS,
)
lead = build_model(CONFIG, algorithm="lead")
lead.fit(TRAIN_DOCUMENTS, TRAIN_REFERENCES)
lead_scored = SCORER.score_model(lead, VAL_DOCUMENTS, VAL_REFERENCES, VAL_FACTS)

FLOOR_METRICS = ("rouge1_f", "rouge2_f", "rouge_l_f", "fact_coverage", "compression")


def mean_scores(frame: pd.DataFrame) -> dict[str, float]:
    # Moyenne des métriques publiées d'une table de prédictions scorée.
    return {name: round(float(frame[name].mean()), 4) for name in FLOOR_METRICS}


floors = pd.DataFrame(
    [
        {"stratégie": "resume_vide", **trivial_floor(), "compression": 0.0},
        {
            "stratégie": "recopie_integrale",
            **copy_rouge,
            "fact_coverage": float("nan"),
            "compression": 1.0,
        },
        {"stratégie": "lead", **mean_scores(lead_scored)},
    ]
)
display(floors.set_index("stratégie").round(4))
print("planchers mesurés sur", len(VAL_DOCUMENTS), "documents de validation")
print("contenu des références mesuré par la copie intégrale :", round(copy_rouge["rouge1_f"], 4))
""",
            context,
        ),
        _insight(
            [
                "Un système muet obtient 0,0 : la métrique est *définie* sur une sortie vide, ce qui "
                "est la condition pour publier un verdict chiffré sur un autre système.",
                "La recopie intégrale obtient un ROUGE non nul et une couverture nulle : elle "
                "contient tout, donc rien de vérifiable. C'est le piège que la couverture des faits "
                "existe pour lever.",
                "Le plancher ``lead`` n'est pas ridicule — il prend les phrases du début, qui "
                "portent souvent l'équipement et le symptôme. C'est pour cela qu'il est la "
                "référence publiée : battre un plancher faible ne prouve rien.",
            ]
        ),
        _md(
            "## 2. Ce que la stack sait produire\n\n"
            "Trois stratégies, deux familles : extractive (on sélectionne des phrases du document, "
            "donc on ne peut pas halluciner une valeur, mais on ne peut pas résumer) et apprise (on "
            "génère, donc on doit être mesuré sur sa fidélité)."
        ),
        _text(
            """
from src.models import available_algorithms, describe_algorithm

rows = []
for name in available_algorithms():
    spec = describe_algorithm(name)
    rows.append(
        {
            "algorithme": name,
            "famille": spec["family"],
            "paramètres appris": spec["requires_fit"],
            "ce qu'il fait": spec["notes"],
        }
    )
display(pd.DataFrame(rows))
""",
            context,
        ),
        _md(
            "### 2.1 Deux baselines extractives sur la même validation\n\n"
            "``lead`` prend les premières phrases, ``textrank`` sélectionne les phrases les plus "
            "centrales puis les diversifie (MMR). Toutes deux écrivent un résumé **fidèle par "
            "construction** : chaque mot vient du document."
        ),
        _text(
            """
from src.training.metrics import per_strategy_frame

SERVED_ALGORITHM = str(CONFIG.model.algorithm)
# Les stratégies comparées sont **celles de la configuration** : le notebook n'invente pas sa
# propre liste, sinon la comparaison publiée ne serait pas celle que `mode=all` reproduit.
EXTRACTIVE = [name for name in STRATEGIES if name != SERVED_ALGORITHM]
print("stratégies comparées :", ", ".join([SERVED_ALGORITHM, *EXTRACTIVE]))

SCORED = [lead_scored]
for algorithm in EXTRACTIVE:
    model = build_model(CONFIG, algorithm=algorithm)
    model.fit(TRAIN_DOCUMENTS, TRAIN_REFERENCES, validation=(VAL_DOCUMENTS, VAL_REFERENCES))
    SCORED.append(SCORER.score_model(model, VAL_DOCUMENTS, VAL_REFERENCES, VAL_FACTS))

extractives = pd.concat(SCORED, ignore_index=True)
display(per_strategy_frame(extractives).round(4))
print("compression moyenne :", round(float(extractives["compression"].mean()), 4))
""",
            context,
        ),
        _insight(
            [
                "``textrank`` sélectionne les phrases centrales mais les coupe pour tenir le budget : "
                "il produit des résumés plus courts que ``lead``, et le ROUGE le sanctionne — la "
                "couverture des faits, elle, reste comparable.",
                "Chaque baseline est ajustée sur le **train** et jugée sur la **validation** : "
                "c'est ce qui rend la comparaison lisible, et c'est la règle que suit le pipeline "
                "d'évaluation (`mode=evaluate`).",
                "Aucune des deux ne peut inventer une valeur : leur fiabilité de 1,0 est structurelle, "
                "pas une performance. Le modèle appris, lui, devra la mériter.",
            ]
        ),
        _md(
            "### 2.2 Un encodeur-décodeur entraîné ici\n\n"
            "Vocabulaire WordPiece partagé appris sur le train, pré-entraînement par débruitage, "
            "puis affinage supervisé. Le notebook ne fait qu'appeler le modèle de la stack : il n'y "
            "a pas de seconde implémentation à maintenir."
        ),
        _text(
            """
import time

from src.models.summarizer import EncoderDecoderSummarizer

learned = build_model(CONFIG, algorithm="transformer_tiny")
started = time.perf_counter()
fit_metrics = learned.fit(
    TRAIN_DOCUMENTS, TRAIN_REFERENCES, validation=(VAL_DOCUMENTS, VAL_REFERENCES)
)
duration = time.perf_counter() - started
print(f"ajustement en {duration:.1f}s :", {key: round(value, 4) for key, value in fit_metrics.items()})
print(learned.summary())
""",
            context,
        ),
        _text(
            """
transformer_scored = SCORER.score_model(learned, VAL_DOCUMENTS, VAL_REFERENCES, VAL_FACTS)
EVERYTHING = pd.concat([extractives, transformer_scored], ignore_index=True)
comparison = per_strategy_frame(EVERYTHING)
display(comparison.round(4))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
comparison.set_index("strategy")["rouge1_f"].plot.bar(ax=axes[0], rot=20,
                                                      title="ROUGE-1 par stratégie (validation)")
comparison.set_index("strategy")["fact_coverage"].plot.bar(ax=axes[1], rot=20,
                                                           title="Couverture des faits (validation)")
plt.tight_layout()
plt.show()

print("part des résumés qui atteignent leur budget (modèle appris) :",
      round(float(transformer_scored["hit_max_length"].mean()), 4))
""",
            context,
        ),
        _insight(
            [
                "Le budget d'entraînement d'un notebook est volontairement minuscule : l'objectif "
                "ici est de vérifier que **la mécanique** tourne et que les métriques répondent, "
                "pas d'atteindre le seuil contractuel, qui est mesuré par `mode=all`.",
                "La part des résumés qui atteignent leur budget se lit avec le ROUGE : un décodeur "
                "qui bute sur sa borne ne conclut pas sa phrase, et un score bas peut venir de là "
                "plutôt que du modèle.",
                "Un modèle appris qui obtient un meilleur ROUGE mais une couverture plus basse a "
                "appris à ressembler à la référence **sans** être vrai : les deux colonnes sont "
                "publiées côte à côte pour que ce cas soit visible.",
            ]
        ),
        _md(
            "## 3. Ce que le modèle regarde\n\n"
            "Un encodeur-décodeur n'est pas une boîte noire : on peut lire la répartition de son "
            "attention sur les phrases du document au premier pas de décodage, et savoir sur quels "
            "jetons son vocabulaire est aveugle."
        ),
        _text(
            """
from src.preprocessing.transformers import split_sentences

example_row = TEST_DOCUMENTS.iloc[0]
example_id, example_text = str(example_row["doc_id"]), str(example_row["text"])

if isinstance(learned, EncoderDecoderSummarizer):
    shares = learned.attention_shares(example_text)
    document_sentences = split_sentences(example_text)
    for index, share in shares.items():
        if index.isdigit() and int(index) < len(document_sentences):
            print(f"phrase {index:>2s} : {share:.3f} — {document_sentences[int(index)][:90]}")
    print()
    print("attention totale répartie sur", len(shares), "phrase(s)")

card = learned.model_card()
print("vocabulaire appris     :", card["vocabulary"]["size"], "pièces")
print("jetons inconnus (train):", card["vocabulary"]["unknown_rate"])
print("paramètres             :", card["n_parameters"])
print("décodage               :", card["architecture"]["decoding"])
""",
            context,
        ),
        _insight(
            [
                "Une part d'attention élevée sur la phrase qui porte l'équipement ou la durée est "
                "une explication vérifiable : elle se compare aux faits saillants, pas à "
                "l'intuition de l'analyste.",
                "Le taux de jetons inconnus est la mesure du vocabulaire : un vocabulaire trop "
                "petit transforme les références de pièces en suites de caractères, et le modèle "
                "ne peut plus les recopier à l'identique.",
                "Le décodage glouton est déclaré dans la fiche modèle : un résumé doit être "
                "reproductible, donc aucune température ni aucun échantillonnage n'est employé.",
            ]
        ),
    ]
    return write_notebook(destination / "04_model_exploration.ipynb", cells)


# ---------------------------------------------------------------------------------------------
# 05 — le pipeline réel, ses artefacts et sa reproductibilité
# ---------------------------------------------------------------------------------------------
def build_05_training(context: NotebookContext, destination: Path) -> Path:
    """Build ``05_training.ipynb``: pipelines réels, artefacts, rechargement, déterminisme.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    cells = [
        *_header(
            context,
            "05",
            "Entraînement — le pipeline réel, ses artefacts et sa reproductibilité",
            [
                "Exécuter les pipelines du projet (`generate-data`, `train`) dans un bac à sable "
                "qui n'écrase aucun artefact de production.",
                "Lire la mécanique d'un encodeur-décodeur : vocabulaire appris, pré-entraînement "
                "par débruitage, affinage supervisé, et ce que chaque étape change dans la perte.",
                "Recharger l'artefact et vérifier qu'il résume **exactement** comme le modèle "
                "ajusté.",
                "Prouver le déterminisme : deux ajustements à graine fixée donnent les mêmes "
                "résumés, jeton pour jeton.",
            ],
        ),
        _md(
            "## 0. Environnement et corpus\n\n"
            "Les pipelines du projet tournent ici sur le corpus réduit du notebook et écrivent dans "
            "`outputs/notebooks/` : `make train` reste intact."
        ),
        _text(SETUP, context),
        _text(LOAD_CORPUS, context),
        _text(
            """
from src.pipelines import DataGenerationPipeline, TrainPipeline

DATA_RESULT = DataGenerationPipeline(CONFIG, paths=NB_PATHS).run()
print("generate-data :", "succès" if DATA_RESULT.succeeded else "échec",
      f"en {DATA_RESULT.duration_seconds:.2f}s")
for message in DATA_RESULT.messages:
    print("  -", message)

TRAIN_RESULT = TrainPipeline(CONFIG, paths=NB_PATHS).run()
print()
print("train :", "succès" if TRAIN_RESULT.succeeded else "échec",
      f"en {TRAIN_RESULT.duration_seconds:.2f}s")
for message in TRAIN_RESULT.messages:
    print("  -", message)
assert TRAIN_RESULT.succeeded, TRAIN_RESULT.messages
OUTCOME = TRAIN_RESULT.payload
""",
            context,
        ),
        _insight(
            [
                "Le notebook appelle **les pipelines du projet**, pas une réimplémentation : ce "
                "qu'on lit ici est exactement ce que fait `python -m src.main mode=train`.",
                "Le bac à sable (`outputs/notebooks/`, données **et** artefacts) est la seule différence : un notebook ne "
                "doit jamais écraser un artefact de production, sinon la reproductibilité du run "
                "publié devient une illusion.",
                "Le pipeline refuse de tourner sans son artefact d'entrée (`mode=evaluate` sans "
                "modèle entraîné) : un rapport plausible sur un modèle vide serait pire qu'une "
                "erreur.",
            ]
        ),
        _md(
            "## 1. La mécanique du modèle\n\n"
            "Trois étapes, dans l'ordre : apprendre le vocabulaire sur le train (documents **et** "
            "résumés), pré-entraîner le décodeur à reconstruire un résumé à partir d'un document "
            "bruité, puis affiner en enseignant-forcé."
        ),
        _text(
            """
fit_metrics = {key: round(value, 4) for key, value in OUTCOME.fit_metrics.items()
               if key.startswith(("pretrain", "loss", "epochs"))}
display(pd.Series(fit_metrics, name="valeur").to_frame())

model = OUTCOME.model
card = model.model_card()
print("architecture :", card["architecture"])
print("vocabulaire  :", card["vocabulary"])
print("entraînement :", {k: v for k, v in card["training"].items() if k != "history"})
print("paramètres   :", card["n_parameters"])
""",
            context,
        ),
        _text(
            """
history = pd.DataFrame(card["training"]["history"] or [])
if not history.empty:
    display(history.round(4))
    history.plot(title="Perte par époque (pré-entraînement puis affinage)", figsize=(8, 4))
    plt.ylabel("perte")
    plt.show()
else:
    print("historique vide : le run a été trop court pour journaliser une époque")
""",
            context,
        ),
        _md(
            "## 2. Les artefacts du run\n\n"
            "Quatre fichiers répondent à quatre questions : le modèle (comment prédire), les "
            "métriques (ce que ça vaut), la fiche modèle (comment c'est fait), la configuration "
            "résolue (avec quels réglages exactement)."
        ),
        _text(
            """
from src.utils.io import read_json

artefact = NB_PATHS.models_dir / "__MODEL_FILE__"
for name in ("__MODEL_FILE__", "model_card.json", "resolved_config.json"):
    path = NB_PATHS.models_dir / name
    print(f"{name:24s} {'écrit' if path.exists() else 'ABSENT':6s} "
          f"({path.stat().st_size if path.exists() else 0} octets)")
for name in ("training_metrics.json",):
    path = NB_PATHS.metrics_dir / name
    print(f"{name:24s} {'écrit' if path.exists() else 'ABSENT':6s} "
          f"({path.stat().st_size if path.exists() else 0} octets)")
assert artefact.exists() and (NB_PATHS.models_dir / "model_card.json").exists()

resolved = read_json(NB_PATHS.models_dir / "resolved_config.json")
print("\\nalgorithme servi dans la configuration résolue :", resolved["model"]["algorithm"])
print("architecture résolue :", {k: resolved["model"]["params"][k]
                                 for k in ("layers", "units", "heads")})
""",
            context,
        ),
        _md(
            "## 3. Recharger l'artefact\n\n"
            "Un artefact qui ne prédit pas exactement comme le modèle ajusté est un artefact cassé — "
            "et cela ne se voit qu'en comparant les deux, ligne à ligne."
        ),
        _text(
            """
from src.models import load_model

reloaded = load_model(artefact, config=CONFIG.model_dump())
comparison_documents = TEST_DOCUMENTS.head(5)
before = model.summarize_frame(comparison_documents)
after = reloaded.summarize_frame(comparison_documents)
identical = bool((before["prediction"] == after["prediction"]).all())
print("résumés identiques après rechargement :", identical)
display(pd.DataFrame({"doc_id": after["doc_id"], "résumé": after["prediction"].str.slice(0, 90)}))
assert identical, "l'artefact rechargé ne résume pas comme le modèle ajusté"
""",
            context,
        ),
        _insight(
            [
                "Le rechargement est testé sur des résumés **générés**, pas sur une matrice de "
                "poids : c'est la sortie qui compte, et un décodage légèrement différent la "
                "changerait sans que personne ne le voie.",
                "Le vocabulaire fait partie de l'artefact : un vocabulaire appris hors du modèle "
                "serait un second artefact à versionner, donc une source de désynchronisation.",
                "L'architecture est écrite dans la fiche modèle : un artefact se relit **et** "
                "s'explique sans le code qui l'a produit.",
            ]
        ),
        _md(
            "## 4. Déterminisme\n\n"
            "Deux ajustements à graine fixée, sur le même split, doivent produire les mêmes "
            "résumés. Sans cette preuve, aucune comparaison entre deux runs n'est interprétable."
        ),
        _text(
            """
from src.models import build_model

first = build_model(CONFIG, algorithm="transformer_tiny")
second = build_model(CONFIG, algorithm="transformer_tiny")
first.fit(TRAIN_DOCUMENTS, TRAIN_REFERENCES)
second.fit(TRAIN_DOCUMENTS, TRAIN_REFERENCES)

sample = TEST_DOCUMENTS.head(3)
first_summaries = first.summarize_frame(sample)["prediction"].tolist()
second_summaries = second.summarize_frame(sample)["prediction"].tolist()
for index, (left, right) in enumerate(zip(first_summaries, second_summaries, strict=True), start=1):
    print(f"{index}. {'identique' if left == right else 'DIFFÉRENT'} — {left[:80]}")
assert first_summaries == second_summaries, "deux ajustements à graine fixée divergent"
print("\\ndéterminisme vérifié sur", len(sample), "résumés")
""",
            context,
        ),
        _insight(
            [
                "La graine est posée sur les trois générateurs (`random`, `numpy`, `torch`) **et** le "
                "nombre de fils d'exécution est fixé à un : les réductions flottantes changent "
                "l'ordre des additions, donc les derniers chiffres d'une perte.",
                "Le décodage est glouton : aucune température, aucun échantillonnage. Un résumé "
                "reproductible est la condition pour qu'un rapport d'évaluation soit vérifiable.",
                "La comparaison porte sur le texte produit, pas sur la perte : deux pertes égales à "
                "la troisième décimale peuvent donner deux résumés différents après un `argmax`.",
            ]
        ),
    ]
    return write_notebook(destination / "05_training.ipynb", cells)


# ---------------------------------------------------------------------------------------------
# 06 — verdict, fidélité, erreurs et recommandations
# ---------------------------------------------------------------------------------------------
def build_06_error_analysis(context: NotebookContext, destination: Path) -> Path:
    """Build ``06_error_analysis.ipynb``: verdict, fidélité, erreurs et recommandations.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    cells = [
        *_header(
            context,
            "06",
            "Analyse d'erreurs, fidélité et recommandations",
            [
                "Lire le verdict contractuel sur le split de **test**, une seule fois, avec sa "
                "marge et ses références mesurées sur les mêmes lignes.",
                "Décomposer la fidélité par type de fait : la moyenne cache toujours le type qui "
                "manque.",
                "Relire les erreurs une par une — fait oublié, valeur inventée, phrase qui bute sur "
                "sa borne — et distinguer les trois familles de correctifs.",
                "Écrire le rapport du projet (tableaux et figures compris) et formuler des "
                "recommandations attachées à un chiffre.",
            ],
        ),
        _md(
            "## 0. Environnement et corpus\n\n"
            "Le notebook ajuste lui-même les stratégies (baseline extractive et modèle de "
            "démonstration) : il n'a besoin d'aucun artefact de `make train` pour tourner, et il "
            "écrit ses sorties dans `outputs/notebooks/`."
        ),
        _text(SETUP, context),
        _text(LOAD_CORPUS, context),
        _md(
            "## 1. Mesurer le test, une fois\n\n"
            "L'évaluateur reçoit le modèle servi **et** ses références. Toutes les stratégies sont "
            "mesurées sur les mêmes lignes de test, avec la même référence : une comparaison entre "
            "deux exécutions différentes ne voudrait rien dire."
        ),
        _text(
            """
from src.evaluation.evaluator import SummaryEvaluator
from src.models import build_model

SERVED = build_model(CONFIG, algorithm="transformer_tiny")
SERVED.fit(TRAIN_DOCUMENTS, TRAIN_REFERENCES, validation=(VAL_DOCUMENTS, VAL_REFERENCES))
baseline = build_model(CONFIG, algorithm="__BASELINE__")
baseline.fit(TRAIN_DOCUMENTS, TRAIN_REFERENCES)

evaluator = SummaryEvaluator(
    SERVED,
    config=CONFIG.model_dump(),
    metrics_config=CONFIG.metrics.model_dump(),
    paths=NB_PATHS,
    text_column=TEXT_COLUMN,
    id_column=ID_COLUMN,
    baselines={"baseline_extractive": baseline},
)
EVALUATION = evaluator.evaluate(TEST_DOCUMENTS, TEST_REFERENCES, TEST_FACTS)
print("verdict :", EVALUATION.verdict)
print("détail  :", EVALUATION.verdict_detail)
display(EVALUATION.per_strategy.round(4))
""",
            context,
        ),
        _insight(
            [
                "Le verdict est lu sur la métrique principale déclarée par la configuration, avec "
                "son seuil : un rapport qui annonce un score sans dire à quoi il se compare n'est "
                "pas un verdict.",
                "La marge (observé moins seuil) est publiable telle quelle : elle dit si le système "
                "est juste au-dessus ou largement au-dessus, ce qui n'est pas la même information "
                "pour décider d'un déploiement.",
                "Les références sont mesurées **sur les mêmes lignes** : comparer le modèle au "
                "score publié dans un autre contexte serait une comparaison invalide.",
            ]
        ),
        _md(
            "## 2. La fidélité, type de fait par type de fait\n\n"
            "La couverture résume en un chiffre ce que le résumé rapporte des faits saillants. Un "
            "type oublié reste invisible dans la moyenne : il faut donc la ventilation, et les "
            "valeurs non supportées — celles que le résumé **invente**."
        ),
        _text(
            """
from src.data.schemas import fact_columns

fidelity = EVALUATION.fidelity
coverage_columns = [column for column in fact_columns() if column in fidelity.columns]
by_type = (
    fidelity[coverage_columns]
    .mean()
    .rename(lambda name: name.replace("covered_", ""))
    .sort_values()
)
display(by_type.round(4).to_frame("couverture moyenne"))

print("couverture globale   :", round(float(fidelity["fact_coverage"].mean()), 4))
print("précision des faits  :", round(float(fidelity["fact_precision"].mean()), 4))
print("valeurs non supportées (total) :", int(fidelity["n_unsupported"].sum()))
print("résumés contenant au moins une valeur inventée :",
      int((fidelity["n_unsupported"] > 0).sum()), "/", len(fidelity))

# La couverture se lit **contre** celle des stratégies extractives, mesurée sur les mêmes lignes :
# une extractive ne peut pas inventer, donc son score est le plancher honnête de cette colonne.
extractives = EVALUATION.per_strategy.loc[EVALUATION.per_strategy["strategy"] != SERVED.strategy]
if not extractives.empty and "fact_coverage" in extractives.columns:
    best = extractives.sort_values("fact_coverage", ascending=False).iloc[0]
    print(f"meilleure couverture extractive : {float(best['fact_coverage']):.4f} ({best['strategy']}) "
          f"| modèle servi : {float(fidelity['fact_coverage'].mean()):.4f}")
""",
            context,
        ),
        _text(
            """
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
by_type.plot.bar(ax=axes[0], rot=30, title="Couverture par type de fait (test)")
fidelity["fact_coverage"].plot.hist(ax=axes[1], bins=20, title="Couverture par document")
plt.tight_layout()
plt.show()
""",
            context,
        ),
        _insight(
            [
                "Un type de fait peu couvert est un correctif ciblé : soit il est rare dans le "
                "train (peu d'exemples à apprendre), soit il n'est pas reformulé de la même façon "
                "que dans la référence.",
                "La précision des faits et la couverture se lisent ensemble : couvrir beaucoup en "
                "inventant n'est pas une amélioration, c'est un déplacement du problème.",
                "L'annotation ne couvre que les valeurs **vérifiables** (équipement, durée, "
                "statut…) : la détection d'hallucination s'arrête là où l'annotation s'arrête, et "
                "le rapport le dit.",
            ]
        ),
        _md(
            "## 3. Où le modèle échoue\n\n"
            "Les segments sont les colonnes du corpus (type d'intervention, urgence, site) : ils "
            "répondent à la question « pour qui le modèle est-il moins bon ? », qui précède "
            "« pourquoi ? »."
        ),
        _text(
            """
segments = EVALUATION.segments
for column in sorted(segments["segment"].unique()) if not segments.empty else []:
    pivot = segments[segments["segment"] == column].sort_values("rouge1_f")
    display(pivot[["value", "n_documents", "rouge1_f", "fact_coverage", "compression"]].round(4))
""",
            context,
        ),
        _text(
            """
if not segments.empty:
    fig, axes = plt.subplots(1, len(sorted(segments["segment"].unique())), figsize=(14, 4), squeeze=False)
    for axis, column in zip(axes[0], sorted(segments["segment"].unique()), strict=False):
        subset = segments[segments["segment"] == column].sort_values("rouge1_f")
        axis.bar(subset["value"], subset["rouge1_f"])
        axis.set_title(f"ROUGE-1 par {column}")
        axis.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.show()
else:
    print("aucun segment déclaré par la configuration")
""",
            context,
        ),
        _md(
            "## 4. Les erreurs, relues une par une\n\n"
            "Le rapport archive les documents où le modèle est le plus loin de sa référence, avec "
            "les faits saillants qu'il aurait fallu rapporter. C'est là qu'on distingue un **fait "
            "oublié** (couverture basse, ROUGE correct) d'une **valeur inventée** (précision basse) "
            "et d'une **phrase qui bute sur sa borne**."
        ),
        _text(
            """
errors = EVALUATION.errors
print("documents archivés :", len(errors))
display(errors.head(5)[[c for c in errors.columns if c in
                       ("doc_id", "rouge1_f", "fact_coverage", "unsupported_facts", "hit_max_length")]])

worst = errors.iloc[0]
worst_id = str(worst["doc_id"])
worst_reference = TEST_REFERENCES.loc[TEST_REFERENCES["doc_id"] == worst_id, "summary"].iloc[0]
worst_fidelity = fidelity.loc[fidelity["doc_id"] == worst_id]

print(f"document {worst_id} : ROUGE-1 {float(worst['rouge1_f']):.4f}, "
      f"couverture {float(worst['fact_coverage']):.4f}")
print("faits saillants du document :")
display(
    TEST_FACTS[(TEST_FACTS["doc_id"] == worst_id) & (TEST_FACTS["salient"] == 1)]
    [["fact_type", "value", "sentence_index"]]
)
print("résumé de référence :", worst_reference[:400])
print("faits rapportés par le modèle :", int(worst_fidelity["n_covered"].sum()),
      "/", int(worst_fidelity["n_salient"].sum()))
if "unsupported_values" in worst_fidelity.columns:
    print("valeurs inventées :", str(worst_fidelity["unsupported_values"].iloc[0]) or "aucune")
""",
            context,
        ),
        _text(
            """
summary_of_worst = (
    EVALUATION.predictions.loc[
        (EVALUATION.predictions["doc_id"] == worst_id)
        & (EVALUATION.predictions["strategy"] == SERVED.strategy)
    ]["prediction"]
)
print("résumé produit par le modèle :", "" if summary_of_worst.empty else summary_of_worst.iloc[0][:400])
print("\\npart des résumés qui atteignent leur budget de longueur (test) :",
      float(EVALUATION.predictions.loc[
          EVALUATION.predictions["strategy"] == SERVED.strategy, "hit_max_length"].mean()))
print("compression moyenne du modèle  :",
      round(float(EVALUATION.predictions.loc[
          EVALUATION.predictions["strategy"] == SERVED.strategy, "compression"].mean()), 4))
print("longueur moyenne de la référence :",
      round(float(TEST_REFERENCES["compression"].mean()), 4), "(compression)")
""",
            context,
        ),
        _insight(
            [
                "Trois familles d'erreurs, trois correctifs : un fait **oublié** se corrige par des "
                "données ou du budget ; une valeur **inventée** par une contrainte de fidélité ; "
                "une phrase **tronquée** par le budget ou par le critère d'arrêt du décodeur.",
                "Un ROUGE bas avec une couverture correcte signale une reformulation différente de "
                "la référence, pas une contre-vérité : le corpus accepte plusieurs formulations, "
                "la référence n'en connaît qu'une.",
                "Relire l'erreur « la pire » est ce qui empêche de conclure sur une moyenne : un "
                "exemple suffit souvent à identifier la cause, et il se cite.",
            ]
        ),
        _md(
            "## 5. Le rapport du projet\n\n"
            "Le constructeur de rapport transforme l'évaluation en artefacts : un rapport Markdown "
            "lisible, des tableaux CSV pour un portail de CI et les figures. Ce sont les mêmes "
            "fichiers que `make evaluate` écrit."
        ),
        _text(
            """
from src.evaluation.reports import ReportBuilder

bundle = ReportBuilder(CONFIG.model_dump(), paths=NB_PATHS).build(EVALUATION)
print("rapport  :", bundle.report.relative_to(PROJECT_ROOT))
for name, path in sorted(bundle.tables.items()):
    print(f"  tableau  : {name:20s} {path.relative_to(PROJECT_ROOT)}")
for name, path in sorted(bundle.figures.items()):
    print(f"  figure   : {name:20s} {path.relative_to(PROJECT_ROOT)}")

print("\\n" + "=" * 90)
print(bundle.report.read_text(encoding="utf-8")[:1500])
""",
            context,
        ),
        _md(
            "## 6. Recommandations\n\n"
            "Chaque recommandation est attachée au chiffre qui la justifie. Une recommandation sans "
            "chiffre est une opinion, et ce notebook n'en publie pas."
        ),
        _text(
            """
observed = float(EVALUATION.verdict_detail.get("observed", 0.0))
threshold = EVALUATION.verdict_detail.get("threshold")
weakest = by_type.index[0]
recommendations = [
    f"1. Le seuil contractuel ({threshold}) n'est pas atteint sur ce run de démonstration "
    f"({observed:.4f}) : `mode=all` entraîne le modèle de référence et publie le verdict réel.",
    f"2. Le type de fait le plus faible est « {weakest} » ({by_type.iloc[0]:.4f} de couverture) : "
    "c'est là qu'un ajustement du corpus ou du budget de longueur a le plus de rendement.",
]
if int(fidelity["n_unsupported"].sum()) > 0:
    recommendations.append(
        f"3. {int(fidelity['n_unsupported'].sum())} valeur(s) inventée(s) détectée(s) : renforcer "
        "la contrainte de fidélité avant d'augmenter la capacité du modèle."
    )
else:
    recommendations.append(
        "3. Aucune valeur inventée détectée sur ce run : la fidélité mesurable est tenue, la "
        "marge de progression est dans la couverture."
    )
served_predictions = EVALUATION.predictions.loc[
    EVALUATION.predictions["strategy"] == SERVED.strategy
]
budget_share = float(served_predictions["hit_max_length"].mean())
recommendations.append(
    f"4. Budget de longueur : {budget_share:.1%} des résumés atteignent leur borne, contre une "
    f"compression de référence de {float(TEST_REFERENCES['compression'].mean()):.4f} : vérifier "
    "le critère d'arrêt du décodeur avant d'accuser le modèle."
)
if not extractives.empty and "fact_coverage" in extractives.columns:
    recommendations.append(
        f"5. Couverture des faits : {float(fidelity['fact_coverage'].mean()):.4f} pour le modèle "
        f"servi contre {float(extractives['fact_coverage'].max()):.4f} pour la meilleure "
        "extractive, sur les mêmes lignes — un extracteur recopie les valeurs par construction, le "
        "démonstrateur réduit doit encore apprendre à le faire."
    )
for line in recommendations:
    print(line)
""",
            context,
        ),
        _insight(
            [
                "Un verdict « non conforme » n'est pas un échec du projet : c'est le résultat d'une "
                "mesure, publié avec sa marge. Le run de référence (`mode=all`), lui, entraîne le "
                "modèle complet et affiche son propre verdict.",
                "Les recommandations sont ordonnées par rendement : d'abord le type de fait le plus "
                "faible, ensuite la fidélité, enfin le budget — jamais « améliorer le modèle » sans "
                "chiffre.",
                "Tous les artefacts écrits ici sont dans `outputs/notebooks/` : le rapport publié "
                "par le projet reste celui de `make evaluate`, et les deux ne se mélangent pas.",
            ]
        ),
    ]
    return write_notebook(destination / "06_error_analysis.ipynb", cells)


# ---------------------------------------------------------------------------------------------
# assemblage
# ---------------------------------------------------------------------------------------------
BUILDERS = (
    build_01_eda,
    build_02_validation,
    build_03_preprocessing,
    build_04_model_exploration,
    build_05_training,
    build_06_error_analysis,
)


def build_all(context: NotebookContext, destination: Path) -> list[Path]:
    """Build the six notebooks of a summarization project.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory of the project.

    Returns:
        The written notebook paths, in order.
    """
    return [builder(context, destination) for builder in BUILDERS]
