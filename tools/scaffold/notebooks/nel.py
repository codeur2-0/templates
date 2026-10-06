"""Notebooks pédagogiques des projets de **reconnaissance d'entités nommées** (famille `nel`).

Un corpus annoté au niveau du **caractère** ne s'explore pas comme une classification de documents :
la question n'est pas « de quel type est ce message » mais « où commence et où finit chaque
mention ». Les six notebooks suivent la progression commune aux autres familles, avec les questions
propres à l'extraction :

* ``01_eda.ipynb`` — cartographier un corpus annoté (mentions par type, offsets, styles, splits) et
  mesurer ce que la **forme seule** explique déjà : une référence de commande, un montant ou une
  date sont des **motifs**, un produit ou un transporteur sont des **noms**. C'est la carte qui dit
  ce qu'une F1 macro vaudra ;
* ``02_validation.ipynb`` — les contrats Pandera des **deux** tables (messages et annotations), le
  lien entre elles (``surface == text[start:end]``, pas de chevauchement, appartenance des mentions
  au split de leur message) et ce qui se passe quand on les casse ;
* ``03_preprocessing.ipynb`` — les traits de forme calculés depuis la surface, la séparabilité des
  signatures, la tokénisation spaCy et l'**alignement** des offsets sur les tokens, puis l'index de
  règles appris sur le train : ce qu'il sait, et la surface réservée qu'il ne peut pas connaître ;
* ``04_model_exploration.ipynb`` — les trois algorithmes de la stack (règles, tagger, hybride)
  comparés sur la validation, l'architecture réellement construite (lue sur le pipeline, pas
  redéclarée), et la lecture de la **confiance comme corroboration** avec un seuil d'automatisation ;
* ``05_training.ipynb`` — le pipeline d'entraînement réel dans un bac à sable, ses artefacts
  (répertoire spaCy, fiche modèle, métriques, configuration résolue), le rechargement à l'identique
  et la preuve de déterminisme ;
* ``06_error_analysis.ipynb`` — le verdict contractuel sur le test, la ventilation par style et par
  canal, le comparatif **surface réservée / surface vue**, les trois familles d'erreurs expliquées
  (inventée, bornes, manquée), les figures et le rapport écrit par le projet.

Les helpers de rendu (``_md``, ``_insight``, ``_objectives``) sont partagés avec les notebooks
tabulaire et texte : une extraction d'entités parle la même langue typographique qu'une régression,
sans en recopier le contenu.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tools.scaffold.notebooks.tabular import _insight, _md, _objectives
from tools.scaffold.utils_notebooks import NotebookNode, code_cell, write_notebook

if TYPE_CHECKING:  # pragma: no cover
    from tools.scaffold.utils_notebooks import NotebookContext

__all__ = ["build_all"]

#: Taille du corpus réduit des notebooks : l'exécution reste rapide tout en laissant plusieurs
#: centaines de mentions par type — assez pour qu'une F1 veuille dire quelque chose.
NB_DOCUMENTS = 320

#: Budget d'époques des notebooks. Le manifeste entraîne le modèle servi à son propre budget
#: (trente époques sur 1 200 messages, publié dans le README) ; un notebook doit rester exécutable
#: sur un clone frais, donc il travaille à budget réduit — et le dit.
NB_EPOCHS = 3

#: Budget de la grille de comparaison des trois algorithmes (notebook 04).
COMPARISON_EPOCHS = 3

#: Seuils de confiance comparés par le notebook 04 : à partir de quel niveau automatiser ?
CONFIDENCE_GRID = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)

#: Précision visée sur les mentions validées automatiquement (la cible de service).
TARGET_AUTOMATION_PRECISION = 0.95


def _measured_results_source(context: NotebookContext) -> str:
    """Render the manifest's measured results as a Python list literal, one row per line.

    Une seule ligne pour plusieurs mesures dépasserait la limite de 100 colonnes du projet : le
    littéral est donc écrit mesure par mesure, exactement comme un développeur l'écrirait à la
    main, et reste indenté à l'intérieur de la cellule.

    Args:
        context: Notebook context.

    Returns:
        The ``REFERENCE_RESULTS`` literal (``[]`` when the manifest declares no measurement).
    """
    rows = [(str(item.label), str(item.value)) for item in context.spec.business.measured_results]
    if not rows:
        return "[]"
    body = "".join(f"    ({label!r}, {value!r}),\n" for label, value in rows)
    return f"[\n{body}]"


def _tokens(context: NotebookContext) -> dict[str, str]:
    """Build the substitution table of the extraction notebooks.

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
        "__NB_EPOCHS__": str(NB_EPOCHS),
        "__PRIMARY__": str(spec.metrics.primary),
        "__MIN_PRIMARY__": str(spec.metrics.min_primary),
        "__TASK__": str(spec.metrics.task),
        "__REPORT_NAME__": str(artifacts.get("report_file", "evaluation_report.md")),
        "__MODEL_FILE__": str(artifacts.get("model_file", "tagger")),
        "__MEASURED_RESULTS__": _measured_results_source(context),
        "__CONFIDENCE_GRID__": repr(list(CONFIDENCE_GRID)),
        "__TARGET_AUTOMATION__": str(TARGET_AUTOMATION_PRECISION),
        # Le budget d'époques n'a de sens que pour une stack entraînée par époques : une stack à
        # passage unique (règles, TF-IDF) reçoit un dictionnaire vide plutôt qu'un paramètre
        # qu'elle ne comprend pas.
        "__COMPARISON_PARAMS__": (
            f'{{"epochs": {COMPARISON_EPOCHS}}}' if context.stack.epochs_based else "{}"
        ),
    }


def _text(source: str, context: NotebookContext) -> NotebookNode:
    """Render a code cell with the extraction tokens substituted.

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


SETUP = """
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display

# --- Racine du projet ---------------------------------------------------------------------------
# Le notebook s'exécute depuis `notebooks/` : on remonte d'un cran pour pouvoir importer `src`.
PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

from src.data.schemas import CANAUX, ENTITY_LABELS, NAME_LABELS, SOURCES, SPLITS, STYLES  # noqa: E402
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

# --- Configuration : exactement celle de `python -m src.main` ------------------------------------
# Les notebooks travaillent sur un corpus réduit (__NB_DOCUMENTS__ messages) et un budget d'époques
# réduit : l'exécution reste rapide, et les valeurs absolues d'un notebook ne sont pas celles de la
# référence publiée par le README.
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
                "seed=__SEED__",
                "log_level=WARNING",
                "train.callbacks.progress_bar=false",
                "train.callbacks.logging_every=100",
            ],
        )
    )

MODEL_NODE = dict(node(CONFIG, "model"))
TEXT_COLUMN = str(MODEL_NODE.get("text_column", "text"))
TARGET_COLUMN = str(MODEL_NODE.get("target", "label"))
ID_COLUMN = str(CONFIG.data.id_column or "msg_id")
SPLIT_COLUMN = str(CONFIG.data.group_column or "split")

# Les notebooks lisent **et écrivent** dans leur propre bac à sable : le corpus réduit et les
# artefacts des notebooks ne touchent ni `data/raw` ni `artifacts/`, donc ceux de `make train`
# restent ceux de la référence.
NB_PATHS = ProjectPaths(
    root=PROJECT_ROOT,
    data_dir=PROJECT_ROOT / "outputs" / "notebooks" / "data",
    artifacts_dir=PROJECT_ROOT / "outputs" / "notebooks" / "artifacts",
).ensure()

print(f"Projet            : {CONFIG.project.name}")
print(f"Tâche             : {CONFIG.metrics.task}")
print(f"Métrique primaire : {CONFIG.metrics.primary} (seuil contractuel : __MIN_PRIMARY__)")
print(f"Algorithme servi  : {CONFIG.model.algorithm} ({CONFIG.model.name})")
print(f"Types d'entités   : {len(ENTITY_LABELS)} — {', '.join(ENTITY_LABELS)}")
print(f"Bac à sable       : {NB_PATHS.artifacts_dir.relative_to(PROJECT_ROOT)}")
"""


LOAD_CORPUS = """
from src.data.generators import SyntheticEntityCorpusGenerator
from src.data.loaders import EntityCorpusLoader

LOADER = EntityCorpusLoader(
    NB_PATHS,
    dataset_name=str(CONFIG.data.dataset_name),
    formats=tuple(CONFIG.data.formats),
    validation_enabled=bool(CONFIG.data.validation.raw),
    lazy_validation=bool(CONFIG.data.validation.lazy),
)

if LOADER.documents_path.exists():
    DOCUMENTS, SPANS = LOADER.load_corpus()
    METADATA = LOADER.load_metadata()
    print(f"Corpus lu depuis le bac à sable : {len(DOCUMENTS)} messages, graine {METADATA['seed']}")
else:
    # Un clone frais n'a pas encore de données : le notebook reste exécutable. Le corpus réduit est
    # persisté **dans le bac à sable** pour que les notebooks suivants et les pipelines travaillent
    # sur exactement les mêmes fichiers.
    bundle = SyntheticEntityCorpusGenerator.from_config(CONFIG.data).generate()
    DOCUMENTS, SPANS, METADATA = bundle.documents, bundle.queries, dict(bundle.metadata)
    LOADER.save_documents(DOCUMENTS)
    LOADER.save_annotations(SPANS)
    LOADER.save_metadata(METADATA)
    print(f"Bac à sable vide : corpus de {len(DOCUMENTS)} messages généré et persisté")

DOCUMENTS = DOCUMENTS.sort_values(ID_COLUMN).reset_index(drop=True)
SPANS = SPANS.sort_values([ID_COLUMN, "start", "end"]).reset_index(drop=True)

TRAIN_DOCUMENTS = LOADER.split("train", frame=DOCUMENTS)
VAL_DOCUMENTS = LOADER.split("val", frame=DOCUMENTS)
CALIBRATION_DOCUMENTS = LOADER.split("calibration", frame=DOCUMENTS)
TEST_DOCUMENTS = LOADER.split("test", frame=DOCUMENTS)
TRAIN_SPANS = LOADER.split_annotations("train", documents=DOCUMENTS, spans=SPANS)
VAL_SPANS = LOADER.split_annotations("val", documents=DOCUMENTS, spans=SPANS)
TEST_SPANS = LOADER.split_annotations("test", documents=DOCUMENTS, spans=SPANS)

print("messages par split :", {name: len(frame) for name, frame in (
    ("train", TRAIN_DOCUMENTS), ("val", VAL_DOCUMENTS),
    ("calibration", CALIBRATION_DOCUMENTS), ("test", TEST_DOCUMENTS))})
print("mentions par split :", {name: len(frame) for name, frame in (
    ("train", TRAIN_SPANS), ("val", VAL_SPANS), ("test", TEST_SPANS))})
"""


# ---------------------------------------------------------------------------------------------
# 01 — exploration du corpus
# ---------------------------------------------------------------------------------------------
def build_01_eda(context: NotebookContext, destination: Path) -> Path:
    """Build ``01_eda.ipynb``: cartographie du corpus annoté et des raccourcis de forme.

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
            "Explorer un corpus annoté au niveau du caractère",
            [
                "Lire les **deux** tables d'un projet d'extraction : les messages, et les mentions "
                "annotées dans ces messages.",
                "Mesurer ce qu'une **règle de forme** retrouverait déjà, type par type : le "
                "raccourci que la F1 micro cache.",
                "Repérer les **surfaces réservées** à l'évaluation et les distracteurs déclarés.",
                "Comparer le corpus des notebooks à la référence publiée par le projet.",
            ],
        ),
        _md("## 1. Mise en place"),
        _text(SETUP, context),
        _text(LOAD_CORPUS, context),
        _md(
            "## 2. Le corpus en chiffres\n\n"
            "Un corpus de reconnaissance d'entités se décrit par ses **mentions**, pas par ses "
            "lignes : le nombre de messages ne dit rien de la difficulté, la répartition des types "
            "et la longueur des mentions, oui."
        ),
        _text(
            """
per_message = SPANS.groupby("msg_id").size()
print(f"messages          : {len(DOCUMENTS)}")
print(f"mentions annotées : {len(SPANS)}")
print(f"par message       : {per_message.mean():.2f} en moyenne "
      f"(min {per_message.min()}, max {per_message.max()})")
print()
print("métadonnées de génération — tailles :", METADATA["entity_counts"])
print("styles :", METADATA["style_shares"], "| canaux :", METADATA["canal_shares"])
""",
            context,
        ),
        _text(
            """
from src.data.schemas import describe_labels, holdout_share, label_counts

comptes = label_counts(SPANS)
reserves = holdout_share(SPANS)
carte_types = pd.DataFrame(
    {
        "type": list(ENTITY_LABELS),
        "mentions": [comptes[label] for label in ENTITY_LABELS],
        "part du corpus": [round(comptes[label] / len(SPANS), 4) for label in ENTITY_LABELS],
        "part réservée": [round(reserves[label], 4) for label in ENTITY_LABELS],
        "famille": ["nom" if label in NAME_LABELS else "motif" for label in ENTITY_LABELS],
        "définition": [describe_labels()[label] for label in ENTITY_LABELS],
    }
)
display(carte_types)
print(f"part réservée au total : {SPANS['holdout'].mean():.2%} des mentions")
""",
            context,
        ),
        _insight(
            [
                "Les types sont déséquilibrés : la F1 **micro** les noie, la F1 **macro** les compte "
                "à égalité — les deux sont publiées pour cette raison.",
                "La colonne « part réservée » est la difficulté du type : une surface de produit "
                "jamais vue à l'entraînement ne peut pas être retrouvée par une liste apprise.",
                "Trois types s'écrivent selon un motif (commande, montant, date), deux reposent sur "
                "un nom (produit, transporteur). Tout le reste du projet découle de cette "
                "distinction.",
            ]
        ),
        _md(
            "## 3. Ce que la forme seule explique déjà\n\n"
            "Avant de juger un modèle, il faut savoir ce qu'une règle d'une ligne obtiendrait. La "
            "fonction `shape_shortcut_scores` apprend, **sur le train**, la signature de forme "
            "dominante de chaque type, puis classe avec sur la validation — une forme jamais vue "
            "n'est reconnue par personne."
        ),
        _text(
            """
from src.features.build_features import shape_separability, shape_shortcut_scores

raccourci = shape_shortcut_scores(DOCUMENTS, SPANS, split="val").set_index("label")
display(
    raccourci[["n_spans", "n_signatures", "n_unseen", "signature", "share_by_shape",
               "precision", "recall", "f1"]]
    .rename(columns={"n_spans": "mentions (val)", "n_signatures": "formes distinctes",
                     "n_unseen": "formes inédites", "signature": "forme dominante (train)",
                     "share_by_shape": "part classée"})
    .round(4)
)
""",
            context,
        ),
        _text(
            """
separabilite = shape_separability(DOCUMENTS, SPANS).sort_values("n_spans", ascending=False)
ambiguës = separabilite[separabilite["n_labels"] > 1]
print(f"signatures de forme observées : {len(separabilite)}")
print(f"signatures partagées par plusieurs types : {len(ambiguës)}")
display(ambiguës.head(10).round(4))
""",
            context,
        ),
        _insight(
            [
                "La règle de forme atteint 0,93 à 1,00 de F1 sur **chaque** type de la "
                "validation, dès que la forme a été vue : sur ce corpus, une signature ne porte "
                "qu'un seul type (`n_labels` = 1 partout) — la forme *nomme* une mention déjà "
                "bornée, elle ne la trouve pas.",
                "`n_unseen` compte les mentions de validation dont la signature est absente du "
                "train (ici le produit) : c'est exactement ce que la surface réservée provoque, et "
                "ce qu'aucune liste de surfaces vue à l'entraînement ne peut rattraper.",
                "La difficulté du projet n'est donc pas de choisir entre deux types sur la même "
                "forme, mais de **trouver** les bornes des mentions et de reconnaître des formes "
                "inédites.",
            ]
        ),
        _md(
            "## 4. Distracteurs déclarés et plancher trivial\n\n"
            "Un corpus pédagogique doit **déclarer** ce qui y ressemble à une entité sans en être "
            "une. Villes, numéros de facture et familles de produits apparaissent dans les textes "
            "et ne sont jamais annotés : un modèle qui les annote perd de la précision."
        ),
        _text(
            """
villes = list(METADATA["distractors"]["cities"])
presentes = [ville for ville in villes if DOCUMENTS[TEXT_COLUMN].str.contains(ville, regex=False).any()]
annotees = sorted(set(SPANS["surface"]) & set(villes))
print("villes présentes dans les textes :", presentes)
print("villes annotées comme entités   :", annotees or "aucune (déclarées distrayantes)")
print("faux positifs que cela promet   :", len(presentes), "pièges à ne pas récompenser")
print()
print("plancher trivial :", METADATA["trivial_floor"])
""",
            context,
        ),
        _text(
            """
from src.visualization.plots import plot_entity_counts, plot_lengths

DISTRIBUTION = LOADER.entity_distribution()
figure_types = plot_entity_counts(DISTRIBUTION, NB_PATHS.figures_dir / "eda_entity_counts.png")
figure_longueurs = plot_lengths(DOCUMENTS, NB_PATHS.figures_dir / "eda_lengths.png")
display(Image(filename=str(figure_types)))
display(Image(filename=str(figure_longueurs)))
""",
            context,
        ),
        _md(
            "## 5. Ce que la référence du projet a mesuré\n\n"
            "Le README publie les chiffres du **run de référence** (corpus complet, budget "
            "d'époques du manifeste). Les valeurs de ce notebook sont celles du corpus réduit : "
            "elles servent à comprendre, pas à citer."
        ),
        _text(
            """
REFERENCE_RESULTS = __MEASURED_RESULTS__
if REFERENCE_RESULTS:
    display(pd.DataFrame(REFERENCE_RESULTS, columns=["mesure", "valeur"]))
else:
    print("Aucune mesure publiée dans le manifeste : voir la section du README du projet.")
print()
print(f"corpus du notebook : {len(DOCUMENTS)} messages, {len(SPANS)} mentions, "
      f"budget d'entraînement {NB_EPOCHS} époque(s) par défaut")
print(f"métrique principale visée : {CONFIG.metrics.primary} >= {CONFIG.metrics.min_primary}")
""",
            context,
        ),
        _insight(
            [
                "Deux tables, un contrat : les annotations ne vivent pas dans le texte, elles "
                "pointent dessus par des offsets — c'est ce que le notebook 02 vérifie.",
                "Un corpus synthétique n'est pas un corpus facile : les surfaces réservées et les "
                "distracteurs sont là pour que les scores veuillent dire quelque chose.",
                "Le plancher trivial (0,0) est publié pour que tout score se lise par rapport à "
                "lui, jamais dans l'absolu.",
            ]
        ),
        _md(
            "## Synthèse\n\n"
            "| Question | Réponse lue ici | Suite |\n"
            "| --- | --- | --- |\n"
            "| Combien de mentions, et de quel type ? | `label_counts`, `entity_distribution` | "
            "notebook 06 (F1 par type) |\n"
            "| Quelle part de chaque type est un **motif** ? | `shape_shortcut_scores` | "
            "notebook 03 (traits de forme) |\n"
            "| Quelle part est réservée à l'évaluation ? | `holdout_share` | notebook 06 "
            "(écart surface réservée / vue) |\n"
            "| Les tables sont-elles conformes ? | — | `02_validation.ipynb` |\n\n"
            "**Suite** : `02_validation.ipynb` fait échouer les contrats pour montrer ce qu'ils "
            "protègent."
        ),
    ]
    return _write(destination, "01_eda", cells)


# ---------------------------------------------------------------------------------------------
# 02 — validation
# ---------------------------------------------------------------------------------------------
def build_02_validation(context: NotebookContext, destination: Path) -> Path:
    """Build ``02_validation.ipynb``: contrats Pandera des deux tables et de l'inférence.

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
            "Contrats de données — les messages, les annotations, et le lien entre eux",
            [
                "Lire les schémas comme des **contrats exécutables** : colonnes, types, domaines de "
                "valeurs.",
                "Casser chaque contrat volontairement : un schéma qu'on n'a jamais vu échouer ne "
                "protège rien.",
                "Vérifier le **lien** entre les deux tables : `surface == text[start:end]`, pas de "
                "chevauchement, splits alignés.",
                "Valider la table de prédictions, entrée de la chaîne d'inférence.",
            ],
        ),
        _md("## 1. Mise en place"),
        _text(SETUP, context),
        _text(LOAD_CORPUS, context),
        _md(
            "## 2. Les schémas, colonne par colonne\n\n"
            "Un schéma Pandera est un objet que l'on peut **afficher** : c'est ce qui le rend "
            "auditable. Les descriptions viennent du code du projet, pas d'un document à part."
        ),
        _text(
            """
from pandera.errors import SchemaError, SchemaErrors

from src.data.schemas import MessagesSchema, SpansSchema, validate_messages, validate_spans


def contract_table(schema: type) -> pd.DataFrame:
    \"\"\"Render a Pandera schema as the table a reader can audit.\"\"\"
    columns = schema.to_schema().columns
    return pd.DataFrame(
        {
            "colonne": list(columns),
            "type": [str(column.dtype) for column in columns.values()],
            "contraintes": [
                ", ".join(check.name for check in column.checks) or "—"
                for column in columns.values()
            ],
            "rôle": [str(column.description or "—") for column in columns.values()],
        }
    )


display(contract_table(MessagesSchema))
display(contract_table(SpansSchema))
""",
            context,
        ),
        _insight(
            [
                "Le lien entre les tables n'est pas une convention : `msg_id` est la clé de "
                "jointure, et son format (`MSG-0001`) est contractuel.",
                "`split` est une **colonne**, pas un paramètre d'entraînement : c'est ce qui rend "
                "deux exécutions comparables.",
                "Le drapeau `holdout` est publié avec les données : la dégradation sur les surfaces "
                "réservées se mesure donc au lieu de se commenter.",
            ]
        ),
        _md(
            "## 3. Le contrat des messages\n\n"
            "Six colonnes suffisent à décrire une ligne : l'identifiant, le texte, le canal, le "
            "style, deux compteurs et la date. `strict = True` signifie qu'une colonne non déclarée "
            "est une **erreur**, pas un détail."
        ),
        _text(
            """
VALIDATED_MESSAGES = validate_messages(DOCUMENTS)
print(f"messages valides : {len(VALIDATED_MESSAGES)} lignes, "
      f"{len(VALIDATED_MESSAGES.columns)} colonnes")

casse = DOCUMENTS.copy()
casse.loc[casse.index[0], "msg_id"] = "1234"          # format d'identifiant invalide
casse.loc[casse.index[1], "text"] = "court"           # texte sous la longueur minimale
casse.loc[casse.index[2], "canal"] = "pigeon"         # canal non déclaré
try:
    validate_messages(casse)
except SchemaError as error:
    print("SchemaError attendue :", str(error).splitlines()[0])
""",
            context,
        ),
        _text(
            """
casse_lazy = DOCUMENTS.copy()
casse_lazy.loc[casse_lazy.index[0], "msg_id"] = "1234"
casse_lazy.loc[casse_lazy.index[1], "text"] = "court"
casse_lazy.loc[casse_lazy.index[2], "canal"] = "pigeon"
try:
    validate_messages(casse_lazy, lazy=True)
except SchemaErrors as errors:
    print(f"lazy=True : {errors.failure_cases['column'].nunique()} colonnes fautives signalées "
          f"en une seule passe")
    display(errors.failure_cases[["column", "check", "failure_case"]].head(6))
""",
            context,
        ),
        _insight(
            [
                "`lazy=True` collecte **toutes** les erreurs d'un coup : sur un fichier de 1 200 "
                "lignes, corriger une colonne par exécution coûte plus cher que la validation.",
                "Le message d'erreur nomme la ligne fautive : c'est ce qui rend un contrat "
                "utilisable par quelqu'un qui n'a pas écrit le schéma.",
                "Le contrat est appliqué **à l'écriture** par le chargeur : un corpus non conforme "
                "ne touche jamais le disque.",
            ]
        ),
        _md(
            "## 4. Le contrat des annotations, et le lien entre les tables\n\n"
            "Une annotation est une paire d'offsets **dans un message**. Le contrat ne peut donc pas "
            "la valider seule : il faut le texte pour vérifier que `surface == text[start:end]`, et "
            "l'appartenance au même split."
        ),
        _text(
            """
from src.data.schemas import corpus_violations, validate_corpus

VALIDATED_SPANS = validate_spans(SPANS)
PROBLEMES = corpus_violations(DOCUMENTS, SPANS)
print(f"annotations valides : {len(VALIDATED_SPANS)}")
print("violations du corpus :", PROBLEMES or "aucune")

decalage = SPANS.copy()
decalage.loc[decalage.index[0], "start"] = decalage.loc[decalage.index[0], "start"] + 1
print("offsets décalés d'un caractère :", corpus_violations(DOCUMENTS, decalage))

inconnu = SPANS.copy()
inconnu.loc[inconnu.index[0], "label"] = "numero_de_dossier"
print("type inventé :", corpus_violations(DOCUMENTS, inconnu))
""",
            context,
        ),
        _text(
            """
chevauchements = []
for message_id, groupe in SPANS.groupby("msg_id"):
    bornes = [tuple(borne) for borne in groupe[["start", "end"]].to_numpy()]
    for index, (start, end) in enumerate(bornes):
        for autre_start, autre_end in bornes[index + 1 :]:
            if min(end, autre_end) - max(start, autre_start) > 0:
                chevauchements.append((message_id, (start, end), (autre_start, autre_end)))
print(f"messages du corpus : {DOCUMENTS['msg_id'].nunique()}")
print(f"chevauchements de mentions : {len(chevauchements)} (aucun n'est une information métier)")
""",
            context,
        ),
        _insight(
            [
                "`surface == text[start:end]` est vérifié sur le corpus entier : sans ce contrôle, "
                "une annotation décalée d'un caractère passerait inaperçue jusqu'à ce qu'un rapport "
                "explique une F1 inexplicable.",
                "Un type inventé est refusé : il deviendrait sinon une classe publiée du rapport, "
                "que personne ne saurait interpréter.",
                "Deux mentions qui se chevauchent dans un même message ne sont pas une ambiguïté "
                "riche : c'est une erreur d'annotation.",
            ]
        ),
        _md(
            "## 5. Le contrat des prédictions\n\n"
            "La table de prédictions est l'entrée de la chaîne d'inférence : mêmes offsets, un type, "
            "une **provenance** (`regle` ou `modele`) et une confiance bornée. On la produit ici "
            "avec le modèle à règles seules, qui n'a rien à apprendre."
        ),
        _text(
            """
from src.data.schemas import validate_predictions
from src.inference.predictor import EntityPredictor
from src.models import build_model

rules_model = build_model(CONFIG, algorithm="gazetteer", params={})
rules_model.fit(TRAIN_DOCUMENTS, TRAIN_SPANS)
predictor = EntityPredictor(rules_model, config=CONFIG.model_dump(), id_column=ID_COLUMN)
PREDICTIONS = predictor.predict(VAL_DOCUMENTS, spans=VAL_SPANS)
print(f"prédictions : {len(PREDICTIONS)} mentions sur {len(VAL_DOCUMENTS)} messages")
display(PREDICTIONS.head(5))
display(validate_predictions(PREDICTIONS).head(3))
""",
            context,
        ),
        _text(
            """
casse_pred = PREDICTIONS.copy()
casse_pred.loc[casse_pred.index[0], "confidence"] = 1.4     # hors [0, 1]
casse_pred.loc[casse_pred.index[1], "source"] = "intuition"  # provenance non déclarée
casse_pred.loc[casse_pred.index[2], "label"] = "facture"     # type inconnu du projet
try:
    validate_predictions(casse_pred)
except SchemaError as error:
    print("SchemaError attendue :", str(error).splitlines()[0])
""",
            context,
        ),
        _insight(
            [
                "La provenance est contractuelle : une mention corroborée par une règle et une "
                "mention proposée par le modèle ne se relisent pas de la même façon.",
                "La confiance est bornée dans [0, 1] par le schéma — c'est un **niveau**, et le "
                "notebook 04 montre comment le lire avant de l'utiliser comme seuil.",
                "La table de prédictions est validée à l'écriture par le chargeur : la chaîne "
                "d'inférence ne peut pas publier une table non conforme.",
            ]
        ),
        _md(
            "## Synthèse\n\n"
            "| Contrat | Objet | Ce qu'il empêche |\n"
            "| --- | --- | --- |\n"
            "| Messages | `MessagesSchema` | identifiants dupliqués, texte tronqué, canal inventé |\n"
            "| Annotations | `SpansSchema` | offsets inversés, type inventé, surface vide |\n"
            "| Corpus | `corpus_violations` | `surface != text[start:end]`, mention hors de son "
            "split, chevauchement |\n"
            "| Prédictions | `PredictedMentionsSchema` | provenance inconnue, confiance hors bornes "
            "|\n\n"
            "**Suite** : `03_preprocessing.ipynb` transforme ces tables en traits mesurables."
        ),
    ]
    return _write(destination, "02_validation", cells)


# ---------------------------------------------------------------------------------------------
# 03 — traits de forme, tokénisation, index de règles
# ---------------------------------------------------------------------------------------------
def build_03_preprocessing(context: NotebookContext, destination: Path) -> Path:
    """Build ``03_preprocessing.ipynb``: traits de forme, alignement des offsets, index de règles.

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
            "Traits de forme, alignement des offsets et index de règles",
            [
                "Calculer les traits d'une mention depuis sa **surface** (longueur, chiffres, "
                "séparateurs, casse) et lire ce qu'ils séparent.",
                "Vérifier que les offsets du corpus tombent sur des **tokens** spaCy : ce que le "
                "modèle apprend dépend de cet alignement.",
                "Construire l'index de règles **sur le train** et mesurer ce qu'il ne peut pas "
                "savoir : les surfaces réservées à l'évaluation.",
                "Comprendre pourquoi un gazetteer n'est pas un modèle : il ne généralise pas.",
            ],
        ),
        _md("## 1. Mise en place"),
        _text(SETUP, context),
        _text(LOAD_CORPUS, context),
        _md(
            "## 1. Les traits de forme d'une mention\n\n"
            "Le modèle apprend ses propres représentations ; ces traits ne le nourrissent pas. Ils "
            "servent à **mesurer** ce que la forme d'une mention dit déjà — et donc à savoir quel "
            "score est mérité par une règle plutôt que par un modèle."
        ),
        _text(
            """
from src.features.build_features import (
    MESSAGE_FEATURES,
    SPAN_FEATURES,
    build_message_features,
    build_span_features,
    describe_features,
    feature_summary,
)

DOCUMENTATION = describe_features()
display(
    pd.DataFrame(
        {"trait": list(SPAN_FEATURES), "définition": [DOCUMENTATION[name] for name in SPAN_FEATURES]}
    )
)

FEATURES = build_span_features(DOCUMENTS, SPANS)
display(FEATURES[list(SPAN_FEATURES)].describe().round(3))
""",
            context,
        ),
        _text(
            """
RESUME_TYPES = feature_summary(FEATURES)
display(RESUME_TYPES.round(3))
print("total des mentions par type :", int(RESUME_TYPES["n_mentions"].sum()), "=", len(SPANS))
""",
            context,
        ),
        _text(
            """
MESSAGES = build_message_features(DOCUMENTS, SPANS)
display(MESSAGES[list(MESSAGE_FEATURES)].describe().round(3))
print("part moyenne du texte couverte par une annotation :",
      f"{MESSAGES['annotated_char_ratio'].mean():.1%}")
print("densité moyenne (mentions pour 100 caractères) :",
      round(float(MESSAGES['mentions_per_100_chars'].mean()), 3))
""",
            context,
        ),
        _insight(
            [
                "Ces traits sont **calculés** à partir de la surface (`surface`, pas une colonne "
                "déclarée) : un corpus qui se contredirait serait détecté ici.",
                "`feature_summary` agrège par type : les types réguliers ont une longueur et une "
                "densité de chiffres stables, les types « nom » beaucoup moins.",
                "`annotated_char_ratio` rappelle qu'une entité est une petite partie du message : "
                "la tâche est un repérage dans du texte, pas un classement de texte.",
            ]
        ),
        _md(
            "## 2. La signature de forme, type par type\n\n"
            "La signature réduit une surface à ses classes de caractères (`Aa-#` : un mot "
            "capitalisé, un séparateur, des chiffres). C'est la lecture la plus pauvre d'une "
            "mention — et celle qui dit si un type est **lisible** par une règle, et où cette "
            "règle s'arrête."
        ),
        _text(
            """
from src.features.build_features import shape_separability

par_type = (
    FEATURES.groupby(["label", "signature"], observed=True)
    .size()
    .rename("mentions")
    .reset_index()
    .sort_values("mentions", ascending=False)
)
display(par_type.head(15))
print("signatures distinctes par type :",
      FEATURES.groupby("label")["signature"].nunique().to_dict())
""",
            context,
        ),
        _text(
            """
SEPARABILITE = shape_separability(DOCUMENTS, SPANS)
print(f"signatures observées : {len(SEPARABILITE)}")
print("signatures qui ne portent qu'un seul type :",
      int((SEPARABILITE["n_labels"] == 1).sum()), "/", len(SEPARABILITE))
display(SEPARABILITE.sort_values("n_spans", ascending=False).head(10).round(4))
""",
            context,
        ),
        _insight(
            [
                "Sur ce corpus, aucune signature n'est partagée par plusieurs types : la forme "
                "ne se trompe pas de **type** — elle ne sait pas **trouver** les bornes, et elle "
                "est muette sur une forme inédite.",
                "Les types « nom » (produit, transporteur) multiplient les signatures : leur forme "
                "ne dit rien de plus que « un mot » ou « deux mots capitalisés », ce qui rend la "
                "liste apprise utile — et bornée aux surfaces déjà vues.",
                "La pureté (`purity`) est la part des mentions d'une signature qui portent son type "
                "dominant : à 1,0, la forme est un indice sûr — jamais une extraction.",
            ]
        ),
        _md(
            "## 3. Tokénisation et alignement des offsets\n\n"
            "Un pipeline spaCy travaille sur des **tokens**, le corpus est annoté en **caractères**. "
            "L'alignement est la jointure entre les deux : une mention qui ne tombe pas sur des "
            "frontières de tokens est ignorée à l'entraînement — et le projet publie le nombre de "
            "mentions ainsi perdues."
        ),
        _text(
            """
import spacy

# Pipeline vide : tokénisation seule, aucun modèle pré-entraîné, aucun téléchargement.
nlp = spacy.blank("fr")

exemple = VAL_DOCUMENTS.iloc[0]
doc = nlp(str(exemple[TEXT_COLUMN]))
print("texte   :", str(exemple[TEXT_COLUMN])[:110], "...")
print("tokens  :", [token.text for token in doc][:16])
print()
for row in SPANS[SPANS["msg_id"] == exemple[ID_COLUMN]].itertuples(index=False):
    aligned = doc.char_span(int(row.start), int(row.end), label=str(row.label))
    rendu = aligned.text if aligned is not None else "DÉSALIGNÉ (aucun token exact)"
    print(f"{str(row.label):12s} {str(row.surface)!r:30s} -> {rendu}")
""",
            context,
        ),
        _text(
            """
TEXTS = dict(zip(DOCUMENTS[ID_COLUMN], DOCUMENTS[TEXT_COLUMN], strict=True))


def alignment_report(documents: pd.DataFrame, spans: pd.DataFrame) -> dict[str, float]:
    \"\"\"Measure how many annotated mentions fall on token boundaries.\"\"\"
    total = aligned = 0
    for message_id, groupe in spans.groupby(ID_COLUMN):
        doc = nlp(str(TEXTS[message_id]))
        for row in groupe.itertuples(index=False):
            total += 1
            if doc.char_span(int(row.start), int(row.end), label=str(row.label)) is not None:
                aligned += 1
    return {
        "mentions": total,
        "alignées": aligned,
        "désalignées": total - aligned,
        "taux d'alignement": round(aligned / max(total, 1), 4),
    }


ALIGNMENT = alignment_report(TRAIN_DOCUMENTS, TRAIN_SPANS)
print("alignement des annotations du train :", ALIGNMENT)

decalé = TRAIN_SPANS.copy()
decalé["start"] = decalé["start"] + 1
print("mêmes annotations décalées d'un caractère :",
      alignment_report(TRAIN_DOCUMENTS, decalé))
""",
            context,
        ),
        _insight(
            [
                "Le générateur écrit les annotations **au moment de l'écriture** du texte : "
                "l'alignement est donc parfait par construction, et il est mesuré pour le prouver.",
                "Un corpus réel arrive souvent désaligné ; c'est exactement ce que le taux "
                "d'alignement du rapport d'entraînement (`alignment_aligned_rate`) publie.",
                "Une mention désalignée n'est pas « presque » apprise : elle est ignorée, donc "
                "comptée comme manquée à l'évaluation.",
            ]
        ),
        _md(
            "## 4. L'index de règles, appris sur le train seulement\n\n"
            "La couche de règles a deux étages : des **motifs** déclarés (commande, montant, date, "
            "transporteur) et un **index** des surfaces de noms vues à l'entraînement. Ce second "
            "étage est une liste : il ne peut pas reconnaître une surface qu'il n'a jamais lue."
        ),
        _text(
            """
from src.models.rules import GazetteerIndex, rule_spans

INDEX = GazetteerIndex.from_spans(TRAIN_SPANS)
print("surfaces indexées par type :",
      {label: len(surfaces) for label, surfaces in INDEX.surfaces.items()})

reserved = SPANS[SPANS["holdout"].astype(bool) & SPANS["label"].isin(NAME_LABELS)]
inconnues = sum(1 for surface in reserved["surface"] if str(surface).casefold() not in INDEX.lookup)
print(f"surfaces réservées de type « nom » : {len(reserved)}")
print(f"dont absentes de l'index (donc introuvables pour lui) : {inconnues}")
""",
            context,
        ),
        _text(
            """
exemple = DOCUMENTS[DOCUMENTS["split"] == "test"].iloc[0]
texte = str(exemple[TEXT_COLUMN])
print("texte :", texte[:120], "...")
print()
for span in rule_spans(texte, INDEX):
    print(f"{span.rule:20s} {span.label:12s} [{span.start}:{span.end}] "
          f"{texte[span.start:span.end]!r}")
""",
            context,
        ),
        _insight(
            [
                "Chaque span de règle publie la règle qui l'a produit (`motif:date`, "
                "`index:produit`) : une prédiction doit toujours pouvoir s'expliquer.",
                "Les motifs généralisent (une date inédite reste une date), l'index non (un nom "
                "inédit n'est rien) : c'est la raison d'être des surfaces réservées.",
                "L'index est construit **après** le découpage : le construire sur le corpus entier "
                "serait une fuite de données, et les tests le vérifient.",
            ]
        ),
        _md(
            "## Synthèse\n\n"
            "| Étage | Objet | Ce qu'il sait | Ce qu'il ignore |\n"
            "| --- | --- | --- | --- |\n"
            "| Traits de forme | `build_span_features` | longueur, chiffres, casse, séparateurs | "
            "le sens du texte |\n"
            "| Tokénisation | `spacy.blank('fr')` | frontières de tokens, offsets | toute "
            "connaissance linguistique |\n"
            "| Motifs | `rule_spans` (patterns) | les écritures régulières | les noms |\n"
            "| Index | `GazetteerIndex.from_spans(TRAIN_SPANS)` | les surfaces vues à "
            "l'entraînement | les surfaces réservées |\n\n"
            "**Suite** : `04_model_exploration.ipynb` compare les trois algorithmes qui utilisent "
            "ces étages."
        ),
    ]
    return _write(destination, "03_preprocessing", cells)


# ---------------------------------------------------------------------------------------------
# 04 — les trois algorithmes, l'architecture et la confiance
# ---------------------------------------------------------------------------------------------
def build_04_model_exploration(context: NotebookContext, destination: Path) -> Path:
    """Build ``04_model_exploration.ipynb``: comparaison des algorithmes et lecture de la confiance.

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
            "Trois algorithmes, une architecture lue, une confiance interprétée",
            [
                "Comparer les trois algorithmes de la stack sur la **validation** : règles, tagger, "
                "hybride.",
                "Lire l'architecture spaCy réellement construite (le projet ne la redéclare pas, "
                "il la publie).",
                "Comprendre ce que « confiance » veut dire ici : un niveau de **corroboration** par "
                "les règles, pas une probabilité.",
                "Arbitrer un seuil d'automatisation sur le split de calibration, puis le lire sur la "
                "validation.",
            ],
        ),
        _md("## 1. Mise en place"),
        _text(SETUP, context),
        _text(LOAD_CORPUS, context),
        _md(
            "## 1. Les algorithmes que la stack sait servir\n\n"
            "La fabrique est la seule porte d'entrée : le reste du projet ne connaît que le contrat "
            "`BaseEntityTagger`. Les trois algorithmes partagent donc exactement les mêmes métriques "
            "et le même format de sortie."
        ),
        _text(
            """
from src.models import ALGORITHMS, available_algorithms, build_model, describe_algorithm

SERVED_ALGORITHM = str(CONFIG.model.algorithm)
ALGORITHMS_TABLE = pd.DataFrame(
    [describe_algorithm(name) for name in available_algorithms(CONFIG.metrics.task)]
).set_index("name")
display(ALGORITHMS_TABLE[["display_name", "learns_weights", "rationale"]])
print("algorithmes déclarés :", list(ALGORITHMS), "| algorithme servi :", SERVED_ALGORITHM)
print("défauts du registre :", describe_algorithm(SERVED_ALGORITHM)["defaults"])
print("hyper-paramètres du manifeste :", dict(MODEL_NODE.get("params", {}) or {}))
""",
            context,
        ),
        _insight(
            [
                "`learns_weights` sépare deux familles : un gazetteer **ne généralise pas** (il "
                "reconnaît), un tagger apprend des représentations.",
                "Les défauts viennent du registre, les hyper-paramètres du manifeste : comparer deux "
                "algorithmes à paramètres explicites (`{}`) est la seule comparaison loyale.",
                "Le contrat commun est ce qui permet de changer d'algorithme sans toucher ni au "
                "trainer, ni à l'évaluateur, ni à l'inférence.",
            ]
        ),
        _md(
            "## 2. Comparaison sur la validation\n\n"
            "Budget réduit (`__NB_EPOCHS__` époques) et corpus réduit : ces chiffres servent à "
            "**choisir**, pas à publier. Un tagger neural à budget réduit n'a pas fini d'apprendre — "
            "le fait est visible dans le tableau, et c'est en soi une information."
        ),
        _text(
            """
from src.evaluation.evaluator import EntityEvaluator


def evaluate_on_validation(model, name: str) -> dict[str, float]:
    \"\"\"Fit one algorithm on the training split and score the validation split.\"\"\"
    model.fit(TRAIN_DOCUMENTS, TRAIN_SPANS)
    result = EntityEvaluator(
        model,
        config=CONFIG.model_dump(),
        paths=NB_PATHS,
        primary_metric=CONFIG.metrics.primary,
        min_primary_metric=CONFIG.metrics.min_primary,
    ).evaluate(VAL_DOCUMENTS, VAL_SPANS)
    return {
        "algorithme": name,
        "poids appris": describe_algorithm(name)["learns_weights"],
        "entity_f1 (micro)": result.metrics["entity_f1"],
        "macro_f1": result.metrics["macro_f1"],
        "precision": result.metrics["entity_precision"],
        "rappel": result.metrics["entity_recall"],
        "bornes exactes": result.metrics["boundary_accuracy"],
        "p50 (ms)": result.metrics["latency_p50_ms"],
        "mentions prédites": len(result.predictions),
    }


COMPARISON = pd.DataFrame(
    [evaluate_on_validation(build_model(CONFIG, algorithm=name, params=__COMPARISON_PARAMS__), name)
     for name in available_algorithms(CONFIG.metrics.task)]
).set_index("algorithme")
display(COMPARISON.round(4))
print("meilleur micro-F1 :", COMPARISON["entity_f1 (micro)"].idxmax(),
      "| meilleur macro-F1 :", COMPARISON["macro_f1"].idxmax())
""",
            context,
        ),
        _insight(
            [
                "Le **micro**-F1 récompense les types fréquents, le **macro**-F1 compte chaque type "
                "à égalité : un algorithme qui sacrifie le transporteur gagne au micro et perd au "
                "macro.",
                "Les règles seules obtiennent un score qui ne doit pas surprendre : sur les types à "
                "motif, elles sont excellentes — c'est mesuré au notebook 01.",
                "Un écart de score plus petit que la variabilité de la graine n'est pas un gain ; le "
                "notebook 05 montre comment mesurer cette variabilité.",
            ]
        ),
        _md(
            "## 3. L'architecture réellement construite\n\n"
            "Le projet ne **redéclare** pas l'architecture du réseau : il la lit sur le pipeline "
            "servi et la publie dans la fiche modèle. Une configuration partielle produirait un "
            "modèle différent de celui que la fiche décrit."
        ),
        _text(
            """
from dataclasses import asdict

from src.models import load_model

SERVED = build_model(CONFIG)          # algorithme et hyper-paramètres du manifeste
SERVED.fit(TRAIN_DOCUMENTS, TRAIN_SPANS)
ARCHITECTURE = SERVED.architecture_report()
print("pipes spaCy      :", ARCHITECTURE["pipe_names"])
print("architecture NER :", ARCHITECTURE.get("architecture"))
print("tok2vec          :", ARCHITECTURE.get("tok2vec"))
print()
CARD = SERVED.model_card(artifact="__MODEL_FILE__")
display(
    pd.Series(
        {key: str(value) for key, value in asdict(CARD).items() if key != "feature_names"}
    ).to_frame("fiche de modèle")
)
print("types servis par le modèle :", SERVED.labels)
""",
            context,
        ),
        _insight(
            [
                "`architecture_report()` lit le pipeline : HashEmbedCNN v2 (largeur 96, fenêtre 1, "
                "`subword` actif) et un tok2vec partagé — l'équivalent explicite du défaut du pipe "
                "`ner`.",
                "La fiche modèle contient les **versions** des bibliothèques : un artefact retrouvé "
                "six mois plus tard reste interprétable.",
                "Les types servis sont appris du corpus (`labels`), pas recopiés d'une "
                "configuration : un type absent du train n'existe pas pour le modèle.",
            ]
        ),
        _md(
            "## 4. La confiance est une corroboration, pas une probabilité\n\n"
            "La stack publie, à côté de chaque mention, une provenance (`regle` ou `modele`) et une "
            "confiance : la part du span confirmée par la couche de règles. Une confiance de 1,0 "
            "signifie « corroborée par une règle », pas « certaine »."
        ),
        _text(
            """
from src.inference.predictor import EntityPredictor
from src.training.metrics import confidence_gap, confidence_table

PREDICTOR = EntityPredictor(SERVED, config=CONFIG.model_dump(), id_column=ID_COLUMN)
PREDICTED_VAL = PREDICTOR.predict(VAL_DOCUMENTS, spans=VAL_SPANS)
display(confidence_table(PREDICTED_VAL, VAL_SPANS, bins=5))
print("provenances :", PREDICTED_VAL["source"].value_counts().to_dict())
print("confiance moyenne par provenance :",
      PREDICTED_VAL.groupby("source")["confidence"].mean().round(3).to_dict())
print("exactitude par provenance :",
      PREDICTED_VAL.groupby("source")["correct"].mean().round(3).to_dict())
print("écart |confiance moyenne des bonnes mentions - 1| :",
      confidence_gap(PREDICTED_VAL, VAL_SPANS))
""",
            context,
        ),
        _insight(
            [
                "Les tranches hautes de confiance sont plus précises que les basses — sinon la "
                "colonne ne servirait à rien, et le projet le dirait.",
                "`confidence_gap` mesure la distance entre la confiance annoncée et la précision "
                "observée : un écart élevé interdit d'appeler cette colonne une probabilité.",
                "Les mentions `regle` sont exactes par construction sur les surfaces apprises : "
                "c'est la partie du score qui vient de la liste, pas du modèle.",
            ]
        ),
        _md(
            "## 5. Un seuil d'automatisation, arbitré sur la calibration\n\n"
            "Si un conseiller ne relit que les mentions douteuses, il faut choisir le seuil : au-"
            "dessus, la mention est validée automatiquement. Le seuil se règle sur le split de "
            "**calibration** et se lit ensuite sur la validation — jamais sur le test."
        ),
        _text(
            """
CALIBRATION_SPANS = LOADER.split_annotations(
    "calibration", documents=DOCUMENTS, spans=SPANS
)
CALIBRATED = PREDICTOR.predict(CALIBRATION_DOCUMENTS, spans=CALIBRATION_SPANS)

rows = []
for threshold in __CONFIDENCE_GRID__:
    selected = CALIBRATED[CALIBRATED["confidence"] >= threshold]
    rows.append(
        {
            "seuil": threshold,
            "mentions validées": len(selected),
            "couverture": round(len(selected) / max(len(CALIBRATED), 1), 4),
            "précision observée": round(float(selected["correct"].mean()), 4) if len(selected) else None,
        }
    )
GRID = pd.DataFrame(rows)
display(GRID)
eligible = GRID[GRID["précision observée"].notna()]
tenable = eligible[eligible["précision observée"] >= __TARGET_AUTOMATION__]
if len(tenable):
    chosen = float(tenable["seuil"].min())
    print(f"seuil retenu : {chosen} (précision visée {__TARGET_AUTOMATION__})")
else:
    chosen = float(eligible["seuil"].max())
    print(f"aucun seuil n'atteint {__TARGET_AUTOMATION__} sur la calibration : "
          f"seuil le plus haut retenu ({chosen})")
""",
            context,
        ),
        _text(
            """
AUTOMATED = PREDICTED_VAL[PREDICTED_VAL["confidence"] >= chosen]
print(f"seuil {chosen} appliqué à la validation :")
print(f"  mentions auto-validées : {len(AUTOMATED)}/{len(PREDICTED_VAL)} "
      f"({len(AUTOMATED) / max(len(PREDICTED_VAL), 1):.1%})")
print(f"  précision observée     : {float(AUTOMATED['correct'].mean()):.4f}" if len(AUTOMATED)
      else "  aucune mention au-dessus du seuil")
print(f"  mentions à relire      : {len(PREDICTED_VAL) - len(AUTOMATED)}")
""",
            context,
        ),
        _insight(
            [
                "Le seuil est un **arbitrage de service** : la couverture dit combien de mentions "
                "on cesse de relire, la précision dit ce que cela coûte.",
                "Régler le seuil sur la calibration et le lire sur la validation est la même "
                "discipline que pour un modèle : le test reste vierge jusqu'au notebook 06.",
                "Un seuil sans mesure de précision n'est pas un seuil, c'est une espérance.",
            ]
        ),
        _md(
            "## Synthèse\n\n"
            "| Algorithme | Ce qu'il apprend | Force | Limite mesurée |\n"
            "| --- | --- | --- | --- |\n"
            "| `gazetteer` | rien (motifs + index du train) | types à motif, latence minimale | "
            "surfaces réservées introuvables |\n"
            "| `tagger` | représentations de tokens | généralise aux formes inédites | budget "
            "d'entraînement, types minoritaires |\n"
            "| `hybride` | les deux | score des motifs + généralisation | complexité de lecture |\n\n"
            "**Suite** : `05_training.ipynb` exécute le pipeline réel, dans un bac à sable."
        ),
    ]
    return _write(destination, "04_model_exploration", cells)


# ---------------------------------------------------------------------------------------------
# 05 — pipeline d'entraînement, artefacts et déterminisme
# ---------------------------------------------------------------------------------------------
def build_05_training(context: NotebookContext, destination: Path) -> Path:
    """Build ``05_training.ipynb``: pipeline réel, artefacts, rechargement et déterminisme.

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
                "Exécuter les pipelines du projet (`generate-data`, `train`) dans un bac à sable "
                "qui n'écrase rien.",
                "Lire les artefacts d'un run : répertoire spaCy, métriques, fiche modèle, "
                "configuration résolue.",
                "Recharger l'artefact et vérifier qu'il prédit **exactement** comme le modèle "
                "ajusté.",
                "Prouver le déterminisme : deux ajustements à graine fixée donnent les mêmes "
                "mentions.",
            ],
        ),
        _md("## 1. Mise en place"),
        _text(SETUP, context),
        _text(LOAD_CORPUS, context),
        _md(
            "## 2. Les pipelines du projet, dans un bac à sable\n\n"
            "Le notebook appelle **les pipelines du projet**, pas une réimplémentation : ce qu'on "
            "lit ici est exactement ce que fait `python -m src.main mode=train`. Les artefacts "
            "partent dans `outputs/notebooks/`, donc ceux de `make train` ne sont jamais écrasés."
        ),
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
        _md(
            "## 3. Les artefacts du run\n\n"
            "L'artefact d'un pipeline spaCy est un **répertoire** : le pipeline complet (tokéniseur, "
            "vocabulaire, reconnaisseur d'entités, configuration), pas un fichier unique. C'est ce "
            "qui permet de le recharger sans reconstruire l'architecture à la main."
        ),
        _text(
            """
ARTIFACT_DIR = NB_PATHS.models_dir / "__MODEL_FILE__"
inside = sorted(
    path.relative_to(ARTIFACT_DIR) for path in ARTIFACT_DIR.rglob("*") if path.is_file()
)
print(f"répertoire du modèle : {ARTIFACT_DIR.relative_to(PROJECT_ROOT)}")
for name in inside:
    print("  -", name)
print()
for name in ("model_card.json", "resolved_config.json"):
    path = NB_PATHS.models_dir / name
    print(f"{name:24s} {'écrit' if path.exists() else 'ABSENT':6s} "
          f"({path.stat().st_size if path.exists() else 0} octets)")
metrics_path = NB_PATHS.metrics_dir / "training_metrics.json"
print(f"{'training_metrics.json':24s} {'écrit' if metrics_path.exists() else 'ABSENT':6s} "
      f"({metrics_path.stat().st_size if metrics_path.exists() else 0} octets)")
assert ARTIFACT_DIR.is_dir() and metrics_path.exists()
""",
            context,
        ),
        _insight(
            [
                "Quatre artefacts répondent à quatre questions : le répertoire spaCy (comment "
                "prédire), les métriques (ce que ça vaut), la fiche modèle (comment c'est fait), la "
                "configuration résolue (avec quels réglages exactement).",
                "La configuration résolue est le seul moyen de rejouer un run : la configuration "
                "Hydra est faite de couches, et une couche qui change change tout.",
                "L'index de règles est **dans** l'artefact : à l'inférence, le modèle n'a pas besoin "
                "du corpus pour retrouver les surfaces qu'il a apprises.",
            ]
        ),
        _md(
            "## 4. Lire les métriques du run\n\n"
            "Les métriques d'entraînement distinguent ce que le modèle a **vu** (`train_*`) de ce "
            "qu'il n'a pas vu (`val_*`). L'écart entre les deux est la première mesure de "
            "généralisation disponible."
        ),
        _text(
            """
METRICS = pd.Series(OUTCOME.metrics).sort_values(ascending=False)
display(METRICS.to_frame("valeur").round(4))
print(f"messages d'entraînement : {OUTCOME.n_train_documents} "
      f"| entités vues : {OUTCOME.n_train_entities}")
print(f"messages de validation  : {OUTCOME.n_val_documents} "
      f"| entités de référence : {OUTCOME.n_val_entities}")
print(f"alignement des annotations : {OUTCOME.metrics['alignment_aligned_rate']:.4f} "
      f"({int(OUTCOME.metrics['alignment_ignored'])} mention(s) ignorée(s))")
display(OUTCOME.per_label.round(4))
""",
            context,
        ),
        _text(
            """
fit = OUTCOME.fit_result
print(f"ajustement : {fit.n_samples} messages, {fit.epochs} époque(s), "
      f"{fit.duration_seconds:.2f}s")
print("paramètres effectifs :", fit.params)
print("extra du run        :", {key: fit.extra[key] for key in sorted(fit.extra)})

HISTORY = pd.DataFrame(OUTCOME.history)
display(HISTORY.round(4))
if "loss" in HISTORY.columns:
    HISTORY["loss"].plot(title="perte d'entraînement par époque", xlabel="époque")
    plt.tight_layout()
""",
            context,
        ),
        _insight(
            [
                "`alignment_ignored = 0` est un résultat du corpus, pas une chance : les annotations "
                "sont écrites avec le texte (notebook 03).",
                "La table par type est publiée **dès l'entraînement** : elle évite de découvrir au "
                "rapport final qu'un type a été sacrifié.",
                "Le budget d'époques du notebook est réduit : la perte ne descend donc pas autant "
                "que dans le run de référence.",
            ]
        ),
        _md(
            "## 5. Recharger et comparer\n\n"
            "Un artefact qui ne se recharge pas ne se déploie pas. Le test compare les mentions "
            "**span par span**, pas une métrique moyenne : un écart d'un seul caractère doit "
            "apparaître."
        ),
        _text(
            """
from src.models import build_model, load_model

RESTORED = load_model(ARTIFACT_DIR, config=CONFIG.model_dump())
print("artefact rechargé :", RESTORED.summary())
print("types servis      :", RESTORED.labels)

TEXTS = [str(text) for text in TEST_DOCUMENTS[TEXT_COLUMN]]
IDENTIFIERS = [str(value) for value in TEST_DOCUMENTS[ID_COLUMN]]

FIRST = build_model(CONFIG)
FIRST.fit(TRAIN_DOCUMENTS, TRAIN_SPANS)
COPY_DIR = FIRST.save(NB_PATHS.models_dir / "notebook_copy")
RELOADED = load_model(COPY_DIR, config=CONFIG.model_dump())

BEFORE = FIRST.predict(TEXTS, ids=IDENTIFIERS)
AFTER = RELOADED.predict(TEXTS, ids=IDENTIFIERS)
identical = [[mention.as_tuple() for mention in found] for found in BEFORE] == [
    [mention.as_tuple() for mention in found] for found in AFTER
]
print("mentions identiques après rechargement :", identical)
print("exemple :", [mention.as_tuple() for mention in BEFORE[0]])
assert identical, "un artefact rechargé doit prédire exactement comme le modèle ajusté"
""",
            context,
        ),
        _text(
            """
SECOND = build_model(CONFIG)
SECOND.fit(TRAIN_DOCUMENTS, TRAIN_SPANS)
AGAIN = SECOND.predict(TEXTS, ids=IDENTIFIERS)
same_run = [[mention.as_tuple() for mention in found] for found in BEFORE] == [
    [mention.as_tuple() for mention in found] for found in AGAIN
]
print("deux ajustements à graine fixée donnent les mêmes mentions :", same_run)
print(f"graine du projet : {CONFIG.seed} | époques : {NB_EPOCHS} | "
      f"messages d'entraînement : {len(TRAIN_DOCUMENTS)}")

variances = pd.DataFrame(
    {
        "run": ["premier ajustement", "second ajustement"],
        "mentions extraites": [sum(len(found) for found in BEFORE), sum(len(found) for found in AGAIN)],
    }
)
display(variances)
""",
            context,
        ),
        _insight(
            [
                "Le déterminisme n'est pas un luxe : sans lui, aucune comparaison entre deux "
                "variantes n'a de sens — on ne saurait pas si l'écart vient du modèle ou du hasard.",
                "La graine est unique et vient de la configuration (`seed`), pas d'un appel caché à "
                "`random()`.",
                "La comparaison porte sur les mentions, pas sur les scores : deux modèles peuvent "
                "afficher la même F1 avec des erreurs différentes.",
            ]
        ),
        _md(
            "## 6. Comparer au run de référence\n\n"
            "`make train` écrit ses artefacts dans `artifacts/`. S'il a tourné sur ce clone, on peut "
            "comparer le run de référence (corpus complet, budget d'époques du manifeste) au run du "
            "notebook — sans jamais confondre les deux."
        ),
        _text(
            """
import json

REFERENCE_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "training_metrics.json"
key = f"val_{CONFIG.metrics.primary}"
notebook_score = float(OUTCOME.metrics[key])
if REFERENCE_PATH.exists():
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    reference_score = float(reference["metrics"].get(key, float("nan")))
    print(f"run de référence (make train) : {key} = {reference_score:.4f} "
          f"({reference['n_train_documents']} messages)")
    print(f"run du notebook               : {key} = {notebook_score:.4f} "
          f"({OUTCOME.n_train_documents} messages, {NB_EPOCHS} époques)")
    print(f"écart                         : {notebook_score - reference_score:+.4f}")
else:
    print("Aucun run de référence dans artifacts/ : lancer `make train` pour comparaison.")
    print(f"run du notebook : {key} = {notebook_score:.4f}")
""",
            context,
        ),
        _insight(
            [
                "Un notebook ne remplace pas un run de référence : corpus réduit et budget réduit "
                "servent à comprendre, pas à publier.",
                "L'écart entre les deux se lit comme un **coût du budget** : plus d'époques et plus "
                "de messages, plus de score — jusqu'à un plateau.",
                "La discipline reste la même : le split de test n'est ouvert qu'au notebook 06.",
            ]
        ),
        _md(
            "## Synthèse\n\n"
            "| Étape | Objet | Artefact |\n"
            "| --- | --- | --- |\n"
            "| Génération | `DataGenerationPipeline` | `data/raw/*.parquet` (bac à sable) |\n"
            "| Entraînement | `TrainPipeline` → `EntityTrainer` | `models/__MODEL_FILE__/` |\n"
            "| Métriques | `EntityTrainingOutcome` | `metrics/training_metrics.json` |\n"
            "| Traçabilité | `ModelCard` | `models/model_card.json` |\n"
            "| Reproductibilité | `seed` de la configuration | `models/resolved_config.json` |\n\n"
            "**Suite** : `06_error_analysis.ipynb` ouvre le test, une seule fois, et transforme les "
            "erreurs en recommandations."
        ),
    ]
    return _write(destination, "05_training", cells)


# ---------------------------------------------------------------------------------------------
# 06 — évaluation sur le test et analyse d'erreurs
# ---------------------------------------------------------------------------------------------
def build_06_error_analysis(context: NotebookContext, destination: Path) -> Path:
    """Build ``06_error_analysis.ipynb``: verdict, erreurs classées, figures et rapport.

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
            "Évaluation sur le test — verdict, erreurs classées et rapport",
            [
                "Ouvrir le split de test **une seule fois** et lire le verdict contractuel.",
                "Ventiler la performance par type, par style et par canal, et comparer les surfaces "
                "**réservées** à celles vues à l'entraînement.",
                "Classer les erreurs (inventée, bornes, manquée) et en montrer un exemple de chaque.",
                "Écrire le rapport du projet, ses tables et ses figures, puis en tirer des "
                "recommandations chiffrées.",
            ],
        ),
        _md("## 1. Mise en place"),
        _text(SETUP, context),
        _text(LOAD_CORPUS, context),
        _md(
            "## 2. Entraîner le modèle à évaluer\n\n"
            "Le test ne s'ouvre qu'après l'entraînement : le modèle est ajusté sur `train`, "
            "surveillé sur `val`, et le split de test sert à **juger**, une fois."
        ),
        _text(
            """
from src.inference.predictor import EntityPredictor
from src.models import build_model

MODEL = build_model(CONFIG)
FIT = MODEL.fit(TRAIN_DOCUMENTS, TRAIN_SPANS)
print(f"modèle : {MODEL.summary()}")
print(f"ajustement : {FIT.n_samples} messages, {FIT.epochs} époque(s), {FIT.duration_seconds:.2f}s")
print("métriques d'entraînement (extrait) :",
      {key: round(value, 4) for key, value in FIT.metrics.items() if key in {"entity_f1", "entity_recall"}})
""",
            context,
        ),
        _md(
            "## 3. Le verdict contractuel\n\n"
            "Le verdict compare la métrique principale à son seuil déclaré "
            "(`metrics.min_primary`). Il est **indéterminé** quand la métrique manque : un contrat "
            "qu'on ne peut pas lire n'est pas un contrat satisfait."
        ),
        _text(
            """
from src.evaluation.evaluator import EntityEvaluator

EVALUATOR = EntityEvaluator(
    MODEL,
    config=CONFIG.model_dump(),
    paths=NB_PATHS,
    primary_metric=CONFIG.metrics.primary,
    min_primary_metric=CONFIG.metrics.min_primary,
)
RESULT = EVALUATOR.evaluate(TEST_DOCUMENTS, TEST_SPANS)

print(f"messages évalués : {RESULT.n_documents} | entités de référence : {RESULT.n_entities}")
print(f"métrique principale : {RESULT.primary_metric} = "
      f"{RESULT.metrics[RESULT.primary_metric]:.4f} (seuil : {RESULT.threshold})")
print(f"verdict : {RESULT.verdict} — {RESULT.verdict_detail.get('message', '')}")
display(pd.Series(RESULT.metrics).sort_values(ascending=False).to_frame("valeur").round(4))
""",
            context,
        ),
        _insight(
            [
                "La métrique principale est **micro** : toutes les mentions comptent pareil. La F1 "
                "macro, publiée à côté, compte les types à égalité — les deux lectures sont "
                "nécessaires.",
                "`partial_f1` compte une mention du bon type qui **recouvre** l'entité : l'écart avec "
                "`entity_f1` est le prix de l'exigence de bornes.",
                "Le verdict porte sur la validation du contrat de service, pas sur la qualité "
                "scientifique du modèle.",
            ]
        ),
        _md(
            "## 4. Par type, par segment, par surface\n\n"
            "Une moyenne cache toujours la même chose : **quelqu'un** perd. Trois ventilations "
            "rendent la perte visible — le type, le segment d'écriture, et la provenance de la "
            "surface."
        ),
        _text(
            """
display(RESULT.per_label.round(4))
FRAGILE = RESULT.per_label[RESULT.per_label["label"] != "micro"].sort_values("f1").iloc[0]
print(f"type le plus fragile : {FRAGILE['label']} (F1 {FRAGILE['f1']:.4f}, "
      f"{int(FRAGILE['support'])} mentions de référence)")
""",
            context,
        ),
        _text(
            """
display(RESULT.segments.round(4))
print("segments publiés :", sorted(set(RESULT.segments["segment"])))
display(RESULT.holdout.round(4))
print(f"rappel sur les surfaces réservées   : {RESULT.metrics['holdout_recall']:.4f}")
print(f"rappel sur les surfaces vues au train : {RESULT.metrics['seen_surface_recall']:.4f}")
print(f"écart (ce que la mémorisation explique) : {RESULT.metrics['holdout_gap']:.4f}")
""",
            context,
        ),
        _insight(
            [
                "La ventilation par **style** est la plus instructive du corpus : les messages "
                "abrégés écrivent les mêmes entités autrement.",
                "L'écart « surface réservée / surface vue » est la mesure honnête de ce que le "
                "modèle a **appris** plutôt que mémorisé.",
                "Un type fragile avec peu de support se lit avec prudence : trois mentions ne font "
                "pas une conclusion.",
            ]
        ),
        _md(
            "## 5. Les trois familles d'erreurs\n\n"
            "Le projet classe les erreurs plutôt que de les compter : une mention **inventée** coûte "
            "de la précision, une mention **manquée** du rappel, et une erreur de **bornes** est un "
            "cas à part — le type est bon, la découpe ne l'est pas."
        ),
        _text(
            """
ERREURS = RESULT.errors
print("erreurs publiées :", len(ERREURS))
if not ERREURS.empty:
    display(ERREURS["kind"].value_counts().to_frame("erreurs"))
    display(
        ERREURS.head(8)[["msg_id", "kind", "label", "surface", "predicted_surface", "context"]]
    )
""",
            context,
        ),
        _text(
            """
if not ERREURS.empty:
    for kind in ("inventee", "bornes", "manquee"):
        exemple = ERREURS[ERREURS["kind"] == kind]
        if exemple.empty:
            print(f"{kind:10s} : aucune erreur de ce type sur le test")
            continue
        row = exemple.iloc[0]
        print(f"{kind:10s} : {row['label']} {str(row['surface'])!r} "
              f"(prédit : {row['predicted_surface']!r})")
""",
            context,
        ),
        _insight(
            [
                "Les erreurs de **bornes** sont les plus coûteuses en apparence : elles comptent "
                "double (faux positif et faux négatif) et le `partial_f1` montre qu'elles "
                "recouvrent souvent la bonne entité.",
                "Une mention **inventée** sur un distracteur déclaré (une ville, une référence de "
                "facture) est une erreur prévisible : le corpus la contenait pour être mesurée.",
                "Le contexte est publié avec chaque erreur : un lecteur peut juger sans rouvrir le "
                "corpus.",
            ]
        ),
        _md(
            "## 6. Les références : plancher trivial et règles seules\n\n"
            "Un score ne se lit pas seul. Deux références sont publiées : le plancher trivial (un "
            "système qui n'annote rien) et la couche de règles **apprise sur le train**, mesurée sur "
            "le même test."
        ),
        _text(
            """
from src.training.metrics import trivial_floor

BASELINES = pd.DataFrame(RESULT.baselines).T
display(BASELINES.round(4))
print("plancher trivial :", trivial_floor())
reference = float(RESULT.baselines.get("regles", {}).get("entity_f1", float("nan")))
gain = RESULT.metrics["entity_f1"] - reference
print(f"gain du modèle sur les règles seules : {gain:+.4f} de F1 micro")
""",
            context,
        ),
        _insight(
            [
                "Le plancher trivial vaut 0,0 par convention : un système qui n'annonce rien n'a "
                "aucune précision.",
                "La référence « règles » est apprise sur le **train** et mesurée sur le test : c'est "
                "la même discipline que pour le modèle.",
                "Un gain plus petit que l'écart sur les surfaces réservées serait un signal clair : "
                "le modèle n'apprendrait pas grand-chose de plus que la liste.",
            ]
        ),
        _md(
            "## 7. Le rapport écrit par le projet\n\n"
            "L'évaluation produit un rapport Markdown, ses tables CSV et ses figures. C'est "
            "l'artefact que lit un lecteur pressé — et celui qui explique chaque chiffre."
        ),
        _text(
            """
from src.evaluation.reports import ReportBuilder

BUNDLE = ReportBuilder(CONFIG.model_dump(), paths=NB_PATHS).build(
    RESULT,
    model_summary={
        "name": MODEL.name,
        "framework": MODEL.framework,
        "algorithm": MODEL.algorithm,
        "task": MODEL.task,
        "labels": MODEL.labels,
        "state": MODEL.state,
    },
    metadata=METADATA,
    documents=DOCUMENTS,
    spans=SPANS,
)
print("verdict du rapport :", BUNDLE.verdict)
print(BUNDLE.summary)
for path in BUNDLE.artifacts:
    relative = Path(path).relative_to(NB_PATHS.artifacts_dir)
    print(f"  - {relative}")
""",
            context,
        ),
        _text(
            """
LINES = BUNDLE.report_path.read_text(encoding="utf-8").splitlines()
print("\\n".join(LINES[:42]))
""",
            context,
        ),
        _text(
            """
from src.visualization.plots import plot_holdout, plot_per_label

figure_types = plot_per_label(RESULT.per_label, NB_PATHS.figures_dir / "test_per_label.png")
figure_reserves = plot_holdout(RESULT.holdout, NB_PATHS.figures_dir / "test_holdout.png")
display(Image(filename=str(figure_types)))
display(Image(filename=str(figure_reserves)))
""",
            context,
        ),
        _insight(
            [
                "Un rapport qui se contente d'un score n'est pas un rapport : celui-ci publie les "
                "tables, les figures et les erreurs, chacun dans un fichier.",
                "Les artefacts du notebook vont dans `outputs/notebooks/` : le rapport de référence "
                "reste celui de `make evaluate`.",
                "Le rapport est régénéré à chaque `mode=evaluate` : il ne peut pas mentir sur l'état "
                "du modèle servi.",
            ]
        ),
        _md(
            "## 8. Recommandations chiffrées\n\n"
            "Ce que ce notebook permet de décider, avec les chiffres qui le justifient."
        ),
        _text(
            """
F1 = RESULT.metrics["entity_f1"]
MACRO = RESULT.metrics["macro_f1"]
BORNES = RESULT.metrics["boundary_accuracy"]
PARTIEL = RESULT.metrics["partial_f1"]
ECART = RESULT.metrics["holdout_gap"]
RAPPEL_RESERVE = RESULT.metrics["holdout_recall"]

print("1. Qualité globale")
print(f"   F1 micro {F1:.4f} / macro {MACRO:.4f} : la hiérarchie des types n'est pas plate.")
print(f"   bornes exactes {BORNES:.4f} contre F1 partielle {PARTIEL:.4f} : "
      f"{PARTIEL - BORNES:.4f} de mentions bien typées mais mal découpées.")
print()
print("2. Ce qui vient de la liste et ce qui vient du modèle")
print(f"   rappel sur surfaces vues {RESULT.metrics['seen_surface_recall']:.4f} contre "
      f"{RAPPEL_RESERVE:.4f} sur les réservées (écart {ECART:.4f}).")
print(f"   les {int(RESULT.holdout['n_entities'].sum())} mentions réservées sont la vraie mesure "
      "de la généralisation.")
print()
print("3. Où agir")
print(f"   type le plus fragile : {FRAGILE['label']} (F1 {FRAGILE['f1']:.4f}) — enrichir ses "
      "surfaces à l'entraînement, pas ses motifs.")
segments = RESULT.segments.set_index(["segment", "value"])
if ("style", "abrege") in segments.index:
    print(f"   style abrégé : F1 {segments.loc[('style', 'abrege'), 'entity_f1']:.4f} — vérifier "
          "les variantes d'écriture des motifs.")
print(
    f"   verdict contractuel : {RESULT.verdict} "
    f"(seuil {RESULT.threshold} sur {RESULT.primary_metric})."
)
""",
            context,
        ),
        _insight(
            [
                "Trois décisions, trois chiffres : enrichir les surfaces des types fragiles, "
                "surveiller l'écart sur les surfaces réservées, garder le seuil contractuel comme "
                "porte de déploiement.",
                "Un modèle qui gagne sur les surfaces vues et perd sur les réservées mémorise : le "
                "tableau des références est là pour le dire.",
                "Le test a été ouvert une fois, dans ce notebook : toute nouvelle décision devra se "
                "prendre sur `val` et se vérifier sur un nouveau test.",
            ]
        ),
        _md(
            "## Synthèse\n\n"
            "| Question | Réponse du notebook | Artefact |\n"
            "| --- | --- | --- |\n"
            "| Le contrat est-il tenu ? | `RESULT.verdict`, `threshold` | `evaluation_report.md` |\n"
            "| Quel type fragile ? | `RESULT.per_label` | `per_label.csv` |\n"
            "| Quel segment perd ? | `RESULT.segments` | `segment_metrics.csv` |\n"
            "| Le modèle mémorise-t-il ? | `holdout_recall`, `holdout_gap` | `holdout.csv` |\n"
            "| Quelles erreurs corriger ? | `RESULT.errors` | `errors.csv` |\n\n"
            "**Fin du parcours** : de la carte du corpus (01) au rapport (06). Reprendre la "
            "configuration, pas le code : `conf/`, puis `python -m src.main mode=all`."
        ),
    ]
    return _write(destination, "06_error_analysis", cells)


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
    """Build the six notebooks of a named-entity-recognition project.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory of the project.

    Returns:
        The written notebook paths, in order.
    """
    return [builder(context, destination) for builder in BUILDERS]
