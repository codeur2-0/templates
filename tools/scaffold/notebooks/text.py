"""Notebooks pédagogiques des projets **texte** (RAG, agents, embeddings, QA documentaire).

Un projet de recherche documentaire ne s'explore pas comme un tableau : il n'y a pas de colonnes à
décrire, mais un **corpus** à comprendre, un **découpage** à choisir et une **décision de refus** à
arbitrer. Les six notebooks gardent la progression des autres modalités, avec les questions propres
à la modalité texte :

* ``01_eda.ipynb`` — cartographier le corpus (sources, sections, longueurs, dates) et surtout le
  **jeu de questions** : parts des difficultés, questions hors corpus, documents annotés par
  question. C'est cette carte qui dit ce qu'un rappel mesurera plus tard ;
* ``02_validation.ipynb`` — les contrats Pandera du texte, et surtout ce qui se passe quand on les
  casse : une règle qu'on n'a jamais vue échouer n'est pas une règle ;
* ``03_preprocessing.ipynb`` — découpage en passages (offsets vérifiés), mots vides, puis
  **mesure** de l'effet de la taille des passages sur le rappel ;
* ``04_model_exploration.ipynb`` — planchers triviaux, comparaison des scorers lexicaux et courbe
  du seuil d'abstention sur le split de calibration ;
* ``05_training.ipynb`` — le pipeline d'entraînement réel dans un bac à sable, ses artefacts et sa
  reproductibilité ;
* ``06_error_analysis.ipynb`` — le verdict contractuel, la ventilation par segment, les questions
  perdues une par une, le comportement d'abstention et des recommandations chiffrées.

Les helpers de rendu (``_md``, ``_code``, ``_insight``, ``_objectives``) sont partagés avec le
squelette tabulaire : un notebook de recherche documentaire et un notebook de classification
parlent la même langue typographique.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tools.scaffold.notebooks.tabular import _insight, _md, _objectives
from tools.scaffold.utils_notebooks import NotebookNode, code_cell, write_notebook

if TYPE_CHECKING:  # pragma: no cover
    from tools.scaffold.utils_notebooks import NotebookContext

__all__ = ["build_all"]

#: Taille du corpus réduit des notebooks (documents) : l'exécution complète reste rapide.
NB_DOCUMENTS = 64

#: Tailles de passage comparées par le notebook de prétraitement, en tokens.
#: 40 et 60 fragmentent un document de ~90 tokens ; 110 est la valeur configurée ; 180 dépasse la
#: longueur d'un document et sert à montrer le plateau.
CHUNK_SIZES = (40, 60, 110, 180)


def _tokens(context: NotebookContext) -> dict[str, str]:
    """Build the substitution table of the text notebooks.

    Args:
        context: Notebook context.

    Returns:
        Mapping of ``__TOKEN__`` to replacement text.
    """
    spec = context.spec
    chunking = dict(spec.preprocessing.get("chunking", {}))
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
        "__CHUNK_TOKENS__": str(chunking.get("max_tokens", 110)),
        "__OVERLAP_TOKENS__": str(chunking.get("overlap_tokens", 30)),
        "__CHUNK_SIZES__": repr(list(CHUNK_SIZES)),
        "__REPORT_NAME__": str(artifacts.get("report_file", "evaluation_report.md")),
        "__MODEL_FILE__": str(artifacts.get("model_file", "model.joblib")),
    }


def _text(source: str, context: NotebookContext) -> NotebookNode:
    """Render a code cell with the text tokens substituted.

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
# Les notebooks travaillent sur un corpus réduit (__NB_DOCUMENTS__ documents) quand `data/raw` est
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

CHUNKING = dict(node(CONFIG, "preprocessing").get("chunking", {}))

print(f"Projet            : {CONFIG.project.name}")
print(f"Tâche             : {CONFIG.metrics.task}")
print(f"Métrique primaire : {CONFIG.metrics.primary} (plancher contractuel : __MIN_PRIMARY__)")
print(f"Algorithme        : {CONFIG.model.algorithm} ({CONFIG.model.name})")
print(f"Découpage         : {CHUNKING.get('max_tokens')} tokens, "
      f"chevauchement {CHUNKING.get('overlap_tokens')}")
print(f"Sorties notebook  : {NB_PATHS.artifacts_dir.relative_to(PROJECT_ROOT)}")
"""

LOAD_CORPUS = """
from src.data.generators import SyntheticCorpusGenerator
from src.data.loaders import TextCorpusLoader

LOADER = TextCorpusLoader(PATHS, formats=CONFIG.data.formats)
if LOADER.documents_path.exists():
    DOCUMENTS = LOADER.load_documents()
    QUERIES = LOADER.load_queries()
    METADATA = LOADER.load_metadata()
    print(f"Corpus lu depuis data/raw : {len(DOCUMENTS)} documents, {len(QUERIES)} questions")
else:
    # Un clone frais n'a pas encore de données : le notebook reste exécutable. Le corpus réduit est
    # persisté pour que les pipelines (train, evaluate) trouvent les mêmes fichiers que le
    # notebook ; `make data` régénère le corpus de référence.
    bundle = SyntheticCorpusGenerator.from_config(CONFIG.data).generate()
    DOCUMENTS, QUERIES, METADATA = bundle.documents, bundle.queries, bundle.metadata
    LOADER.save_documents(DOCUMENTS)
    LOADER.save_queries(QUERIES)
    LOADER.save_metadata(METADATA)
    print(f"data/raw absent : corpus de {len(DOCUMENTS)} documents généré et persisté")

CALIBRATION = QUERIES[QUERIES["split"] == "calibration"].reset_index(drop=True)
VALIDATION = QUERIES[QUERIES["split"] == "val"].reset_index(drop=True)
TEST = QUERIES[QUERIES["split"] == "test"].reset_index(drop=True)
ANSWERABLE = QUERIES[QUERIES["answer_type"] != "unanswerable"].reset_index(drop=True)
# Sous-ensemble utilisé pour **régler** quoi que ce soit : la validation seule. Le test ne sert
# qu'à juger, jamais à choisir (y compris dans un notebook).
VAL_ANSWERABLE = VALIDATION[VALIDATION["answer_type"] != "unanswerable"].reset_index(drop=True)
print("splits :", {name: len(frame) for name, frame in
                   (("calibration", CALIBRATION), ("val", VALIDATION), ("test", TEST))})
"""


# ---------------------------------------------------------------------------------------------
# 01 — exploration du corpus et du jeu de questions
# ---------------------------------------------------------------------------------------------
def build_01_eda(context: NotebookContext, destination: Path) -> Path:
    """Build ``01_eda.ipynb``: cartographie du corpus et du jeu de questions.

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
            "Exploration du corpus et des questions",
            [
                "Décrire un corpus documentaire avant de l'indexer : sources, sections, longueurs, "
                "dates de publication.",
                "Lire un jeu de questions annotées : difficultés, intentions, questions hors corpus "
                "et nombre de documents pertinents par question.",
                "Comprendre pourquoi la part de questions hors corpus conditionne la lecture de "
                "toutes les métriques de rappel.",
            ],
        ),
        _text(SETUP, context),
        _md(
            "## 1. Le corpus\n\n"
            "Un assistant documentaire ne vaut que par son corpus : ce que l'organisation a écrit, "
            "où, quand et pour qui. Les quatre lectures ci-dessous — source, section, longueur, "
            "date — suffisent à détecter les deux pathologies classiques : une source qui écrase "
            "toutes les autres, et un corpus périmé dont les réponses ne sont plus valides."
        ),
        _text(LOAD_CORPUS, context),
        _text(
            """
DOCUMENTS.head(3)[["doc_id", "title", "section", "source", "published_at", "n_tokens"]]
""",
            context,
        ),
        _text(
            """
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
DOCUMENTS["source"].value_counts().plot.bar(ax=axes[0], title="Documents par source", rot=30)
DOCUMENTS["section"].value_counts().head(10).plot.bar(
    ax=axes[1], title="Documents par section", rot=30
)
DOCUMENTS["n_tokens"].plot.hist(ax=axes[2], bins=30, title="Longueur des documents (tokens)")
plt.tight_layout()
plt.show()

print("période couverte       :", DOCUMENTS["published_at"].min().date(),
      "->", DOCUMENTS["published_at"].max().date())
print("identifiants dupliqués :", int(DOCUMENTS["doc_id"].duplicated().sum()))
print("sections distinctes    :", DOCUMENTS["section"].nunique())
""",
            context,
        ),
        _insight(
            [
                "Une source qui concentre la majorité des documents oriente toutes les réponses : "
                "la ventilation par source est un prérequis à la lecture du rappel.",
                "Les longueurs fixent le nombre de passages : un document de 40 tokens ne produit "
                "qu'un passage, un document de 400 en produit quatre.",
                "Des identifiants dupliqués casseraient les jointures entre passages, réponses et "
                "questions : le contrat les refuse, et cette cellule les compte.",
            ]
        ),
        _md(
            "## 2. Le jeu de questions\n\n"
            "Le corpus dit ce qui est **écrit**, les questions disent ce qui est **demandé**. La "
            "difficulté annotée n'est pas décorative : elle indique à quel titre une question "
            "compte. Une question hors corpus ne se juge pas au rappel mais à l'abstention."
        ),
        _text(
            """
summary = (
    QUERIES.assign(hors_corpus=QUERIES["answer_type"] == "unanswerable")
    .groupby("difficulty")
    .agg(
        questions=("query_id", "size"),
        documents_annotes=("n_gold_docs", "mean"),
        hors_corpus=("hors_corpus", "sum"),
    )
    .round(2)
)
summary
""",
            context,
        ),
        _text(
            """
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
QUERIES["split"].value_counts().sort_index().plot.bar(
    ax=axes[0], title="Questions par split", rot=0
)
QUERIES["difficulty"].value_counts().plot.bar(
    ax=axes[1], title="Questions par difficulté", rot=30
)
QUERIES["intent"].value_counts().plot.bar(ax=axes[2], title="Questions par intention", rot=30)
plt.tight_layout()
plt.show()

print("questions hors corpus     :", int((QUERIES["answer_type"] == "unanswerable").sum()))
print("questions multi-document  :", int((QUERIES["n_gold_docs"] > 1).sum()))
print("longueur moyenne question :", round(QUERIES["question"].str.len().mean(), 1), "caractères")
""",
            context,
        ),
        _text(
            """
# Une question répondable est annotée par les documents qui la portent **et** par l'extrait exact
# qui contient la réponse ; une question hors corpus n'a ni l'un ni l'autre.
easy = ANSWERABLE.sort_values("n_gold_docs").iloc[0]
out_of_corpus = QUERIES[QUERIES["answer_type"] == "unanswerable"].iloc[0]
display(
    pd.DataFrame([easy, out_of_corpus])[
        ["question", "reference_answer", "gold_doc_ids", "difficulty", "answer_type"]
    ]
)
print("extrait annoté de la question répondable :", str(easy["gold_span"])[:200])
""",
            context,
        ),
        _insight(
            [
                "Trois familles de questions cohabitent : retrouver une formulation identique "
                "(facile), comprendre sans les mêmes mots (paraphrase), agréger deux sources "
                "(multi-document). Elles n'appellent pas les mêmes correctifs.",
                "La part de questions hors corpus est le seul segment où « ne pas répondre » est "
                "la bonne réponse : sans elle, un assistant qui répond toujours paraît excellent.",
                "Le split de calibration sert à régler la décision de refus : ni la validation ni "
                "le test ne doivent y servir, sinon la mesure publiée est fausse.",
            ]
        ),
    ]
    return write_notebook(destination / "01_eda.ipynb", cells)


# ---------------------------------------------------------------------------------------------
# 02 — validation des contrats
# ---------------------------------------------------------------------------------------------
def build_02_validation(context: NotebookContext, destination: Path) -> Path:
    """Build ``02_validation.ipynb``: contrats Pandera du corpus et des questions.

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
                "Lire les contrats du projet : documents, questions, passages et prédictions.",
                "Vérifier que le corpus réel les respecte, puis observer ce qui se passe quand un "
                "fichier ne les respecte plus.",
                "Comprendre la règle « question hors corpus ⇔ extrait vide », qui empêche une "
                "annotation incohérente de gonfler le rappel.",
            ],
        ),
        _text(SETUP, context),
        _md(
            "## 1. Les contrats\n\n"
            "Un contrat de données est du code exécutable : il nomme chaque colonne, son type et "
            "ses règles croisées. C'est aussi la documentation la plus fiable du jeu de données, "
            "puisqu'elle est vérifiée à chaque exécution du pipeline."
        ),
        _text(
            """
from src.data.schemas import (
    ChunksSchema,
    DocumentsSchema,
    PredictedAnswersSchema,
    QueriesSchema,
    validate_documents,
    validate_queries,
)

for schema in (DocumentsSchema, QueriesSchema, ChunksSchema, PredictedAnswersSchema):
    print(f"{schema.__name__:22s} -> {', '.join(schema.to_schema().columns)}")
""",
            context,
        ),
        _md("### 1.1 Le corpus respecte ses contrats"),
        _text(
            LOAD_CORPUS
            + "\n"
            + """
validated_documents = validate_documents(DOCUMENTS)
validated_queries = validate_queries(QUERIES)
print(f"{len(validated_documents)} documents valides, {len(validated_queries)} questions valides")

# La convention est explicite : une question hors corpus désigne un document leurre (celui qui en
# est le plus proche) mais **aucun extrait** — c'est l'extrait qui porte la réponse. Le contrat
# vérifie que les deux marqueurs sont d'accord : « hors corpus » <=> « extrait vide ».
unanswerable = validated_queries["answer_type"] == "unanswerable"
empty_span = validated_queries["gold_span"].str.strip() == ""
assert bool((unanswerable == empty_span).all())
print("règle vérifiée : answer_type == 'unanswerable' <=> extrait annoté vide")
print(
    "questions hors corpus désignant tout de même un document leurre :",
    int((unanswerable & (validated_queries["gold_doc_ids"] != "")).sum()),
    "-> leur pertinence est jugée nulle, c'est l'extrait qui compte",
)
""",
            context,
        ),
        _md(
            "## 2. Casser volontairement les contrats\n\n"
            "Chaque corruption ci-dessous correspond à un incident réel : identifiant mal formé, "
            "source inconnue, colonne manquante, question annotée sans document. Le message "
            "d'erreur de Pandera est affiché tel quel — c'est lui qu'on lira en production."
        ),
        _text(
            """
from typing import Any


def show_failure(label: str, frame: pd.DataFrame, validator: Any) -> None:
    \"\"\"Afficher la première ligne du message d'erreur d'un contrat.\"\"\"
    try:
        validator(frame)
    except Exception as error:  # on veut le message du contrat, quel qu'il soit
        print(f"[refusé] {label} : {str(error).splitlines()[0]}")
    else:
        print(f"[accepté] {label} : aucune erreur (le contrat est trop laxiste)")


bad_id = DOCUMENTS.head(1).copy()
bad_id.loc[bad_id.index[0], "doc_id"] = "doc-1"
show_failure("identifiant hors format", bad_id, validate_documents)

bad_source = DOCUMENTS.head(1).copy()
bad_source.loc[bad_source.index[0], "source"] = "wiki_inconnu"
show_failure("source inconnue", bad_source, validate_documents)

missing_column = DOCUMENTS.drop(columns=["text"])
show_failure("colonne manquante", missing_column, validate_documents)
""",
            context,
        ),
        _text(
            """
answerable = ANSWERABLE.head(1).copy()
answerable.loc[answerable.index[0], "gold_doc_ids"] = ""
show_failure("question répondable sans document annoté", answerable, validate_queries)

sabotaged = QUERIES[QUERIES["answer_type"] == "unanswerable"].head(1).copy()
sabotaged.loc[sabotaged.index[0], "gold_span"] = "Un extrait qui n'existe pas."
show_failure("question hors corpus avec extrait", sabotaged, validate_queries)
""",
            context,
        ),
        _text(
            """
# Le mode *lazy* rassemble toutes les violations en une seule erreur : c'est ce qu'on veut dans un
# rapport de qualité de données, où l'on préfère corriger plusieurs défauts en un seul passage.
broken = DOCUMENTS.head(3).copy()
broken.loc[broken.index[0], "doc_id"] = "nope"
broken.loc[broken.index[1], "source"] = "wiki_inconnu"
try:
    validate_documents(broken, lazy=True)
except Exception as error:
    cases = getattr(error, "failure_cases", None)
    print("violations détectées :", 0 if cases is None else len(cases))
    if cases is not None:
        display(cases)
""",
            context,
        ),
        _insight(
            [
                "Un contrat protège la **jointure** : identifiants, catégories, cohérence croisée. "
                "Sans lui, une question mal annotée se traduit par un rappel nul dont personne ne "
                "retrouve l'origine.",
                "Le mode strict refuse toute colonne non déclarée : un fichier personnalisé ne peut "
                "pas injecter silencieusement une colonne dans le corpus publié.",
                "La règle « hors corpus ⇔ extrait vide » est la seule qui distingue proprement "
                "« je ne sais pas » de « je réponds n'importe quoi ».",
            ]
        ),
    ]
    return write_notebook(destination / "02_validation.ipynb", cells)


# ---------------------------------------------------------------------------------------------
# 03 — découpage, tokens et taille des passages
# ---------------------------------------------------------------------------------------------
def build_03_preprocessing(context: NotebookContext, destination: Path) -> Path:
    """Build ``03_preprocessing.ipynb``: découpage, tokens, effet de la taille des passages.

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
            "Découpage en passages et vocabulaire",
            [
                "Découper un document en passages dont les offsets pointent réellement sur le "
                "texte annoncé.",
                "Observer le filtrage des mots vides et son effet sur le vocabulaire.",
                "Mesurer l'effet de la taille des passages sur le rappel@5, au lieu de la choisir "
                "au ressenti.",
            ],
        ),
        _text(SETUP, context),
        _md(
            "## 1. Du texte brut aux passages\n\n"
            "Le découpage est la décision la plus structurante d'un système de recherche : un "
            "passage trop court perd le contexte, un passage trop long dilue le signal. Ici le "
            "découpage se fait **par phrase** et conserve ses positions : un passage doit pouvoir "
            "être affiché tel quel dans une réponse sourcée."
        ),
        _text(LOAD_CORPUS, context),
        _text(
            """
from src.preprocessing.transformers import DocumentChunker, StopWordFilter, tokenize

sample = DOCUMENTS.iloc[0]
chunker = DocumentChunker(**CHUNKING)
chunks = chunker.chunk_document(str(sample["doc_id"]), str(sample["text"]))
print(f"document {sample['doc_id']} : {sample['n_tokens']} tokens -> {len(chunks)} passages")
for chunk in chunks:
    segment = str(sample["text"])[chunk.start_char : chunk.end_char]
    verdict = "offsets exacts" if segment.strip() == chunk.text.strip() else "OFFSETS INCOHÉRENTS"
    print(f"  {chunk.chunk_id} [{chunk.start_char}:{chunk.end_char}] "
          f"{chunk.n_tokens:>3} tokens — {verdict}")
""",
            context,
        ),
        _text(
            """
print("texte du premier passage, tel qu'il sera cité :")
print(chunks[0].text[:400], "…")
print()
print("tokens du même passage :", tokenize(chunks[0].text)[:15], "…")
""",
            context,
        ),
        _md(
            "## 2. Tokénisation et mots vides\n\n"
            "La recherche lexicale raisonne sur des termes. Retirer les mots vides français est ce "
            "qui rend le score discriminant : sans cela, « de », « la » et « pour » dominent et "
            "deux textes sans rapport paraissent proches."
        ),
        _text(
            """
raw_tokens = tokenize("Le solde de congés payés est de 25 jours ouvrables par an.")
filtered = StopWordFilter().transform(raw_tokens)
print("tokens bruts   :", raw_tokens)
print("tokens filtrés :", filtered)
print(f"{len(raw_tokens) - len(filtered)} mots vides retirés sur {len(raw_tokens)}")
""",
            context,
        ),
        _md(
            "## 3. Le corpus découpé\n\n"
            "On applique le pipeline de prétraitement au corpus entier, puis on regarde ce qui est "
            "réellement indexé : nombre de passages, longueur moyenne et distribution."
        ),
        _text(
            """
from src.preprocessing.pipelines import TextPreprocessor

PREPROCESSOR = TextPreprocessor.from_config(node(CONFIG, "preprocessing"))
CHUNKS = PREPROCESSOR.prepare_corpus(DOCUMENTS)
print(f"{len(DOCUMENTS)} documents -> {len(CHUNKS)} passages indexés")
print(f"longueur moyenne d'un passage : {CHUNKS['n_tokens'].mean():.1f} tokens")
display(CHUNKS.head(3)[["chunk_id", "doc_id", "chunk_index", "n_tokens"]])
""",
            context,
        ),
        _text(
            """
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
CHUNKS["n_tokens"].plot.hist(ax=axes[0], bins=25, title="Longueur des passages (tokens)")
CHUNKS.groupby("doc_id").size().plot.hist(
    ax=axes[1], bins=15, title="Passages produits par document"
)
axes[0].set_xlabel("tokens")
axes[1].set_xlabel("passages")
plt.tight_layout()
plt.show()
""",
            context,
        ),
        _md(
            "## 4. Effet de la taille des passages sur le rappel\n\n"
            "Chaque taille testée re-découpe et ré-indexe le corpus. C'est exactement l'expérience "
            "à mener avant de choisir un découpage : elle coûte quelques secondes avec un index "
            "lexical, et elle remplace une intuition par une mesure."
        ),
        _text(
            """
import numpy as np
from src.models import build_model
from src.training.losses_metrics import retrieval_metrics

gold = [{item for item in str(ids).split(",") if item}
        for ids in VAL_ANSWERABLE["gold_doc_ids"]]

rows = []
for size in __CHUNK_SIZES__:
    variant_config = CONFIG.model_dump()
    variant_config["preprocessing"]["chunking"] = {
        "max_tokens": size,
        "overlap_tokens": max(size // 4, 10),
    }
    variant = build_model(variant_config)
    fit_result = variant.fit(DOCUMENTS)
    print(f"taille {size:>4} tokens -> {fit_result.n_chunks} passages "
          f"(vocabulaire {fit_result.extra.get('vocabulary_size')} termes)")
    rankings = [
        [passage.doc_id for passage in variant.retrieve(str(question), 5)]
        for question in VAL_ANSWERABLE["question"]
    ]
    rows.append(
        {
            "taille": size,
            "passages": fit_result.n_chunks,
            "vocabulaire": fit_result.extra.get("vocabulary_size", float("nan")),
            "recall_at_5": retrieval_metrics(rankings, gold, ks=(5,))["recall_at_5"],
        }
    )

benchmark = pd.DataFrame(rows)
n_questions = len(VAL_ANSWERABLE)
# Un rappel mesuré sur ~50 questions porte son propre bruit : l'intervalle de confiance évite de
# conclure sur un écart de deux points. C'est la différence entre un notebook et une croyance.
benchmark["intervalle_95"] = 1.96 * np.sqrt(
    benchmark["recall_at_5"] * (1.0 - benchmark["recall_at_5"]) / n_questions
)
print(f"questions de validation utilisées : {n_questions} "
      f"(longueur moyenne d'un document : {DOCUMENTS['n_tokens'].mean():.0f} tokens)")
benchmark.round(4)
""",
            context,
        ),
        _text(
            """
best = benchmark.loc[benchmark["recall_at_5"].idxmax()]
configured = benchmark.loc[benchmark["taille"] == __CHUNK_TOKENS__]
print(f"meilleure taille observée : {int(best['taille'])} tokens "
      f"(rappel@5 = {best['recall_at_5']:.3f} ± {best['intervalle_95']:.3f})")
if len(configured):
    gap = float(best["recall_at_5"] - configured["recall_at_5"].iloc[0])
    verdict = "au-delà du bruit" if abs(gap) > float(best["intervalle_95"]) else "dans le bruit"
    print(f"écart au découpage configuré ({__CHUNK_TOKENS__} tokens) : {gap:+.3f} -> {verdict}")
plateau = benchmark.groupby("passages")["taille"].apply(list)
for passages, sizes in plateau.items():
    if len(sizes) > 1:
        print(f"tailles {sizes} : mêmes {passages} passages indexés, donc même mesure "
              "— au-delà de la longueur d'un document, un découpage plus grand ne change rien")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(benchmark["taille"], benchmark["passages"], marker="o")
axes[0].set(title="Passages indexés", xlabel="taille d'un passage (tokens)", ylabel="passages")
axes[1].errorbar(benchmark["taille"], benchmark["recall_at_5"],
                 yerr=benchmark["intervalle_95"], marker="o", color="darkgreen", capsize=4)
axes[1].set(title="Rappel@5 et son intervalle à 95 %", xlabel="taille d'un passage (tokens)",
            ylabel="recall@5")
plt.tight_layout()
plt.show()
""",
            context,
        ),
        _insight(
            [
                "Le résultat est un **plateau**, pas un optimum : dès qu'un passage couvre tout le "
                "document, agrandir la taille ne change plus ni l'index ni la mesure.",
                "Toute différence inférieure à l'intervalle de confiance affiché est du bruit : "
                "sur quelques dizaines de questions, décider sur un écart de deux points revient à "
                "tirer à pile ou face. Il faut plus de questions, pas plus d'intuition.",
                "Chaque taille re-indexe le corpus : l'expérience n'est acceptable que parce que "
                "l'index lexical se construit en quelques secondes — un argument de plus pour "
                "commencer par une baseline lexicale.",
                "Le nombre de passages indexés fixe la latence et la mémoire : le choix du "
                "découpage est un arbitrage mesuré, pas une constante recopiée.",
            ]
        ),
    ]
    return write_notebook(destination / "03_preprocessing.ipynb", cells)


# ---------------------------------------------------------------------------------------------
# 04 — planchers, scorers et seuil de refus
# ---------------------------------------------------------------------------------------------
def _deduplication_section(context: NotebookContext) -> list[NotebookNode]:
    """Build the notebook cells that measure the near-duplicates of the corpus.

    The section is added to the families whose corpus writes the same fact several times: the dense
    stack exposes ``nearest_neighbours`` precisely for that second use case, and the measured ratio
    is the one the family publishes.

    Args:
        context: Notebook context.

    Returns:
        The markdown, code and insight cells of the section.
    """
    return [
        _md(
            "## 4. Dédoublonnage — le second usage de l'index\n\n"
            "Chaque fait du corpus est écrit trois fois (notice, fiche commerciale, note SAV) : "
            "c'est ce qui rend le rappel généreux, et c'est aussi un travail à part entière — "
            "retrouver les quasi-doublons d'une fiche sans écrire de recherche par mots-clés. "
            "L'index répond aux deux questions : ``retrieve`` pour une question, "
            "``nearest_neighbours`` pour un texte déjà indexé, en excluant la fiche interrogée."
        ),
        _text(
            """
def fact_and_reference(title: str) -> tuple[str, str]:
    \"\"\"Lire le fait et la référence d'un titre de fiche (style — fait — produit (REF)).\"\"\"
    parts = [part.strip() for part in title.split(" — ")]
    return parts[1], parts[2].rsplit("(", 1)[1].rstrip(")")


titles = dict(zip(DOCUMENTS["doc_id"], DOCUMENTS["title"], strict=True))
same_fact = 0
example: tuple[str, list[str]] | None = None
for record in DOCUMENTS.to_dict(orient="records"):
    neighbours = MODEL.nearest_neighbours(
        str(record["text"]), 2, exclude_doc_id=str(record["doc_id"])
    )
    if all(
        fact_and_reference(str(titles[neighbour.doc_id]))
        == fact_and_reference(str(record["title"]))
        for neighbour in neighbours
    ):
        same_fact += 1
        if example is None:
            example = (
                str(record["doc_id"]),
                [str(neighbour.doc_id) for neighbour in neighbours],
            )
print(f"{same_fact}/{len(DOCUMENTS)} fiches ont leurs deux autres écritures en tête")
print(f"exemple : {example[0]} est voisin de {', '.join(example[1])}")
example_frame = DOCUMENTS[DOCUMENTS["doc_id"].isin([example[0], *example[1]])]
example_frame[["doc_id", "title", "section"]]
""",
            context,
        ),
        _insight(
            [
                "Un index ne sert pas seulement à répondre à des questions : le même objet "
                "retrouve les quasi-doublons d'un texte, et une propriété du corpus — trois "
                "écritures par fait — devient une mesure.",
                "Le test de la stack refait cette vérification sur toutes les fiches : une "
                "propriété annoncée au README et vérifiée sur un échantillon ne prouve rien.",
                "Le dédoublonnage se juge sur ce qu'on en fait : ici il compte les écritures "
                "multiples, il ne les supprime pas — trois sources valent mieux qu'une pour un "
                "référentiel publié par plusieurs équipes.",
            ]
        ),
    ]


def build_04_model_exploration(context: NotebookContext, destination: Path) -> Path:
    """Build ``04_model_exploration.ipynb``: planchers, scorers et seuil d'abstention.

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
            "Exploration : planchers, scorers et seuil de refus",
            [
                "Installer les planchers triviaux (tirage aléatoire, ordre du corpus) avant toute "
                "comparaison.",
                "Comparer les scorers lexicaux à découpage, corpus et questions identiques.",
                "Choisir le seuil d'abstention sur le split de calibration et lire ce que ce choix "
                "coûte en couverture.",
            ],
        ),
        _text(SETUP, context),
        _md(
            "## 1. Les planchers\n\n"
            "Sur un corpus où chaque question désigne un thème précis, un tirage aléatoire trouve "
            "parfois un document du bon thème. Sans ce plancher, on ne sait pas si un rappel de "
            "0,2 est un résultat ou du hasard : c'est la première ligne à écrire dans un rapport."
        ),
        _text(LOAD_CORPUS, context),
        _text(
            """
import random

from src.training.losses_metrics import retrieval_metrics

gold = [{item for item in str(ids).split(",") if item}
        for ids in VAL_ANSWERABLE["gold_doc_ids"]]

rng = random.Random(CONFIG.seed)
all_docs = DOCUMENTS["doc_id"].tolist()
random_ranking = [
    rng.sample(all_docs, k=min(10, len(all_docs))) for _ in range(len(VAL_ANSWERABLE))
]
corpus_order = [all_docs[:10] for _ in range(len(VAL_ANSWERABLE))]

baselines = pd.DataFrame(
    {
        "tirage aléatoire": retrieval_metrics(random_ranking, gold, ks=(1, 5, 10)),
        "ordre du corpus": retrieval_metrics(corpus_order, gold, ks=(1, 5, 10)),
    }
).T
baselines[[f"recall_at_{k}" for k in (1, 5, 10)] + ["mrr"]].round(4)
""",
            context,
        ),
        _md(
            "## 2. Comparer les scorers\n\n"
            "La stack sert plusieurs algorithmes — la cellule suivante les énumère : deux scorers ne "
            "pondèrent pas la fréquence des termes de la même façon et deux index ne stockent pas la "
            "même dimension. Le protocole est figé — même découpage, même corpus, mêmes questions — "
            "pour que l'écart observé vienne de l'algorithme et de rien d'autre."
        ),
        _text(
            """
from src.models import available_algorithms, build_model

print("algorithmes servis par la stack :", available_algorithms())

comparison: dict[str, dict[str, float]] = {}
for algorithm in available_algorithms():
    candidate = build_model(CONFIG.model_dump(), algorithm=algorithm)
    fit_result = candidate.fit(DOCUMENTS, CALIBRATION)
    rankings = [
        [passage.doc_id for passage in candidate.retrieve(str(question), 10)]
        for question in VAL_ANSWERABLE["question"]
    ]
    metrics = retrieval_metrics(rankings, gold, ks=(1, 5, 10))
    # Chaque stack publie son empreinte : taille du vocabulaire pour un index lexical, dimension
    # stockée et fidélité de projection pour un index dense. Une clé absente n'est pas inventée.
    footprint_keys = {
        "vocabulary_size": "vocabulaire",
        "dimension": "dimension",
        "projection_fidelity": "fidélité",
    }
    footprint = {
        label: float(fit_result.extra[key])
        for key, label in footprint_keys.items()
        if key in fit_result.extra
    }
    comparison[algorithm] = {
        "passages": float(fit_result.n_chunks),
        **footprint,
        "recall_at_1": metrics["recall_at_1"],
        "recall_at_5": metrics["recall_at_5"],
        "recall_at_10": metrics["recall_at_10"],
        "mrr": metrics["mrr"],
    }
comparison_frame = pd.DataFrame(comparison).T
comparison_frame.round(4)
""",
            context,
        ),
        _text(
            """
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
comparison_frame[["recall_at_1", "recall_at_5", "recall_at_10"]].plot.bar(ax=axes[0], rot=0)
axes[0].set(title="Rappel par scorer (validation)", ylabel="rappel")
comparison_frame[["mrr"]].plot.bar(ax=axes[1], rot=0, color="darkorange", legend=False)
axes[1].set(title="MRR — rang du premier passage pertinent", ylabel="mrr")
plt.tight_layout()
plt.show()
""",
            context,
        ),
        _insight(
            [
                "Les planchers ne sont pas une formalité : ils fixent le niveau à battre, et un "
                "rapport qui les omet laisse croire qu'un score médiocre est un progrès.",
                "Comparer deux scorers à découpage différent ne mesure rien : le protocole doit "
                "être identique, seul le scorer change.",
                "Un vocabulaire plus grand n'est pas un avantage en soi : ce qui compte est de "
                "classer le bon passage en tête.",
            ]
        ),
        _md(
            "## 3. Le seuil de refus\n\n"
            "Répondre à une question dont la réponse n'est pas dans le corpus est l'erreur la plus "
            "coûteuse d'un assistant : elle est crédible et fausse. Le seuil est réglé sur le split "
            "de **calibration**, puis on mesure ce qu'il coûte sur les questions qui, elles, ont "
            "une réponse."
        ),
        _text(
            """
MODEL = build_model(CONFIG.model_dump())
FIT_RESULT = MODEL.fit(DOCUMENTS, CALIBRATION)
print(f"{FIT_RESULT.n_chunks} passages indexés en {FIT_RESULT.duration_seconds:.2f}s")
print(f"vocabulaire : {FIT_RESULT.extra.get('vocabulary_size')} termes")
print(f"seuil d'abstention retenu par la calibration : {MODEL.abstention_threshold:.4f}")

scores_answerable: list[float] = []
scores_out_of_corpus: list[float] = []
for record in CALIBRATION.to_dict(orient="records"):
    scores = [passage.score for passage in MODEL.retrieve(str(record["question"]), 5)]
    target = (
        scores_out_of_corpus if record["answer_type"] == "unanswerable" else scores_answerable
    )
    target.append(float(max(scores, default=0.0)))
""",
            context,
        ),
        _text(
            """
import numpy as np

candidates = np.unique(np.concatenate([scores_answerable, scores_out_of_corpus]))
sweep = pd.DataFrame(
    [
        {
            "seuil": float(candidate),
            "refus corrects": float(np.mean(np.asarray(scores_out_of_corpus) < candidate)),
            "couverture": float(np.mean(np.asarray(scores_answerable) >= candidate)),
        }
        for candidate in candidates
    ]
)
sweep["moyenne des deux"] = 0.5 * (sweep["refus corrects"] + sweep["couverture"])
best = sweep.loc[sweep["moyenne des deux"].idxmax()]
print(f"meilleur compromis (calibration) : seuil {best['seuil']:.3f} — "
      f"{best['refus corrects']:.0%} de refus corrects "
      f"pour {best['couverture']:.0%} de couverture")
display(sweep.iloc[:: max(len(sweep) // 10, 1)].round(3))
""",
            context,
        ),
        _text(
            """
plt.figure(figsize=(8, 4))
plt.plot(sweep["seuil"], sweep["refus corrects"], label="refus corrects (hors corpus)")
plt.plot(sweep["seuil"], sweep["couverture"], label="couverture (questions avec réponse)")
plt.plot(sweep["seuil"], sweep["moyenne des deux"], linestyle="--", color="black",
         label="moyenne des deux")
plt.axvline(MODEL.abstention_threshold, color="red", linestyle=":", label="seuil retenu")
plt.xlabel("score du meilleur passage")
plt.ylabel("taux")
plt.title("Arbitrage du seuil de refus")
plt.legend()
plt.show()
""",
            context,
        ),
        _insight(
            [
                "Un seuil ne s'optimise pas par exactitude globale : sur un jeu majoritairement "
                "répondable, répondre toujours donne une bonne exactitude et zéro refus utile.",
                "L'arbitrage s'énonce en deux nombres — refus corrects et couverture — et c'est au "
                "métier de choisir le point de fonctionnement, pas au modèle.",
                "Si la calibration n'apporte pas un gain réel, le code conserve le seuil configuré "
                "et l'écrit dans les journaux : un réglage inventé serait pire qu'un réglage absent.",
            ]
        ),
    ]
    # Seule la stack dense expose ``nearest_neighbours`` : la variante lexicale de la même famille
    # garde le notebook à trois sections.
    if context.spec.stack == "embedding":
        cells.extend(_deduplication_section(context))
    return write_notebook(destination / "04_model_exploration.ipynb", cells)


# ---------------------------------------------------------------------------------------------
# 05 — entraînement dans les conditions de production
# ---------------------------------------------------------------------------------------------
def build_05_training(context: NotebookContext, destination: Path) -> Path:
    """Build ``05_training.ipynb``: le pipeline d'entraînement réel dans un bac à sable.

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
            "Entraînement de production",
            [
                "Exécuter le pipeline d'entraînement réel, sans le réécrire dans le notebook.",
                "Lire les artefacts produits : passages, métriques, fiche de modèle et "
                "configuration résolue.",
                "Vérifier la reproductibilité : reconstruire l'index retrouve les mêmes passages.",
            ],
        ),
        _text(SETUP, context),
        _md(
            "## 1. Le pipeline, pas une approximation\n\n"
            "Un notebook qui « refait » l'entraînement finit toujours par diverger du code de "
            "production. Ici on appelle `TrainPipeline`, exactement comme `python -m src.main "
            "mode=train`, en redirigeant ses écritures vers `outputs/notebooks`."
        ),
        _text(LOAD_CORPUS, context),
        _text(
            """
from src.pipelines.train_pipeline import TrainPipeline

TRAIN_RESULT = TrainPipeline(CONFIG, paths=NB_PATHS).run()
print("statut :", TRAIN_RESULT.status)
for message in TRAIN_RESULT.messages:
    print("-", message)
print()
print("artefacts écrits :")
for artifact in TRAIN_RESULT.artifacts:
    print("  -", Path(artifact).relative_to(PROJECT_ROOT))
""",
            context,
        ),
        _text(
            """
metrics = pd.Series(TRAIN_RESULT.metrics).sort_index()
validation_metrics = metrics[
    metrics.index.str.startswith(
        ("val_recall", "val_mrr", "val_ndcg", "val_answer", "val_citation", "val_abstention",
         "val_latency")
    )
]
print("métriques de validation (la validation surveille, le test jugera) :")
validation_metrics.round(4).to_frame("valeur")
""",
            context,
        ),
        _md(
            "### 1.1 La fiche de modèle\n\n"
            "Elle est écrite à côté de l'artefact et dit ce que le fichier sérialisé ne dit pas : "
            "quelle configuration, quelles métriques, quelles limites connues."
        ),
        _text(
            """
from src.models import build_model, load_model
from src.utils.io import read_json

ARTIFACT = NB_PATHS.models_dir / "__MODEL_FILE__"
TRAINED = load_model(ARTIFACT, config=CONFIG.model_dump())
# La fiche **archivée** est celle qu'a écrite le pipeline : c'est elle qui porte les métriques du
# run. `TRAINED.model_card()` en reconstruit une vide, exactement comme une instance rechargée ne
# connaît pas l'historique d'entraînement.
card = read_json(NB_PATHS.models_dir / "model_card.json")
for key in ("model_name", "framework", "algorithm", "task", "n_documents", "n_chunks",
            "created_at", "artifact"):
    print(f"  {key:14s}: {card.get(key)}")
print()
print("paramètres effectifs :")
display(pd.Series(card.get("params", {})).to_frame("valeur"))
print(f"métriques archivées : {len(card.get('metrics', {}))}")
print("notes de la fiche :")
for note in card.get("notes", [])[:4]:
    print("  -", note)
""",
            context,
        ),
        _text(
            """
chunks = TRAINED.chunks
display(chunks.head(3)[["chunk_id", "doc_id", "chunk_index", "n_tokens"]])
print(f"{len(chunks)} passages indexés")
print("fichiers de passages :", [path.name for path in NB_PATHS.processed_dir.glob("chunks.*")])
print("taille moyenne d'un passage :", round(float(chunks["n_tokens"].mean()), 1), "tokens")
""",
            context,
        ),
        _md(
            "## 2. Reproductibilité\n\n"
            "Reconstruire l'index doit retrouver les mêmes passages et les mêmes scores : c'est la "
            "condition pour que deux variantes soient comparables, et pour qu'un incident puisse "
            "être rejoué à l'identique."
        ),
        _text(
            """
rebuilt = build_model(CONFIG.model_dump())
rebuilt.fit(DOCUMENTS)
probe = str(TEST["question"].iloc[0])
first = [passage.chunk_id for passage in TRAINED.retrieve(probe, 5)]
second = [passage.chunk_id for passage in rebuilt.retrieve(probe, 5)]
print("question de sonde :", probe)
print("index entraîné    :", first)
print("index reconstruit :", second)
print("identiques        :", first == second)
""",
            context,
        ),
        _text(
            """
resolved = read_json(NB_PATHS.models_dir / "resolved_config.json")
print("configuration résolue, clés de premier niveau :", sorted(resolved)[:12])
print()
print("Cette copie permet de rejouer un run : l'artefact seul ne dit pas quel override "
      "l'a produit.")
""",
            context,
        ),
        _insight(
            [
                "L'indexation est un `fit` au sens du contrat : elle produit des artefacts, des "
                "métriques et une fiche de modèle, comme n'importe quel autre modèle.",
                "Les passages sont persistés : sans eux, impossible d'expliquer pourquoi une "
                "question a échoué, ni de rejouer une évaluation.",
                "La configuration résolue est archivée avec l'artefact : c'est elle qui rend le "
                "résultat reproductible, pas la graine seule.",
            ]
        ),
    ]
    return write_notebook(destination / "05_training.ipynb", cells)


# ---------------------------------------------------------------------------------------------
# 06 — analyse d'erreurs et recommandations
# ---------------------------------------------------------------------------------------------
def build_06_error_analysis(context: NotebookContext, destination: Path) -> Path:
    """Build ``06_error_analysis.ipynb``: verdict, segments, abstention et recommandations.

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
            "Analyse d'erreurs et recommandations",
            [
                "Lire le verdict contractuel de l'évaluateur de production sur le split de test.",
                "Ventiler les résultats par difficulté et par intention : trois segments, trois "
                "correctifs différents.",
                "Mesurer le comportement d'abstention, puis proposer des actions reliées à un "
                "chiffre.",
            ],
        ),
        _text(SETUP, context),
        _md(
            "## 1. Le verdict\n\n"
            "L'évaluation de production écrit un rapport et un fichier de métriques ; on les relit "
            "ici sans les recalculer, pour qu'un notebook ne puisse jamais publier un chiffre "
            "différent de celui du rapport. C'est cette même chaîne qui tourne en CI : "
            "l'évaluation charge l'artefact entraîné, mesure le split de test une seule fois et "
            "compare la métrique principale au plancher trivial."
        ),
        _text(LOAD_CORPUS, context),
        _text(
            """
from src.models import build_model, load_model
from src.pipelines.evaluation_pipeline import EvaluationPipeline

ARTIFACT = NB_PATHS.models_dir / "__MODEL_FILE__"
if not ARTIFACT.exists():
    # Notebook exécuté seul : on construit un index de travail dans le bac à sable plutôt que de
    # lire `artifacts/models/`, qui appartient à `make train`.
    print("artefact absent de outputs/notebooks : construction d'un index de travail")
    bootstrap = build_model(CONFIG.model_dump())
    bootstrap.fit(DOCUMENTS, CALIBRATION)
    bootstrap.save(ARTIFACT)

TRAINED_MODEL = load_model(ARTIFACT, config=CONFIG.model_dump())
EVALUATION = EvaluationPipeline(CONFIG, paths=NB_PATHS).run()
print("statut :", EVALUATION.status)
for message in EVALUATION.messages:
    print("-", message)
""",
            context,
        ),
        _text(
            """
from IPython.display import Markdown

REPORT_PATH = NB_PATHS.reports_dir / "__REPORT_NAME__"
display(Markdown(REPORT_PATH.read_text(encoding="utf-8")))
""",
            context,
        ),
        _md(
            "## 2. Résultats par segment\n\n"
            "Un rappel global peut cacher un segment excellent et un autre en échec : ce sont les "
            "écarts qui dictent le travail suivant."
        ),
        _text(
            """
RESULT = EVALUATION.payload
segments = pd.DataFrame(RESULT.segments).T.astype(float)
columns = [name for name in ("n_questions", "recall_at_5", "mrr", "abstention_rate")
           if name in segments.columns]
segments[columns].round(3).sort_values("recall_at_5")
""",
            context,
        ),
        _text(
            """
per_question = RESULT.per_question.copy()
per_question["trouvé"] = per_question["first_relevant_rank"] > 0
answerable_rows = per_question[per_question["answer_type"] != "unanswerable"]
by_difficulty = (
    answerable_rows.groupby("difficulty")
    .agg(
        questions=("query_id", "size"),
        rappel_moyen=("recall_at_5", "mean"),
        rang_moyen=("first_relevant_rank", lambda values: values[values > 0].mean()),
        non_trouvées=("trouvé", lambda values: int((~values).sum())),
    )
    .round(3)
)
by_difficulty
""",
            context,
        ),
        _md(
            "### 2.1 Les questions perdues, une par une\n\n"
            "Un taux d'échec n'est pas une explication. On regarde les questions concernées : "
            "partagent-elles une formulation, un thème, une source ?"
        ),
        _text(
            """
lost = answerable_rows[~answerable_rows["trouvé"]].sort_values("top_score", ascending=False)
print(f"{len(lost)} questions sans passage pertinent dans le top-5 "
      f"sur {len(answerable_rows)} questions répondables")
display(lost[["question", "difficulty", "intent", "n_gold_docs", "top_score"]].head(10))
""",
            context,
        ),
        _text(
            """
# Ce que le système répond sur une question perdue : cela distingue « mauvais classement » (le bon
# passage existe mais arrive trop loin) de « réponse construite de toutes pièces ».
if len(lost):
    record = lost.iloc[0]
    answer = TRAINED_MODEL.answer(str(record["question"]), 5, query_id=str(record["query_id"]))
    print("question       :", record["question"])
    print("réponse        :", answer.text or "(abstention)")
    print("meilleur score :", round(answer.top_score, 3))
    display(pd.DataFrame([passage.to_row() for passage in answer.passages]))
""",
            context,
        ),
        _md(
            "## 3. Le comportement d'abstention\n\n"
            "C'est la mesure que l'on oublie : un assistant qui répond toujours obtient de bons "
            "scores de rappel et invente des réponses sur les questions hors corpus."
        ),
        _text(
            """
out_of_corpus = per_question[per_question["answer_type"] == "unanswerable"]
refusal_rate = float(out_of_corpus["abstained"].mean()) if len(out_of_corpus) else float("nan")
coverage = 1.0 - float(answerable_rows["abstained"].mean())
print(f"questions hors corpus            : {len(out_of_corpus)}")
print(f"refus corrects (taux)            : {refusal_rate:.1%}")
print(f"couverture des questions servies : {coverage:.1%}")
print(f"fausses réponses restantes       : {int((out_of_corpus['abstained'] == 0).sum())}")
""",
            context,
        ),
        _text(
            """
from IPython.display import Image

figures = sorted(NB_PATHS.figures_dir.glob("*.png"))
print("figures produites par l'évaluation :", [figure.name for figure in figures])
for figure in figures[:2]:
    display(Image(filename=str(figure)))
""",
            context,
        ),
        _md(
            "## 4. Recommandations\n\n"
            "Chaque recommandation est reliée à une mesure produite plus haut : ce qui n'est pas "
            "mesuré ici doit être mesuré avant d'être corrigé."
        ),
        _text(
            """
# On classe les difficultés **répondables** : le segment hors corpus n'appelle pas un meilleur
# modèle mais une abstention, qui est mesurée juste au-dessus.
difficulty = {
    key.split("=", 1)[1]: row
    for key, row in segments.iterrows()
    if key.startswith("difficulty=") and key != "difficulty=hors_corpus"
}
worst = min(difficulty, key=lambda name: difficulty[name]["recall_at_5"])
recommendations = [
    {
        "action": f"Traiter en priorité le segment « {worst} »",
        "preuve": (f"rappel@5 = {difficulty[worst]['recall_at_5']:.3f} sur "
                   f"{int(difficulty[worst]['n_questions'])} questions"),
        "effort": "moyen (modèle plus riche, latence supérieure)",
    },
    {
        "action": "Arbitrer le seuil de refus avec le métier",
        "preuve": f"{refusal_rate:.1%} de refus corrects pour {coverage:.1%} de couverture",
        "effort": "faible (décision, pas de code)",
    },
    {
        "action": "Enrichir le corpus là où les questions sont perdues",
        "preuve": f"{len(lost)} questions sans passage pertinent dans le top-5",
        "effort": "élevé (travail éditorial)",
    },
]
pd.DataFrame(recommendations)
""",
            context,
        ),
        _insight(
            [
                "Le verdict contractuel se lit avec son référentiel : un rappel@5 ne vaut que "
                "comparé au plancher trivial mesuré sur le même corpus.",
                "Facile, paraphrase et multi-document appellent trois correctifs différents : un "
                "réglage global les dégraderait tous les trois.",
                "L'abstention est une fonctionnalité mesurée, pas un aveu d'échec : refuser une "
                "question hors corpus est le comportement attendu d'un assistant documentaire.",
            ]
        ),
    ]
    return write_notebook(destination / "06_error_analysis.ipynb", cells)


#: Builders, dans l'ordre de lecture.
BUILDERS = (
    build_01_eda,
    build_02_validation,
    build_03_preprocessing,
    build_04_model_exploration,
    build_05_training,
    build_06_error_analysis,
)


def build_all(context: NotebookContext, destination: Path) -> list[Path]:
    """Build the six notebooks of a text project.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory of the project.

    Returns:
        The written notebook paths, in order.
    """
    return [builder(context, destination) for builder in BUILDERS]
