"""Construction des six notebooks pédagogiques de la modality **tabulaire**.

Un notebook ``.ipynb`` est un JSON structuré : le générer avec un template texte produit des
fichiers fragiles (échappements, virgules, cellules cassées). Chaque cellule est donc
construite en Python via :mod:`nbformat`, ce qui garantit un notebook **valide et exécutable**.

Les six notebooks imposés par le standard du dépôt :

``01_eda``                exploration descriptive complète (profil, distributions, cible, corrélations)
``02_validation``         contrats Pandera, dont un **échec volontaire** commenté
``03_preprocessing``      nettoyage, feature engineering et démonstration d'anti-fuite
``04_model_exploration``  comparaison d'algorithmes, hyperparamètres, courbe d'apprentissage
``05_training``           entraînement de production (callbacks, artefacts, stabilité)
``06_error_analysis``     analyse d'erreurs, arbitrage de seuil et recommandations concrètes

Chaque notebook est **autonome** (il recalcule ce dont il a besoin), écrit dans
``outputs/notebooks`` (ignoré par git) et se termine par une synthèse actionnable.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from nbformat.notebooknode import NotebookNode

from tools.scaffold.context import names_with_role
from tools.scaffold.utils_notebooks import (
    NotebookContext,
    code_cell,
    markdown_cell,
    write_notebook,
)

#: Nombre de lignes utilisé par les notebooks : réaliste mais exécutable en quelques secondes.
NB_ROWS = 1500
#: Époques utilisées par les notebooks pour une stack sans boucle d'époques (sans effet).
NB_EPOCHS_DEFAULT = 3
#: Époques utilisées par les notebooks pour une stack itérative (deep learning).
NB_EPOCHS_DEEP = 12

#: Tâches qui apprennent à partir d'une cible.
SUPERVISED_TASKS = frozenset({"binary", "multiclass", "regression", "forecasting", "ranking"})

#: Tâches dont la métrique primaire se maximise.
MAXIMIZE_TASKS = frozenset({"binary", "multiclass", "clustering", "ranking", "anomaly"})


# ---------------------------------------------------------------------------------------
# Helpers de rendu
# ---------------------------------------------------------------------------------------
def _tokens(context: NotebookContext) -> dict[str, str]:
    """Build the substitution table used by every code template.

    Les cellules sont écrites avec des jetons ``__XXX__`` plutôt qu'avec des f-strings : le
    code des notebooks contient beaucoup d'accolades (dicts, f-strings internes) et les
    doubler partout rendrait les templates illisibles.

    Args:
        context: Notebook context.

    Returns:
        Mapping of token to replacement text.
    """
    spec = context.spec
    data = spec.data
    return {
        "__ROWS__": str(NB_ROWS),
        "__SEED__": str(data.seed),
        "__TITLE__": spec.title,
        "__PROJECT__": spec.project_slug,
        "__DATASET__": data.dataset_name,
        "__TARGET__": str(data.target or "target"),
        "__HAS_TARGET__": repr(data.target is not None),
        "__ID__": str(data.id_column or "id"),
        "__HAS_ID__": repr(bool(data.id_column)),
        "__TIME__": str(data.time_column or ""),
        "__HAS_TIME__": repr(bool(data.time_column)),
        "__TASK__": str(spec.metrics.task),
        "__PRIMARY__": str(spec.metrics.primary),
        "__ALGO__": str(spec.model.algorithm),
        # Le libellé de la baseline dépend de la tâche : classe majoritaire vs médiane.
        "__BASELINE_LABEL__": (
            "baseline (médiane)" if _is_regression(context) else "baseline (classe majoritaire)"
        ),
        "__MODEL_CLASS__": context.model_class,
        "__FRAMEWORK__": spec.stack_key,
        "__DROP__": context.py_list(
            names_with_role(data.columns, ["identifier", "timestamp", "metadata", "group"])
        ),
        "__NUMERIC__": context.py_list(context.numeric_features),
        "__CATEGORICAL__": context.py_list(context.categorical_features),
        "__MIN_PRIMARY__": repr(spec.metrics.min_primary),
        "__SUPERVISED__": repr(spec.metrics.task in SUPERVISED_TASKS),
        "__PARAM_GRID__": repr(_param_grid(spec)),
        # Budget d'entraînement réduit dans les notebooks : sans effet sur les stacks sans
        # époques (sklearn, boosting), il borne l'exécution des stacks itératives (deep learning).
        "__NB_EPOCHS__": str(NB_EPOCHS_DEEP if context.stack.epochs_based else NB_EPOCHS_DEFAULT),
    }


def _is_regression(context: NotebookContext) -> bool:
    """Return ``True`` when the project predicts a continuous target.

    Args:
        context: Notebook context.

    Returns:
        ``True`` for a regression / forecasting task, ``False`` otherwise.
    """
    return str(getattr(context.spec.metrics, "task", "")) in {"regression", "forecasting"}


def _is_clustering(context: NotebookContext) -> bool:
    """Return ``True`` when the project segments data without any target.

    Args:
        context: Notebook context.

    Returns:
        ``True`` for a clustering task, ``False`` otherwise.
    """
    return str(getattr(context.spec.metrics, "task", "")) == "clustering"


def _render(source: str, context: NotebookContext) -> str:
    """Substitute the manifest tokens inside a code template.

    Args:
        source: Raw template source.
        context: Notebook context.

    Returns:
        The rendered source.
    """
    rendered = source
    for token, value in _tokens(context).items():
        rendered = rendered.replace(token, value)
    return rendered


def _code(source: str, context: NotebookContext) -> NotebookNode:
    """Create a rendered code cell."""
    return code_cell(_render(source, context))


def _md(source: str) -> NotebookNode:
    """Create a Markdown cell (no substitution: the prose is written directly)."""
    return markdown_cell(source)


def _insight(lines: Sequence[str], title: str = "Ce qu'il faut retenir") -> NotebookNode:
    """Create the Markdown insight cell that follows an output.

    Args:
        lines: Bullet points (French).
        title: Section title.

    Returns:
        The Markdown cell.
    """
    body = "\n".join(f"- {line}" for line in lines)
    return markdown_cell(f"**{title}**\n\n{body}")


def _objectives(context: NotebookContext, items: Sequence[str]) -> NotebookNode:
    """Render the pedagogical objectives cell of a notebook.

    Args:
        context: Notebook context.
        items: Notebook-specific objectives.

    Returns:
        The Markdown cell.
    """
    shared = list(context.spec.learning_objectives[:3])
    lines = "\n".join(f"1. {item}" for item in items)
    extra = ""
    if shared:
        extra = "\n\n**Objectifs transverses du dépôt**\n\n" + "\n".join(
            f"- {item}" for item in shared
        )
    return markdown_cell(f"## Objectifs pédagogiques\n\n{lines}{extra}")


# ---------------------------------------------------------------------------------------
# Blocs de code partagés
# ---------------------------------------------------------------------------------------
SETUP = """
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

# --- Racine du projet ---------------------------------------------------------------------------
# Le notebook s'exécute depuis `notebooks/` : on remonte d'un cran pour pouvoir importer `src`.
PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from loguru import logger  # noqa: E402

from src.schemas.config import validate_config  # noqa: E402
from src.utils.paths import ProjectPaths  # noqa: E402

# --- Réglages d'affichage -----------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25})
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 170)
logger.remove()
logger.add(sys.stderr, level="WARNING")

# --- Configuration : exactement celle de `python -m src.main` ------------------------------------
# Les notebooks travaillent sur un échantillon réduit (__ROWS__ lignes) : l'exécution complète
# reste sous la minute, tout en conservant des distributions réalistes.
NB_ROWS = __ROWS__

GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=str(PROJECT_ROOT / "conf"), version_base=None):
    CONFIG = validate_config(
        compose(
            config_name="config",
            overrides=[
                "mode=train",
                f"data.n_samples={NB_ROWS}",
                "seed=__SEED__",
                "log_level=WARNING",
                "++train.epochs=__NB_EPOCHS__",
                "train.callbacks.progress_bar=false",
            ],
        )
    )

PATHS = ProjectPaths.from_root(PROJECT_ROOT)
# Les notebooks écrivent leurs artefacts dans `outputs/notebooks` (ignoré par git) afin de ne
# jamais écraser ceux produits par `make train`.
NB_PATHS = ProjectPaths.from_root(PROJECT_ROOT / "outputs" / "notebooks").ensure()

print(f"Projet            : {CONFIG.project.name}")
print(f"Tâche             : {CONFIG.metrics.task}")
print(f"Métrique primaire : {CONFIG.metrics.primary} (seuil cible : __MIN_PRIMARY__)")
print(f"Cible             : {CONFIG.data.target or 'aucune (apprentissage non supervisé)'}")
print(f"Algorithme        : {CONFIG.model.algorithm} ({CONFIG.model.name})")
print(f"Lignes (notebook) : {NB_ROWS}")
"""

LOAD_RAW = """
from src.data.generators import SyntheticDataGenerator
from src.data.loaders import RawDataLoader

raw_path = PATHS.data_file(CONFIG.data.dataset_name)
if raw_path.exists():
    # Cas nominal : le dataset a été généré par `make data`, on passe par le loader validant.
    raw = RawDataLoader(PATHS, dataset_name=CONFIG.data.dataset_name).load()
    print(f"Dataset lu depuis {raw_path.relative_to(PROJECT_ROOT)}")
else:
    # Le notebook reste exécutable sur un clone frais : on génère en mémoire.
    raw = SyntheticDataGenerator(n_samples=NB_ROWS, seed=CONFIG.data.seed).generate()
    print("data/raw vide : génération synthétique en mémoire (`make data` la persiste)")

raw = raw.head(NB_ROWS).reset_index(drop=True)
print(f"shape = {raw.shape}")
raw.head()
"""

PREPARE = '''
from src.data.loaders import DatasetSplitter, feature_target_split
from src.features.build_features import FeatureBuilder, select_feature_columns, split_by_dtype
from src.preprocessing.pipelines import PreprocessingPipeline


def prepare_matrices(frame: pd.DataFrame, config: Any) -> dict[str, Any]:
    """Reproduce what ``TrainPipeline`` does, on the notebook-sized dataset.

    La fonction reprend **exactement** l'enchaînement de production : split → feature
    engineering (appris sur train uniquement) → preprocessing (appris sur train uniquement).
    C'est ce qui rend les chiffres de ce notebook comparables à ceux de `make train`.

    Args:
        frame: Raw dataset.
        config: Validated application configuration.

    Returns:
        Mapping with splits, fitted objects and model-ready matrices.
    """
    target = config.data.target
    drop_columns = list(config.data.drop_columns)

    splitter = DatasetSplitter.from_config(config.model_dump(), seed=config.seed)
    splits = splitter.split(frame, target=target)

    builder = FeatureBuilder.from_config(config.model_dump(), target=target)
    if builder.recipes:
        builder.fit(splits.train)
    enriched = {
        "train": builder.transform(splits.train),
        "val": None if splits.val is None else builder.transform(splits.val),
        "test": builder.transform(splits.test),
    }

    train_frame = enriched["train"]
    feature_columns = select_feature_columns(train_frame, drop_columns=drop_columns, target=target)
    numeric, categorical = split_by_dtype(train_frame, feature_columns)
    explicit = config.preprocessing.model_dump().get("columns") or {}
    numeric = list(explicit.get("numeric") or numeric)
    categorical = list(explicit.get("categorical") or categorical)

    pipeline = PreprocessingPipeline(
        numeric_features=numeric,
        categorical_features=categorical,
        config=config.preprocessing.model_dump(),
        target=target,
    )
    X_train_frame, y_train = feature_target_split(train_frame, target, drop_columns)
    X_train = pipeline.fit_transform(X_train_frame, y_train)

    def project(split: pd.DataFrame | None) -> tuple[pd.DataFrame | None, Any]:
        if split is None:
            return None, None
        _, labels = feature_target_split(split, target, drop_columns)
        return pipeline.transform(split.loc[:, X_train_frame.columns]), labels

    X_val, y_val = project(enriched["val"])
    X_test, y_test = project(enriched["test"])

    return {
        "splits": splits,
        "enriched": enriched,
        "builder": builder,
        "pipeline": pipeline,
        "numeric": numeric,
        "categorical": categorical,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": list(pipeline.feature_names_out),
    }


PREPARED = prepare_matrices(raw, CONFIG)
print("train :", PREPARED["X_train"].shape)
print("val   :", None if PREPARED["X_val"] is None else PREPARED["X_val"].shape)
print("test  :", PREPARED["X_test"].shape)
print(f"features livrées au modèle : {len(PREPARED['feature_names'])}")
PREPARED["X_train"].head()
'''

FIT_MODEL = """
from src.models import build_model

MODEL = build_model(CONFIG, feature_names=PREPARED["feature_names"])
FIT_RESULT = MODEL.fit(
    PREPARED["X_train"],
    PREPARED["y_train"],
    X_val=PREPARED["X_val"],
    y_val=PREPARED["y_val"],
    callbacks=[],
)
print(MODEL.summary())
pd.Series(FIT_RESULT.metrics, name="métrique").to_frame("valeur")
"""


# ---------------------------------------------------------------------------------------
# 01 — Exploration
# ---------------------------------------------------------------------------------------
def build_01_eda(context: NotebookContext, destination: Path) -> Path:
    """Build ``01_eda.ipynb``.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    spec = context.spec
    data = spec.data
    cells: list[NotebookNode] = [
        _md(
            f"""# 01 — Exploration des données (EDA)

**Projet** : {spec.title}
**Cas métier** : {spec.business.problem}
**Jeu de données** : `{data.dataset_name}` — {data.title}

> {data.description}
"""
        ),
        _objectives(
            context,
            [
                "Charger un jeu de données tabulaire et en établir le profil (types, manquants, doublons).",
                "Lire une distribution : détecter déséquilibre, outliers et colinéarité **avant** de modéliser.",
                "Relier chaque observation statistique à une conséquence métier ou de modélisation.",
                "Produire les figures qui serviront de référence dans les notebooks suivants.",
            ],
        ),
        _md(
            """## 0. Environnement

Toute la configuration vient de **Hydra** (`conf/`) : aucune valeur métier n'est codée en dur
dans ce notebook. Si `data/raw` est vide, le générateur synthétique du projet prend le relais
(voir `make data`)."""
        ),
        _code(SETUP, context),
        _insight(
            [
                "`CONFIG` est l'objet **Pydantic** validé : une clé incohérente échoue ici, pas en production.",
                "Les notebooks travaillent sur un échantillon réduit pour rester rapides ; `make train` utilise `data.n_samples` complet.",
                "`NB_PATHS` isole les écritures du notebook dans `outputs/notebooks`.",
            ],
            title="Pourquoi ce bloc d'initialisation",
        ),
        _md(
            """## 1. Chargement et premier contact

On ne regarde jamais un dataset sans vérifier trois choses : sa **forme** (lignes x colonnes),
ses **types** (un numérique lu comme texte casse tout) et ses **premières lignes** (les valeurs
ont-elles du sens métier ?)."""
        ),
        _code(LOAD_RAW, context),
        _insight(
            [
                f"Le contrat `{data.dataset_name}` est documenté dans `data/README.md` : chaque colonne y a une signification métier.",
                "Les identifiants et horodatages ne sont **pas** des features : ils servent à tracer et à splitter.",
            ]
        ),
        _code(
            """
from src.data.schemas import describe_schema, validation_report

# Types déclarés (contrat Pandera) vs types réellement lus : toute divergence est un signal.
contract = describe_schema("raw")
observed = pd.DataFrame({"dtype_lu": {str(k): str(v) for k, v in raw.dtypes.items()}})
contract.join(observed)[["dtype", "dtype_lu", "nullable", "unique", "checks"]]
""",
            context,
        ),
        _insight(
            [
                "La colonne `dtype` vient du **contrat**, `dtype_lu` de la source : elles doivent correspondre.",
                "Les `checks` (bornes, valeurs autorisées) sont la mémoire des règles métier — ils seront testés au notebook 02.",
            ]
        ),
        _code(
            """
report = validation_report(raw)
summary = pd.DataFrame(
    {
        "indicateur": ["lignes", "colonnes", "cellules manquantes", "taux de manquants", "mémoire (Ko)"],
        "valeur": [
            report["n_rows"],
            report["n_columns"],
            report["missing_cells"],
            f"{report['missing_rate']:.2%}",
            round(report["memory_kb"], 1),
        ],
    }
)
summary
""",
            context,
        ),
        _insight(
            [
                "Un taux de manquants global faible peut cacher une colonne très incomplète : regarder **par colonne**.",
                "La mémoire indique si le dataset tient en RAM (sinon : pyarrow, chunking ou échantillonnage).",
            ]
        ),
        _md(
            "## 2. Valeurs manquantes\n\nOù, combien, et surtout : **manquant au hasard ou pas** ? Un manquant informatif (ex. score de satisfaction non renseigné par les clients mécontents) est un signal, pas seulement un problème technique."
        ),
        _code(
            """
missing = raw.isna().sum()
missing_frame = (
    pd.DataFrame({"manquants": missing, "taux": (missing / len(raw)).round(4)})
    .loc[lambda frame: frame["manquants"] > 0]
    .sort_values("manquants", ascending=False)
)
missing_frame
""",
            context,
        ),
        _code(
            """
if missing_frame.empty:
    print("Aucune valeur manquante dans cet échantillon.")
else:
    fig, axis = plt.subplots(figsize=(7.5, 0.55 * len(missing_frame) + 1.6))
    axis.barh(missing_frame.index[::-1], missing_frame["taux"][::-1] * 100, color="#d1495b")
    axis.set_xlabel("Cellules manquantes (%)")
    axis.set_title("Valeurs manquantes par colonne")
    fig.tight_layout()
    plt.show()
""",
            context,
        ),
        _insight(
            [
                "L'imputation doit être **apprise sur le train** (moyenne/médiane/constante) puis appliquée aux autres splits.",
                "Ajouter un indicateur binaire « valeur manquante » est souvent rentable quand le manquant est informatif.",
                f"Notes du générateur : {'; '.join(data.notes[:2]) if data.notes else 'valeurs manquantes injectées volontairement (cas pédagogique)'}.",
            ]
        ),
        _md("## 3. Distributions numériques"),
        _code(
            """
numeric_columns = [
    column for column in raw.columns if pd.api.types.is_numeric_dtype(raw[column])
]
numeric_columns = [column for column in numeric_columns if column != CONFIG.data.target]

n_plots = len(numeric_columns)
n_cols = 3
n_rows = int(np.ceil(n_plots / n_cols)) if n_plots else 1
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.7 * n_rows))
for axis, column in zip(np.atleast_1d(axes).ravel(), numeric_columns, strict=False):
    raw[column].hist(bins=30, ax=axis, color="#005f73", edgecolor="white")
    axis.set_title(column, fontsize=9)
    axis.tick_params(labelsize=7)
for axis in np.atleast_1d(axes).ravel()[len(numeric_columns):]:
    axis.axis("off")
fig.suptitle("Distributions des variables numériques", y=1.005)
fig.tight_layout()
plt.show()
""",
            context,
        ),
        _code(
            """
raw[numeric_columns].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T.round(2)
""",
            context,
        ),
        _insight(
            [
                "Une distribution très asymétrique (max ≫ p99) justifie un **winsorising** ou un `log1p` plutôt qu'une suppression d'outliers.",
                "Des échelles hétérogènes (euros, Go, unités) imposent un **scaling** pour les modèles sensibles à la distance (SVM, k-NN, réseaux).",
                "Comparer `mean` et `50%` : un écart important signale une queue lourde.",
            ]
        ),
        _md("## 4. Variables catégorielles"),
        _code(
            """
categorical_columns = [
    column
    for column in raw.columns
    if not pd.api.types.is_numeric_dtype(raw[column])
    and column not in [*CONFIG.data.drop_columns, str(CONFIG.data.target)]
]

for column in categorical_columns:
    counts = raw[column].astype(str).value_counts()
    print(f"--- {column} ({len(counts)} modalités) ---")
    print((counts / len(raw)).map("{:.1%}".format).to_string())
""",
            context,
        ),
        _insight(
            [
                "Une modalité ultra-rare (< 1 %) doit être regroupée dans un bucket `rare` : sinon l'encodage one-hot crée des colonnes quasi vides et instables.",
                "Une cardinalité élevée (identifiants, codes postaux) appelle un **target encoding** régularisé plutôt qu'un one-hot.",
            ]
        ),
    ]

    if data.target:
        if _is_regression(context):
            from tools.scaffold.notebooks.regression import target_cells

            cells += target_cells(context)
        else:
            cells += _target_cells(context)
    elif _is_clustering(context):
        # Aucune cible : la section 5 porte sur l'échelle des variables, la structure visible en
        # ACP et les colonnes de diagnostic (métadonnées, jamais des features).
        from tools.scaffold.notebooks.clustering import structure_cells

        cells += structure_cells(context)

    cells += [
        _md("## 6. Colinéarité et structure"),
        _code(
            """
correlation = raw[numeric_columns].corr(numeric_only=True)
fig, axis = plt.subplots(figsize=(6.6, 5.4))
image = axis.imshow(correlation.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
axis.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right", fontsize=7)
axis.set_yticks(range(len(correlation.index)), correlation.index, fontsize=7)
for row in range(correlation.shape[0]):
    for column in range(correlation.shape[1]):
        value = correlation.iloc[row, column]
        axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6,
                  color="black" if abs(value) < 0.6 else "white")
fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
axis.set_title("Corrélations de Pearson (variables numériques)")
fig.tight_layout()
plt.show()
""",
            context,
        ),
        _insight(
            [
                "Deux features corrélées à > 0.9 n'apportent presque rien ensemble : en garder une simplifie le modèle et son explication.",
                "Les arbres sont robustes à la colinéarité ; les modèles linéaires/régularisés voient leurs coefficients devenir instables.",
                "La corrélation ne capture pas les relations **non linéaires** : la vérifier par des graphes cible vs feature.",
            ]
        ),
        _code(
            """
# Outliers : comptage par la règle de l'IQR (1.5 x écart interquartile).
rows = []
for column in numeric_columns:
    series = raw[column].dropna()
    if series.empty:
        continue
    low, high = series.quantile([0.25, 0.75])
    iqr = high - low
    outliers = int(((series < low - 1.5 * iqr) | (series > high + 1.5 * iqr)).sum())
    rows.append({"colonne": column, "outliers_iqr": outliers, "part": outliers / max(len(series), 1)})
outlier_frame = pd.DataFrame(rows).sort_values("outliers_iqr", ascending=False)
outlier_frame.head(8).round(4)
""",
            context,
        ),
        _insight(
            [
                "La règle IQR **signale**, elle ne tranche pas : un outlier peut être un client légitime (grand compte, pic saisonnier).",
                "Le winsorising (clip aux quantiles 1-99 %) conserve les lignes et les labels, contrairement à la suppression.",
            ]
        ),
        _code(
            """
# Intégrité : unicité de la clé et doublons complets.
key = CONFIG.data.id_column
duplicates = int(raw.duplicated().sum())
key_duplicates = int(raw[key].duplicated().sum()) if key and key in raw.columns else 0
print(f"doublons complets            : {duplicates}")
print(f"doublons sur la clé '{key}' : {key_duplicates}")
unique_keys = raw[key].nunique() if key in raw.columns else "n/a"
print(f"identifiants uniques         : {unique_keys} / {len(raw)}")
""",
            context,
        ),
        _md("## 7. Synthèse de l'exploration"),
        _insight(
            [str(item) for item in data.insights] or ["Aucun insight déclaré dans le manifeste."],
            title="Lectures clés de ce jeu de données",
        ),
        _md(
            """### Décisions de modélisation issues de l'EDA

| Observation | Décision |
| --- | --- |
| Valeurs manquantes localisées | Imputation apprise sur le train (notebook 03) |
| Échelles hétérogènes | Scaling numérique obligatoire |
| Outliers légitimes | Winsorising plutôt que suppression |
| Modalités rares | Regroupement `rare` avant encodage |
| Colinéarité | Surveiller l'importance des features (notebook 04) |

**Suite** : `02_validation.ipynb` transforme ces observations en **contrats exécutables** (Pandera).
"""
        ),
    ]
    return write_notebook(destination / "01_eda.ipynb", cells)


def _target_cells(context: NotebookContext) -> list[NotebookNode]:
    """Build the target-analysis cells (supervised tasks only).

    Args:
        context: Notebook context.

    Returns:
        The cells to insert in notebook 01.
    """
    data = context.spec.data
    positive_rate = data.positive_rate
    rate_text = (
        f"taux de classe positive attendu ≈ {positive_rate:.0%}"
        if positive_rate is not None
        else "répartition à observer"
    )
    return [
        _md(
            f"## 5. La cible\n\n{rate_text}. C'est la variable la plus importante du notebook : son déséquilibre conditionne le choix des métriques."
        ),
        _code(
            """
target = CONFIG.data.target
distribution = raw[target].value_counts(normalize=True).sort_index()
counts = raw[target].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
counts.plot.bar(ax=axes[0], color=["#0a9396", "#d1495b"] if len(counts) == 2 else "#0a9396", edgecolor="white")
axes[0].set_title(f"Distribution de `{target}` (effectifs)")
axes[0].set_ylabel("nombre de lignes")
distribution.plot.bar(ax=axes[1], color="#ee9b00", edgecolor="white")
axes[1].set_title("Répartition relative")
axes[1].set_ylabel("part")
for axis in axes:
    axis.tick_params(axis="x", rotation=0)
fig.tight_layout()
plt.show()

print(f"classes : {list(counts.index)}")
print(f"classe majoritaire : {float(distribution.max()):.1%} -> accuracy triviale")
""",
            context,
        ),
        _insight(
            [
                "L'accuracy d'un modèle « classe majoritaire » est déjà élevée : ce n'est **pas** une preuve de performance.",
                "En déséquilibre, piloter sur le **rappel de la classe positive** et le **PR AUC** plutôt que sur l'accuracy ou le ROC AUC.",
                "Le ré-échantillonnage (SMOTE, pondération) se décide ici, et s'applique **uniquement sur le train**.",
            ]
        ),
        _code(
            """
# Signal par variable catégorielle : écart de taux de cible entre modalités.
target = CONFIG.data.target
rows = []
for column in categorical_columns:
    grouped = raw.groupby(raw[column].astype(str))[target].agg(["mean", "count"])
    if grouped.empty or grouped["mean"].nunique() < 2:
        continue
    spread = float(grouped["mean"].max() - grouped["mean"].min())
    rows.append(
        {
            "variable": column,
            "écart_de_taux": round(spread, 4),
            "modalité_la_plus_risquée": grouped["mean"].idxmax(),
            "taux_max": round(float(grouped["mean"].max()), 4),
            "modalité_la_moins_risquée": grouped["mean"].idxmin(),
            "taux_min": round(float(grouped["mean"].min()), 4),
        }
    )
signal_frame = pd.DataFrame(rows).sort_values("écart_de_taux", ascending=False)
signal_frame
""",
            context,
        ),
        _insight(
            [
                "Un écart de taux important entre modalités = signal exploitable directement par le modèle.",
                "Un écart proche de 0 ne veut pas dire « inutile » : la variable peut interagir avec une autre.",
                "Ces écarts se lisent en **points de pourcentage** : les présenter ainsi aux métiers évite les malentendus.",
            ]
        ),
        _code(
            """
# Signal par variable numérique : taux de cible par quartile.
target = CONFIG.data.target
plots = [column for column in numeric_columns if raw[column].nunique() > 4][:6]
fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.4))
for axis, column in zip(np.atleast_1d(axes).ravel(), plots, strict=False):
    buckets = pd.qcut(raw[column], q=4, duplicates="drop")
    grouped = raw.groupby(buckets, observed=True)[target].mean()
    grouped.plot.bar(ax=axis, color="#005f73", edgecolor="white")
    axis.set_title(f"Taux de cible par quartile — {column}", fontsize=8.5)
    axis.set_ylabel("")
    axis.tick_params(labelsize=7, axis="x", rotation=20)
for axis in np.atleast_1d(axes).ravel()[len(plots):]:
    axis.axis("off")
fig.tight_layout()
plt.show()
""",
            context,
        ),
        _insight(
            [
                "Une relation **monotone** (le taux monte avec la variable) est facilement apprise, y compris par un modèle linéaire.",
                "Une relation en **U** impose des transformations (binning, splines) ou un modèle non linéaire.",
                "Le binning par quantiles est aussi une feature déclarée dans `conf/preprocessing/default.yaml` (recette `bin`).",
            ]
        ),
    ]


# ---------------------------------------------------------------------------------------
# 02 — Contrats de données
# ---------------------------------------------------------------------------------------
def build_02_validation(context: NotebookContext, destination: Path) -> Path:
    """Build ``02_validation.ipynb`` (including a deliberate validation failure).

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    spec = context.spec
    cells: list[NotebookNode] = [
        _md(
            f"""# 02 — Contrats de données avec Pandera

**Projet** : {spec.title}
**Objectif** : transformer les règles métier découvertes en EDA en **contrats exécutables**.

Un contrat de données vaut par ses deux propriétés :

1. il **accepte** les données conformes (sinon il bloque la production pour rien) ;
2. il **refuse** les données corrompues avec un message exploitable (sinon il ne sert à rien).

Ce notebook démontre les deux — y compris en **provoquant volontairement** des échecs.
"""
        ),
        _objectives(
            context,
            [
                "Lire un schéma `DataFrameModel` comme une documentation (types, bornes, catégories, unicité).",
                "Provoquer et interpréter un échec de validation (`failure_cases`).",
                "Distinguer les trois contrats du cycle de vie : brut, transformé, inférence.",
                "Utiliser le mode `lazy` pour remonter toutes les erreurs d'un coup.",
            ],
        ),
        _code(SETUP, context),
        _code(LOAD_RAW, context),
        _insight(
            [
                "Le `RawDataLoader` applique déjà le contrat au chargement : une source déviante échoue **ici**, pas en entraînement.",
                "`validate=False` existe pour inspecter des données cassées sans exception (diagnostic).",
            ]
        ),
        _md(
            """## 1. Le contrat des données brutes

Trois schémas cohabitent dans `src/data/schemas.py` :

| Schéma | Appliqué sur | Politique |
| --- | --- | --- |
| `RawDataSchema` | `data/raw` juste après chargement | strict, types et bornes imposés |
| `ProcessedDataSchema` | matrice livrée au modèle | 100 % numérique, zéro NaN |
| `InferenceDataSchema` | toute requête de prédiction | colonnes optionnelles, nulls tolérés |"""
        ),
        _code(
            """
from src.data.schemas import RawDataSchema, schema_to_markdown

display(Markdown(schema_to_markdown("raw")))
""",
            context,
        ),
        _insight(
            [
                "Cette table est **générée depuis le code** : elle ne peut pas diverger de l'implémentation.",
                "`nullable=False` sur la clé et `unique` garantissent l'intégrité du jeu (pas de doublon silencieux).",
                "Les `checks` (bornes, `isin`) sont la traduction directe des règles métier du README.",
            ]
        ),
        _code(
            """
from src.data.schemas import describe_schema

describe_schema("raw")
""",
            context,
        ),
        _md("## 2. Validation nominale : le contrat doit passer"),
        _code(
            """
from src.data.schemas import validate_frame, validation_report

validated = validate_frame(raw, "raw")
profile = validation_report(validated)
print(f"validation OK | lignes={profile['n_rows']} | colonnes={profile['n_columns']}")
print(f"              | cellules manquantes={profile['missing_cells']}")
validated.head(3)
""",
            context,
        ),
        _insight(
            [
                "La coercition (`coerce=True`) absorbe les différences Parquet/CSV : un entier lu comme flottant reste valide.",
                "Un contrat qui ne passe **jamais** en local est un contrat mal calibré — le vérifier fait partie du travail.",
            ]
        ),
        _md(
            """## 3. Échecs volontaires — la partie la plus utile du notebook

On corrompt **délibérément** le dataset pour vérifier que le contrat mord. Chaque cellule
isole une violation ; l'exception est capturée puis affichée avec ses `failure_cases`."""
        ),
        _code(
            '''
try:
    import pandera.pandas as pa
except ModuleNotFoundError:  # pandera < 0.26
    import pandera as pa

SchemaViolation = (pa.errors.SchemaError, pa.errors.SchemaErrors)


def show_violation(label: str, frame: pd.DataFrame) -> None:
    """Validate a deliberately corrupted frame and explain the failure.

    Args:
        label: Human readable name of the injected corruption.
        frame: Corrupted dataset.
    """
    try:
        RawDataSchema.validate(frame, lazy=True)
    except SchemaViolation as error:
        cases = getattr(error, "failure_cases", None)
        print(f"[REFUSÉ] {label}")
        if cases is not None:
            print(cases.head(6).to_string(index=False))
        else:
            print(error)
        return
    print(f"[ACCEPTÉ — ATTENTION] {label} : le contrat ne couvre pas ce cas")
''',
            context,
        ),
        _code(
            """
numeric_bounded = [
    name
    for name, column in RawDataSchema.to_schema().columns.items()
    if any(getattr(check, "name", "") in {"ge", "greater_than_or_equal_to", "in_range"}
           for check in (getattr(column, "checks", []) or []))
]
column = numeric_bounded[0]
corrupted = raw.copy()
corrupted.loc[corrupted.index[:5], column] = 10_000_000
show_violation(f"valeur hors bornes sur `{column}` (10 000 000)", corrupted)
""",
            context,
        ),
        _insight(
            [
                "`failure_cases` donne la **colonne**, le **check** et les **valeurs** en échec : le diagnostic est immédiat.",
                "En production, cette erreur doit faire échouer le run (fail fast) plutôt que d'entraîner un modèle sur des données fausses.",
            ]
        ),
        _code(
            """
corrupted = raw.copy()
corrupted["colonne_non_declaree"] = 0
show_violation("colonne non déclarée (strict=True)", corrupted)
""",
            context,
        ),
        _code(
            """
corrupted = raw.drop(columns=[raw.columns[-1]])
show_violation("colonne manquante", corrupted)
""",
            context,
        ),
        _code(
            """
# Une colonne catégorielle **textuelle** contrainte par `isin` : c'est la seule que l'on puisse
# corrompre avec une chaîne. Un booléen codé 0/1 porte souvent un `isin` lui aussi, mais pandas
# refuse d'y écrire du texte (TypeError) — ce n'est pas la violation que l'on veut démontrer.
textual_dtypes = ("str", "object", "category")
categorical_checked = []
for name, column in RawDataSchema.to_schema().columns.items():
    dtype = str(getattr(column, "dtype", "")).lower()
    if not any(marker in dtype for marker in textual_dtypes):
        continue
    checks = getattr(column, "checks", []) or []
    if any(getattr(check, "name", "") == "isin" for check in checks):
        categorical_checked.append(str(name))
if categorical_checked:
    column = categorical_checked[0]
    corrupted = raw.copy()
    corrupted.loc[corrupted.index[:3], column] = "modalite_inexistante"
    show_violation(f"catégorie hors liste sur `{column}`", corrupted)
else:
    print("Aucune colonne contrainte par `isin` dans ce schéma.")
""",
            context,
        ),
        _insight(
            [
                "Une nouvelle modalité non déclarée est le bug silencieux le plus fréquent après un changement de SI amont.",
                "Deux réponses possibles : mettre à jour le contrat (évolution légitime) ou refuser (régression).",
            ]
        ),
    ]

    if spec.data.id_column:
        cells += [
            _code(
                """
corrupted = raw.copy()
corrupted.loc[corrupted.index[1], CONFIG.data.id_column] = corrupted.loc[corrupted.index[0], CONFIG.data.id_column]
show_violation(f"clé dupliquée sur `{CONFIG.data.id_column}`", corrupted)
""",
                context,
            ),
            _code(
                """
corrupted = raw.copy()
corrupted.loc[corrupted.index[0], CONFIG.data.id_column] = None
show_violation(f"clé nulle sur `{CONFIG.data.id_column}`", corrupted)
""",
                context,
            ),
            _insight(
                [
                    "Une clé dupliquée crée une **fuite** entre splits : la même observation peut se retrouver en train et en test.",
                    "C'est pourquoi `assert_no_overlap()` est testé dans `tests/test_loaders.py`.",
                ]
            ),
        ]

    cells += [
        _md("## 4. Mode `lazy` : tout remonter d'un coup"),
        _code(
            """
column = numeric_bounded[0]
corrupted = raw.copy()
corrupted.loc[corrupted.index[:20], column] = -1_000_000
corrupted.loc[corrupted.index[20:40], column] = 1_000_000
show_violation(f"40 violations sur `{column}` (mode lazy)", corrupted)
""",
            context,
        ),
        _insight(
            [
                "Sans `lazy`, pandera s'arrête à la première erreur : on découvre les problèmes un par un.",
                "Avec `lazy`, le rapport complet permet de corriger le flux amont en une fois (configurable via `data.validation.lazy`).",
            ]
        ),
        _md("## 5. Contrat des données transformées"),
        _code(
            """
from src.data.schemas import ProcessedDataSchema

try:
    ProcessedDataSchema.validate(raw)
except SchemaViolation as error:
    print("[REFUSÉ comme attendu] la matrice brute n'est pas numérique :")
    print(str(error)[:320])

numeric_matrix = raw.select_dtypes(include=[np.number]).dropna().head(50)
print("matrice numérique valide ->", ProcessedDataSchema.validate(numeric_matrix).shape)
""",
            context,
        ),
        _insight(
            [
                "Ce contrat est la **dernière ligne de défense** avant le modèle : aucune feature texte, aucun NaN, aucun infini.",
                "Il est appliqué automatiquement par `TrainPipeline` quand `data.validation.processed: true`.",
            ]
        ),
        _md("## 6. Contrat d'inférence : tolérant mais pas laxiste"),
        _code(
            """
from src.data.schemas import InferenceDataSchema

payload = raw.drop(columns=[CONFIG.data.target]).head(20).copy() if CONFIG.data.target else raw.head(20).copy()
payload.iloc[0, 0] = None                      # valeur manquante tolérée
payload["colonne_du_client"] = "web"           # colonne supplémentaire tolérée
print("payload accepté ->", InferenceDataSchema.validate(payload).shape)

bad_payload = payload.copy()
if numeric_bounded:
    bad_payload[numeric_bounded[0]] = "pas-un-nombre"
try:
    InferenceDataSchema.validate(bad_payload)
except SchemaViolation as error:
    print("[REFUSÉ] type incohérent :", str(error)[:200])
""",
            context,
        ),
        _insight(
            [
                "La cible est absente d'un payload d'inférence : le contrat ne doit pas l'exiger (`required=False`).",
                "Tolérer les nulls et les colonnes en trop **sans** renoncer aux checks de type : c'est l'équilibre recherché.",
                "Une requête partielle est corrigée par le preprocessing ; une requête incohérente est refusée avec un message clair.",
            ]
        ),
        _md(
            """## 7. Où la validation s'exécute vraiment

| Emplacement | Contrat | Déclencheur |
| --- | --- | --- |
| `RawDataLoader.load()` | `RawDataSchema` | chaque chargement de données |
| `TrainPipeline._preprocess()` | `ProcessedDataSchema` | avant entraînement |
| `Predictor.predict()` | `InferenceDataSchema` | chaque requête de prédiction |
| `tests/test_data_schemas.py` | les trois | chaque commit (`make test`) |

### Checklist à reproduire sur un nouveau projet

1. Écrire le schéma **avant** le code de chargement (le contrat guide l'implémentation).
2. Ajouter un test d'acceptation (données conformes) **et** un test de refus (données corrompues).
3. Versionner le schéma avec le code : une évolution de contrat est un changement d'API.
4. Journaliser les `failure_cases` : ce sont eux qui font gagner du temps en incident.
"""
        ),
    ]
    return write_notebook(destination / "02_validation.ipynb", cells)


# ---------------------------------------------------------------------------------------
# 03 — Preprocessing
# ---------------------------------------------------------------------------------------
def build_03_preprocessing(context: NotebookContext, destination: Path) -> Path:
    """Build ``03_preprocessing.ipynb`` (feature engineering + leakage demonstration).

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    spec = context.spec
    cells: list[NotebookNode] = [
        _md(
            f"""# 03 — Preprocessing et feature engineering

**Projet** : {spec.title}
**Objectif** : passer d'un dataset brut à une matrice numérique propre, **sans fuite de données**.

Ordre imposé (et testé dans `tests/test_preprocessing.py`) :

1. **split** d'abord (train / val / test),
2. **feature engineering** appris sur le train,
3. **preprocessing** (imputation, outliers, encodage, scaling) appris sur le train,
4. application aux autres splits — jamais l'inverse.
"""
        ),
        _objectives(
            context,
            [
                "Construire des features **déclaratives** (recettes YAML) plutôt que du code ad hoc.",
                "Comprendre pourquoi toute statistique apprise sur le test fausse l'évaluation.",
                "Configurer imputation, winsorising, encodage et scaling depuis `conf/preprocessing`.",
                "Persister et recharger le pipeline : l'inférence doit reproduire exactement l'entraînement.",
            ],
        ),
        _code(SETUP, context),
        _code(LOAD_RAW, context),
        _md("## 1. Splitter avant toute transformation"),
        _code(
            """
from src.data.loaders import DatasetSplitter, assert_no_overlap

splitter = DatasetSplitter.from_config(CONFIG.model_dump(), seed=CONFIG.data.seed)
splits = splitter.split(raw, target=CONFIG.data.target)

frames = [frame for frame in (splits.train, splits.val, splits.test) if frame is not None]
assert_no_overlap(*frames, key=CONFIG.data.id_column)
pd.DataFrame([splits.sizes], index=["lignes"]).T.assign(
    part=lambda frame: (frame["lignes"] / len(raw)).map("{:.1%}".format)
)
""",
            context,
        ),
        _insight(
            [
                "`assert_no_overlap` est un garde-fou de fuite : aucune clé ne doit apparaître dans deux splits.",
                "La stratification conserve la répartition de la cible (indispensable en déséquilibre).",
                "Le split est **configuré** (`train.split.*`), jamais codé en dur : un sweep Hydra peut le faire varier.",
            ]
        ),
        _md("## 2. Feature engineering déclaratif"),
        _code(
            """
from src.features.build_features import FeatureBuilder

builder = FeatureBuilder.from_config(CONFIG.model_dump(), target=CONFIG.data.target)
recipes = pd.DataFrame(builder.describe())
recipes if not recipes.empty else print("Aucune recette déclarée dans conf/preprocessing/default.yaml")
""",
            context,
        ),
        _insight(
            [
                "Chaque recette est **déclarée en YAML** : ajouter une feature ne demande aucune modification de code.",
                "Le builder **refuse** une recette qui lirait la cible : la fuite la plus classique est bloquée à la configuration.",
                "Les recettes `bin` et `group_stat` apprennent des statistiques sur le train → elles nécessitent `fit()`.",
            ]
        ),
        _code(
            """
if builder.recipes:
    builder.fit(splits.train)          # statistiques apprises sur le train uniquement
    enriched_train = builder.transform(splits.train)
    enriched_test = builder.transform(splits.test)
    created = [column for column in enriched_train.columns if column not in splits.train.columns]
    print(f"features créées : {created}")
    enriched_train[created].describe().T.round(3)
else:
    enriched_train, enriched_test, created = splits.train, splits.test, []
    print("Aucune feature dérivée.")
""",
            context,
        ),
        _code(
            """
# Stabilité d'une feature dérivée entre train et test : un bon signal doit se transporter.
if created:
    column = created[0]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0))
    enriched_train[column].hist(bins=25, ax=axes[0], color="#005f73", edgecolor="white")
    axes[0].set_title(f"train — {column}", fontsize=9)
    enriched_test[column].hist(bins=25, ax=axes[1], color="#ee9b00", edgecolor="white")
    axes[1].set_title(f"test — {column}", fontsize=9)
    fig.tight_layout()
    plt.show()
    print(f"moyenne train = {enriched_train[column].mean():.4f}")
    print(f"moyenne test  = {enriched_test[column].mean():.4f}")
else:
    print("Aucune feature dérivée à comparer.")
""",
            context,
        ),
        _insight(
            [
                "Deux distributions très différentes entre train et test signalent une **dérive** (ou un split temporel).",
                "Une feature dont la moyenne change beaucoup apportera peu en production : à surveiller (voir `mlops/model-monitoring`).",
            ]
        ),
        _md("## 3. Le pipeline de preprocessing"),
        _code(PREPARE, context),
        _insight(
            [
                "`prepare_matrices()` reproduit exactement `TrainPipeline` : les chiffres du notebook sont comparables à `make train`.",
                "Le nombre de features livrées dépasse celui des colonnes brutes : l'encodage one-hot **explose** les modalités.",
            ]
        ),
        _code(
            """
report = PREPARED["pipeline"].report.to_dict()
pd.DataFrame(
    {
        "indicateur": [
            "lignes en entrée",
            "colonnes en entrée",
            "features en sortie",
            "manquants avant",
            "manquants après",
            "encodeur",
            "scaler",
        ],
        "valeur": [
            report["n_rows_in"],
            report["n_columns_in"],
            report["n_output_features"],
            report["missing_before"],
            report["missing_after"],
            report["encoder"],
            report["scaler"],
        ],
    }
)
""",
            context,
        ),
        _insight(
            [
                "`missing_before` > 0 et `missing_after` = 0 : l'imputation a fait son travail (et c'est **tracé**).",
                "Le rapport est sérialisable : il alimente les runs MLflow/W&B et le rapport d'évaluation.",
            ]
        ),
        _code(
            """
from src.data.schemas import ProcessedDataSchema

X_train, X_test = PREPARED["X_train"], PREPARED["X_test"]
print("contrat processed :", "OK" if ProcessedDataSchema.validate(X_train) is not None else "KO")
print("dtypes uniques    :", sorted({str(dtype) for dtype in X_train.dtypes}))
print("NaN restants      :", int(X_train.isna().to_numpy().sum()))
X_train.iloc[:5, :6]
""",
            context,
        ),
        _md("## 4. Démonstration d'anti-fuite"),
        _code(
            """
# Le scaler est appris sur le train : le test est centré/réduit avec les statistiques du TRAIN.
train_mean = float(PREPARED["X_train"].to_numpy().mean())
test_mean = float(PREPARED["X_test"].to_numpy().mean())

# Contre-exemple volontaire : un pipeline appris sur train+test (fuite) puis comparé.
leaky = PreprocessingPipeline(
    numeric_features=PREPARED["numeric"],
    categorical_features=PREPARED["categorical"],
    config=CONFIG.preprocessing.model_dump(),
    target=CONFIG.data.target,
)
combined = pd.concat(
    [
        PREPARED["enriched"]["train"].drop(columns=list(CONFIG.data.drop_columns) + ([CONFIG.data.target] if CONFIG.data.target else []), errors="ignore"),
        PREPARED["enriched"]["test"].drop(columns=list(CONFIG.data.drop_columns) + ([CONFIG.data.target] if CONFIG.data.target else []), errors="ignore"),
    ],
    ignore_index=True,
)
_ = leaky.fit_transform(combined)

print(f"moyenne train (pipeline correct) : {train_mean:+.4f}")
print(f"moyenne test  (pipeline correct) : {test_mean:+.4f}  <- proche de 0 sans être 0 : normal")
print(f"colonnes du pipeline fuyard      : {len(leaky.feature_names_out)}")
print("Le pipeline fuyard a vu le test : ses statistiques en dépendent (illustration uniquement).")
""",
            context,
        ),
        _insight(
            [
                "La moyenne du **train** est ≈ 0 (le scaler l'a centré) ; celle du **test** ne l'est pas exactement : c'est la signature d'un pipeline honnête.",
                "Un `StandardScaler` ajusté sur train+test injecte de l'information du test dans le train → métriques surévaluées.",
                "Même règle pour l'imputation, le winsorising, le target encoding et le binning par quantiles.",
            ]
        ),
        _md("## 5. Les transformateurs maison, un par un"),
        _code(
            """
from src.preprocessing.transformers import (
    ColumnSelector,
    DataFrameScaler,
    Log1pTransformer,
    OutlierClipper,
    RareCategoryGrouper,
    TypeCaster,
)

numeric_columns = PREPARED["numeric"]
categorical_columns = PREPARED["categorical"]
frame = PREPARED["enriched"]["train"]

clipper = OutlierClipper((0.01, 0.99), columns=numeric_columns[:3]).fit(frame)
print("bornes apprises (train) :")
for column, bounds in list(clipper.bounds_.items())[:3]:
    print(f"  {column:<28} [{bounds[0]:.2f} ; {bounds[1]:.2f}]")
clipped = clipper.transform(frame)
print("cellules écrêtées       :", clipper.report)
""",
            context,
        ),
        _code(
            """
if categorical_columns:
    grouper = RareCategoryGrouper(min_frequency=0.01, max_categories=10).fit(frame[categorical_columns])
    kept = {column: len(values) for column, values in grouper.kept_.items()}
    grouped = grouper.transform(frame[categorical_columns])
    print("modalités conservées :", kept)
    print("modalités `rare`     :", int((grouped.astype(str) == "rare").to_numpy().sum()))
else:
    print("Aucune colonne catégorielle dans ce projet.")
""",
            context,
        ),
        _code(
            """
# Comparaison des stratégies de scaling sur une variable asymétrique.
column = numeric_columns[0]
strategies = ["none", "standard", "minmax", "robust"]
fig, axes = plt.subplots(1, len(strategies), figsize=(4.0 * len(strategies), 2.8), sharey=False)
for axis, strategy in zip(axes, strategies, strict=False):
    scaled = DataFrameScaler(strategy).fit_transform(frame[[column]])
    scaled[column].hist(bins=30, ax=axis, color="#0a9396", edgecolor="white")
    axis.set_title(
        f"{strategy}  |  mean={scaled[column].mean():+.2f}  std={scaled[column].std():.2f}",
        fontsize=9,
    )
fig.suptitle(f"Effet du scaling sur `{column}`", y=1.04)
fig.tight_layout()
plt.show()
""",
            context,
        ),
        _insight(
            [
                "`standard` : moyenne 0, écart-type 1 — le choix par défaut des modèles linéaires et des réseaux.",
                "`minmax` : ramène dans [0, 1] mais **sensibilise aux outliers** (le max définit l'échelle).",
                "`robust` : utilise médiane et écart interquartile — recommandé quand la queue est lourde.",
                "Les arbres (forêts, boosting) sont **invariants** au scaling : le scaler ne change rien à leurs splits.",
            ]
        ),
        _code(
            """
# TypeCaster : projette un payload sur le contrat (utile en inférence,
# où des colonnes non déclarées arrivent du système appelant).
caster = TypeCaster(numeric_columns=numeric_columns[:3], categorical_columns=categorical_columns[:2]).fit(frame)
noisy = frame.assign(colonne_inutile="x")
projected = caster.transform(noisy)
print("colonnes du payload bruité :", list(noisy.columns)[-3:])
print("colonnes après projection  :", list(projected.columns))
print("dtypes                     :", {str(k): str(v) for k, v in projected.dtypes.items()})
""",
            context,
        ),
        _md("## 6. Persistance : l'inférence doit reproduire l'entraînement"),
        _code(
            """
test_frame = PREPARED["enriched"]["test"].loc[:, PREPARED["numeric"] + PREPARED["categorical"]]
artifact = PREPARED["pipeline"].save(NB_PATHS.models_dir / "preprocessing_notebook.joblib")
restored = PreprocessingPipeline.load(artifact)

before = PREPARED["pipeline"].transform(test_frame)
after = restored.transform(test_frame)

print("artefact      :", artifact.relative_to(PROJECT_ROOT))
print("features      :", before.shape[1])
print("reproductible :", bool(np.allclose(before.to_numpy(), after.to_numpy())))
""",
            context,
        ),
        _insight(
            [
                "Le pipeline est **sérialisé avec le modèle** : en inférence, aucune statistique n'est recalculée.",
                "Si `reproductible` est faux, l'artefact est corrompu ou les versions de librairies diffèrent (d'où la fiche modèle).",
            ]
        ),
        _md(
            """## 7. Synthèse

| Étape | Où c'est configuré | Test associé |
| --- | --- | --- |
| Split | `conf/train/default.yaml` (`split`) | `tests/test_loaders.py` |
| Features dérivées | `conf/preprocessing/default.yaml` (`features`) | `tests/test_preprocessing.py` |
| Imputation / outliers / encodage / scaling | `conf/preprocessing/default.yaml` | `tests/test_preprocessing.py` |
| Contrat de sortie | `src/data/schemas.py` (`ProcessedDataSchema`) | `tests/test_data_schemas.py` |

**Suite** : `04_model_exploration.ipynb` compare les algorithmes sur cette matrice.
"""
        ),
    ]
    return write_notebook(destination / "03_preprocessing.ipynb", cells)


# ---------------------------------------------------------------------------------------
# 04 — Exploration de modèles
# ---------------------------------------------------------------------------------------
def _param_grid(spec: Any) -> dict[str, list[Any]]:
    """Return the hyper-parameter grid declared for the notebooks (optional).

    La grille est **déclarée dans le manifeste** (``extras.notebook_param_grid``) : elle dépend
    de l'algorithme de la stack, donc ni du notebook ni du code applicatif.

    Args:
        spec: Validated project manifest.

    Returns:
        Mapping of parameter name to candidate values (empty when not declared).
    """
    grid = spec.extras.get("notebook_param_grid") or {}
    return {str(key): list(values) for key, values in dict(grid).items()}


def build_04_model_exploration(context: NotebookContext, destination: Path) -> Path:
    """Build ``04_model_exploration.ipynb``.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    spec = context.spec
    grid = _param_grid(spec)
    cells: list[NotebookNode] = [
        _md(
            f"""# 04 — Exploration et comparaison de modèles

**Projet** : {spec.title}
**Modèle configuré** : `{spec.model.algorithm}` ({spec.model.display_name})
**Pourquoi ce choix** : {spec.model.rationale}

Ce notebook répond à la question que tout relecteur pose : *« pourquoi cet algorithme ? »*.
La réponse doit être **chiffrée** : baseline, comparaison d'algorithmes, sensibilité aux
hyperparamètres, courbe d'apprentissage et importance des features.
"""
        ),
        _objectives(
            context,
            [
                "Commencer par une **baseline** : sans elle, aucune performance n'est interprétable.",
                "Comparer les algorithmes disponibles dans la stack via `available_algorithms()`.",
                "Distinguer sous-apprentissage et sur-apprentissage avec une courbe d'apprentissage.",
                "Lire l'importance des features pour décider quoi instrumenter ensuite.",
            ],
        ),
        _code(SETUP, context),
        _code(LOAD_RAW, context),
        _code(PREPARE, context),
        _md("## 1. La baseline d'abord"),
        _code(
            """
from src.evaluation.evaluator import Evaluator
from src.models import build_model
from src.training.losses_metrics import MetricCalculator, MetricInputs

BASE_MODEL = build_model(CONFIG, feature_names=PREPARED["feature_names"], algorithm="__ALGO__")
_ = BASE_MODEL.fit(PREPARED["X_train"], PREPARED["y_train"], X_val=PREPARED["X_val"], y_val=PREPARED["y_val"], callbacks=[])

EVALUATOR = Evaluator.from_config(BASE_MODEL, CONFIG.model_dump(), NB_PATHS)
baseline_metrics = EVALUATOR.compare_to_baseline(PREPARED["X_val"], PREPARED["y_val"], baseline="dummy")

calculator = MetricCalculator(task=CONFIG.metrics.task, metrics=CONFIG.metrics.all_metrics)
model_metrics = calculator.evaluate(
    MetricInputs(
        y_true=PREPARED["y_val"],
        y_pred=BASE_MODEL.predict(PREPARED["X_val"]),
        y_proba=BASE_MODEL.predict_proba(PREPARED["X_val"]) if BASE_MODEL.supports_proba else None,
        X=PREPARED["X_val"],
    )
)

comparison = pd.DataFrame(
    {
        "modèle": ["__ALGO__", "__BASELINE_LABEL__"],
        CONFIG.metrics.primary: [model_metrics.get(CONFIG.metrics.primary, float("nan")),
                                 baseline_metrics.get(f"baseline_{CONFIG.metrics.primary}", float("nan"))],
    }
)
comparison.round(4)
""",
            context,
        ),
        _insight(
            [
                "Un modèle qui ne bat pas la baseline n'apporte **aucune** valeur : il ne doit pas aller en production.",
                f"La métrique primaire du projet est `{spec.metrics.primary}` (sens : {spec.metrics.direction}) ; le seuil de qualité déclaré est {spec.metrics.min_primary}.",
                "La baseline est recalculée sur le **même** split de validation : la comparaison est loyale.",
            ]
        ),
        _md("## 2. Comparaison des algorithmes de la stack (réglages par défaut)"),
        _code(
            """
from src.models.factory import available_algorithms

ALGORITHMS = available_algorithms(CONFIG.metrics.task)
print(f"{len(ALGORITHMS)} algorithmes disponibles pour la tâche '{CONFIG.metrics.task}' :")
print(ALGORITHMS)

rows = []
# Comparaison **loyale** : chaque algorithme est construit avec ses réglages par défaut.
# Les `model.params` configurés sont réglés pour l'algorithme retenu ; les réutiliser
# ailleurs fausserait le classement (`max_leaf_nodes` bride une forêt aléatoire).
# Le modèle configuré et réglé est, lui, évalué en section 1.
for algorithm in ALGORITHMS:
    try:
        candidate = build_model(
            CONFIG, feature_names=PREPARED["feature_names"], algorithm=algorithm, params={}
        )
        result = candidate.fit(
            PREPARED["X_train"], PREPARED["y_train"],
            X_val=PREPARED["X_val"], y_val=PREPARED["y_val"], callbacks=[],
        )
        values = calculator.evaluate(
            MetricInputs(
                y_true=PREPARED["y_val"],
                y_pred=candidate.predict(PREPARED["X_val"]),
                y_proba=candidate.predict_proba(PREPARED["X_val"]) if candidate.supports_proba else None,
                X=PREPARED["X_val"],
            )
        )
        rows.append(
            {
                "algorithme": algorithm,
                CONFIG.metrics.primary: values.get(CONFIG.metrics.primary, float("nan")),
                **{name: values.get(name, float("nan")) for name in CONFIG.metrics.secondary[:3]},
                "secondes": round(result.duration_seconds, 2),
            }
        )
    except Exception as error:  # un algorithme incompatible ne doit pas casser l'exploration
        rows.append({"algorithme": algorithm, CONFIG.metrics.primary: float("nan"), "secondes": float("nan")})
        print(f"  ! {algorithm} ignoré : {type(error).__name__}: {error}")

# Le meilleur en tête, quel que soit le sens de la métrique (AUC : décroissant, RMSE : croissant).
meilleur_d_abord = str(CONFIG.metrics.direction) == "minimize"
ranking = (
    pd.DataFrame(rows)
    .sort_values(CONFIG.metrics.primary, ascending=meilleur_d_abord)
    .reset_index(drop=True)
)
ranking.round(4)
""",
            context,
        ),
        _insight(
            [
                "Le classement se lit **avec** le temps d'entraînement : un gain de 0.005 pour 20x plus lent est rarement rentable.",
                "Chaque algorithme est comparé **à réglages par défaut** : un algorithme perdant ici peut gagner une fois réglé (section 3).",
                "Un écart faible entre algorithmes indique que la limite vient des **données**, pas du modèle.",
                "Les valeurs manquantes (NaN) signalent une métrique non définie pour l'algorithme (ex. probabilités absentes).",
            ]
        ),
        _code(
            """
fig, axis = plt.subplots(figsize=(8.2, 0.45 * len(ranking) + 1.8))
axis.barh(ranking["algorithme"][::-1], ranking[CONFIG.metrics.primary][::-1], color="#005f73")
baseline_value = baseline_metrics.get(f"baseline_{CONFIG.metrics.primary}", float("nan"))
if np.isfinite(baseline_value):
    axis.axvline(baseline_value, color="#d1495b", linestyle="--", linewidth=1.2, label="baseline")
    axis.legend(fontsize=8)
axis.set_xlabel(f"{CONFIG.metrics.primary} (validation)")
axis.set_title("Comparaison des algorithmes")
fig.tight_layout()
plt.show()
""",
            context,
        ),
    ]

    if grid:
        cells += [
            _md("## 3. Sensibilité aux hyperparamètres"),
            _code(
                """
import itertools

GRID = __PARAM_GRID__
combinations = list(itertools.product(*[GRID[name] for name in GRID]))
print(f"{len(combinations)} combinaisons testées sur {list(GRID)}")

rows = []
for combination in combinations:
    params = dict(zip(GRID, combination, strict=True))
    candidate = build_model(CONFIG, feature_names=PREPARED["feature_names"], params=params)
    _ = candidate.fit(
        PREPARED["X_train"], PREPARED["y_train"],
        X_val=PREPARED["X_val"], y_val=PREPARED["y_val"], callbacks=[],
    )
    values = calculator.evaluate(
        MetricInputs(
            y_true=PREPARED["y_val"],
            y_pred=candidate.predict(PREPARED["X_val"]),
            y_proba=candidate.predict_proba(PREPARED["X_val"]) if candidate.supports_proba else None,
        )
    )
    row = {str(name): str(value) for name, value in params.items()}
    row[CONFIG.metrics.primary] = values.get(CONFIG.metrics.primary, float("nan"))
    rows.append(row)

grid_frame = pd.DataFrame(rows).sort_values(
    CONFIG.metrics.primary, ascending=str(CONFIG.metrics.direction) == "minimize"
)
grid_frame.round(4)
""",
                context,
            ),
            _insight(
                [
                    "La meilleure ligne du tableau est un **candidat**, pas une décision : elle est choisie sur la validation.",
                    "Un modèle très complexe qui n'améliore pas la validation sur-apprend : revenir au plus simple.",
                    "Ces valeurs appartiennent dans `conf/model/default.yaml` (`model.params`), jamais dans le code.",
                ]
            ),
        ]

    cells += [
        _md(f"## {4 if grid else 3}. Courbe d'apprentissage — faut-il plus de données ?"),
        _code(
            """
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
curve = []
for fraction in fractions:
    size = max(int(len(PREPARED["X_train"]) * fraction), 30)
    subsample = PREPARED["X_train"].iloc[:size]
    labels = None if PREPARED["y_train"] is None else PREPARED["y_train"].iloc[:size]
    candidate = build_model(CONFIG, feature_names=PREPARED["feature_names"])
    train_result = candidate.fit(subsample, labels, callbacks=[])
    train_score = float(train_result.metrics.get(CONFIG.metrics.primary, float("nan")))
    val_score = float("nan")
    if PREPARED["X_val"] is not None and PREPARED["y_val"] is not None:
        val_values = calculator.evaluate(
            MetricInputs(
                y_true=PREPARED["y_val"],
                y_pred=candidate.predict(PREPARED["X_val"]),
                y_proba=candidate.predict_proba(PREPARED["X_val"]) if candidate.supports_proba else None,
            )
        )
        val_score = float(val_values.get(CONFIG.metrics.primary, float("nan")))
    curve.append({"lignes": size, "train": train_score, "validation": val_score})

curve_frame = pd.DataFrame(curve)
fig, axis = plt.subplots(figsize=(6.6, 3.6))
axis.plot(curve_frame["lignes"], curve_frame["train"], marker="o", label="train")
axis.plot(curve_frame["lignes"], curve_frame["validation"], marker="s", label="validation")
axis.set_xlabel("lignes d'entraînement")
axis.set_ylabel(CONFIG.metrics.primary)
axis.set_title("Courbe d'apprentissage")
axis.legend(fontsize=8)
fig.tight_layout()
plt.show()
curve_frame.round(4)
""",
            context,
        ),
        _insight(
            [
                "Courbes **écartées et plates** ⇒ sur-apprentissage : régulariser plutôt qu'ajouter des données.",
                "Courbes **encore montantes** ⇒ plus de données aiderait vraiment (argument budgétaire chiffré).",
                "Un train parfait (1.0) avec une validation médiocre est le signal d'une fuite ou d'un modèle trop complexe.",
            ]
        ),
        _md(f"## {5 if grid else 4}. Importance des features"),
        _code(
            """
importance = EVALUATOR.feature_importance(PREPARED["X_val"], y=PREPARED["y_val"])
if importance.empty:
    print("Ce modèle n'expose pas d'importance (permutation indisponible sur ce split).")
else:
    display(importance.head(12).round(4))
    top = importance.head(12).iloc[::-1]
    fig, axis = plt.subplots(figsize=(7.6, 0.4 * len(top) + 1.4))
    axis.barh(top["feature"], top["importance"], color="#0a9396")
    axis.set_xlabel(f"importance ({top['method'].iloc[0] if 'method' in top else 'naine'})")
    axis.set_title("Top 12 des features")
    fig.tight_layout()
    plt.show()
""",
            context,
        ),
        _insight(
            [
                "L'importance **native** (Gini, coefficients) est rapide mais biaisée vers les variables à forte cardinalité.",
                "L'importance par **permutation** est plus lente mais model-agnostic : c'est celle à citer en comité.",
                "Une feature dominante impose une vigilance particulière sur sa disponibilité et sa dérive en production.",
            ]
        ),
        _md(
            f"""## Synthèse — choix du modèle

- Algorithme retenu par la configuration : **`{spec.model.algorithm}`** ({spec.model.display_name}).
- Justification documentée : {spec.model.rationale}
- Alternatives évaluées ici : {", ".join(f"`{name}`" for name in spec.model.alternatives) or "voir le tableau de comparaison"}.

**Règle de décision** : on retient le modèle le plus **simple** dont la métrique primaire est à
moins de ~1 point du meilleur, et dont le coût d'inférence est compatible avec `{spec.business.cadence}`.

**Suite** : `05_training.ipynb` entraîne le modèle retenu dans les conditions de production.
"""
        ),
    ]
    return write_notebook(destination / "04_model_exploration.ipynb", cells)


# ---------------------------------------------------------------------------------------
# 05 — Entraînement
# ---------------------------------------------------------------------------------------
def build_05_training(context: NotebookContext, destination: Path) -> Path:
    """Build ``05_training.ipynb``.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    spec = context.spec
    cells: list[NotebookNode] = [
        _md(
            f"""# 05 — Entraînement dans les conditions de production

**Projet** : {spec.title}
**Ce que fait `make train`** : charger → valider → splitter → features → preprocessing →
entraîner → métriques de validation → **persister** (modèle, fiche, métriques, splits).

Ce notebook exécute exactement le même chemin, mais de façon instrumentée : chaque brique est
visible, mesurable et rejouable.
"""
        ),
        _objectives(
            context,
            [
                "Piloter un entraînement par **objets** (Trainer, callbacks) plutôt que par un script monolithique.",
                "Lire un `TrainingOutcome` : métriques, durée, historique, artefacts.",
                "Mesurer la **stabilité** d'un modèle (plusieurs graines) avant de conclure.",
                "Vérifier le garde-fou de qualité déclaré en configuration.",
            ],
        ),
        _code(SETUP, context),
        _code(LOAD_RAW, context),
        _code(PREPARE, context),
        _md("## 1. Le modèle, construit depuis la configuration"),
        _code(FIT_MODEL, context),
        _insight(
            [
                "`build_model` est la **seule** porte d'entrée : le reste du projet ne connaît que `BaseModel`.",
                "Changer de stack (XGBoost, PyTorch, …) ne modifie ni le trainer, ni l'évaluateur, ni l'inférence.",
                "Les hyperparamètres viennent de `conf/model/default.yaml` : aucun n'est codé en dur ici.",
            ]
        ),
        _md("## 2. Callbacks : instrumenter sans polluer la boucle d'entraînement"),
        _code(
            """
from src.training.callbacks import (
    EarlyStoppingCallback,
    LoggingCallback,
    MetricHistoryCallback,
    MetricThresholdCallback,
)
from src.training.trainer import Trainer, TrainingData

TRAINING_DATA = TrainingData(
    X_train=PREPARED["X_train"],
    y_train=PREPARED["y_train"],
    X_val=PREPARED["X_val"],
    y_val=PREPARED["y_val"],
    feature_names=PREPARED["feature_names"],
    task=CONFIG.metrics.task,
)

CALLBACKS = [
    LoggingCallback(every=1),
    MetricHistoryCallback(),
    EarlyStoppingCallback(
        monitor=CONFIG.train.early_stopping.monitor,
        patience=CONFIG.train.early_stopping.patience,
        mode=CONFIG.train.early_stopping.mode,
    ),
    MetricThresholdCallback(
        monitor=f"val_{CONFIG.metrics.primary}",
        threshold=float(CONFIG.metrics.min_primary or 0.0),
        mode="max" if CONFIG.metrics.direction == "maximize" else "min",
    ),
]
[callback.name for callback in CALLBACKS]
""",
            context,
        ),
        _code(
            """
TRAINER = Trainer(
    MODEL,
    config=CONFIG.model_dump(),
    paths=NB_PATHS,
    metric_names=CONFIG.metrics.all_metrics,
    task=CONFIG.metrics.task,
    callbacks=CALLBACKS,
)
OUTCOME = TRAINER.train(TRAINING_DATA)

metrics_frame = pd.DataFrame(
    {"métrique": list(OUTCOME.metrics), "valeur": [OUTCOME.metrics[name] for name in OUTCOME.metrics]}
).sort_values("valeur", ascending=False)
print(f"durée : {OUTCOME.duration_seconds:.2f}s | artefacts : {len(OUTCOME.artifacts)}")
metrics_frame.round(4).reset_index(drop=True)
""",
            context,
        ),
        _insight(
            [
                "Les métriques préfixées `val_` viennent du **split de validation** : elles ne sont jamais calculées sur le test.",
                "Le test reste vierge jusqu'au notebook 06 — c'est la condition d'une estimation honnête.",
                "La durée est tracée : un entraînement qui double soudainement signale une dérive de données ou de configuration.",
            ]
        ),
        _md("## 3. Artefacts produits"),
        _code(
            """
import json

artifacts = pd.DataFrame(
    {"artefact": list(OUTCOME.artifacts), "chemin": [OUTCOME.artifacts[name] for name in OUTCOME.artifacts]}
)
display(artifacts)

card_path = OUTCOME.artifacts.get("model_card")
if card_path:
    card = json.loads(Path(card_path).read_text(encoding="utf-8"))
    print("fiche modèle — clés :", sorted(card))
    print("features attendues  :", len(card["feature_names"]))
    print("versions librairies :", card["library_versions"])
""",
            context,
        ),
        _insight(
            [
                "La **fiche modèle** (features, paramètres, versions, métriques) rend un artefact reproductible et auditable.",
                "Sans elle, un `.joblib` retrouvé dans six mois est inutilisable : impossible de savoir quoi lui donner en entrée.",
                "Les artefacts du notebook sont écrits dans `outputs/notebooks` ; `make train` écrit dans `artifacts/`.",
            ]
        ),
        _md("## 4. Stabilité : plusieurs graines, même conclusion ?"),
        _code(
            """
from src.training.losses_metrics import MetricCalculator, MetricInputs

scores = []
for seed in (7, 21, 42):
    candidate = build_model(CONFIG, feature_names=PREPARED["feature_names"])
    candidate.random_state = seed
    _ = candidate.fit(PREPARED["X_train"], PREPARED["y_train"], X_val=PREPARED["X_val"], y_val=PREPARED["y_val"], callbacks=[])
    values = MetricCalculator(task=CONFIG.metrics.task, metrics=[CONFIG.metrics.primary]).evaluate(
        MetricInputs(
            y_true=PREPARED["y_val"],
            y_pred=candidate.predict(PREPARED["X_val"]),
            y_proba=candidate.predict_proba(PREPARED["X_val"]) if candidate.supports_proba else None,
            # Les métriques internes d'un clustering (silhouette, Davies-Bouldin) ont besoin de la
            # matrice de features : sans `X`, elles sont simplement ignorées par le registre.
            X=PREPARED["X_val"],
        )
    )
    scores.append(values.get(CONFIG.metrics.primary, float("nan")))

scores_array = np.asarray(scores, dtype="float64")
stability = pd.DataFrame(
    {
        "graine": [7, 21, 42],
        CONFIG.metrics.primary: scores_array.round(4),
    }
)
mean = float(np.nanmean(scores_array))
std = float(np.nanstd(scores_array, ddof=1)) if len(scores_array) > 1 else 0.0
print(f"moyenne = {mean:.4f} | ecart-type = {std:.4f}")
print(f"intervalle +/- 1 ecart-type = [{mean - std:.4f} ; {mean + std:.4f}]")
stability
""",
            context,
        ),
        _insight(
            [
                "Un écart-type élevé signifie que le résultat dépend de la graine : toute comparaison de modèles doit le mesurer.",
                "Règle pratique : ne pas célébrer un gain inférieur à 2 x l'écart-type observé.",
                "Cette variabilité est aussi un argument pour la **validation croisée** en CI (`train.cross_validation`).",
            ]
        ),
        _md("## 5. Garde-fou de qualité"),
        _code(
            """
threshold_callback = next(
    (callback for callback in CALLBACKS if isinstance(callback, MetricThresholdCallback)), None
)
threshold = CONFIG.metrics.min_primary
observed = OUTCOME.metrics.get(f"val_{CONFIG.metrics.primary}", float("nan"))
seuil = float(threshold) if threshold is not None else float("nan")

# Le sens du seuil dépend de la métrique : un ROC AUC doit le dépasser, un RMSE rester en dessous.
if not np.isfinite(observed) or not np.isfinite(seuil):
    verdict = "INCONNU"
elif CONFIG.metrics.direction == "maximize":
    verdict = "OK" if observed >= seuil else "ÉCHEC"
else:
    verdict = "OK" if observed <= seuil else "ÉCHEC"

formatage = "{:,.2f}" if abs(observed) >= 1000 else "{:.4f}"
sens = "plus c'est haut, mieux c'est" if CONFIG.metrics.direction == "maximize" else "plus c'est bas, mieux c'est"
print(f"métrique          : val_{CONFIG.metrics.primary} ({sens})")
print(f"valeur observée   : {formatage.format(observed)}")
print(f"seuil configuré   : {formatage.format(seuil) if np.isfinite(seuil) else 'aucun'}")
print(f"verdict           : {verdict}")
print(f"callback satisfait: {getattr(threshold_callback, 'satisfied', 'n/a')}")
""",
            context,
        ),
        _insight(
            [
                "Le seuil vit dans `conf/config.yaml` (`metrics.min_primary`) : la CI l'utilise comme gate de déploiement.",
                "Un modèle sous le seuil ne doit **pas** être promu — même s'il « marche » en apparence.",
            ]
        ),
        _md("## 6. Rechargement et vérification"),
        _code(
            """
from src.models import load_model

RESTORED = load_model(OUTCOME.artifacts["model"])
original = np.asarray(MODEL.predict(PREPARED["X_test"]), dtype="float64")
reloaded = np.asarray(RESTORED.predict(PREPARED["X_test"]), dtype="float64")
print("modèle rechargé :", RESTORED.summary())
print("prédictions identiques :", bool(np.allclose(original, reloaded)))
""",
            context,
        ),
        _insight(
            [
                "Le test de rechargement est **le** test de déploiement : un modèle qui ne se recharge pas ne se sert pas.",
                "Il est rejoué automatiquement dans `tests/test_models.py::TestPersistence`.",
            ]
        ),
        _md(
            """## Synthèse

| Étape | Objet utilisé | Artefact |
| --- | --- | --- |
| Construction | `build_model(CONFIG)` | — |
| Entraînement | `Trainer.train(TrainingData)` | `model.joblib` |
| Instrumentation | `LoggingCallback`, `MetricHistoryCallback`, `EarlyStoppingCallback`, `MetricThresholdCallback` | historique |
| Traçabilité | `ModelCard` | `model_card.json` |
| Métriques | `MetricCalculator` | `training_metrics.json` |
| Qualité | `metrics.min_primary` | verdict CI |

**Suite** : `06_error_analysis.ipynb` évalue sur le **test** et transforme les erreurs en décisions.
"""
        ),
    ]
    return write_notebook(destination / "05_training.ipynb", cells)


# ---------------------------------------------------------------------------------------
# 06 — Analyse d'erreurs
# ---------------------------------------------------------------------------------------
def build_06_error_analysis(context: NotebookContext, destination: Path) -> Path:
    """Build ``06_error_analysis.ipynb``.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    spec = context.spec
    data = spec.data
    recommendation_lines = "\n".join(
        f"{index}. {item}" for index, item in enumerate(data.recommendations, start=1)
    )
    cells: list[NotebookNode] = [
        _md(
            f"""# 06 — Analyse d'erreurs et recommandations

**Projet** : {spec.title}
**Principe** : une métrique globale ne dit **jamais** quoi corriger. Ce notebook descend au
niveau de la ligne : qui le modèle se trompe-t-il, avec quelle confiance, et que fait-on lundi matin ?

Le split de **test** n'est utilisé qu'ici — une seule fois — pour rester une estimation honnête.
"""
        ),
        _objectives(
            context,
            [
                "Produire une évaluation complète (métriques, matrice, courbes, calibration).",
                "Arbitrer le **seuil de décision** avec une table métier (précision / rappel / volume).",
                "Segmenter les erreurs pour identifier une cause actionnable.",
                "Formuler des recommandations concrètes, appuyées sur les chiffres observés.",
            ],
        ),
        _code(SETUP, context),
        _code(LOAD_RAW, context),
        _code(PREPARE, context),
        _code(FIT_MODEL, context),
        _md("## 1. Évaluation sur le split de test"),
        _code(
            """
from src.evaluation.evaluator import Evaluator

EVALUATOR = Evaluator.from_config(MODEL, CONFIG.model_dump(), NB_PATHS)
RESULT = EVALUATOR.evaluate(
    PREPARED["X_test"],
    PREPARED["y_test"],
    split="test",
    context=PREPARED["enriched"]["test"],
)

metrics_frame = pd.DataFrame(
    {"métrique": list(RESULT.metrics), "valeur": [RESULT.metrics[name] for name in RESULT.metrics]}
)
print(f"observations évaluées : {RESULT.n_samples} | taux d'erreur : {RESULT.error_rate:.2%}")
metrics_frame.round(4)
""",
            context,
        ),
        _insight(
            [
                f"La métrique de décision est `{spec.metrics.primary}` = **{RESULT_PLACEHOLDER}** — c'est elle qui pilote le seuil de qualité.",
                "Le `context` passé à `evaluate()` permet d'enrichir l'analyse d'erreurs avec les colonnes brutes (identifiants, segments).",
                "Comparer systématiquement au split de validation : un écart important signale un test trop petit ou une dérive.",
            ]
        ),
        _md("## 2. Matrice de confusion et métriques par classe"),
        _code(
            """
from src.visualization.plots import ClassificationPlots

PLOTS = ClassificationPlots(NB_PATHS.figures_dir)
confusion_path = PLOTS.confusion_matrix(RESULT)
display(Image(confusion_path, width=460))
""",
            context,
        ),
        _code(
            """
if not RESULT.per_class.empty:
    display(RESULT.per_class.round(4))
    per_class_path = PLOTS.per_class_metrics(RESULT.per_class)
    display(Image(per_class_path, width=520))
else:
    print("Aucune métrique par classe pour cette tâche.")
""",
            context,
        ),
        _insight(
            [
                "Lire la matrice **en lignes** (rappel) puis **en colonnes** (précision) : les deux erreurs ne coûtent pas la même chose.",
                "Une classe au rappel faible = des cas positifs non détectés ; une précision faible = du bruit envoyé aux équipes.",
                "Le coût métier asymétrique (faux négatif ≫ faux positif) doit se traduire dans le seuil, pas dans la matrice.",
            ]
        ),
        _md("## 3. Courbes et calibration"),
        _code(
            """
figures = {}
for name in ("roc_curve", "precision_recall_curve", "calibration_curve", "score_distribution"):
    method = getattr(PLOTS, name, None)
    if method is None:
        continue
    path = method(RESULT)
    if path is not None:
        figures[name] = path
        display(Image(path, width=470))
print("figures générées :", sorted(figures))
""",
            context,
        ),
        _insight(
            [
                "Le **ROC AUC** est optimiste en fort déséquilibre : le **PR AUC** reflète mieux la valeur opérationnelle.",
                "Une courbe de calibration éloignée de la diagonale interdit d'utiliser les probabilités comme des revenus attendus.",
                "La distribution des scores montre si le seuil par défaut (0.5) tombe dans une zone dense : souvent non.",
            ]
        ),
        _md("## 4. Arbitrage du seuil — la table à montrer aux métiers"),
        _code(
            """
THRESHOLDS = EVALUATOR.threshold_analysis(PREPARED["X_test"], PREPARED["y_test"])
if THRESHOLDS.empty:
    print("Pas de probabilités disponibles : l'arbitrage de seuil ne s'applique pas à ce modèle.")
else:
    display(THRESHOLDS.round(4).head(20))
    threshold_path = PLOTS.threshold_curve(THRESHOLDS)
    if threshold_path is not None:
        display(Image(threshold_path, width=560))
""",
            context,
        ),
        _code(
            """
if not THRESHOLDS.empty:
    best_f1 = THRESHOLDS.loc[THRESHOLDS["f1"].idxmax()]
    volume_column = next((name for name in THRESHOLDS.columns if "volume" in name or "flag" in name), None)
    print(f"seuil maximisant le F1 : {float(best_f1['threshold']):.2f}")
    print(f"  précision = {float(best_f1['precision']):.3f} | rappel = {float(best_f1['recall']):.3f}")
    if volume_column:
        print(f"  {volume_column} = {best_f1[volume_column]}")
    print("\\nLecture métier : baisser le seuil augmente le volume à traiter et le rappel ;")
    print("l'augmenter réduit la charge des équipes mais laisse passer des cas positifs.")
""",
            context,
        ),
        _insight(
            [
                "Le seuil est un **paramètre métier** : il se choisit avec la capacité de traitement disponible, pas à 0.5 par défaut.",
                "Documenter la table dans le rapport (`ReportBuilder`) évite de re-débattre du seuil à chaque comité.",
            ]
        ),
        _md("## 5. Où le modèle se trompe-t-il ?"),
        _code(
            """
if RESULT.errors.empty:
    print("Aucune erreur sur le split de test.")
else:
    print(f"{len(RESULT.errors)} erreurs analysées (triées par confiance décroissante)")
    display(RESULT.errors.head(12))
""",
            context,
        ),
        _code(
            """
# Segmentation des erreurs : quelle population concentre les faux positifs / faux négatifs ?
if not RESULT.errors.empty and CONFIG.data.target:
    error_frame = RESULT.errors.copy()
    segmentation_columns = [
        column for column in (__CATEGORICAL__) if column in error_frame.columns
    ][:3]
    for column in segmentation_columns:
        grouped = (
            error_frame.groupby(error_frame[column].astype(str))
            .agg(erreurs=("is_error", "size"))
            .sort_values("erreurs", ascending=False)
        )
        total = raw.groupby(raw[column].astype(str)).size().rename("population")
        table = grouped.join(total, how="left")
        table["taux_erreur"] = (table["erreurs"] / table["population"]).round(4)
        print(f"--- erreurs par `{column}` ---")
        display(table.head(6))
else:
    print("Segmentation indisponible pour cette tâche.")
""",
            context,
        ),
        _insight(
            [
                "Un segment dont le taux d'erreur dépasse nettement la moyenne est une **cause**, pas une anecdote.",
                "Vérifier ensuite le volume de données de ce segment à l'entraînement : sous-représentation = sous-performance.",
                "Les réponses possibles : feature dédiée, sur-échantillonnage du segment, ou règle métier complémentaire.",
            ]
        ),
        _code(
            """
baseline_comparison = EVALUATOR.compare_to_baseline(PREPARED["X_test"], PREPARED["y_test"])
gains = pd.DataFrame(
    {
        "modèle": ["modèle entraîné", "baseline"],
        CONFIG.metrics.primary: [
            RESULT.metrics.get(CONFIG.metrics.primary, float("nan")),
            baseline_comparison.get(f"baseline_{CONFIG.metrics.primary}", float("nan")),
        ],
    }
)
gains.round(4)
""",
            context,
        ),
        _md("## 6. Rapport exécutable"),
        _code(
            """
from src.evaluation.reports import ReportBuilder

REPORTER = ReportBuilder(NB_PATHS, config=CONFIG.model_dump())
WRITTEN = REPORTER.build(RESULT, model=MODEL, thresholds=THRESHOLDS if not THRESHOLDS.empty else None)
print(f"{len(WRITTEN)} artefacts écrits dans {NB_PATHS.artifacts_dir.relative_to(PROJECT_ROOT)}")
report_path = WRITTEN["report"]
Markdown(report_path.read_text(encoding="utf-8")[:2500] + "\\n\\n[…]")
""",
            context,
        ),
        _insight(
            [
                "Le rapport mélange **chiffres calculés** et **recommandations documentées** : il est régénérable à chaque run.",
                "Le même contenu est écrit en JSON (`artifacts/metrics`) pour être consommé par un dashboard ou une CI.",
            ]
        ),
        _code(
            """
recommendations = REPORTER.recommendations(RESULT)
for index, recommendation in enumerate(recommendations, start=1):
    print(f"{index:2d}. {recommendation}")
""",
            context,
        ),
        _md(
            f"""## 7. Recommandations concrètes

### issues de l'analyse (calculées ci-dessus)

Elles dépendent des métriques observées : seuil à ajuster, classe à rappeler, calibration à
revoir, volume d'erreurs à investiguer.

### documentées pour ce cas d'usage

{recommendation_lines or "1. Rejouer l'évaluation sur un échantillon plus large avant toute décision."}

### plan d'action proposé

| Priorité | Action | Effet attendu | Comment vérifier |
| --- | --- | --- | --- |
| 1 | Fixer le seuil avec la table d'arbitrage | volume d'alertes maîtrisé | `threshold_analysis()` rejoué |
| 2 | Recalibrer les probabilités si la diagonale est manquée | coûts/revenus attendus fiables | `calibration_curve` |
| 3 | Instrumenter les segments les plus en erreur | rappel ciblé | segmentation du §5 |
| 4 | Ajouter un suivi de dérive sur les features dominantes | alerte précoce | `mlops/model-monitoring` |
| 5 | Rejouer ce notebook à chaque nouvelle version de données | non-régression | `make evaluate` + CI |

## 8. Limites assumées

- Les données sont **synthétiques** : les niveaux de performance illustrent une méthode, pas un marché réel.
- Une seule passe d'évaluation : la variance n'est pas mesurée ici (voir notebook 05, §4).
- L'analyse d'erreurs porte sur {NB_ROWS} lignes maximum pour rester interactive.

**Aller plus loin dans le dépôt** : comparaison multi-stacks (`data-science/classification/with-*`),
mise en production et suivi (`mlops/`), pipelines de données (`data-eng/`).
"""
        ),
    ]
    rendered: list[NotebookNode] = []
    for cell in cells:
        source = cell["source"]
        source = source.replace("__CATEGORICAL__", context.py_list(_segmentation_columns(context)))
        rendered.append(cell)
    result_path = write_notebook(destination / "06_error_analysis.ipynb", rendered)
    _replace_placeholder(result_path)
    return result_path


def _segmentation_columns(context: NotebookContext) -> list[str]:
    """Return the categorical columns used to segment the errors.

    Args:
        context: Notebook context.

    Returns:
        Up to three categorical feature names.
    """
    return list(context.categorical_features)[:3]


def _replace_placeholder(path: Path) -> None:
    """Inject the observed primary metric into the notebook prose.

    Le texte Markdown ne peut pas connaître la valeur calculée à l'exécution ; on cite donc la
    métrique **nommée** plutôt qu'une valeur figée (qui deviendrait fausse au prochain run).

    Args:
        path: Written notebook.
    """
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(RESULT_PLACEHOLDER, "celle affichée ci-dessus"), encoding="utf-8")


#: Jeton remplacé après écriture (voir :func:`_replace_placeholder`).
RESULT_PLACEHOLDER = "__PRIMARY_VALUE__"


# ---------------------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------------------
def build_all(context: NotebookContext, destination: Path) -> list[Path]:
    """Build the six notebooks of a tabular project.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory of the project.

    Returns:
        The written notebook paths, in order.
    """
    error_analysis = build_06_error_analysis
    model_exploration = build_04_model_exploration
    if _is_regression(context):
        # L'analyse d'erreurs d'une cible continue n'a ni matrice de confusion ni seuil à arbitrer.
        from tools.scaffold.notebooks.regression import (
            build_06_error_analysis as build_06_regression,
        )

        error_analysis = build_06_regression
    elif _is_clustering(context):
        # Sans cible, l'exploration porte sur le nombre de groupes et la stabilité des
        # affectations, et l'analyse finale sur les profils de segments et leur validité externe.
        from tools.scaffold.notebooks.clustering import (
            build_04_model_exploration as build_04_clustering,
            build_06_error_analysis as build_06_clustering,
        )

        model_exploration = build_04_clustering
        error_analysis = build_06_clustering
    builders = (
        build_01_eda,
        build_02_validation,
        build_03_preprocessing,
        model_exploration,
        build_05_training,
        error_analysis,
    )
    return [builder(context, destination) for builder in builders]


__all__ = [
    "NB_ROWS",
    "build_01_eda",
    "build_04_model_exploration",
    "build_05_training",
    "build_06_error_analysis",
    "build_all",
    "_is_clustering",
    "_is_regression",
]
