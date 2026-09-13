"""Cellules de notebooks spécifiques à la tâche **anomaly** (détection de fraude non supervisée).

Le squelette des six notebooks (``notebooks/tabular.py``) est partagé par toutes les tâches
tabulaires ; seules les cellules qui parlent de *cible*, de *probabilité de classe* ou d'*erreur de
prédiction* doivent changer. Ce module fournit :

* :func:`structure_cells` — la section 5 de ``01_eda.ipynb`` : pas de cible à prédire, donc on
  regarde la **prévalence**, le pouvoir discriminant **univarié** de chaque variable (pour montrer
  qu'aucune ne suffit) et le caractère **informatif** des valeurs manquantes ;
* :func:`build_04_model_exploration` — le plancher aléatoire et les règles métier existantes, la
  comparaison des algorithmes de la stack, la sensibilité à la « contamination », la stabilité du
  classement entre graines, puis la grille d'hyperparamètres ;
* :func:`build_06_error_analysis` — l'évaluation complète, l'arbitrage **au budget**
  d'investigation, la couverture par mode opératoire, le lift par décile, l'analyse des fraudes
  manquées et des fausses alertes, les facteurs contributifs, les diagnostics et le rapport.

Les helpers de rendu (``_code``, ``_md``, ``_insight``, jetons ``__XXX__``) sont réutilisés tels
quels : un notebook de détection d'anomalies et un notebook de classification partagent la même
grammaire.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tools.scaffold.notebooks.tabular import (
    LOAD_RAW,
    PREPARE,
    SETUP,
    _code,
    _insight,
    _md,
    _objectives,
    _replace_placeholder,
)
from tools.scaffold.utils_notebooks import NotebookNode, write_notebook

if TYPE_CHECKING:  # pragma: no cover
    from tools.scaffold.utils_notebooks import NotebookContext

__all__ = ["build_04_model_exploration", "build_06_error_analysis", "structure_cells"]

#: Colonnes de diagnostic produites par le générateur (métadonnées, jamais des features).
LABEL = "is_fraud"
SCHEME = "fraud_scheme"


# ---------------------------------------------------------------------------------------
# 01 — Prévalence, signal univarié, manquants informatifs
# ---------------------------------------------------------------------------------------
_PREVALENCE_CELL = f"""
label_column = "{LABEL}"
scheme_column = "{SCHEME}"
labels = raw[label_column].to_numpy().astype(int)

prevalence = float(labels.mean())
n_frauds = int(labels.sum())
print(f"transactions            : {{len(raw)}}")
print(f"fraudes confirmées      : {{n_frauds}}")
print(f"prévalence              : {{prevalence:.3%}}")
print(f"plancher PR AUC         : {{prevalence:.4f}}   <- score aléatoire")
print()
print("Ce que vaut un détecteur qui ne signale rien :")
print(f"  accuracy : {{1 - prevalence:.4f}}   (flatteur et inutile)")
print(f"  rappel   : 0.0000   (aucune fraude capturée)")
print(f"  PR AUC   : {{prevalence:.4f}}   (égal au plancher)")

schemes = (
    raw[scheme_column]
    .fillna("légitime")
    .value_counts()
    .rename_axis("population")
    .to_frame("effectif")
)
schemes["part du flux"] = (schemes["effectif"] / len(raw)).round(4)
schemes["part de la fraude"] = (schemes["effectif"] / max(n_frauds, 1)).round(3)
display(schemes)

comparatif = raw.groupby(raw[scheme_column].fillna("légitime"), observed=True).agg(
    montant_médian=("amount_eur", "median"),
    vélocité_moyenne=("transactions_24h", "mean"),
    échecs_moyens=("failed_attempts_1h", "mean"),
    part_nuit=("is_night", "mean"),
    part_3ds=("three_ds_authenticated", "mean"),
    chargebacks_moyens=("previous_chargebacks_12m", "mean"),
).round(3)
display(comparatif)
"""

_SEPARATION_CELL = """
from sklearn.metrics import roc_auc_score

feature_columns = [
    column for column in raw.columns if column not in set(CONFIG.data.drop_columns)
]
numeric_columns = [
    column for column in feature_columns if pd.api.types.is_numeric_dtype(raw[column])
]

rows = []
for column in numeric_columns:
    values = pd.to_numeric(raw[column], errors="coerce")
    if values.notna().sum() < 100 or values.nunique() < 2:
        continue
    filled = values.fillna(values.median()).to_numpy(dtype="float64")
    auc = float(roc_auc_score(labels, filled))
    rows.append(
        {
            "feature": column,
            "auc_univariée": max(auc, 1.0 - auc),
            "sens": "fraude = valeurs hautes" if auc >= 0.5 else "fraude = valeurs basses",
            "médiane_fraude": round(float(values[labels == 1].median()), 2),
            "médiane_légitime": round(float(values[labels == 0].median()), 2),
            "manquants_%": round(float(values.isna().mean()) * 100, 2),
        }
    )

separation = pd.DataFrame(rows).sort_values("auc_univariée", ascending=False).reset_index(drop=True)
display(separation)

fig, axis = plt.subplots(figsize=(7.8, 0.34 * len(separation) + 1.8))
axis.barh(
    separation["feature"][::-1],
    (separation["auc_univariée"][::-1] - 0.5) * 2,
    color="#0a9396",
)
axis.axvline(0.0, color="#ae2012", linestyle="--", lw=1.2, label="AUC 0,50 : aucun signal")
axis.set_xlabel("pouvoir discriminant univarié (2 x |AUC - 0,5|)")
axis.set_title("Aucune variable ne suffit : le signal est multivarié")
axis.legend(fontsize=8)
fig.tight_layout()
plt.show()
"""

_MISSING_CELL = """
nullable_columns = [
    column
    for column in ("device_age_days", "session_duration_sec", "billing_shipping_distance_km")
    if column in raw.columns
]

missing = pd.DataFrame(
    {
        "manquants légitime %": [
            round(float(raw.loc[labels == 0, column].isna().mean()) * 100, 2)
            for column in nullable_columns
        ],
        "manquants fraude %": [
            round(float(raw.loc[labels == 1, column].isna().mean()) * 100, 2)
            for column in nullable_columns
        ],
    },
    index=nullable_columns,
)
missing["écart (points)"] = (missing["manquants fraude %"] - missing["manquants légitime %"]).round(2)
display(missing)

# Ce que produirait une imputation silencieuse : la valeur médiane écrase le signal.
for column in nullable_columns:
    values = pd.to_numeric(raw[column], errors="coerce")
    naive = values.fillna(values.median())
    auc_naive = float(roc_auc_score(labels, naive.to_numpy(dtype="float64")))
    indicator = values.isna().to_numpy().astype(int)
    auc_indicator = float(roc_auc_score(labels, indicator)) if indicator.sum() else float("nan")
    print(
        f"{column:<32} AUC valeur imputée = {max(auc_naive, 1 - auc_naive):.3f} | "
        f"AUC indicateur de manquant = {max(auc_indicator, 1 - auc_indicator):.3f}"
    )
"""


def structure_cells(context: NotebookContext) -> list[NotebookNode]:
    """Build the anomaly-analysis cells of notebook 01 for an **unsupervised** project.

    Args:
        context: Notebook context.

    Returns:
        The cells to insert in notebook 01 (section 5).
    """
    return [
        _md(
            f"""## 5. Pas de cible à prédire : qu'est-ce qu'une anomalie ici ?

Ce projet n'a **aucune variable cible** (`target: null`). Le détecteur est entraîné sans
étiquette, et les étiquettes ne servent qu'à *mesurer* la performance. Pourquoi ?

1. **l'étiquette arrive tard** — la fraude n'est confirmée qu'au chargeback, 30 à 90 jours après
   la transaction. Un modèle supervisé apprendrait donc sur une vérité partielle et périmée ;
2. **l'étiquette est biaisée** — seules les transactions déjà alertées par les règles existantes
   sont investiguées, donc étiquetées. Un schéma inédit est invisible dans les étiquettes ;
3. **la prévalence est très faible** — à ~1,8 % de fraude, l'accuracy est une métrique
   inexploitable : « ne rien signaler » obtient 98,2 %.

Deux colonnes de **métadonnées** existent dans ce jeu synthétique : `{LABEL}` (fraude confirmée)
et `{SCHEME}` (mode opératoire). Elles sont exclues des features par `drop_columns` : elles servent
à juger le détecteur, jamais à l'entraîner.

L'exploration répond donc à trois questions : quelle est la **prévalence** (et le plancher de
performance qui en découle), une variable **seule** suffit-elle à séparer la fraude, et les valeurs
**manquantes** portent-elles de l'information ?
"""
        ),
        _code(_PREVALENCE_CELL, context),
        _insight(
            [
                "La prévalence fixe le **plancher** : un score aléatoire obtient une PR AUC égale à la part de fraude. Toute performance publiée doit être lue relativement à ce plancher, jamais dans l'absolu.",
                "L'accuracy est ici un piège : un détecteur qui ne signale rien obtient ~98 % d'accuracy et 0 % de rappel. C'est la raison du choix de la PR AUC comme métrique primaire.",
                "Les quatre modes opératoires n'ont pas le même profil : vélocité explosive pour le card testing, géographie incohérente pour la prise de compte, historique de chargebacks pour la fraude amicale. Aucun détecteur univarié ne peut couvrir les quatre.",
            ]
        ),
        _code(_SEPARATION_CELL, context),
        _insight(
            [
                "Aucune variable n'atteint une AUC univariée élevée : le signal est **multivarié** et conditionnel (un montant élevé n'est anormal que relativement au panier habituel et à la catégorie).",
                "Les features de vélocité (`transactions_24h`, `distinct_merchants_24h`, `failed_attempts_1h`) dominent le classement univarié : c'est cohérent avec le card testing, mais elles ne couvrent pas la fraude amicale.",
                "`amount_eur` seul est un mauvais détecteur : les achats de luxe légitimes occupent exactement la même zone. C'est le premier piège du générateur, et la raison pour laquelle le ratio au panier habituel existe.",
            ]
        ),
        _code(_MISSING_CELL, context),
        _insight(
            [
                "Les manquants sont **informatifs** (MNAR) : une empreinte d'appareil bloquée est deux fois plus fréquente en fraude. Une imputation silencieuse à la médiane détruit ce signal.",
                "L'indicateur de manquant porte souvent plus d'information que la valeur imputée : c'est un feature à part entière, à déclarer explicitement dans `conf/preprocessing/default.yaml`.",
                "`billing_shipping_distance_km` est manquant pour les retraits en magasin : ici le manquant est légitime et non risqué. Même symptôme, causes opposées — d'où la nécessité de documenter la mécanique de collecte avant d'imputer.",
            ]
        ),
    ]


# ---------------------------------------------------------------------------------------
# 04 — Exploration : plancher, algorithmes, contamination, stabilité, grille
# ---------------------------------------------------------------------------------------
_FIT_CELL = """
from src.models import build_model

# Tâche non supervisée : `fit` reçoit `None` en guise de cible. Le modèle apprend la région de
# densité « normale » sur le train uniquement ; la validation sert à choisir le point de
# fonctionnement, jamais à ajuster les hyperparamètres sur les fraudes.
MODEL = build_model(CONFIG, feature_names=PREPARED["feature_names"])
FIT_RESULT = MODEL.fit(
    PREPARED["X_train"], None, X_val=PREPARED["X_val"], y_val=None, callbacks=[]
)
print(MODEL.summary())
print(f"entraînement : {FIT_RESULT.duration_seconds:.2f} s")
"""


_BASELINE_CELL = f"""
from src.training.losses_metrics import MetricCalculator, MetricInputs

RANKING_METRICS = [
    name
    for name in [CONFIG.metrics.primary, *CONFIG.metrics.secondary]
    if name in {{"pr_auc", "roc_auc", "recall_at_budget", "precision_at_budget"}}
]

val_frame = PREPARED["enriched"]["val"]
if val_frame is None or PREPARED["X_val"] is None:
    raise RuntimeError(
        "Split de validation absent : renseignez `train.split.val_size` dans conf/config.yaml"
    )
VAL_LABELS = val_frame["{LABEL}"].to_numpy().astype(int)
# Le point de fonctionnement est **lu dans la configuration** (noeud `fraud_detection` de
# `conf/config.yaml`) : le notebook et le pipeline d'inférence pilotent au même budget.
BUDGET_RATE = float(
    dict(getattr(CONFIG, "fraud_detection", {{}}) or {{}}).get("budget_rate", 0.02)
)
VAL_BUDGET = max(1, round(BUDGET_RATE * len(VAL_LABELS)))
calculator = MetricCalculator(
    task=CONFIG.metrics.task, metrics=RANKING_METRICS, extra={{"budget": VAL_BUDGET}}
)

prevalence = float(VAL_LABELS.mean())
print(f"transactions de validation : {{len(VAL_LABELS)}}")
print(f"fraudes confirmées         : {{int(VAL_LABELS.sum())}}")
print(f"prévalence                 : {{prevalence:.3%}}")
print(f"budget d'investigation     : {{VAL_BUDGET}} alertes ({{BUDGET_RATE:.1%}} du flux)")
print(f"plancher PR AUC            : {{prevalence:.4f}}")

rng = np.random.default_rng(0)
amount = pd.to_numeric(val_frame["amount_eur"], errors="coerce").fillna(0.0).to_numpy()
velocity = pd.to_numeric(val_frame["transactions_24h"], errors="coerce").fillna(0.0).to_numpy()
failures = pd.to_numeric(val_frame["failed_attempts_1h"], errors="coerce").fillna(0.0).to_numpy()

references = {{
    "score aléatoire (plancher)": rng.random(len(VAL_LABELS)),
    "règle métier : montant > P99": (amount > np.quantile(amount, 0.99)).astype(float),
    "règle métier : vélocité 24 h >= 5": (velocity >= 5).astype(float),
    "règle métier : échecs 1 h >= 2": (failures >= 2).astype(float),
    "règles combinées (OU)": (
        (amount > np.quantile(amount, 0.99)) | (velocity >= 5) | (failures >= 2)
    ).astype(float),
    # Le détecteur configuré rejoint le tableau : c'est la comparaison qui décide. Un modèle qui
    # ne bat pas les règles métier qu'il est censé remplacer ne doit pas être mis en production.
    f"détecteur configuré ({{MODEL.algorithm}})": np.asarray(
        MODEL.predict(PREPARED["X_val"]), dtype="float64"
    ).ravel(),
}}

rows = []
for name, scores in references.items():
    values = calculator.evaluate(MetricInputs(y_true=VAL_LABELS, y_pred=None, y_proba=scores))
    rows.append(
        {{
            "référence": name,
            **{{key: round(float(value), 4) for key, value in values.items()}},
        }}
    )
baseline_table = pd.DataFrame(rows)
display(baseline_table)
"""

_ALGORITHMS_CELL = """
from src.models.factory import available_algorithms

ALGORITHMS = available_algorithms(CONFIG.metrics.task)
print(f"{len(ALGORITHMS)} algorithmes disponibles pour la tâche '{CONFIG.metrics.task}' :")
print(ALGORITHMS)

rows = []
# Comparaison **loyale** : `params={}` construit chaque algorithme avec ses réglages par défaut.
# Les `model.params` configurés sont propres à la forêt d'isolation (`n_estimators`,
# `max_samples`, `max_features`) : les injecter dans un one-class SVM lèverait une erreur, et les
# régler « à la main » pour chaque concurrent fausserait le classement. Le détecteur configuré et
# réglé est, lui, évalué en section 1.
for algorithm in ALGORITHMS:
    try:
        candidate = build_model(
            CONFIG, feature_names=PREPARED["feature_names"], algorithm=algorithm, params={}
        )
        result = candidate.fit(
            PREPARED["X_train"], None, X_val=PREPARED["X_val"], y_val=None, callbacks=[]
        )
        scores = np.asarray(candidate.predict(PREPARED["X_val"]), dtype="float64").ravel()
        values = calculator.evaluate(
            MetricInputs(y_true=VAL_LABELS, y_pred=None, y_proba=scores)
        )
        row = {
            "algorithme": algorithm,
            CONFIG.metrics.primary: values.get(CONFIG.metrics.primary, float("nan")),
        }
        for name in RANKING_METRICS[1:]:
            row[name] = values.get(name, float("nan"))
        row["score moyen"] = round(float(np.mean(scores)), 4)
        row["secondes"] = round(result.duration_seconds, 2)
        rows.append(row)
    except Exception as error:  # un algorithme incompatible ne doit pas casser l'exploration
        rows.append(
            {
                "algorithme": algorithm,
                CONFIG.metrics.primary: float("nan"),
                "erreur": str(error)[:90],
            }
        )

ascending = CONFIG.metrics.direction == "minimize"
ranking = pd.DataFrame(rows).sort_values(
    CONFIG.metrics.primary, ascending=ascending, na_position="last"
)
display(ranking.round(4))
"""

_CONTAMINATION_CELL = """
# La « contamination » est la part d'anomalies que le détecteur *suppose* a priori. Elle déplace
# le seuil interne de décision, mais — point crucial — elle ne change pas le CLASSEMENT produit par
# le score continu. D'où l'intérêt de piloter au budget plutôt qu'au seuil par défaut.
CONTAMINATIONS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)

rows = []
for contamination in CONTAMINATIONS:
    candidate = build_model(
        CONFIG,
        feature_names=PREPARED["feature_names"],
        params={"contamination": contamination},
    )
    candidate.fit(PREPARED["X_train"], None, X_val=PREPARED["X_val"], y_val=None, callbacks=[])
    scores = np.asarray(candidate.predict(PREPARED["X_val"]), dtype="float64").ravel()
    values = calculator.evaluate(
        MetricInputs(y_true=VAL_LABELS, y_pred=None, y_proba=scores)
    )
    flagged = (scores >= np.quantile(scores, 1 - contamination)).astype(int)
    rows.append(
        {
            "contamination": contamination,
            CONFIG.metrics.primary: values.get(CONFIG.metrics.primary, float("nan")),
            **{name: values.get(name, float("nan")) for name in RANKING_METRICS[1:]},
            "alertes au seuil interne": int(flagged.sum()),
            "précision au seuil interne": round(
                float((flagged & VAL_LABELS).sum() / max(int(flagged.sum()), 1)), 4
            ),
        }
    )

contamination_table = pd.DataFrame(rows)
display(contamination_table.round(4))

fig, axis = plt.subplots(figsize=(7.4, 4.0))
axis.plot(
    contamination_table["contamination"] * 100,
    contamination_table[CONFIG.metrics.primary],
    marker="o",
    color="#0a9396",
    label=CONFIG.metrics.primary,
)
axis.axhline(float(VAL_LABELS.mean()), color="#ae2012", ls="--", lw=1.2, label="plancher (prévalence)")
second = axis.twinx()
second.plot(
    contamination_table["contamination"] * 100,
    contamination_table["alertes au seuil interne"],
    color="#ee9b00",
    ls=":",
    marker="s",
    ms=4,
    label="alertes au seuil interne",
)
second.set_ylabel("nombre d'alertes", color="#ee9b00")
second.grid(False)
axis.set_xscale("log")
axis.set_xlabel("contamination déclarée (%)")
axis.set_ylabel(CONFIG.metrics.primary)
axis.set_title("La contamination déplace le seuil, pas le classement")
lines, labels_ = axis.get_legend_handles_labels()
extra_lines, extra_labels = second.get_legend_handles_labels()
axis.legend(lines + extra_lines, labels_ + extra_labels, fontsize=8, loc="center left")
fig.tight_layout()
plt.show()
"""

_STABILITY_CELL = """
from scipy.stats import spearmanr

STABILITY_SEEDS = (7, 19, 123)
reference_scores = np.asarray(MODEL.predict(PREPARED["X_val"]), dtype="float64").ravel()
reference_top = set(np.argsort(-reference_scores, kind="stable")[:VAL_BUDGET].tolist())

rows = []
for seed in STABILITY_SEEDS:
    candidate = build_model(
        CONFIG,
        feature_names=PREPARED["feature_names"],
        params={"random_state": seed},
    )
    candidate.fit(PREPARED["X_train"], None, X_val=PREPARED["X_val"], y_val=None, callbacks=[])
    scores = np.asarray(candidate.predict(PREPARED["X_val"]), dtype="float64").ravel()
    top = set(np.argsort(-scores, kind="stable")[:VAL_BUDGET].tolist())
    values = calculator.evaluate(MetricInputs(y_true=VAL_LABELS, y_pred=None, y_proba=scores))
    rows.append(
        {
            "graine": seed,
            CONFIG.metrics.primary: values.get(CONFIG.metrics.primary, float("nan")),
            "corrélation de rang (Spearman)": round(
                float(spearmanr(reference_scores, scores).statistic), 4
            ),
            "recouvrement du top-budget": round(len(top & reference_top) / max(len(top), 1), 4),
        }
    )

stability_table = pd.DataFrame(rows)
display(stability_table.round(4))
print(
    f"dispersion de la métrique primaire entre graines : "
    f"{float(stability_table[CONFIG.metrics.primary].std()):.4f}"
)
"""

_GRID_CELL = """
import itertools

# Grille déclarée dans le manifeste du projet (`extras.notebook_param_grid`) et injectée ici au
# moment du build : le notebook reste une source unique de vérité côté configuration, et la grille
# explorée est versionnée avec le projet plutôt qu'improvisée à chaque exécution.
GRID = __PARAM_GRID__
combinations = list(itertools.product(*[GRID[name] for name in GRID]))
print(f"{len(combinations)} combinaisons testées sur {list(GRID)}")

grid_rows = []
for combination in combinations:
    params = dict(zip(GRID, combination, strict=True))
    candidate = build_model(CONFIG, feature_names=PREPARED["feature_names"], params=params)
    result = candidate.fit(
        PREPARED["X_train"], None, X_val=PREPARED["X_val"], y_val=None, callbacks=[]
    )
    scores = np.asarray(candidate.predict(PREPARED["X_val"]), dtype="float64").ravel()
    values = calculator.evaluate(MetricInputs(y_true=VAL_LABELS, y_pred=None, y_proba=scores))
    row = {str(name): str(value) for name, value in params.items()}
    row[CONFIG.metrics.primary] = values.get(CONFIG.metrics.primary, float("nan"))
    for name in RANKING_METRICS[1:]:
        row[name] = values.get(name, float("nan"))
    row["secondes"] = round(result.duration_seconds, 2)
    grid_rows.append(row)

grid_results = pd.DataFrame(grid_rows).sort_values(
    CONFIG.metrics.primary, ascending=ascending, na_position="last"
)
display(grid_results.round(4))
"""


def build_04_model_exploration(context: NotebookContext, destination: Path) -> Path:
    """Build ``04_model_exploration.ipynb`` for an anomaly-detection project.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    spec = context.spec
    cells: list[NotebookNode] = [
        _md(
            f"""# 04 — Exploration et comparaison de détecteurs d'anomalies

**Projet** : {spec.title}
**Modèle configuré** : `{spec.model.algorithm}` ({spec.model.display_name})
**Pourquoi ce choix** : {spec.model.rationale}

En détection d'anomalies non supervisée, la question « pourquoi cet algorithme ? » se dédouble :

* **quel score ?** — tous les détecteurs ne rendent pas la même quantité : une forêt d'isolation
  produit un score d'anormalité continu, un auto-encodeur une erreur de reconstruction, un
  one-class SVM une distance à la frontière. Seul un **score ordonnable** permet de piloter au
  budget ;
* **quel point de fonctionnement ?** — la « contamination » déclarée fixe un seuil par défaut, mais
  la vraie contrainte est la **capacité d'investigation** des analystes.

Ce notebook répond aux deux **par des chiffres** : plancher aléatoire, règles métier existantes,
comparaison d'algorithmes, sensibilité à la contamination, stabilité du classement entre graines.
"""
        ),
        _objectives(
            context,
            [
                "Commencer par le **plancher** : un score aléatoire obtient une PR AUC égale à la prévalence, et les règles métier existantes donnent la référence à battre.",
                "Comparer les algorithmes de la stack **à données et budget égaux** : la comparaison n'a de sens qu'à protocole identique.",
                "Comprendre que la **contamination** déplace le seuil sans changer le classement — donc pourquoi on pilote au budget.",
                "Mesurer la **stabilité du classement** entre graines : une file d'alertes qui change tous les jours n'est pas exploitable.",
            ],
        ),
        _code(SETUP, context),
        _code(LOAD_RAW, context),
        _code(PREPARE, context),
        _md(
            """## 0. Le détecteur configuré

On entraîne d'abord le modèle déclaré dans `conf/model/default.yaml` : il servira de référence
dans toutes les sections suivantes (comparaison aux règles métier, stabilité, sensibilité)."""
        ),
        _code(_FIT_CELL, context),
        _md("## 1. Le plancher d'abord : score aléatoire et règles métier"),
        _code(_BASELINE_CELL, context),
        _insight(
            [
                "Le score aléatoire donne une PR AUC égale à la prévalence : c'est le **plancher**. Toute valeur publiée doit être comparée à lui, jamais lue dans l'absolu.",
                "Les règles métier (montant > P99, vélocité ≥ 5, échecs ≥ 2) ont une précision correcte mais un rappel faible : elles ne capturent que les schémas déjà vus. C'est exactement la limite que le détecteur appris doit repousser.",
                "La règle « montant > P99 » illustre le piège des outliers légitimes : elle alerte sur les achats de luxe, qui ne sont pas de la fraude.",
                f"La métrique de décision du projet est `{spec.metrics.primary}` (sens : {spec.metrics.direction}) ; le seuil de qualité déclaré est {spec.metrics.min_primary}.",
            ]
        ),
        _md("## 2. Comparaison des algorithmes de la stack"),
        _code(_ALGORITHMS_CELL, context),
        _insight(
            [
                "Une forêt d'isolation suppose qu'une anomalie est **facile à isoler** par des coupes aléatoires ; un auto-encodeur suppose qu'elle est **mal reconstruite** par un goulot d'étranglement. Les deux hypothèses ne capturent pas les mêmes schémas.",
                "Un one-class SVM est sensible à l'échelle et au volume : sur des millions de transactions, son coût quadratique le rend inutilisable sans échantillonnage.",
                "Un écart de PR AUC inférieur à 2x la dispersion entre graines (section 4) n'est pas un signal : ne pas choisir un algorithme sur un écart de cet ordre.",
            ]
        ),
        _md("## 3. Sensibilité à la contamination déclarée"),
        _code(_CONTAMINATION_CELL, context),
        _insight(
            [
                "La PR AUC et le rappel **au budget** sont quasi constants : la contamination ne change pas le classement, seulement le seuil interne. C'est la justification du pilotage au budget.",
                "Le nombre d'alertes au seuil interne, lui, suit mécaniquement la contamination : déclarer 20 % d'anomalies dans un flux qui en contient 1,8 % sature l'équipe d'analyse.",
                "En production, la contamination se règle sur la **capacité d'investigation** (ici 2 % du flux), pas sur une estimation de la prévalence réelle — qui est inconnue par définition.",
            ]
        ),
        _md("## 4. Stabilité : le classement survit-il à un changement de graine ?"),
        _code(_STABILITY_CELL, context),
        _insight(
            [
                "Le **recouvrement du top-budget** mesure ce que le métier redoute : une transaction alertée hier qui ne l'est plus aujourd'hui, à comportement inchangé, décrédibilise l'outil auprès des analystes.",
                "La corrélation de rang de Spearman est plus exigeante que le recouvrement du top-K : elle vérifie tout le classement, pas seulement la tête de file.",
                "Une forêt d'isolation est stabilisée en augmentant `n_estimators` ; un auto-encodeur en fixant toutes les graines (initialisation, découpage des lots) et en réduisant le taux d'apprentissage.",
            ]
        ),
        _md("## 5. Sensibilité aux hyperparamètres"),
        _code(_GRID_CELL, context),
        _insight(
            [
                "Le tri respecte le **sens** de la métrique (`direction: maximize` pour une PR AUC) : un tri ascendant par défaut classerait les pires détecteurs en premier.",
                "Une grille se lit aussi par sa **dispersion** : si toutes les combinaisons se tiennent en 0,01 de PR AUC, le détecteur est robuste et le réglage fin n'est pas le levier principal — les features le sont.",
                "Gare au sur-ajustement sur le split de validation : avec quelques dizaines de fraudes seulement, l'écart-type d'une PR AUC est de l'ordre de 0,03 à 0,05. Choisir le meilleur point d'une grille sur un seul split est un biais classique.",
            ]
        ),
        _md(
            """## 6. Choix argumenté

| Critère | Lecture | Décision |
| --- | --- | --- |
| PR AUC (validation) | capacité de classement en forte imbalance | doit dépasser nettement la prévalence |
| Rappel au budget | fraude capturée à capacité constante | critère métier principal |
| Précision au budget | coût analyste par alerte | borne le volume d'alertes acceptable |
| Recouvrement du top-budget entre graines | reproductibilité de la file d'alertes | ≥ 0,85 avant mise en production |
| Coût d'entraînement / d'inférence | fenêtre de batch et latence temps réel | < 50 ms par transaction en scoring |

**Règle de décision retenue** : choisir le détecteur qui maximise le rappel **au budget déclaré**,
à précision au budget acceptable, puis vérifier la stabilité entre graines. Un gain de PR AUC
inférieur à la dispersion entre graines ne justifie pas un changement d'algorithme : il justifie un
travail sur les features (vélocité à fenêtre courte, indicateurs de manquants).
"""
        ),
    ]
    result_path = write_notebook(destination / "04_model_exploration.ipynb", cells)
    _replace_placeholder(result_path)
    return result_path


# ---------------------------------------------------------------------------------------
# 06 — Évaluation au budget, couverture par schéma, erreurs, recommandations
# ---------------------------------------------------------------------------------------
_EVALUATE_CELL = """
from src.evaluation.evaluator import Evaluator

EVALUATOR = Evaluator.from_config(MODEL, CONFIG.model_dump(), NB_PATHS)
RESULT = EVALUATOR.evaluate(
    PREPARED["X_test"],
    None,
    split="test",
    context=PREPARED["enriched"]["test"],
)

metrics_frame = pd.DataFrame(
    {"métrique": list(RESULT.metrics), "valeur": [RESULT.metrics[name] for name in RESULT.metrics]}
)
print(f"transactions évaluées   : {RESULT.n_samples}")
print(f"fraudes confirmées      : {int(RESULT.extras['n_frauds'])}")
print(f"prévalence              : {RESULT.prevalence:.3%}")
print(f"budget d'investigation  : {RESULT.budget} alertes ({RESULT.extras['budget_rate']:.1%})")
print(f"seuil de score retenu   : {RESULT.threshold:.4f}")
print(f"lift au budget          : {RESULT.lift_at_budget:.1f}x")
print(f"plancher (aléatoire)    : {float(RESULT.extras['baseline']['random_pr_auc']):.4f}")
display(metrics_frame.round(4))
"""

_BUDGET_CELL = """
budget_table = pd.DataFrame(RESULT.curves["budget_tradeoff"])
display(budget_table.round(4))

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2))
axes[0].plot(budget_table["budget_rate"] * 100, budget_table["recall"], marker="o", ms=4,
             color="#0a9396", label="rappel")
axes[0].plot(budget_table["budget_rate"] * 100, budget_table["precision"], marker="s", ms=4,
             color="#ae2012", label="précision")
axes[0].axvline(float(RESULT.extras["budget_rate"]) * 100, color="#ee9b00", ls="--", lw=1.6,
                label=f"budget retenu ({RESULT.extras['budget_rate']:.0%})")
axes[0].set_xlabel("volume d'alertes (% du flux)")
axes[0].set_ylabel("rappel / précision")
axes[0].set_ylim(0, 1.02)
axes[0].set_title("Arbitrage capacité d'analyse ↔ fraude capturée")
axes[0].legend(fontsize=8)

axes[1].plot(budget_table["alerts"], budget_table["frauds_captured"], marker="o", ms=4,
             color="#0a9396", label="détecteur")
random_capture = budget_table["alerts"] * float(RESULT.prevalence)
axes[1].plot(budget_table["alerts"], random_capture, ls="--", color="#ae2012",
             label="tirage aléatoire")
axes[1].set_xlabel("nombre d'alertes traitées")
axes[1].set_ylabel("fraudes capturées")
axes[1].set_title("Courbe de capture : le classement fait le gain")
axes[1].legend(fontsize=8)
fig.tight_layout()
plt.show()
"""

_SCHEME_CELL = """
SCHEME_LABELS = {
    "card_not_present": "Card not present",
    "account_takeover": "Prise de compte",
    "synthetic_identity": "Identité synthétique",
    "friendly_fraud": "Fraude amicale",
    "legitimate": "Légitime",
}

per_scheme = RESULT.per_scheme
if per_scheme.empty:
    print("Aucune fraude confirmée dans ce split : couverture non mesurable.")
else:
    table = per_scheme.copy()
    table["mode opératoire"] = [
        SCHEME_LABELS.get(str(name), str(name)) for name in table["fraud_scheme"]
    ]
    display(
        table[
            [
                "mode opératoire", "frauds", "share_of_fraud", "captured_at_budget",
                "recall_at_budget", "median_rank", "best_rank", "worst_rank",
            ]
        ].round(3)
    )

    fig, axis = plt.subplots(figsize=(7.6, 3.8))
    ordered = table.sort_values("recall_at_budget")
    axis.barh(
        [SCHEME_LABELS.get(str(name), str(name)) for name in ordered["fraud_scheme"]],
        ordered["recall_at_budget"],
        color="#0a9396",
    )
    for position, (recall, count) in enumerate(
        zip(ordered["recall_at_budget"], ordered["frauds"], strict=True)
    ):
        axis.text(float(recall) + 0.01, position, f"{float(recall):.2f} (n={int(count)})",
                  va="center", fontsize=8)
    axis.set_xlim(0, 1.15)
    axis.set_xlabel("rappel au budget")
    axis.set_title("Couverture par mode opératoire")
    fig.tight_layout()
    plt.show()
"""

_LIFT_CELL = """
lift_rows = RESULT.curves.get("lift_by_decile") or []
if not lift_rows:
    print("Lift non calculable sur ce split.")
else:
    lift_table = pd.DataFrame(lift_rows)
    display(lift_table.round(4))

    frauds = pd.Series(RESULT.predictions.loc[RESULT.predictions["is_fraud"] == 1, "rank"])
    captured = np.array(
        [(frauds <= position).sum() for position in range(1, len(RESULT.predictions) + 1)]
    )
    share = captured / max(int(frauds.size), 1)
    inspected = np.arange(1, len(RESULT.predictions) + 1) / len(RESULT.predictions)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2))
    axes[0].bar(
        [f"D{int(value)}" for value in lift_table["decile"]],
        lift_table["lift"],
        color=["#ee9b00" if int(value) >= 8 else "#0a9396" for value in lift_table["decile"]],
    )
    axes[0].axhline(1.0, color="#ae2012", ls="--", lw=1.2, label="lift = 1 (aléatoire)")
    axes[0].set_ylabel("lift sur la prévalence")
    axes[0].set_title("Concentration de la fraude par décile de score")
    axes[0].legend(fontsize=8)

    axes[1].plot(inspected * 100, share * 100, color="#0a9396", lw=2.0, label="détecteur")
    axes[1].plot([0, 100], [0, 100], color="#ae2012", ls="--", lw=1.2, label="aléatoire")
    axes[1].axvline(float(RESULT.extras["budget_rate"]) * 100, color="#ee9b00", ls="--", lw=1.6,
                    label="budget retenu")
    axes[1].set_xlabel("part du flux inspectée (%)")
    axes[1].set_ylabel("part de la fraude capturée (%)")
    axes[1].set_title("Courbe de capture cumulative")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    plt.show()
"""

_ERRORS_CELL = """
rows = RESULT.predictions
errors = RESULT.errors
context_columns = [
    column
    for column in (
        "transaction_id", "amount_eur", "merchant_category", "channel", "transactions_24h",
        "failed_attempts_1h", "device_age_days", "three_ds_authenticated",
        "amount_to_customer_avg_ratio", "previous_chargebacks_12m",
    )
    if column in errors.columns
]

outcomes = rows["outcome"].value_counts(normalize=True) * 100
print("Répartition des décisions au budget :")
display(outcomes.round(2).to_frame("%"))

for kind, title in (
    ("false_negative", "Fraudes manquées les mieux classées (marge de progrès)"),
    ("false_positive", "Fausses alertes les plus convaincantes (coût analyste)"),
    ("true_positive", "Fraudes capturées (ce que le détecteur sait faire)"),
):
    subset = errors[errors["error_kind"] == kind].head(8)
    if subset.empty:
        continue
    print(f"\\n### {title}")
    display(subset[["rank", "score", "fraud_scheme", *context_columns]].round(3))

# Les fausses alertes sont-elles des outliers légitimes ? On vérifie sur les colonnes brutes.
false_alarms = rows[rows["outcome"] == "false_positive"]
if not false_alarms.empty and "amount_eur" in PREPARED["enriched"]["test"].columns:
    test_frame = PREPARED["enriched"]["test"].reset_index(drop=True)
    amount = pd.to_numeric(test_frame["amount_eur"], errors="coerce")
    high_value = float((amount.loc[false_alarms.index] > amount.quantile(0.99)).mean())
    print(
        f"\\nPart des fausses alertes dont le montant dépasse le P99 du flux : {high_value:.1%}"
    )
"""

_DRIVERS_CELL = """
importance = RESULT.feature_importance
if importance.empty:
    print("Importance par permutation non calculable sur ce split.")
else:
    top = importance.head(15)
    display(top.round(5))

    fig, axis = plt.subplots(figsize=(7.8, 0.34 * len(top) + 1.8))
    axis.barh(
        top["feature"][::-1],
        top["importance_mean"][::-1],
        xerr=top["importance_std"][::-1],
        color=["#ae2012" if value < 0 else "#0a9396" for value in top["importance_mean"][::-1]],
    )
    axis.axvline(0.0, color="#495057", lw=1.0)
    axis.set_xlabel("importance par permutation (recouvrement du top-budget)")
    axis.set_title("Facteurs contributifs du score d'anomalie")
    fig.tight_layout()
    plt.show()

    print(f"\\npart de l'importance portée par la première feature : {float(top.iloc[0]['share']):.1%}")
    negative = importance[importance["importance_mean"] < 0]
    print(f"features à importance négative : {list(negative['feature']) or 'aucune'}")
"""

_DIAGNOSTICS_CELL = """
from src.evaluation.reports import ReportBuilder

BUILDER = ReportBuilder(NB_PATHS, config=CONFIG.model_dump())
diagnostics = BUILDER.diagnostics(RESULT)

if not diagnostics:
    print("Aucun signal de faiblesse marqué sur ce split.")
for title, detail in diagnostics:
    print(f"\\n### {title}\\n{detail}")

print("\\n--- seuils du cas d'usage ---")
for name, value in sorted(BUILDER.thresholds.items()):
    print(f"  {name}: {value}")
"""

_RECOMMENDATIONS_CELL = """
recommendations = BUILDER.recommendations(RESULT)
for index, item in enumerate(recommendations, start=1):
    print(f"{index}. {item}")
"""

_REPORT_CELL = """
artifacts = BUILDER.build(RESULT, model=MODEL)
for kind, path in sorted(artifacts.items()):
    relative = path.relative_to(PROJECT_ROOT) if path.is_absolute() else path
    print(f"{kind:<18} -> {relative}")

display(Markdown(f"### Rapport généré\\n\\n`{artifacts['report']}`"))
if "figure_pr_curve" in artifacts:
    display(Image(filename=str(artifacts["figure_pr_curve"]), width=720))
if "figure_budget_tradeoff" in artifacts:
    display(Image(filename=str(artifacts["figure_budget_tradeoff"]), width=760))
"""


def build_06_error_analysis(context: NotebookContext, destination: Path) -> Path:
    """Build ``06_error_analysis.ipynb`` for an anomaly-detection project.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    spec = context.spec
    cells: list[NotebookNode] = [
        _md(
            f"""# 06 — Analyse des erreurs et recommandations

**Projet** : {spec.title}
**Détecteur évalué** : `{spec.model.algorithm}` ({spec.model.display_name})
**Métrique primaire** : `{spec.metrics.primary}` (seuil cible : {spec.metrics.min_primary})

Une fraude ne se résume pas à une matrice de confusion : le coût d'une fausse alerte est du temps
analyste, le coût d'une fraude manquée est une perte nette. Ce notebook évalue le détecteur **au
point de fonctionnement réel** (le budget d'investigation), puis dissèque ses erreurs pour en tirer
des recommandations actionnables.

Les étiquettes utilisées ici (`{LABEL}`, `{SCHEME}`) sont des **métadonnées** : le modèle ne les a
jamais vues, ni à l'entraînement ni dans la matrice de features.
"""
        ),
        _objectives(
            context,
            [
                "Évaluer le détecteur sur le split de test **hors échantillon** et comparer au plancher aléatoire.",
                "Traduire le score en **décision de capacité** : table d'arbitrage volume d'alertes → rappel / précision / lift.",
                "Vérifier la **couverture par mode opératoire** : un bon score global peut masquer un schéma non détecté.",
                "Disséquer les **erreurs** (fraudes manquées, fausses alertes) et les **facteurs contributifs**, puis formuler des recommandations chiffrées.",
            ],
        ),
        _code(SETUP, context),
        _code(LOAD_RAW, context),
        _code(PREPARE, context),
        _code(_FIT_CELL, context),
        _md("## 1. Évaluation hors échantillon"),
        _code(_EVALUATE_CELL, context),
        _insight(
            [
                "La PR AUC se lit **relativement à la prévalence** : à 1,8 % de fraude, un plancher de 0,018 signifie qu'une PR AUC de 0,40 représente un gain d'un facteur ~22, et non « 40 % de réussite ».",
                "La ROC AUC reste élevée même quand la file d'alertes est noyée sous les faux positifs : c'est pourquoi elle est secondaire ici, et la PR AUC primaire.",
                "Le lift au budget est la traduction métier du classement : à capacité égale, combien de fois plus de fraude qu'un tirage aléatoire.",
            ]
        ),
        _md("## 2. Le point de fonctionnement : arbitrer au budget d'investigation"),
        _code(_BUDGET_CELL, context),
        _insight(
            [
                "Le seuil ne se choisit pas sur un F1 abstrait mais sur la **capacité réelle** de l'équipe : la colonne « fausses alertes » est un coût en jours-analystes.",
                "La courbe de capture montre le gain du classement : à 2 % du flux inspecté, un bon détecteur capture une part de la fraude très supérieure à la diagonale aléatoire.",
                "Doubler le volume d'alertes achète du rappel au prix d'une précision qui chute : c'est un arbitrage de direction, pas un réglage de modèle.",
            ]
        ),
        _md("## 3. Couverture par mode opératoire"),
        _code(_SCHEME_CELL, context),
        _insight(
            [
                "Un rappel global de 0,6 peut cacher un rappel de 0,1 sur un schéma minoritaire — et c'est souvent celui qui croît (card testing automatisé, identités synthétiques).",
                "La fraude amicale est structurellement la plus difficile : la transaction est légitime dans sa forme, seul l'historique de contestations trahit le comportement. Un détecteur transactionnel ne peut pas la voir.",
                "La surveillance de production doit être **ventilée par schéma**, pas agrégée : sinon une dégradation ciblée passe inaperçue tant que la métrique globale tient.",
            ]
        ),
        _md("## 4. Concentration de la fraude : lift par décile"),
        _code(_LIFT_CELL, context),
        _insight(
            [
                "Le lift du décile supérieur mesure la rentabilité de l'investigation : c'est le chiffre à présenter à la direction des risques.",
                "Si le lift est proche de 1 dans tous les déciles, le score n'ordonne rien — même si la ROC AUC semble correcte.",
                "La courbe de capture cumulative sert à dimensionner l'équipe : elle répond directement à « combien d'alertes pour capturer 70 % de la fraude ? ».",
            ]
        ),
        _md("## 5. Analyse des erreurs"),
        _code(_ERRORS_CELL, context),
        _insight(
            [
                "Les **fraudes manquées les mieux classées** sont la marge de progrès accessible : elles portent une signature presque suffisante. Si elles se concentrent sur un schéma, c'est ce schéma qu'il faut outiller.",
                "Les **fausses alertes les plus convaincantes** sont souvent des outliers légitimes (achat de luxe, voyageur d'affaires nocturne, paiement d'entreprise). Elles relèvent d'une liste blanche métier, pas d'un re-réglage du modèle.",
                "Comparer les fraudes capturées aux fraudes manquées, colonne par colonne, identifie la variable qui fait la différence : c'est le prochain feature à construire.",
            ]
        ),
        _md("## 6. Facteurs contributifs (importance par permutation)"),
        _code(_DRIVERS_CELL, context),
        _insight(
            [
                "La permutation mesure la chute du **recouvrement du top-budget** quand une colonne est mélangée : cette définition est identique pour tous les détecteurs, ce qui permet de les comparer.",
                "Une importance **négative** signifie que permuter la colonne améliore le classement : la feature apporte du bruit et doit être retirée ou retravaillée.",
                "Une importance concentrée sur une seule variable (> 45 %) rend le détecteur fragile : il se réduit presque à une règle univariée, que les fraudeurs contournent en premier.",
            ]
        ),
        _md("## 7. Diagnostics automatiques"),
        _code(_DIAGNOSTICS_CELL, context),
        _insight(
            [
                "Les diagnostics sont produits par le même code que le rapport : une faiblesse détectée ici le sera aussi en production, sans intervention manuelle.",
                "Les seuils affichés viennent du cas d'usage (`extras.quality` du manifeste) et sont écrasables par `metrics.thresholds` dans la configuration Hydra : aucun chiffre n'est codé en dur.",
            ]
        ),
        _md("## 8. Recommandations"),
        _code(_RECOMMENDATIONS_CELL, context),
        _insight(
            [
                "Les recommandations **mesurées** (issues des chiffres du split) précèdent les bonnes pratiques documentées du cas d'usage : c'est ce qui les rend actionnables.",
                "Trois leviers reviennent systématiquement en détection de fraude : les features de vélocité à fenêtre courte, les indicateurs de manquants, et la liste blanche des outliers légitimes.",
                "Le détecteur ne remplace pas les règles métier : il couvre les schémas inédits, elles couvrent les obligations réglementaires (LCB-FT). Les deux se combinent dans un classement unique.",
            ]
        ),
        _md("## 9. Génération du rapport d'évaluation"),
        _code(_REPORT_CELL, context),
        _insight(
            [
                "Le rapport Markdown, son équivalent JSON, les tables CSV (budget, couverture par schéma, erreurs) et les figures sont écrits dans `outputs/notebooks/` pour ne jamais écraser les artefacts de `make train`.",
                "Le JSON est la source consommable par un tableau de bord ou un registre de modèles (voir `mlops/model-registry`) ; le Markdown est la source lisible par le responsable fraude.",
                "Rejouer `make evaluate` régénère exactement ces artefacts à partir des modèles entraînés : l'évaluation est reproductible, pas seulement l'entraînement.",
            ]
        ),
    ]
    result_path = write_notebook(destination / "06_error_analysis.ipynb", cells)
    _replace_placeholder(result_path)
    return result_path
