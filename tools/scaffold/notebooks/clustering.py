"""Cellules de notebooks spécifiques à la tâche **clustering** (segmentation non supervisée).

Le squelette des six notebooks (``notebooks/tabular.py``) est partagé par toutes les tâches
tabulaires ; seules les cellules qui parlent de *cible*, de *probabilité de classe* ou d'*erreur de
prédiction* doivent changer. Ce module fournit :

* :func:`structure_cells` — l'analyse de structure dans ``01_eda.ipynb`` (pas de cible : on regarde
  l'échelle des variables, la variance expliquée par l'ACP et les colonnes de diagnostic) ;
* :func:`build_04_model_exploration` — le notebook d'exploration : plancher aléatoire, choix du
  nombre de groupes, comparaison d'algorithmes à k fixé, stabilité entre graines, sensibilité aux
  hyperparamètres ;
* :func:`build_06_error_analysis` — le notebook d'analyse : profils de segments en unités brutes,
  drivers en écarts-types, validité externe (churn observé), affectations fragiles, diagnostics et
  recommandations actionnables.

Les helpers de rendu (``_code``, ``_md``, ``_insight``, jetons ``__XXX__``) sont réutilisés tels
quels : un notebook de clustering et un notebook de classification partagent la même grammaire.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tools.scaffold.notebooks.tabular import (
    FIT_MODEL,
    LOAD_RAW,
    PREPARE,
    RESULT_PLACEHOLDER,
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


# ---------------------------------------------------------------------------------------
# 01 — Structure latente (pas de cible)
# ---------------------------------------------------------------------------------------
_SCALE_CELL = """
feature_columns = [
    column for column in raw.columns if column not in set(CONFIG.data.drop_columns)
]
numeric_columns = [
    column for column in feature_columns if pd.api.types.is_numeric_dtype(raw[column])
]

scale = pd.DataFrame(
    {
        "min": raw[numeric_columns].min(),
        "médiane": raw[numeric_columns].median(),
        "max": raw[numeric_columns].max(),
        "écart-type": raw[numeric_columns].std(),
    }
)
print("Rapport max/écart-type — plus il est grand, plus la variable domine une distance brute :")
scale.assign(
    **{"max / écart-type": (scale["max"] / scale["écart-type"]).round(1)}
).sort_values("max / écart-type", ascending=False).round(2)
"""

_PCA_CELL = """
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ACP sur les features numériques standardisées : c'est l'espace dans lequel le clustering
# travaillera (à l'encodage des catégorielles près).
matrix = raw[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(
    raw[numeric_columns].median(numeric_only=True)
)
scaled = StandardScaler().fit_transform(matrix.to_numpy(dtype="float64"))

pca = PCA(n_components=4, random_state=CONFIG.seed)
projected = pca.fit_transform(scaled)

variance = pd.DataFrame(
    {
        "axe": [f"PC{index}" for index in range(1, len(pca.explained_variance_ratio_) + 1)],
        "variance_expliquée": pca.explained_variance_ratio_.round(4),
        "cumul": pca.explained_variance_ratio_.cumsum().round(4),
    }
)
print(variance.to_string(index=False))

diagnostic_column = "latent_segment" if "latent_segment" in raw.columns else None
map_frame = pd.DataFrame(projected[:, :2], columns=["x", "y"])
fig, axis = plt.subplots(figsize=(8.4, 5.6))
if diagnostic_column:
    # Coloration par le profil latent INJECTÉ PAR LE GÉNÉRATEUR : diagnostic pédagogique
    # uniquement. En production, cette colonne n'existe pas — le clustering est aveugle.
    map_frame["profil"] = raw[diagnostic_column].to_numpy()
    for level, group in map_frame.groupby("profil", observed=True):
        axis.scatter(group["x"], group["y"], s=13, alpha=0.55, label=str(level), edgecolors="none")
    axis.legend(title="profil latent (diagnostic)", fontsize=8, title_fontsize=8, markerscale=1.6)
else:
    axis.scatter(map_frame["x"], map_frame["y"], s=13, alpha=0.55, color="#005f73", edgecolors="none")
axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
axis.set_title("Structure du nuage de clients (ACP 2D)")
fig.tight_layout()
plt.show()

loadings = pd.DataFrame(pca.components_[0:2].T, index=numeric_columns, columns=["PC1", "PC2"])
loadings.reindex(loadings.abs().sum(axis=1).sort_values(ascending=False).index).head(8).round(3)
"""

_DIAGNOSTIC_CELL = """
diagnostic_columns = [
    column for column in ("latent_segment", "churned_next_90d") if column in raw.columns
]

if not diagnostic_columns:
    print("Aucune colonne de diagnostic dans ce jeu de données.")
elif "latent_segment" in raw and "churned_next_90d" in raw:
    external = raw.groupby("latent_segment", observed=True).agg(
        clients=("churned_next_90d", "size"),
        churn_90j=("churned_next_90d", "mean"),
    )
    external["part"] = external["clients"] / len(raw)
    print("Validité externe disponible : le churn observé à 90 jours, par profil latent")
    display(external.round(3).sort_values("churn_90j", ascending=False))
    spread = external["churn_90j"].max() - external["churn_90j"].min()
    print(f"écart de churn entre profils : {spread:.1%}")
    print("Répartition des profils latents (parts attendues dans la population) :")
    display((raw["latent_segment"].value_counts(normalize=True) * 100).round(1).to_frame("%"))
else:
    print("Colonnes de diagnostic présentes :", diagnostic_columns)
"""


def structure_cells(context: NotebookContext) -> list[NotebookNode]:
    """Build the structure-analysis cells of notebook 01 for an **unsupervised** project.

    Args:
        context: Notebook context.

    Returns:
        The cells to insert in notebook 01 (section 5).
    """
    return [
        _md(
            """## 5. Pas de cible : quelle structure cherche-t-on ?

Il n'y a **aucune variable à prédire** ici. L'exploration ne consiste donc pas à expliquer une
cible, mais à répondre à trois questions qui conditionnent toute la suite :

1. **échelle** — les variables sont-elles comparables ? Une distance euclidienne sur des euros et
   des parts (0-1) est dominée par les euros : le clustering serait une segmentation par chiffre
   d'affaires, rien d'autre ;
2. **structure** — existe-t-il une géométrie exploitable (groupes, densités) ou un continuum ?
   L'ACP donne une première réponse honnête : si deux axes expliquent tout, la segmentation sera
   une découpe de plan ;
3. **diagnostic** — de quoi disposera-t-on pour juger la segmentation *après* coup ? Deux colonnes
   de métadonnées existent dans ce jeu synthétique (profil latent injecté, churn observé à 90
   jours). Elles sont **exclues des features** par `drop_columns` : elles servent à valider la
   méthode, pas à l'entraîner.
"""
        ),
        _code(_SCALE_CELL, context),
        _insight(
            [
                "Sans standardisation, `revenue_12m_eur` (milliers d'euros) écrase `discount_share` (0 à 1) : la distance euclidienne ne verrait que le chiffre d'affaires.",
                "C'est la première cause d'échec d'un clustering en production — et elle est invisible si l'on regarde seulement la silhouette (elle aussi calculée sur des distances).",
                "Les queues lourdes (revenus, sessions) appellent un `log1p` **avant** le scaling : c'est exactement ce que fait `conf/preprocessing/default.yaml`.",
            ]
        ),
        _code(_PCA_CELL, context),
        _insight(
            [
                "Si les profils latents se séparent nettement sur les deux premiers axes, la structure est apprenable ; s'ils se mélangent, aucune méthode ne les séparera proprement — et c'est le cas ici par construction (chevauchement volontaire).",
                "Les loadings disent **quoi** portent les axes : un PC1 dominé par le chiffre d'affaires et la fréquence est un axe de valeur, un PC2 dominé par la part promotionnelle est un axe de comportement.",
                "Une variance expliquée faible sur les deux premiers axes n'est pas un échec : la structure vit alors en dimension supérieure, et la carte 2D sous-estime la séparation réelle.",
            ]
        ),
        _code(_DIAGNOSTIC_CELL, context),
        _insight(
            [
                "Ces deux colonnes sont des **métadonnées** : `drop_columns` les exclut de la matrice de features. Elles ne servent qu'à juger la segmentation une fois produite.",
                "Le churn observé à 90 jours est la mesure de **validité externe** : une segmentation dont les groupes ont tous le même taux de churn est statistiquement propre et commercialement inutile.",
                "L'accord avec le profil latent (ARI, NMI) est un diagnostic pédagogique : en production, personne ne connaît la « vraie » segmentation — c'est pourquoi les critères internes et métier priment.",
            ]
        ),
    ]


# ---------------------------------------------------------------------------------------
# 04 — Exploration : plancher, choix de k, algorithmes, stabilité
# ---------------------------------------------------------------------------------------
_BASELINE_CELL = """
from src.evaluation.evaluator import Evaluator
from src.models import build_model
from src.training.losses_metrics import MetricCalculator, MetricInputs

BASE_MODEL = build_model(CONFIG, feature_names=PREPARED["feature_names"])
_ = BASE_MODEL.fit(PREPARED["X_train"], None, X_val=PREPARED["X_val"], y_val=None, callbacks=[])

EVALUATOR = Evaluator.from_config(BASE_MODEL, CONFIG.model_dump(), NB_PATHS)
random_baseline = EVALUATOR.compare_to_baseline(PREPARED["X_val"], baseline="random_assignment")
single_baseline = EVALUATOR.compare_to_baseline(PREPARED["X_val"], baseline="single_cluster")

calculator = MetricCalculator(task=CONFIG.metrics.task, metrics=CONFIG.metrics.all_metrics)
model_metrics = calculator.evaluate(
    MetricInputs(
        y_true=None,
        y_pred=BASE_MODEL.predict(PREPARED["X_val"]),
        X=PREPARED["X_val"],
    )
)

secondary = list(CONFIG.metrics.secondary[:2])
comparison = pd.DataFrame(
    {
        "segmentation": [
            f"{CONFIG.model.algorithm} (configuré)",
            "affectation aléatoire (même k)",
            "groupe unique",
        ],
        CONFIG.metrics.primary: [
            model_metrics.get(CONFIG.metrics.primary, float("nan")),
            random_baseline.get(f"baseline_{CONFIG.metrics.primary}", float("nan")),
            single_baseline.get(f"baseline_{CONFIG.metrics.primary}", float("nan")),
        ],
    }
)
for name in secondary:
    comparison[name] = [
        model_metrics.get(name, float("nan")),
        random_baseline.get(f"baseline_{name}", float("nan")),
        single_baseline.get(f"baseline_{name}", float("nan")),
    ]
comparison.round(4)
"""

_K_SELECTION_CELL = """
k_table = EVALUATOR.k_selection(PREPARED["X_train"])
display(k_table.round(3))

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))

axes[0].plot(k_table["k"], k_table["silhouette"], marker="o", color="#005f73", label="silhouette")
axes[0].set_xlabel("nombre de groupes (k)")
axes[0].set_ylabel("silhouette", color="#005f73")
twin = axes[0].twinx()
twin.plot(k_table["k"], k_table["davies_bouldin"], marker="s", color="#ae2012")
twin.set_ylabel("Davies-Bouldin (plus bas = mieux)", color="#ae2012")
twin.grid(False)
best_k = int(k_table.loc[k_table["silhouette"].idxmax(), "k"])
axes[0].axvline(best_k, color="#94d2bd", linestyle="--")
axes[0].set_title(f"Pic de silhouette : k = {best_k}")

axes[1].plot(k_table["k"], k_table["inertia"], marker="o", color="#bb3e03")
axes[1].set_xlabel("nombre de groupes (k)")
axes[1].set_ylabel("inertie intra-cluster")
axes[1].set_title("Coude d'inertie et taille du plus petit groupe")
twin2 = axes[1].twinx()
twin2.plot(k_table["k"], k_table["min_cluster_share"] * 100, marker="^", color="#6a4c93", linewidth=1.2)
twin2.axhline(EVALUATOR.min_cluster_share * 100, color="#6a4c93", linestyle=":")
twin2.set_ylabel("plus petit groupe (%)", color="#6a4c93")
twin2.grid(False)

fig.tight_layout()
plt.show()

configured_k = int((CONFIG.model.params or {}).get("n_clusters", best_k))
print(f"k retenu par la configuration : {configured_k}")
print(f"k du pic de silhouette        : {best_k}")
lowest_db_k = int(k_table.loc[k_table["davies_bouldin"].idxmin(), "k"])
print(f"Davies-Bouldin minimal        : k = {lowest_db_k}")
K_CHOSEN = configured_k
"""

_ALGORITHMS_CELL = """
from src.models.factory import available_algorithms

ALGORITHMS = available_algorithms(CONFIG.metrics.task)
print(f"{len(ALGORITHMS)} algorithmes disponibles pour la tâche '{CONFIG.metrics.task}' :")
print(ALGORITHMS)

# Comparaison **loyale** : tous les algorithmes sont entraînés avec le MÊME nombre de groupes.
# `n_clusters` (KMeans) et `n_components` (mélange gaussien) sont fournis ensemble : chaque
# estimateur ignore le paramètre qui ne le concerne pas (journalisé en debug par la fabrique).
rows = []
for algorithm in ALGORITHMS:
    try:
        candidate = build_model(
            CONFIG,
            feature_names=PREPARED["feature_names"],
            algorithm=algorithm,
            params={"n_clusters": K_CHOSEN, "n_components": K_CHOSEN},
        )
        result = candidate.fit(
            PREPARED["X_train"], None, X_val=PREPARED["X_val"], y_val=None, callbacks=[]
        )
        labels = candidate.predict(PREPARED["X_val"])
        values = calculator.evaluate(
            MetricInputs(y_true=None, y_pred=labels, X=PREPARED["X_val"])
        )
        shares = pd.Series(labels).value_counts(normalize=True)
        row = {
            "algorithme": algorithm,
            CONFIG.metrics.primary: values.get(CONFIG.metrics.primary, float("nan")),
        }
        for name in CONFIG.metrics.secondary[:2]:
            row[name] = values.get(name, float("nan"))
        row["plus petit groupe (%)"] = (
            round(float(shares.min()) * 100, 2) if len(shares) else float("nan")
        )
        row["groupes"] = int(pd.Series(labels).nunique())
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
ranking.round(4)
"""

_STABILITY_CELL = """
from sklearn.metrics import adjusted_rand_score

reference_labels = np.asarray(BASE_MODEL.predict(PREPARED["X_val"])).ravel()

stability_rows = []
for seed in (7, 21, 42):
    candidate = build_model(CONFIG, feature_names=PREPARED["feature_names"])
    candidate.random_state = seed
    _ = candidate.fit(
        PREPARED["X_train"], None, X_val=PREPARED["X_val"], y_val=None, callbacks=[]
    )
    labels = np.asarray(candidate.predict(PREPARED["X_val"])).ravel()
    values = calculator.evaluate(MetricInputs(y_true=None, y_pred=labels, X=PREPARED["X_val"]))
    stability_rows.append(
        {
            "graine": seed,
            "ari_vs_référence": round(float(adjusted_rand_score(reference_labels, labels)), 4),
            CONFIG.metrics.primary: round(
                float(values.get(CONFIG.metrics.primary, float("nan"))), 4
            ),
        }
    )

stability = pd.DataFrame(stability_rows)
display(stability)
mean_ari = float(stability["ari_vs_référence"].mean())
scores = stability[CONFIG.metrics.primary].to_numpy(dtype="float64")
print(f"ARI moyen entre graines : {mean_ari:.3f}")
print(f"{CONFIG.metrics.primary} : moyenne {np.nanmean(scores):.4f} ± {np.nanstd(scores, ddof=1):.4f}")
"""

_GRID_CELL = """
import itertools

# Grille déclarée dans le manifeste (injectée dans `conf/model/default.yaml`) : aucune valeur
# n'est codée en dur dans le notebook.
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
    labels = candidate.predict(PREPARED["X_val"])
    values = calculator.evaluate(MetricInputs(y_true=None, y_pred=labels, X=PREPARED["X_val"]))
    shares = pd.Series(labels).value_counts(normalize=True)
    row = {str(name): str(value) for name, value in params.items()}
    row[CONFIG.metrics.primary] = values.get(CONFIG.metrics.primary, float("nan"))
    for name in CONFIG.metrics.secondary[:2]:
        row[name] = values.get(name, float("nan"))
    row["plus petit groupe (%)"] = (
        round(float(shares.min()) * 100, 2) if len(shares) else float("nan")
    )
    row["secondes"] = round(result.duration_seconds, 2)
    grid_rows.append(row)

grid_results = pd.DataFrame(grid_rows).sort_values(
    CONFIG.metrics.primary, ascending=ascending, na_position="last"
)
grid_results.round(4)
"""


def build_04_model_exploration(context: NotebookContext, destination: Path) -> Path:
    """Build ``04_model_exploration.ipynb`` for a clustering project.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    spec = context.spec
    cells: list[NotebookNode] = [
        _md(
            f"""# 04 — Exploration et comparaison de modèles

**Projet** : {spec.title}
**Modèle configuré** : `{spec.model.algorithm}` ({spec.model.display_name})
**Pourquoi ce choix** : {spec.model.rationale}

En apprentissage non supervisé, la question « pourquoi cet algorithme ? » se dédouble :

* **combien de groupes ?** — le nombre de segments est un choix, pas une découverte automatique ;
* **quelle méthode ?** — à k fixé, les algorithmes ne produisent pas la même géométrie (sphères de
  taille comparable pour KMeans, ellipsoïdes et probabilités d'appartenance pour un mélange
  gaussien).

Ce notebook répond aux deux **par des chiffres** : plancher aléatoire, critères internes, taille
des groupes, stabilité entre graines.
"""
        ),
        _objectives(
            context,
            [
                "Commencer par le **plancher** : une affectation aléatoire de même taille (silhouette ≈ 0).",
                "Choisir le nombre de groupes par triangulation (silhouette, Davies-Bouldin, coude d'inertie, taille minimale).",
                "Comparer les algorithmes de la stack **à k fixé** : la comparaison n'a de sens qu'à structure égale.",
                "Mesurer la **stabilité** des affectations entre graines : une segmentation instable ne pilote pas de campagnes.",
            ],
        ),
        _code(SETUP, context),
        _code(LOAD_RAW, context),
        _code(PREPARE, context),
        _md("## 1. Le plancher d'abord : une affectation aléatoire"),
        _code(_BASELINE_CELL, context),
        _insight(
            [
                "Une affectation aléatoire de même taille a une silhouette proche de 0 : c'est le **plancher**. Toute segmentation publiée doit s'en détacher nettement.",
                "Le groupe unique est dégénéré (silhouette indéfinie) : il rappelle que « ne pas segmenter » n'est pas une option mesurable, c'est une absence de modèle.",
                f"La métrique de décision du projet est `{spec.metrics.primary}` (sens : {spec.metrics.direction}) ; le seuil de qualité déclaré est {spec.metrics.min_primary}.",
            ]
        ),
        _md("## 2. Combien de groupes ? Triangulation des critères"),
        _code(_K_SELECTION_CELL, context),
        _insight(
            [
                "Les critères **ne convergent pas toujours** : pic de silhouette, minimum de Davies-Bouldin et coude d'inertie peuvent désigner des k différents. C'est normal — ils ne mesurent pas la même chose.",
                "La taille du plus petit groupe est un critère **métier** : sous ~3 % de la base, un segment ne justifie pas une campagne dédiée (coût, lisibilité, volume statistique).",
                "Le k retenu ici est celui de la configuration : l'écart avec le pic statistique est un arbitrage assumé et documenté, pas une erreur.",
            ]
        ),
        _md("## 3. Comparaison des algorithmes de la stack (à k fixé)"),
        _code(_ALGORITHMS_CELL, context),
        _insight(
            [
                "KMeans suppose des groupes **convexes** de taille comparable ; un mélange gaussien autorise des ellipsoïdes et produit des probabilités d'appartenance (donc une confiance par client).",
                "MiniBatchKMeans échange un peu de qualité contre un coût mémoire constant : c'est l'option des bases de plusieurs millions de clients.",
                "Un écart de silhouette inférieur à 0.02 entre deux algorithmes n'est pas un signal : il faut le comparer à la variabilité entre graines (section 4).",
            ]
        ),
        _md("## 4. Stabilité : les clients gardent-ils leur segment d'une graine à l'autre ?"),
        _code(_STABILITY_CELL, context),
        _insight(
            [
                "L'**ARI entre graines** mesure ce que le métier redoute le plus : un client qui change de segment d'un jour à l'autre reçoit des messages contradictoires.",
                "KMeans converge vers des optima locaux : augmenter `n_init` (nombre de ré-initialisations) est le premier levier de stabilité, avant tout changement d'algorithme.",
                "Règle pratique : ne pas célébrer un gain de silhouette inférieur à 2× l'écart-type observé entre graines.",
            ]
        ),
        _md("## 5. Sensibilité aux hyperparamètres"),
        _code(_GRID_CELL, context),
        _insight(
            [
                "Le tri respecte le **sens** de la métrique (`direction: maximize` pour une silhouette) : un tri ascendant par défaut classerait les pires modèles en premier.",
                "Une grille se lit aussi par sa **dispersion** : si toutes les combinaisons se tiennent en 0.01 de silhouette, le modèle est robuste — et le réglage fin n'est pas le levier principal.",
                "Attention au sur-ajustement sur le critère interne : maximiser la silhouette peut produire des groupes géométriquement nets mais métier-inutiles (micro-groupes, séparation par la région).",
            ]
        ),
        _md(
            """## 6. Choix argumenté

| Critère | Lecture | Décision |
| --- | --- | --- |
| Silhouette (validation) | structure réelle vs bruit | doit dépasser le seuil du cas d'usage |
| Davies-Bouldin | compacité / séparation | plus bas = mieux, mais sensible aux outliers |
| Taille du plus petit groupe | exploitabilité CRM | ≥ 3 % de la base, sinon fusion |
| ARI entre graines | reproductibilité des affectations | ≥ 0.85 avant mise en production |
| Coût d'entraînement | fenêtre de batch nocturne | MiniBatchKMeans si > 1 M de clients |

**Règle de décision retenue** : choisir le plus petit k qui satisfait la silhouette et la
stabilité, puis vérifier la lisibilité métier des profils (notebook 06). Un k supérieur n'est
justifié que s'il sépare un enjeu d'action distinct.
"""
        ),
    ]
    result_path = write_notebook(destination / "04_model_exploration.ipynb", cells)
    _replace_placeholder(result_path)
    return result_path


# ---------------------------------------------------------------------------------------
# 06 — Analyse de segmentation et recommandations
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
print(f"clients évalués          : {RESULT.n_samples}")
print(f"groupes produits         : {RESULT.n_clusters}")
print(f"plus petit groupe        : {RESULT.min_cluster_share:.1%}")
print(f"stabilité (ARI moyen)    : {RESULT.stability_ari:.3f}")
print(f"accord latent (ARI)      : {RESULT.ari_latent:.3f}")
print(f"écart de churn           : {RESULT.churn_spread:.1%}")
print(f"groupes dégénérés        : {RESULT.degenerate_clusters or 'aucun'}")
metrics_frame.round(4)
"""

_PROFILES_CELL = """
profiles = RESULT.per_segment
preferred = {
    "cluster", "size", "share", "mean_distance", "mean_silhouette", "churn_rate",
    "dominant_latent", "latent_purity",
}
display_columns = [
    column for column in profiles.columns
    if column in preferred or str(column).startswith("mean_")
]
display(profiles[display_columns].round(3))

revenue_column = next(
    (column for column in profiles.columns if str(column).endswith("revenue_12m_eur")), None
)
if revenue_column:
    contribution = (profiles[revenue_column] * profiles["share"]).sort_values(ascending=False)
    total = float(contribution.sum())
    print("Contribution relative au chiffre d'affaires (à lire en relatif) :")
    for cluster, value in contribution.items():
        share = value / total if total else float("nan")
        print(f"  segment {cluster} : {share:.1%}")
"""

_DRIVERS_CELL = """
centroids = RESULT.feature_importance
top_drivers = (
    centroids[centroids["rank"] <= 4]
    .pivot_table(index="cluster", columns="feature", values="centroid_z")
    .round(2)
)
display(top_drivers)

pivot = centroids.pivot_table(index="feature", columns="cluster", values="centroid_z")
strength = centroids.groupby("feature")["centroid_z"].apply(lambda series: float(series.abs().max()))
keep = strength.sort_values(ascending=False).head(12).index.tolist()
pivot = pivot.loc[[name for name in keep if name in pivot.index]]

fig, axis = plt.subplots(figsize=(0.9 * len(pivot.columns) + 6.4, 0.42 * len(pivot) + 2.0))
image = axis.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
axis.set_xticks(range(len(pivot.columns)), [f"segment {name}" for name in pivot.columns], fontsize=8)
axis.set_yticks(range(len(pivot.index)), pivot.index, fontsize=8)
for row in range(pivot.shape[0]):
    for column in range(pivot.shape[1]):
        value = pivot.iloc[row, column]
        axis.text(column, row, f"{value:+.1f}", ha="center", va="center", fontsize=7,
                  color="black" if abs(value) < 1.4 else "white")
fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03, label="écart-type vs client moyen")
axis.set_title("Profil des segments : ce qui distingue chaque groupe")
fig.tight_layout()
plt.show()
"""

_EXTERNAL_CELL = """
external = dict(RESULT.extras.get("external_validity", {}))
stability = dict(RESULT.extras.get("stability", {}))

print("Accord avec la structure latente (diagnostic pédagogique) :")
print(f"  ARI        : {float(external.get('ari_latent', float('nan'))):.3f}")
print(f"  NMI        : {float(external.get('nmi_latent', float('nan'))):.3f}")
print(f"  V-measure  : {float(external.get('v_measure_latent', float('nan'))):.3f}")

churn = external.get("churn_by_cluster", {})
if churn:
    churn_frame = pd.DataFrame(
        {"segment": list(churn), "churn_90j": [float(value) * 100 for value in churn.values()]}
    ).sort_values("churn_90j", ascending=False)
    display(churn_frame.round(1))
    fig, axis = plt.subplots(figsize=(7.6, 3.8))
    axis.bar(churn_frame["segment"].astype(str), churn_frame["churn_90j"], color="#ae2012")
    axis.set_ylabel("churn à 90 jours (%)")
    axis.set_xlabel("segment")
    axis.set_title("Validité externe : la segmentation sépare-t-elle un risque réel ?")
    fig.tight_layout()
    plt.show()

print("Stabilité des affectations (ARI contre l'affectation de référence) :")
for seed, value in dict(stability.get("per_seed", {})).items():
    print(f"  graine {seed} : {float(value):.3f}")
print(f"  moyenne : {float(stability.get('mean_ari', float('nan'))):.3f}")
"""

_BORDERLINE_CELL = """
rows = RESULT.predictions
confidence = rows["confidence"].value_counts(normalize=True)
print("Confiance des affectations :")
display((confidence * 100).round(1).to_frame("%"))

borderline = RESULT.errors
display_columns = [
    column for column in ("cluster", "silhouette", "distance_to_centroid", "confidence")
    if column in borderline.columns
]
display(borderline[display_columns].head(12).round(3))

negative_share = float((pd.to_numeric(rows["silhouette"], errors="coerce") < 0).mean())
print(f"clients plus proches d'un autre groupe (silhouette < 0) : {negative_share:.1%}")

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
values = pd.to_numeric(rows["silhouette"], errors="coerce").dropna()
axes[0].hist(values, bins=36, color="#0a9396", edgecolor="white")
axes[0].axvline(float(values.mean()), color="#ae2012", linestyle="--",
                label=f"moyenne {values.mean():.3f}")
axes[0].axvline(0.0, color="#495057", linestyle=":")
axes[0].legend(fontsize=8)
axes[0].set_title("Silhouette individuelle")
axes[0].set_xlabel("silhouette")

projection = dict(RESULT.curves.get("pca_map", {}))
if projection.get("x"):
    scatter_frame = pd.DataFrame(
        {"x": projection["x"], "y": projection["y"],
         "segment": [str(value) for value in projection["cluster"]]}
    )
    for segment, group in scatter_frame.groupby("segment", observed=True):
        axes[1].scatter(group["x"], group["y"], s=12, alpha=0.55, label=f"segment {segment}",
                        edgecolors="none")
    axes[1].legend(fontsize=7, markerscale=1.8, title="segment", title_fontsize=7)
axes[1].set_title("Carte ACP des clients évalués")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
fig.tight_layout()
plt.show()
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
    print(f"{kind:<12} -> {relative}")

display(Markdown(f"### Rapport généré\\n\\n`{artifacts['report']}`"))
if "profile_heatmap" in artifacts:
    display(Image(filename=str(artifacts["profile_heatmap"]), width=760))
"""


def build_06_error_analysis(context: NotebookContext, destination: Path) -> Path:
    """Build ``06_error_analysis.ipynb`` for a clustering project.

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
            f"""# 06 — Analyse de la segmentation et recommandations

**Projet** : {spec.title}
**Principe** : une silhouette globale ne dit **jamais** si la segmentation sert le métier. Un
clustering peut être géométriquement propre et produire des groupes inexploitables (micro-groupes,
profils illisibles, clients à la frontière). Ce notebook descend au niveau du **segment** puis du
**client** : qui est dans chaque groupe, qu'est-ce qui le définit, quels clients sont mal affectés,
et que fait-on lundi matin ?

Le split de **test** n'est utilisé qu'ici — une seule fois — pour rester une estimation honnête.
"""
        ),
        _objectives(
            context,
            [
                "Produire une évaluation complète : critères internes, tailles de groupes, confiance des affectations.",
                "Profiler chaque segment **en unités brutes** (euros, jours, commandes) et identifier ses drivers en écarts-types.",
                "Vérifier la **validité externe** : les groupes séparent-ils un comportement observé après coup (churn à 90 jours) ?",
                "Repérer les affectations fragiles et les groupes dégénérés, puis formuler des recommandations chiffrées.",
            ],
        ),
        _code(SETUP, context),
        _code(LOAD_RAW, context),
        _code(PREPARE, context),
        _code(FIT_MODEL, context),
        _md("## 1. Évaluation sur le split de test"),
        _code(_EVALUATE_CELL, context),
        _insight(
            [
                f"La métrique de décision est `{spec.metrics.primary}` = **{RESULT_PLACEHOLDER}** — c'est elle qui pilote le seuil de qualité (`metrics.min_primary`).",
                "Les critères internes (silhouette, Davies-Bouldin, Calinski-Harabasz) jugent la **géométrie** ; la taille des groupes et la stabilité jugent l'**exploitabilité** ; le churn observé juge l'**utilité métier**.",
                "Un seul de ces trois regards ne suffit jamais : c'est la raison d'être de ce notebook.",
            ]
        ),
        _md("## 2. Profils des segments (unités brutes)"),
        _code(_PROFILES_CELL, context),
        _insight(
            [
                "Un profil se lit en **unités métier** : panier moyen en euros, récence en jours, part promotionnelle en pourcentage. Les écarts-types standardisés du modèle ne parlent ni au CRM ni au marketing.",
                "La colonne `dominant_latent` (diagnostic) montre quel profil injecté le groupe retrouve, et `latent_purity` la part de ce profil dans le groupe : une pureté de 60 % signifie que 4 clients sur 10 viennent d'ailleurs.",
                "La concentration de la valeur est l'argument de priorisation : si 8 % des clients portent 40 % du chiffre d'affaires, la rétention de ce groupe passe avant toute acquisition.",
            ]
        ),
        _md("## 3. Ce qui définit chaque segment (écarts-types)"),
        _code(_DRIVERS_CELL, context),
        _insight(
            [
                "Les coordonnées du centroïde sont des **tailles d'effet** : `+1,8` sur la part promotionnelle signifie « 1,8 écart-type au-dessus du client moyen ». C'est ce qui rend un groupe nommable.",
                "Un groupe dont aucun driver ne sort de ±0,5 écart-type est un groupe « moyen » : il n'a pas de personnalité et doit probablement être fusionné.",
                "Vigilance sur les variables **dérivées** (ratios, panier moyen, palier de fidélité) : si elles dominent un centroïde, la segmentation redécouvre une identité arithmétique ou une règle métier, pas un comportement.",
            ]
        ),
        _md("## 4. Validité externe et stabilité"),
        _code(_EXTERNAL_CELL, context),
        _insight(
            [
                "L'ARI latent n'est **pas** un objectif de production : personne ne connaît la vraie segmentation. C'est un outil de mise au point qui dit si la méthode retrouve une structure connue.",
                "Le churn observé à 90 jours est la vraie validation : une segmentation dont les groupes diffèrent de 20 points de churn pilote directement une campagne de rétention.",
                "Une stabilité faible (ARI < 0.85) interdit la mise en production, quelle que soit la silhouette : les clients changeraient de groupe à chaque ré-entraînement.",
            ]
        ),
        _md("## 5. Affectations fragiles et carte des clients"),
        _code(_BORDERLINE_CELL, context),
        _insight(
            [
                "La part de clients à silhouette négative est plus informative que la moyenne : ce sont les clients **à la frontière**, ceux pour lesquels une campagne ciblée est un pari.",
                "La carte ACP rend le chevauchement visible : des groupes qui s'interpénètrent sur le plan ne sont pas nécessairement mal séparés en dimension supérieure, mais ils signalent un continuum comportemental.",
                "En production, ces clients doivent déclencher une règle métier (campagne générique, revue humaine) plutôt qu'un message personnalisé : c'est le rôle du niveau de confiance publié par le prédicteur.",
            ]
        ),
        _md("## 6. Diagnostics : ce qui affaiblit cette segmentation"),
        _code(_DIAGNOSTICS_CELL, context),
        _insight(
            [
                "Les diagnostics sont **calculés**, pas récités : chaque faiblesse observée (groupe dégénéré, instabilité, faible validité externe, frontières floues, variable circulaire) déclenche sa propre hypothèse.",
                "Les seuils viennent de la configuration (`metrics.thresholds`) : un critère de qualité qui vit dans le code n'est pas auditable.",
                "Un diagnostic sans chiffre associé n'est pas une hypothèse, c'est une opinion.",
            ]
        ),
        _md("## 7. Recommandations"),
        _code(_RECOMMENDATIONS_CELL, context),
        _md(
            f"""### déclenchées par les chiffres observés

Les recommandations ci-dessus sont produites par `ReportBuilder.recommendations()` : les premières
répondent aux diagnostics mesurés sur ce split, les suivantes sont les bonnes pratiques du cas
d'usage.

### documentées pour ce cas d'usage

{recommendation_lines or "1. Rejouer la segmentation sur un échantillon plus large avant toute décision."}

### plan d'action proposé

| Priorité | Action | Effet attendu | Comment vérifier |
| --- | --- | --- | --- |
| 1 | Nommer chaque segment et lui associer une action CRM (message, canal, remise) | segmentation adoptée par le métier | `segmentation.labels` dans `conf/config.yaml` |
| 2 | Router les clients à confiance faible vers une campagne générique | baisse des messages inadaptés | `artifacts/reports/segment_assignments.csv` |
| 3 | Stabiliser l'entraînement (graine fixée, `n_init` augmenté) avant mise en production | ARI inter-graines ≥ 0.85 | notebook 04, §4 |
| 4 | Retirer les variables circulaires (palier de fidélité) de l'espace de description | segmentation comportementale, pas arithmétique | notebook 06, §3 (drivers) |
| 5 | Instrumenter la dérive (taille des groupes, distribution des distances) | alerte précoce | `mlops/model-monitoring` |
| 6 | Rejouer ce notebook à chaque nouvelle version de données | non-régression | `make evaluate` + CI |

## 8. Limites assumées

- Les données sont **synthétiques** : les niveaux de qualité illustrent une méthode, pas une base clients réelle.
- `latent_segment` et `churned_next_90d` sont des **métadonnées** : exclues des features, elles ne servent qu'au diagnostic. En production, aucune des deux n'existe au moment de segmenter.
- Une seule passe d'évaluation sur un split unique : la dérive temporelle de la base clients n'est pas mesurée ici.
- Le générateur mélange volontairement une partie des clients entre deux profils : une silhouette très élevée signerait une structure artificielle, pas une bonne méthode.
- L'analyse porte sur {spec.data.n_samples} clients générés, dont une fraction en test : les segments rares restent peu observés.

**Aller plus loin dans le dépôt** : comparaison multi-stacks (`data-science/regression/with-*`,
`data-science/classification/with-*`), mise en production et suivi (`mlops/`), pipelines de données
(`data-eng/`).
"""
        ),
        _md("## 9. Génération du rapport et des figures"),
        _code(_REPORT_CELL, context),
        _insight(
            [
                "Le rapport Markdown, son équivalent JSON, les profils CSV et les figures sont des **artefacts versionnables** : ils rendent la segmentation auditable six mois plus tard.",
                "`make evaluate` rejoue exactement cette chaîne de façon non interactive — c'est le même code, pas une copie.",
            ]
        ),
    ]
    result_path = write_notebook(destination / "06_error_analysis.ipynb", cells)
    _replace_placeholder(result_path)
    return result_path
