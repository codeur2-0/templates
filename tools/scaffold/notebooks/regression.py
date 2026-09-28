"""Cellules de notebooks spécifiques à la tâche **régression**.

Le squelette des six notebooks (``notebooks/tabular.py``) est partagé par toutes les tâches
tabulaires ; seules les cellules qui parlent de *classe*, de *probabilité* ou de *seuil de
décision* doivent changer. Ce module fournit :

* :func:`target_cells` — l'analyse de la cible continue dans ``01_eda.ipynb`` (distribution,
  asymétrie, prix unitaire, médiane par segment) ;
* :func:`build_06_error_analysis` — le notebook d'analyse d'erreurs en régression (résidus,
  biais segmenté, couverture de la fourchette métier, pires erreurs, recommandations).

Les helpers de rendu (``_code``, ``_md``, ``_insight``, jetons ``__XXX__``) sont réutilisés tels
quels : un notebook de régression et un notebook de classification partagent la même grammaire.
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

__all__ = ["build_06_error_analysis", "target_cells"]


# ---------------------------------------------------------------------------------------
# 01 — Analyse de la cible continue
# ---------------------------------------------------------------------------------------
def target_cells(context: NotebookContext) -> list[NotebookNode]:
    """Build the target-analysis cells of notebook 01 for a **continuous** target.

    Args:
        context: Notebook context.

    Returns:
        The cells to insert in notebook 01 (section 5).
    """
    return [
        _md(
            """## 5. La cible

Une cible continue ne se lit pas comme une étiquette : ce qui compte ici est la **forme** de la
distribution (asymétrie, queue lourde), l'**échelle** (les erreurs absolues croissent avec la
valeur) et la **stabilité** dans le temps. Ces trois lectures décident de la transformation de
cible, de la métrique de pilotage et de la largeur de fourchette à publier.
"""
        ),
        _code(
            """
target = CONFIG.data.target
values = raw[target].astype("float64")

summary = pd.DataFrame(
    {
        "statistique": [
            "effectif", "min", "P5", "médiane", "moyenne", "P95", "max",
            "écart-type", "asymétrie", "aplatissement",
        ],
        "valeur": [
            len(values), values.min(), values.quantile(0.05), values.median(), values.mean(),
            values.quantile(0.95), values.max(), values.std(), values.skew(), values.kurt(),
        ],
    }
)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
axes[0].hist(values, bins=50, color="#0a9396", edgecolor="white")
axes[0].axvline(values.median(), color="#d1495b", linestyle="--", label="médiane")
axes[0].axvline(values.mean(), color="#ee9b00", linestyle=":", label="moyenne")
axes[0].set_title(f"Distribution de `{target}`")
axes[0].set_xlabel(target)
axes[0].set_ylabel("nombre de biens")
axes[0].legend(fontsize=8)
axes[1].hist(values.clip(upper=values.quantile(0.99)), bins=50, color="#005f73", edgecolor="white")
axes[1].set_title("Zoom : 99 % de la population")
axes[1].set_xlabel(target)
fig.tight_layout()
plt.show()

summary.round(1)
""",
            context,
        ),
        _insight(
            [
                "Moyenne ≫ médiane = distribution **asymétrique à droite** : quelques biens d'exception tirent la moyenne. La RMSE hérite de cette sensibilité, pas la MAE.",
                "Une cible asymétrique se modélise mieux en **log** (`log1p` ou cible transformée) : la variance devient homogène et l'erreur relative devient l'erreur absolue.",
                "Ne **jamais** supprimer les valeurs extrêmes légitimes : les winsoriser (0,5-99,5 %) ou les traiter par une métrique robuste (erreur médiane) est préférable.",
            ]
        ),
        _code(
            """
# Asymétrie : la cible devient-elle gaussienne une fois passée en log ?
target = CONFIG.data.target
values = raw[target].astype("float64")
logged = np.log(values.clip(lower=1.0))

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
for axis, series, title in (
    (axes[0], values, "échelle brute"),
    (axes[1], logged, "échelle logarithmique"),
):
    axis.hist(series, bins=50, color="#94d2bd", edgecolor="white")
    axis.set_title(f"Asymétrie = {series.skew():.2f} — {title}")
    axis.set_xlabel(target)
fig.tight_layout()
plt.show()

print(f"asymétrie brute : {values.skew():.3f}")
print(f"asymétrie log   : {logged.skew():.3f}")
""",
            context,
        ),
        _insight(
            [
                "Une asymétrie qui passe de > 1 à ≈ 0 en log confirme un processus **multiplicatif** : les facteurs de prix se cumulent en pourcentages, pas en euros.",
                "Conséquence pratique : entraîner sur le log-prix, puis revenir en euros avec une correction de biais (`exp(mu + sigma²/2)`), sinon on sous-estime systématiquement.",
                "Le générateur de ce projet implémente exactement ce mécanisme — l'observer ici, c'est vérifier que la donnée ressemble au métier.",
            ]
        ),
        _code(
            """
# Prix unitaire : la même lecture, ramenée à la surface (le ratio que le métier manipule).
target = CONFIG.data.target
unit_column = next(
    (column for column in ("surface_m2", "surface", "area_m2") if column in raw.columns), None
)
if unit_column is None:
    print("Aucune colonne de surface : le prix unitaire n'est pas calculable ici.")
else:
    unit_price = raw[target] / raw[unit_column].replace(0, np.nan)
    print(f"prix unitaire médian : {unit_price.median():,.0f} / {unit_column}")
    p10 = unit_price.quantile(0.10)
    p90 = unit_price.quantile(0.90)
    print(f"P10 / P90            : {p10:,.0f} / {p90:,.0f}")

    by_surface = raw.assign(_bucket=pd.qcut(raw[unit_column], q=5, duplicates="drop")).groupby(
        "_bucket", observed=True
    ).agg(
        n=(target, "size"),
        prix_medien=(target, "median"),
        surface_mediane=(unit_column, "median"),
    )
    by_surface["prix_m2_medien"] = by_surface["prix_medien"] / by_surface["surface_mediane"]
    display(by_surface.round(1))
""",
            context,
        ),
        _insight(
            [
                "Si le prix au m² **décroît** avec la surface, la relation prix/surface est **concave** : un modèle linéaire sur-évaluera les grands biens et sous-évaluera les studios.",
                "Ce ratio est aussi l'unité de lecture du métier : le présenter dans le rapport rend l'estimation crédible.",
                "Une feature dérivée (`surface_per_room`, `tax_per_surface`) est déjà déclarée dans `conf/preprocessing/default.yaml` : la configuration, pas le code, porte ces choix.",
            ]
        ),
        _code(
            """
# Signal par variable catégorielle : médiane de la cible par modalité (et volume associé).
target = CONFIG.data.target
rows = []
for column in categorical_columns:
    grouped = raw.groupby(raw[column].astype(str))[target].agg(["median", "mean", "count"])
    if grouped.empty or grouped["median"].nunique() < 2:
        continue
    rows.append(
        {
            "variable": column,
            "modalités": len(grouped),
            "médiane_min": round(float(grouped["median"].min()), 0),
            "médiane_max": round(float(grouped["median"].max()), 0),
            "ratio_max_min": round(float(grouped["median"].max() / max(grouped["median"].min(), 1.0)), 2),
            "modalité_la_plus_chère": grouped["median"].idxmax(),
        }
    )
signal_frame = pd.DataFrame(rows).sort_values("ratio_max_min", ascending=False)
signal_frame
""",
            context,
        ),
        _insight(
            [
                "Un `ratio_max_min` élevé (≥ 1,5) entre modalités = un facteur de prix puissant, directement exploitable par le modèle.",
                "Un ratio proche de 1 ne signifie pas « inutile » : la variable peut interagir avec une autre (étage × ascenseur).",
                "Le volume par modalité compte autant que l'écart : un segment à 30 observations ne justifie pas une feature dédiée.",
            ]
        ),
        _code(
            """
# Signal par variable numérique : médiane de la cible par quartile.
target = CONFIG.data.target
plots = [column for column in numeric_columns if raw[column].nunique() > 4][:6]
fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.4))
for axis, column in zip(np.atleast_1d(axes).ravel(), plots, strict=False):
    buckets = pd.qcut(raw[column], q=4, duplicates="drop")
    grouped = raw.groupby(buckets, observed=True)[target].median()
    grouped.plot.bar(ax=axis, color="#005f73", edgecolor="white")
    axis.set_title(f"Médiane de la cible par quartile — {column}", fontsize=8.5)
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
                "Une relation **monotone** est apprise facilement, y compris par un modèle linéaire.",
                "Une relation en **U** ou **concave** impose des transformations (binning, log, splines) ou un modèle non linéaire : c'est précisément le cas de la surface et de l'année de construction.",
                "Le binning par quantiles est aussi une recette déclarée dans `conf/preprocessing/default.yaml` (`bin`) : l'EDA et la configuration racontent la même histoire.",
            ]
        ),
    ]


# ---------------------------------------------------------------------------------------
# 06 — Analyse d'erreurs (régression)
# ---------------------------------------------------------------------------------------
def build_06_error_analysis(context: NotebookContext, destination: Path) -> Path:
    """Build ``06_error_analysis.ipynb`` for a regression project.

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
**Principe** : une métrique globale ne dit **jamais** quoi corriger. En régression, une RMSE
correcte peut cacher une sous-estimation systématique d'un quartier entier. Ce notebook descend
au niveau de la ligne : où le modèle se trompe-t-il, dans quel sens, et que fait-on lundi matin ?

Le split de **test** n'est utilisé qu'ici — une seule fois — pour rester une estimation honnête.
"""
        ),
        _objectives(
            context,
            [
                "Produire une évaluation complète (métriques globales, résidus, couverture de fourchette).",
                "Détecter un **biais segmenté** : le modèle se trompe-t-il toujours dans le même sens ?",
                "Identifier les pires erreurs et leur cause probable (queue de distribution, marché fin).",
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
print(f"observations évaluées : {RESULT.n_samples}")
print(f"biais moyen           : {RESULT.bias:,.0f}")
print(f"couverture ± {EVALUATOR.tolerance_pct:.0f} %   : {RESULT.coverage:.1%}")
print(f"part hors fourchette  : {RESULT.error_rate:.1%}")
metrics_frame.round(4)
""",
            context,
        ),
        _insight(
            [
                f"La métrique de décision est `{spec.metrics.primary}` = **{RESULT_PLACEHOLDER}** — c'est elle qui pilote le seuil de qualité.",
                "La RMSE se lit **avec** la MAE : un rapport RMSE/MAE élevé (> 1,6) signale une queue d'erreur lourde, pas un modèle globalement mauvais.",
                "Le `context` passé à `evaluate()` permet d'enrichir l'analyse avec les colonnes brutes (identifiant, quartier, surface).",
                "La couverture de fourchette est la traduction métier de l'erreur : c'est elle qui détermine si l'estimation est publiable.",
            ]
        ),
        _md("## 2. Prédit vs observé — le premier réflexe"),
        _code(
            """
from src.visualization.plots import RegressionPlots

PLOTS = RegressionPlots(NB_PATHS.figures_dir)
for name in ("predicted_vs_actual", "residuals_vs_predicted"):
    path = getattr(PLOTS, name)(RESULT)
    if path is not None:
        display(Image(path, width=520))
""",
            context,
        ),
        _insight(
            [
                "Un nuage **centré sur la diagonale** sans structure = pas de biais global. Un nuage qui s'écarte de la diagonale aux extrémités = le modèle « lisse » la queue de distribution.",
                "Le graphique des résidus est le test d'**hétéroscédasticité** : un cône qui s'ouvre avec la valeur prédite est attendu sur un prix (erreur multiplicative).",
                "La ligne rouge (résidu moyen par classe) doit rester proche de zéro : une dérive monotone indique un effet non appris ou une retransformation biaisée.",
            ]
        ),
        _md("## 3. Distribution de l'erreur et couverture de la fourchette"),
        _code(
            """
for name in ("error_distribution", "error_by_bucket", "coverage_by_bucket", "error_breakdown"):
    path = getattr(PLOTS, name)(RESULT)
    if path is not None:
        display(Image(path, width=540))
""",
            context,
        ),
        _code(
            """
# Lecture chiffrée de la fourchette métier : ce que le produit publie réellement.
frame = RESULT.predictions
tolerance = EVALUATOR.tolerance_pct
print(f"fourchette publiée  : ± {tolerance:.0f} %")
print(f"couverture observée : {RESULT.coverage:.1%} (cible : >= 70 %)")
print(f"erreur médiane      : {frame['absolute_error'].median():,.0f}")
print(f"erreur P95          : {frame['absolute_error'].quantile(0.95):,.0f}")
print(f"erreur relative P95 : {frame['relative_error_pct'].abs().quantile(0.95):.2f} %")
print(f"biais relatif moyen : {frame['relative_error_pct'].mean():+.2f} %")
""",
            context,
        ),
        _insight(
            [
                "Une couverture sous 70 % n'est pas forcément un échec du modèle : la fourchette est peut-être **trop étroite** pour le niveau de bruit réel. Les deux leviers existent (modèle, largeur).",
                "L'erreur **médiane** décrit le portefeuille typique, la P95 décrit le risque : les publier ensemble évite les débats stériles sur « la » bonne métrique.",
                "Un biais relatif moyen non nul (> ±2 %) se corrige souvent sans ré-entraîner : correction de retransformation log, ou recalibrage par segment.",
            ]
        ),
        _md("## 4. Analyse par segment — là où se décide la prochaine itération"),
        _code(
            """
segments = RESULT.per_segment
if segments.empty:
    print("Aucun axe de segmentation exploitable sur ce run.")
else:
    display(segments.round(3).head(20))
    worst = segments.assign(_abs_bias=segments["bias_pct"].abs()).sort_values(
        "_abs_bias", ascending=False
    ).head(5)
    print("--- segments les plus biaisés ---")
    display(worst.drop(columns=["_abs_bias"]).round(3))
""",
            context,
        ),
        _insight(
            [
                "Un segment dont la couverture s'effondre concentre le risque métier : c'est lui qu'il faut instrumenter en premier, pas la métrique globale.",
                "Vérifier ensuite le **volume** d'apprentissage du segment (`n`) : sous-représentation = sous-performance, et la réponse est donnée (feature, sur-échantillonnage, modèle dédié).",
                "Un biais de signe constant sur un segment est un signal fort : soit une feature manque, soit la transformation de cible n'est pas cohérente sur cette population.",
            ]
        ),
        _md("## 5. Où le modèle se trompe-t-il ?"),
        _code(
            """
if RESULT.errors.empty:
    print("Aucune erreur enregistrée sur le split de test.")
else:
    print(f"{len(RESULT.errors)} pires erreurs (triées par erreur relative décroissante)")
    display(RESULT.errors.head(12).round(2))
""",
            context,
        ),
        _code(
            """
# Segmentation des erreurs : quelle population concentre les erreurs relatives ?
frame = RESULT.predictions
segmentation_columns = [column for column in (__CATEGORICAL__) if column in frame.columns][:3]
for column in segmentation_columns:
    table = (
        frame.groupby(frame[column].astype(str))
        .agg(
            n=("y_true", "size"),
            erreur_relative_mediane=("relative_error_pct", "median"),
            biais_moyen=("residual", "mean"),
            couverture=("within_tolerance", "mean"),
        )
        .sort_values("couverture")
    )
    print(f"--- erreurs par `{column}` ---")
    display(table.round(3))
""",
            context,
        ),
        _insight(
            [
                "Les pires erreurs sont presque toujours les biens **atypiques** (très grande surface, marché fin, état exceptionnel) : le modèle n'a pas assez de voisins pour les situer.",
                "Une erreur isolée n'est pas un bug ; un **groupe** d'erreurs du même signe sur le même segment en est un (ou une feature manquante).",
                "Réponse possible : publier une fourchette élargie et orienter ces biens vers une revue humaine — c'est exactement ce que fait `Predictor` avec le niveau de confiance.",
            ]
        ),
        _code(
            """
baseline_comparison = EVALUATOR.compare_to_baseline(PREPARED["X_test"], PREPARED["y_test"])
gains = pd.DataFrame(
    {
        "modèle": ["modèle entraîné", "baseline (médiane)"],
        CONFIG.metrics.primary: [
            RESULT.metrics.get(CONFIG.metrics.primary, float("nan")),
            baseline_comparison.get(f"baseline_{CONFIG.metrics.primary}", float("nan")),
        ],
    }
)
gains.round(2)
""",
            context,
        ),
        _insight(
            [
                "Un modèle qui ne bat pas la baseline (médiane) n'apporte **aucune** valeur : il ne doit pas aller en production.",
                "En régression, comparer aussi le **R²** : il exprime directement le gain sur la variance expliquée, ce qui parle davantage aux métiers qu'une RMSE en euros.",
                "La baseline est recalculée sur le **même** split de test : la comparaison est loyale.",
            ]
        ),
        _md("## 6. Rapport exécutable"),
        _code(
            """
from src.evaluation.reports import ReportBuilder

REPORTER = ReportBuilder(NB_PATHS, config=CONFIG.model_dump())
WRITTEN = REPORTER.build(RESULT, model=MODEL)
print(f"{len(WRITTEN)} artefacts écrits dans {NB_PATHS.artifacts_dir.relative_to(PROJECT_ROOT)}")
report_path = WRITTEN["report"]
Markdown(report_path.read_text(encoding="utf-8")[:2500] + "\\n\\n[…]")
""",
            context,
        ),
        _insight(
            [
                "Le rapport mélange **chiffres calculés** et **recommandations documentées** : il est régénérable à chaque run.",
                "Le même contenu est écrit en JSON (`artifacts/metrics/regression_report.json`) et en CSV (`error_by_segment.csv`, `top_errors.csv`) pour un dashboard ou une CI.",
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

Elles dépendent des métriques observées : couverture de fourchette à revoir, biais segmenté à
recalibrer, transformation de cible à corriger, segments à instrumenter.

### documentées pour ce cas d'usage

{recommendation_lines or "1. Rejouer l'évaluation sur un échantillon plus large avant toute décision."}

### plan d'action proposé

| Priorité | Action | Effet attendu | Comment vérifier |
| --- | --- | --- | --- |
| 1 | Recalibrer les segments biaisés (facteur multiplicatif par segment) | biais relatif < 2 % partout | §4 rejoué sur le test |
| 2 | Ajuster la largeur de fourchette au niveau de confiance | couverture ≥ 70 % | `coverage_by_bucket` |
| 3 | Ajouter les features exogènes manquantes (transactions voisines, tension du marché) | MAPE en baisse | notebook 04, comparaison d'algorithmes |
| 4 | Instrumenter la dérive du marché (prix médian observé vs estimé) | alerte précoce | `mlops/model-monitoring` |
| 5 | Rejouer ce notebook à chaque nouvelle version de données | non-régression | `make evaluate` + CI |

## 8. Limites assumées

- Les données sont **synthétiques** : les niveaux de performance illustrent une méthode, pas un marché réel.
- Une seule passe d'évaluation : la variance n'est pas mesurée ici (voir notebook 05, §4).
- L'analyse porte sur {spec.data.n_samples} lignes générées, dont une fraction en test : les segments rares restent peu observés.
- La fourchette publiée est un intervalle **heuristique** (± pourcentage ajusté par la confiance), pas un intervalle de prédiction statistique ; un modèle quantile (ou conforme) serait nécessaire pour un niveau de confiance garanti.

**Aller plus loin dans le dépôt** : comparaison multi-stacks (`data-science/regression/with-*`),
mise en production et suivi (`mlops/`), pipelines de données (`data-eng/`).
"""
        ),
    ]
    result_path = write_notebook(destination / "06_error_analysis.ipynb", cells)
    _replace_placeholder(result_path)
    return result_path
