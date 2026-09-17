"""Cellules de notebooks spécifiques à la tâche **ranking** (recommandation de catalogue).

Le squelette des six notebooks (``notebooks/tabular.py``) est partagé par toutes les tâches
tabulaires ; seules les cellules qui parlent de *cible*, de *probabilité de classe* ou d'*erreur de
prédiction* doivent changer. En classement, ce qui change est plus profond qu'un libellé : trois
ruptures de raisonnement traversent tout le notebook.

* **l'unité d'analyse n'est pas la ligne mais l'utilisateur.** Une précision calculée sur toutes
  les lignes mêle un client très actif à un nouveau venu et ne décrit aucun des deux. Chaque
  métrique de ce module est calculée par utilisateur puis moyennée, et une cellule est consacrée à
  montrer l'écart entre les deux lectures ;
* **un score absolu ne veut rien dire.** Le notebook part donc du plancher (tirage aléatoire),
  passe par la référence gratuite (tri par popularité) et se termine par le **plafond atteignable**
  publié par le générateur. La place du modèle dans cette fourchette est la seule lecture honnête ;
* **la pertinence n'est pas le seul objectif.** Couverture du catalogue, concentration, démarrage à
  froid et emplacements perdus en rupture sont mesurés au même titre que le NDCG, parce qu'un
  moteur peut améliorer son NDCG en se refermant sur dix best-sellers.

Ce module fournit :

* :func:`structure_cells` — la section 5 de ``01_eda.ipynb`` : la cible de pertinence, la structure
  par utilisateur, la relation popularité/pertinence qui fait la force de la référence gratuite, et
  les segments qui contraignent la publication ;
* :func:`build_04_model_exploration` — plancher, référence popularité, plafond oracle, comparaison
  des algorithmes à protocole identique, sensibilité à la coupure K, stabilité entre graines, sonde
  de fuite, puis grille d'hyperparamètres ;
* :func:`build_06_error_analysis` — évaluation complète par l'évaluateur de production, verdict sur
  les objectifs, ventilation par segment, démarrage à froid, santé du catalogue, arbitrage marge,
  backtest chronologique confronté à son plancher de bruit, pires mauvais classements, importance
  par permutation, hypothèses, recommandations et rapport.

Les helpers de rendu (``_code``, ``_md``, ``_insight``, jetons ``__XXX__``) sont réutilisés tels
quels : un notebook de recommandation et un notebook de classification partagent la même grammaire.
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

#: Nombre de lignes générées dans les notebooks de classement. Il faut assez d'utilisateurs pour
#: que le NDCG moyen ait un sens : à 30 candidats par utilisateur, 6 000 lignes donnent 200
#: utilisateurs, soit une erreur-type d'environ 0,02 sur le NDCG — assez pour distinguer deux
#: algorithmes, pas assez pour publier un écart de 0,005.
NB_ROWS_RANKING = 6000


# ---------------------------------------------------------------------------------------
# Helpers de métriques, injectés dans les notebooks
# ---------------------------------------------------------------------------------------
# Ces fonctions sont volontairement **réécrites dans le notebook** plutôt qu'importées depuis
# `src.training.losses_metrics` : un notebook pédagogique doit montrer le calcul, pas seulement son
# résultat. Le notebook 06, lui, utilise l'évaluateur de production — c'est exactement le point :
# comprendre le calcul ici, consommer l'objet industriel là.
RANKING_HELPERS = '''
import numpy as np
import pandas as pd


def discounts(size: int) -> np.ndarray:
    """Return the NDCG positional discounts 1 / log2(rank + 1).

    Le discount logarithmique est ce qui distingue le NDCG d'une simple précision : être pertinent
    en première position vaut plus qu'en dixième, parce que l'attention de l'utilisateur décroît
    avec la position. log2(rank + 1) est la forme standard, retenue parce qu'elle décroît vite au
    début de la liste et lentement ensuite — exactement le profil d'attention observé.
    """
    if size <= 0:
        return np.zeros(0)
    return 1.0 / np.log2(np.arange(2, size + 2, dtype="float64"))


def dcg(gains: np.ndarray) -> float:
    """Return the discounted cumulative gain of an ordered gain vector."""
    gains = np.asarray(gains, dtype="float64")
    return float((gains * discounts(gains.size)).sum()) if gains.size else 0.0


def ndcg_at_k(truth: np.ndarray, score: np.ndarray, k: int) -> float:
    """Return the NDCG@K of one list: realised gain over the best possible gain.

    Retourner 0 quand la liste ne contient rien de pertinent est un choix : un utilisateur sans
    intention ne peut pas être satisfait, et compter sa liste comme « parfaite » récompenserait un
    moteur qui publie peu. Ces utilisateurs sont donc exclus du calcul par l'appelant.
    """
    truth = np.asarray(truth, dtype="float64")
    order = np.argsort(-np.asarray(score, dtype="float64"), kind="stable")
    ideal = dcg(np.sort(truth)[::-1][:k])
    return dcg(truth[order][:k]) / ideal if ideal > 0 else 0.0


def precision_at_k(truth: np.ndarray, score: np.ndarray, k: int) -> float:
    """Return the share of the K published slots that are relevant."""
    order = np.argsort(-np.asarray(score, dtype="float64"), kind="stable")
    selected = np.asarray(truth, dtype="float64")[order][:k]
    return float(selected.sum() / selected.size) if selected.size else 0.0


def recall_at_k(truth: np.ndarray, score: np.ndarray, k: int) -> float:
    """Return the share of the available relevance captured by the K published slots."""
    truth = np.asarray(truth, dtype="float64")
    total = float(truth.sum())
    if total <= 0:
        return float("nan")
    order = np.argsort(-np.asarray(score, dtype="float64"), kind="stable")
    return float(truth[order][:k].sum() / total)


def average_precision_at_k(truth: np.ndarray, score: np.ndarray, k: int) -> float:
    """Return AP@K: the mean of the precisions measured at each relevant hit."""
    order = np.argsort(-np.asarray(score, dtype="float64"), kind="stable")
    selected = np.asarray(truth, dtype="float64")[order][:k]
    if selected.sum() <= 0:
        return 0.0
    precisions = np.cumsum(selected) / np.arange(1, selected.size + 1, dtype="float64")
    return float((precisions * selected).sum() / min(selected.sum(), k))


def hit_rate_at_k(truth: np.ndarray, score: np.ndarray, k: int) -> float:
    """Return 1 when at least one relevant item lands in the K published slots."""
    order = np.argsort(-np.asarray(score, dtype="float64"), kind="stable")
    return 1.0 if float(np.asarray(truth, dtype="float64")[order][:k].sum()) > 0 else 0.0


def ranking_metrics(frame: pd.DataFrame, score_column: str, k: int = 10) -> pd.Series:
    """Compute every ranking metric **per user**, then average.

    C'est la fonction la plus importante du notebook : elle matérialise le fait que l'unité
    d'évaluation est l'utilisateur. Les utilisateurs sans aucun candidat pertinent sont exclus,
    parce qu'aucun classement ne peut les satisfaire.
    """
    rows = {"ndcg": [], "precision": [], "recall": [], "map": [], "hit": []}
    for _, group in frame.groupby("user_id", observed=True, sort=False):
        truth = group[TARGET].to_numpy(dtype="float64")
        if truth.sum() <= 0:
            continue
        score = group[score_column].to_numpy(dtype="float64")
        rows["ndcg"].append(ndcg_at_k(truth, score, k))
        rows["precision"].append(precision_at_k(truth, score, k))
        rows["recall"].append(recall_at_k(truth, score, k))
        rows["map"].append(average_precision_at_k(truth, score, k))
        rows["hit"].append(hit_rate_at_k(truth, score, k))
    if not rows["ndcg"]:
        return pd.Series(dtype="float64")
    values = pd.Series({key: float(np.nanmean(item)) for key, item in rows.items()})
    values["utilisateurs"] = float(len(rows["ndcg"]))
    return values.round(4)


def score_frame(PREPARED: dict[str, Any], split: str, model: Any) -> pd.DataFrame:
    """Return the enriched split rows with the model score, ready for per-user metrics.

    Le cadre enrichi (avant pré-traitement) porte les identifiants et les colonnes métier ; la
    matrice pré-traitée porte ce que le modèle consomme. Les deux sont alignés ligne à ligne, ce
    qui permet de scorer sur l'une et d'analyser sur l'autre.
    """
    frame = PREPARED["enriched"][split].reset_index(drop=True)
    matrix = PREPARED[f"X_{split}"]
    probabilities = model.predict_proba(matrix)
    scored = frame.copy()
    scored["score"] = probabilities[:, -1] if probabilities.ndim == 2 else probabilities
    return scored
'''


# ---------------------------------------------------------------------------------------
# 01 — La cible de pertinence et la structure du jeu de candidats
# ---------------------------------------------------------------------------------------
_TARGET_STRUCTURE_CELL = '''
# Aucun nom de colonne n'est écrit dans ce notebook : l'évaluateur du projet résout ses réglages
# depuis `conf/config.yaml` (bloc `recommendation`, avec repli sur le noeud `data`). L'EDA décrit
# donc exactement les colonnes que le rapport de production utilisera — renommer un champ dans la
# configuration suffit, aucune édition de notebook n'est nécessaire.
from src.evaluation.evaluator import RankingSettings

SETTINGS = RankingSettings.resolve(CONFIG.model_dump(mode="json"))

TARGET = CONFIG.data.target
GROUP = SETTINGS.group_column
ARTICLE = SETTINGS.item_column
PERIODE = SETTINGS.time_column or CONFIG.data.time_column
print(f"cible                : {TARGET}")
print(f"unité de publication : {GROUP}")
print(f"article              : {ARTICLE}")
print(f"popularité           : {SETTINGS.popularity_column}")
print(f"intention récente    : {SETTINGS.intent_column}")
print(f"disponibilité        : {SETTINGS.stock_column}")
print(f"activité             : {SETTINGS.activity_column} "
      f"(froid <= {SETTINGS.cold_user_max_orders})")
print(f"marge                : {SETTINGS.margin_column}")
print(f"coupure publiée      : top-{SETTINGS.top_k}")
print()

prevalence = float(raw[TARGET].mean())
per_user = raw.groupby(GROUP, observed=True)[TARGET].agg(["size", "sum", "mean"])

print(f"couples (utilisateur, candidat) : {len(raw):,}")
print(f"utilisateurs distincts          : {per_user.shape[0]:,}")
print(f"candidats par utilisateur       : {per_user['size'].min()} à {per_user['size'].max()}"
      f" (médiane {per_user['size'].median():.0f})")
print(f"pertinence globale              : {prevalence:.4f}")
print()
print("Répartition du nombre de candidats pertinents par utilisateur :")
distribution = per_user["sum"].value_counts().sort_index()
display(
    pd.DataFrame(
        {
            "utilisateurs": distribution,
            "part": (distribution / len(per_user)).round(4),
        }
    )
)

sans_intention = int((per_user["sum"] == 0).sum())
print(f"utilisateurs sans AUCUN candidat pertinent : {sans_intention}"
      f" ({sans_intention / len(per_user):.1%})")
print("Ces utilisateurs sont exclus de toutes les métriques de classement : aucun ordre")
print("ne peut les satisfaire, et les compter comme « parfaits » récompenserait un moteur")
print("qui publie peu.")
'''

_POPULARITY_LINK_CELL = '''
# La question qui décide de tout : la popularité porte-t-elle un signal de pertinence ?
# Si non, le tri par popularité est une référence ridicule et la comparaison n'apprend rien.
# Si oui — et c'est le cas réel — c'est une référence forte qu'un modèle doit battre nettement.
audience = SETTINGS.popularity_column

# Les colonnes « témoin » du tableau ne sont pas écrites à la main : on prend les trois champs
# numériques les plus corrélés à la cible (hors audience et cible). Le choix est donc dicté par le
# jeu de données, et le tableau reste pertinent si le schéma change.
numeriques = raw.select_dtypes("number").drop(
    columns=[column for column in (TARGET, audience) if column in raw.columns], errors="ignore"
)
# `TARGET` vient d'être écartée de `numeriques` : la corrélation se calcule donc contre la
# colonne brute, pas contre la matrice de corrélation interne.
correlations = (
    numeriques.corrwith(raw[TARGET].astype("float64")).abs().sort_values(ascending=False)
)
temoins = list(correlations.head(3).index)
print(f"colonnes témoin retenues (|corrélation| à la cible) : {temoins}")
print()

agregats = {
    f"{audience} médian": (audience, "median"),
    "pertinence": (TARGET, "mean"),
    "effectif": (TARGET, "size"),
}
for colonne in temoins:
    agregats[f"{colonne} moyen"] = (colonne, "mean")

lien = (
    raw.groupby(pd.qcut(raw[audience], 10, labels=False, duplicates="drop"), observed=True)
    .agg(**agregats)
    .round(4)
    .rename_axis("décile d'audience")
)
display(lien)

correlation = float(raw[[audience, TARGET]].corr(numeric_only=True).iloc[0, 1])
print(f"corrélation de Pearson {audience} / {TARGET} : {correlation:+.4f}")
print()
print("Lecture : la pertinence croît avec l'audience, mais elle ne s'y réduit pas —")
print("le dixième décile n'est pas pertinent à 100 %. C'est exactement l'espace dans")
print("lequel un modèle doit travailler : capter la part d'audience qui est du signal,")
print("et ajouter l'affinité individuelle que l'audience ne porte pas.")
'''

_CONSTRAINTS_CELL = '''
# Ce qui contraint la publication, indépendamment de la pertinence. Chaque indicateur est calculé
# sur une colonne résolue depuis la configuration ; une colonne absente du schéma est écartée au
# lieu de faire échouer le notebook.
def _part(colonne: str, masque) -> tuple[str, float]:
    """Return the indicator name and its share, or NaN when the column is unavailable."""
    if colonne not in raw.columns:
        return colonne, float("nan")
    return colonne, float(masque(raw[colonne]).mean())


lignes: list[dict[str, object]] = []

colonne, valeur = _part(SETTINGS.stock_column, lambda serie: serie <= 0)
lignes.append(
    {
        "indicateur": f"candidats en rupture ({colonne} <= 0)",
        "part": valeur,
        "conséquence sur la publication": "un article indisponible ne doit jamais occuper un "
        "emplacement : le filtre s'applique après le classement",
    }
)

colonne, valeur = _part(SETTINGS.intent_column, lambda serie: serie <= 0)
lignes.append(
    {
        "indicateur": f"candidats sans intention récente ({colonne} <= 0)",
        "part": valeur,
        "conséquence sur la publication": "la référence d'exploitation pure ne peut rien publier "
        "ici : le moteur explore plus qu'il n'exploite",
    }
)

if SETTINGS.activity_column in raw.columns:
    activite = raw.groupby(GROUP, observed=True)[SETTINGS.activity_column].first()
    part_froid = float((activite <= SETTINGS.cold_user_max_orders).mean())
else:
    part_froid = float("nan")
lignes.append(
    {
        "indicateur": f"utilisateurs en démarrage froid ({SETTINGS.activity_column} <= "
        f"{SETTINGS.cold_user_max_orders})",
        "part": part_froid,
        "conséquence sur la publication": "aucun historique exploitable : le service rendu ne peut "
        "reposer que sur la fiche article et le contexte",
    }
)

if SETTINGS.margin_column in raw.columns:
    marge = raw[SETTINGS.margin_column]
    part_marge = float((marge <= marge.median()).mean())
else:
    part_marge = float("nan")
lignes.append(
    {
        "indicateur": f"candidats sous la médiane de marge ({SETTINGS.margin_column})",
        "part": part_marge,
        "conséquence sur la publication": "pertinence et valeur ne coïncident pas : l'arbitrage se "
        "règle dans le score de publication, pas dans le modèle",
    }
)

sans_intention_part = float((per_user["sum"] == 0).mean())
lignes.append(
    {
        "indicateur": f"utilisateurs sans aucun candidat pertinent ({TARGET} toujours 0)",
        "part": sans_intention_part,
        "conséquence sur la publication": "exclus des métriques de classement : aucun ordre "
        "ne peut les satisfaire",
    }
)

contraintes = pd.DataFrame(lignes).set_index("indicateur")
contraintes["part"] = contraintes["part"].astype("float64").round(4)
display(contraintes)

print()
print(f"références du catalogue : {raw[ARTICLE].nunique():,}")
if PERIODE and PERIODE in raw.columns:
    periode = pd.to_datetime(raw[PERIODE])
    print(f"fenêtre observée        : {periode.min().date()} → {periode.max().date()}")
    print(f"nombre de périodes      : {periode.dt.to_period('D').nunique()} jours distincts")

# Cardinalité des champs catégoriels : ce sont les leviers de segmentation du rapport d'erreur.
categorielles = raw.select_dtypes(include=["object", "string", "category", "bool"]).columns
if len(categorielles):
    cardinalites = pd.DataFrame(
        {
            "colonne": list(categorielles),
            "modalités": [int(raw[colonne].nunique()) for colonne in categorielles],
            "part de la modalité majoritaire": [
                round(float(raw[colonne].astype(str).value_counts(normalize=True).iloc[0]), 4)
                for colonne in categorielles
            ],
        }
    ).sort_values("modalités", ascending=False)
    print()
    display(cardinalites)
    print("Une modalité majoritaire à plus de 90 % porte peu d'information ; une cardinalité")
    print("proche du nombre de lignes est un identifiant déguisé, à écarter des features.")
'''


def structure_cells(context: NotebookContext) -> list[NotebookNode]:
    """Build the target section of ``01_eda.ipynb`` for a ranking project.

    Une cible de classement ne se décrit pas comme une cible de classification. Ce qui compte ici
    n'est pas sa distribution marginale — une prévalence de 14 % se résume en une ligne — mais la
    **structure du jeu de candidats** : combien de candidats par utilisateur, combien d'utilisateurs
    sans aucune intention, et surtout si la popularité porte déjà le signal (ce qui détermine la
    difficulté réelle du problème et la valeur de la référence gratuite).

    Args:
        context: Notebook context.

    Returns:
        The cells of section 5.
    """
    spec = context.spec
    return [
        _md(
            f"""## 5. La cible de classement et la structure du jeu de candidats

La cible `{spec.data.target}` vaut 1 quand le couple (utilisateur, article candidat) correspond à
une intention réelle. Mais décrire sa distribution ne dit rien d'utile : ce qui gouverne la
difficulté du problème, c'est la **structure** du jeu — combien de candidats par utilisateur,
combien d'utilisateurs sans aucune intention observable, et si la popularité des articles porte déjà
le signal de pertinence.

Cette dernière question est décisive. Si la popularité ne disait rien de la pertinence, le tri par
popularité — le moteur gratuit que le site utilise aujourd'hui — serait une référence ridicule, et
n'importe quel modèle la battrait sans mérite. Si elle en dit beaucoup, la référence est forte et
le gain du modèle doit se mesurer contre elle."""
        ),
        _code(_TARGET_STRUCTURE_CELL, context),
        _insight(
            [
                "Chaque utilisateur est scoré sur le **même nombre de candidats** : c'est une exigence de comparabilité, sans laquelle Precision@K et Recall@K ne mesureraient pas la même chose d'un utilisateur à l'autre.",
                "Une part notable d'utilisateurs n'a aucun candidat pertinent. Ils sont exclus des métriques de classement : les inclure reviendrait à récompenser un moteur qui publie peu, puisque publier rien ne peut pas être faux.",
                "La prévalence globale borne le plancher : un moteur qui tire au hasard obtient une précision égale à cette prévalence. C'est la première valeur à connaître avant de commenter un chiffre.",
            ]
        ),
        _md(
            """### 5.1 La popularité porte-t-elle le signal ?

C'est la question qui fixe la difficulté du problème. On regarde la pertinence moyenne par décile
d'audience : si elle croît fortement, le tri par popularité est une référence solide ; si elle est
plate, la référence est vide."""
        ),
        _code(_POPULARITY_LINK_CELL, context),
        _insight(
            [
                "La pertinence **croît** avec l'audience : les articles beaucoup vus sont beaucoup vus parce qu'ils plaisent largement. Le tri par popularité n'est donc pas une référence aléatoire, c'est un vrai moteur — gratuit, explicable et robuste.",
                "Mais elle ne s'y **réduit** pas : le dernier décile n'est pas pertinent à 100 %, et les déciles intermédiaires se chevauchent. C'est dans cet écart que travaille un modèle, en ajoutant l'affinité individuelle que l'audience agrégée ne peut pas porter.",
                "Cette relation explique aussi le risque de boucle de rétroaction : exposer ce qui est déjà populaire augmente son audience, qui augmente son score, qui augmente son exposition. La mesurer ici, c'est se donner la référence contre laquelle on surveillera la dérive en production.",
            ]
        ),
        _md(
            """### 5.2 Ce qui contraint la publication

La pertinence n'est pas la seule contrainte. Un moteur qui publierait uniquement ce qui est
pertinent montrerait des articles en rupture de stock, dix articles de la même catégorie, ou
laisserait les nouveaux utilisateurs sans recommandation exploitable. Ces contraintes se mesurent
avant de modéliser, parce qu'elles déterminent ce que la publication devra filtrer."""
        ),
        _code(_CONSTRAINTS_CELL, context),
        _insight(
            [
                "La majorité des candidats n'a jamais été consultée par l'utilisateur : le moteur est structurellement en **exploration**, pas en exploitation. Un modèle qui ne sait scorer que ce qui a déjà été vu est donc inutile sur la plus grande part du catalogue.",
                "Les articles en rupture ne doivent jamais occuper un emplacement. Le filtrer dans le modèle est fragile — le stock change toutes les heures, le modèle toutes les semaines — donc le filtre est appliqué après le classement, dans la couche de publication.",
                "La part d'utilisateurs froids est un **objectif** à part entière, pas une note de bas de page : un moteur excellent pour les clients fidèles et médiocre pour les nouveaux venus est un problème d'acquisition déguisé en problème de précision.",
            ]
        ),
    ]


# ---------------------------------------------------------------------------------------
# 04 — Plancher, référence, plafond, algorithmes, coupure K, stabilité, fuite, grille
# ---------------------------------------------------------------------------------------
_FIT_CELL = """
from src.models import build_model

MODEL = build_model(CONFIG, feature_names=PREPARED["feature_names"])
# `build_model(params=...)` **remplace** les réglages de `conf/model/default.yaml`. Pour ne faire
# varier qu'un seul facteur à la fois dans les sections suivantes (algorithme, coupure, graine,
# grille), on part donc toujours de cette copie des réglages configurés.
CONFIGURED_PARAMS = dict(CONFIG.model.params)
FIT_RESULT = MODEL.fit(
    PREPARED["X_train"],
    PREPARED["y_train"],
    X_val=PREPARED["X_val"],
    y_val=PREPARED["y_val"],
    callbacks=[],
)
print(MODEL.summary())
print(f"entraînement : {FIT_RESULT.duration_seconds:.2f} s")
"""

_SCORE_HELPERS = '''
# Aucun nom de colonne n'est écrit dans ce notebook : l'évaluateur du projet résout ses réglages
# depuis `conf/config.yaml` (bloc `recommendation`, avec repli sur le noeud `data`). Le notebook
# utilise exactement les mêmes colonnes que le rapport de production — changer le schéma ne
# demande qu'un changement de configuration, pas une édition de notebook.
from src.evaluation.evaluator import RankingSettings

CONFIG_DICT = CONFIG.model_dump(mode="json")
SETTINGS = RankingSettings.resolve(CONFIG_DICT)


def rank_scores(model: Any, matrix: pd.DataFrame) -> np.ndarray:
    """Return a continuous ranking score, whatever the estimator exposes.

    Un score de classement n'a pas besoin d'être une probabilité : il lui faut seulement être
    **ordonnable**. Un SVM sans calibration probabiliste classe très bien par sa distance à la
    frontière, un arbre par sa probabilité, un modèle de factorisation par son produit scalaire.
    Cette fonction prend ce que l'estimateur sait produire, dans l'ordre de préférence.
    """
    try:
        probabilities = np.asarray(model.predict_proba(matrix), dtype="float64")
        return probabilities[:, -1].ravel() if probabilities.ndim == 2 else probabilities.ravel()
    except Exception:
        estimator = getattr(model, "estimator_", None) or getattr(model, "model_", None)
        for attribute in ("decision_function", "score_samples", "predict"):
            function = getattr(estimator, attribute, None)
            if callable(function):
                values = np.asarray(function(matrix), dtype="float64")
                return values[:, -1].ravel() if values.ndim == 2 else values.ravel()
        raise


VAL = PREPARED["enriched"]["val"].reset_index(drop=True).copy()
VAL["score"] = rank_scores(MODEL, PREPARED["X_val"])
TARGET = CONFIG.data.target
GROUPE = SETTINGS.group_column
ARTICLE = SETTINGS.item_column
TOP_K = int(SETTINGS.top_k)
print(f"split de validation : {len(VAL):,} candidats, {VAL[GROUPE].nunique()} utilisateurs")
print(f"coupure publiée     : top-{TOP_K}")
print(f"colonnes résolues   : groupe={GROUPE} article={ARTICLE} "
      f"popularité={SETTINGS.popularity_column} intention={SETTINGS.intent_column}")
'''

_BASELINES_CELL = '''
# Quatre moteurs, un seul protocole : mêmes candidats, mêmes utilisateurs, même coupure.
rng = np.random.default_rng(CONFIG.seed)
VAL["score_aleatoire"] = rng.random(len(VAL))
VAL["score_popularite"] = VAL[SETTINGS.popularity_column].astype("float64")
VAL["score_intention"] = VAL[SETTINGS.intent_column].astype("float64")

references = {
    "tirage aléatoire": "score_aleatoire",
    "tri par popularité": "score_popularite",
    "intention récente": "score_intention",
    f"modèle configuré ({MODEL.algorithm})": "score",
}
table = pd.DataFrame(
    {name: ranking_metrics(VAL, column, TOP_K) for name, column in references.items()}
).T
table["couverture catalogue"] = [
    round(
        VAL.assign(
            rang=VAL.groupby(GROUPE, observed=True)[column].rank(ascending=False, method="first")
        )
        .query("rang <= @TOP_K")[ARTICLE]
        .nunique()
        / VAL[ARTICLE].nunique(),
        4,
    )
    for column in references.values()
]
display(table)

plancher = float(table.loc["tirage aléatoire", "ndcg"])
reference = float(table.loc["tri par popularité", "ndcg"])
modele = float(table.loc[f"modèle configuré ({MODEL.algorithm})", "ndcg"])
print(f"plancher (aléatoire)      : {plancher:.4f}")
print(f"référence (popularité)    : {reference:.4f}")
print(f"modèle                    : {modele:.4f}")
print(f"gain sur la popularité    : {(modele - reference) / reference:+.1%}")
'''

_CEILING_CELL = '''
# Le générateur connaît la probabilité de pertinence **avant** bruit : il peut donc classer
# parfaitement tout ce qui est connaissable. Ce classement oracle définit le plafond atteignable,
# et la distance qui reste n'est pas une marge de progression — c'est du bruit irréductible.
import json

# Le plafond n'est publié que par le générateur synthétique : sur données réelles il n'existe pas.
# La variable est donc initialisée à NaN **avant** la branche conditionnelle, pour que les cellules
# suivantes puissent la tester sans dépendre de l'existence du fichier.
plafond = float("nan")

metadata_path = PATHS.data_file("generation_metadata", fmt="json", stage="raw")
if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    plafond = float(metadata.get("ndcg_ceiling_oracle") or float("nan"))
    publie = {
        "aléatoire": metadata.get("ndcg_baseline_random"),
        "popularité": metadata.get("ndcg_baseline_popularity"),
        "plafond oracle": plafond,
        "marge totale": metadata.get("ndcg_headroom"),
        "coupure de référence": metadata.get("top_k_reference"),
    }
    display(pd.Series(publie, name=f"NDCG@{TOP_K} publié par le générateur").to_frame())

    chemin = plafond - plancher
    progression = (modele - plancher) / chemin if chemin > 0 else float("nan")
    print(f"le modèle a parcouru {progression:.1%} du chemin entre l'aléatoire et le plafond")
    print(f"part du plafond capturée : {modele / plafond:.1%}")
else:
    print("Aucune métadonnée de génération : le plafond n'est pas publié.")
    print("Sur données réelles c'est le cas normal — on ne compare alors qu'aux références.")
'''

_PER_USER_CELL = '''
# La même donnée, deux lectures. C'est la démonstration la plus importante du notebook : la
# métrique « globale » n'est pas une version bruitée de la métrique par utilisateur, c'est une
# AUTRE question, qui mélange des utilisateurs de volumes très différents.
par_utilisateur = ranking_metrics(VAL, "score", TOP_K)

utilisateurs = VAL[GROUPE].nunique()
ordre = np.argsort(-VAL["score"].to_numpy(), kind="stable")
verite = VAL[TARGET].to_numpy(dtype="float64")
global_precision = float(verite[ordre][: TOP_K * utilisateurs].mean())
global_ndcg = ndcg_at_k(verite, VAL["score"].to_numpy(), TOP_K * utilisateurs)

volumes = VAL.groupby(GROUPE, observed=True).size()
print(f"lecture GLOBALE (toutes lignes confondues)")
print(f"  précision sur les {TOP_K}x{volumes.size} premiers rangs : {global_precision:.4f}")
print(f"  NDCG sur une liste unique de {TOP_K * volumes.size} places  : {global_ndcg:.4f}")
print()
print(f"lecture PAR UTILISATEUR (moyenne de {int(par_utilisateur['utilisateurs'])} listes)")
print(f"  précision@{TOP_K} : {par_utilisateur['precision']:.4f}")
print(f"  NDCG@{TOP_K}      : {par_utilisateur['ndcg']:.4f}")
print()
print("Écart de précision : "
      f"{par_utilisateur['precision'] - global_precision:+.4f}")
print()
print("Pourquoi la lecture par utilisateur est la seule valide ici :")
display(
    pd.DataFrame(
        {
            "utilisateurs": volumes.value_counts().sort_index(),
        }
    ).rename_axis("candidats par utilisateur").reset_index()
)
print("Un utilisateur très actif contribue à des dizaines de lignes, un nouveau venu à une")
print("seule : la moyenne globale est donc pondérée par l'activité, et décrit le comportement")
print("du moteur sur les gros clients — pas le service rendu à un utilisateur moyen.")
'''

_CUTOFF_CELL = '''
# Publier 5 ou 20 articles ne mesure pas la même qualité. Cette table est l'outil de décision
# du K : elle montre le compromis structurel (la précision chute, le rappel monte) et le prix en
# diversité (la couverture catalogue progresse, la concentration recule).
lignes = []
for coupure in (3, 5, TOP_K, 15, 20):
    valeurs = ranking_metrics(VAL, "score", coupure)
    published = VAL.assign(
        rang=VAL.groupby(GROUPE, observed=True)["score"].rank(ascending=False, method="first")
    ).query("rang <= @coupure")
    counts = published[ARTICLE].value_counts()
    shares = counts.to_numpy(dtype="float64") / float(counts.sum())
    lignes.append(
        {
            "K": coupure,
            "NDCG": valeurs["ndcg"],
            "précision": valeurs["precision"],
            "rappel": valeurs["recall"],
            "MAP": valeurs["map"],
            "hit-rate": valeurs["hit"],
            "couverture catalogue": round(counts.size / VAL[ARTICLE].nunique(), 4),
            "Herfindahl": round(float((shares**2).sum()), 4),
            "retenu": coupure == TOP_K,
        }
    )
coupures = pd.DataFrame(lignes).set_index("K")
display(coupures)
print(f"K retenu par la configuration : {TOP_K}")
'''

_ALGORITHMS_CELL = '''
import warnings

from src.models.factory import available_algorithms

ALGORITHMS = available_algorithms(CONFIG.metrics.task)
print(f"{len(ALGORITHMS)} algorithmes servent la tâche '{CONFIG.metrics.task}' :")
print(ALGORITHMS)

lignes = []
avertissements: dict[str, dict[str, int]] = {}
# Comparaison **loyale** : `params={}` construit chaque algorithme avec ses réglages par défaut.
# Les `model.params` configurés sont propres au boosting par histogrammes (`max_leaf_nodes`,
# `max_features`, `learning_rate`) : les injecter dans une régression logistique lèverait une
# erreur, et les régler à la main pour chaque concurrent fausserait le classement. Le modèle
# configuré ET réglé est, lui, mesuré en section 1.
for algorithm in ALGORITHMS:
    try:
        # Les avertissements des bibliothèques sont capturés puis restitués en fin de cellule :
        # les masquer cacherait une information diagnostique (convergence, classes rares), les
        # laisser inonder la sortie noierait le tableau. Ni l'un ni l'autre.
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            candidate = build_model(
                CONFIG, feature_names=PREPARED["feature_names"], algorithm=algorithm, params={}
            )
            result = candidate.fit(
                PREPARED["X_train"],
                PREPARED["y_train"],
                X_val=PREPARED["X_val"],
                y_val=PREPARED["y_val"],
                callbacks=[],
            )
            scores = rank_scores(candidate, PREPARED["X_val"])
        comptes: dict[str, int] = {}
        for item in captured:
            premiere = str(item.message).strip().splitlines()[0][:78]
            cle = f"{item.category.__name__} : {premiere}"
            comptes[cle] = comptes.get(cle, 0) + 1
        if comptes:
            avertissements[algorithm] = comptes
        mesure = ranking_metrics(VAL.assign(score_candidat=scores), "score_candidat", TOP_K)
        published = (
            VAL.assign(
                rang=pd.Series(scores, index=VAL.index)
                .groupby(VAL[GROUPE])
                .rank(ascending=False, method="first")
            )
            .query("rang <= @TOP_K")[ARTICLE]
            .nunique()
        )
        lignes.append(
            {
                "algorithme": algorithm,
                f"NDCG@{TOP_K}": mesure["ndcg"],
                f"précision@{TOP_K}": mesure["precision"],
                f"rappel@{TOP_K}": mesure["recall"],
                "couverture": round(published / VAL[ARTICLE].nunique(), 4),
                "secondes": round(result.duration_seconds, 2),
            }
        )
    except Exception as error:  # un algorithme incompatible ne doit pas casser l'exploration
        lignes.append(
            {
                "algorithme": algorithm,
                f"NDCG@{TOP_K}": float("nan"),
                f"précision@{TOP_K}": float("nan"),
                f"rappel@{TOP_K}": float("nan"),
                "couverture": float("nan"),
                "secondes": float("nan"),
            }
        )
        avertissements[algorithm] = {f"échec : {type(error).__name__} : {error}"[:120]: 1}

comparaison = pd.DataFrame(lignes).set_index("algorithme").sort_values(
    f"NDCG@{TOP_K}", ascending=False
)
display(comparaison)
print(f"référence à battre (popularité) : {reference:.4f}")
print(f"plafond atteignable             : {plafond:.4f}" if not np.isnan(plafond) else "")

if avertissements:
    print("\\n--- avertissements et échecs capturés ---")
    for algorithm, comptes in avertissements.items():
        for message, nombre in comptes.items():
            print(f"  {algorithm:<24} x{nombre}  {message}")
'''

_SEEDS_CELL = '''
# Un écart entre deux algorithmes inférieur à la dispersion entre graines n'est pas un gain :
# c'est un tirage. Trois graines suffisent à estimer cette dispersion sur un volume de notebook.
lignes = []
for graine in (CONFIG.seed, CONFIG.seed + 1, CONFIG.seed + 2):
    candidate = build_model(
        CONFIG,
        feature_names=PREPARED["feature_names"],
        params={**CONFIGURED_PARAMS, "random_state": graine},
    )
    candidate.fit(
        PREPARED["X_train"],
        PREPARED["y_train"],
        X_val=PREPARED["X_val"],
        y_val=PREPARED["y_val"],
        callbacks=[],
    )
    scores = rank_scores(candidate, PREPARED["X_val"])
    mesure = ranking_metrics(VAL.assign(score_graine=scores), "score_graine", TOP_K)
    lignes.append(
        {
            "graine": graine,
            f"NDCG@{TOP_K}": mesure["ndcg"],
            f"précision@{TOP_K}": mesure["precision"],
            f"rappel@{TOP_K}": mesure["recall"],
        }
    )
graines = pd.DataFrame(lignes).set_index("graine")
display(graines)

ecart_type = float(graines[f"NDCG@{TOP_K}"].std(ddof=1))
print(f"NDCG moyen {graines[f'NDCG@{TOP_K}'].mean():.4f} | écart-type entre graines {ecart_type:.4f}")
print(f"amplitude  {graines[f'NDCG@{TOP_K}'].max() - graines[f'NDCG@{TOP_K}'].min():.4f}")
print()
print("Tout écart entre deux algorithmes inférieur à cette dispersion ne justifie pas un")
print("changement de modèle : il justifie un travail sur les features.")
'''

_LEAK_CELL = '''
# Sonde de fuite : aucune colonne observable ne doit approcher le plafond. Si une seule y
# parvenait, elle porterait l'information que le générateur garde pour lui — et le modèle
# « appris » ne serait qu'un proxy de la réponse.
colonnes = [
    column
    for column in VAL.columns
    if column not in {TARGET, "score", "score_aleatoire", "score_popularite", "score_intention"}
    and pd.api.types.is_numeric_dtype(VAL[column])
]
lignes = []
for column in colonnes:
    valeurs = VAL[column].astype("float64")
    if valeurs.nunique() < 2:
        continue
    mesure = ranking_metrics(VAL.assign(sonde=valeurs), "sonde", TOP_K)
    lignes.append({"colonne": column, f"NDCG@{TOP_K}": mesure["ndcg"]})
sondes = (
    pd.DataFrame(lignes)
    .set_index("colonne")
    .sort_values(f"NDCG@{TOP_K}", ascending=False)
    .head(12)
)
display(sondes)

meilleure = float(sondes[f"NDCG@{TOP_K}"].iloc[0])
print(f"meilleure colonne seule  : {meilleure:.4f}")
print(f"référence popularité     : {reference:.4f}")
print(f"modèle complet           : {modele:.4f}")
print(f"plafond oracle           : {plafond:.4f}" if not np.isnan(plafond) else "")
print()
if not np.isnan(plafond):
    print(f"la meilleure colonne atteint {meilleure / plafond:.1%} du plafond : "
          "aucune fuite univariée.")
    print("Le modèle complet dépasse nettement chaque colonne prise isolément : il combine")
    print("l'information, il ne la recopie pas.")
'''

_GRID_CELL = """
# Grille explorée par ce notebook : elle est **déclarée dans le manifeste du projet**
# (`extras.notebook_param_grid`), donc versionnée avec lui, centrée sur le point retenu. Le
# notebook rejoue ainsi la décision d'hyperparamètres au lieu de la paraphraser.
GRID: dict[str, list[Any]] = __PARAM_GRID__

if not GRID:
    print("Aucune grille déclarée pour ce projet : la recherche d'hyperparamètres n'est pas")
    print("rejouée dans le notebook, le point configuré est celui de conf/model/default.yaml.")
else:
    import itertools

    from src.models import build_model

    combinaisons = list(itertools.product(*[GRID[key] for key in GRID]))
    print(f"{len(combinaisons)} combinaisons explorées : {GRID}")

    lignes = []
    for combinaison in combinaisons:
        params = {**CONFIGURED_PARAMS, **dict(zip(GRID.keys(), combinaison, strict=True))}
        candidate = build_model(CONFIG, feature_names=PREPARED["feature_names"], params=params)
        candidate.fit(
            PREPARED["X_train"],
            PREPARED["y_train"],
            X_val=PREPARED["X_val"],
            y_val=PREPARED["y_val"],
            callbacks=[],
        )
        scores = rank_scores(candidate, PREPARED["X_val"])
        mesure = ranking_metrics(VAL.assign(score_grille=scores), "score_grille", TOP_K)
        lignes.append(
            {
                **params,
                f"NDCG@{TOP_K}": mesure["ndcg"],
                f"précision@{TOP_K}": mesure["precision"],
                f"rappel@{TOP_K}": mesure["recall"],
            }
        )

    colonnes_grille = list(GRID.keys())
    grille = pd.DataFrame(lignes)
    display(
        grille[[*colonnes_grille, f"NDCG@{TOP_K}", f"précision@{TOP_K}", f"rappel@{TOP_K}"]]
        .sort_values(f"NDCG@{TOP_K}", ascending=False)
        .reset_index(drop=True)
    )
    meilleur = grille.loc[grille[f"NDCG@{TOP_K}"].idxmax(), colonnes_grille].to_dict()
    configure = {key: CONFIGURED_PARAMS.get(key) for key in colonnes_grille}
    print(f"meilleur point de la grille : {meilleur}")
    print(f"point configuré             : {configure}")
    ecart = float(grille[f"NDCG@{TOP_K}"].max() - grille[f"NDCG@{TOP_K}"].min())
    print(f"amplitude NDCG sur toute la grille : {ecart:.4f}")

    # La dispersion entre graines mesurée en section 7 borne ce qu'on peut conclure. Elle est
    # lue depuis l'espace global pour que la cellule reste exécutable hors ordre.
    bruit = globals().get("ecart_type")
    if meilleur == configure:
        print("Le point configuré est bien celui que la grille préfère : la décision rejoue.")
    elif bruit is not None and ecart <= float(bruit):
        print(f"AUCUN point de la grille ne se détache : l'amplitude totale ({ecart:.4f}) reste")
        print(f"sous la dispersion entre graines ({float(bruit):.4f}). Le point configuré est donc")
        print("défendable, et chercher un meilleur réglage sur ce volume serait ajuster du bruit.")
    else:
        print("Le point configuré n'est pas le meilleur de cette grille, et l'écart dépasse la")
        print("dispersion entre graines : à trancher en connaissance de cause (coût, variance,")
        print("couverture), pas sur le seul NDCG.")
"""


# ---------------------------------------------------------------------------
# 06 — Analyse d'erreur : verdict, segments, froid, couverture, bruit, causes
# ---------------------------------------------------------------------------
_EVALUATE_CELL = '''
from src.evaluation.evaluator import Evaluator

# L'évaluateur du projet fait tout le travail de mesure : il calcule les métriques par
# utilisateur, les références, la couverture, le backtest chronologique, les pires classements et
# l'importance des features. Le notebook ne recalcule rien — il **interprète** et prolonge là où
# le rapport écrit ne peut pas aller (lire des lignes, tester une hypothèse, chiffrer un arbitrage).
# `Evaluator.from_config` attend un **mapping** (comme le pipeline `scripts/evaluate.py`, qui
# passe le dictionnaire Hydra). `CONFIG` est le modèle pydantic validé : on le sérialise, ce qui
# inclut le bloc `recommendation` déclaré dans `conf/config.yaml`.
evaluator = Evaluator.from_config(MODEL, CONFIG.model_dump(mode="json"), PATHS)
RESULT = evaluator.evaluate(
    PREPARED["X_test"],
    PREPARED["y_test"],
    split="test",
    context=PREPARED["enriched"]["test"],
)
VERDICTS = RESULT.extras["verdicts"]

print(f"split        : {RESULT.split} | {RESULT.n_samples:,} candidats")
print(f"utilisateurs : {RESULT.extras['users_evaluated']} évalués "
      f"sur {RESULT.extras['users_total']}")
print(f"métrique clé : {RESULT.primary_metric} = {RESULT.primary_value:.4f}")
print(f"statut       : {VERDICTS['statut']}")
'''

_VERDICT_CELL = '''
# Chaque objectif est une phrase du contrat de service, pas un nombre décoratif. Le tableau dit
# ce qui est tenu, ce qui ne l'est pas, et surtout **pourquoi** l'objectif existe.
# `VERDICTS` a la forme produite par l'évaluateur : une liste d'objectifs sous la clé
    # `objectifs`, et le décompte global sous `statut` / `respectes` / `mesurables`.
entries = list(VERDICTS.get("objectifs") or [])
if entries:
    objectifs = pd.DataFrame(entries).set_index("critere")
    display(objectifs[["mesure", "seuil", "operateur", "verdict"]])
    print()
    print(f"{VERDICTS.get('respectes')}/{VERDICTS.get('mesurables')} objectifs mesurables "
          f"respectés — statut : {VERDICTS.get('statut')}")
    print()
    for entry in entries:
        marque = {"respecte": "OK ", "non_respecte": "KO ", "non_mesurable": "-- "}.get(
            str(entry.get("verdict")), "?? "
        )
        print(f"  {marque}{entry.get('critere')} : {entry.get('lecture')}")
else:  # forme plate : {nom: booléen}
    objectifs = pd.DataFrame(
        [{"objectif": name, "statut": "acquis" if ok else "manqué"} for name, ok in VERDICTS.items()]
    )
    display(objectifs)
    print(f"{int((objectifs['statut'] == 'acquis').sum())}/{len(objectifs)} objectifs acquis")
print()
print("Lecture : un objectif manqué n'est pas une note, c'est une décision à prendre.")
print("« NDCG sous le seuil » → travailler les features ou le modèle.")
print("« couverture sous le seuil » → contrainte de diversité à publier, pas un bug de score.")
print("« rappel froid sous le seuil » → le moteur ne sert pas les nouveaux : problème produit.")
'''

_SEGMENTS_CELL = '''
# La moyenne cache toujours un segment. Ici les segments sont ceux qui correspondent à des
# décisions : le démarrage froid (produit), la catégorie (catalogue), la période (saisonnalité).
segments = RESULT.per_segment
if segments.empty:
    print("Aucune ventilation produite (colonnes de segmentation absentes du contexte).")
else:
    # L'index brut est un numéro de ligne : sans libellé, `idxmax()` renverrait « 1 » au lieu de
    # « segment_utilisateur / chaud ». On construit donc un index lisible avant tout classement.
    if {"axe", "segment"}.issubset(segments.columns):
        segments = segments.set_index(
            segments["axe"].astype(str) + " / " + segments["segment"].astype(str)
        )
    display(segments)

    print()
    print("Ce que la ventilation dit de façon exploitable :")
    numerique = segments.select_dtypes("number")
    if not numerique.empty and "ndcg" in numerique.columns:
        meilleur = numerique["ndcg"].idxmax()
        pire = numerique["ndcg"].idxmin()
        ecart = float(numerique.loc[meilleur, "ndcg"] - numerique.loc[pire, "ndcg"])
        print(f"  segment le mieux servi : {meilleur} (NDCG {numerique.loc[meilleur, 'ndcg']:.4f})")
        print(f"  segment le moins servi : {pire} (NDCG {numerique.loc[pire, 'ndcg']:.4f})")
        print(f"  écart                  : {ecart:+.4f}")
        print()
        print("Un écart important entre segments justifie un modèle ou un réglage par segment,")
        print("ou au minimum un objectif de service distinct — pas une moyenne unique.")

    if not RESULT.baselines.empty:
        print()
        print("Références mesurées sur le même split :")
        display(RESULT.baselines)
'''

_COLD_CELL = '''
# Le démarrage froid est LE point de rupture d'un moteur de recommandation : c'est là que la
# référence « intention récente » tombe à zéro (pas d'historique) et que le catalogue populaire
# ne dit rien de la personne. On lit donc ce que le moteur publie réellement pour ces utilisateurs.
ratio_froid = RESULT.extras.get("cold_start_recall_ratio", float("nan"))
print(f"ratio de rappel froid / global : {ratio_froid:.2f}")
print("1.00 = les nouveaux sont servis exactement comme les habitués")
print("<0.80 = le moteur dégrade systématiquement le service des nouveaux arrivants")
print()

published = RESULT.predictions
if published.empty:
    print("Aucune publication produite.")
else:
    colonne_activite = evaluator.settings.activity_column
    colonne_groupe = evaluator.settings.group_column
    # Le cadre publié ne porte que les colonnes nécessaires à la recommandation : l'activité de
    # l'utilisateur reste dans le cadre enrichi du split. On la recolle par identifiant, ce qui
    # évite de supposer que les deux cadres partagent leurs colonnes.
    contexte = PREPARED["enriched"]["test"]
    if colonne_activite not in published.columns and colonne_activite in contexte.columns:
        activite = contexte.groupby(colonne_groupe, observed=True)[colonne_activite].first()
        published = published.assign(
            **{colonne_activite: published[colonne_groupe].map(activite)}
        )
    if colonne_activite in published.columns:
        froid = published[published[colonne_activite] <= evaluator.settings.cold_user_max_orders]
        chaud = published[published[colonne_activite] > evaluator.settings.cold_user_max_orders]
        comparaison = pd.DataFrame(
            {
                "utilisateurs": pd.Series(
                    {
                        "démarrage froid": froid[colonne_groupe].nunique(),
                        "habitués": chaud[colonne_groupe].nunique(),
                    }
                ),
                "publications": pd.Series({"démarrage froid": len(froid), "habitués": len(chaud)}),
                "score moyen": pd.Series(
                    {
                        "démarrage froid": froid["y_pred"].mean() if "y_pred" in froid else float("nan"),
                        "habitués": chaud["y_pred"].mean() if "y_pred" in chaud else float("nan"),
                    }
                ),
                "articles distincts": pd.Series(
                    {
                        "démarrage froid": froid[evaluator.settings.item_column].nunique(),
                        "habitués": chaud[evaluator.settings.item_column].nunique(),
                    }
                ),
            }
        )
        display(comparaison)

        print()
        print("Diagnostic : pour un utilisateur froid, le moteur ne peut s'appuyer que sur les")
        print("features **article** (popularité, note, prix, marge) et **contexte** (période,")
        print("canal). S'il publie exactement le même top-K pour tous les nouveaux, c'est que")
        print("ces features dominent — et la personnalisation n'existe pas encore pour eux.")
        if froid[colonne_groupe].nunique() > 0:
            premiers = (
                froid.groupby(colonne_groupe, observed=True)[evaluator.settings.item_column]
                .apply(lambda items: tuple(items[:5]))
                .head(6)
            )
            print()
            print("Top-5 publié pour quelques utilisateurs froids :")
            display(premiers.rename("top-5").to_frame())
            distincts = len(set(premiers.tolist()))
            print(f"{distincts} listes distinctes sur {premiers.size} utilisateurs froids")
            if distincts == 1:
                print("=> liste identique pour tous : le moteur est en mode « catalogue populaire ».")
            else:
                print("=> les listes diffèrent : le contexte et l'article produisent de la variation.")
    else:
        print(f"Colonne d'activité '{colonne_activite}' absente des publications :")
        print("le segment froid ne peut pas être isolé dans ce notebook.")
'''

_COVERAGE_CELL = '''
# Un moteur pertinent mais monotone est un moteur rejeté : il épuise le catalogue, lasse les
# utilisateurs et négocie mal avec les fournisseurs. La couverture et la concentration ne sont pas
# des métriques de confort, ce sont des contraintes de service.
couverture = RESULT.extras.get("catalog_coverage", float("nan"))
concentration = RESULT.extras.get("herfindahl_top_k", float("nan"))
biais_popularite = RESULT.extras.get("popularity_bias_share", float("nan"))
rupture = RESULT.extras.get("out_of_stock_published_share", float("nan"))

table = pd.Series(
    {
        "couverture catalogue (top-K union)": couverture,
        "minimum exigé": evaluator.settings.coverage_min,
        "Herfindahl des publications": concentration,
        "part des slots prise par le décile chaud": biais_popularite,
        "publications en rupture de stock": rupture,
    },
    name="valeur",
).to_frame()
display(table)

print()
print("Lecture :")
print(f"  {couverture:.1%} du catalogue apparaît dans au moins un top-K "
      f"(minimum exigé {evaluator.settings.coverage_min:.0%})")
if biais_popularite == biais_popularite:
    print(f"  {biais_popularite:.1%} des emplacements reviennent au décile d'articles le plus vu :")
    print("  c'est la mesure directe du biais de popularité du moteur.")
if rupture == rupture:
    print(f"  {rupture:.1%} des publications concernent un article en rupture : gaspillage")
    print("  d'emplacement, corrigé à l'inférence par le filtre de disponibilité.")

if not published.empty:
    colonne_article = evaluator.settings.item_column
    compte = published[colonne_article].value_counts()
    parts = compte.to_numpy(dtype="float64") / float(compte.sum())
    courbe = pd.DataFrame(
        {
            "part du catalogue": [0.01, 0.05, 0.10, 0.25, 0.50],
        }
    )
    cumulees = []
    for part in courbe["part du catalogue"]:
        n = max(1, round(part * compte.size))
        cumulees.append(round(float(parts[:n].sum()), 4))
    courbe["part des publications"] = cumulees
    display(courbe)
    print()
    print("Si 1 % du catalogue capte une part disproportionnée des publications, la")
    print("diversité est le prochain chantier — avant toute amélioration du NDCG.")
'''

_BACKTEST_CELL = '''
# Un NDCG moyen sur toute la période cache les saisons. Le backtest découpe le test en plis
# chronologiques et mesure la dispersion. La question décisive n'est pas « ça varie ? » — ça varie
# toujours — mais « ça varie plus que le bruit d'échantillonnage ? ».
temporel = RESULT.backtest
if temporel.empty:
    print("Backtest non produit (colonne temporelle absente du contexte).")
else:
    display(temporel)
    dispersion = RESULT.extras.get("dispersion_backtest", {})
    if dispersion:
        display(pd.Series(dispersion, name="dispersion entre plis").to_frame())
        amplitude = float(dispersion.get("amplitude", float("nan")))
        plafond_bruit = float(dispersion.get("plancher_bruit", dispersion.get("bruit", float("nan"))))
        seuil = evaluator.settings.backtest_spread_max
        print()
        print(f"amplitude observée   : {amplitude:.4f}")
        print(f"seuil contractuel    : {seuil:.4f}")
        if plafond_bruit == plafond_bruit:
            print(f"plancher de bruit    : {plafond_bruit:.4f}")
        print()
        if amplitude <= seuil:
            print("Sous le seuil : la qualité est stable d'une période à l'autre.")
            print("ATTENTION à ne pas lire « stable » comme « bon » : un moteur médiocre peut")
            print("être très régulièrement médiocre. Le backtest mesure la dispersion, pas le niveau.")
        else:
            print("Au-dessus du seuil : chercher une cause (saisonnalité, dérive du catalogue,")
            print("vieillissement des features) avant de retoucher les hyperparamètres.")

    colonne_temps = evaluator.settings.time_column
    if colonne_temps and colonne_temps in PREPARED["enriched"]["test"].columns:
        test = PREPARED["enriched"]["test"]
        periode = test.groupby(
            pd.to_datetime(test[colonne_temps]).dt.to_period("M"), observed=True
        ).size()
        print()
        print("Volume de candidats par mois dans le test (le contexte du backtest) :")
        display(periode.rename("candidats").to_frame())
'''

_WORST_CELL = '''
# Les pires classements sont la matière première du diagnostic : on y lit ce que le moteur a
# publié, ce qu'il aurait dû publier, et quelle feature l'a trompé.
erreurs = RESULT.errors
if erreurs.empty:
    print("Aucun mauvais classement produit par l'évaluateur.")
else:
    display(erreurs.head(12))
    print()
    print("Colonnes utiles pour le diagnostic :")
    for colonne in erreurs.columns:
        print(f"  {colonne}")
    print()
    colonne_groupe = evaluator.settings.group_column
    if colonne_groupe in erreurs.columns:
        pires = erreurs[colonne_groupe].value_counts().head(5)
        print("Utilisateurs les plus mal servis :")
        display(pires.rename("occurrences").to_frame())
        print()
        print("Un utilisateur qui revient plusieurs fois dans les pires classements n'est pas")
        print("un accident : c'est un profil que les features ne décrivent pas. C'est le")
        print("signal d'une feature manquante (intention, saison, budget), pas d'un réglage.")
'''

_IMPORTANCE_CELL = '''
# L'importance par permutation répond à la seule question qui compte pour un classement : si je
# détruis l'information de cette colonne, de combien le NDCG tombe-t-il ? Une feature importante
# pour prédire la ligne peut être inutile pour ordonner la liste — et inversement.
importance = RESULT.feature_importance
if importance.empty:
    print("Aucune importance produite (le modèle n'expose pas de scores exploitables).")
else:
    display(importance)

    colonne_delta = next(
        (
            name
            for name in importance.columns
            if any(
                marque in name.lower()
                for marque in ("importance", "delta", "perte", "drop", "cout", "coût")
            )
        ),
        None,
    )
    if colonne_delta is not None:
        classement = importance.sort_values(colonne_delta, ascending=False)
        print()
        print(f"Classement par {colonne_delta} (chute de la métrique quand la colonne est mélangée) :")
        display(classement.head(10))
        print()
        premier = classement.iloc[0]
        print(f"Feature dominante : {premier.name}")
        print("Si une seule feature porte l'essentiel du classement, le moteur est fragile :")
        print("sa qualité disparaît avec la disponibilité de cette colonne.")
    else:
        print()
        print("Colonnes produites : " + ", ".join(importance.columns))

    print()
    print("Mise en garde : une colonne de popularité dominante est attendue — c'est la")
    print("référence à battre. Ce qui doit inquiéter, c'est une colonne qui reproduirait la")
    print("cible (fuite) ou une colonne d'identité utilisateur (mémorisation sans généralisation).")
'''

_RECOMMENDATIONS_CELL = '''
# Recommandations. Chacune est formulée comme une action vérifiable : ce qu'on change, l'effet
# attendu sur quelle métrique, et comment on le mesure. Une recommandation sans métrique cible
# est une opinion.
recommandations = [
    {
        "n°": 1,
        "action": "Filtrer les ruptures de stock APRÈS le classement et republier le candidat "
        "suivant (déjà implémenté dans `src/inference/predictor.py`).",
        "effet attendu": "supprime les emplacements perdus sans toucher au NDCG mesuré",
        "comment vérifier": "part des publications en rupture = 0 sur le test d'inférence",
    },
    {
        "n°": 2,
        "action": "Ajouter des features d'intention de court terme (sessions en cours, panier, "
        "recherches) plutôt que d'augmenter la capacité du modèle.",
        "effet attendu": "NDCG et rappel, surtout sur le segment chaud",
        "comment vérifier": "gain de NDCG supérieur à la dispersion entre graines (notebook 04)",
    },
    {
        "n°": 3,
        "action": "Publier un top-K dédié au démarrage froid : tri par popularité pondérée par la "
        "note et la nouveauté, faute d'historique.",
        "effet attendu": "ratio de rappel froid vers 1.00",
        "comment vérifier": "ratio rappel froid / global, par segment, sur le backtest",
    },
    {
        "n°": 4,
        "action": "Introduire une contrainte de diversité (max par catégorie) dans la publication.",
        "effet attendu": "couverture et Herfindahl, au prix d'un léger recul du NDCG",
        "comment vérifier": "table de sensibilité à K et arbitrage marge du notebook 04",
    },
    {
        "n°": 5,
        "action": "Arbitrer explicitement pertinence contre marge dans le score de publication, "
        "avec un poids configurable.",
        "effet attendu": "valeur par publication, NDCG en léger recul",
        "comment vérifier": "courbe marge/NDCG du rapport, point de fonctionnement retenu",
    },
    {
        "n°": 6,
        "action": "Rejouer le backtest à chaque réentraînement et comparer l'amplitude au plancher "
        "de bruit, jamais au seul seuil contractuel.",
        "effet attendu": "détection des dérives saisonnières avant qu'elles soient visibles",
        "comment vérifier": "amplitude inter-plis et son erreur type dans le rapport",
    },
]
display(pd.DataFrame(recommandations).set_index("n°"))

print()
print("Ce que ce notebook NE recommande PAS, volontairement :")
print("  - augmenter la profondeur ou le nombre d'arbres : la grille du notebook 04 montre")
print("    que le gain reste sous la dispersion entre graines ;")
print("  - évaluer en AUC ou en précision globale : ces métriques mélangent des utilisateurs")
print("    de volumes différents et ne décrivent pas le service rendu ;")
print("  - choisir le K sur le NDCG seul : le K est un arbitrage qualité/diversité/coût.")
'''

_FIGURES_CELL = """
# Les figures du rapport : elles servent à trancher une décision, pas à illustrer le code. Le
# générateur est celui de la production (`RankingPlots`, utilisé par `src/evaluation/reports.py`),
# et les fichiers sont écrits dans `outputs/notebooks` — jamais dans `artifacts/`, que
# `make evaluate` possède.
from src.visualization.plots import RankingPlots

figures = NB_PATHS.figures_dir
try:
    produites = RankingPlots(figures).save_all(RESULT)
    print(f"{len(produites)} figures produites dans {figures}")
    for nom, chemin in sorted(produites.items()):
        print(f"  - {nom:<26} {chemin.name}")
except Exception as error:  # une figure manquante ne doit pas masquer l'analyse
    produites = {}
    print(f"Figures non produites ({type(error).__name__}: {error}).")
    print("Le rapport écrit (`scripts/evaluate.py`) reste la source de vérité.")

# Deux figures suffisent à résumer le verdict : la qualité contre ses seuils, et la position du
# modèle entre le plancher et le plafond.
for cle in ("metrics_bar", "ceiling_position"):
    chemin = produites.get(cle)
    if chemin is not None and Path(chemin).exists():
        display(Image(filename=str(chemin)))
"""


def build_04_model_exploration(context: NotebookContext, destination: Path) -> Path:
    """Build ``04_model_exploration.ipynb`` for a ranking project.

    L'exploration d'un moteur de classement ne ressemble pas à celle d'un classifieur : il n'y a
    pas de « meilleure accuracy », il y a un plancher, une référence métier à battre, un plafond
    connaissable, et une coupure de publication qui change la question posée. Ce notebook installe
    ces quatre repères avant toute comparaison d'algorithmes.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    spec = context.spec
    cells: list[NotebookNode] = [
        _md(
            f"""# 04 — Exploration et comparaison de modèles de classement

**Projet** : {spec.title}
**Modèle configuré** : `{spec.model.algorithm}` ({spec.model.display_name})
**Pourquoi ce choix** : {spec.model.rationale}

Un moteur de recommandation ne se choisit pas comme un classifieur. Il n'existe pas de
« meilleure exactitude » : il existe un **plancher** (le tirage aléatoire), une **référence
métier** (le tri par popularité, que le site sait déjà faire sans modèle), un **plafond**
(ce que la donnée permet de connaître au mieux) et une **coupure de publication** qui change la
question posée. Ce notebook installe ces quatre repères, puis compare les algorithmes à protocole
identique — mêmes candidats, mêmes utilisateurs, même coupure, mêmes graines."""
        ),
        _objectives(
            context,
            [
                "Mesurer le plancher, la référence métier et le plafond atteignable.",
                "Comprendre pourquoi la métrique se calcule **par utilisateur**, jamais globalement.",
                "Comparer les algorithmes de la stack à protocole identique, dispersion comprise.",
                "Choisir la coupure K en connaissance de cause (qualité contre diversité).",
                "Vérifier l'absence de fuite par une sonde univariée.",
            ],
        ),
        _md(
            """## 0. Mise en place

Même configuration que `python -m src.main`, à une différence près : le volume est réduit
(`NB_ROWS`) pour que le notebook s'exécute en quelques secondes. Les distributions restent
réalistes, mais les écarts fins ne sont plus lisibles — c'est assumé, et la section 7 mesure
précisément ce que le bruit autorise à conclure."""
        ),
        _code(SETUP, context),
        _code(LOAD_RAW, context),
        _code(PREPARE, context),
        _md(
            """## 1. Le protocole : des métriques par utilisateur

Une liste de recommandation se juge **liste par liste**. Les fonctions ci-dessous sont écrites
dans le notebook, et non importées, pour que leur définition soit lisible : le NDCG normalise le
gain par le gain idéal, la précision et le rappel se lisent à la coupure K, et la moyenne se fait
sur les utilisateurs — pas sur les lignes."""
        ),
        _code(RANKING_HELPERS, context),
        _code(_FIT_CELL, context),
        _md(
            """Le score de classement n'a pas besoin d'être une probabilité : il lui faut être
ordonnable. La fonction ci-dessous prend ce que l'estimateur sait produire, ce qui permet de
comparer loyalement un arbre (probabilité) et un SVM (distance à la frontière). Les colonnes
utilisées ensuite sont résolues depuis la configuration, jamais écrites dans le notebook."""
        ),
        _code(_SCORE_HELPERS, context),
        _md(
            """## 2. Plancher, référence métier et modèle

Quatre moteurs, un seul protocole. Le tirage aléatoire borne ce qu'on peut obtenir sans
information ; le tri par popularité est ce que le site sait déjà faire sans modèle ; l'intention
récente est l'exploitation pure (remontrer ce que la personne a vu). **Un modèle qui ne bat pas
la popularité de façon nette ne justifie pas son coût de mise en œuvre.**"""
        ),
        _code(_BASELINES_CELL, context),
        _insight(
            [
                "L'aléatoire n'est pas zéro : avec des candidats de volumes comparables, il obtient un NDCG non nul. Toute progression se mesure à partir de ce plancher.",
                "La référence à battre est la popularité, pas l'aléatoire. C'est elle qui figure au contrat du projet.",
                "Un moteur qui se contente de redisposer les articles les plus vus n'ajoute rien : l'écart à la référence est la seule justification du modèle.",
            ]
        ),
        _md(
            """## 3. Le plafond : ce que la donnée permet de connaître

Le générateur synthétique connaît la probabilité de pertinence **avant** bruit : il peut donc
classer parfaitement tout ce qui est connaissable. Ce classement oracle définit le plafond, et la
distance qui reste n'est pas une marge de progression — c'est du bruit irréductible. Sur données
réelles ce plafond n'est pas publié : on ne compare alors qu'aux références, et c'est précisément
pourquoi la section 2 existe."""
        ),
        _code(_CEILING_CELL, context),
        _insight(
            [
                "Un modèle qui capture une large part du plafond est au bout de ce que la donnée autorise : la suite du travail est sur les features, pas sur les hyperparamètres.",
                "Le plafond sépare « le modèle est faible » de « la donnée est bruitée ». Sans lui, on optimise du bruit.",
            ]
        ),
        _md(
            """## 4. Lecture globale contre lecture par utilisateur

C'est la démonstration la plus importante du notebook. La métrique « globale » n'est pas une
version bruitée de la métrique par utilisateur : c'est **une autre question**, qui pondère chaque
individu par son volume de candidats."""
        ),
        _code(_PER_USER_CELL, context),
        _insight(
            [
                "Une précision globale mélange des utilisateurs très actifs et des nouveaux venus : elle décrit le moteur sur les gros clients, pas le service rendu à un utilisateur moyen.",
                "Toute métrique de classement se moyenne sur l'unité de décision — ici l'utilisateur, ailleurs la session ou la requête.",
                "C'est aussi ce qui rend l'AUC inadaptée : elle compare des paires d'articles appartenant à des utilisateurs différents, paires que personne ne verra jamais côte à côte.",
            ]
        ),
        _md(
            """## 5. Sensibilité à la coupure K

Publier 5 ou 20 articles ne mesure pas la même qualité. Cette table est l'outil de décision du K :
elle montre le compromis structurel (la précision chute, le rappel monte) et son prix en diversité
(la couverture progresse, la concentration recule)."""
        ),
        _code(_CUTOFF_CELL, context),
        _insight(
            [
                "Le NDCG varie peu avec K quand le haut de liste est bon : c'est la signature d'un classement correct, pas d'un hasard heureux.",
                "La précision chute mécaniquement avec K, le rappel monte : choisir K sur la seule précision revient à choisir K = 1, donc à ne rien recommander.",
                "Le K retenu est une contrainte produit (surface d'affichage réelle), déclarée dans `conf/config.yaml` — pas un hyperparamètre à optimiser.",
            ]
        ),
        _md(
            """## 6. Comparaison des algorithmes à protocole identique

`params={{}}` construit chaque algorithme avec ses réglages par défaut : les `model.params`
configurés sont propres au boosting par histogrammes et les injecter ailleurs lèverait une erreur.
Le modèle configuré **et réglé** est mesuré en section 2 ; ici on compare les familles."""
        ),
        _code(_ALGORITHMS_CELL, context),
        _insight(
            [
                "Un algorithme qui échoue est conservé dans le tableau avec ses avertissements : masquer un échec, c'est perdre une information diagnostique.",
                "Le classement se lit avec la dispersion de la section 7 : un écart inférieur à l'écart-type entre graines n'est pas un gain.",
                "La couverture accompagne chaque ligne : un algorithme plus précis mais plus concentré coûte en diversité, et ce coût est réel.",
            ]
        ),
        _md(
            """## 7. Stabilité entre graines

Trois graines suffisent, sur le volume d'un notebook, à estimer la dispersion d'un même
algorithme. Cette dispersion est l'étalon de toute comparaison : en dessous, on mesure un tirage."""
        ),
        _code(_SEEDS_CELL, context),
        _insight(
            [
                "Choisir le vainqueur d'une grille sur une seule graine, c'est choisir un bruit : la décision doit survivre au changement de graine.",
                "La dispersion entre graines est aussi le plancher de ce qu'on peut promettre en production : un gain annoncé plus petit ne sera pas vérifiable.",
            ]
        ),
        _md(
            """## 8. Sonde de fuite

Aucune colonne observable ne doit approcher le plafond. Si une seule y parvenait, elle porterait
l'information que le générateur garde pour lui — et le modèle « appris » ne serait qu'un proxy de
la réponse. Le test est univarié volontairement : c'est le plus sévère."""
        ),
        _code(_LEAK_CELL, context),
        _insight(
            [
                "La meilleure colonne seule reste proche de la référence de popularité : c'est attendu, l'audience est le signal observable le plus fort.",
                "Le modèle complet dépasse nettement chaque colonne isolée : il combine l'information, il ne la recopie pas.",
                "Sur données réelles, ce test se refait à chaque nouvelle feature — une fuite s'introduit par un joint de données, pas par un algorithme.",
            ]
        ),
        _md(
            """## 9. Grille d'hyperparamètres

La grille est déclarée dans le manifeste du projet et centrée sur le point retenu, pour que le
notebook **rejoue** la décision au lieu de la paraphraser."""
        ),
        _code(_GRID_CELL, context),
        _insight(
            [
                "Le point configuré doit sortir de la grille : si ce n'est pas le cas, la décision est à documenter (coût, variance, couverture).",
                "La grille du notebook est volontairement étroite : la recherche large appartient au pipeline, pas à un notebook pédagogique.",
            ]
        ),
        _md(
            """## Conclusion

Quatre repères ont été installés : le plancher aléatoire, la référence de popularité, le plafond
oracle et la coupure de publication. Les algorithmes ont été comparés à protocole identique, la
dispersion entre graines a borné ce qu'on peut conclure, et la sonde de fuite a écarté le scénario
le plus dangereux — un modèle qui recopie la réponse.

Le notebook 05 entraîne le modèle retenu sur le volume complet, avec les callbacks et la
journalisation de production. Le notebook 06 analyse les erreurs de classement et formule les
recommandations."""
        ),
    ]
    result_path = write_notebook(destination / "04_model_exploration.ipynb", cells)
    _replace_placeholder(result_path)
    return result_path


def build_06_error_analysis(context: NotebookContext, destination: Path) -> Path:
    """Build ``06_error_analysis.ipynb`` for a ranking project.

    L'analyse d'erreur d'un moteur de classement ne cherche pas « les lignes mal prédites » : elle
    cherche **les utilisateurs mal servis**, les segments sacrifiés, les emplacements perdus et les
    causes actionnables. Ce notebook s'appuie sur l'évaluateur du projet (qui mesure tout) et
    prolonge là où le rapport écrit ne peut pas aller : lire des publications, tester une
    hypothèse, chiffrer un arbitrage.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    spec = context.spec
    cells: list[NotebookNode] = [
        _md(
            f"""# 06 — Analyse d'erreur et recommandations

**Projet** : {spec.title}
**Modèle analysé** : `{spec.model.algorithm}` ({spec.model.display_name})
**Métrique de contrat** : `{spec.metrics.primary}` (seuil `{spec.metrics.min_primary}`)

Une métrique moyenne ne dit jamais qui est mal servi. Ce notebook part du verdict produit par
l'évaluateur du projet, puis descend : par segment, par utilisateur, par emplacement publié. Il se
termine par des recommandations **vérifiables** — ce qu'on change, l'effet attendu, et la métrique
qui le prouve."""
        ),
        _objectives(
            context,
            [
                "Lire le verdict objectif par objectif, et savoir quoi faire de chacun.",
                "Identifier les segments mal servis et le coût du démarrage froid.",
                "Distinguer une instabilité réelle du bruit d'échantillonnage.",
                "Remonter des pires classements à une cause actionnable.",
                "Formuler des recommandations chiffrées et vérifiables.",
            ],
        ),
        _md(
            """## 0. Mise en place

La journalisation est descendue au niveau `ERROR` dans ce projet. Ce n'est pas un réglage de
confort : `fit` émet un avertissement par métrique `*_at_k` (`missing input(s) ['groups']`), parce
que la matrice pré-traitée ne porte plus l'identité de l'utilisateur et que le registre refuse, à
juste titre, de calculer une métrique **par utilisateur** sans groupe. Ces métriques sont
recalculées ici sur le cadre enrichi, et en production par le `Trainer` qui reçoit `groups_val`.
Les avertissements n'apporteraient donc rien — et noieraient les tableaux."""
        ),
        _code(SETUP, context),
        _code(LOAD_RAW, context),
        _code(PREPARE, context),
        _code(RANKING_HELPERS, context),
        _code(_FIT_CELL, context),
        _code(_SCORE_HELPERS, context),
        _md(
            """## 1. Le verdict d'abord

L'évaluateur du projet calcule les métriques par utilisateur, les références, la couverture, le
backtest chronologique, les pires classements et l'importance des features. Le notebook ne recalcule
rien : il interprète."""
        ),
        _code(_EVALUATE_CELL, context),
        _code(_VERDICT_CELL, context),
        _insight(
            [
                "Un objectif manqué n'est pas une note, c'est une décision à prendre — et chaque objectif pointe vers un chantier différent.",
                "Le statut global (`conforme`, `partiel`, `non_conforme`) est contractuel : il conditionne la mise en production, pas la discussion technique.",
            ]
        ),
        _md(
            """## 2. Ventilation par segments

La moyenne cache toujours un segment. Les segments affichés correspondent à des décisions :
démarrage froid (produit), catégorie (catalogue), période (saisonnalité)."""
        ),
        _code(_SEGMENTS_CELL, context),
        _insight(
            [
                "Un écart important entre segments justifie un réglage par segment, ou au minimum un objectif de service distinct — jamais une moyenne unique.",
                "Les références mesurées sur le même split permettent de savoir si un segment faible l'est aussi pour la popularité : si oui, le problème vient de la donnée, pas du modèle.",
            ]
        ),
        _md(
            """## 3. Le démarrage froid

C'est le point de rupture d'un moteur de recommandation : la référence « intention récente » y
tombe à zéro, et le catalogue populaire ne dit rien de la personne. On lit donc ce que le moteur
publie **réellement** pour ces utilisateurs."""
        ),
        _code(_COLD_CELL, context),
        _insight(
            [
                "Un ratio de rappel froid proche de 1 signifie que les nouveaux sont servis comme les habitués ; nettement sous 1, le moteur les dégrade systématiquement.",
                "Si la liste publiée est identique pour tous les nouveaux, le moteur est en mode « catalogue populaire » : la personnalisation n'existe pas encore pour eux.",
                "Pour un utilisateur froid, seules les features article et contexte portent de l'information — d'où l'intérêt de la note, de la nouveauté et de la saisonnalité.",
            ]
        ),
        _md(
            """## 4. Couverture, concentration et biais de popularité

Un moteur pertinent mais monotone est un moteur rejeté : il épuise le catalogue, lasse les
utilisateurs et négocie mal avec les fournisseurs. Ce sont des contraintes de service, pas des
métriques de confort."""
        ),
        _code(_COVERAGE_CELL, context),
        _insight(
            [
                "La part des emplacements captée par le décile d'articles le plus vu est la mesure directe du biais de popularité du moteur.",
                "Les publications en rupture de stock sont des emplacements perdus : elles se corrigent à l'inférence par un filtre de disponibilité, pas par le score.",
                "Si 1 % du catalogue capte une part disproportionnée des publications, la diversité est le chantier suivant — avant toute amélioration du NDCG.",
            ]
        ),
        _md(
            """## 5. Backtest et plancher de bruit

Un NDCG moyen sur toute la période cache les saisons. La question décisive n'est pas « ça varie ? »
— ça varie toujours — mais « ça varie **plus que le bruit** d'échantillonnage d'un pli ? »."""
        ),
        _code(_BACKTEST_CELL, context),
        _insight(
            [
                "Le seuil contractuel d'amplitude est calibré au-dessus du plancher de bruit : en dessous de ce plancher, le critère mesurerait le hasard.",
                "« Stable » ne veut pas dire « bon » : un moteur médiocre peut être très régulièrement médiocre. Le backtest mesure la dispersion, pas le niveau.",
                "Une amplitude qui dépasse le seuil appelle une cause (saisonnalité, dérive du catalogue, vieillissement des features) avant tout retouchage d'hyperparamètres.",
            ]
        ),
        _md(
            """## 6. Les pires classements

C'est la matière première du diagnostic : on y lit ce que le moteur a publié, ce qu'il aurait dû
publier, et quelle information lui a manqué."""
        ),
        _code(_WORST_CELL, context),
        _insight(
            [
                "Un utilisateur qui revient plusieurs fois dans les pires classements n'est pas un accident : c'est un profil que les features ne décrivent pas.",
                "Le signal appelle une feature manquante (intention, saison, budget), pas un réglage de capacité.",
            ]
        ),
        _md(
            """## 7. Importance par permutation

La seule question qui compte pour un classement : si je détruis l'information de cette colonne, de
combien la métrique tombe-t-elle ? Une feature importante pour prédire la ligne peut être inutile
pour ordonner la liste — et inversement."""
        ),
        _code(_IMPORTANCE_CELL, context),
        _insight(
            [
                "Une colonne de popularité dominante est attendue : c'est la référence à battre, pas une anomalie.",
                "Ce qui doit inquiéter, c'est une colonne qui reproduirait la cible (fuite) ou une identité utilisateur (mémorisation sans généralisation).",
                "Si une seule feature porte l'essentiel du classement, le moteur est fragile : sa qualité disparaît avec la disponibilité de cette colonne.",
            ]
        ),
        _md(
            """## 8. Hypothèses et recommandations

Chaque recommandation est une action vérifiable : ce qu'on change, l'effet attendu, la métrique qui
le prouve. Une recommandation sans métrique cible est une opinion."""
        ),
        _code(_RECOMMENDATIONS_CELL, context),
        _insight(
            [
                "Les recommandations portent d'abord sur la donnée et la publication (features d'intention, filtre de stock, top-K froid, diversité) : c'est là que le gain dépasse la dispersion entre graines.",
                "Augmenter la capacité du modèle est explicitement écarté tant que le gain reste sous le bruit mesuré au notebook 04.",
                "Le choix de K et l'arbitrage marge sont des décisions produit : elles se prennent sur des courbes, pas sur une métrique unique.",
            ]
        ),
        _md("## 9. Figures du rapport"),
        _code(_FIGURES_CELL, context),
        _md(
            """## Conclusion

Le verdict a été lu objectif par objectif, les segments mal servis ont été identifiés, le démarrage
froid a été mesuré au niveau des publications réelles, la couverture et le biais de popularité ont
été chiffrés, et l'instabilité a été comparée à son plancher de bruit. Les recommandations qui en
découlent sont actionnables et vérifiables.

Ce qui distingue cette analyse d'un rapport de classification : l'unité n'est jamais la ligne,
toujours l'utilisateur ; et le critère de succès n'est jamais la métrique seule, toujours la
métrique **avec** ses contraintes de service (couverture, disponibilité, équité entre segments)."""
        ),
    ]
    result_path = write_notebook(destination / "06_error_analysis.ipynb", cells)
    _replace_placeholder(result_path)
    return result_path
