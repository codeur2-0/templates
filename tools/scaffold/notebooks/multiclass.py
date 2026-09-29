"""Cellules de notebooks spécifiques à la tâche **classification multi-classes**.

Le squelette des six notebooks (``notebooks/tabular.py``) est partagé par toutes les tâches
tabulaires ; seules les cellules qui parlent de *classe positive*, de *seuil* ou de *ROC binaire*
doivent changer. Un classifieur multi-classes pose d'autres questions :

* **01, section 5** (:func:`structure_cells`) — comment les modes se répartissent, quelles
  combinaisons de capteurs les séparent, et pourquoi la règle actuelle se trompe ;
* **04** (:func:`build_04_model_exploration`) — plancher, règle métier et plafond **avant** toute
  comparaison d'algorithmes, apport des features physiques, effet de la pondération de classes
  sur la décision, grille de réglage, dispersion entre graines ;
* **06** (:func:`build_06_error_analysis`) — verdict de l'évaluateur de production, confusions lues
  physiquement, argmax contre décision à coût minimal, revue experte, calibration, plafond
  structurel et plan d'action.

Les helpers de rendu (``_code``, ``_md``, ``_insight``, jetons ``__XXX__``) sont ceux de
``tabular.py`` : les notebooks de toutes les tâches partagent la même grammaire.
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

__all__ = [
    "NB_ROWS_MULTICLASS",
    "build_04_model_exploration",
    "build_06_error_analysis",
    "structure_cells",
]

#: Volume des notebooks multi-classes : le volume nominal du pipeline. Avec six classes dont la
#: plus rare pèse ~6 %, un échantillon de 1 500 lignes ne laisserait qu'une vingtaine d'alarmes de
#: ce mode dans le split de test — un rappel mesuré sur vingt lignes est du bruit.
NB_ROWS_MULTICLASS = 6000

# ---------------------------------------------------------------------------------------
# Blocs de code partagés par les notebooks 04 et 06
# ---------------------------------------------------------------------------------------
REFERENCES = """
import json

from src.data.generators import SyntheticDataGenerator

# Le plafond oracle n'est calculable que par le générateur : lui seul connaît les vraies
# probabilités. On rejoue donc la génération configurée (mêmes options, même graine, même volume)
# dans `outputs/notebooks/data/raw` : les références y sont publiées exactement comme en
# production, et l'évaluateur du notebook les lira au même endroit que celui du pipeline.
REFERENCE_GENERATOR = SyntheticDataGenerator.from_config(CONFIG.model_dump(), NB_PATHS)
REFERENCE_GENERATOR.run()
GENERATION = json.loads((NB_PATHS.raw_dir / "generation_metadata.json").read_text(encoding="utf-8"))
same_rows = GENERATION["n_samples"] == len(raw)
print(f"références calculées sur {GENERATION['n_samples']} alarmes (identiques à `raw` : {same_rows})")

REFERENCE_TABLE = pd.DataFrame(
    {
        "référence": [
            "plancher : classe majoritaire",
            "règle actuelle : code automate",
            "plafond : oracle (vraies probabilités)",
        ],
        "macro-F1": [
            GENERATION["f1_macro_baseline_majority"],
            GENERATION["f1_macro_baseline_alarm_code"],
            GENERATION["f1_macro_ceiling_oracle"],
        ],
        "accuracy": [
            GENERATION["accuracy_baseline_majority"],
            GENERATION["accuracy_baseline_alarm_code"],
            GENERATION["accuracy_ceiling_oracle"],
        ],
    }
)
REFERENCE_TABLE.round(4)
"""

VALIDATION_TOOLS = '''
from src.evaluation.decision import DiagnosisSettings, minimum_cost_decision, realised_costs
from src.models import build_model
from src.models.factory import available_algorithms
from src.schemas.config import validate_config
from src.training.losses_metrics import MetricCalculator, MetricInputs

SETTINGS = DiagnosisSettings.resolve(CONFIG.model_dump())
LABELS = list(SETTINGS.class_order)
CALCULATOR = MetricCalculator(task=CONFIG.metrics.task, metrics=CONFIG.metrics.all_metrics)

# Les boucles d'exploration entraînent des dizaines de modèles : la validation croisée interne
# (un diagnostic du `fit`, pas une décision) y est coupée pour garder le notebook sous quelques
# minutes. Le split de validation reste l'arbitre, exactement comme en production.
_payload = CONFIG.model_dump()
_payload["model"]["cross_validation"] = {**dict(_payload["model"].get("cross_validation") or {}), "enabled": False}
CONFIG_FAST = validate_config(_payload)


def business_order(model: Any, probabilities: np.ndarray) -> np.ndarray:
    """Reorder the probability columns of ``model`` in the business class order."""
    classes = [str(value) for value in model.classes_]
    return np.asarray(probabilities)[:, [classes.index(label) for label in LABELS]]


def top_label_ece(truth: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of the top-label confidence (business-ordered columns)."""
    confidence = probabilities.max(axis=1)
    correct = np.asarray(LABELS)[probabilities.argmax(axis=1)] == truth
    index = np.clip(np.digitize(confidence, np.linspace(0, 1, bins + 1)[1:-1], right=True), 0, bins - 1)
    return float(sum((index == b).mean() * abs(correct[index == b].mean() - confidence[index == b].mean())
                     for b in range(bins) if (index == b).any()))


def score_on_validation(model: Any, prepared: dict[str, Any]) -> dict[str, float]:
    """Macro-F1, calibration and decision cost of a fitted model on the validation split."""
    truth = np.asarray(prepared["y_val"]).astype(str)
    raw_probabilities = model.predict_proba(prepared["X_val"])
    probabilities = business_order(model, raw_probabilities)
    predicted = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    values = CALCULATOR.evaluate(
        MetricInputs(
            y_true=truth,
            y_pred=predicted,
            y_proba=probabilities,
            extra={"classes": LABELS},
        )
    )
    decided = minimum_cost_decision(probabilities, LABELS, SETTINGS)
    no_failure = SETTINGS.no_failure_class
    real = truth != no_failure
    return {
        "macro_f1": float(values.get("f1_macro", float("nan"))),
        "balanced_accuracy": float(values.get("balanced_accuracy", float("nan"))),
        "log_loss": float(values.get("log_loss", float("nan"))),
        "ece": top_label_ece(truth, probabilities),
        "coût_argmax": float(realised_costs(truth, predicted, LABELS, SETTINGS).mean()),
        "coût_minimal": float(realised_costs(truth, decided, LABELS, SETTINGS).mean()),
        "pannes_acquittées": float(np.mean(decided[real] == no_failure)) if real.any() else float("nan"),
    }


def fit_candidate(config: Any, prepared: dict[str, Any], **overrides: Any) -> Any:
    """Build and fit a model on the prepared matrices (no callbacks)."""
    candidate = build_model(config, feature_names=prepared["feature_names"], **overrides)
    candidate.fit(prepared["X_train"], prepared["y_train"], X_val=prepared["X_val"], y_val=prepared["y_val"], callbacks=[])
    return candidate


# Référence linéaire de la stack : `logistic_regression` en scikit-learn, `linear` pour les
# réseaux, `*_linear` pour les boosters. Elle sert de témoin aux sections 4 et 7.
LINEAR = next(
    (name for name in available_algorithms(CONFIG.metrics.task) if "logistic" in name or "linear" in name),
    None,
)
print(f"{len(LABELS)} modes : {LABELS}")
print(f"référence linéaire de la stack : {LINEAR}")
'''


# ---------------------------------------------------------------------------------------
# 01 — Section 5 : la cible et sa structure
# ---------------------------------------------------------------------------------------
def structure_cells(context: NotebookContext) -> list[NotebookNode]:
    """Build the target-structure cells of notebook 01 for a **multiclass** target.

    Args:
        context: Notebook context.

    Returns:
        The cells to insert in notebook 01 (section 5).
    """
    return [
        _md(
            """## 5. La cible : six modes de défaillance

Une cible multi-classes ne se résume pas à un taux : il faut savoir **combien** pèse chaque mode
(le déséquilibre fixe la métrique), **ce qui les sépare** (quelles combinaisons de capteurs), et
**pourquoi la règle actuelle se trompe** (c'est la valeur que le modèle doit apporter).
"""
        ),
        _code(
            """
target = CONFIG.data.target
order = [label for label in CONFIG.model_dump()["diagnosis"]["class_order"] if label in set(raw[target])]
counts = raw[target].value_counts().reindex(order)
shares = counts / counts.sum()

fig, axis = plt.subplots(figsize=(8.6, 3.4))
axis.bar(order, counts, color="#0a9396", edgecolor="white")
for position, (count, share) in enumerate(zip(counts, shares, strict=True)):
    axis.text(position, count, f"{share:.1%}", ha="center", va="bottom", fontsize=8)
axis.set_title(f"Répartition de `{target}` ({len(raw)} alarmes)")
axis.set_ylabel("alarmes")
axis.tick_params(axis="x", rotation=15)
fig.tight_layout()
plt.show()

majority_f1 = shares.max() / (1 + shares.max()) * 2 / len(order)
print(f"rapport majoritaire / minoritaire : {counts.max() / counts.min():.1f}")
print(f"accuracy de la classe majoritaire : {shares.max():.1%}")
print(f"macro-F1 de la classe majoritaire : {majority_f1:.3f}")
""",
            context,
        ),
        _insight(
            [
                "Six classes de 30 % à 6 % : répondre toujours « fausse alarme » donne 30 % d'accuracy mais un macro-F1 proche de 0,08. C'est pour cela que le **macro-F1** pilote ce projet : chaque mode compte autant, quelle que soit sa fréquence.",
                "Le mode le plus rare (`random_failure`) ne pèse que quelques centaines d'alarmes : ses métriques seront les plus bruitées, et le split stratifié est obligatoire pour qu'il existe dans chaque partie.",
                "La métrique n'est pas un détail technique : un modèle réglé sur l'accuracy apprendrait à ignorer les modes rares, qui sont justement ceux qui coûtent cher à rater.",
            ]
        ),
        _code(
            """
# Ce que dit le code automate, confronté au mode réellement constaté (part de chaque code).
alarm_column = CONFIG.model_dump()["diagnosis"]["alarm_code_column"]
routing = pd.crosstab(raw[alarm_column], raw[target], normalize="index").reindex(columns=order).round(3)
rule = CONFIG.model_dump()["diagnosis"]["alarm_code_rule"]
routing.insert(0, "mode visé par la règle", [rule.get(code, "-") for code in routing.index])
routing["exact"] = [routing.loc[code, rule[code]] if rule.get(code) in routing.columns else float("nan") for code in routing.index]
routing
""",
            context,
        ),
        _insight(
            [
                "La colonne `exact` est la part des alarmes d'un code pour lesquelles la règle envoie la bonne équipe : c'est la **précision de la règle actuelle**, code par code.",
                "Le code générique `E901` mélange fausses alarmes et vraies pannes, dont toutes les défaillances aléatoires : la règle les acquitte toutes, ce qui est l'erreur la plus chère du métier.",
                "Les règles de l'automate ignorent la gamme de produit : le seuil de surcharge réel dépend pourtant de la gamme (L/M/H), d'où des surcharges mal routées sur les pièces haute précision.",
            ]
        ),
        _code(
            """
# Les modes ne se séparent pas sur un capteur, mais sur des combinaisons physiques.
frame = raw.dropna(subset=["torque_nm"]).sample(n=min(2500, len(raw)), random_state=CONFIG.seed)
gap = frame["process_temperature_k"] - frame["air_temperature_k"]
power = frame["torque_nm"] * frame["rotational_speed_rpm"] * 2 * np.pi / 60
strain = frame["tool_wear_min"] * frame["torque_nm"]
palette = dict(zip(order, plt.cm.tab10.colors, strict=False))
colours = frame[target].map(palette)

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
axes[0].scatter(frame["rotational_speed_rpm"], gap, c=colours, s=6, alpha=0.6)
axes[0].axhline(8.6, color="#9b2226", linestyle="--", linewidth=1)
axes[0].axvline(1380, color="#9b2226", linestyle="--", linewidth=1)
axes[0].set_xlabel("vitesse (tr/min)")
axes[0].set_ylabel("écart process - air (K)")
axes[0].set_title("Dissipation thermique : écart < 8,6 K à basse vitesse")
axes[1].scatter(frame["rotational_speed_rpm"], power, c=colours, s=6, alpha=0.6)
axes[1].axhline(3500, color="#9b2226", linestyle="--", linewidth=1)
axes[1].axhline(9000, color="#9b2226", linestyle="--", linewidth=1)
axes[1].set_xlabel("vitesse (tr/min)")
axes[1].set_ylabel("puissance ≈ couple x vitesse (W)")
axes[1].set_title("Défaut de puissance : hors [3 500, 9 000] W")
axes[2].scatter(frame["tool_wear_min"], strain, c=colours, s=6, alpha=0.6)
axes[2].axvline(200, color="#9b2226", linestyle="--", linewidth=1)
for quality, threshold in {"L": 11000, "M": 12000, "H": 13000}.items():
    axes[2].axhline(threshold, color="#495057", linestyle=":", linewidth=1)
    axes[2].text(5, threshold, f"seuil {quality}", fontsize=7, va="bottom")
axes[2].set_xlabel("usure outil (min)")
axes[2].set_ylabel("usure x couple (min.N.m)")
axes[2].set_title("Usure (> 200 min) et surcharge (seuil par gamme)")
handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=colour, label=label) for label, colour in palette.items()]
fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout()
plt.show()
""",
            context,
        ),
        _insight(
            [
                "Chaque mode occupe une **zone** délimitée par un seuil physique sur une combinaison de capteurs — jamais sur un capteur seul. C'est ce qui justifie les trois features déclarées dans `conf/preprocessing/default.yaml` (écart de température, puissance, usure x couple).",
                "Les zones se chevauchent : un outil usé sous fort couple est à la fois dans la zone « usure » et dans la zone « surcharge ». Ces chevauchements annoncent les confusions de la matrice du notebook 06.",
                "Les défaillances aléatoires sont dispersées partout : aucune zone ne les annonce. C'est la signature d'un plafond structurel, qu'aucun modèle ne franchira.",
            ]
        ),
        _code(
            """
# Profil médian des capteurs par mode : la lecture tabulaire des mêmes zones.
sensors = [column for column in numeric_columns if column in raw.columns]
profile = raw.groupby(target)[sensors].median().reindex(order).round(1)
profile["alarmes"] = counts
profile
""",
            context,
        ),
        _insight(
            [
                "Une médiane très différente d'un mode à l'autre (usure pour `tool_wear`, couple pour `overstrain`) confirme un signal fort ; des médianes proches (`false_alarm` et `random_failure`) annoncent une confusion inévitable.",
                "La vibration monte avec l'usure et l'effort : c'est un signal faible mais réel, et le seul capteur absent d'une partie du parc (imputation au notebook 03).",
                "Ce tableau est aussi un contrôle de cohérence métier : un mode dont le profil contredirait la physique signalerait un problème d'étiquetage, pas un problème de modèle.",
            ]
        ),
    ]


# ---------------------------------------------------------------------------------------
# 04 — Exploration des modèles
# ---------------------------------------------------------------------------------------
def build_04_model_exploration(context: NotebookContext, destination: Path) -> Path:
    """Build ``04_model_exploration.ipynb`` for a multiclass project.

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

Avant de comparer des algorithmes, il faut savoir **où se situe le jeu** : un plancher (la classe
majoritaire), la règle que le métier applique déjà (le code automate) et le plafond atteignable
(l'oracle du générateur). Un macro-F1 de 0,65 ne veut rien dire seul ; à 0,01 du plafond, il veut
dire que le signal est épuisé et que la suite se joue sur la **décision**, pas sur le modèle.
"""
        ),
        _objectives(
            context,
            [
                "Installer plancher, règle métier et plafond **avant** toute comparaison d'algorithmes.",
                "Comparer les algorithmes de la stack à protocole identique, sur le macro-F1 **et** sur la calibration.",
                "Mesurer l'apport des features physiques déclarées en configuration.",
                "Vérifier ce que la pondération de classes change à la décision à coût minimal.",
                "Borner ce que la dispersion entre graines autorise à conclure.",
            ],
        ),
        _code(SETUP, context),
        _code(LOAD_RAW, context),
        _code(PREPARE, context),
        _md("## 1. Plancher, règle actuelle et plafond"),
        _code(REFERENCES, context),
        _insight(
            [
                "Le plafond oracle n'est pas 1 : le mode est **tiré** de probabilités physiques (bruit irréductible) et `random_failure` n'a aucun signal. Un modèle qui dépasserait ce plafond aurait une fuite.",
                "L'écart entre la règle actuelle et le plafond est la **marge de valeur** du projet : c'est elle que le modèle doit parcourir.",
                "Toutes ces références sont calculées sur les mêmes alarmes que le reste du notebook.",
            ]
        ),
        _code(VALIDATION_TOOLS, context),
        _md("## 2. Le modèle configuré, sur la validation"),
        _code(
            """
BASE_MODEL = fit_candidate(CONFIG_FAST, PREPARED)
BASE_SCORES = score_on_validation(BASE_MODEL, PREPARED)
position = (BASE_SCORES["macro_f1"] - GENERATION["f1_macro_baseline_alarm_code"]) / (
    GENERATION["f1_macro_ceiling_oracle"] - GENERATION["f1_macro_baseline_alarm_code"]
)
print(f"part du chemin règle -> plafond parcourue : {position:.0%}")
pd.Series(BASE_SCORES, name="__ALGO__").to_frame().round(4)
""",
            context,
        ),
        _insight(
            [
                "La part du chemin parcourue entre la règle et le plafond est la seule lecture honnête d'un macro-F1 : proche de 100 %, le signal est épuisé.",
                "Le coût à la décision minimale est très inférieur au coût de l'argmax : la décision compte ici autant que le modèle — c'est l'objet du notebook 06.",
                "La validation sert à choisir ; le test n'est ouvert qu'une fois, au notebook 06.",
            ]
        ),
        _md("## 3. Les algorithmes de la stack, à protocole identique"),
        _code(
            """
import time

from src.models.factory import available_algorithms

rows = []
for algorithm in available_algorithms(CONFIG.metrics.task):
    started = time.perf_counter()
    try:
        candidate = fit_candidate(CONFIG_FAST, PREPARED, algorithm=algorithm, params={})
        scores = score_on_validation(candidate, PREPARED)
    except Exception as error:  # un algorithme incompatible ne doit pas casser l'exploration
        print(f"  ! {algorithm} ignoré : {type(error).__name__}: {error}")
        continue
    rows.append({"algorithme": algorithm, **scores, "secondes": round(time.perf_counter() - started, 1)})

COMPARISON = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
COMPARISON.round(4)
""",
            context,
        ),
        _insight(
            [
                "Chaque algorithme est évalué **à réglages par défaut** : le classement dit quelle famille capte le signal, pas laquelle est la mieux réglée.",
                "Regarder l'ECE et le log loss **à côté** du macro-F1 : deux modèles de même macro-F1 ne se valent pas si l'un est sur-confiant, parce que la décision à coût minimal multiplie ses probabilités par des euros.",
                "Des écarts de quelques millièmes entre les meilleurs algorithmes sont sous la dispersion entre graines (section 7) : ils ne justifient aucun choix.",
            ]
        ),
        _md("## 4. Ce que rapportent les features physiques"),
        _code(
            """
_without = CONFIG_FAST.model_dump()
_without["preprocessing"]["features"] = []
CONFIG_NO_PHYSICS = validate_config(_without)
PREPARED_NO_PHYSICS = prepare_matrices(raw, CONFIG_NO_PHYSICS)

rows = []
for algorithm in [name for name in (LINEAR, "__ALGO__") if name]:
    for label, config, prepared in (
        ("capteurs bruts", CONFIG_NO_PHYSICS, PREPARED_NO_PHYSICS),
        ("+ features physiques", CONFIG_FAST, PREPARED),
    ):
        candidate = fit_candidate(config, prepared, algorithm=algorithm)
        scores = score_on_validation(candidate, prepared)
        rows.append({"algorithme": algorithm, "features": label, "macro_f1": scores["macro_f1"], "ece": scores["ece"]})
pd.DataFrame(rows).pivot(index="algorithme", columns="features", values="macro_f1").round(4)
""",
            context,
        ),
        _insight(
            [
                "Un modèle linéaire ne sait pas fabriquer un produit couple x vitesse : lui fournir les features physiques lui rend une grande part de son retard sur le boosting.",
                "Les arbres trouvent une partie de ces combinaisons seuls, mais au prix de nombreux découpages : les features déclarées les rendent directement accessibles.",
                "Ces features sont des **recettes de configuration** (`conf/preprocessing/default.yaml`), apprises sur le train : aucune ligne de code métier n'a été écrite pour les obtenir.",
            ]
        ),
        _md("## 5. Pondérer les classes : utile à l'argmax, inutile à la décision"),
        _code(
            """
# `class_weight: auto` est la forme commune aux stacks qui acceptent une pondération : chaque
# stack la traduit (« balanced » en scikit-learn et LightGBM, poids par classe dans la perte des
# réseaux). Une stack qui ne l'accepte pas la filtre : la section le signale au lieu de comparer
# deux modèles identiques.
rows = []
for label, extra_params in (("sans pondération", {}), ("class_weight=auto", {"class_weight": "auto"})):
    params = {**dict(CONFIG.model.params), **extra_params}
    candidate = fit_candidate(CONFIG_FAST, PREPARED, params=params)
    if extra_params and "class_weight" not in dict(candidate.params):
        print("Cette stack n'accepte pas `class_weight` : il faudrait des poids d'échantillons.")
    scores = score_on_validation(candidate, PREPARED)
    truth = np.asarray(PREPARED["y_val"]).astype(str)
    predicted = np.asarray(LABELS)[business_order(candidate, candidate.predict_proba(PREPARED["X_val"])).argmax(axis=1)]
    rare = [label_ for label_ in LABELS if label_ not in SETTINGS.structural_classes][-1]
    rows.append({"variante": label, **scores, f"rappel {rare}": float(np.mean(predicted[truth == rare] == rare))})
pd.DataFrame(rows).set_index("variante").round(4)
""",
            context,
        ),
        _insight(
            [
                "La pondération relève le rappel des modes rares **à l'argmax** : elle déplace les probabilités vers ces modes.",
                "Mais elle **déforme** ces probabilités (ECE en hausse) et n'améliore pas le coût à la décision minimale : l'asymétrie des erreurs est déjà portée par la matrice de coûts.",
                "Règle pratique : pondérer quand la décision est l'argmax ; laisser les probabilités fidèles quand la décision passe par des coûts. Ce projet est dans le second cas.",
            ]
        ),
    ]
    grid = dict(spec.extras.get("notebook_param_grid") or {})
    if grid:
        cells += [
            _md("## 6. Sensibilité aux hyperparamètres"),
            _code(
                """
import itertools

GRID = __PARAM_GRID__
rows = []
for combination in itertools.product(*GRID.values()):
    overrides = dict(zip(GRID, combination, strict=True))
    candidate = fit_candidate(CONFIG_FAST, PREPARED, params={**dict(CONFIG.model.params), **overrides})
    scores = score_on_validation(candidate, PREPARED)
    rows.append({**{key: str(value) for key, value in overrides.items()}, "macro_f1": scores["macro_f1"], "ece": scores["ece"], "log_loss": scores["log_loss"]})
pd.DataFrame(rows).sort_values("macro_f1", ascending=False).round(4)
""",
                context,
            ),
            _insight(
                [
                    "Plus d'itérations ou plus de feuilles n'améliorent pas le macro-F1 quand le signal tient en quelques seuils : le modèle mémorise le bruit et devient sur-confiant (ECE, log loss).",
                    "Le point retenu (`conf/model/default.yaml`) est le plus simple dont le macro-F1 est au niveau du meilleur **et** dont la calibration est la plus fiable.",
                ]
            ),
        ]
    cells += [
        _md(f"## {7 if grid else 6}. Dispersion entre graines : ce que les écarts veulent dire"),
        _code(
            """
rows = []
for seed in (7, 42, 123, 2024, 999):
    _payload = CONFIG_FAST.model_dump()
    _payload["seed"] = seed
    config_seed = validate_config(_payload)
    prepared_seed = prepare_matrices(raw, config_seed)
    for algorithm in [name for name in (LINEAR, "__ALGO__") if name]:
        candidate = fit_candidate(config_seed, prepared_seed, algorithm=algorithm)
        rows.append({"graine": seed, "algorithme": algorithm, "macro_f1": score_on_validation(candidate, prepared_seed)["macro_f1"]})
SEEDS = pd.DataFrame(rows)
SEEDS.groupby("algorithme")["macro_f1"].agg(["mean", "std", "min", "max"]).round(4)
""",
            context,
        ),
        _insight(
            [
                "L'écart-type entre graines est l'unité de mesure des différences : un gain inférieur à cet écart-type n'est pas un gain.",
                "Un écart entre familles (linéaire contre boosting) bien supérieur à la dispersion est, lui, une conclusion solide.",
                "Le split de validation ne compte que quelques dizaines d'alarmes du mode le plus rare : sa métrique est la plus instable, d'où l'intérêt de plusieurs graines.",
            ]
        ),
        _md(
            f"""## Synthèse — choix du modèle

- Algorithme retenu : **`{spec.model.algorithm}`** ({spec.model.display_name}).
- Justification : {spec.model.rationale}
- Alternatives évaluées : {", ".join(f"`{name.split(' : ')[0]}`" for name in spec.model.alternatives) or "voir le tableau de comparaison"}.

**Règle de décision** : le modèle le plus simple dont le macro-F1 est au niveau du meilleur à la
dispersion près, **et** dont les probabilités sont assez fidèles pour la décision à coût minimal.

**Suite** : `05_training.ipynb` entraîne ce modèle dans les conditions de production ;
`06_error_analysis.ipynb` l'évalue une seule fois sur le test et construit la décision.
"""
        ),
    ]
    return write_notebook(destination / "04_model_exploration.ipynb", cells)


# ---------------------------------------------------------------------------------------
# 06 — Analyse d'erreurs et décision
# ---------------------------------------------------------------------------------------
def build_06_error_analysis(context: NotebookContext, destination: Path) -> Path:
    """Build ``06_error_analysis.ipynb`` for a multiclass project.

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
            f"""# 06 — Analyse d'erreurs, décision et recommandations

**Projet** : {spec.title}
**Principe** : un classifieur multi-classes ne décide rien par lui-même. L'argmax dit *quel mode
est le plus probable* ; le métier veut savoir *quelle équipe envoyer*, et toutes les erreurs n'ont
pas le même prix. Ce notebook part du verdict de l'évaluateur de production, descend jusqu'aux
confusions et aux alarmes mal diagnostiquées, puis construit la décision : coût minimal et revue
experte des cas incertains.

Le split de **test** n'est utilisé qu'ici — une seule fois — pour rester une estimation honnête.
"""
        ),
        _objectives(
            context,
            [
                "Lire le verdict contractuel de l'évaluateur de production, objectif par objectif.",
                "Situer le modèle entre la règle actuelle et le plafond atteignable.",
                "Lire la matrice de confusion comme une carte des causes physiques d'erreur.",
                "Remplacer l'argmax par une décision à coût minimal et chiffrer le gain.",
                "Choisir un seuil de revue experte sur la courbe couverture / coût.",
                "Reconnaître un plafond structurel et le documenter au lieu de le masquer.",
            ],
        ),
        _code(SETUP, context),
        _code(LOAD_RAW, context),
        _code(PREPARE, context),
        _code(FIT_MODEL, context),
        _code(REFERENCES, context),
        _md("## 1. Le verdict de l'évaluateur de production"),
        _code(
            """
from src.evaluation.evaluator import Evaluator

EVALUATOR = Evaluator.from_config(MODEL, CONFIG.model_dump(), NB_PATHS)
RESULT = EVALUATOR.evaluate(
    PREPARED["X_test"], PREPARED["y_test"], split="test", context=PREPARED["enriched"]["test"]
)
verdict = "conforme" if RESULT.is_compliant else "non conforme"
print(f"verdict : {verdict} ({RESULT.extras['objectives_met']} objectifs atteints)")
RESULT.verdict
""",
            context,
        ),
        _insight(
            [
                f"La métrique de décision est `{spec.metrics.primary}` = **{RESULT_PLACEHOLDER}** ; le verdict la complète par six objectifs métier (règle battue, rappel par mode, pannes acquittées, calibration, coût, latence).",
                "Un objectif « n/a » n'est pas un succès : il signale une donnée manquante (par exemple le code automate absent du contexte).",
                "Ce verdict est exactement celui du rapport `artifacts/reports/evaluation_report.md` : même objet, mêmes seuils (`diagnosis.objectives`).",
            ]
        ),
        _md("## 2. Le modèle entre la règle actuelle et le plafond"),
        _code(
            """
from src.visualization.plots import MulticlassPlots

PLOTS = MulticlassPlots(NB_PATHS.figures_dir)
display(RESULT.references.round(4))
display(Image(PLOTS.references(RESULT), width=560))
""",
            context,
        ),
        _insight(
            [
                "La référence qui compte est la règle actuelle, pas la classe majoritaire : battre un plancher trivial ne prouve rien au métier.",
                "Le plafond oracle est calculé sur le jeu complet, le modèle sur le test : un modèle légèrement au-dessus du plafond est dans le bruit d'échantillonnage, pas au-delà du possible.",
                "Une fois le plafond atteint, plus aucun réglage ne rapporte : la valeur restante est dans la décision (section 4) et dans de nouvelles données (capteurs, historique).",
            ]
        ),
        _md("## 3. Qualité par mode et confusions"),
        _code(
            """
display(RESULT.per_class.round(4))
display(Image(PLOTS.confusion_matrix(RESULT), width=560))
RESULT.confusions.head(8)
""",
            context,
        ),
        _code(
            """
# Lecture métier des paires confondues, déclarée dans `diagnosis.confusion_notes`.
notes = {
    frozenset(part.strip() for part in key.split("/")): note
    for key, note in (CONFIG.model_dump()["diagnosis"].get("confusion_notes") or {}).items()
}
for _, row in RESULT.confusions.head(6).iterrows():
    pair = frozenset({row["classe réelle"], row["classe prédite"]})
    reading = notes.get(pair, "pas de lecture documentée")
    print(f"- {row['classe réelle']} -> {row['classe prédite']} ({row['effectif']} alarmes)")
    print(f"  {reading}")
""",
            context,
        ),
        _insight(
            [
                "Une matrice de confusion se lit **ligne par ligne** : chaque ligne dit où partent les alarmes d'un mode réel. Les cellules hors diagonale sont des causes, pas des chiffres.",
                "Les confusions les plus fréquentes suivent les chevauchements physiques vus au notebook 01 (usure et surcharge, thermique et puissance) : le modèle se trompe là où la physique est ambiguë.",
                "Une confusion vers `false_alarm` est d'une autre nature qu'une confusion entre deux pannes : la première laisse la machine tourner, la seconde envoie seulement la mauvaise équipe.",
            ]
        ),
        _md("## 4. Argmax ou décision à coût minimal"),
        _code(
            """
from src.evaluation.decision import DiagnosisSettings

SETTINGS = DiagnosisSettings.resolve(CONFIG.model_dump(), classes=list(MODEL.classes_))
costs = pd.DataFrame(SETTINGS.cost_matrix(RESULT.labels), index=RESULT.labels, columns=RESULT.labels)
print("Matrice de coûts (EUR, vraie classe en ligne, décision en colonne)")
display(costs)
display(RESULT.decision.round(4))
display(Image(PLOTS.decision_costs(RESULT), width=600))
""",
            context,
        ),
        _code(
            """
# Des alarmes où la décision à coût minimal contredit l'argmax : pourquoi ?
frame = RESULT.predictions
changed = frame[frame["decision"] != frame["y_pred"]]
print(f"{len(changed)} alarmes ({len(changed) / len(frame):.1%}) où la décision diffère de l'argmax")
columns = ["y_true", "y_pred", "decision", "confidence", *[f"proba_{label}" for label in RESULT.labels]]
changed[columns].head(8).round(3)
""",
            context,
        ),
        _insight(
            [
                "La décision à coût minimal change la réponse quand le mode le plus probable est « fausse alarme » mais qu'une panne reste plausible : une probabilité de panne de 40 % suffit à justifier un déplacement, parce qu'acquitter une panne coûte vingt fois plus cher.",
                "Le prix de cette prudence est visible dans le tableau : quelques déplacements inutiles de plus, beaucoup moins de pannes acquittées, et un coût moyen par alarme nettement inférieur.",
                "La matrice de coûts est un **contrat métier** (`diagnosis.costs`) : la modifier change la décision sans ré-entraîner le modèle.",
            ]
        ),
        _md("## 5. Revue experte : combien d'alarmes confier à un humain ?"),
        _code(
            """
display(RESULT.abstention.round(4))
display(Image(PLOTS.abstention(RESULT), width=600))
display(Image(PLOTS.confidence_histogram(RESULT), width=600))
best = RESULT.abstention.loc[RESULT.abstention["cost_per_alarm"].idxmin()]
print(f"seuil configuré                    : {SETTINGS.review_threshold:.2f}")
print(f"seuil le moins coûteux sur le test : {best['threshold']:.2f}")
""",
            context,
        ),
        _insight(
            [
                "Les diagnostics erronés se concentrent aux faibles confiances : c'est ce qui rend la revue experte rentable — on relit peu d'alarmes pour éviter beaucoup d'erreurs.",
                "Le seuil se choisit sur le **coût**, pas sur l'exactitude : relire coûte aussi (`diagnosis.costs.expert_review`), et au-delà d'un certain seuil on paie des revues pour des cas faciles.",
                "Le seuil le moins coûteux du test est une information, pas une décision : il se confirme sur la validation avant de modifier `diagnosis.review_threshold`.",
            ]
        ),
        _md("## 6. Les probabilités sont-elles fidèles ?"),
        _code(
            """
display(RESULT.calibration)
display(Image(PLOTS.reliability(RESULT), width=520))
print(f"ECE (erreur de calibration top-label) : {RESULT.ece:.3f}")
""",
            context,
        ),
        _insight(
            [
                "Une courbe sous la diagonale signale un modèle **sur-confiant** : il annonce 90 % et se trompe plus d'une fois sur dix. La décision à coût minimal hériterait de cette erreur.",
                "Un boosting profond est typiquement sur-confiant ; le modèle livré est volontairement petit pour garder des probabilités fidèles (notebook 04, section 6).",
                "Si l'ECE dérive en production, le correctif est un recalibrage (isotonic sur une fenêtre récente), pas un ré-entraînement complet.",
            ]
        ),
        _md("## 7. Le plafond structurel : `random_failure`"),
        _code(
            """
structural = list(SETTINGS.structural_classes)
frame = RESULT.predictions
for label in structural:
    rows = frame[frame["y_true"] == label]
    if rows.empty:
        continue
    print(f"--- {label} : {len(rows)} alarmes de test")
    oracle_recall = GENERATION["recall_per_class_oracle"].get(label, float("nan"))
    print(f"rappel du modèle   : {float((rows['y_pred'] == label).mean()):.3f}")
    print(f"rappel de l'oracle : {oracle_recall:.3f}")
    display(rows["y_pred"].value_counts(normalize=True).round(3).to_frame("part des prédictions"))
    display(rows["decision"].value_counts(normalize=True).round(3).to_frame("part des décisions"))
""",
            context,
        ),
        _insight(
            [
                "Le rappel de l'oracle sur ce mode est nul : **aucun** modèle ne peut faire mieux, puisque rien d'observable ne l'annonce. Ce n'est pas un défaut de réglage à corriger mais une limite à documenter.",
                "La décision à coût minimal n'acquitte pourtant pas ces alarmes : elle envoie une équipe sur les cas incertains, ce qui limite la casse même sans diagnostic exact.",
                "Le levier est ailleurs : maintenance préventive, nouveaux capteurs, ou historique de la machine. Le modèle n'est pas le bon outil pour ce mode.",
            ]
        ),
        _md("## 8. Où se concentrent les erreurs ?"),
        _code(
            """
frame = RESULT.predictions
for column in [name for name in ("product_quality", "alarm_code", "shift", "production_line") if name in frame.columns]:
    table = (
        frame.groupby(column)
        .agg(alarmes=("is_error", "size"), taux_erreur=("is_error", "mean"), confiance=("confidence", "mean"))
        .sort_values("taux_erreur", ascending=False)
    )
    print(f"--- erreurs par `{column}`")
    display(table.round(3))
print("--- erreurs les plus confiantes (elles échappent à la revue experte)")
RESULT.errors.head(10)
""",
            context,
        ),
        _insight(
            [
                "Un segment au taux d'erreur nettement supérieur (une gamme, un code automate) est un candidat à une feature dédiée ou à une revue ciblée.",
                "Les erreurs confiantes sont les plus dangereuses : elles ne partent pas en revue. Les relire une à une est le meilleur moyen de découvrir une variable manquante.",
                "Une erreur isolée n'est pas un bug ; un groupe d'erreurs de même nature sur le même segment en est un.",
            ]
        ),
        _md("## 9. Rapport exécutable"),
        _code(
            """
from src.evaluation.reports import ReportBuilder

REPORTER = ReportBuilder(NB_PATHS, config=CONFIG.model_dump())
WRITTEN = REPORTER.build(RESULT, model=MODEL)
print(f"{len(WRITTEN)} artefacts écrits dans {NB_PATHS.artifacts_dir.relative_to(PROJECT_ROOT)}")
for index, recommendation in enumerate(REPORTER.recommendations(RESULT), start=1):
    print(f"{index:2d}. {recommendation}")
""",
            context,
        ),
        _md(
            f"""## 10. Recommandations concrètes

### issues de l'analyse (calculées ci-dessus)

Elles dépendent des mesures : décision à coût minimal, seuil de revue, recalibrage, mode le moins
bien rappelé, confusion la plus fréquente.

### documentées pour ce cas d'usage

{recommendation_lines or "1. Rejouer l'évaluation sur un échantillon plus large avant toute décision."}

### plan d'action proposé

| Priorité | Action | Effet attendu | Comment vérifier |
| --- | --- | --- | --- |
| 1 | Router sur la décision à coût minimal | coût par alarme en baisse, pannes acquittées quasi nulles | §4 rejoué sur le test |
| 2 | Revue experte sous le seuil de confiance | erreurs confiantes seules restantes | courbe d'abstention (§5) |
| 3 | Surveiller l'ECE et la répartition des modes | alerte avant la dérive de la décision | `mlops/model-monitoring` |
| 4 | Instrumenter les machines sans capteur de vibration | meilleure séparation usure / surcharge | notebook 04, ablation |
| 5 | Rejouer ce notebook à chaque nouvelle version de données | non-régression | `make evaluate` + CI |

## 11. Limites assumées

- Les données sont **synthétiques** : la physique est plausible, les niveaux de performance illustrent une méthode.
- Les coûts de `diagnosis.costs` sont indicatifs : seul leur ordre de grandeur relatif est défendable sans données financières réelles.
- `random_failure` n'est pas diagnosticable par construction ; tout gain affiché sur ce mode serait du bruit.
- Une seule passe d'évaluation sur le test : la dispersion est mesurée au notebook 04 (graines), pas ici.
"""
        ),
    ]
    result_path = write_notebook(destination / "06_error_analysis.ipynb", cells)
    _replace_placeholder(result_path)
    return result_path
