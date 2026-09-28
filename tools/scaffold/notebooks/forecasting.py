"""Cellules de notebooks spécifiques à la tâche **forecasting** (prévision de série temporelle).

Le squelette des six notebooks (``notebooks/tabular.py``) est partagé par toutes les tâches
tabulaires ; une prévision ajoute trois questions que ce squelette ne pose pas, et ce module les
traite :

* **quand sait-on quoi ?** — une prévision se juge d'abord à son *contrat d'antériorité*. Une
  variable calculée sur le jour cible avec la vérité du jour cible n'est pas une feature, c'est une
  fuite : le MAPE s'effondre et le modèle est inutilisable en production. La section ajoutée à
  ``01_eda.ipynb`` audite ce contrat colonne par colonne, et ``04`` le viole **volontairement** pour
  montrer à quoi ressemble un modèle qui triche ;
* **à quoi se comparer ?** — la référence n'est pas « zéro erreur » mais le **naif saisonnier** que
  l'opérateur produit déjà à la main. Tout le notebook 04 est construit autour de ce plancher ;
* **publier quoi ?** — un nombre seul ne se pilote pas. Le notebook 06 mesure la couverture réelle
  des intervalles, la ventile entre jours calmes et régimes extrêmes, et teste le recalibrage
  adaptatif que le rapport d'évaluation recommande.

Ce module fournit :

* :func:`structure_cells` — la section « structure temporelle » de ``01_eda.ipynb`` : la série
  quotidienne, la thermo-sensibilité asymétrique, les régimes, l'audit d'antériorité, les références
  naïves et la structure (origine, horizon) du panel ;
* :func:`build_04_model_exploration` — plancher naïf, comparaison des familles d'algorithmes,
  thermo-sensibilité apprise par le modèle linéaire, stratégie multi-horizons, grille
  d'hyperparamètres, backtest avec ré-entraînement et sonde de fuite ;
* :func:`build_05_training` — validation chronologique contre validation aléatoire, choix de la
  perte, écart d'apprentissage, artefacts persistés et coût de ré-entraînement ;
* :func:`build_06_error_analysis` — erreur par horizon, par régime, par saison, pires journées,
  autocorrélation des résidus, couverture des intervalles (statique contre adaptative), facteurs
  contributifs et recommandations chiffrées.

Les helpers de rendu (``_code``, ``_md``, ``_insight``, jetons ``__XXX__``) sont réutilisés tels
quels : un notebook de prévision et un notebook de régression partagent la même grammaire.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tools.scaffold.notebooks.tabular import (
    LOAD_RAW,
    NB_ROWS_FORECAST,
    PREPARE,
    RESULT_PLACEHOLDER,
    SETUP,
    _code,
    _insight,
    _md,
    _objectives,
)
from tools.scaffold.utils_notebooks import NotebookNode, write_notebook

if TYPE_CHECKING:  # pragma: no cover
    from tools.scaffold.utils_notebooks import NotebookContext

__all__ = [
    "NB_ROWS_FORECAST",
    "build_04_model_exploration",
    "build_05_training",
    "build_06_error_analysis",
    "structure_cells",
]

#: Bloc d'initialisation commun aux notebooks 04, 05 et 06 : noms de colonnes lus dans la
#: configuration résolue (jamais codés en dur) et petites fonctions de mesure réutilisées.
FORECAST_HELPERS = '''
# --- Contrat de prévision : tout est lu dans la configuration, rien n'est codé en dur -----------
FORECAST_CONF = dict(CONFIG.model_dump().get("load_forecasting") or {})
HORIZON_COLUMN = str(FORECAST_CONF.get("horizon_column") or "horizon_days")
HORIZONS = tuple(int(value) for value in (FORECAST_CONF.get("horizons") or ()))
LONG_HORIZON = int(FORECAST_CONF.get("long_horizon") or (max(HORIZONS) if HORIZONS else 7))
INTERVAL_LEVEL = float(FORECAST_CONF.get("interval_level") or 0.90)
INTERVAL_METHOD = str(FORECAST_CONF.get("interval_method") or "normalized_conformal")
INTERVAL_SCALE = str(FORECAST_CONF.get("interval_scale_column") or "load_last_observed")
BACKTEST_FOLDS = int(FORECAST_CONF.get("backtest_folds") or 5)

TARGET = str(CONFIG.data.target)
TIME_COLUMN = str(CONFIG.data.time_column or "origin_date")
TARGET_DATE = "target_date" if "target_date" in raw.columns else TIME_COLUMN
EVENT_COLUMN = "event_type" if "event_type" in raw.columns else None
NAIVE_COLUMN = "load_seasonal_naive" if "load_seasonal_naive" in raw.columns else None
PERSIST_COLUMN = "load_last_observed" if "load_last_observed" in raw.columns else None
TEMP_FORECAST = "temperature_forecast_c" if "temperature_forecast_c" in raw.columns else None


def mape(truth: Any, predicted: Any) -> float:
    """Mean absolute percentage error, in percent, ignoring zero denominators.

    Args:
        truth: Observed values.
        predicted: Forecast values.

    Returns:
        The MAPE in percent (``nan`` when nothing is measurable).
    """
    observed = np.asarray(truth, dtype="float64")
    forecast = np.asarray(predicted, dtype="float64")
    usable = np.isfinite(observed) & np.isfinite(forecast) & (np.abs(observed) > 1e-8)
    if not usable.any():
        return float("nan")
    return float(np.mean(np.abs((observed[usable] - forecast[usable]) / observed[usable])) * 100.0)


def mase(truth: Any, predicted: Any, reference: Any) -> float:
    """Mean absolute scaled error: model error over the naive reference error.

    Args:
        truth: Observed values.
        predicted: Forecast values.
        reference: Naive reference forecast on the same rows.

    Returns:
        The MASE (below 1 means better than the reference).
    """
    observed = np.asarray(truth, dtype="float64")
    forecast = np.asarray(predicted, dtype="float64")
    naive = np.asarray(reference, dtype="float64")
    usable = np.isfinite(observed) & np.isfinite(forecast) & np.isfinite(naive)
    scale = float(np.mean(np.abs(observed[usable] - naive[usable])))
    if not usable.any() or scale < 1e-9:
        return float("nan")
    return float(np.mean(np.abs(observed[usable] - forecast[usable])) / scale)


def daily_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse the (origin, horizon) panel into one row per target day.

    Le panel contient plusieurs lignes par jour cible (une par horizon) qui portent **la même**
    consommation : la série quotidienne se reconstruit en dédupliquant sur la date cible.

    Args:
        frame: Raw panel.

    Returns:
        One row per target day, sorted chronologically.
    """
    unique = frame.drop_duplicates(subset=[TARGET_DATE]).copy()
    unique[TARGET_DATE] = pd.to_datetime(unique[TARGET_DATE])
    return unique.sort_values(TARGET_DATE).reset_index(drop=True)


print(f"contrat de prévision : horizons={HORIZONS} colonne='{HORIZON_COLUMN}' "
      f"intervalle={INTERVAL_METHOD} (niveau {INTERVAL_LEVEL:.0%}, échelle '{INTERVAL_SCALE}')")
print(f"cible='{TARGET}' origine='{TIME_COLUMN}' cible_date='{TARGET_DATE}' "
      f"régimes='{EVENT_COLUMN}' naif='{NAIVE_COLUMN}'")
'''


# ---------------------------------------------------------------------------------------
# 01 — Structure temporelle, thermo-sensibilité, régimes, antériorité, références naïves
# ---------------------------------------------------------------------------------------
_SERIES_CELL = """
# La série quotidienne : trois saisonnalités superposées et une rupture de régime.
daily = daily_series(raw)
daily["doy"] = daily[TARGET_DATE].dt.dayofyear
daily["weekday"] = daily[TARGET_DATE].dt.dayofweek

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
axes[0].plot(daily[TARGET_DATE], daily[TARGET], lw=0.7, color="#1f4e79")
axes[0].set_title(f"Série quotidienne ({len(daily)} jours) — cible `{TARGET}`")
axes[0].set_ylabel("MW")
axes[0].grid(alpha=0.3)

weekly = daily.groupby("weekday")[TARGET].mean()
axes[1].bar(
    ["lun", "mar", "mer", "jeu", "ven", "sam", "dim"],
    weekly.reindex(range(7)).to_numpy(),
    color="#2e75b6",
)
axes[1].set_title("Profil hebdomadaire (moyenne par jour de la semaine)")
axes[1].set_ylabel("MW")
axes[1].grid(alpha=0.3, axis="y")

monthly = daily.groupby(daily[TARGET_DATE].dt.month)[TARGET].mean()
axes[2].plot(monthly.index, monthly.to_numpy(), marker="o", color="#c00000")
axes[2].set_title("Profil annuel (moyenne par mois)")
axes[2].set_ylabel("MW")
axes[2].set_xlabel("mois")
axes[2].grid(alpha=0.3)
fig.tight_layout()
plt.show()

print(f"période couverte : {daily[TARGET_DATE].min().date()} -> {daily[TARGET_DATE].max().date()}")
print(f"nombre d'années civiles : {daily[TARGET_DATE].dt.year.nunique()}")
print(f"niveau : moyenne={daily[TARGET].mean():.0f} MW, médiane={daily[TARGET].median():.0f} MW, "
      f"min={daily[TARGET].min():.0f}, max={daily[TARGET].max():.0f}")
print(f" amplitude semaine : {weekly.max() - weekly.min():.0f} MW "
      f"({100 * (weekly.max() - weekly.min()) / weekly.mean():.1f} % du niveau moyen)")
print(f" amplitude année   : {monthly.max() - monthly.min():.0f} MW "
      f"({100 * (monthly.max() - monthly.min()) / monthly.mean():.1f} % du niveau moyen)")
"""

_THERMO_CELL = """
# Thermo-sensibilité : la relation consommation / température, et son asymétrie.
# La température **vraie** du jour cible n'est volontairement pas dans le jeu (contrat
# d'antériorité) : on travaille avec la prévision de température, qui est ce dont dispose le modèle.
if TEMP_FORECAST is None:
    print("aucune colonne de prévision de température dans ce jeu : section sans objet")
else:
    panel = raw.drop_duplicates(subset=[TARGET_DATE]).copy()
    bins = np.arange(-15, 40, 2.5)
    panel["temp_bin"] = pd.cut(panel[TEMP_FORECAST], bins=bins)
    profile = panel.groupby("temp_bin", observed=True).agg(
        jours=(TARGET, "size"),
        consommation=(TARGET, "mean"),
        ecart_type=(TARGET, "std"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].scatter(panel[TEMP_FORECAST], panel[TARGET], s=4, alpha=0.25, color="#2e75b6")
    centre = profile.index.map(lambda interval: (interval.left + interval.right) / 2)
    axes[0].plot(centre, profile["consommation"], color="#c00000", lw=2, label="moyenne par palier")
    axes[0].set_xlabel(f"prévision de température du jour cible ({TEMP_FORECAST}, °C)")
    axes[0].set_ylabel(f"{TARGET} (MW)")
    axes[0].set_title("Thermo-sensibilité : une courbe, pas une droite")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    heating = panel[panel[TEMP_FORECAST] < 14.0]
    cooling = panel[panel[TEMP_FORECAST] > 22.0]
    if len(heating) > 20 and len(cooling) > 20:
        slope_heat = np.polyfit(heating[TEMP_FORECAST], heating[TARGET], 1)[0]
        slope_cool = np.polyfit(cooling[TEMP_FORECAST], cooling[TARGET], 1)[0]
        axes[1].bar(
            ["chauffe (< 14 °C)", "climatisation (> 22 °C)"],
            [-slope_heat, slope_cool],
            color=["#c00000", "#2e75b6"],
        )
        axes[1].set_ylabel("MW par °C")
        axes[1].set_title("Gradient par régime (positif = la consommation monte)")
        axes[1].grid(alpha=0.3, axis="y")
        print(f"gradient de chauffe        : {slope_heat:7.1f} MW/°C "
              f"({100 * slope_heat / daily[TARGET].mean():+.2f} %/°C) sur {len(heating)} jours")
        print(f"gradient de climatisation  : {slope_cool:7.1f} MW/°C "
              f"({100 * slope_cool / daily[TARGET].mean():+.2f} %/°C) sur {len(cooling)} jours")
        print(f"rapport chauffe / clim     : {abs(slope_heat / slope_cool):.1f}x")
    fig.tight_layout()
    plt.show()
"""

_REGIME_CELL = """
# Régimes exceptionnels : ce qu'ils coûtent, et surtout ce qu'ils coûtent EN ERREUR.
if EVENT_COLUMN is None:
    print("aucune colonne de régime dans ce jeu : section sans objet")
else:
    panel = raw.drop_duplicates(subset=[TARGET_DATE]).copy()
    if NAIVE_COLUMN:
        panel["erreur_naive_pct"] = 100.0 * np.abs(
            (panel[TARGET] - panel[NAIVE_COLUMN]) / panel[TARGET]
        )
    regimes = panel.groupby(EVENT_COLUMN).agg(
        jours=(TARGET, "size"),
        consommation_moyenne=(TARGET, "mean"),
        consommation_max=(TARGET, "max"),
    )
    if NAIVE_COLUMN:
        regimes["erreur_naive_pct"] = panel.groupby(EVENT_COLUMN)["erreur_naive_pct"].mean()
    regimes["part_des_jours_pct"] = 100.0 * regimes["jours"] / regimes["jours"].sum()
    if NAIVE_COLUMN:
        total_error = (panel["erreur_naive_pct"] / 100.0 * panel[TARGET]).groupby(
            panel[EVENT_COLUMN]
        ).sum()
        regimes["part_de_l_erreur_pct"] = 100.0 * total_error / total_error.sum()
    print(regimes.round(2).to_string())
    print()

    episodes = (
        panel.assign(jour=pd.to_datetime(panel[TARGET_DATE]).dt.date)
        .groupby([EVENT_COLUMN, "jour"], as_index=False)
        .size()
        .rename(columns={"size": "horizons"})
    )
    if len(episodes):
        grouped = episodes[episodes[EVENT_COLUMN] != "none"].groupby(EVENT_COLUMN)["jour"].agg(
            jours="count", premier="min", dernier="max"
        )
        print("jours par régime (chaque date compte une fois, tous horizons confondus) :")
        print(grouped.to_string())
"""

_ANTERIORITY_CELL = """
# AUDIT DU CONTRAT D'ANTÉRIORITÉ — la cellule la plus importante de ce notebook.
# Une feature est légale si et seulement si elle est connue à l'origine (date de publication) ou
# connue par avance (calendrier, prévision météo). Toute colonne qui dépend de la vérité du jour
# cible est une fuite : elle rend le MAPE excellent en backtest et le modèle inutilisable en réel.
panel = raw.copy()
panel["origine"] = pd.to_datetime(panel[TIME_COLUMN])
panel["cible"] = pd.to_datetime(panel[TARGET_DATE])
panel["ecart_jours"] = (panel["cible"] - panel["origine"]).dt.days

print("=== 1. cohérence de l'horizon déclaré ===")
print(panel["ecart_jours"].value_counts().sort_index().to_string())
if HORIZON_COLUMN in panel.columns:
    incoherent = int((panel["ecart_jours"] != panel[HORIZON_COLUMN]).sum())
    print(f"lignes où l'écart de dates diffère de `{HORIZON_COLUMN}` : {incoherent}")

print()
print("=== 2. classement des colonnes par disponibilité ===")
known_at_origin, known_in_advance, metadata, suspect = [], [], [], []
for column in panel.columns:
    if column in {"sample_id", TIME_COLUMN, TARGET_DATE, "ecart_jours", "origine", "cible"}:
        metadata.append(column)
    elif column.startswith("target_") or column in {"hdd_target", "cdd_target"}:
        known_in_advance.append(column)
    elif column == TARGET:
        metadata.append(column)
    elif column.startswith(("load_", "temperature_")):
        known_at_origin.append(column)
    else:
        suspect.append(column)
print(f"connues à l'origine (historique)     : {known_at_origin}")
print(f"connues par avance (calendrier/météo): {known_in_advance}")
print(f"métadonnées (jamais features)        : {metadata}")
print(f"à inspecter manuellement             : {suspect}")

print()
print("=== 3. vérification quantitative : les colonnes « historiques » datent-elles d'avant ? ===")
# `load_seasonal_naive` doit être la consommation du même jour de la semaine, UNE SEMAINE avant la
# cible. Si elle coïncide avec la cible, c'est une fuite ; on le mesure plutôt que de le supposer.
if NAIVE_COLUMN:
    daily = daily_series(raw).set_index(TARGET_DATE)[TARGET]
    expected = panel["cible"].map(
        lambda stamp: daily.get(stamp - pd.Timedelta(days=7), np.nan)
    )
    match_naive = float(np.nanmean(np.isclose(panel[NAIVE_COLUMN], expected, rtol=1e-3)))
    match_target = float(np.nanmean(np.isclose(panel[NAIVE_COLUMN], panel[TARGET], rtol=1e-3)))
    print(f"`{NAIVE_COLUMN}` == valeur 7 jours avant la cible : {match_naive:.1%} des lignes")
    print(f"`{NAIVE_COLUMN}` == cible du jour             : {match_target:.1%} des lignes "
          "(doit rester bas : un taux élevé signerait une fuite)")

if PERSIST_COLUMN:
    expected_last = panel["origine"].map(lambda stamp: daily.get(stamp, np.nan))
    match_persist = float(np.nanmean(np.isclose(panel[PERSIST_COLUMN], expected_last, rtol=1e-3)))
    print(f"`{PERSIST_COLUMN}` == valeur au jour d'origine    : {match_persist:.1%} des lignes")

print()
print("=== 4. la prévision météo est-elle entachée de son erreur ? ===")
# Une prévision météo parfaite serait une fuite déguisée : le modèle apprendrait une relation
# déterministe qui n'existe pas en exploitation. On vérifie qu'elle diffère de la réalisation.
if TEMP_FORECAST:
    anomalies = (panel[TEMP_FORECAST] - panel["temperature_anomaly_c"]).abs()
    print(f"écart prévision / normale saisonnière : moyenne={anomalies.mean():.2f} °C")
    print(f"dispersion de la prévision par horizon (°C, écart-type) :")
    print(panel.groupby(HORIZON_COLUMN)[TEMP_FORECAST].std().round(2).to_string())
"""

_NAIVE_CELL = """
# Les références naïves : le seul adversaire qui compte.
# Un modèle de prévision ne se juge pas à son MAPE absolu mais à son gain sur ce qu'un opérateur
# produit déjà sans modèle. Trois références, toutes construites avec l'information disponible à
# l'origine uniquement.
references = {}
if NAIVE_COLUMN:
    references["naif saisonnier (même jour, -7 j)"] = raw[NAIVE_COLUMN]
if PERSIST_COLUMN:
    references["persistance (dernière valeur connue)"] = raw[PERSIST_COLUMN]
for name, column in [("moyenne glissante 7 j", "load_rolling_mean_7d"),
                     ("climatologie 28 j", "load_rolling_mean_28d")]:
    if column in raw.columns:
        references[name] = raw[column]
references["moyenne globale (niveau constant)"] = pd.Series(
    float(raw[TARGET].mean()), index=raw.index
)

truth = raw[TARGET]
rows = []
for name, forecast in references.items():
    rows.append(
        {
            "référence": name,
            "MAPE %": round(mape(truth, forecast), 3),
            "MASE": round(mase(truth, forecast, references.get(
                "naif saisonnier (même jour, -7 j)", forecast)), 3),
            "MAE MW": round(float(np.mean(np.abs(truth - forecast))), 1),
            "biais %": round(float(np.mean((forecast - truth) / truth)) * 100.0, 2),
        }
    )
baseline_table = pd.DataFrame(rows).sort_values("MAPE %").reset_index(drop=True)
print(baseline_table.to_string(index=False))
print()
BEST_NAIVE = float(baseline_table["MAPE %"].iloc[0])
print(f"plancher à battre : {BEST_NAIVE:.2f} % de MAPE")

# Les critères de succès ne sont pas recopiés ici : ils sont lus dans le module de rapport, qui les
# tient lui-même du manifeste. Un seuil dupliqué dans un notebook finit toujours par diverger.
from src.evaluation.reports import DEFAULT_THRESHOLDS  # noqa: E402

print(f"critères déclarés : MAPE <= {DEFAULT_THRESHOLDS['mape_max']:.1f} %, "
      f"gain sur la meilleure référence >= "
      f"{DEFAULT_THRESHOLDS['improvement_vs_naive_min']:.0f} %, "
      f"MASE <= {DEFAULT_THRESHOLDS['mase_max']:.2f}")
print(f"à {BEST_NAIVE:.2f} % de MAPE naïf, le seuil de {DEFAULT_THRESHOLDS['mape_max']:.1f} % "
      f"exige un gain d'au moins "
      f"{100.0 * (1.0 - DEFAULT_THRESHOLDS['mape_max'] / BEST_NAIVE):.0f} %")
"""

_HORIZON_CELL = """
# La structure du panel : ce n'est pas une série, c'est un ensemble de problèmes de prévision.
counts = raw.groupby(HORIZON_COLUMN).size()
print(f"origines distinctes        : {raw[TIME_COLUMN].nunique()}")
print(f"jours cibles distincts     : {raw[TARGET_DATE].nunique()}")
print(f"lignes                     : {len(raw)}")
print(f"horizons publiés           : {sorted(counts.index.tolist())}")
print(counts.rename("lignes").to_string())
print()
overlap = raw.groupby(TARGET_DATE)[TIME_COLUMN].nunique()
print(f"jours cibles vus depuis plusieurs origines : {int((overlap > 1).sum())} "
      f"(sur {len(overlap)})")
print()
# La même consommation est donc connue plusieurs fois, avec une information d'origine différente :
# c'est ce qui permet d'apprendre l'effet de l'horizon sans dupliquer l'information cible.
sample_day = raw[raw[TARGET_DATE] == raw[TARGET_DATE].max()]
columns = [TIME_COLUMN, TARGET_DATE, HORIZON_COLUMN, TARGET]
if PERSIST_COLUMN:
    columns.append(PERSIST_COLUMN)
print("le dernier jour cible, vu depuis chaque origine :")
print(sample_day[columns].sort_values(HORIZON_COLUMN).to_string(index=False))
"""


def structure_cells(context: NotebookContext) -> list[NotebookNode]:
    """Build the temporal-structure section of ``01_eda.ipynb``.

    Args:
        context: Notebook context.

    Returns:
        Markdown, code and insight cells, in execution order.
    """
    spec = context.spec
    cells: list[NotebookNode] = [
        _md(
            """## 5. Structure temporelle : ce qui se prévoit, et avec quoi

Une prévision de consommation n'est pas un problème tabulaire ordinaire : chaque ligne est un
**couple (origine, horizon)** et la cible est la consommation d'un jour **futur**. Trois
conséquences structurent toute l'analyse qui suit :

1. la série porte **plusieurs saisonnalités superposées** (annuelle, hebdomadaire, et une composante
   météo qui n'est pas saisonnière) ;
2. la relation à la température est **asymétrique** — le chauffage pèse beaucoup plus que la
   climatisation dans le parc français ;
3. le jeu contient des **régimes exceptionnels** (vague de froid, canicule, arrêt industriel) qui
   sont rares en jours mais dominants en coût d'erreur.

Et une règle absolue, vérifiée chiffre à l'appui en section 5.4 : **rien dans les features ne doit
dépendre de la vérité du jour cible**."""
        ),
        _md("### 5.1 La série quotidienne et ses saisonnalités"),
        _code(FORECAST_HELPERS + _SERIES_CELL, context),
        _insight(
            [
                "La série est **non stationnaire à trois échelles** : une tendance lente (effacement, "
                "efficacité énergétique), un cycle annuel marqué (chauffage) et un cycle hebdomadaire "
                "(tertiaire fermé le week-end). Un modèle qui ne verrait que le niveau moyen perdrait "
                "l'essentiel.",
                "L'amplitude annuelle est très supérieure à l'amplitude hebdomadaire : c'est la "
                "thermo-sensibilité qui pilote le niveau, le calendrier pilote la forme. Cela fixe "
                "l'ordre des priorités en feature engineering.",
                "Le panel contient plusieurs lignes par jour cible (une par horizon) : la "
                "reconstruction de la série quotidienne passe par une déduplication sur la date "
                "cible, ce que fait `daily_series()`.",
            ]
        ),
        _md(
            """### 5.2 Thermo-sensibilité : une courbe, pas une droite

La température vraie du jour cible **n'est pas dans le jeu** — ce serait une fuite. On travaille
donc avec la prévision de température, entachée de son erreur réelle, qui est exactement ce dont
dispose le modèle le matin de la publication."""
        ),
        _code(_THERMO_CELL, context),
        _insight(
            [
                "La relation est en **deux pentes** : forte et négative sous ~14 °C (chauffage), "
                "faible et positive au-dessus de ~22 °C (climatisation), plate entre les deux. Un "
                "modèle linéaire global la rate ; c'est la première raison d'utiliser des arbres.",
                "Le gradient de chauffe est plusieurs fois le gradient de climatisation : le parc est "
                "peu climatisé. Conséquence opérationnelle — une vague de froid coûte plus cher en "
                "erreur de prévision qu'une canicule de même amplitude.",
                "La dispersion autour de la moyenne par palier reste importante : elle contient "
                "l'effet du jour de la semaine et le bruit irréductible de la série. C'est cette "
                "dispersion qui fixe le plancher de MAPE atteignable.",
            ]
        ),
        _md("### 5.3 Régimes exceptionnels : rares en jours, dominants en erreur"),
        _code(_REGIME_CELL, context),
        _insight(
            [
                "Les jours de régime exceptionnel représentent une faible part des lignes mais une "
                "part **disproportionnée de l'erreur totale** : ce sont eux qui justifient un "
                "indicateur de confiance et une revue humaine, pas le MAPE moyen.",
                "Le naif saisonnier est particulièrement mauvais pendant ces épisodes, parce qu'il "
                "recopie une semaine qui ne ressemblait pas à celle-ci. C'est là que le modèle "
                "appris gagne le plus — et là aussi qu'il peut perdre le plus s'il extrapole.",
                "Un arrêt industriel **baisse** la consommation : le traiter comme une anomalie à "
                "retrancher plutôt qu'à prévoir est une erreur de conception fréquente.",
            ]
        ),
        _md(
            """### 5.4 Audit du contrat d'antériorité

Cette section n'est pas décorative : c'est le test qui décide si le projet est honnête. Une
variable calculée avec la vérité du jour cible donne un MAPE proche de zéro en backtest et un
modèle **inutilisable** en production, puisque cette vérité n'existe pas encore à 6 h du matin."""
        ),
        _code(_ANTERIORITY_CELL, context),
        _insight(
            [
                "Trois familles de colonnes, et une seule règle : **connue à l'origine** (historique "
                "de consommation, température observée), **connue par avance** (calendrier du jour "
                "cible, prévision météo) ou **métadonnée** (la cible elle-même, la date cible, le "
                "type de régime) — cette dernière est exclue de la matrice de features par "
                "`drop_columns`.",
                "La vérification est **quantitative** : le naif saisonnier coïncide avec la valeur de "
                "la semaine précédente, pas avec la cible du jour. Un taux de coïncidence élevé avec "
                "la cible signerait une fuite ; on le mesure au lieu de le supposer.",
                "Les colonnes `target_*` sont du calendrier : légitimes, parce que le calendrier "
                "d'un jour futur est connu. Les degrés-jours `hdd_target`/`cdd_target` sont calculés "
                "sur la **prévision** de température, jamais sur la réalisation — c'est le détail "
                "qui fait la différence entre une feature et une fuite.",
                "La prévision météo diffère de la réalisation, et son erreur croît avec l'horizon. "
                "Une prévision météo parfaite serait elle aussi une fuite déguisée.",
            ]
        ),
        _md("### 5.5 Les références naïves : le plancher à battre"),
        _code(_NAIVE_CELL, context),
        _insight(
            [
                "Le **naif saisonnier** (recopier la consommation du même jour de la semaine "
                "précédente) est la référence métier : c'est ce que fait l'opérateur sans modèle. "
                "Toute la valeur du projet se mesure en gain sur cette référence.",
                "La persistance est mauvaise dès que l'horizon s'allonge ou que le jour de la "
                "semaine change ; la climatologie 28 jours est mauvaise dès que la météo bouge. "
                "Chaque référence échoue sur une dimension différente — le modèle doit les battre "
                "toutes.",
                "Un gain inférieur à ~30 % ne justifie pas la complexité d'exploitation (astreinte, "
                "ré-entraînement, surveillance de dérive). C'est un critère de succès déclaré dans "
                "le manifeste, pas une opinion.",
            ]
        ),
        _md("### 5.6 Le panel (origine, horizon)"),
        _code(_HORIZON_CELL, context),
        _insight(
            [
                "Un même jour cible apparaît plusieurs fois, vu depuis des origines différentes : le "
                "modèle apprend donc **l'effet de l'horizon** comme une variable, au lieu d'entraîner "
                "un modèle par échéance.",
                "Ce choix a un coût (un compromis entre horizons) et un bénéfice (une seule chaîne à "
                "maintenir, plus de données par modèle). La section 3 du notebook 04 mesure les deux.",
                "Le split **chronologique sur l'origine** est obligatoire : couper aléatoirement "
                "placerait dans l'entraînement des lignes dont le jour cible est déjà dans le test, "
                "ce qui est une fuite par recouvrement de fenêtre.",
            ]
        ),
    ]
    del spec
    return cells


# ---------------------------------------------------------------------------------------
# 04 — Plancher naïf, familles d'algorithmes, stratégie multi-horizons, grille, fuite
# ---------------------------------------------------------------------------------------
_FLOOR_CELL = """
# Le plancher, mesuré sur le split de TEST : ce que produirait l'opérateur sans modèle.
# Comparer sur l'entraînement n'aurait aucun sens — le naif saisonnier y est avantagé par
# construction, puisqu'il recopie une semaine déjà vue.
truth = PREPARED["y_test"]
floor = {}
if NAIVE_COLUMN:
    floor["naif saisonnier"] = PREPARED["splits"].test[NAIVE_COLUMN].to_numpy(dtype="float64")
if PERSIST_COLUMN:
    floor["persistance"] = PREPARED["splits"].test[PERSIST_COLUMN].to_numpy(dtype="float64")
for label, column in [("moyenne glissante 7 j", "load_rolling_mean_7d"),
                      ("climatologie 28 j", "load_rolling_mean_28d")]:
    if column in PREPARED["splits"].test.columns:
        floor[label] = PREPARED["splits"].test[column].to_numpy(dtype="float64")

rows = []
for label, forecast in floor.items():
    rows.append({
        "référence": label,
        "MAPE %": round(mape(truth, forecast), 3),
        "MAE MW": round(float(np.mean(np.abs(np.asarray(truth, dtype="float64") - forecast))), 1),
    })
floor_table = pd.DataFrame(rows).sort_values("MAPE %").reset_index(drop=True)
print(floor_table.to_string(index=False))
FLOOR_MAPE = float(floor_table["MAPE %"].iloc[0])
BEST_REFERENCE = floor_table["référence"].iloc[0]
print(f"\\nplancher sur le test : {FLOOR_MAPE:.2f} % ({BEST_REFERENCE})")
"""

_ALGORITHMS_CELL = """
# Comparaison des familles d'algorithmes, à données et protocole identiques.
# Chacun est entraîné avec ses réglages par défaut (`params={}`) : c'est la comparaison loyale. Le
# modèle configuré est aussi mesuré avec ses hyperparamètres réglés, pour séparer « la famille » de
# « le réglage ».
import time  # noqa: E402

from src.models import build_model  # noqa: E402

CONFIGURED = str(CONFIG.model.algorithm)
candidates = ["ridge", "random_forest", "gradient_boosting", "svm", CONFIGURED]
naive_test = (
    PREPARED["splits"].test[NAIVE_COLUMN].to_numpy(dtype="float64") if NAIVE_COLUMN else None
)

rows = []
fitted = {}
for algorithm in candidates:
    try:
        started = time.perf_counter()
        model = build_model(
            CONFIG, feature_names=PREPARED["feature_names"], algorithm=algorithm, params={}
        )
        model.fit(PREPARED["X_train"], PREPARED["y_train"])
        elapsed = time.perf_counter() - started
        forecast = np.asarray(model.predict(PREPARED["X_test"]), dtype="float64").ravel()
        fitted[algorithm] = model
        rows.append({
            "algorithme": algorithm,
            "MAPE test %": round(mape(PREPARED["y_test"], forecast), 3),
            "MASE": round(mase(PREPARED["y_test"], forecast, naive_test), 3)
            if naive_test is not None else None,
            "gain sur le naif %": round(100.0 * (1.0 - mape(PREPARED["y_test"], forecast)
                                                 / FLOOR_MAPE), 1),
            "MAE MW": round(float(np.mean(np.abs(
                np.asarray(PREPARED["y_test"], dtype="float64") - forecast))), 1),
            "entraînement s": round(elapsed, 2),
        })
    except Exception as error:  # noqa: BLE001 - un échec d'algorithme est un résultat du tableau
        rows.append({"algorithme": algorithm, "MAPE test %": None, "erreur": str(error)[:80]})

algorithm_table = pd.DataFrame(rows)
print(algorithm_table.to_string(index=False))
"""

_THERMO_LEARNED_CELL = """
# Ce que le modèle linéaire a appris : la thermo-sensibilité en MW par °C.
# C'est la raison d'entraîner une Ridge alors qu'elle est battue par les arbres : elle rend la
# relation lisible dans l'unité du métier, et cette lecture est un contrôle de sanity du modèle.
if "ridge" not in fitted:
    print("la Ridge n'a pas pu être entraînée : section sans objet")
else:
    names = list(PREPARED["feature_names"])
    estimator = getattr(fitted["ridge"], "estimator_", None)
    coefficients = np.asarray(getattr(estimator, "coef_", [])).ravel()
    if coefficients.size != len(names):
        print(f"taille inattendue : {coefficients.size} coefficients pour {len(names)} features")
    else:
        level = float(np.asarray(PREPARED["y_train"], dtype="float64").mean())
        names_series = pd.Series(names)
        weather_mask = names_series.str.contains("temp|hdd|cdd", regex=True)
        calendar_mask = names_series.str.startswith("target_")
        history_mask = names_series.str.startswith("load_")
        contributions = (
            pd.DataFrame({"feature": names, "coefficient": coefficients})
            .assign(
                absolu=lambda frame: frame["coefficient"].abs(),
                pct_du_niveau=lambda frame: 100.0 * frame["coefficient"] / level,
                famille=np.select(
                    [weather_mask, calendar_mask, history_mask],
                    ["météo", "calendrier", "historique"],
                    default="autre",
                ),
            )
            .sort_values("absolu", ascending=False)
            .reset_index(drop=True)
        )
        print("15 plus fortes contributions (coefficients standardisés) :")
        print(contributions.head(15).round(3).to_string(index=False))
        print()
        totals = contributions.groupby("famille")["coefficient"].apply(
            lambda values: float(values.abs().sum())
        )
        for family in ("météo", "calendrier", "historique", "autre"):
            if family in totals.index:
                print(f"poids cumulé {family:12s} : {totals[family]:.2f}")
"""

_HORIZON_STRATEGY_CELL = """
# Un modèle unique avec l'horizon en feature, contre un modèle par horizon : la section mesure
# ce que le choix configuré (modèle unique) coûte et ce qu'il rapporte.
from src.models import build_model  # noqa: E402

test_frame = PREPARED["splits"].test
train_frame = PREPARED["splits"].train
horizons_present = sorted(int(value) for value in test_frame[HORIZON_COLUMN].unique())

# Stratégie A : le modèle configuré, tous horizons confondus (ce que fait le pipeline).
unique_model = build_model(
    CONFIG, feature_names=PREPARED["feature_names"], params=dict(CONFIG.model.params)
)
unique_model.fit(PREPARED["X_train"], PREPARED["y_train"])
forecast_unique = np.asarray(unique_model.predict(PREPARED["X_test"]), dtype="float64").ravel()

# Stratégie B : un modèle par horizon, entraîné sur les seules lignes de cet horizon.
columns = list(PREPARED["X_train"].columns) if hasattr(PREPARED["X_train"], "columns") else None
forecast_per_horizon = np.full(len(test_frame), np.nan)
cost = []
for horizon in horizons_present:
    train_mask = (train_frame[HORIZON_COLUMN].to_numpy() == horizon)
    test_mask = (test_frame[HORIZON_COLUMN].to_numpy() == horizon)
    if columns is None:
        print("matrices non typées DataFrame : comparaison impossible dans ce notebook")
        break
    model = build_model(
        CONFIG, feature_names=PREPARED["feature_names"], params=dict(CONFIG.model.params)
    )
    model.fit(PREPARED["X_train"].loc[train_mask], PREPARED["y_train"][train_mask])
    started = time.perf_counter()
    forecast_per_horizon[test_mask] = np.asarray(
        model.predict(PREPARED["X_test"].loc[test_mask]), dtype="float64"
    ).ravel()
    cost.append(time.perf_counter() - started)

rows = []
truth_test = np.asarray(PREPARED["y_test"], dtype="float64")
for horizon in horizons_present:
    mask = test_frame[HORIZON_COLUMN].to_numpy() == horizon
    rows.append({
        "horizon": f"J+{horizon}",
        "lignes": int(mask.sum()),
        "modèle unique MAPE %": round(mape(truth_test[mask], forecast_unique[mask]), 3),
        "modèle dédié MAPE %": round(mape(truth_test[mask], forecast_per_horizon[mask]), 3)
        if np.isfinite(forecast_per_horizon[mask]).all() else None,
    })
strategy_table = pd.DataFrame(rows)
strategy_table.loc["global"] = [
    "global",
    len(truth_test),
    round(mape(truth_test, forecast_unique), 3),
    round(mape(truth_test, forecast_per_horizon), 3)
    if np.isfinite(forecast_per_horizon).all() else None,
]
print(strategy_table.to_string(index=False))
print(f"\\nmodèles à maintenir : 1 (stratégie unique) contre {len(horizons_present)} (dédiés)")
"""

_GRID_CELL = """
# Grille d'hyperparamètres, évaluée sur le split de VALIDATION chronologique.
# Jamais sur le test : le test ne sert qu'à la mesure finale, une seule fois.
from itertools import product  # noqa: E402

from src.models import build_model  # noqa: E402

GRID = __PARAM_GRID__
keys = sorted(GRID)
truth_val = np.asarray(PREPARED["y_val"], dtype="float64")
naive_val = (
    PREPARED["splits"].val[NAIVE_COLUMN].to_numpy(dtype="float64")
    if NAIVE_COLUMN and PREPARED["splits"].val is not None else None
)

rows = []
for values in product(*(GRID[key] for key in keys)):
    override = dict(zip(keys, values, strict=True))
    model = build_model(
        CONFIG,
        feature_names=PREPARED["feature_names"],
        params={**dict(CONFIG.model.params), **override},
    )
    model.fit(PREPARED["X_train"], PREPARED["y_train"])
    forecast = np.asarray(model.predict(PREPARED["X_val"]), dtype="float64").ravel()
    rows.append({
        **{key: override[key] for key in keys},
        "MAPE val %": round(mape(truth_val, forecast), 3),
        "MASE val": round(mase(truth_val, forecast, naive_val), 3) if naive_val is not None else None,
        "MAPE train %": round(mape(PREPARED["y_train"], np.asarray(
            model.predict(PREPARED["X_train"]), dtype="float64").ravel()), 3),
    })

grid_table = pd.DataFrame(rows).sort_values("MAPE val %").reset_index(drop=True)
print(grid_table.to_string(index=False))
grid_table["écart d'apprentissage"] = (grid_table["MAPE val %"] - grid_table["MAPE train %"]).round(3)
print("\\navec l'écart train/validation (mesure du surapprentissage) :")
print(grid_table[[*keys, "MAPE train %", "MAPE val %", "écart d'apprentissage"]]
      .sort_values("MAPE val %").to_string(index=False))
BEST_PARAMS = {key: grid_table.iloc[0][key] for key in keys}
print(f"\\nmeilleur point de la grille sur validation : {BEST_PARAMS}")
print(f"paramètres configurés dans conf/model/default.yaml : "
      f"{ {key: CONFIG.model.params.get(key) for key in keys} }")
"""

_BACKTEST_CELL = """
# Backtest à origine glissante AVEC ré-entraînement : le protocole que le rapport ne peut pas
# se permettre en routine (il score le modèle déjà entraîné, à coût nul). Ici on paie le coût,
# sur un nombre de replis réduit, pour vérifier que le classement des replis n'est pas un artefact.
from src.models import build_model  # noqa: E402

ordered = PREPARED["splits"].test.sort_values(TIME_COLUMN).reset_index(drop=True)
folds = max(BACKTEST_FOLDS, 2)
edges = np.linspace(0, len(ordered), folds + 1).astype(int)

rows = []
for index in range(folds):
    block = ordered.iloc[edges[index]: edges[index + 1]]
    if len(block) < 20:
        continue
    # Fenêtre d'entraînement étendue : tout ce qui précède le repli, dans train + val.
    # Les colonnes à projeter sont celles d'**avant** pré-traitement (`frame_columns`) : après
    # one-hot, les modalités n'existent pas encore dans le cadre enrichi.
    history = pd.concat([PREPARED["splits"].train, PREPARED["splits"].val], ignore_index=True)
    X_history = PREPARED["pipeline"].transform(
        PREPARED["builder"].transform(history).loc[:, PREPARED["frame_columns"]]
    )
    y_history = history[TARGET].to_numpy(dtype="float64")
    X_block = PREPARED["pipeline"].transform(
        PREPARED["builder"].transform(block).loc[:, PREPARED["frame_columns"]]
    )
    model = build_model(
        CONFIG, feature_names=PREPARED["feature_names"], params=dict(CONFIG.model.params)
    )
    model.fit(X_history, y_history)
    forecast = np.asarray(model.predict(X_block), dtype="float64").ravel()
    truth_block = block[TARGET].to_numpy(dtype="float64")
    naive_block = (
        block[NAIVE_COLUMN].to_numpy(dtype="float64") if NAIVE_COLUMN else None
    )
    rows.append({
        "repli": index + 1,
        "période": f"{pd.to_datetime(block[TIME_COLUMN]).min().date()} -> "
                   f"{pd.to_datetime(block[TIME_COLUMN]).max().date()}",
        "lignes": len(block),
        "MAPE %": round(mape(truth_block, forecast), 3),
        "MASE": round(mase(truth_block, forecast, naive_block), 3) if naive_block is not None else None,
        "naif %": round(mape(truth_block, naive_block), 3) if naive_block is not None else None,
    })

backtest_table = pd.DataFrame(rows)
print(backtest_table.to_string(index=False))
if len(backtest_table) > 1:
    spread = float(backtest_table["MAPE %"].std())
    print(f"\\ndispersion entre replis : {spread:.3f} point de MAPE")
    print(f"dérive premier -> dernier : "
          f"{backtest_table['MAPE %'].iloc[-1] - backtest_table['MAPE %'].iloc[0]:+.3f} point")
"""

_LEAKAGE_CELL = """
# SONDE DE FUITE : on viole volontairement le contrat d'antériorité pour voir à quoi ressemble
# un modèle qui triche. Cette cellule est un avertissement, pas une amélioration.
from src.models import build_model  # noqa: E402

leaked_train = PREPARED["X_train"].copy()
leaked_test = PREPARED["X_test"].copy()
leaked_train["FUTURE_LOAD"] = np.asarray(PREPARED["y_train"], dtype="float64")
leaked_test["FUTURE_LOAD"] = np.asarray(PREPARED["y_test"], dtype="float64")

model = build_model(
    CONFIG,
    feature_names=[*PREPARED["feature_names"], "FUTURE_LOAD"],
    params=dict(CONFIG.model.params),
)
model.fit(leaked_train, PREPARED["y_train"])
forecast = np.asarray(model.predict(leaked_test), dtype="float64").ravel()
leaked_mape = mape(PREPARED["y_test"], forecast)
honest_mape = mape(PREPARED["y_test"], forecast_unique)

print(f"MAPE avec la cible du jour futur en feature : {leaked_mape:.4f} %")
print(f"MAPE du modèle honnête                      : {honest_mape:.4f} %")
print(f"plancher naïf                               : {FLOOR_MAPE:.4f} %")
print()
print("Un MAPE proche de zéro n'est jamais une bonne nouvelle en prévision : c'est la signature")
print("d'une fuite. Le bruit irréductible de cette série (AR(1) multiplicatif + erreur de")
print("prévision météo) fixe un plancher structurel bien au-dessus de zéro.")
"""

_DECISION_CELL = """
# Décision : ce que les mesures ci-dessus justifient.
from src.evaluation.reports import DEFAULT_THRESHOLDS  # noqa: E402

chosen = algorithm_table.dropna(subset=["MAPE test %"]).sort_values("MAPE test %")
print("=== synthèse ===")
print(f"plancher naïf sur le test        : {FLOOR_MAPE:.2f} % ({BEST_REFERENCE})")
print(f"meilleur algorithme par défaut   : {chosen.iloc[0]['algorithme']} "
      f"({chosen.iloc[0]['MAPE test %']:.2f} %)")
print(f"modèle configuré, réglé (grille) : {grid_table.iloc[0]['MAPE val %']:.2f} % sur validation")
print(f"modèle qui triche (sonde)        : {leaked_mape:.4f} % — à ne jamais publier")
print()
print("critères du manifeste :")
for key in ("mape_max", "improvement_vs_naive_min", "mase_max", "mape_long_horizon_max"):
    print(f"  {key:28s} = {DEFAULT_THRESHOLDS[key]}")
print()
print("Ce que ces chiffres justifient :")
print("  * la famille retenue (arbres) bat le linéaire dès que la thermo-sensibilité est une")
print("    courbe et que les interactions calendrier x météo comptent ;")
print("  * le linéaire garde une valeur : il publie la thermo-sensibilité en MW/°C, que le")
print("    boosting ne rend qu'au prix d'une importance par permutation ;")
print("  * le modèle unique multi-horizons coûte quelques dixièmes de point face à des modèles")
print("    dédiés, et évite de maintenir quatre chaînes d'entraînement — arbitrage documenté ;")
print("  * le réglage retenu est celui qui minimise l'écart train/validation, pas celui qui")
print("    minimise le MAPE de validation seul : un écart large annonce une dégradation en")
print("    exploitation sur les régimes rares.")
"""


def build_04_model_exploration(context: NotebookContext, destination: Path) -> Path:
    """Build ``04_model_exploration.ipynb`` for a forecasting project.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    spec = context.spec
    cells: list[NotebookNode] = [
        _md(
            f"""# 04 — Exploration des modèles de prévision

**Projet** : {spec.title}
**Modèle configuré** : `{spec.model.algorithm}` ({spec.model.display_name})
**Pourquoi ce choix** : {spec.model.rationale}

En prévision, « explorer les modèles » ne veut pas dire essayer beaucoup d'algorithmes et garder le
meilleur. Trois questions précèdent ce choix, dans cet ordre :

1. **quel est le plancher ?** — la référence naïve que l'opérateur produit déjà. Sans elle, un MAPE
   de 4 % ne veut rien dire : il peut être excellent ou catastrophique ;
2. **quelle famille peut représenter la physique du problème ?** — la thermo-sensibilité est une
   courbe, le calendrier est catégoriel, les régimes sont rares : toutes les familles ne peuvent pas
   représenter cela ;
3. **le résultat est-il honnête ?** — une prévision se trompe de 0,1 % quand elle triche. La sonde de
   fuite en fin de notebook montre volontairement ce chiffre pour qu'on apprenne à s'en méfier.

Les alternatives déclarées dans le manifeste : {"; ".join(spec.model.alternatives[:4])}.
"""
        ),
        _objectives(
            context,
            [
                "Mesurer le **plancher naïf sur le test** avant tout modèle : c'est la seule "
                "comparaison qui décide de la valeur opérationnelle.",
                "Comparer les familles d'algorithmes **à protocole identique** (mêmes données, "
                "mêmes réglages par défaut) pour séparer l'effet de la famille de l'effet du "
                "réglage.",
                "Lire la **thermo-sensibilité apprise** par le modèle linéaire, en MW par °C : un "
                "contrôle de cohérence métier qu'aucune métrique ne remplace.",
                "Arbitrer **modèle unique multi-horizons contre modèles dédiés**, chiffre à l'appui.",
                "Régler les hyperparamètres **sur la validation chronologique**, en regardant "
                "l'écart train/validation et pas seulement le score.",
                "Exécuter un **backtest avec ré-entraînement**, le protocole complet que le rapport "
                "de routine approxime.",
                "Provoquer une **fuite volontaire** pour reconnaître sa signature.",
            ],
        ),
        _code(SETUP, context),
        _code(LOAD_RAW, context),
        _code(FORECAST_HELPERS, context),
        _code(PREPARE, context),
        _md(
            """## 1. Le plancher : ce que produit l'opérateur sans modèle

Toutes les comparaisons de ce notebook sont rapportées à ce tableau. Un modèle qui ne bat pas la
meilleure référence de 30 % au moins ne justifie pas son coût d'exploitation."""
        ),
        _code(_FLOOR_CELL, context),
        _insight(
            [
                "Le naif saisonnier est la référence la plus forte, parce que la consommation est "
                "dominée par le cycle hebdomadaire : recopier la semaine dernière capture déjà "
                "l'essentiel.",
                "La persistance et la climatologie sont nettement plus mauvaises, chacune pour une "
                "raison différente — la persistance ignore le jour de la semaine, la climatologie "
                "ignore la météo. Le modèle doit faire mieux que les trois à la fois.",
                "Ce plancher est mesuré sur le **test** : le mesurer sur l'entraînement avantagerait "
                "le naif, qui recopie une semaine déjà vue.",
            ]
        ),
        _md("## 2. Comparaison des familles d'algorithmes"),
        _code(_ALGORITHMS_CELL, context),
        _insight(
            [
                "La Ridge est battue parce que la thermo-sensibilité est une **courbe** : elle ne "
                "peut la représenter qu'au prix de features fabriquées à la main (degrés-jours, "
                "termes croisés). Les arbres la capturent sans transformation.",
                "La forêt aléatoire et le boosting par histogrammes sont proches ; le boosting gagne "
                "généralement sur ce volume, et surtout il est **natif valeurs manquantes** — une "
                "panne de capteur météo ne casse pas la publication du matin.",
                "Le boosting séquentiel historique (`gradient_boosting`) est plusieurs fois plus "
                "lent à précision comparable : c'est la raison du choix des histogrammes.",
                "Le SVR à noyau RBF tient sur quelques milliers de lignes mais son coût est "
                "quadratique et il ne gère pas les manquants : inutilisable sur un périmètre "
                "national.",
                "Un écart de MAPE inférieur à la dispersion entre replis du backtest (section 5) "
                "n'est pas un signal : ne pas choisir une famille sur un écart de cet ordre.",
            ]
        ),
        _md(
            """## 3. Ce que le modèle linéaire a appris : la thermo-sensibilité en MW/°C

On entraîne la Ridge même en sachant qu'elle est battue : elle rend la relation dans l'unité du
métier. Un coefficient de température qui aurait le mauvais signe est une erreur que le MAPE ne
révèle pas."""
        ),
        _code(_THERMO_LEARNED_CELL, context),
        _insight(
            [
                "Les coefficients sont exprimés sur des features **standardisées** : ils sont "
                "comparables entre eux, mais pas directement en MW/°C. La conversion passe par "
                "l'écart-type de la colonne de température dans le pré-traitement ajusté.",
                "Le bloc météo domine, suivi du calendrier et de l'historique récent : c'est "
                "l'ordre attendu pour une consommation tertiaire/résidentielle, et le vérifier est "
                "un contrôle de sanity gratuit.",
                "Le boosting ne produit pas cette lecture ; il faut passer par une importance par "
                "permutation (notebook 06), qui dit **quelle** variable compte mais pas **dans quel "
                "sens** ni en quelle unité.",
            ]
        ),
        _md("## 4. Un modèle multi-horizons, ou un modèle par horizon ?"),
        _code(_HORIZON_STRATEGY_CELL, context),
        _insight(
            [
                "Le modèle unique traite l'horizon comme une feature ordinaire : il apprend un "
                "compromis, là où un modèle dédié peut spécialiser ses splits sur l'échéance.",
                "L'écart mesuré est faible, et il est à comparer au coût : quatre modèles à "
                "entraîner, surveiller, versionner et ré-entraîner, quatre jeux d'hyperparamètres "
                "qui dérivent indépendamment.",
                "Le choix configuré (modèle unique) est donc un arbitrage **d'exploitation**, pas un "
                "renoncement à la précision. Si l'écart dépassait un demi-point de MAPE au J+7, "
                "l'arbitrage serait à revoir — c'est exactement ce que cette section permet de "
                "surveiller.",
            ]
        ),
        _md(
            """## 5. Réglage sur validation chronologique

La grille déclarée dans le manifeste (`extras.notebook_param_grid`) est rejouée ici. Deux règles :
elle est évaluée sur la **validation**, jamais sur le test ; et on regarde l'**écart
train/validation**, pas seulement le score."""
        ),
        _code(_GRID_CELL, context),
        _insight(
            [
                "Le nombre de feuilles est le réglage le plus sensible : trop de feuilles sur une "
                "série bruitée mémorise les pointes isolées, et cela se voit d'abord dans l'écart "
                "train/validation avant de se voir dans le score.",
                "Le sous-échantillonnage de colonnes (`max_features`) décorrèle les arbres, ce qui "
                "compte ici parce que plusieurs colonnes sont des transformations monotones les "
                "unes des autres (température, anomalie, degrés-jours).",
                "Le pas d'apprentissage et le nombre d'itérations sont quasi interchangeables à "
                "produit constant : la grille est plate sur cet axe, donc on retient le point le "
                "plus économe.",
                "Un point de grille qui gagne 0,02 point de MAPE mais double l'écart "
                "train/validation est un mauvais choix : il achète du score en backtest avec de la "
                "fragilité en exploitation.",
            ]
        ),
        _md(
            """## 6. Backtest à origine glissante **avec** ré-entraînement

Le rapport d'évaluation score le modèle déjà entraîné sur chaque repli (`fixed_model`) : c'est une
mesure de dérive, à coût nul. Le protocole complet ré-entraîne sur une fenêtre étendue à chaque
repli, ce qui est le comportement réel d'un modèle rafraîchi périodiquement."""
        ),
        _code(_BACKTEST_CELL, context),
        _insight(
            [
                "La dispersion entre replis mesure la **stabilité** opérationnelle : un MAPE moyen "
                "correct mais très dispersé est plus coûteux à exploiter qu'un MAPE légèrement "
                "supérieur et stable, parce que l'astreinte ne peut pas s'y fier.",
                "Une dérive croissante du premier au dernier repli indique un modèle qui vieillit "
                "(thermo-sensibilité du parc, nouveaux usages) : le levier est la fréquence de "
                "ré-entraînement, pas la grille d'hyperparamètres.",
                "Comparer chaque repli au naif **sur ce repli** est essentiel : un repli de vague de "
                "froid a un naif plus mauvais, donc un gain apparent plus flatteur.",
            ]
        ),
        _md("## 7. Sonde de fuite : reconnaître un modèle qui triche"),
        _code(_LEAKAGE_CELL, context),
        _insight(
            [
                "Ajouter la consommation du jour cible comme feature fait s'effondrer le MAPE. "
                "C'est **exactement** ce que produit une fuite temporelle discrète : une colonne "
                "anodine (degrés-jours calculés sur la réalisation, décalage mal indexé, moyenne "
                "glissante centrée) suffit.",
                "Le plancher structurel de cette série est bien au-dessus de zéro : bruit AR(1) "
                "multiplicatif plus erreur de prévision météo. Un MAPE anormalement bas est donc un "
                "signal d'alerte, pas une réussite.",
                "La parade n'est pas un test unique mais un **contrat** : chaque colonne est classée "
                "« connue à l'origine », « connue par avance » ou « métadonnée », et les "
                "métadonnées sont exclues par `drop_columns` — audit rejoué en section 5.4 du "
                "notebook 01.",
            ]
        ),
        _md("## 8. Décision"),
        _code(_DECISION_CELL, context),
        _insight(
            [
                "Le choix final n'est pas « le meilleur MAPE » mais le meilleur couple "
                "**précision / coût d'exploitation**, sous contrainte d'honnêteté (aucune fuite, "
                "validation chronologique, comparaison au naif).",
                "Toutes les décisions de ce notebook sont traçables : les hyperparamètres retenus "
                "sont écrits dans `conf/model/default.yaml` avec le commentaire de la mesure qui "
                "les justifie, et la grille est rejouable ici.",
                "Ce qui reste à faire au notebook 05 : vérifier que la validation chronologique "
                "n'est pas optimiste, arbitrer la perte d'entraînement, et mesurer ce que "
                "l'entraînement persiste pour l'inférence.",
            ]
        ),
    ]
    return write_notebook(Path(destination) / "04_model_exploration.ipynb", cells)


# ---------------------------------------------------------------------------------------
# 05 — Entraînement : validation chronologique, perte, surapprentissage, artefacts
# ---------------------------------------------------------------------------------------
_TRAIN_CELL = """
# Entraînement avec l'objet de production (Trainer), dans un répertoire temporaire.
# Un notebook ne doit jamais écraser les artefacts de `make train` : il les reproduit à côté.
import tempfile  # noqa: E402

from src.models import build_model  # noqa: E402
from src.training.callbacks import MetricHistoryCallback, MetricThresholdCallback  # noqa: E402
from src.training.trainer import Trainer, TrainingData  # noqa: E402
from src.utils.paths import ProjectPaths  # noqa: E402

SANDBOX = ProjectPaths(root=Path(tempfile.mkdtemp(prefix="notebook-05-")))
SANDBOX.ensure()

MODEL = build_model(CONFIG, feature_names=PREPARED["feature_names"])
history = MetricHistoryCallback()
threshold = MetricThresholdCallback(
    f"val_{CONFIG.metrics.primary}",
    float(CONFIG.metrics.min_primary),
    mode="max" if str(CONFIG.metrics.direction) == "maximize" else "min",
)
trainer = Trainer(
    MODEL,
    config=CONFIG.train.model_dump(),
    paths=SANDBOX,
    metric_names=CONFIG.metrics.all_metrics,
    task=CONFIG.metrics.task,
    callbacks=[history, threshold],
    min_primary_metric=CONFIG.metrics.min_primary,
    primary_metric=f"val_{CONFIG.metrics.primary}",
    primary_direction=str(CONFIG.metrics.direction),
)
outcome = trainer.train(
    TrainingData(
        X_train=PREPARED["X_train"],
        y_train=PREPARED["y_train"],
        X_val=PREPARED["X_val"],
        y_val=PREPARED["y_val"],
        feature_names=PREPARED["feature_names"],
        task=CONFIG.metrics.task,
    )
)

PRIMARY_KEY = f"val_{CONFIG.metrics.primary}"
print(f"métrique primaire : {PRIMARY_KEY} = {outcome.metrics.get(PRIMARY_KEY, float('nan')):.4f}")
print(f"seuil déclaré     : {CONFIG.metrics.min_primary} (sens : {CONFIG.metrics.direction})")
print(f"gate du callback  : {'satisfait' if threshold.satisfied else 'NON satisfait'}")
print(f"durée             : {outcome.duration_seconds:.2f} s")
print(f"artefacts écrits  : {sorted(outcome.artifacts)}")
print(f"bac à sable       : {SANDBOX.root.name} (les artefacts de `make train` ne sont pas touchés)")
print()
print("métriques de validation :")
pd.Series(
    {key: round(float(value), 4) for key, value in outcome.metrics.items()
     if isinstance(value, (int, float))},
    name="valeur",
).to_frame()
"""

_TEMPORAL_VS_RANDOM_CELL = """
# Validation chronologique contre validation aléatoire : la mesure de l'optimisme.
# Une validation croisée aléatoire sur une série temporelle mélange les époques : des lignes
# d'entraînement encadrent temporellement des lignes de test, ce qui est une fuite par recouvrement.
from sklearn.model_selection import KFold  # noqa: E402

from src.models import build_model  # noqa: E402

X_all = pd.concat([PREPARED["X_train"], PREPARED["X_val"]], ignore_index=True)
y_all = pd.concat(
    [pd.Series(np.asarray(PREPARED["y_train"], dtype="float64")),
     pd.Series(np.asarray(PREPARED["y_val"], dtype="float64"))],
    ignore_index=True,
)
frame_all = pd.concat(
    [PREPARED["splits"].train, PREPARED["splits"].val], ignore_index=True
)

random_scores = []
for train_index, test_index in KFold(n_splits=5, shuffle=True, random_state=CONFIG.seed).split(X_all):
    model = build_model(CONFIG, feature_names=PREPARED["feature_names"],
                        params=dict(CONFIG.model.params))
    model.fit(X_all.iloc[train_index], y_all.to_numpy()[train_index])
    forecast = np.asarray(model.predict(X_all.iloc[test_index]), dtype="float64").ravel()
    random_scores.append(mape(y_all.to_numpy()[test_index], forecast))

# Équivalent chronologique : cinq replis consécutifs, sans mélange
# (chaque repli s'entraîne sur tout ce qui précède son bloc test).
N_FOLDS = 5
edges = np.linspace(0, len(X_all), N_FOLDS + 2).astype(int)  # N replis -> N+2 bornes
temporal_scores = []
for index in range(N_FOLDS):
    train_index = np.arange(0, edges[index + 1])
    test_index = np.arange(edges[index + 1], edges[index + 2])
    if len(test_index) == 0:
        continue
    model = build_model(CONFIG, feature_names=PREPARED["feature_names"],
                        params=dict(CONFIG.model.params))
    model.fit(X_all.iloc[train_index], y_all.to_numpy()[train_index])
    forecast = np.asarray(model.predict(X_all.iloc[test_index]), dtype="float64").ravel()
    temporal_scores.append(mape(y_all.to_numpy()[test_index], forecast))

comparison = pd.DataFrame({
    "protocole": ["KFold aléatoire", "replis chronologiques"],
    "MAPE moyen %": [round(float(np.mean(random_scores)), 3),
                     round(float(np.mean(temporal_scores)), 3)],
    "écart-type": [round(float(np.std(random_scores)), 3),
                   round(float(np.std(temporal_scores)), 3)],
    "replis": [len(random_scores), len(temporal_scores)],
})
print(comparison.to_string(index=False))
optimism = float(comparison["MAPE moyen %"].iloc[0] - comparison["MAPE moyen %"].iloc[1])
print(f"\\noptimisme du protocole aléatoire : {optimism:+.3f} point de MAPE")
print("périodes couvertes par chaque repli chronologique :")
for index in range(5):
    block = frame_all.iloc[edges[index + 1]: edges[index + 2]]
    if len(block):
        print(f"  repli {index + 1} : {pd.to_datetime(block[TIME_COLUMN]).min().date()} -> "
              f"{pd.to_datetime(block[TIME_COLUMN]).max().date()} ({len(block)} lignes)")
"""

_LOSS_CELL = """
# Perte d'entraînement : le métier pilote au MAPE, l'estimateur optimise une perte.
# `squared_error` optimise la RMSE ; `absolute_error` est alignée sur le MAPE mais produit des
# gradients discontinus. La comparaison est le seul moyen honnête de trancher.
from src.models import build_model  # noqa: E402

rows = []
for loss in ("squared_error", "absolute_error"):
    params = {**dict(CONFIG.model.params), "loss": loss}
    try:
        model = build_model(CONFIG, feature_names=PREPARED["feature_names"], params=params)
        model.fit(PREPARED["X_train"], PREPARED["y_train"])
        forecast = np.asarray(model.predict(PREPARED["X_test"]), dtype="float64").ravel()
        truth = np.asarray(PREPARED["y_test"], dtype="float64")
        rows.append({
            "perte": loss,
            "MAPE test %": round(mape(truth, forecast), 3),
            "MAE MW": round(float(np.mean(np.abs(truth - forecast))), 1),
            "RMSE MW": round(float(np.sqrt(np.mean((truth - forecast) ** 2))), 1),
            "biais %": round(float(np.mean((forecast - truth) / truth)) * 100.0, 2),
            "P95 erreur %": round(float(np.percentile(np.abs((truth - forecast) / truth)) * 100.0), 2),
        })
    except Exception as error:  # noqa: BLE001 - une perte non supportée est un résultat
        rows.append({"perte": loss, "MAPE test %": None, "erreur": str(error)[:70]})

loss_table = pd.DataFrame(rows)
print(loss_table.to_string(index=False))
"""

_OVERFIT_CELL = """
# Écart d'apprentissage : train / validation / test, et son pilotage par la complexité.
from src.models import build_model  # noqa: E402

truth_train = np.asarray(PREPARED["y_train"], dtype="float64")
truth_val = np.asarray(PREPARED["y_val"], dtype="float64")
truth_test = np.asarray(PREPARED["y_test"], dtype="float64")

rows = []
for leaves in (7, 15, 31, 63, 127):
    params = {**dict(CONFIG.model.params), "max_leaf_nodes": leaves}
    model = build_model(CONFIG, feature_names=PREPARED["feature_names"], params=params)
    model.fit(PREPARED["X_train"], PREPARED["y_train"])
    scores = {
        "train": mape(truth_train, np.asarray(model.predict(PREPARED["X_train"]), dtype="float64")),
        "val": mape(truth_val, np.asarray(model.predict(PREPARED["X_val"]), dtype="float64")),
        "test": mape(truth_test, np.asarray(model.predict(PREPARED["X_test"]), dtype="float64")),
    }
    rows.append({
        "feuilles": leaves,
        "MAPE train %": round(scores["train"], 3),
        "MAPE val %": round(scores["val"], 3),
        "MAPE test %": round(scores["test"], 3),
        "écart train/val": round(scores["val"] - scores["train"], 3),
        "rapport val/train": round(scores["val"] / scores["train"], 2)
        if scores["train"] > 1e-9 else None,
    })

overfit_table = pd.DataFrame(rows)
print(overfit_table.to_string(index=False))
print(f"\\nconfiguration retenue : max_leaf_nodes={CONFIG.model.params.get('max_leaf_nodes')}")

fig, axis = plt.subplots(figsize=(8, 4.2))
axis.plot(overfit_table["feuilles"], overfit_table["MAPE train %"], marker="o", label="train")
axis.plot(overfit_table["feuilles"], overfit_table["MAPE val %"], marker="s", label="validation")
axis.plot(overfit_table["feuilles"], overfit_table["MAPE test %"], marker="^", label="test")
axis.set_xscale("log", base=2)
axis.set_xlabel("nombre maximal de feuilles par arbre")
axis.set_ylabel("MAPE (%)")
axis.set_title("L'écart d'apprentissage se lit avant le score")
axis.legend()
axis.grid(alpha=0.3)
plt.show()
"""

_ARTEFACTS_CELL = """
# Ce que l'entraînement persiste, et ce dont l'inférence a besoin.
# La chaîne de publication casse silencieusement si un seul de ces objets manque ou est désaligné.
artifacts_root = PATHS.artifacts_dir
rows = []
for relative, purpose in [
    ("models/model.joblib", "modèle entraîné (prédiction ponctuelle)"),
    ("models/preprocessing.joblib", "pré-traitement ajusté sur train (mêmes colonnes, mêmes échelles)"),
    ("models/feature_builder.joblib", "recettes de features apprises sur train"),
    ("models/resolved_config.json", "configuration résolue : horizons, méthode d'intervalle, seuils"),
    ("models/model_card.json", "carte du modèle : identité, métriques, empreinte des données"),
    ("reports/intervals.csv", "calibration des intervalles par horizon (méthode + score + échelle)"),
    ("reports/per_horizon.csv", "erreur attendue par horizon, publiée avec chaque prévision"),
]:
    path = artifacts_root / relative
    rows.append({
        "artefact": relative,
        "présent": path.exists(),
        "taille": f"{path.stat().st_size / 1024:.1f} ko" if path.exists() else "-",
        "rôle": purpose,
    })
artifacts_table = pd.DataFrame(rows)
print(artifacts_table.to_string(index=False))
print()

# Aller-retour d'inférence : le prédicteur reconstruit-il la même prévision que le modèle nu ?
from src.inference.predictor import Predictor  # noqa: E402

try:
    predictor = Predictor.from_config(CONFIG.model_dump(), PATHS)
    sample = predictor.sample_inputs(n=4) if hasattr(predictor, "sample_inputs") else None
    if sample is not None:
        published = predictor.predict(sample)
        columns = [column for column in [predictor.horizon_column, f"{TARGET}_prediction",
                                         "lower_mw", "upper_mw", "expected_mape_pct",
                                         "confidence", "interval_method"]
                   if column in published.columns]
        print("publication du prédicteur (4 enregistrements) :")
        print(published[columns].to_string(index=False))
except Exception as error:  # noqa: BLE001 - l'absence d'artefacts est un cas nominal en notebook
    print(f"prédicteur indisponible (lancez `make train` puis `make evaluate`) : {error}")
"""

_RETRAIN_CELL = """
# Coût et déclencheur de ré-entraînement : la question que pose l'astreinte, pas le notebook.
import time  # noqa: E402

from src.models import build_model  # noqa: E402

started = time.perf_counter()
model = build_model(CONFIG, feature_names=PREPARED["feature_names"],
                    params=dict(CONFIG.model.params))
model.fit(PREPARED["X_train"], PREPARED["y_train"])
fit_seconds = time.perf_counter() - started
forecast = np.asarray(model.predict(PREPARED["X_test"]), dtype="float64").ravel()
score_now = mape(PREPARED["y_test"], forecast)

started = time.perf_counter()
model.predict(PREPARED["X_test"].head(1))
predict_seconds = time.perf_counter() - started

print(f"entraînement ({len(PREPARED['X_train'])} lignes) : {fit_seconds:.2f} s")
print(f"prédiction unitaire                        : {predict_seconds * 1000:.1f} ms")
print(f"MAPE de référence sur le test              : {score_now:.3f} %")
print()

# Déclencheur de dérive : on compare l'erreur des derniers jours à celle du reste de la période.
# Un test statistique simple vaut mieux qu'un seuil arbitraire sur le MAPE, parce qu'il tient
# compte du volume : 50 lignes ne prouvent rien, 500 si.
ordered = PREPARED["splits"].test.sort_values(TIME_COLUMN).reset_index(drop=True)
X_ordered = PREPARED["pipeline"].transform(
    PREPARED["builder"].transform(ordered).loc[:, PREPARED["frame_columns"]]
)
truth_ordered = ordered[TARGET].to_numpy(dtype="float64")
forecast_ordered = np.asarray(model.predict(X_ordered), dtype="float64").ravel()
errors = 100.0 * np.abs((truth_ordered - forecast_ordered) / truth_ordered)

window = max(len(errors) // 5, 20)
recent, past = errors[-window:], errors[:-window]
welch = (recent.mean() - past.mean()) / np.sqrt(
    recent.var(ddof=1) / len(recent) + past.var(ddof=1) / len(past)
)
print(f"fenêtre récente : {window} lignes "
      f"({pd.to_datetime(ordered[TIME_COLUMN]).iloc[-window:].min().date()} -> "
      f"{pd.to_datetime(ordered[TIME_COLUMN]).max().date()})")
print(f"  erreur moyenne récente : {recent.mean():.2f} %")
print(f"  erreur moyenne passée  : {past.mean():.2f} %")
print(f"  statistique de Welch   : {welch:+.2f} (seuil de décision ±1,96)")
print(f"  conclusion             : "
      f"{'ré-entraîner' if welch > 1.96 else 'aucune dérive significative'}")
"""


def build_05_training(context: NotebookContext, destination: Path) -> Path:
    """Build ``05_training.ipynb`` for a forecasting project.

    Args:
        context: Notebook context.
        destination: ``notebooks/`` directory.

    Returns:
        The written path.
    """
    spec = context.spec
    cells: list[NotebookNode] = [
        _md(
            f"""# 05 — Entraînement et validation temporelle

**Projet** : {spec.title}
**Modèle** : `{spec.model.algorithm}` ({spec.model.display_name})
**Métrique de décision** : `{spec.metrics.primary}` (sens : {spec.metrics.direction}, seuil déclaré
{spec.metrics.min_primary})

Entraîner un modèle de prévision est la partie facile. Ce qui distingue un projet exploitable d'un
prototype, ce sont quatre vérifications que ce notebook exécute **chiffres à l'appui** :

1. la validation est-elle **chronologique** ? Un protocole aléatoire sur une série temporelle est
   optimiste, et on mesure ici de combien ;
2. la **perte** optimisée est-elle celle que le métier pilote ? `squared_error` optimise la RMSE,
   alors que la décision se prend au MAPE ;
3. l'**écart d'apprentissage** est-il maîtrisé ? Un modèle qui mémorise le bruit AR(1) de la série
   tient en backtest et s'effondre en exploitation ;
4. les **artefacts** persistés suffisent-ils à l'inférence ? La chaîne de publication casse
   silencieusement s'il en manque un.
"""
        ),
        _objectives(
            context,
            [
                "Entraîner avec l'objet de production (`Trainer`), callbacks compris, sans toucher "
                "aux artefacts du pipeline.",
                "Mesurer l'**optimisme** d'une validation croisée aléatoire face à des replis "
                "chronologiques.",
                "Arbitrer la **perte d'entraînement** (`squared_error` contre `absolute_error`) sur "
                "quatre critères, pas un.",
                "Piloter l'**écart train/validation** par la complexité des arbres.",
                "Vérifier que chaque artefact persisté a un rôle précis dans la chaîne "
                "d'inférence, et faire un aller-retour de publication.",
                "Chiffrer le **coût de ré-entraînement** et tester un déclencheur de dérive.",
            ],
        ),
        _code(SETUP, context),
        _code(LOAD_RAW, context),
        _code(FORECAST_HELPERS, context),
        _code(PREPARE, context),
        _md(
            """## 1. Entraînement avec l'objet de production

Le `Trainer` est celui qu'utilise `make train` : mêmes callbacks, mêmes métriques, même politique
de seuil. La seule différence est le répertoire de destination, temporaire, pour ne pas écraser les
artefacts publiés."""
        ),
        _code(_TRAIN_CELL, context),
        _insight(
            [
                "Les callbacks ne sont pas de la décoration : `MetricThresholdCallback` transforme "
                "le seuil de qualité du manifeste en **gate d'exécution**, donc un ré-entraînement "
                "qui dégrade le modèle est signalé au lieu d'être publié.",
                "`MetricHistoryCallback` accumule l'historique des métriques. Sur un estimateur sans "
                "époque il n'a qu'une ligne, mais c'est lui qui alimente les courbes d'apprentissage "
                "des stacks itératives : le même code sert les deux familles.",
                "Les métriques de validation sont préfixées `val_` et celles d'entraînement ne le "
                "sont pas : c'est ce qui permet de calculer l'écart d'apprentissage sans ambiguïté.",
            ]
        ),
        _md(
            """## 2. Validation chronologique contre validation aléatoire

La question n'est pas théorique : elle décide si les chiffres annoncés au métier sont crédibles.
On exécute les deux protocoles sur les **mêmes** données."""
        ),
        _code(_TEMPORAL_VS_RANDOM_CELL, context),
        _insight(
            [
                "Le protocole aléatoire est systématiquement **optimiste** : chaque repli contient "
                "des lignes dont les voisines temporelles immédiates sont dans l'entraînement. Sur "
                "une série à forte autocorrélation, cela revient à évaluer une interpolation, pas "
                "une prévision.",
                "L'écart mesuré est le prix de l'erreur de protocole. Il se reporte directement sur "
                "la décision : un modèle qui « passe » le seuil en KFold aléatoire peut le manquer "
                "en replis chronologiques.",
                "Les replis chronologiques couvrent des périodes **différentes** (donc des régimes "
                "différents) : leur dispersion n'est pas du bruit, c'est de l'information sur la "
                "stabilité saisonnière du modèle.",
                "C'est la raison pour laquelle `conf/model/default.yaml` désactive la validation "
                "croisée aléatoire : le backtest à origine glissante de l'évaluateur la remplace.",
            ]
        ),
        _md(
            """## 3. La perte d'entraînement face à la métrique de décision

Le métier pilote au MAPE. L'estimateur, lui, optimise une perte. Les aligner n'est pas
automatiquement le bon choix — voici les deux côtés du compromis."""
        ),
        _code(_LOSS_CELL, context),
        _insight(
            [
                "`absolute_error` aligne la perte sur le MAPE, mais ses gradients sont discontinus "
                "autour de zéro : l'optimisation est moins stable et l'ensemble converge plus "
                "lentement.",
                "`squared_error` pénalise davantage les grosses erreurs, ce qui est **souhaitable** "
                "ici : une pointe hivernale ratée coûte plus cher qu'une erreur moyenne en été. La "
                "RMSE plus élevée n'est pas un défaut, c'est la trace de cette priorité.",
                "Le biais est le critère à surveiller : une perte absolue tend à prévoir la médiane, "
                "donc à sous-prévoir les pointes — exactement ce qu'un acheteur d'énergie redoute.",
                "Le choix configuré (`squared_error`) est donc un arbitrage documenté, pas un "
                "réglage par défaut subi.",
            ]
        ),
        _md("## 4. Écart d'apprentissage : ce que la complexité achète et ce qu'elle coûte"),
        _code(_OVERFIT_CELL, context),
        _insight(
            [
                "Quand le nombre de feuilles augmente, le MAPE d'entraînement chute et celui de "
                "validation cesse de suivre : la divergence **est** le surapprentissage. Le point "
                "retenu est le dernier avant la divergence, pas le meilleur score de validation.",
                "Le rapport val/train est plus parlant que l'écart absolu : un modèle à 1,7 % en "
                "entraînement et 3,5 % en validation (rapport ≈ 2) est sain sur une série bruitée ; "
                "un rapport proche de 1 signale une validation trop proche de l'entraînement, donc "
                "probablement mal découpée.",
                "Le test suit la validation de près, ce qui est le signal recherché : si le test "
                "s'écartait nettement, la période de test contiendrait un régime absent de la "
                "validation — c'est précisément ce que le notebook 06 vérifie sur les intervalles.",
            ]
        ),
        _md("## 5. Artefacts persistés et aller-retour d'inférence"),
        _code(_ARTEFACTS_CELL, context),
        _insight(
            [
                "Quatre objets sont nécessaires pour publier : le modèle, le pré-traitement, le "
                "constructeur de features et la **configuration résolue**. Sans ce dernier, "
                "l'évaluateur et le prédicteur devraient deviner la colonne d'horizon ou la méthode "
                "d'intervalle.",
                "Les tables de calibration (`intervals.csv`, `per_horizon.csv`) font partie du "
                "contrat d'inférence : le prédicteur y lit l'intervalle et l'erreur attendue, donc "
                "une évaluation absente dégrade la publication au lieu de la rendre fausse.",
                "L'aller-retour prouve l'alignement : le prédicteur reconstruit les mêmes features "
                "que l'entraînement, dans le même ordre. Un désalignement de colonnes est l'incident "
                "le plus fréquent en mise en production, et il est silencieux.",
            ]
        ),
        _md("## 6. Coût de ré-entraînement et déclencheur de dérive"),
        _code(_RETRAIN_CELL, context),
        _insight(
            [
                "Le coût d'entraînement se compte en secondes sur ce volume : la vraie contrainte "
                "n'est pas le calcul mais la **revue** qui accompagne chaque nouveau modèle "
                "(validation des seuils, comparaison au précédent, journalisation).",
                "Le déclencheur de dérive compare la fenêtre récente au reste de la période avec un "
                "test de Welch : il tient compte du volume, donc il ne déclenche pas sur 30 lignes "
                "bruitées.",
                "Une dérive confirmée appelle un ré-entraînement ; une dérive **saisonnière** "
                "récurrente appelle autre chose — une variable de régime ou un recalibrage. "
                "Distinguer les deux est l'objet du notebook 06.",
            ]
        ),
    ]
    return write_notebook(Path(destination) / "05_training.ipynb", cells)


# ---------------------------------------------------------------------------------------
# 06 — Erreur par horizon, par régime, par saison, intervalles, recommandations
# ---------------------------------------------------------------------------------------
_EVALUATE_CELL = """
# Évaluation complète sur le split de TEST — utilisé une seule fois, ici.
from src.evaluation.evaluator import Evaluator  # noqa: E402

EVALUATOR = Evaluator.from_config(MODEL, CONFIG.model_dump(), NB_PATHS)
RESULT = EVALUATOR.evaluate(
    PREPARED["X_test"],
    PREPARED["y_test"],
    split="test",
    context=PREPARED["enriched"]["test"],
)

metrics_frame = pd.DataFrame({
    "métrique": list(RESULT.metrics),
    "valeur": [round(float(RESULT.metrics[name]), 4) for name in RESULT.metrics],
})
print(f"lignes évaluées        : {RESULT.n_samples}")
print(f"métrique primaire      : {RESULT.primary_metric} = {RESULT.primary_value:.4f}")
print(f"biais moyen            : {RESULT.bias:+.2f} %")
print(f"bande de tolérance ± {EVALUATOR.tolerance_pct:.0f} % : {RESULT.coverage:.1%} des lignes")
print(f"couverture d'intervalle: {RESULT.interval_coverage:.1%} "
      f"(nominal {INTERVAL_LEVEL:.0%})")
metrics_frame
"""

_PER_HORIZON_CELL = """
# Erreur par horizon : quatre produits différents, pas un.
from src.visualization.plots import ForecastingPlots  # noqa: E402

PLOTS = ForecastingPlots(NB_PATHS.figures_dir)
table = RESULT.per_horizon
columns = [column for column in [
    "horizon", "rows", "mape_pct", "smape_pct", "mae_mw", "rmse_mw", "bias_pct",
    "naive_mape_pct", "improvement_vs_naive_pct", "interval_coverage_pct",
] if column in table.columns]
print(table[columns].round(3).to_string(index=False))
print()
for name in ("forecast_vs_actual", "error_by_horizon", "improvement_vs_baselines"):
    path = getattr(PLOTS, name)(RESULT)
    if path is not None:
        display(Image(path, width=620))
"""

_PER_REGIME_CELL = """
# Erreur par régime : là où se perd l'argent.
predictions = RESULT.predictions
if EVENT_COLUMN and EVENT_COLUMN in predictions.columns:
    truth_safe = predictions["y_true"].where(predictions["y_true"].abs() > 1e-9)
    regime = (
        predictions.assign(biais_pct=100.0 * predictions["residual"] / truth_safe)
        .groupby(EVENT_COLUMN)
        .agg(
            lignes=("ape_pct", "size"),
            mape_pct=("ape_pct", "mean"),
            p95_pct=("ape_pct", lambda values: float(np.percentile(values, 95))),
            biais_pct=("biais_pct", "mean"),
            erreur_totale_mw=("absolute_error", "sum"),
        )
        .sort_values("mape_pct", ascending=False)
    )
    regime["part_des_lignes_pct"] = (100.0 * regime["lignes"] / regime["lignes"].sum()).round(2)
    regime["part_de_l_erreur_pct"] = (
        100.0 * regime["erreur_totale_mw"] / regime["erreur_totale_mw"].sum()
    ).round(2)
    print(regime.round(2).to_string())
    print()
    path = PLOTS.error_by_regime(RESULT)
    if path is not None:
        display(Image(path, width=620))
else:
    print("aucune colonne de régime dans le contexte évalué")
"""

_PER_SEASON_CELL = """
# Erreur par mois : un biais saisonnier se corrige par une variable de régime, pas par des arbres.
if "target_month" in predictions.columns:
    month_column = "target_month"
elif TIME_COLUMN in predictions.columns:
    month_column = pd.to_datetime(predictions[TIME_COLUMN]).dt.month
else:
    month_column = None

if month_column is not None:
    months = (
        predictions[month_column] if isinstance(month_column, str) else month_column
    )
    season = (
        predictions.assign(mois=months)
        .groupby("mois")
        .agg(
            lignes=("ape_pct", "size"),
            mape_pct=("ape_pct", "mean"),
            biais_pct=("relative_error_pct", "mean"),
        )
        .sort_values("mape_pct", ascending=False)
    )
    print(season.round(2).to_string())
    print()
    worst = season.index[0]
    print(f"pire mois : {worst} (MAPE {season['mape_pct'].iloc[0]:.2f} %, "
          f"biais {season['biais_pct'].iloc[0]:+.2f} %)")
    path = PLOTS.bias_by_month(RESULT)
    if path is not None:
        display(Image(path, width=620))
"""

_WORST_CELL = """
# Les pires journées : à lire une par une avant de conclure.
errors = RESULT.errors
columns = [column for column in [
    "sample_id", TIME_COLUMN, TARGET_DATE, HORIZON_COLUMN, EVENT_COLUMN, "y_true", "y_pred",
    "ape_pct", "residual",
] if column and column in errors.columns]
print(errors[columns].head(12).round(2).to_string(index=False))
print()
path = PLOTS.worst_errors(RESULT)
if path is not None:
    display(Image(path, width=620))
print()
# Une erreur isolée sur un arrêt industriel inédit n'appelle pas le même correctif qu'une erreur
# systématique sur tous les lundis de vacances scolaires : on distingue les deux par la répétition.
if EVENT_COLUMN and EVENT_COLUMN in errors.columns:
    repetition = errors.head(50)[EVENT_COLUMN].value_counts()
    print("composition des 50 pires lignes par régime :")
    print(repetition.to_string())
"""

_ACF_CELL = """
# Autocorrélation des résidus : ce que le modèle laisse de structure temporelle.
residual = predictions["residual"].to_numpy(dtype="float64")
ordered = predictions.sort_values(TIME_COLUMN) if TIME_COLUMN in predictions.columns else predictions
residual = ordered["residual"].to_numpy(dtype="float64")
n = len(residual)
noise_threshold = 2.0 / np.sqrt(max(n, 1))

rows = []
for lag in (1, 2, 3, 7, 14, 28):
    if lag >= n:
        continue
    coefficient = float(np.corrcoef(residual[lag:], residual[:-lag])[0, 1])
    rows.append({
        "décalage (jours)": lag,
        "autocorrélation": round(coefficient, 3),
        "seuil de bruit": round(noise_threshold, 3),
        "significative": bool(abs(coefficient) > noise_threshold),
        "lecture": (
            "inertie non capturée : un décalage supplémentaire ou un terme AR aiderait"
            if lag == 1 and abs(coefficient) > noise_threshold
            else "saisonnalité hebdomadaire résiduelle : le profil par jour est mal appris"
            if lag == 7 and abs(coefficient) > noise_threshold
            else "structure temporelle résiduelle"
            if abs(coefficient) > noise_threshold
            else "compatible avec du bruit"
        ),
    })
acf_table = pd.DataFrame(rows)
print(acf_table.to_string(index=False))
print()
fig, axis = plt.subplots(figsize=(8, 3.6))
axis.bar(acf_table["décalage (jours)"], acf_table["autocorrélation"], color="#2e75b6")
axis.axhline(noise_threshold, color="#c00000", ls="--", lw=1, label="±2/√n (bruit blanc)")
axis.axhline(-noise_threshold, color="#c00000", ls="--", lw=1)
axis.set_xlabel("décalage (jours)")
axis.set_ylabel("autocorrélation des résidus")
axis.set_title("Un résidu autocorrélé n'est pas du bruit : c'est de l'information non apprise")
axis.legend()
axis.grid(alpha=0.3, axis="y")
plt.show()
print(f"\\nécart-type des résidus : {residual.std():.1f} | "
      f"erreur relative médiane : {predictions['ape_pct'].median():.2f} % | "
      f"P95 : {predictions['ape_pct'].quantile(0.95):.2f} %")
"""

_INTERVALS_CELL = """
# Couverture réelle des intervalles, ventilée — puis le recalibrage adaptatif recommandé.
table = RESULT.intervals
columns = [column for column in [
    "horizon", "rows", "level_pct", "method", "scale_column", "score_quantile", "mean_width_mw",
    "relative_width_pct", "coverage_pct", "coverage_calm_pct", "coverage_extreme_pct", "n_extreme",
] if column in table.columns]
print(table[columns].round(3).to_string(index=False))
print()
path = PLOTS.interval_coverage(RESULT)
if path is not None:
    display(Image(path, width=620))

# --- Expérience : recalibrage adaptatif ---------------------------------------------------------
# Un système réel connaît chaque matin l'erreur d'hier. On ajoute donc au pool de calibration les
# résidus des lignes **chronologiquement précédentes**, ce que la publication statique ne fait pas.
val_frame = PREPARED["splits"].val
val_forecast = np.asarray(MODEL.predict(PREPARED["X_val"]), dtype="float64").ravel()
val_truth = np.asarray(PREPARED["y_val"], dtype="float64")
if INTERVAL_SCALE in val_frame.columns:
    val_scale = np.abs(val_frame[INTERVAL_SCALE].to_numpy(dtype="float64"))
else:
    val_scale = np.abs(val_forecast)
val_scores = np.abs(val_forecast - val_truth) / np.where(val_scale > 1e-9, val_scale, 1.0)
val_horizons = val_frame[HORIZON_COLUMN].to_numpy()

evaluated = predictions.copy()
if TIME_COLUMN in evaluated.columns:
    evaluated = evaluated.sort_values(TIME_COLUMN).reset_index(drop=True)
scale_eval = (
    np.abs(evaluated[INTERVAL_SCALE].to_numpy(dtype="float64"))
    if INTERVAL_SCALE in evaluated.columns
    else np.abs(evaluated["y_pred"].to_numpy(dtype="float64"))
)
scale_eval = np.where(scale_eval > 1e-9, scale_eval, float(np.median(scale_eval)))
truth_eval = evaluated["y_true"].to_numpy(dtype="float64")
forecast_eval = evaluated["y_pred"].to_numpy(dtype="float64")
residual_eval = evaluated["residual"].to_numpy(dtype="float64")
horizons_eval = evaluated[HORIZON_COLUMN].to_numpy()
calm_eval = (
    (evaluated[EVENT_COLUMN].astype(str) == "none").to_numpy()
    if EVENT_COLUMN and EVENT_COLUMN in evaluated.columns
    else np.ones(len(evaluated), dtype=bool)
)

rows = []
for horizon in sorted(pd.unique(horizons_eval)):
    static_pool = val_scores[val_horizons == horizon]
    static_quantile = float(np.quantile(static_pool, INTERVAL_LEVEL))
    test_mask = horizons_eval == horizon
    static_hit = np.abs(truth_eval[test_mask] - forecast_eval[test_mask]) <= (
        static_quantile * scale_eval[test_mask]
    )
    running = list(static_pool)
    adaptive_hit = np.zeros(int(test_mask.sum()), dtype=bool)
    for position, index in enumerate(np.where(test_mask)[0]):
        current = float(np.quantile(np.asarray(running), INTERVAL_LEVEL))
        adaptive_hit[position] = abs(truth_eval[index] - forecast_eval[index]) <= (
            current * scale_eval[index]
        )
        running.append(abs(residual_eval[index]) / scale_eval[index])
    rows.append({
        "horizon": f"J+{horizon}",
        "lignes": int(test_mask.sum()),
        "quantile statique": round(static_quantile, 4),
        "couverture statique %": round(100.0 * float(static_hit.mean()), 1),
        "couverture adaptative %": round(100.0 * float(adaptive_hit.mean()), 1),
        "gain (points)": round(100.0 * (float(adaptive_hit.mean()) - float(static_hit.mean())), 1),
        "couverture adapt. jours calmes %": round(
            100.0 * float(adaptive_hit[calm_eval[test_mask]].mean()), 1
        ) if calm_eval[test_mask].any() else None,
        "couverture adapt. régime extrême %": round(
            100.0 * float(adaptive_hit[~calm_eval[test_mask]].mean()), 1
        ) if (~calm_eval[test_mask]).any() else None,
    })

adaptive_table = pd.DataFrame(rows)
print()
print("=== intervalle conforme normalisé : statique contre recalibrage adaptatif ===")
print(adaptive_table.to_string(index=False))
print(f"\\ncouverture globale statique   : "
      f"{100.0 * float((np.abs(truth_eval - forecast_eval) <= 0).mean()):.1f} % (rappel : "
      f"{RESULT.interval_coverage:.1%} mesurée par l'évaluateur)")
GLOBAL_STATIC = float(np.mean([row["couverture statique %"] for row in rows]))
GLOBAL_ADAPTIVE = float(np.mean([row["couverture adaptative %"] for row in rows]))
print(f"moyenne par horizon : statique {GLOBAL_STATIC:.1f} % -> adaptative "
      f"{GLOBAL_ADAPTIVE:.1f} % (gain {GLOBAL_ADAPTIVE - GLOBAL_STATIC:+.1f} point)")
"""

_DRIVERS_CELL = """
# Facteurs contributifs : quelles variables portent la prévision.
importance = RESULT.feature_importance
if importance is None or getattr(importance, "empty", True):
    print("le modèle n'expose pas d'importance : section sans objet")
else:
    frame = importance.copy()
    name_column = "feature" if "feature" in frame.columns else frame.columns[0]
    value_column = "importance" if "importance" in frame.columns else frame.columns[-1]
    frame["famille"] = np.select(
        [
            frame[name_column].astype(str).str.contains("temp|hdd|cdd", regex=True),
            frame[name_column].astype(str).str.startswith("target_"),
            frame[name_column].astype(str).str.startswith("load_"),
        ],
        ["météo", "calendrier", "historique"],
        default="autre",
    )
    print(frame.sort_values(value_column, ascending=False).head(15).round(4).to_string(index=False))
    print()
    print("poids par famille :")
    print(frame.groupby("famille")[value_column].sum().sort_values(ascending=False)
          .round(3).to_string())
    path = PLOTS.feature_importance(RESULT)
    if path is not None:
        display(Image(path, width=620))
"""

_DIAGNOSTICS_CELL = """
# Synthèse des diagnostics produits par l'évaluateur.
extras = dict(RESULT.extras or {})
keys = [
    "naive_mape", "improvement_vs_naive_pct", "residual_std", "bias_pct", "median_ape_pct",
    "p95_ape_pct", "worst_month_bias_pct", "interval_method", "interval_scale_column",
    "interval_source",
]
print("=== diagnostics ===")
for key in keys:
    if key in extras:
        value = extras[key]
        print(f"  {key:28s} = {value:.4f}" if isinstance(value, (int, float))
              else f"  {key:28s} = {value}")

backtest = RESULT.backtest
if backtest is not None and not backtest.empty:
    print()
    print("=== backtest à origine glissante (modèle figé) ===")
    print(backtest.round(3).to_string(index=False))
    path = PLOTS.backtest_stability(RESULT)
    if path is not None:
        display(Image(path, width=620))

segments = RESULT.per_segment
if segments is not None and not segments.empty:
    print()
    print("=== erreur par segment déclaré ===")
    print(segments.round(3).head(20).to_string(index=False))
"""

_RECOMMENDATIONS_CELL = """
# Recommandations : chacune est rattachée à une mesure de ce notebook, pas à une opinion.
from src.evaluation.reports import DEFAULT_THRESHOLDS  # noqa: E402

EXTRAS = dict(RESULT.extras or {})
BIAS_PCT = float(EXTRAS.get("bias_pct", float("nan")))
GAIN_PCT = float(EXTRAS.get("improvement_vs_naive_pct", float("nan")))
horizon_table = RESULT.per_horizon
long_horizon = (
    horizon_table[horizon_table["horizon"] == LONG_HORIZON]["mape_pct"].iloc[0]
    if "horizon" in horizon_table.columns and (horizon_table["horizon"] == LONG_HORIZON).any()
    else float("nan")
)
print("=== état mesuré ===")
print(f"  MAPE global                : {RESULT.metrics.get('mape', float('nan')):.3f} % "
      f"(seuil {DEFAULT_THRESHOLDS['mape_max']:.1f} %)")
print(f"  MAPE à J+{LONG_HORIZON}                 : {long_horizon:.3f} % "
      f"(seuil {DEFAULT_THRESHOLDS['mape_long_horizon_max']:.1f} %)")
print(f"  gain sur le naif saisonnier: {GAIN_PCT:.1f} % "
      f"(seuil {DEFAULT_THRESHOLDS['improvement_vs_naive_min']:.0f} %)")

print(f"  couverture d'intervalle    : {100.0 * RESULT.interval_coverage:.1f} % "
      f"(nominal {INTERVAL_LEVEL:.0%}, seuil "
      f"{DEFAULT_THRESHOLDS['interval_coverage_min']:.0f} %)")
print(f"  biais moyen                : {BIAS_PCT:+.2f} % "
      f"(seuil ±{DEFAULT_THRESHOLDS['bias_max']:.1f} %)")
print(f"  recalibrage adaptatif      : {GLOBAL_STATIC:.1f} % -> {GLOBAL_ADAPTIVE:.1f} % "
      f"({GLOBAL_ADAPTIVE - GLOBAL_STATIC:+.1f} point)")
print()
print("=== plan d'action, par ordre de rapport gain / coût ===")
from src.evaluation.reports import ReportBuilder  # noqa: E402

REPORTER = ReportBuilder(NB_PATHS, config=CONFIG.model_dump())
for index, recommendation in enumerate(REPORTER.recommendations(RESULT), start=1):
    print(f"{index:2d}. {recommendation}")
"""

_REPORT_CELL = """
# Le rapport généré par le pipeline : mêmes chiffres, mêmes seuils, régénérable à chaque run.
from src.evaluation.reports import ReportBuilder  # noqa: E402

REPORTER = ReportBuilder(NB_PATHS, config=CONFIG.model_dump())
WRITTEN = REPORTER.build(RESULT, model=MODEL)
print(f"{len(WRITTEN)} artefacts écrits dans "
      f"{NB_PATHS.artifacts_dir.relative_to(PROJECT_ROOT)}")
for name, artefact in sorted(WRITTEN.items()):
    print(f"  {name:22s} {Path(artefact).name}")
print()
Markdown(Path(WRITTEN["report"]).read_text(encoding="utf-8")[:3000] + "\\n\\n[…]")
"""


def build_06_error_analysis(context: NotebookContext, destination: Path) -> Path:
    """Build ``06_error_analysis.ipynb`` for a forecasting project.

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
**Principe** : un MAPE global ne dit **jamais** quoi corriger. En prévision, 4 % d'erreur moyenne
peuvent cacher 15 % sur les vagues de froid et 2 % le reste du temps — et ce sont les 15 % qui
coûtent cher, parce que ce sont les jours où l'écart au prévu se paie au prix spot.

Ce notebook descend au niveau de la ligne, dans quatre directions : **par horizon** (quatre produits
différents), **par régime** (là où se perd l'argent), **par saison** (un biais saisonnier ne se
corrige pas avec des arbres), et **dans le temps** (résidus autocorrélés = structure non apprise).
Il mesure enfin la couverture réelle des intervalles et teste le recalibrage adaptatif que le
rapport recommande.

Le split de **test** n'est utilisé qu'ici — une seule fois.
"""
        ),
        _objectives(
            context,
            [
                "Produire l'évaluation complète avec l'objet de production (`Evaluator`), y compris "
                "les intervalles et le backtest.",
                "Ventiler l'erreur **par horizon, par régime, par mois** : trois lectures qui "
                "appellent trois correctifs différents.",
                "Lire les **pires journées** une par une, et distinguer l'erreur isolée de l'erreur "
                "systématique.",
                "Mesurer l'**autocorrélation des résidus** : ce qui reste de structure temporelle "
                "est de l'information non apprise, pas du bruit.",
                "Vérifier la **couverture réelle des intervalles** et tester le recalibrage "
                "adaptatif, en le ventilant entre jours calmes et régimes extrêmes.",
                "Formuler des recommandations **chiffrées**, chacune rattachée à une mesure.",
            ],
        ),
        _code(SETUP, context),
        _code(LOAD_RAW, context),
        _code(FORECAST_HELPERS, context),
        _code(PREPARE, context),
        _code(
            """
from src.models import build_model  # noqa: E402

MODEL = build_model(CONFIG, feature_names=PREPARED["feature_names"])
MODEL.fit(PREPARED["X_train"], PREPARED["y_train"],
          X_val=PREPARED["X_val"], y_val=PREPARED["y_val"], callbacks=[])
print(MODEL.summary())
""",
            context,
        ),
        _md("## 1. Évaluation complète sur le split de test"),
        _code(_EVALUATE_CELL, context),
        _insight(
            [
                f"La métrique de décision est `{spec.metrics.primary}` = **{RESULT_PLACEHOLDER}**. "
                "Elle se lit avec le gain sur le naif saisonnier : sans cette comparaison, un MAPE "
                "de 4 % est impossible à qualifier.",
                "Deux « couvertures » cohabitent et ne mesurent pas la même chose : la **bande de "
                "tolérance** (part des prévisions à moins de ±5 % du réalisé, lecture métier) et la "
                "**couverture d'intervalle** (part des réalisations dans l'intervalle publié, "
                "lecture statistique). Les confondre est une erreur classique de revue.",
                "Le `context` passé à `evaluate()` apporte les colonnes brutes (régime, date cible, "
                "échelle de dispersion) : sans lui, la ventilation par régime serait impossible.",
            ],
        ),
        _md("## 2. Erreur par horizon : quatre produits, pas un"),
        _code(_PER_HORIZON_CELL, context),
        _insight(
            [
                "L'erreur croît avec l'horizon pour **deux** raisons distinctes : la prévision météo "
                "se dégrade, et l'information de court terme (dernière valeur connue) perd en "
                "pertinence. Les séparer demande de regarder la colonne de prévision de "
                "température, pas seulement le MAPE.",
                "Le gain sur le naif **décroît** avec l'horizon : c'est attendu, le naif saisonnier "
                "est d'autant plus fort que la cible est proche d'une semaine déjà observée.",
                "Publier un seul chiffre pour tous les horizons reviendrait à garantir le J+7 au "
                "prix du J+1. L'erreur attendue est donc publiée **par horizon** avec chaque "
                "prévision.",
            ],
        ),
        _md("## 3. Erreur par régime : là où se perd l'argent"),
        _code(_PER_REGIME_CELL, context),
        _insight(
            [
                "La colonne « part de l'erreur » est celle qui décide : un régime à 4 % des lignes "
                "mais 20 % de l'erreur totale est une priorité, alors qu'un régime à 20 % des "
                "lignes et 10 % de l'erreur ne l'est pas.",
                "Le **biais** par régime est plus parlant que le MAPE : une sous-prévision "
                "systématique en vague de froid fait acheter en urgence au prix spot, une "
                "sur-prévision fait sur-engager de la production. Les deux coûtent, pas de la même "
                "façon.",
                "Un régime jamais vu en entraînement ne se corrige pas par plus d'arbres : il se "
                "corrige par une variable de régime, ou se signale par un indicateur de confiance "
                "qui déclenche une revue humaine.",
            ],
        ),
        _md("## 4. Erreur par saison : le biais qui se corrige autrement"),
        _code(_PER_SEASON_CELL, context),
        _insight(
            [
                "Un biais concentré sur quelques mois est presque toujours un effet saisonnier mal "
                "appris (thermo-sensibilité sous-estimée en hiver, effet vacances sur-estimé en "
                "été).",
                "Le correctif n'est pas plus de complexité : c'est une variable de régime, un "
                "recalibrage saisonnier, ou une feature de profil par mois — toutes choses que le "
                "constructeur de features déclaratif sait produire.",
                "Un biais **stable** sur tous les mois est un autre problème : il se corrige par la "
                "perte d'entraînement ou par un terme correctif global.",
            ],
        ),
        _md("## 5. Les pires journées, lues une par une"),
        _code(_WORST_CELL, context),
        _insight(
            [
                "Une erreur isolée sur un arrêt industriel inédit n'appelle pas le même correctif "
                "qu'une erreur systématique sur tous les lundis de vacances scolaires : la "
                "composition des 50 pires lignes par régime tranche la question.",
                "Le P95 d'erreur se lit avec le MAPE : un P95 très au-dessus du MAPE moyen signale "
                "une queue lourde, donc un risque opérationnel que la moyenne cache.",
                "Ces journées sont candidates à la revue humaine **avant** publication : le "
                "prédicteur produit déjà l'indicateur de confiance qui les signale.",
            ],
        ),
        _md("## 6. Autocorrélation des résidus : ce qui reste de structure"),
        _code(_ACF_CELL, context),
        _insight(
            [
                "Un résidu **autocorrélé n'est pas du bruit** : c'est de la dépendance temporelle "
                "que le modèle n'a pas capturée. À décalage 1, cela appelle un terme AR ou un "
                "décalage supplémentaire ; à décalage 7, un profil hebdomadaire mieux appris.",
                "Le seuil ±2/√n est la référence de bruit blanc : en dessous, l'autocorrélation "
                "observée est compatible avec le hasard, et il ne faut rien corriger.",
                "Une autocorrélation résiduelle a une seconde conséquence, plus coûteuse : elle "
                "invalide l'hypothèse d'indépendance sur laquelle repose un intervalle gaussien. "
                "C'est une raison de plus de calibrer les bornes sur des quantiles empiriques.",
            ],
        ),
        _md(
            """## 7. Intervalles : couverture réelle, ventilation et recalibrage adaptatif

La couverture nominale est un engagement. Cette section mesure ce qui est **réellement** couvert,
par horizon et par régime, puis exécute le recalibrage quotidien qu'un système réel peut faire
puisque l'erreur d'hier est connue ce matin."""
        ),
        _code(_INTERVALS_CELL, context),
        _insight(
            [
                "La couverture globale est la moyenne de deux populations très différentes : les "
                "jours calmes, proches du nominal, et les jours de régime exceptionnel, où "
                "l'intervalle ne protège **rien**. Publier le seul chiffre global serait trompeur.",
                "Ce n'est pas un défaut de calibration mais sa limite structurelle : la fenêtre de "
                "validation ne contient pas le régime, donc aucun quantile appris sur elle ne peut "
                "le couvrir. Élargir la fenêtre pour couvrir toutes les saisons est le correctif de "
                "fond.",
                "La normalisation par le niveau (`INTERVAL_SCALE`) est ce qui rend l'intervalle "
                "utilisable d'une saison à l'autre : le bruit de la série est multiplicatif, donc "
                "une largeur absolue calibrée en été est deux fois trop étroite en hiver.",
                "Le recalibrage adaptatif gagne quelques points sans élargir davantage l'intervalle, "
                "parce qu'il suit la dispersion **récente** au lieu de la dispersion moyenne de la "
                "fenêtre de calibration. Son coût est nul : il réutilise les résidus déjà publiés.",
                "Il ne corrige toutefois pas les régimes inédits — c'est le rôle de l'indicateur de "
                "confiance et de la revue humaine, pas de l'intervalle.",
            ],
        ),
        _md("## 8. Facteurs contributifs"),
        _code(_DRIVERS_CELL, context),
        _insight(
            [
                "L'importance par permutation dit **quelle** variable compte, pas dans quel sens ni "
                "en quelle unité : pour la thermo-sensibilité en MW/°C, il faut repasser par le "
                "modèle linéaire du notebook 04.",
                "Le regroupement par famille (météo / calendrier / historique) est plus utile que le "
                "classement individuel : il dit quelle **source** de données mérite un "
                "investissement (un meilleur contrat météo, un calendrier plus fin).",
                "Une variable dominante qui serait une métadonnée du jour cible signerait une "
                "fuite : c'est le dernier contrôle du contrat d'antériorité.",
            ],
        ),
        _md("## 9. Diagnostics consolidés"),
        _code(_DIAGNOSTICS_CELL, context),
        _insight(
            [
                "Le backtest à modèle figé mesure la **stabilité temporelle**, pas la performance : "
                "une dérive croissante du premier au dernier repli annonce un modèle qui vieillit.",
                "La dispersion entre replis est un critère de succès à part entière — un modèle "
                "instable coûte plus cher à exploiter qu'un modèle légèrement moins précis et "
                "stable.",
                "Les segments déclarés dans la configuration donnent une lecture complémentaire "
                "(niveau de consommation, jour de la semaine) sans avoir à réécrire le notebook.",
            ],
        ),
        _md(
            f"""## 10. Recommandations

Les recommandations ci-dessous sont produites par le générateur de rapport à partir des mesures de
ce notebook. Celles déclarées dans le manifeste, à titre de référence métier :

{recommendation_lines}
"""
        ),
        _code(_RECOMMENDATIONS_CELL, context),
        _md("## 11. Rapport exécutable"),
        _code(_REPORT_CELL, context),
        _insight(
            [
                "Le rapport mélange **chiffres calculés** et **verdict sur les seuils déclarés** : "
                "il est régénérable à chaque run et ne dépend d'aucune saisie manuelle.",
                "Le même contenu est écrit en Markdown (lecture), en JSON (CI, dashboard) et en CSV "
                "(tableurs, revue métier) : trois consommateurs, une seule source de vérité.",
                "Ce qui reste ouvert, assumé : une seule région, une prévision météo sans "
                "scénarios P10/P50/P90, et un backtest à modèle figé en routine. Chacun de ces "
                "points est un chantier identifié, pas un angle mort.",
            ],
        ),
    ]
    return write_notebook(Path(destination) / "06_error_analysis.ipynb", cells)
