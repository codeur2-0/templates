# Prévision de consommation électrique multi-horizons avec scikit-learn

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-classic-ml-orange)
![Hydra](https://img.shields.io/badge/config-Hydra-89b482)
![Pandera](https://img.shields.io/badge/validation-Pandera-16a34a)
![pytest](https://img.shields.io/badge/tests-pytest-2ea44f)
![Licence](https://img.shields.io/badge/license-MIT-lightgrey)

> **Domaine** : `data-science` · **Problématique** : `time-series-forecasting` · **Stack** : `scikit-learn`
> **Emplacement** : `data-science/time-series-forecasting/with-sklearn/`
> **Temps de lecture** : ~15 min · **Temps d'exécution complet** : < 5 min sur un laptop

---

## 1. À propos / Objectifs

Pipeline de prévision de bout en bout (série quotidienne synthétique de consommation régionale ->
expansion en couples origine/horizon avec contrat d'antériorité -> contrats Pandera -> features de
décalage, de momentum et de thermo-sensibilité -> gradient boosting par histogrammes -> MAPE et MASE
ventilés par horizon, backtest à origine glissante, comparaison aux références naïves, intervalles
par quantiles de résidus, publication des sept prochains jours) pour une équipe Prévisions,
entièrement piloté par Hydra et structuré en objets testables. La vérité terrain du jour cible
(`load_mw`, `event_type`) est une métadonnée : elle sert à mesurer, jamais à entraîner.

Le réseau doit annoncer chaque matin sa consommation prévisionnelle pour les sept prochains jours :
c'est cette annonce qui pilote les achats sur les marchés de gros, le programme d'appel des groupes
de production et les alertes « équilibre menace » en cas d'écart. Aujourd'hui la prévision est
construite à la main à partir d'un profil hebdomadaire moyen et d'un ajustement météo au jugement de
l'analyste. Résultat : un écart moyen de 4 à 6 % qui double pendant les vagues de froid, alors que
chaque point d'erreur sur 2 500 MW représente environ 25 MW à compenser en urgence — au prix spot du
jour, souvent le plus élevé de l'année.

**Ce que cet exemple démontre**

1. Construire un jeu supervisé par expansion temporelle (origine x horizon) et formaliser le contrat d'antériorité de chaque feature : connue à l'origine, connue par avance, ou interdite.
2. Comprendre pourquoi un split chronologique s'impose et ce que coûte concrètement une validation croisée aléatoire sur une série temporelle.
3. Comparer un modèle appris à trois références triviales (persistance, naif saisonnier, moyenne glissante) et quantifier la valeur ajoutée réelle plutôt que le R².
4. Choisir les bonnes métriques de prévision : MAPE et ses pièges (asymétrie, division par zéro), sMAPE, MASE, et toujours ventiler par horizon.
5. Valider par backtest à origine glissante et lire la stabilité de l'erreur dans le temps plutôt qu'un score unique.
6. Modéliser la thermo-sensibilité (MW par degré) et les saisonnalités (annuelle, hebdomadaire, jours fériés, vacances scolaires) sans les confondre.
7. Traiter les manquants d'une variable exogène (capteur météo en panne, biaisés vers l'hiver) sans imputation silencieuse.
8. Passer d'une prévision ponctuelle à un intervalle exploitable, construit par horizon, et mesurer sa couverture réelle.
9. Diagnostiquer les résidus d'une prévision : autocorrélation, saisonnalité résiduelle, biais par régime, par horizon et par mois.
10. Analyser les erreurs sur les épisodes extrêmes (vague de froid, canicule, arrêt industriel) et produire des recommandations opérationnelles pour l'astreinte et les achats.
11. Industrialiser l'ensemble : configuration Hydra, artefacts traçables, tests pytest, rapports générés, prévision batch et unitaire.

**Pourquoi scikit-learn ici ?**

- Référence absolue pour le ML tabulaire : pipelines composables, transformers custom, métriques.
- `ColumnTransformer` évite toute fuite de données : le scaling/encodage est appris sur le train uniquement.

---

## 2. Stack technologique

| Rôle | Outil | Version minimale | Pourquoi |
| --- | --- | --- | --- |
| Framework ML | **scikit-learn** | joblib>=1.3 | classic-ml |
| Configuration | **Hydra** (`hydra-core`) | 1.3 | Composition YAML, overrides CLI, multirun |
| Validation des données | **Pandera** | 0.20 | Contrats de données exécutables (`DataFrameModel`) |
| Typage de la config | **Pydantic** | 2.5 | Échec immédiat sur une configuration invalide |
| Données | **pandas**, **numpy**, **pyarrow** | 2.2 / 1.26 / 15.0 | Parquet typé et compressé |
| Visualisation | **matplotlib**, **seaborn** | 3.8 / 0.13 | Figures reproductibles (rapports + notebooks) |
| Logging | **loguru** | 0.7 | Logs structurés, rotation, interception stdlib |
| Progression | **tqdm** | 4.66 | Barres de progression des entraînements |
| Tests | **pytest** | 8.0 | Unitaires + smoke de bout en bout |
| Qualité | **ruff**, **mypy** | 0.5 / 1.10 | Lint/format unique, typage statique |

Dépendances exactes : [`requirements.txt`](requirements.txt) et [`pyproject.toml`](pyproject.toml).

---

## 3. Cas d'usage

Le réseau doit annoncer chaque matin sa consommation prévisionnelle pour les sept prochains jours :
c'est cette annonce qui pilote les achats sur les marchés de gros, le programme d'appel des groupes
de production et les alertes « équilibre menace » en cas d'écart. Aujourd'hui la prévision est
construite à la main à partir d'un profil hebdomadaire moyen et d'un ajustement météo au jugement de
l'analyste. Résultat : un écart moyen de 4 à 6 % qui double pendant les vagues de froid, alors que
chaque point d'erreur sur 2 500 MW représente environ 25 MW à compenser en urgence — au prix spot du
jour, souvent le plus élevé de l'année.

| | |
| --- | --- |
| **Qui** (persona / consommateur) | Équipe Prévisions d'un gestionnaire de réseau de distribution électrique (data scientist qui industrialise le modèle, analyste marché qui engage les achats d'énergie, et astreinte conduite qui équilibre l'offre et la demande en J+1). |
| **Problème** | Prédire la consommation électrique quotidienne régionale (en MW) pour les horizons J+1, J+2, J+3 et J+7, à partir de l'histoire de consommation disponible **au matin de l'origine** et de la prévision météo connue à ce même instant, puis publier une prévision assortie d'un intervalle et d'un indicateur de confiance. |
| **Entrées** | Historique de consommation quotidienne (décalages 1, 2, 7, 14 et 28 jours, moyennes et dispersion glissantes 7 et 28 jours), ancrage saisonnier (même jour de la semaine précédente), prévision de température pour le jour cible et son écart à la normale saisonnière, degrés-jours de chauffe et de froid, calendrier du jour cible (jour de la semaine, week-end, jour férié, vacances scolaires), horizon visé. |
| **Sorties** | Consommation prévisionnelle en MW par horizon, écart relatif attendu (MAPE par horizon), intervalle de prévision, indicateur de confiance (régime stable / épisode extrême / rupture d'historique) et facteurs dominants de la prévision (température, calendrier, niveau récent). |
| **Valeur métier** | Réduction des achats d'urgence et des pénalités d'écart, meilleure anticipation des pointes de froid et de canicule, traçabilité de la prévision (chiffre, intervalle, raisons) devant le régulateur, et détection automatique des jours où la prévision doit être revue par un humain. |
| **Cadence** | Prévision publiée chaque matin à 06h00 pour les sept jours à venir ; ré-entraînement hebdomadaire sur l'historique glissant, et recalibrage immédiat après un épisode extrême. |

**Objectif de modélisation** — Prédire la consommation électrique quotidienne régionale (en MW) pour les horizons J+1, J+2,
J+3 et J+7, à partir de l'histoire de consommation disponible **au matin de l'origine** et
de la prévision météo connue à ce même instant, puis publier une prévision assortie d'un
intervalle et d'un indicateur de confiance.

**Critères de réussite**

- [x] MAPE ≤ 4,0 % sur le split de test chronologique, tous horizons confondus. Mesuré : 3,692 % (J+1 3,555 | J+2 3,733 | J+3 3,596 | J+7 3,886), pour un plancher structurel de bruit à ~2,8 %.
- [x] Le modèle doit battre le **naif saisonnier** (même jour de la semaine précédente) d'au moins 30 % en MAPE : c'est le seul seuil qui prouve une valeur ajoutée. Mesuré : +59,1 % (9,029 % -> 3,692 %).
- [x] MAPE ≤ 6,0 % à l'horizon J+7 : la dégradation avec l'horizon doit rester maîtrisée (le J+7 pilote les achats hebdomadaires). Mesuré : 3,886 %, soit 0,33 point au-dessus du J+1.
- [x] MASE ≤ 0,75 : l'erreur doit être inférieure à celle du naif saisonnier pris comme référence d'échelle. Mesuré : 0,420.
- [x] Biais moyen < 1,5 % et aucun mois dont le biais dépasse 4 % : une prévision systématiquement basse fait acheter trop cher en urgence. Mesuré : +0,887 % en moyenne, pire mois décembre à 3,261 %.
- [x] Couverture de l'intervalle de prévision ≥ 85 % pour un niveau nominal de 90 %. Mesuré : 85,42 % avec le conformal normalisé (quantiles de résidus 66,5 %, conformal 73,2 %, adaptatif 79,9 %).
- [x] Reproductibilité : deux exécutions avec la même graine produisent exactement les mêmes métriques. Mesuré : métriques identiques au bit près ; dispersion inter-replis du backtest 0,491.

**Contraintes**

- Aucune donnée réelle de consommation : le dataset de ce dépôt est synthétique et hors ligne.
- Interdiction absolue de fuite temporelle : une feature ne peut utiliser que l'information disponible au matin de l'origine (historique) ou connue par avance (calendrier, prévision météo).
- Le split d'évaluation est chronologique : jamais de mélange aléatoire, jamais de validation croisée aléatoire sur des séries temporelles.
- Le modèle doit s'exécuter sur CPU en moins de 5 secondes pour les 7 horizons d'une journée (contrainte d'astreinte). Mesuré : 49 ms en moyenne sur 20 appels, soit une marge d'un facteur 100.
- Les épisodes extrêmes (vague de froid, canicule, arrêt industriel) ne doivent pas être rognés : ce sont précisément les jours où la prévision vaut de l'argent.

---

## 4. Données

Les données sont **synthétiques**, générées localement par
[`src/data/generators.py`](src/data/generators.py) : aucune donnée réelle, aucun téléchargement,
aucune information confidentielle. Elles sont néanmoins construites pour être **crédibles
métier** (corrélations réalistes, bruit, valeurs manquantes, outliers légitimes).

| Propriété | Valeur |
| --- | --- |
| Jeu de données | `regional_electricity_load` |
| Volume par défaut | 4 800 lignes |
| Formats écrits | parquet, csv |
| Emplacement | `data/raw/regional_electricity_load.parquet` (et `.csv`) |
| Cible | `load_mw` |
| Identifiant | `sample_id` |
| Colonne temporelle | `origin_date` |
| Graine | `42` (reproductible) |

**Schéma**

| Colonne | Type | Rôle | Signification métier |
| --- | --- | --- | --- |
| `sample_id` | str | identifier | Identifiant unique du couple (origine, horizon) |
| `origin_date` | datetime | timestamp | Date de l'origine : dernier jour dont la consommation est connue au moment de la prévision |
| `horizon_days` | int | feature | Horizon visé en jours (1, 2, 3 ou 7) — le modèle apprend la dégradation avec l'horizon (jours) |
| `target_date` | datetime | metadata | Date prévue (origine + horizon) — sert à l'analyse d'erreur, jamais au modèle |
| `target_month` | str | metadata | Mois du jour cible, pour la ventilation saisonnière des erreurs |
| `event_type` | str | metadata | Nature de l'épisode touchant le jour cible (aucun, vague de froid, canicule, arrêt industriel) |
| `is_extreme_event` | bool | metadata | Vrai si le jour cible appartient à un épisode extrême — cible prioritaire de l'analyse d'erreur |
| `load_mw` | float | target | Consommation électrique régionale observée le jour cible (MW) |
| `load_last_observed` | float | feature | Dernière consommation connue : le jour de l'origine lui-même (ancre de persistance) (MW) |
| `load_lag_1d` | float | feature | Consommation un jour avant l'origine (MW) |
| `load_lag_2d` | float | feature | Consommation deux jours avant l'origine (MW) |
| `load_lag_7d` | float | feature | Consommation sept jours avant l'origine (même jour de la semaine que l'origine) (MW) |
| `load_lag_14d` | float | feature | Consommation quatorze jours avant l'origine (MW) |
| `load_lag_28d` | float | feature | Consommation vingt-huit jours avant l'origine (niveau de référence mensuel) (MW) |
| `load_rolling_mean_7d` | float | feature | Moyenne glissante 7 jours de la consommation, terminée à l'origine (MW) |
| `load_rolling_std_7d` | float | feature | Écart-type glissant 7 jours : volatilité récente du réseau (MW) |
| `load_rolling_min_7d` | float | feature | Minimum glissant 7 jours : plancher récent (détecte un creux de vacances) (MW) |
| `load_rolling_mean_28d` | float | feature | Moyenne glissante 28 jours : niveau de fond mensuel (MW) |
| `load_seasonal_naive` | float | feature | Ancrage saisonnier : consommation du même jour de la semaine, une semaine avant le jour cible (toujours ≤ origine, donc licite) (MW) |
| `temperature_forecast_c` | float | feature | Prévision de température moyenne pour le jour cible, disponible à l'origine — entachée d'une erreur croissante avec l'horizon (°C) |
| `temperature_forecast_prev_c` | float | feature | Prévision de température de la veille du jour cible (observation à J+1) — permet au modèle de reconstruire l'inertie thermique du bâti (°C) |
| `temperature_anomaly_c` | float | feature | Écart de la prévision de température à la normale saisonnière du jour cible (°C) |
| `temperature_lag_1d` | float | feature | Température observée à l'origine (°C) |
| `temperature_rolling_mean_7d` | float | feature | Moyenne glissante 7 jours de la température observée, terminée à l'origine (°C) |
| `hdd_target` | float | feature | Degrés-jours de chauffe du jour cible : max(0, 18 °C - température prévue) (°C) |
| `cdd_target` | float | feature | Degrés-jours de froid du jour cible : max(0, température prévue - 24 °C) (°C) |
| `target_weekday` | str | feature | Jour de la semaine du jour cible (connu par avance) |
| `target_is_weekend` | bool | feature | Vrai si le jour cible tombe un samedi ou un dimanche |
| `target_is_holiday` | bool | feature | Vrai si le jour cible est un jour férié national français |
| `target_school_holiday` | bool | feature | Vrai si le jour cible tombe pendant les vacances scolaires (effet résidentiel marqué) |
| `target_doy_sin` | float | feature | Composante sinus du jour de l'année cible (encodage cyclique de la saisonnalité annuelle) |
| `target_doy_cos` | float | feature | Composante cosinus du jour de l'année cible |

Détail complet (distributions attendues, remarques) : [`data/README.md`](data/README.md).

---

## 5. Arborescence

```text
data-science/time-series-forecasting/with-sklearn/
├── README.md                    # ← ce fichier
├── pyproject.toml               # métadonnées, dépendances, ruff / mypy / pytest
├── requirements.txt             # dépendances runtime
├── requirements-dev.txt         # dépendances de développement
├── Makefile                     # make data / train / evaluate / predict / verify
├── .gitignore
│
├── conf/                        # configuration Hydra (source de vérité)
│   ├── config.yaml              # config principale + defaults
│   ├── data/default.yaml        # dataset, cible, split, contrats de validation
│   ├── model/default.yaml       # algorithme + hyperparamètres
│   ├── train/default.yaml       # boucle d'entraînement, callbacks, artefacts
│   ├── preprocessing/default.yaml  # nettoyage, encodage, scaling, features dérivées
│   └── hydra/local.yaml         # répertoires de sortie Hydra (run / sweep)
│
├── data/                        # données (générées, jamais committées)
│   ├── README.md                # documentation métier du dataset
│   ├── raw/                     # données brutes synthétiques
│   ├── processed/               # données transformées (parquet)
│   └── external/                # données tierces
│
├── src/                         # code source (package Python, 100 % typé)
│   ├── main.py                  # point d'entrée @hydra.main
│   ├── data/                    # loaders, schémas Pandera, générateur synthétique
│   ├── preprocessing/           # transformers custom + pipeline (fit/transform/save)
│   ├── features/                # feature engineering piloté par la configuration
│   ├── models/                  # BaseModel (ABC) + implémentation scikit-learn
│   ├── training/                # Trainer, callbacks, métriques
│   ├── evaluation/              # Evaluator + génération de rapports
│   ├── inference/               # Predictor (batch + enregistrement unitaire)
│   ├── pipelines/               # orchestration bout-en-bout (4 pipelines)
│   ├── schemas/                 # configuration typée (Pydantic)
│   ├── utils/                   # logging, chemins, IO, helpers
│   └── visualization/           # figures (rapports + notebooks)
│
├── scripts/                     # CLI minces (compose la config Hydra puis délègue à src/)
│   ├── generate_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── notebooks/                   # 6 notebooks pédagogiques exécutables
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_data_validation_and_schemas.ipynb
│   ├── 03_preprocessing_and_features.ipynb
│   ├── 04_model_exploration.ipynb
│   ├── 05_training_and_tracking.ipynb
│   └── 06_evaluation_and_error_analysis.ipynb
│
├── tests/                       # pytest (schémas, loaders, preprocessing, modèle, training)
│   ├── conftest.py
│   ├── test_data_schemas.py
│   ├── test_loaders.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_training.py
│
└── artifacts/                   # produits générés (ignorés par git)
    ├── models/                  # modèle + pipeline de preprocessing + model card
    ├── metrics/                 # métriques JSON
    ├── reports/                 # rapports Markdown/HTML + prédictions
    └── figures/                 # graphiques PNG
```

---

## 6. Architecture & design

### 6.1 Principes directeurs

| Principe | Traduction concrète dans ce projet |
| --- | --- |
| **SRP** | Un module = une responsabilité : `data` charge, `preprocessing` transforme, `training` entraîne, `evaluation` mesure, `inference` prédit. |
| **DIP** | Tout dépend de l'abstraction `BaseModel` (`src/models/base.py`), jamais d'un framework précis. |
| **OCP** | Ajouter un algorithme = ajouter une classe, sans modifier le trainer ni l'évaluateur. |
| **Config as code** | Aucune constante métier dans le code : tout vient de `conf/` (Hydra) et est validé par Pydantic. |
| **Data contracts** | Chaque DataFrame traverse un schéma Pandera avant d'être consommé. |
| **Reproductibilité** | Graine fixée partout, artefacts horodatés, données régénérables à l'identique. |
| **Testabilité** | Les classes reçoivent leurs dépendances par le constructeur (injection), pas de singleton caché. |

### 6.2 Classes principales

```text
AppConfig (Pydantic)                 ← conf/*.yaml validé et typé
   │
   ├── ProjectPaths                  ← arborescence disque (résolue depuis src/, pas depuis le CWD)
   │
   ├── SyntheticDataGenerator        ← produit data/raw/*.parquet (+ .csv)
   ├── RawDataLoader / ProcessedDataLoader ← lecture + validation Pandera
   ├── FeatureBuilder                ← features dérivées déclaratives (ratio, bin, interaction…)
   ├── PreprocessingPipeline         ← ColumnTransformer : imputation, clipping, scaling, encodage
   │
   ├── SklearnModel(BaseModel)   ← implémentation scikit-learn
   │        fit / predict / predict_proba / save / load
   ├── Trainer                       ← split, callbacks, métriques, artefacts
   ├── Evaluator                     ← métriques + matrice/rapport d'erreurs
   ├── ReportBuilder                 ← Markdown + figures dans artifacts/
   └── Predictor                     ← inférence validée (batch + 1 enregistrement)
```

### 6.3 Flux de bout en bout

```text
mode=generate-data → SyntheticDataGenerator → data/raw/*.parquet
mode=train         → RawDataLoader (validation Pandera)
                     → FeatureBuilder → PreprocessingPipeline.fit_transform
                     → SklearnModel.fit (callbacks)
                     → Evaluator.evaluate → ReportBuilder → artifacts/{models,metrics,reports,figures}
mode=evaluate      → chargement des artefacts → nouvelle évaluation + rapports
mode=predict       → InferenceDataSchema → Predictor.predict → artifacts/reports/predictions.csv
```

### 6.4 Comparaison avec les autres stacks

Le **même cas d'usage** est implémenté avec d'autres frameworks dans les dossiers voisins.
La structure, les contrats de données et les métriques étant identiques, la comparaison est
directe (qualité, temps d'entraînement, lisibilité du code) :

- `../../with-sklearn/` — sklearn
- `../../with-xgboost/` — xgboost
- `../../with-lightgbm/` — lightgbm
- `../../with-pytorch/` — pytorch
- `../../with-keras/` — keras

---

## 7. Prérequis

- **Python ≥ 3.10** (3.11 ou 3.12 recommandés)
- `pip` ≥ 23 (ou `uv`, ou `conda`)
- ~1 Go d'espace disque pour les dépendances, < 50 Mo pour les données et artefacts
- Aucun GPU nécessaire, **aucun accès réseau** requis après installation des dépendances

---

## 8. Installation

**Option A — venv (recommandé)**

```bash
cd data-science/time-series-forecasting/with-sklearn
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-dev.txt   # runtime + tests + lint + notebooks
```

**Option B — uv (plus rapide)**

```bash
cd data-science/time-series-forecasting/with-sklearn
uv venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

**Option C — installation éditable du projet**

```bash
cd data-science/time-series-forecasting/with-sklearn
pip install -e ".[dev]"
```

Vérification :

```bash
python -c "import src, pandera, hydra; print('environnement OK')"
```

---

## 9. Génération des données

```bash
make data
# ou, de façon équivalente :
python scripts/generate_data.py
python -m src.main mode=generate-data
```

Options utiles :

```bash
python scripts/generate_data.py data.n_samples=10000          # plus de volume
python scripts/generate_data.py seed=7                        # autre tirage
python scripts/generate_data.py data.formats=[parquet]        # Parquet uniquement
```

Résultat : `data/raw/regional_electricity_load.parquet` (+ `.csv` pour la lecture humaine)
et un journal de génération détaillé (lignes, colonnes, taux de manquants, distribution de la cible).

---

## 10. Utilisation

### (a) Via le point d'entrée Hydra `src/main.py`

```bash
# Enchaînement complet : données → entraînement → évaluation → prédiction
python -m src.main mode=all

# Entraînement seul
python -m src.main mode=train

# Overrides Hydra (aucun fichier à modifier)
python -m src.main mode=train ++train.epochs=10 data.n_samples=2000 log_level=DEBUG
python -m src.main mode=train model.params.max_iter=250

# Afficher la configuration composée sans rien exécuter
python -m src.main --cfg job

# Sweep (multirun) sur plusieurs valeurs
python -m src.main --multirun data.n_samples=1000,4000 seed=1,2
```

### (b) Via les scripts CLI

```bash
python scripts/generate_data.py            # 1. données
python scripts/train.py                    # 2. entraînement + évaluation + artefacts
python scripts/evaluate.py                 # 3. rapports et figures
python scripts/predict.py                  # 4. prédictions sur un échantillon
python scripts/predict.py predict.input=data/raw/regional_electricity_load.csv predict.n_samples=10
```

### (c) Depuis un notebook (ou du code Python)

```python
from hydra import compose, initialize_config_dir
from src.main import main_flow

with initialize_config_dir(config_dir="conf", version_base=None):
    cfg = compose(config_name="config", overrides=["mode=train", "data.n_samples=1500"])

results = main_flow(cfg)
print(results["train"].metrics)
```

Ou directement avec les classes (sans Hydra) :

```python
from src.data.generators import SyntheticDataGenerator
from src.models.model import SklearnModel
from src.pipelines import TrainPipeline
from src.schemas.config import validate_config

data = SyntheticDataGenerator(n_samples=1000, seed=42).run()
```

### (d) Via le Makefile

```bash
make help        # liste des cibles
make all         # données → train → éval → prédiction
make smoke       # exécution rapide (petit dataset)
make verify      # lint + mypy + pytest + smoke
```

---

## 11. Configuration

Toute la configuration vit dans `conf/` et est composée par Hydra :

| Fichier | Contenu | Exemples d'override |
| --- | --- | --- |
| `conf/config.yaml` | racine : `mode`, `seed`, `paths`, `metrics`, `project` | `mode=evaluate`, `seed=7`, `log_level=DEBUG` |
| `conf/data/default.yaml` | dataset, cible, colonnes ignorées, contrats de validation | `data.n_samples=5000`, `data.validation.strict=false` |
| `conf/model/default.yaml` | algorithme et hyperparamètres | `model.params.max_iter=…` |
| `conf/train/default.yaml` | split, epochs, callbacks, noms d'artefacts | `++train.epochs=20`, `train.split.test_size=0.25` |
| `conf/preprocessing/default.yaml` | imputation, scaling, encodage, features dérivées | `preprocessing.numeric.scaler=robust`, `preprocessing.categorical.encoder=ordinal` |
| `conf/hydra/local.yaml` | répertoires de sortie Hydra (`outputs/`, `multirun/`) | `hydra.run.dir=outputs/debug` |
| `conf/config.yaml` -> bloc `load_forecasting` | Réglages métier de la famille, au même niveau que `mode` et `seed` : 9 clés (horizon_column, horizons, long_horizon, interval_level, interval_method, interval_scale_column, backtest_folds, …), chacune commentée dans le fichier — c'est là qu'on change le métier sans toucher au code. | `load_forecasting.horizon_column=horizon_days`, `load_forecasting.long_horizon=7` |

Règles appliquées :

1. **Aucune valeur magique** dans le code : tout paramètre est déclaré dans `conf/`.
2. La configuration est **validée et typée** au démarrage (`src/schemas/config.py` → Pydantic).
3. `hydra.job.chdir=false` : le répertoire courant n'est jamais modifié, les chemins sont
   résolus depuis `src/utils/paths.py` (donc identiques en CLI, en notebook et en test).
4. `++` ajoute une clé absente du schéma ; sans `++` la clé doit déjà exister.

---

## 12. Résultats attendus

Après `make all`, le dépôt local contient :

| Chemin | Contenu |
| --- | --- |
| `data/raw/regional_electricity_load.parquet` (+ `csv`) | jeu de données synthétique, 4 800 lignes |
| `data/raw/generation_metadata.json` | recette de génération : graine, options, empreinte du jeu, fichiers écrits |
| `data/processed/split_{train,val,test}.parquet` | splits avant transformation (l'évaluation et l'inférence rejouent exactement le même découpage) |
| `data/processed/features_{X_train,X_val,X_test}.parquet` | matrices prêtes pour le modèle |
| `artifacts/models/model.joblib` | modèle entraîné |
| `artifacts/models/preprocessing.joblib` | pipeline de preprocessing ajusté (aucune fuite) |
| `artifacts/models/feature_builder.joblib` | construction des features dérivées, ajustée sur le train uniquement |
| `artifacts/models/model_card.json` | carte du modèle (params, métriques, features, date) |
| `artifacts/models/resolved_config.json` | configuration Hydra résolue : l'artefact entraîné porte sa recette exacte |
| `artifacts/metrics/training_metrics.json` | métriques d'entraînement et de validation |
| `artifacts/metrics/evaluation_metrics.json` | métriques sur le split de test + verdict des seuils |
| `artifacts/reports/evaluation_report.md` | rapport lisible (métriques, analyse d'erreurs, recommandations) |
| `artifacts/reports/per_horizon.csv` | MAPE / MAE / biais / couverture ventilés par horizon |
| `artifacts/reports/segments.csv` | ventilation par régime extrême, week-end, jour férié, vacances |
| `artifacts/reports/baselines.csv` | références triviales (persistance, naif saisonnier, moyennes glissantes) |
| `artifacts/reports/intervals.csv` | bornes, largeur relative et méthode de calibration par prévision |
| `artifacts/reports/backtest.csv` | replis chronologiques et dispersion de la performance |
| `artifacts/reports/errors.csv` | pires erreurs, avec leurs références naïves |
| `artifacts/reports/predictions.csv` | prédictions sur l'échantillon de démonstration |
| `artifacts/figures/*.png` | prévision vs réel, erreur par horizon et par régime, couverture d'intervalle, diagnostics de résidus, biais mensuel, stabilité du backtest |
| `outputs/<date>/<heure>/` | configuration composée + logs Hydra |

Métrique principale : **`mape`** (sens `minimize`, seuil de smoke test : ≤ 4.0).
Métriques secondaires : smape, mase, mae, rmse, r2.

Résultats de l'exécution de référence (`make all`, graine 42) :

| Indicateur | Valeur mesurée |
| --- | --- |
| MAPE global (test, 960 lignes) | **3,692 %** |
| MAE | **99,4 MW** |
| sMAPE / MASE / R² | **3,67 % / 0,420 / 0,953** |
| Biais moyen (pire mois : décembre) | **+0,887 % (3,261 %)** |
| MAPE du naif saisonnier -> gain du modèle | **9,029 % -> +59,1 %** |
| Couverture d'intervalle (nominal 90 %) | **85,42 %** |
| MAPE J+1 / J+2 / J+3 / J+7 | **3,555 / 3,733 / 3,596 / 3,886 %** |
| MAPE sur les 20 jours d'épisode extrême | **9,42 % (biais -9,22 %, limite documentée)** |
| Dispersion inter-replis du backtest | **0,491** |
| Latence d'inférence pour une journée (7 horizons) | **49 ms** |
| Features en entrée du modèle | **41 (32 colonnes brutes + encodage)** |

Ces valeurs sont reproductibles à l'identique ; elles proviennent du split de test, jamais
du split d'entraînement, et le détail complet est dans `artifacts/reports/evaluation_report.md`.

---

## 13. Notebooks

Tous les notebooks sont **exécutables de bout en bout** (`make notebooks`) et documentés
cellule par cellule, comme un support de formation pour juniors.


| Notebook | Ce qu'on y apprend |
| --- | --- |
| `01_eda.ipynb` | EDA structurée : types, manquants, distributions univariées et bivariées, corrélations, outliers, puis **10-15 insights** actionnables. |
| `02_validation.ipynb` | Pourquoi des contrats de données : définition d'un `DataFrameModel` Pandera, validation réussie, puis **corruption volontaire** pour observer l'échec et le message d'erreur. |
| `03_preprocessing.ipynb` | Construction du pipeline : imputation, clipping, scaling, encodage, features dérivées, et démonstration de l'absence de fuite (fit sur train uniquement). |
| `04_model_exploration.ipynb` | *Exploration des modèles de prévision* — Plancher naïf mesuré **sur le test** avant tout modèle, comparaison des familles d'algorithmes à protocole identique, thermo-sensibilité apprise lue en MW par °C comme contrôle de cohérence métier, arbitrage modèle unique contre modèle par horizon, grille de réglage et sonde de fuite temporelle. |
| `05_training.ipynb` | *Entraînement et validation temporelle* — Entraînement avec l'objet de production dans un bac à sable — les artefacts du pipeline ne sont pas touchés —, mesure de l'**optimisme** d'une validation croisée aléatoire face à des replis chronologiques, arbitrage de la perte d'entraînement sur quatre critères plutôt qu'un, et pilotage de l'écart train/validation par la complexité des arbres. |
| `06_error_analysis.ipynb` | *Analyse d'erreurs et recommandations* — Évaluation complète avec l'objet de production (intervalles et backtest compris), ventilation de l'erreur **par horizon, par régime et par mois** — trois lectures qui appellent trois correctifs différents —, lecture des pires journées une par une, autocorrélation des résidus, recalibrage des intervalles et recommandations opérationnelles. |

---

## 14. Tests

```bash
make test                # pytest -q
python -m pytest -v      # détaillé
python -m pytest tests/test_data_schemas.py -v
python -m pytest --cov=src --cov-report=term-missing
```

Ce qui est testé :

| Fichier | Objet du test |
| --- | --- |
| `tests/test_data_schemas.py` | Les schémas Pandera acceptent les données valides **et** rejettent les données corrompues (types, bornes, valeurs autorisées, colonnes manquantes). |
| `tests/test_loaders.py` | Chargement Parquet/CSV, validation appliquée, gestion des erreurs, split train/val/test. |
| `tests/test_preprocessing.py` | Transformers (fit/transform), absence de fuite, cohérence des colonnes en sortie, persistance. |
| `tests/test_models.py` | Contrat `BaseModel` : fit → predict → predict_proba, shape, déterminisme, sauvegarde/rechargement, garde-fous (modèle non entraîné, colonnes manquantes). |
| `tests/test_training.py` | Le `Trainer` produit des métriques, des callbacks fonctionnent (early stopping), les artefacts sont écrits. |
| `tests/test_pipeline.py` | Bout en bout : chaque pipeline (`data`, `train`, `evaluation`, `inference`) s'exécute sur une configuration réduite, écrit ses artefacts et refuse une entrée invalide. |

Les tests utilisent des **fixtures légères** (`tests/conftest.py`) : petit dataset synthétique
et configuration réduite, donc exécution en quelques secondes.

---

## 15. Bonnes pratiques appliquées

- **OOP systématique** : générateur, loaders, preprocessing, modèle, trainer, évaluateur,
  predictor et pipelines sont des classes à responsabilité unique.
- **Classe abstraite `BaseModel`** : contrat commun `fit / predict / predict_proba / save / load`, ce qui permet
  de changer de framework sans toucher au reste du code (DIP).
- **Type hints partout** + `mypy` configuré ; docstrings Google sur toutes les entités publiques.
- **Hydra** pour toute la configuration, **Pydantic** pour la valider au démarrage.
- **Pandera** à trois endroits critiques : données brutes, données transformées, données d'inférence.
- **Aucune fuite de données** : le preprocessing est appris sur le train seul et persisté.
- **Parquet** pour les données intermédiaires (typé, compressé, lecture partielle), CSV pour l'humain.
- **Artefacts traçables** : model card JSON, métriques JSON, rapport Markdown, figures PNG.
- **Reproductibilité** : graine propagée à Python/NumPy/scikit-learn, données régénérables à l'identique.
- **Chemins robustes** : `pathlib.Path` uniquement, racine résolue depuis `src/`, jamais depuis le CWD.
- **Logs structurés** avec loguru (interception du `logging` stdlib des librairies tierces).
- **Tests unitaires concrets** (comportements, pas `assert True`) + smoke test de bout en bout.
- **Notebooks documentés** : chaque bloc de code est précédé d'une intention et suivi d'une lecture du résultat.

---

## 16. Structure du code expliquée

| Dossier | Responsabilité | Points d'attention |
| --- | --- | --- |
| `src/data/` | Génération synthétique, chargement, **contrats Pandera** | Le générateur est déterministe ; les schémas sont la documentation exécutable des données. |
| `src/preprocessing/` | Transformers custom + pipeline sklearn-compatible | `fit` sur train uniquement, `transform` partout ; persistable. |
| `src/features/` | Feature engineering **déclaratif** (recettes en YAML) | Ajouter une feature = ajouter une entrée dans `conf/preprocessing/default.yaml`. |
| `src/models/` | `BaseModel` (ABC) + implémentation scikit-learn | Aucune logique de training loop ici : le modèle expose un contrat. |
| `src/training/` | `Trainer`, callbacks, registre de métriques | Les effets de bord (logs, early stopping) sont des callbacks, pas du code inline. |
| `src/evaluation/` | `Evaluator` + `ReportBuilder` | Les métriques sont calculées une seule fois puis sérialisées. |
| `src/inference/` | `Predictor` | Valide l'entrée avec `InferenceDataSchema` avant de prédire. |
| `src/pipelines/` | Orchestration bout-en-bout | Retourne des `PipelineResult` (statut, métriques, artefacts) exploitables en CI. |
| `src/schemas/` | Configuration typée (Pydantic) | Fait échouer immédiatement une configuration invalide. |
| `src/utils/` | Logging, chemins, IO, helpers | Zéro dépendance métier : réutilisable tel quel ailleurs. |
| `src/visualization/` | Figures des rapports et notebooks | Rendu non interactif (`Agg`), fichiers écrits dans `artifacts/figures/`. |
| `scripts/` | CLI minces | Composent la config Hydra puis délèguent à `src/` — jamais de logique métier. |
| `tests/` | Qualité | Fixtures légères, assertions sur les comportements. |
| `conf/` | Configuration | Source de vérité unique. |

---

## 17. Dépannage

| Symptôme | Cause probable | Solution |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'src'` | Exécution depuis un autre répertoire | `cd data-science/time-series-forecasting/with-sklearn` puis `python -m src.main …` (ou `export PYTHONPATH=.`) |
| `FileNotFoundError: data/raw/...` | Données non générées | `make data` (ou `python scripts/generate_data.py`) |
| `SchemaError` Pandera au chargement | Dataset corrompu / régénéré avec un autre schéma | `make clean-artifacts && make data` |
| `Could not find a version that satisfies the requirement scikit-learn` | Index PyPI inaccessible / Python trop ancien | Python ≥ 3.10, `pip install --upgrade pip`, vérifier le proxy |
| Erreur `Key 'X' not in ...` sur un override Hydra | Clé absente du schéma | Préfixer l'override avec `++` (ex. `++train.epochs=5`) |
| Les chemins pointent vers `outputs/...` | Un outil a changé le CWD | `hydra.job.chdir=false` est déjà actif ; les chemins viennent de `src/utils/paths.py` |
| Tests lents | Entraînement complet dans les tests | Les fixtures utilisent un petit dataset ; `pytest -m "not slow"` |

Pour rejouer une configuration exacte : Hydra sauvegarde la config composée dans
`outputs/<date>/<heure>/.hydra/config.yaml` — copiez-la avec `--config-path`/`--config-name`.

---

## 18. Références

- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [sklearn Pipeline & ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html)
- [Hydra — Documentation officielle](https://hydra.cc/docs/intro/)
- [Pandera — Data validation](https://pandera.readthedocs.io/)
- [Pydantic v2 — Data validation](https://docs.pydantic.dev/latest/)
- [OmegaConf — Configuration structurée](https://omegaconf.readthedocs.io/)
- [scikit-learn — Pipelines et composite estimators](https://scikit-learn.org/stable/modules/compose.html)
- [Google Python Style Guide — Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [Cookiecutter Data Science — Standard d'arborescence](https://cookiecutter-data-science.drivendata.org/)

---

## 19. Licence & contribution

Code fourni à des fins pédagogiques, licence MIT. Pour proposer une amélioration : conserver la
structure imposée, ajouter des tests, mettre à jour ce README et vérifier `make verify`.

*Dernière génération : 2026-09-13 · projet `ds-forecasting-sklearn`*
