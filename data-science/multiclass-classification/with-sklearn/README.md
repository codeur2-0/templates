# Diagnostic du mode de défaillance machine avec scikit-learn

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-classic-ml-orange)
![Hydra](https://img.shields.io/badge/config-Hydra-89b482)
![Pandera](https://img.shields.io/badge/validation-Pandera-16a34a)
![pytest](https://img.shields.io/badge/tests-pytest-2ea44f)
![Licence](https://img.shields.io/badge/license-MIT-lightgrey)

> **Domaine** : `data-science` · **Problématique** : `multiclass-classification` · **Stack** : `scikit-learn`
> **Emplacement** : `data-science/multiclass-classification/with-sklearn/`
> **Temps de lecture** : ~15 min · **Temps d'exécution complet** : < 5 min sur un laptop

---

## 1. À propos / Objectifs

Pipeline multi-classes de bout en bout (alarmes machine synthétiques à physique explicite ->
contrats Pandera -> features physiques déclarées en configuration -> gradient boosting par
histogrammes -> évaluation face au plancher majoritaire, au routage actuel par code automate et au
plafond oracle -> décision à coût minimal et revue experte des cas incertains) pour une équipe
maintenance, entièrement piloté par Hydra et structuré en objets testables.

Chaque alarme automate arrête la machine et déclenche une intervention. Aujourd'hui, l'équipe
envoyée est choisie d'après le code d'alarme émis par l'automate, calculé par des règles à seuil
fixes qui ignorent la gamme de produit et les zones de bordure : une fois sur deux environ, la
mauvaise spécialité se déplace, la machine reste arrêtée pendant qu'on envoie la bonne, et une panne
réelle classée « fausse alarme » est acquittée alors que la machine continue de se dégrader.

**Ce que cet exemple démontre**

1. Composer un pipeline scikit-learn multi-classes propre : ColumnTransformer, features physiques déclarées, fit sur le train uniquement.
2. Utiliser une classe abstraite BaseModel pour rendre le framework interchangeable.
3. Lire des métriques multi-classes en contexte déséquilibré (macro-F1, balanced accuracy, MCC, AUC un-contre-tous).
4. Remplacer l'argmax par une décision à coût minimal et un seuil de revue experte chiffrés sur le métier.
5. Situer un modèle entre le plancher majoritaire, la règle métier actuelle et le plafond atteignable.

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

Chaque alarme automate arrête la machine et déclenche une intervention. Aujourd'hui, l'équipe
envoyée est choisie d'après le code d'alarme émis par l'automate, calculé par des règles à seuil
fixes qui ignorent la gamme de produit et les zones de bordure : une fois sur deux environ, la
mauvaise spécialité se déplace, la machine reste arrêtée pendant qu'on envoie la bonne, et une panne
réelle classée « fausse alarme » est acquittée alors que la machine continue de se dégrader.

| | |
| --- | --- |
| **Qui** (persona / consommateur) | Responsable maintenance d'un atelier d'usinage (fraiseuses CNC sur quatre lignes), avec une équipe de techniciens répartis par spécialité (outillage, thermique, électrique, mécanique) et un data scientist qui industrialise le modèle ; le superviseur de production consomme le diagnostic dans son outil de GMAO. |
| **Problème** | Prédire, au moment de l'alarme et à partir des seuls capteurs disponibles, le mode de défaillance parmi six (fausse alarme, usure outil, dissipation thermique, défaut de puissance, surcharge mécanique, défaillance aléatoire), afin d'envoyer directement la bonne équipe — ou de demander un avis expert quand le modèle doute. |
| **Entrées** | Contexte de production (gamme de produit L/M/H, ligne, équipe), code d'alarme automate, et relevés capteurs à l'instant de l'alarme : températures air et process, vitesse de broche, couple, usure outil, débit de refroidissement, vibration (capteur absent sur une partie du parc), heures depuis la dernière maintenance. |
| **Sorties** | Mode de défaillance prédit, probabilité de chacun des six modes, confiance et marge entre les deux hypothèses les plus probables, décision à coût minimal, équipe à envoyer, drapeau de revue experte quand la confiance est insuffisante, et justification lisible. |
| **Valeur métier** | Moins de déplacements inutiles et de seconds passages, arrêts machine raccourcis, et surtout moins de pannes réelles acquittées par erreur — l'erreur la plus chère, parce que la machine continue de produire des pièces non conformes jusqu'à la casse. |
| **Cadence** | Diagnostic calculé à chaque alarme (appel unitaire), modèle ré-entraîné chaque mois sur les interventions clôturées. |

**Objectif de modélisation** — Prédire, au moment de l'alarme et à partir des seuls capteurs disponibles, le mode de
défaillance parmi six (fausse alarme, usure outil, dissipation thermique, défaut de
puissance, surcharge mécanique, défaillance aléatoire), afin d'envoyer directement la bonne
équipe — ou de demander un avis expert quand le modèle doute.

**Critères de réussite**

- [ ] Macro-F1 ≥ 0,60 sur le split de test hors échantillon : les six modes comptent autant, quelle que soit leur fréquence.
- [ ] Le modèle bat le routage actuel par code d'alarme d'au moins 20 points de macro-F1 : sans cet écart, la règle de l'automate suffit.
- [ ] Rappel ≥ 0,55 sur chacun des cinq modes prédictibles (tous sauf `random_failure`, sans signal observable par construction).
- [ ] Au plus 5 % des pannes réelles acquittées comme « fausse alarme » par la décision retenue : c'est l'erreur qui laisse une machine se dégrader.
- [ ] Probabilités exploitables pour la décision à coût minimal : erreur de calibration top-label (ECE) < 0,08.
- [ ] Coût moyen par alarme inférieur d'au moins 40 % à celui du routage actuel, avec la même matrice de coûts.
- [ ] Temps d'inférence < 50 ms pour un lot de 1 000 alarmes (contrainte de la GMAO).
- [ ] Reproductibilité : deux exécutions avec la même seed produisent exactement les mêmes métriques.

**Contraintes**

- Aucune donnée industrielle réelle : le jeu de ce dépôt est synthétique, sa physique est inspirée des jeux publics de maintenance prédictive.
- Le modèle tourne sur CPU, à côté de la GMAO, sans service externe.
- Toute décision automatisée reste explicable (probabilités par mode, seconde hypothèse, justification) et réversible (revue experte sous le seuil de confiance).

---

## 4. Données

Les données sont **synthétiques**, générées localement par
[`src/data/generators.py`](src/data/generators.py) : aucune donnée réelle, aucun téléchargement,
aucune information confidentielle. Elles sont néanmoins construites pour être **crédibles
métier** (corrélations réalistes, bruit, valeurs manquantes, outliers légitimes).

| Propriété | Valeur |
| --- | --- |
| Jeu de données | `machine_failure_diagnosis` |
| Volume par défaut | 6 000 lignes |
| Formats écrits | parquet, csv |
| Emplacement | `data/raw/machine_failure_diagnosis.parquet` (et `.csv`) |
| Cible | `failure_mode` |
| Identifiant | `alarm_id` |
| Graine | `42` (reproductible) |

**Schéma**

| Colonne | Type | Rôle | Signification métier |
| --- | --- | --- | --- |
| `alarm_id` | str | identifier | Identifiant unique de l'alarme |
| `alarm_timestamp` | datetime | timestamp | Date et heure de levée de l'alarme |
| `product_quality` | category | feature | Gamme de la pièce usinée (L = standard, M = intermédiaire, H = haute précision) |
| `production_line` | category | feature | Ligne de production de la machine |
| `shift` | category | feature | Équipe en poste au moment de l'alarme |
| `alarm_code` | category | feature | Code d'alarme émis par l'automate (règles à seuil fixes, une part de codes corrompus) |
| `air_temperature_k` | float | feature | Température de l'air ambiant (K) |
| `process_temperature_k` | float | feature | Température du process (zone de coupe) (K) |
| `rotational_speed_rpm` | float | feature | Vitesse de rotation de la broche (tr/min) |
| `torque_nm` | float | feature | Couple de coupe (N.m) |
| `tool_wear_min` | int | feature | Temps d'utilisation cumulé de l'outil en place (min) |
| `coolant_flow_l_min` | float | feature | Débit du circuit de refroidissement (L/min) |
| `vibration_mm_s` | float | feature | Vitesse vibratoire efficace de la broche (mm/s) |
| `hours_since_maintenance` | float | feature | Heures de fonctionnement depuis la dernière maintenance préventive (h) |
| `failure_mode` | str | target | Mode de défaillance constaté à la clôture de l'intervention |

Détail complet (distributions attendues, remarques) : [`data/README.md`](data/README.md).

---

## 5. Arborescence

```text
data-science/multiclass-classification/with-sklearn/
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
│   ├── 01_eda.ipynb
│   ├── 02_validation.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_model_exploration.ipynb
│   ├── 05_training.ipynb
│   └── 06_error_analysis.ipynb
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
cd data-science/multiclass-classification/with-sklearn
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-dev.txt   # runtime + tests + lint + notebooks
```

**Option B — uv (plus rapide)**

```bash
cd data-science/multiclass-classification/with-sklearn
uv venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

**Option C — installation éditable du projet**

```bash
cd data-science/multiclass-classification/with-sklearn
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

Résultat : `data/raw/machine_failure_diagnosis.parquet` (+ `.csv` pour la lecture humaine)
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
python -m src.main mode=train model.params.max_iter=100

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
python scripts/predict.py predict.input=data/raw/machine_failure_diagnosis.csv predict.n_samples=10
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
| `conf/config.yaml` -> bloc `diagnosis` | Réglages métier de la famille, au même niveau que `mode` et `seed` : 10 clés (class_order, no_failure_class, structural_classes, alarm_code_column, alarm_code_rule, team_by_class, costs, …), chacune commentée dans le fichier — c'est là qu'on change le métier sans toucher au code. | `diagnosis.no_failure_class=false_alarm`, `diagnosis.alarm_code_column=alarm_code` |

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
| `data/raw/machine_failure_diagnosis.parquet` (+ `csv`) | jeu de données synthétique, 6 000 lignes |
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
| `artifacts/reports/per_class.csv` | précision, rappel, F1, AUC un-contre-tous et équipe, pour chaque classe |
| `artifacts/reports/decision_policies.csv` | coût et profil d'erreur de l'argmax, de la décision à coût minimal, de la revue experte et de la règle actuelle |
| `artifacts/reports/abstention_curve.csv` | couverture, exactitude automatisée et coût selon le seuil de revue experte |
| `artifacts/reports/top_errors.csv` | diagnostics erronés les plus confiants, avec leurs probabilités et leur contexte |
| `artifacts/reports/predictions.csv` | prédictions sur l'échantillon de démonstration |
| `artifacts/figures/*.png` | matrice de confusion normalisée, qualité par classe, références, ROC un-contre-tous, fiabilité, revue experte, coût des politiques de décision |
| `outputs/<date>/<heure>/` | configuration composée + logs Hydra |

Métrique principale : **`f1_macro`** (sens `maximize`, seuil de smoke test : ≥ 0.55).
Métriques secondaires : balanced_accuracy, accuracy, mcc, log_loss, roc_auc_ovr, f1_weighted, precision_macro, recall_macro.

Résultats de l'exécution de référence (`make all`, graine 42) :

| Indicateur | Valeur mesurée |
| --- | --- |
| Macro-F1 (test) | **0,671 (routage actuel 0,375 ; plafond oracle 0,679)** |
| Accuracy / balanced accuracy / MCC | **0,777 / 0,667 / 0,723** |
| Log loss / AUC un-contre-tous / ECE | **0,694 / 0,908 / 0,026** |
| Coût par alarme : argmax / coût minimal / routage actuel | **476 / 172 / 428 EUR** |
| Pannes réelles acquittées : argmax / coût minimal / routage actuel | **22,4 % / 0,0 % / 14,4 %** |
| Rappel le plus faible d'un mode prédictible (overstrain) | **0,593** |
| Rappel de random_failure (modèle / oracle) | **0,000 / 0,000** |
| Latence pour 1 000 alarmes | **21 ms** |
| Verdict contractuel | **7/7 objectifs atteints** |

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
| `04_model_exploration.ipynb` | *Exploration et comparaison de modèles* — Plancher (classe majoritaire), règle métier actuelle et plafond oracle installés **avant** toute comparaison, algorithmes comparés sur le macro-F1 **et** la calibration, apport chiffré des features physiques, effet de la pondération de classes sur la décision à coût minimal, grille de réglage et dispersion entre graines. |
| `05_training.ipynb` | *Entraînement dans les conditions de production* — Entraînement piloté par **objets** (`Trainer`, callbacks) plutôt que par un script monolithique, lecture d'un `TrainingOutcome` (métriques, durée, historique, artefacts), stabilité mesurée sur plusieurs graines avant de conclure, et garde-fou de qualité déclaré en configuration. |
| `06_error_analysis.ipynb` | *Analyse d'erreurs, décision et recommandations* — Verdict contractuel de l'évaluateur de production, position du modèle entre la règle actuelle et le plafond, confusions lues comme des causes physiques, **argmax contre décision à coût minimal**, choix du seuil de revue experte sur la courbe couverture / coût, calibration, plafond structurel d'une classe sans signal et plan d'action. |

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
| `ModuleNotFoundError: No module named 'src'` | Exécution depuis un autre répertoire | `cd data-science/multiclass-classification/with-sklearn` puis `python -m src.main …` (ou `export PYTHONPATH=.`) |
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

*Dernière génération : 2026-10-06 · projet `ds-multiclass-sklearn`*
