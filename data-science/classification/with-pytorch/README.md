# Prédiction d'attrition client (churn télécom) avec PyTorch

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-deep-learning-orange)
![Hydra](https://img.shields.io/badge/config-Hydra-89b482)
![Pandera](https://img.shields.io/badge/validation-Pandera-16a34a)
![pytest](https://img.shields.io/badge/tests-pytest-2ea44f)
![Licence](https://img.shields.io/badge/license-MIT-lightgrey)

> **Domaine** : `data-science` · **Problématique** : `classification` · **Stack** : `PyTorch`
> **Emplacement** : `data-science/classification/with-pytorch/`
> **Temps de lecture** : ~15 min · **Temps d'exécution complet** : < 5 min sur un laptop

---

## 1. À propos / Objectifs

Pipeline tabulaire de bout en bout (données synthétiques -> contrats Pandera -> feature engineering
-> réseau de neurones PyTorch entraîné par descente de gradient explicite -> évaluation et analyse
d'erreurs) pour un cas de churn télécom, entièrement piloté par Hydra et structuré en objets
testables.

L'opérateur perd chaque mois une part significative de ses abonnés au profit de concurrents.
Acquérir un client coûte 5 à 7 fois plus cher que d'en conserver un, et les campagnes de rétention
actuelles sont déclenchées au feeling, trop tard et sur trop de monde (coût inutile, fatigue
commerciale).

**Ce que cet exemple démontre**

1. Écrire une boucle d'entraînement PyTorch explicite : DataLoader, forward, perte, backward, optimizer.step.
2. Comprendre ce que coûte le deep learning sur données tabulaires et quand lui préférer un booster.
3. Régulariser un réseau : dropout, weight decay, early stopping avec restauration du meilleur état.
4. Ordonnancer le taux d'apprentissage (cosine, plateau) et lire son effet sur les courbes d'apprentissage.
5. Traiter un déséquilibre de classes avec `pos_weight` sur une entropie croisée binaire avec logits.
6. Garantir le déterminisme : graine avant initialisation, mélange généré par un Generator semé, CPU.
7. Sauvegarder un checkpoint (poids + métadonnées) et le recharger sans pickler l'objet Python.

**Pourquoi PyTorch ici ?**

- `nn.Module` explicite, boucle d'entraînement écrite à la main : pédagogie maximale.
- Déterminisme : `torch.manual_seed`, `torch.use_deterministic_algorithms` et `num_workers=0` par défaut.

---

## 2. Stack technologique

| Rôle | Outil | Version minimale | Pourquoi |
| --- | --- | --- | --- |
| Framework ML | **PyTorch** | torch>=2.2 | deep-learning |
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

L'opérateur perd chaque mois une part significative de ses abonnés au profit de concurrents.
Acquérir un client coûte 5 à 7 fois plus cher que d'en conserver un, et les campagnes de rétention
actuelles sont déclenchées au feeling, trop tard et sur trop de monde (coût inutile, fatigue
commerciale).

| | |
| --- | --- |
| **Qui** (persona / consommateur) | Équipe Customer Success / Marketing d'un opérateur télécom (abonnement B2C), avec un data scientist qui industrialise le modèle et un CRM qui consomme les scores. |
| **Problème** | Identifier, avant la résiliation, les abonnés les plus susceptibles de partir dans les 30 prochains jours, afin de cibler une action de rétention (offre, geste commercial, appel sortant) sur une population réduite et pertinente. |
| **Entrées** | Fiche abonné : ancienneté, type de contrat, service internet, moyen de paiement, région, charges mensuelles et cumulées, tickets support des 6 derniers mois, consommation data, nombre de services souscrits, promotion active, score de satisfaction déclaré. |
| **Sorties** | Probabilité de churn par abonné, décision binaire selon un seuil piloté par le métier, segment de risque (faible / modéré / élevé / critique) et principaux facteurs explicatifs. |
| **Valeur métier** | Réduction du taux d'attrition, baisse du coût des campagnes de rétention (ciblage), meilleure allocation des appels sortants vers les clients à fort enjeu. |
| **Cadence** | Score recalculé quotidiennement (batch nocturne) et exposé au CRM. |

**Objectif de modélisation** — Identifier, avant la résiliation, les abonnés les plus susceptibles de partir dans les 30
prochains jours, afin de cibler une action de rétention (offre, geste commercial, appel
sortant) sur une population réduite et pertinente.

**Critères de réussite**

- [ ] ROC AUC ≥ 0.75 sur le split de test hors échantillon (le générateur injecte un bruit irréductible).
- [ ] Rappel de la classe « churn » ≥ 0.55 au seuil retenu : on préfère surveiller trop que rater un départ.
- [ ] Précision ≥ 0.50 sur la population alertée pour que le coût de rétention reste rentable.
- [ ] Probabilités exploitables : écart de calibration moyen < 0.10 (sinon recalibrage).
- [ ] Temps d'inférence < 50 ms pour un batch de 1 000 abonnés (contrainte CRM).
- [ ] Reproductibilité : deux exécutions avec la même seed produisent exactement les mêmes métriques.

**Contraintes**

- Aucune donnée personnelle sensible : le dataset de ce dépôt est synthétique.
- Le modèle doit tourner sur CPU, sans dépendance à un service externe.
- Toute décision automatisée doit rester explicable (segments + facteurs clés).

---

## 4. Données

Les données sont **synthétiques**, générées localement par
[`src/data/generators.py`](src/data/generators.py) : aucune donnée réelle, aucun téléchargement,
aucune information confidentielle. Elles sont néanmoins construites pour être **crédibles
métier** (corrélations réalistes, bruit, valeurs manquantes, outliers légitimes).

| Propriété | Valeur |
| --- | --- |
| Jeu de données | `telecom_churn` |
| Volume par défaut | 4 000 lignes |
| Formats écrits | parquet, csv |
| Emplacement | `data/raw/telecom_churn.parquet` (et `.csv`) |
| Cible | `churned` |
| Identifiant | `customer_id` |
| Taux de classe positive | ~26 % |
| Graine | `42` (reproductible) |

**Schéma**

| Colonne | Type | Rôle | Signification métier |
| --- | --- | --- | --- |
| `customer_id` | str | identifier | Identifiant unique de l'abonné |
| `signup_date` | datetime | timestamp | Date de souscription de l'abonnement |
| `tenure_months` | int | feature | Ancienneté de l'abonné (mois) |
| `contract_type` | category | feature | Type de contrat souscrit |
| `internet_service` | category | feature | Service internet associé à l'abonnement |
| `payment_method` | category | feature | Moyen de paiement utilisé |
| `region` | category | feature | Région commerciale de rattachement |
| `monthly_charges` | float | feature | Montant mensuel facturé (EUR) |
| `total_charges` | float | feature | Cumul facturé depuis la souscription (EUR) |
| `support_tickets_6m` | int | feature | Nombre de tickets support ouverts sur 6 mois (tickets) |
| `avg_monthly_data_gb` | float | feature | Consommation data mensuelle moyenne (Go) |
| `num_products` | int | feature | Nombre de services/produits souscrits (bundle) (produits) |
| `has_promotion` | bool | feature | Bénéficie d'une promotion active (1 = oui) |
| `satisfaction_score` | float | feature | Score de satisfaction déclaré (enquête CSAT) (/10) |
| `churned` | int | target | Attrition observée dans les 30 jours (1 = parti) |

Détail complet (distributions attendues, remarques) : [`data/README.md`](data/README.md).

---

## 5. Arborescence

```text
data-science/classification/with-pytorch/
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
│   ├── models/                  # BaseModel (ABC) + implémentation PyTorch
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
   ├── PyTorchModel(BaseModel)   ← implémentation PyTorch
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
                     → PyTorchModel.fit (callbacks)
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
- `../../with-tensorflow/` — tensorflow
- `../../with-keras/` — keras
- `../../with-mlflow/` — mlflow

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
cd data-science/classification/with-pytorch
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-dev.txt   # runtime + tests + lint + notebooks
```

**Option B — uv (plus rapide)**

```bash
cd data-science/classification/with-pytorch
uv venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

**Option C — installation éditable du projet**

```bash
cd data-science/classification/with-pytorch
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

Résultat : `data/raw/telecom_churn.parquet` (+ `.csv` pour la lecture humaine)
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
python -m src.main mode=train model.params.hidden_layers=[64, 32]

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
python scripts/predict.py predict.input=data/raw/telecom_churn.csv predict.n_samples=10
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
from src.models.model import PyTorchModel
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
| `conf/model/default.yaml` | algorithme et hyperparamètres | `model.params.hidden_layers=…` |
| `conf/train/default.yaml` | split, epochs, callbacks, noms d'artefacts | `++train.epochs=20`, `train.split.test_size=0.25` |
| `conf/preprocessing/default.yaml` | imputation, scaling, encodage, features dérivées | `preprocessing.numeric.scaler=robust`, `preprocessing.categorical.encoder=ordinal` |
| `conf/hydra/local.yaml` | répertoires de sortie Hydra (`outputs/`, `multirun/`) | `hydra.run.dir=outputs/debug` |
| `conf/config.yaml` -> bloc `decision` | Réglages métier de la famille, au même niveau que `mode` et `seed` : 3 clés (threshold, retention_cost_eur, customer_lifetime_value_eur), chacune commentée dans le fichier — c'est là qu'on change le métier sans toucher au code. | `decision.threshold=0.5`, `decision.retention_cost_eur=25.0` |

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
| `data/raw/telecom_churn.parquet` (+ `csv`) | jeu de données synthétique, 4 000 lignes |
| `data/raw/generation_metadata.json` | recette de génération : graine, options, empreinte du jeu, fichiers écrits |
| `data/processed/split_{train,val,test}.parquet` | splits avant transformation (l'évaluation et l'inférence rejouent exactement le même découpage) |
| `data/processed/features_{X_train,X_val,X_test}.parquet` | matrices prêtes pour le modèle |
| `artifacts/models/model.pt` | modèle entraîné |
| `artifacts/models/preprocessing.joblib` | pipeline de preprocessing ajusté (aucune fuite) |
| `artifacts/models/feature_builder.joblib` | construction des features dérivées, ajustée sur le train uniquement |
| `artifacts/models/model_card.json` | carte du modèle (params, métriques, features, date) |
| `artifacts/models/resolved_config.json` | configuration Hydra résolue : l'artefact entraîné porte sa recette exacte |
| `artifacts/metrics/training_metrics.json` | métriques d'entraînement et de validation |
| `artifacts/metrics/evaluation_metrics.json` | métriques sur le split de test + verdict des seuils |
| `artifacts/reports/evaluation_report.md` | rapport lisible (métriques, analyse d'erreurs, recommandations) |
| `artifacts/reports/top_errors.csv` | exemples les plus mal classés, avec leurs probabilités et les features en cause |
| `artifacts/reports/predictions.csv` | prédictions sur l'échantillon de démonstration |
| `artifacts/figures/*.png` | matrice de confusion, courbes ROC/PR, calibration |
| `outputs/<date>/<heure>/` | configuration composée + logs Hydra |

Métrique principale : **`roc_auc`** (sens `maximize`, seuil de smoke test : ≥ 0.7).
Métriques secondaires : pr_auc, accuracy, balanced_accuracy, precision, recall, f1, log_loss.

---

## 13. Notebooks

Tous les notebooks sont **exécutables de bout en bout** (`make notebooks`) et documentés
cellule par cellule, comme un support de formation pour juniors.


| Notebook | Ce qu'on y apprend |
| --- | --- |
| `01_eda.ipynb` | EDA structurée : types, manquants, distributions univariées et bivariées, corrélations, outliers, puis **10-15 insights** actionnables. |
| `02_validation.ipynb` | Pourquoi des contrats de données : définition d'un `DataFrameModel` Pandera, validation réussie, puis **corruption volontaire** pour observer l'échec et le message d'erreur. |
| `03_preprocessing.ipynb` | Construction du pipeline : imputation, clipping, scaling, encodage, features dérivées, et démonstration de l'absence de fuite (fit sur train uniquement). |
| `04_model_exploration.ipynb` | *Exploration et comparaison de modèles* — Comparaison baseline + modèles candidats en validation croisée, table de métriques, choix argumenté du modèle. |
| `05_training.ipynb` | *Entraînement dans les conditions de production* — Entraînement piloté par **objets** (`Trainer`, callbacks) plutôt que par un script monolithique, lecture d'un `TrainingOutcome` (métriques, durée, historique, artefacts), stabilité mesurée sur plusieurs graines avant de conclure, et garde-fou de qualité déclaré en configuration. |
| `06_error_analysis.ipynb` | *Analyse d'erreurs et recommandations* — Matrice de confusion, rapport par classe, exemples mal prédits, hypothèses sur les causes et recommandations concrètes. |

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
- **Reproductibilité** : graine propagée à Python/NumPy/PyTorch, données régénérables à l'identique.
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
| `src/models/` | `BaseModel` (ABC) + implémentation PyTorch | Aucune logique de training loop ici : le modèle expose un contrat. |
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
| `ModuleNotFoundError: No module named 'src'` | Exécution depuis un autre répertoire | `cd data-science/classification/with-pytorch` puis `python -m src.main …` (ou `export PYTHONPATH=.`) |
| `FileNotFoundError: data/raw/...` | Données non générées | `make data` (ou `python scripts/generate_data.py`) |
| `SchemaError` Pandera au chargement | Dataset corrompu / régénéré avec un autre schéma | `make clean-artifacts && make data` |
| `Could not find a version that satisfies the requirement pytorch` | Index PyPI inaccessible / Python trop ancien | Python ≥ 3.10, `pip install --upgrade pip`, vérifier le proxy |
| Erreur `Key 'X' not in ...` sur un override Hydra | Clé absente du schéma | Préfixer l'override avec `++` (ex. `++train.epochs=5`) |
| Les chemins pointent vers `outputs/...` | Un outil a changé le CWD | `hydra.job.chdir=false` est déjà actif ; les chemins viennent de `src/utils/paths.py` |
| `UserWarning: Deterministic behavior ...` | Algorithmes déterministes activés | Attendu : `set_seed(deterministic=True)` — peut ralentir légèrement l'entraînement |
| Tests lents | Entraînement complet dans les tests | Les fixtures utilisent un petit dataset ; `pytest -m "not slow"` |

Pour rejouer une configuration exacte : Hydra sauvegarde la config composée dans
`outputs/<date>/<heure>/.hydra/config.yaml` — copiez-la avec `--config-path`/`--config-name`.

---

## 18. Références

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [torch.nn](https://pytorch.org/docs/stable/nn.html)
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

*Dernière génération : 2026-09-13 · projet `ds-classification-pytorch`*
