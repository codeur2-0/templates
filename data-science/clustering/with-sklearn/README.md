# Segmentation client d'un e-commerçant (RFM étendu) avec scikit-learn

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-classic-ml-orange)
![Hydra](https://img.shields.io/badge/config-Hydra-89b482)
![Pandera](https://img.shields.io/badge/validation-Pandera-16a34a)
![pytest](https://img.shields.io/badge/tests-pytest-2ea44f)
![Licence](https://img.shields.io/badge/license-MIT-lightgrey)

> **Domaine** : `data-science` · **Problématique** : `clustering` · **Stack** : `scikit-learn`
> **Emplacement** : `data-science/clustering/with-sklearn/`
> **Temps de lecture** : ~15 min · **Temps d'exécution complet** : < 5 min sur un laptop

---

## 1. À propos / Objectifs

Pipeline tabulaire de bout en bout (données synthétiques -> contrats Pandera -> feature engineering
-> KMeans sur espace standardisé -> profils de segments, validité externe et affectation des
nouveaux clients avec niveau de confiance) pour la segmentation CRM d'un e-commerçant, entièrement
piloté par Hydra et structuré en objets testables. Aucun label n'est utilisé : la qualité se juge
sur des critères internes, de robustesse et métier.

La segmentation actuelle est artisanale : trois seuils RFM (récence, fréquence, montant) posés à la
main dans un tableur, revus une fois par an. Elle classe 62 % des clients dans un seul groupe «
standard », ignore complètement le comportement promotionnel, le taux de retour et la pression sur
le service client, et ne dit rien des nouveaux inscrits qui n'ont pas encore commandé. Résultat :
les campagnes sont indifférenciées, le coût d'acquisition augmente, et le churn des clients à fort
potentiel n'est détecté qu'après six mois d'inactivité.

**Ce que cet exemple démontre**

1. Construire un pipeline non supervisé sans fuite : les colonnes de diagnostic sont exclues des features par configuration.
2. Choisir le nombre de groupes par triangulation (silhouette, Davies-Bouldin, coude d'inertie, taille minimale).
3. Comprendre l'effet de l'échelle et des queues de distribution sur une distance euclidienne (log, winsorising, standardisation).
4. Comparer KMeans, MiniBatchKMeans et GaussianMixture à k fixé, sur un même protocole.
5. Mesurer la stabilité des affectations entre graines (ARI) avant toute mise en production.
6. Profiler des segments en unités métier et identifier leurs drivers en écarts-types.
7. Valider en externe : séparer un comportement observé après coup (churn à 90 jours) sans l'avoir utilisé.
8. Publier une affectation assortie d'une confiance et d'une action CRM, et router les cas frontières.
9. Industrialiser l'ensemble : configuration Hydra, artefacts traçables, tests pytest, rapports générés.

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

La segmentation actuelle est artisanale : trois seuils RFM (récence, fréquence, montant) posés à la
main dans un tableur, revus une fois par an. Elle classe 62 % des clients dans un seul groupe «
standard », ignore complètement le comportement promotionnel, le taux de retour et la pression sur
le service client, et ne dit rien des nouveaux inscrits qui n'ont pas encore commandé. Résultat :
les campagnes sont indifférenciées, le coût d'acquisition augmente, et le churn des clients à fort
potentiel n'est détecté qu'après six mois d'inactivité.

| | |
| --- | --- |
| **Qui** (persona / consommateur) | Équipe CRM & Data d'un e-commerçant français (mode et maison, ~250 000 clients actifs), avec un data scientist qui construit la segmentation, un responsable CRM qui pilote les campagnes d'acquisition et de rétention, et un outil de marketing automation qui consomme le segment de chaque client pour choisir le message, le canal et la pression publicitaire. |
| **Problème** | Découvrir **sans étiquette préalable** des groupes de clients homogènes et actionnables à partir de leur comportement d'achat et d'engagement sur douze mois glissants, choisir le nombre de groupes de façon argumentée (et non arbitraire), profiler chaque groupe, puis affecter tout nouveau client au groupe le plus proche avec un niveau de confiance. |
| **Entrées** | Agrégats client sur douze mois glissants : récence du dernier achat, nombre de commandes, chiffre d'affaires, panier moyen, largeur du catalogue achetée, part du chiffre d'affaires réalisée en promotion, taux de retour, tickets de service client, ouvertures de newsletter, sessions web et part mobile, palier de fidélité, canal d'acquisition, région, NPS déclaré, consentement marketing, date d'inscription. |
| **Sorties** | Identifiant de segment, probabilité d'appartenance (quand l'algorithme en produit), distance au centre du segment, niveau de confiance de l'affectation, profil du segment (effectif, part du chiffre d'affaires, comportement dominant, taux de churn observé) et recommandation d'action CRM associée. |
| **Valeur métier** | Campagnes différenciées (message, canal, remise) au lieu d'un envoi unique, détection précoce des clients à fort potentiel qui décrochent, arbitrage du budget promotionnel (ne plus remiser les clients qui achètent au prix fort), et mesure de la valeur par segment pour prioriser les investissements d'acquisition. |
| **Cadence** | Segmentation recalculée chaque nuit sur les clients actifs (affectation incrémentale des nouveaux), profils et seuils revus mensuellement avec le CRM, ré-entraînement complet trimestriel. |

**Objectif de modélisation** — Découvrir **sans étiquette préalable** des groupes de clients homogènes et actionnables à
partir de leur comportement d'achat et d'engagement sur douze mois glissants, choisir le
nombre de groupes de façon argumentée (et non arbitraire), profiler chaque groupe, puis
affecter tout nouveau client au groupe le plus proche avec un niveau de confiance.

**Critères de réussite**

- [ ] Silhouette moyenne ≥ 0.25 sur le split de test hors échantillon (structure de groupes réelle, pas du bruit).
- [ ] Indice de Davies-Bouldin ≤ 1.60 et Calinski-Harabasz ≥ 900 : compacité et séparation simultanées.
- [ ] Aucun groupe dégénéré : chaque segment représente ≥ 3 % des clients (sinon il est fusionné ou supprimé).
- [ ] Stabilité : ARI ≥ 0.85 entre deux entraînements de graines différentes (une segmentation qui bouge à chaque run n'est pas exploitable).
- [ ] Validité externe : écart de taux de churn à 90 jours ≥ 20 points entre le segment le plus sain et le plus à risque.
- [ ] Accord avec la structure latente injectée dans le générateur : ARI ≥ 0.45 (diagnostic pédagogique, indisponible en production).
- [ ] Reproductibilité : deux exécutions avec la même seed produisent exactement les mêmes affectations.

**Contraintes**

- Aucune donnée réelle de client : le dataset de ce dépôt est synthétique et hors ligne.
- Aucune étiquette de segment en production : la variable latente du générateur sert uniquement de diagnostic pédagogique et n'est jamais une feature.
- L'affectation d'un client doit être explicable en une phrase au CRM (profil du segment + distance).
- Traitement sur CPU de 250 000 clients en moins de 5 minutes (fenêtre de batch nocturne).
- Vigilance RGPD et équité : la région et le canal d'acquisition sont des proxies socio-économiques ; leur influence sur l'affectation doit être documentée et surveillée.

---

## 4. Données

Les données sont **synthétiques**, générées localement par
[`src/data/generators.py`](src/data/generators.py) : aucune donnée réelle, aucun téléchargement,
aucune information confidentielle. Elles sont néanmoins construites pour être **crédibles
métier** (corrélations réalistes, bruit, valeurs manquantes, outliers légitimes).

| Propriété | Valeur |
| --- | --- |
| Jeu de données | `retail_customer_base` |
| Volume par défaut | 6,000 lignes |
| Formats écrits | parquet, csv |
| Emplacement | `data/raw/retail_customer_base.parquet` (et `.csv`) |
| Cible | *aucune* (apprentissage non supervisé) |
| Identifiant | `customer_id` |
| Colonne temporelle | `signup_date` |
| Graine | `42` (reproductible) |

**Schéma**

| Colonne | Type | Rôle | Signification métier |
| --- | --- | --- | --- |
| `customer_id` | str | identifier | Identifiant unique du client |
| `signup_date` | datetime | timestamp | Date d'inscription (première création de compte) |
| `last_order_date` | datetime | timestamp | Date de la dernière commande |
| `tenure_months` | int | feature | Ancienneté du compte en mois (mois) |
| `recency_days` | int | feature | Nombre de jours depuis la dernière commande (jours) |
| `orders_12m` | int | feature | Nombre de commandes sur 12 mois glissants (commandes) |
| `revenue_12m_eur` | float | feature | Chiffre d'affaires réalisé sur 12 mois glissants (EUR) |
| `avg_basket_eur` | float | feature | Panier moyen (chiffre d'affaires / nombre de commandes) (EUR) |
| `distinct_categories_12m` | int | feature | Nombre de catégories de produits achetées (largeur du catalogue) (catégories) |
| `discount_share` | float | feature | Part du chiffre d'affaires réalisée avec une remise (%) |
| `return_rate` | float | feature | Taux de retour (articles retournés / articles commandés) (%) |
| `support_tickets_12m` | int | feature | Nombre de contacts au service client (réclamation, SAV, question livraison) (tickets) |
| `newsletter_opens_12m` | int | feature | Nombre d'ouvertures de newsletter sur 12 mois (ouvertures) |
| `web_sessions_12m` | int | feature | Nombre de sessions web ou application sur 12 mois (sessions) |
| `mobile_share` | float | feature | Part des sessions réalisées sur mobile (%) |
| `loyalty_tier` | category | feature | Palier du programme de fidélité |
| `acquisition_channel` | category | feature | Canal d'acquisition du client |
| `region` | category | feature | Région de livraison principale |
| `nps_score` | int | feature | Note de recommandation déclarée (Net Promoter Score individuel) (/10) |
| `opt_in_marketing` | bool | feature | Consentement aux communications marketing (1 = oui) |
| `latent_segment` | category | metadata | Profil latent injecté par le générateur — diagnostic pédagogique uniquement |
| `churned_next_90d` | bool | metadata | Aucune commande dans les 90 jours suivant la date de référence (validité externe) |

Détail complet (distributions attendues, remarques) : [`data/README.md`](data/README.md).

---

## 5. Arborescence

```text
data-science/clustering/with-sklearn/
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
- `../../with-pytorch/` — pytorch
- `../../with-tensorflow/` — tensorflow

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
cd data-science/clustering/with-sklearn
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-dev.txt   # runtime + tests + lint + notebooks
```

**Option B — uv (plus rapide)**

```bash
cd data-science/clustering/with-sklearn
uv venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

**Option C — installation éditable du projet**

```bash
cd data-science/clustering/with-sklearn
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

Résultat : `data/raw/retail_customer_base.parquet` (+ `.csv` pour la lecture humaine)
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
python -m src.main mode=train model.params.n_clusters=6

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
python scripts/predict.py predict.input=data/raw/retail_customer_base.csv predict.n_samples=10
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
| `conf/model/default.yaml` | algorithme et hyperparamètres | `model.params.n_clusters=…` |
| `conf/train/default.yaml` | split, epochs, callbacks, noms d'artefacts | `++train.epochs=20`, `train.split.test_size=0.25` |
| `conf/preprocessing/default.yaml` | imputation, scaling, encodage, features dérivées | `preprocessing.numeric.scaler=robust`, `preprocessing.categorical.encoder=ordinal` |
| `conf/hydra/local.yaml` | répertoires de sortie Hydra (`outputs/`, `multirun/`) | `hydra.run.dir=outputs/debug` |

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
| `data/raw/retail_customer_base.parquet` | jeu de données synthétique (6,000 lignes) |
| `data/processed/*.parquet` | splits et données transformées |
| `artifacts/models/model.joblib` | modèle entraîné |
| `artifacts/models/preprocessing.joblib` | pipeline de preprocessing ajusté (aucune fuite) |
| `artifacts/models/model_card.json` | carte du modèle (params, métriques, features, date) |
| `artifacts/metrics/training_metrics.json` | métriques d'entraînement et d'évaluation |
| `artifacts/reports/evaluation_report.md` | rapport lisible (métriques, analyse d'erreurs, recommandations) |
| `artifacts/reports/predictions.csv` | prédictions sur l'échantillon de démonstration |
| `artifacts/figures/*.png` | figures spécifiques à la tâche |
| `outputs/<date>/<heure>/` | configuration composée + logs Hydra |

Métrique principale : **`silhouette`** (cible de smoke test : ≥ 0.25).
Métriques secondaires : calinski_harabasz, davies_bouldin, n_clusters.

---

## 13. Notebooks

Tous les notebooks sont **exécutables de bout en bout** (`make notebooks`) et documentés
cellule par cellule, comme un support de formation pour juniors.

| Notebook | Ce qu'on y apprend |
| --- | --- |
| `01_exploratory_analysis.ipynb` | EDA structurée : types, manquants, distributions univariées et bivariées, corrélations, outliers, puis **10-15 insights** actionnables. |
| `02_data_validation_and_schemas.ipynb` | Pourquoi des contrats de données : définition d'un `DataFrameModel` Pandera, validation réussie, puis **corruption volontaire** pour observer l'échec et le message d'erreur. |
| `03_preprocessing_and_features.ipynb` | Construction du pipeline : imputation, clipping, scaling, encodage, features dérivées, et démonstration de l'absence de fuite (fit sur train uniquement). |
| `04_model_exploration.ipynb` | Comparaison baseline + modèles candidats en validation croisée, table de métriques, choix argumenté du modèle. |
| `05_training_and_tracking.ipynb` | Entraînement instrumenté : callbacks, courbes d'apprentissage, métriques suivies, sauvegarde des artefacts. |
| `06_evaluation_and_error_analysis.ipynb` | **Analyse d'erreurs** : matrice de confusion, rapport par classe, exemples mal prédits, hypothèses sur les causes et recommandations concrètes. |

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
| `ModuleNotFoundError: No module named 'src'` | Exécution depuis un autre répertoire | `cd data-science/clustering/with-sklearn` puis `python -m src.main …` (ou `export PYTHONPATH=.`) |
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

*Dernière génération : 2026-09-13 · projet `ds-clustering-sklearn`*
