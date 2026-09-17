# Moteur de recommandation de catalogue e-commerce avec scikit-learn

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-classic-ml-orange)
![Hydra](https://img.shields.io/badge/config-Hydra-89b482)
![Pandera](https://img.shields.io/badge/validation-Pandera-16a34a)
![pytest](https://img.shields.io/badge/tests-pytest-2ea44f)
![Licence](https://img.shields.io/badge/license-MIT-lightgrey)

> **Domaine** : `data-science` · **Problématique** : `recommendation` · **Stack** : `scikit-learn`
> **Emplacement** : `data-science/recommendation/with-sklearn/`
> **Temps de lecture** : ~15 min · **Temps d'exécution complet** : < 5 min sur un laptop

---

## 1. À propos / Objectifs

Pipeline de recommandation de bout en bout (catalogue synthétique de 900 références x 1 200
utilisateurs -> génération de 36 000 couples (utilisateur, candidat) avec facteur latent d'attrait
et probabilité de pertinence -> contrats Pandera -> features d'affinité, d'intention, de signal
collaboratif et de fiche article -> gradient boosting par histogrammes -> classement par utilisateur
puis publication d'un top-10 sous contraintes métier) pour une équipe Personnalisation, entièrement
piloté par Hydra et structuré en objets testables. L'évaluation est menée **par utilisateur** et
jamais ligne à ligne : NDCG@10, Precision@10, Recall@10, MAP@10 et hit-rate confrontés au tirage
aléatoire, au tri par popularité et au plafond atteignable publié par le générateur, avec couverture
catalogue, concentration, démarrage à froid, emplacements perdus en rupture et backtest
chronologique. La probabilité oracle est une métadonnée : elle sert à borner la qualité, jamais à
entraîner.

Le site expose chaque visiteur à un catalogue de plusieurs milliers de références. Le classement
historique est un tri par popularité : il écrase la longue traîne, ignore l'affinité individuelle et
recommande des articles indisponibles. Une refonte du moteur de recommandation doit prouver son
apport **hors échantillon et par utilisateur**, pas seulement sur une moyenne globale.

**Ce que cet exemple démontre**

1. Construire un jeu de candidats (utilisateur x article) plutôt qu'une matrice d'interactions, et comprendre ce que ce choix autorise : consommer les features de fiche article, scorer un article neuf, publier sous contraintes métier.
2. Formaliser le contrat d'antériorité de chaque feature : connue avant la session (fiche article, historique agrégé), connue au moment de la session (intention récente), ou interdite (issue de la session elle-même).
3. Comprendre pourquoi l'unité d'évaluation est l'utilisateur et non la ligne : une précision calculée sur toutes les lignes mêle un client très actif à un nouveau venu et ne décrit aucun des deux.
4. Choisir les bonnes métriques de classement : NDCG et son discount logarithmique, Precision@K contre Recall@K, MAP, hit-rate — et savoir laquelle pilote quelle décision produit.
5. Confronter systématiquement un modèle à trois références (tirage aléatoire, tri par popularité, exploitation de l'intention récente) et à un plafond atteignable publié par le générateur, plutôt que commenter un NDCG absolu dénué de référence.
6. Comprendre pourquoi un générateur de données doit faire de la popularité une référence **forte mais battable** : sans facteur latent d'attrait commun aux vues et à la pertinence, le tri par popularité est ridicule et la comparaison n'apprend rien.
7. Mesurer ce que le NDCG ne voit pas : couverture catalogue, concentration (Herfindahl), biais de popularité, et la boucle de rétroaction qu'ils amorcent en production.
8. Traiter le démarrage à froid comme un objectif à part entière et mesurer le rappel des nouveaux utilisateurs rapporté au rappel global.
9. Appliquer les contraintes métier (disponibilité du stock, diversité par catégorie) comme des filtres post-classement explicites et testables, plutôt que de les laisser au modèle.
10. Chiffrer l'arbitrage entre pertinence et marge au lieu de le trancher implicitement dans le code de publication.
11. Analyser les erreurs de classement en deux familles — oublis et fausses promesses — dont les causes et les correctifs diffèrent, et les classer par coût NDCG.
12. Mesurer l'importance par permutation sur le NDCG plutôt que sur l'impureté : détecter qu'un moteur dont le score s'effondre quand on mélange l'audience est un trieur par popularité habillé en modèle.
13. Valider par backtest chronologique et lire la dispersion de la qualité entre périodes plutôt qu'un score unique.
14. Industrialiser l'ensemble : configuration Hydra, artefacts traçables, tests pytest, rapports générés, publication batch et unitaire d'un top-K.

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

Le site expose chaque visiteur à un catalogue de plusieurs milliers de références. Le classement
historique est un tri par popularité : il écrase la longue traîne, ignore l'affinité individuelle et
recommande des articles indisponibles. Une refonte du moteur de recommandation doit prouver son
apport **hors échantillon et par utilisateur**, pas seulement sur une moyenne globale.

| | |
| --- | --- |
| **Qui** (persona / consommateur) | Équipe Personnalisation d'un e-commerçant : data scientist qui industrialise le modèle de classement, responsable CRM qui pilote le taux de conversion et le panier moyen, et équipe catalogue qui surveille l'exposition de la longue traîne et la disponibilité des références. |
| **Problème** | Classer les articles candidats d'un utilisateur — les scorer puis publier un top-K — à partir des interactions passées, de la fiche article et du profil utilisateur, sans jamais utiliser d'information postérieure à la session scorée. |
| **Entrées** | Journal de candidats scorés : identifiants utilisateur et article, profil utilisateur (ancienneté, commandes, dépenses, amplitude d'exploration, canal, région, statut membre), fiche article (catégorie, prix, note, avis, stock, ancienneté, marge, audience, taux de conversion, promotion), signaux croisés utilisateur-article (affinité catégorie et marque, écart au prix habituel, consultations récentes, taux d'achat des utilisateurs similaires) et date de session. |
| **Sorties** | Score de pertinence par couple (utilisateur, article), top-K ordonné par utilisateur, indicateur de confiance (utilisateur froid, article jamais vu, candidat peu peuplé), couverture catalogue du top-K et marge attendue — car classer par pertinence et classer par marge ne donnent pas le même top-K, et l'arbitrage doit être explicite. |
| **Valeur métier** | Hausse du taux de conversion et du panier moyen, baisse du taux de retour (un article pertinent est un article gardé), exposition de la longue traîne au lieu du seul best-seller, et suppression des recommandations d'articles indisponibles qui dégradent la confiance. |
| **Cadence** | Scores recalculés à chaque ouverture de session (le classement doit refléter la dernière interaction), popularité rafraîchie plusieurs fois par jour, ré-entraînement quotidien sur fenêtre glissante et ré-indexation complète après un changement de catalogue. |

**Objectif de modélisation** — Classer les articles candidats d'un utilisateur — les scorer puis publier un top-K — à
partir des interactions passées, de la fiche article et du profil utilisateur, sans jamais
utiliser d'information postérieure à la session scorée.

**Critères de réussite**

- [ ] NDCG@10 ≥ 0,45 sur le split de test chronologique, pertinence binaire. Références du jeu : un scoreur aléatoire vaut 0,230, le tri par popularité 0,334, et le plafond atteignable (score latent parfait, bruit compris) 0,722 — l'ordre publié compte autant que le fait de recommander les bons articles.
- [ ] Precision@10 ≥ 0,22 : au moins un article sur cinq publié dans le top-10 est réellement pertinent (popularité : 0,163).
- [ ] Recall@10 ≥ 0,55 : le top-10 capture plus de la moitié de la pertinence disponible pour un utilisateur (popularité : 0,434).
- [ ] Le modèle doit battre la baseline **popularité** (classement par audience 7 jours) d'au moins 25 % en NDCG@10 : sans cet écart, un simple tri par best-seller suffit et le modèle n'a pas de raison d'exister.
- [ ] Couverture catalogue ≥ 25 % des références présentes dans au moins un top-10 du test. Ce critère distingue deux moteurs de même pertinence : le tri par popularité n'expose que 8,4 % du catalogue et échoue à ce critère, ce qui est précisément le mécanisme par lequel la longue traîne meurt.
- [ ] Le rappel@10 sur les utilisateurs froids (au plus 1 commande sur 12 mois, ~29 % des utilisateurs) doit valoir au moins 60 % du rappel global : un moteur qui abandonne les nouveaux venus perd exactement les clients qu'il faut convertir. Ce critère est la sanction directe de l'interdiction d'utiliser les identifiants comme features.
- [ ] Reproductibilité : deux exécutions avec la même graine produisent exactement les mêmes métriques, et la génération est déterministe à la graine près.

**Contraintes**

- Aucune donnée réelle : le catalogue, les utilisateurs et les interactions de ce dépôt sont synthétiques et hors ligne.
- `user_id` et `item_id` sont des **identifiants**, jamais des features : un modèle qui mémorise un identifiant ne sait rien dire d'un utilisateur ou d'un article jamais vu. La généralisation passe par les signaux croisés.
- Le split est chronologique : jamais de mélange aléatoire. Évaluer sur des sessions passées avec un modèle entraîné sur des sessions futures, c'est mesurer un moteur qui triche.
- Aucune statistique postérieure à la session scorée : l'audience et le taux de conversion d'un article sont des fenêtres glissantes **antérieures**, jamais la période cible.
- Un article indisponible ne doit pas être publié : le filtre stock est appliqué après scoring, ce qui est mesuré séparément car il modifie la couverture.
- Latence de classement < 50 ms pour 200 candidats sur CPU (contrainte de temps réel session).

---

## 4. Données

Les données sont **synthétiques**, générées localement par
[`src/data/generators.py`](src/data/generators.py) : aucune donnée réelle, aucun téléchargement,
aucune information confidentielle. Elles sont néanmoins construites pour être **crédibles
métier** (corrélations réalistes, bruit, valeurs manquantes, outliers légitimes).

| Propriété | Valeur |
| --- | --- |
| Jeu de données | `ecommerce_candidate_impressions` |
| Volume par défaut | 36 000 lignes |
| Formats écrits | parquet, csv |
| Emplacement | `data/raw/ecommerce_candidate_impressions.parquet` (et `.csv`) |
| Cible | `relevance` |
| Identifiant | `sample_id` |
| Colonne temporelle | `session_date` |
| Graine | `42` (reproductible) |

**Schéma**

| Colonne | Type | Rôle | Signification métier |
| --- | --- | --- | --- |
| `sample_id` | str | identifier | Identifiant unique du couple (utilisateur, article candidat) |
| `user_id` | str | group | Identifiant utilisateur — clé de regroupement des métriques de classement, jamais une feature |
| `item_id` | str | identifier | Identifiant article du catalogue — jamais une feature, pour les mêmes raisons |
| `session_date` | datetime | timestamp | Date de la session au cours de laquelle le candidat a été scoré — clé du split chronologique |
| `user_tenure_days` | int | feature | Ancienneté du compte utilisateur en jours (jours) |
| `user_orders_12m` | int | feature | Nombre de commandes de l'utilisateur sur les 12 derniers mois — pilote le segment froid / tiède / chaud (commandes) |
| `user_spend_eur_12m` | float | feature | Dépense cumulée de l'utilisateur sur 12 mois (EUR) |
| `user_sessions_30d` | int | feature | Sessions ouvertes sur les 30 derniers jours — intensité d'usage récente (sessions) |
| `user_distinct_categories_12m` | int | feature | Nombre de catégories distinctes achetées sur 12 mois — amplitude d'exploration (catégories) |
| `user_avg_basket_eur` | float | feature | Panier moyen de l'utilisateur (EUR) |
| `user_typical_price_eur` | float | feature | Prix médian des articles achetés par l'utilisateur — référence pour mesurer l'écart de prix d'un candidat (EUR) |
| `user_return_rate` | float | feature | Part des commandes retournées sur 12 mois — un retour fréquent signale une pertinence mal calibrée (ratio) |
| `user_price_band_pref` | category | feature | Gamme de prix préférée de l'utilisateur |
| `user_channel` | category | feature | Canal principal d'achat (le classement mobile privilégie les fiches courtes) |
| `user_region` | str | feature | Région de livraison — effet sur la disponibilité et les délais |
| `user_is_member` | bool | feature | Adhésion au programme de fidélité |
| `item_category` | category | feature | Catégorie catalogue de l'article |
| `item_price_eur` | float | feature | Prix de vente de l'article (EUR) |
| `item_rating_avg` | float | feature | Note moyenne de l'article (/5) |
| `item_reviews_count` | int | feature | Nombre d'avis publiés — preuve sociale, mais aussi proxy de l'âge de l'article (avis) |
| `item_stock_units` | int | feature | Stock disponible au moment du scoring — un stock nul interdit la publication (unités) |
| `item_age_days` | int | feature | Ancienneté de l'article au catalogue (jours) |
| `item_margin_pct` | float | feature | Marge brute de l'article — permet d'arbitrer explicitement pertinence contre valeur (%) |
| `item_views_7d` | int | feature | Vues de l'article sur les 7 jours **précédant** la session — fenêtre strictement antérieure, donc sans fuite (vues) |
| `item_conversion_rate_30d` | float | feature | Taux de conversion de la fiche sur 30 jours glissants antérieurs (ratio) |
| `item_is_promoted` | bool | feature | Article poussé par une opération marketing au moment de la session |
| `user_category_affinity` | float | feature | Part des dépenses de l'utilisateur dans la catégorie de l'article — le signal croisé le plus fort (ratio) |
| `user_brand_affinity` | float | feature | Affinité de l'utilisateur à la marque de l'article (historique d'achats et de consultations) (ratio) |
| `price_gap_pct` | float | feature | Écart relatif entre le prix du candidat et le prix habituel de l'utilisateur — un écart fort fait chuter la pertinence (%) |
| `days_since_last_view` | int | feature | Jours écoulés depuis la dernière consultation de cet article par cet utilisateur (jours) |
| `user_item_views_30d` | int | feature | Consultations de ce candidat par cet utilisateur sur 30 jours glissants antérieurs (vues) |
| `similar_users_buy_rate` | float | feature | Taux d'achat de ce candidat par les utilisateurs les plus proches (signal collaboratif précalculé sur fenêtre antérieure) (ratio) |
| `relevance` | int | target | Pertinence observée : 1 si l'utilisateur a ajouté au panier ou acheté, 0 sinon |

Détail complet (distributions attendues, remarques) : [`data/README.md`](data/README.md).

---

## 5. Arborescence

```text
data-science/recommendation/with-sklearn/
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
cd data-science/recommendation/with-sklearn
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-dev.txt   # runtime + tests + lint + notebooks
```

**Option B — uv (plus rapide)**

```bash
cd data-science/recommendation/with-sklearn
uv venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

**Option C — installation éditable du projet**

```bash
cd data-science/recommendation/with-sklearn
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

Résultat : `data/raw/ecommerce_candidate_impressions.parquet` (+ `.csv` pour la lecture humaine)
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
python -m src.main mode=train model.params.max_iter=300

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
python scripts/predict.py predict.input=data/raw/ecommerce_candidate_impressions.csv predict.n_samples=10
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
| `conf/config.yaml` -> bloc `recommendation` | Réglages métier de la famille, au même niveau que `mode` et `seed` : 16 clés (top_k, cutoffs, group_column, item_column, popularity_column, intent_column, stock_column, …), chacune commentée dans le fichier — c'est là qu'on change le métier sans toucher au code. | `recommendation.top_k=10`, `recommendation.group_column=user_id` |

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
| `data/raw/ecommerce_candidate_impressions.parquet` (+ `csv`) | jeu de données synthétique, 36 000 lignes |
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
| `artifacts/reports/predictions.csv` | prédictions sur l'échantillon de démonstration |
| `artifacts/figures/*.png` | figures spécifiques à la tâche |
| `outputs/<date>/<heure>/` | configuration composée + logs Hydra |

Métrique principale : **`ndcg_at_k`** (sens `maximize`, seuil de smoke test : ≥ 0.45).
Métriques secondaires : precision_at_k, recall_at_k, map_at_k, hit_rate_at_k.

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
| `ModuleNotFoundError: No module named 'src'` | Exécution depuis un autre répertoire | `cd data-science/recommendation/with-sklearn` puis `python -m src.main …` (ou `export PYTHONPATH=.`) |
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

*Dernière génération : 2026-09-17 · projet `ds-recommendation-sklearn`*
