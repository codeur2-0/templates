# Routage de tickets de support par TF-IDF — la référence explicable

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn (TF-IDF, classification de texte)](https://img.shields.io/badge/scikit--learn%20(TF--IDF,%20classification%20de%20texte)-classic-ml-orange)
![Hydra](https://img.shields.io/badge/config-Hydra-89b482)
![Pandera](https://img.shields.io/badge/validation-Pandera-16a34a)
![pytest](https://img.shields.io/badge/tests-pytest-2ea44f)
![Licence](https://img.shields.io/badge/license-MIT-lightgrey)

> **Domaine** : `ai-eng` · **Problématique** : `text-classification` · **Stack** : `scikit-learn (TF-IDF, classification de texte)`
> **Emplacement** : `ai-eng/text-classification/with-tfidf_classifier/`
> **Temps de lecture** : ~15 min · **Temps d'exécution complet** : < 5 min sur un laptop

---

## 1. À propos / Objectifs

Six catégories de support, 1 200 tickets synthétiques écrits trois fois — vocabulaire de la
catégorie, paraphrase, bruit partagé — et une classe « autre » volontairement sans vocabulaire
propre. Le classifieur est une régression logistique multinomiale sur TF-IDF : un poids par terme et
par classe, donc une décision qui se relit terme à terme et une explication affichable au
conseiller. Ce projet est la **référence** de la famille : il chiffre ce qu'un modèle qui lit les
mots obtient quand un tiers des tickets ne les emploie pas, publie l'écart entre le segment
canonique et le segment paraphrase, et laisse le notebook 04 comparer les trois algorithmes de la
stack sur le split de validation.

Le support reçoit 1 200 tickets par semaine par quatre canaux (formulaire, courriel, chat,
courrier). Aujourd'hui, chaque ticket est lu par un conseiller qui le réaffecte à la main au bon
service : facturation, livraison, produit défectueux, remboursement, compte client — ou « autre »
quand la demande ne rentre dans aucune case. Le réaiguillage coûte une demi-journée par semaine et,
surtout, il se voit : un ticket de remboursement routé vers la livraison repart pour un tour et le
client attend deux jours de plus.

**Ce que cet exemple démontre**

1. Structurer un projet text-classification prêt pour la production (OOP, typage, tests, configuration déclarative).
2. Valider les données avec des contrats exécutables **Pandera** à chaque étape critique.
3. Configurer l'intégralité du projet avec **Hydra** (aucune valeur codée en dur).
4. Rendre le résultat **reproductible** : graine fixée, artefacts horodatés, rapports générés.

**Pourquoi scikit-learn (TF-IDF, classification de texte) ici ?**

- Classifieur linéaire **explicable** : un poids par terme et par classe, donc une décision qui se relit terme à terme (`explain`).
- Trois algorithmes servis : `tfidf_logreg` (référence), `tfidf_nb` (bayésien naïf complémentaire, témoin rapide) et `tfidf_centroid` (plancher sans apprentissage de poids).
- Aucun synonyme connu : la paraphrase est la limite mesurée de la stack, et le taux de termes hors vocabulaire est publié dans la fiche de modèle.

---

## 2. Stack technologique

| Rôle | Outil | Version minimale | Pourquoi |
| --- | --- | --- | --- |
| Framework ML | **scikit-learn (TF-IDF, classification de texte)** | joblib>=1.3 | classic-ml |
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

Le support reçoit 1 200 tickets par semaine par quatre canaux (formulaire, courriel, chat,
courrier). Aujourd'hui, chaque ticket est lu par un conseiller qui le réaffecte à la main au bon
service : facturation, livraison, produit défectueux, remboursement, compte client — ou « autre »
quand la demande ne rentre dans aucune case. Le réaiguillage coûte une demi-journée par semaine et,
surtout, il se voit : un ticket de remboursement routé vers la livraison repart pour un tour et le
client attend deux jours de plus.

| | |
| --- | --- |
| **Qui** (persona / consommateur) | Responsable du support de niveau 1 d'un distributeur de matériel électronique (300 personnes), avec un ingénieur IA qui outille le routage des tickets et une équipe qualité qui mesure les réclamations mal orientées. |
| **Problème** | Attribuer automatiquement à chaque ticket sa catégorie de service, **avec une décision explicable** (les termes qui ont pesé), sans jamais sacrifier la classe minoritaire « autre » au profit de l'exactitude globale — et sans dépendre d'un vocabulaire que le client n'emploie pas forcément. |
| **Entrées** | Un ticket de support : texte libre reçu par formulaire, courriel, chat ou courrier, avec sa date de réception, sa priorité déclarée par le client et son canal d'arrivée. |
| **Sorties** | La catégorie du ticket, la confiance associée, la probabilité de chacune des six classes, les termes qui ont décidé (pour les modèles linéaires) et une table de prédictions jointe aux artefacts, prête à être comparée au routage humain. |
| **Valeur métier** | Le temps de réaiguillage passe de la demi-journée par semaine à une relecture des seuls tickets douteux, et chaque erreur devient diagnosticable : la ventilation par style montre que le modèle lit les mots plutôt qu'il ne comprend la demande, ce qui est un résultat **publié**, pas caché. |
| **Cadence** | Entraînement mensuel sur les tickets étiquetés ; classification en ligne à la réception. |

**Objectif de modélisation** — Attribuer automatiquement à chaque ticket sa catégorie de service, **avec une décision
explicable** (les termes qui ont pesé), sans jamais sacrifier la classe minoritaire « autre
» au profit de l'exactitude globale — et sans dépendre d'un vocabulaire que le client
n'emploie pas forcément.

**Critères de réussite**

- [ ] F1 macro ≥ 0,70 sur le split de test : la métrique principale pèse chaque classe à égalité, donc la classe minoritaire « autre » compte autant que les cinq autres.
- [ ] Exactitude ≥ 0,75 : le plancher trivial (classe majoritaire) vaut ≈ 0,20, il est publié à côté des métriques pour que le chiffre soit situé.
- [ ] Écart canonique / paraphrase **publié** : l'écart de F1 macro entre les tickets écrits avec le vocabulaire de leur catégorie et ceux qui ne l'utilisent pas mesure la limite de l'approche, et figure dans le rapport.
- [ ] La priorité déclarée et le canal d'arrivée sont des distracteurs **déclarés** : leur gain de raccourci reste sous 0,20 par rapport à la meilleure règle constante, et la suite de tests le vérifie à chaque exécution.
- [ ] Chaque prédiction est explicable : les modèles servis exposent les termes qui ont décidé, et l'explication ne cite que des termes réellement présents dans le texte.
- [ ] Le classifieur est reproductible (graine fixée) et son artefact, rechargé, classe exactement comme le modèle ajusté — probabilités et explications comprises.
- [ ] Aucune dépendance réseau à l'exécution : corpus synthétique, vocabulaire appris sur le train, aucune clé d'API.
- [ ] Les tournures du test ne sont **jamais** vues à l'entraînement (deux formulations par classe et par style sont réservées aux splits d'évaluation) : la F1 publiée mesure une généralisation, pas une recopie.

**Contraintes**

- Aucune donnée personnelle : le corpus est synthétique et les textes sont des gabarits paramétrés.
- La classe « autre » ne doit jamais être absorbée : c'est elle qui justifie une métrique macro plutôt qu'une exactitude globale.
- Une réponse n'est acceptée que si elle est explicable : un score sans termes associés n'est pas relu par un conseiller.

---

## 4. Données

Les données sont **synthétiques**, générées localement par
[`src/data/generators.py`](src/data/generators.py) : aucune donnée réelle, aucun téléchargement,
aucune information confidentielle. Elles sont néanmoins construites pour être **crédibles
métier** (corrélations réalistes, bruit, valeurs manquantes, outliers légitimes).

| Propriété | Valeur |
| --- | --- |
| Jeu de données | `support_tickets` |
| Volume par défaut | 1 200 lignes |
| Formats écrits | parquet, csv |
| Emplacement | `data/raw/support_tickets.parquet` (et `.csv`) |
| Cible | `label` |
| Identifiant | `doc_id` |
| Colonne temporelle | `published_at` |
| Graine | `42` (reproductible) |

**Schéma**

| Colonne | Type | Rôle | Signification métier |
| --- | --- | --- | --- |
| `doc_id` | str | identifier | Identifiant stable du ticket (clé de jointure, jamais une feature). |
| `text` | str | feature | Texte du ticket tel qu'il a été reçu, gabarit paramétré par produit et référence de commande. |
| `label` | category | target | Catégorie du ticket : la cible du modèle, celle que le conseiller aurait choisie. |
| `source` | category | feature | Canal d'arrivée du ticket : formulaire, courriel, chat ou courrier. |
| `style` | category | metadata | Style rédactionnel : canonique (vocabulaire de la catégorie), paraphrase (la même demande sans ce vocabulaire), bruite (formule de politesse et signature partagées). |
| `priority` | category | metadata | Priorité déclarée par le client : un distracteur, tiré indépendamment du libellé. |
| `published_at` | datetime | timestamp | Date de réception du ticket (les dates remontent sur six mois). |
| `n_tokens` | int | metadata | Nombre de tokens du texte, calculé avec le tokenizer du projet. |
| `split` | category | group | Découpage du corpus (train / val / calibration / test), stratifié par classe et par style au moment de la génération. |

Détail complet (distributions attendues, remarques) : [`data/README.md`](data/README.md).

---

## 5. Arborescence

```text
ai-eng/text-classification/with-tfidf_classifier/
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
│   ├── models/                  # BaseModel (ABC) + implémentation scikit-learn (TF-IDF, classification de texte)
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
   ├── TfidfClassifier(BaseModel)   ← implémentation scikit-learn (TF-IDF, classification de texte)
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
                     → TfidfClassifier.fit (callbacks)
                     → Evaluator.evaluate → ReportBuilder → artifacts/{models,metrics,reports,figures}
mode=evaluate      → chargement des artefacts → nouvelle évaluation + rapports
mode=predict       → InferenceDataSchema → Predictor.predict → artifacts/reports/predictions.csv
```

### 6.4 Comparaison avec les autres stacks

Le **même cas d'usage** est implémenté avec d'autres frameworks dans les dossiers voisins.
La structure, les contrats de données et les métriques étant identiques, la comparaison est
directe (qualité, temps d'entraînement, lisibilité du code) :

- `../../with-tfidf_classifier/` — tfidf_classifier
- `../../with-transformers/` — transformers

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
cd ai-eng/text-classification/with-tfidf_classifier
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-dev.txt   # runtime + tests + lint + notebooks
```

**Option B — uv (plus rapide)**

```bash
cd ai-eng/text-classification/with-tfidf_classifier
uv venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

**Option C — installation éditable du projet**

```bash
cd ai-eng/text-classification/with-tfidf_classifier
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

Résultat : `data/raw/support_tickets.parquet` (+ `.csv` pour la lecture humaine)
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
python -m src.main mode=train model.params.lexical={'mode': 'tfidf', 'ngram_range': [1, 2], 'min_df': 2, 'sublinear_tf': True}

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
python scripts/predict.py predict.input=data/raw/support_tickets.csv predict.n_samples=10
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
from src.models.model import TfidfClassifier
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
| `conf/model/default.yaml` | algorithme et hyperparamètres | `model.params.lexical=…` |
| `conf/train/default.yaml` | split, epochs, callbacks, noms d'artefacts | `++train.epochs=20`, `train.split.test_size=0.25` |
| `conf/preprocessing/default.yaml` | imputation, scaling, encodage, features dérivées | `preprocessing.numeric.scaler=robust`, `preprocessing.categorical.encoder=ordinal` |
| `conf/hydra/local.yaml` | répertoires de sortie Hydra (`outputs/`, `multirun/`) | `hydra.run.dir=outputs/debug` |
| `conf/config.yaml` -> bloc `train` | Réglages métier de la famille, au même niveau que `mode` et `seed` : 1 clés (artifacts), chacune commentée dans le fichier — c'est là qu'on change le métier sans toucher au code. | `train.artifacts={'model_file': 'classifier.joblib'}` |

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
| `data/raw/support_tickets.parquet` (+ `csv`) | jeu de données synthétique, 1 200 lignes |
| `data/raw/generation_metadata.json` | recette de génération : graine, options, empreinte du jeu, fichiers écrits |
| `data/processed/chunks.parquet` (+ `csv`) | passages indexés : le texte découpé, ses métadonnées et les annotations conservées |
| `artifacts/models/classifier.joblib` | modèle entraîné |
| `artifacts/models/preprocessing.joblib` | pipeline de preprocessing ajusté (aucune fuite) |
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

Métrique principale : **`macro_f1`** (sens `maximize`, seuil de smoke test : ≥ 0.7).
Métriques secondaires : accuracy, weighted_f1, balanced_accuracy, macro_precision, macro_recall, cohen_kappa, mean_confidence, expected_calibration_error, latency_p50_ms, latency_p95_ms, coverage.

Résultats de l'exécution de référence (`make all`, graine 42) :

| Indicateur | Valeur mesurée |
| --- | --- |
| F1 macro sur le split de test (six classes, chacune pesant pareil) | **0,8408** |
| Exactitude sur le split de test (plancher trivial : 0,2034 ; tirage stratifié : 0,1661) | **0,8531** |
| F1 macro du segment canonique (93 tickets) / du segment paraphrase (52) | **0,9262 / 0,7352** |
| F1 macro du segment bruité (32 tickets : politesse et signature partagées) | **0,7487** |
| Référence « règle à mots-clés » publiée avec le corpus | **0,5492** |
| Kappa de Cohen / exactitude équilibrée | **0,8217 / 0,8337** |
| Écart de calibration (confiance moyenne 0,6168 contre exactitude 0,8531) | **0,2363** |
| Latence p50 / p95 d'une prédiction (TF-IDF, 6 classes) | **0,67 ms / 0,78 ms** |
| Formulations réservées par classe et par style, jamais vues à l'entraînement | **2 sur 6** |

Ces valeurs sont reproductibles à l'identique ; elles proviennent du split de test, jamais
du split d'entraînement, et le détail complet est dans `artifacts/reports/evaluation_report.md`.

---

## 13. Notebooks

Tous les notebooks sont **exécutables de bout en bout** (`make notebooks`) et documentés
cellule par cellule, comme un support de formation pour juniors.


| Notebook | Ce qu'on y apprend |
| --- | --- |
| `01_eda.ipynb` | Exploration d'un **corpus documentaire** : sources, sections, longueurs, dates de publication, doublons d'identifiant, puis lecture du **jeu de questions** (difficultés, intentions, questions hors corpus, documents annotés par question). |
| `02_validation.ipynb` | Contrats Pandera du texte : colonnes, formats d'identifiants et cohérence « question hors corpus ⇔ extrait vide », puis **corruption volontaire** (identifiant hors format, source inconnue, question répondable sans document) pour lire le message d'échec. |
| `03_preprocessing.ipynb` | Découpage en passages : offsets vérifiés contre le texte source, chevauchement, filtrage des mots vides, puis **mesure** de l'effet de la taille des passages sur le rappel@5 — avec l'intervalle de confiance qui distingue un écart réel d'un bruit d'échantillonnage. |
| `04_model_exploration.ipynb` | Planchers triviaux installés d'abord (tirage aléatoire, ordre du corpus), comparaison des scorers lexicaux **à protocole identique**, puis courbe du seuil d'abstention sur le split de calibration : refus corrects contre couverture, jamais une exactitude globale. |
| `05_training.ipynb` | *Entraînement dans les conditions de production* — Le pipeline réel (`TrainPipeline`) exécuté dans un bac à sable, lecture des artefacts (passages indexés, fiche de modèle, configuration résolue) et **preuve de reproductibilité** : reconstruire l'index retrouve les mêmes passages. |
| `06_error_analysis.ipynb` | *Analyse d'erreurs et recommandations* — Verdict contractuel, ventilation par difficulté et par intention, questions perdues examinées une par une, comportement d'abstention mesuré, puis recommandations reliées à un chiffre. |

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
| `tests/test_data_schemas.py` | Les contrats Pandera acceptent un corpus valide **et** rejettent les données corrompues : identifiant hors format, source inconnue, question hors corpus annotée avec un extrait, colonne manquante. |
| `tests/test_loaders.py` | Chargement Parquet/CSV du corpus et des questions annotées, validation appliquée, répartition calibration / validation / test et rejet d'un fichier corrompu. |
| `tests/test_preprocessing.py` | Découpage en passages (offsets exacts, chevauchement, longueur), tokenisation et mots vides, vectoriseur lexical (BM25 / TF-IDF) et **persistance** de l'estimateur appris. |
| `tests/test_models.py` | Contrat `BaseModel` : fit → retrieve → answer, abstention sous le seuil, déterminisme, sauvegarde / rechargement de l'index et garde-fous (modèle non entraîné, `k` invalide). |
| `tests/test_training.py` | Le `Trainer` produit des métriques par question (recall@k, MRR, nDCG), les callbacks fonctionnent, et une question hors corpus ne fausse pas le rappel. |
| `tests/test_evaluation.py` | La couche qui publie les chiffres : questions hors corpus **exclues du dénominateur** du rappel, `citation_precision` non applicable (et non nulle) quand rien n'est cité, ventilation par segment qui totalise les questions évaluées, et rapport dont le verdict découle de la mesure. |
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
- **Reproductibilité** : graine propagée à Python/NumPy/scikit-learn (TF-IDF, classification de texte), données régénérables à l'identique.
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
| `src/models/` | `BaseModel` (ABC) + implémentation scikit-learn (TF-IDF, classification de texte) | Aucune logique de training loop ici : le modèle expose un contrat. |
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
| `ModuleNotFoundError: No module named 'src'` | Exécution depuis un autre répertoire | `cd ai-eng/text-classification/with-tfidf_classifier` puis `python -m src.main …` (ou `export PYTHONPATH=.`) |
| `FileNotFoundError: data/raw/...` | Données non générées | `make data` (ou `python scripts/generate_data.py`) |
| `SchemaError` Pandera au chargement | Dataset corrompu / régénéré avec un autre schéma | `make clean-artifacts && make data` |
| `Could not find a version that satisfies the requirement scikit-learn (tf-idf, classification de texte)` | Index PyPI inaccessible / Python trop ancien | Python ≥ 3.10, `pip install --upgrade pip`, vérifier le proxy |
| Erreur `Key 'X' not in ...` sur un override Hydra | Clé absente du schéma | Préfixer l'override avec `++` (ex. `++train.epochs=5`) |
| Les chemins pointent vers `outputs/...` | Un outil a changé le CWD | `hydra.job.chdir=false` est déjà actif ; les chemins viennent de `src/utils/paths.py` |
| Tests lents | Entraînement complet dans les tests | Les fixtures utilisent un petit dataset ; `pytest -m "not slow"` |

Pour rejouer une configuration exacte : Hydra sauvegarde la config composée dans
`outputs/<date>/<heure>/.hydra/config.yaml` — copiez-la avec `--config-path`/`--config-name`.

---

## 18. Références

- [scikit-learn : classification de textes](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [scikit-learn : régression logistique](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
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

*Dernière génération : 2026-10-06 · projet `ai-eng-text-classification-tfidf`*
