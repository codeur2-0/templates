# Support client sur fiches produit avec scikit-learn (BM25) — QA extractive de référence

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn (TF-IDF / BM25)](https://img.shields.io/badge/scikit--learn%20(TF--IDF%20/%20BM25)-classic-ml-orange)
![Hydra](https://img.shields.io/badge/config-Hydra-89b482)
![Pandera](https://img.shields.io/badge/validation-Pandera-16a34a)
![pytest](https://img.shields.io/badge/tests-pytest-2ea44f)
![Licence](https://img.shields.io/badge/license-MIT-lightgrey)

> **Domaine** : `ai-eng` · **Problématique** : `question-answering` · **Stack** : `scikit-learn (TF-IDF / BM25)`
> **Emplacement** : `ai-eng/question-answering/with-tfidf/`
> **Temps de lecture** : ~15 min · **Temps d'exécution complet** : < 5 min sur un laptop

---

## 1. À propos / Objectifs

Réponse aux questions du support client par la phrase exacte de la fiche produit (corpus synthétique
de 120 fiches -> contrats Pandera -> découpage -> index BM25 appris sur le train -> classement des
fiches -> phrase copiée du meilleur passage ou abstention explicite), entièrement piloté par Hydra
et structuré en objets testables. Le contrat de cette famille est la **phrase exacte** :
l'évaluation mesure l'exact match et le F1 de réponse en plus du rappel, confronte le tout au tirage
aléatoire et à l'ordre du corpus, puis ventile les résultats par difficulté (factuelle, paraphrase,
multi-document) et par comportement d'abstention sur les questions hors corpus.

Le service client répond chaque jour à des questions factuelles — durée de garantie, délai de
retour, coût des frais, prise en charge SAV — dont la réponse existe déjà, écrite noir sur blanc,
dans les fiches produit internes. Les conseillers répondent de mémoire : une durée de garantie
annoncée de 24 mois au lieu de 36 coûte un litige, et chaque réponse approximative nourrit un
contentieux.

**Ce que cet exemple démontre**

1. Écrire un générateur de corpus dont la réponse de référence est une phrase du corpus, afin que l'exact match mesure la sélection de la bonne phrase.
2. Séparer le vocabulaire des fiches (garantie, délai, montant, panne, compatibilité) pour que la question ait une seule bonne fiche.
3. Distinguer une question difficile d'une question ambiguë : la paraphrase s'éloigne du texte, elle ne retire pas l'entité qui identifie la réponse.
4. Composer une réponse multi-document (deux phrases, deux fiches) et mesurer le plafond que cela crée.
5. Traiter l'abstention comme une décision mesurée : une réponse refusée coûte, une réponse inventée coûte davantage.
6. Comparer exact match et F1 de réponse : le premier exige la formulation exacte, le second pardonne la reformulation.
7. Lire une matrice de confusion d'abstention par segment plutôt qu'une exactitude globale.
8. Industrialiser : index persisté, artefact rechargé à l'identique, seuil calibré hors du split de test.

**Pourquoi scikit-learn (TF-IDF / BM25) ici ?**

- Reference lexicale hors ligne : TF-IDF + cosinus, et BM25 (Okapi) implemente a la main avec les memes statistiques de corpus.
- Aucun modele pre-entraine, aucun telechargement : le vocabulaire est appris sur le train et persiste avec l'index.
- Sert de baseline a toutes les familles texte : un modele neuronal qui ne la bat pas ne merite pas son cout.

---

## 2. Stack technologique

| Rôle | Outil | Version minimale | Pourquoi |
| --- | --- | --- | --- |
| Framework ML | **scikit-learn (TF-IDF / BM25)** | joblib>=1.3 | classic-ml |
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

Le service client répond chaque jour à des questions factuelles — durée de garantie, délai de
retour, coût des frais, prise en charge SAV — dont la réponse existe déjà, écrite noir sur blanc,
dans les fiches produit internes. Les conseillers répondent de mémoire : une durée de garantie
annoncée de 24 mois au lieu de 36 coûte un litige, et chaque réponse approximative nourrit un
contentieux.

| | |
| --- | --- |
| **Qui** (persona / consommateur) | Support client de niveau 1 d'un e-commerçant de 300 personnes, avec un responsable qualité qui répond aux litiges et un ingénieur IA chargé d'outiller l'équipe. |
| **Problème** | Répondre à une question client par la **phrase exacte** de la fiche produit qui fait foi, en citant la fiche, et refuser explicitement de répondre quand aucune fiche ne contient la réponse — plutôt que de produire une phrase plausible. |
| **Entrées** | Question d'un client (ou d'un conseiller) en langage naturel, et un corpus de fiches produit versionnées : titre, section, espace wiki d'origine, date de publication, texte. |
| **Sorties** | La phrase de la fiche qui répond à la question (copiée, pas reformulée), l'identifiant de la fiche citée, le score de confiance du meilleur passage et une décision explicite de réponse ou d'abstention. |
| **Valeur métier** | Zéro réponse inventée sur les sujets contractuels (garantie, retour, frais), temps de réponse divisé par deux au niveau 1, et une réponse auditable : chaque phrase produite est retrouvable dans une fiche datée, ce qui rend un litige arbitrable. |
| **Cadence** | Index reconstruit à chaque publication de fiche produit ; réponse en ligne pour le support. |

**Objectif de modélisation** — Répondre à une question client par la **phrase exacte** de la fiche produit qui fait foi, en
citant la fiche, et refuser explicitement de répondre quand aucune fiche ne contient la
réponse — plutôt que de produire une phrase plausible.

**Critères de réussite**

- [ ] Exact match ≥ 0,55 sur les questions extractives du split de test : la phrase doit être la bonne, pas seulement proche (baseline mesurée : 0,62).
- [ ] F1 de réponse ≥ 0,70 : une phrase tronquée ou complétée coûte des points (baseline mesurée : 0,74).
- [ ] recall@1 ≥ 0,75 et recall@5 ≥ 0,95 : la phrase citée doit venir du meilleur passage, pas seulement d'un passage retrouvé (baseline mesurée : 0,83 et 1,00).
- [ ] Précision des citations ≥ 0,85 : une réponse n'annexe pas la phrase d'une fiche deux fois moins bien classée (baseline mesurée : 0,98).
- [ ] Taux de réponse sur les questions hors corpus ≤ 0,20 : une fiche citée à tort est pire qu'un refus (baseline mesurée : 0,17).
- [ ] Exactitude équilibrée de la décision réponse/abstention ≥ 0,75 : refuser une question répondable coûte aussi cher que répondre à côté (baseline mesurée : 0,81).
- [ ] Couverture des questions répondables ≥ 0,75 : un refus coûte une réponse que le corpus contenait (baseline mesurée : 0,79).
- [ ] Latence p95 < 250 ms sur CPU, sans appel réseau, pour un corpus de quelques centaines de fiches (baseline mesurée : 5,1 ms).
- [ ] Reproductibilité : deux exécutions à graine fixée produisent exactement les mêmes métriques.

**Contraintes**

- Aucune clé API et aucun téléchargement : l'assistant tourne hors ligne par défaut.
- Le corpus est synthétique : aucune donnée client, aucune donnée personnelle.
- Toute réponse doit citer une fiche identifiable dans le corpus.

---

## 4. Données

Les données sont **synthétiques**, générées localement par
[`src/data/generators.py`](src/data/generators.py) : aucune donnée réelle, aucun téléchargement,
aucune information confidentielle. Elles sont néanmoins construites pour être **crédibles
métier** (corrélations réalistes, bruit, valeurs manquantes, outliers légitimes).

| Propriété | Valeur |
| --- | --- |
| Jeu de données | `product_support_faq` |
| Volume par défaut | 120 lignes |
| Formats écrits | parquet, csv |
| Emplacement | `data/raw/product_support_faq.parquet` (et `.csv`) |
| Cible | *aucune* (apprentissage non supervisé) |
| Identifiant | `doc_id` |
| Colonne temporelle | `published_at` |
| Graine | `42` (reproductible) |

**Schéma**

| Colonne | Type | Rôle | Signification métier |
| --- | --- | --- | --- |
| `doc_id` | str | identifier | Identifiant stable de la fiche (clé de jointure, jamais une feature). |
| `title` | str | metadata | Titre de la fiche : le type d'article et le produit concerné. |
| `section` | category | feature | Section éditoriale : definition, politique, calcul, procedure, contact. |
| `source` | category | feature | Espace wiki d'origine : wiki_juridique, wiki_support, wiki_finance, wiki_it. |
| `published_at` | datetime | timestamp | Date de publication de la fiche (sert à calculer son ancienneté). |
| `n_tokens` | int | metadata | Nombre de tokens du texte, calculé avec le tokenizer du projet. |
| `text` | str | feature | Texte intégral de la fiche : c'est lui qui contient la phrase canonique. |

Détail complet (distributions attendues, remarques) : [`data/README.md`](data/README.md).

---

## 5. Arborescence

```text
ai-eng/question-answering/with-tfidf/
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
│   ├── models/                  # BaseModel (ABC) + implémentation scikit-learn (TF-IDF / BM25)
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
   ├── TfidfModel(BaseModel)   ← implémentation scikit-learn (TF-IDF / BM25)
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
                     → TfidfModel.fit (callbacks)
                     → Evaluator.evaluate → ReportBuilder → artifacts/{models,metrics,reports,figures}
mode=evaluate      → chargement des artefacts → nouvelle évaluation + rapports
mode=predict       → InferenceDataSchema → Predictor.predict → artifacts/reports/predictions.csv
```

### 6.4 Comparaison avec les autres stacks

Le **même cas d'usage** est implémenté avec d'autres frameworks dans les dossiers voisins.
La structure, les contrats de données et les métriques étant identiques, la comparaison est
directe (qualité, temps d'entraînement, lisibilité du code) :

- `../../with-tfidf/` — tfidf
- `../../with-langchain/` — langchain

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
cd ai-eng/question-answering/with-tfidf
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-dev.txt   # runtime + tests + lint + notebooks
```

**Option B — uv (plus rapide)**

```bash
cd ai-eng/question-answering/with-tfidf
uv venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

**Option C — installation éditable du projet**

```bash
cd ai-eng/question-answering/with-tfidf
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

Résultat : `data/raw/product_support_faq.parquet` (+ `.csv` pour la lecture humaine)
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
python -m src.main mode=train model.params.abstention_threshold=0.0

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
python scripts/predict.py predict.input=data/raw/product_support_faq.csv predict.n_samples=10
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
from src.models.model import TfidfModel
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
| `conf/model/default.yaml` | algorithme et hyperparamètres | `model.params.abstention_threshold=…` |
| `conf/train/default.yaml` | split, epochs, callbacks, noms d'artefacts | `++train.epochs=20`, `train.split.test_size=0.25` |
| `conf/preprocessing/default.yaml` | imputation, scaling, encodage, features dérivées | `preprocessing.numeric.scaler=robust`, `preprocessing.categorical.encoder=ordinal` |
| `conf/hydra/local.yaml` | répertoires de sortie Hydra (`outputs/`, `multirun/`) | `hydra.run.dir=outputs/debug` |
| `conf/config.yaml` -> bloc `retrieval` | Réglages métier de la famille, au même niveau que `mode` et `seed` : 3 clés (answer_k, abstention_threshold, rerank), chacune commentée dans le fichier — c'est là qu'on change le métier sans toucher au code. | `retrieval.answer_k=5`, `retrieval.abstention_threshold=0.0` |

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
| `data/raw/product_support_faq.parquet` (+ `csv`) | jeu de données synthétique, 120 lignes |
| `data/raw/generation_metadata.json` | recette de génération : graine, options, empreinte du jeu, fichiers écrits |
| `data/processed/split_{train,val,test}.parquet` | splits avant transformation (l'évaluation et l'inférence rejouent exactement le même découpage) |
| `data/processed/features_{X_train,X_val,X_test}.parquet` | matrices prêtes pour le modèle |
| `artifacts/models/lexical_index.joblib` | modèle entraîné |
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

Métrique principale : **`recall_at_1`** (sens `maximize`, seuil de smoke test : ≥ 0.7).
Métriques secondaires : recall_at_3, recall_at_5, mrr, ndcg_at_10, answer_exact_match, answer_f1, citation_precision, citation_recall, abstention_accuracy, abstention_balanced_accuracy, answer_coverage.

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
- **Reproductibilité** : graine propagée à Python/NumPy/scikit-learn (TF-IDF / BM25), données régénérables à l'identique.
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
| `src/models/` | `BaseModel` (ABC) + implémentation scikit-learn (TF-IDF / BM25) | Aucune logique de training loop ici : le modèle expose un contrat. |
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
| `ModuleNotFoundError: No module named 'src'` | Exécution depuis un autre répertoire | `cd ai-eng/question-answering/with-tfidf` puis `python -m src.main …` (ou `export PYTHONPATH=.`) |
| `FileNotFoundError: data/raw/...` | Données non générées | `make data` (ou `python scripts/generate_data.py`) |
| `SchemaError` Pandera au chargement | Dataset corrompu / régénéré avec un autre schéma | `make clean-artifacts && make data` |
| `Could not find a version that satisfies the requirement scikit-learn (tf-idf / bm25)` | Index PyPI inaccessible / Python trop ancien | Python ≥ 3.10, `pip install --upgrade pip`, vérifier le proxy |
| Erreur `Key 'X' not in ...` sur un override Hydra | Clé absente du schéma | Préfixer l'override avec `++` (ex. `++train.epochs=5`) |
| Les chemins pointent vers `outputs/...` | Un outil a changé le CWD | `hydra.job.chdir=false` est déjà actif ; les chemins viennent de `src/utils/paths.py` |
| Tests lents | Entraînement complet dans les tests | Les fixtures utilisent un petit dataset ; `pytest -m "not slow"` |

Pour rejouer une configuration exacte : Hydra sauvegarde la config composée dans
`outputs/<date>/<heure>/.hydra/config.yaml` — copiez-la avec `--config-path`/`--config-name`.

---

## 18. Références

- [scikit-learn : extraction de features textuelles](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
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

*Dernière génération : 2026-10-05 · projet `ai-eng-qa-tfidf`*
