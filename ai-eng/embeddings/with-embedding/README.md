# Index vectoriel dense d'un catalogue dupliqué (hachage + SVD) — recherche et dédoublonnage

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![Index vectoriel dense (hachage + SVD)](https://img.shields.io/badge/Index%20vectoriel%20dense%20(hachage%20+%20SVD)-ai-engineering-orange)
![Hydra](https://img.shields.io/badge/config-Hydra-89b482)
![Pandera](https://img.shields.io/badge/validation-Pandera-16a34a)
![pytest](https://img.shields.io/badge/tests-pytest-2ea44f)
![Licence](https://img.shields.io/badge/license-MIT-lightgrey)

> **Domaine** : `ai-eng` · **Problématique** : `embeddings` · **Stack** : `Index vectoriel dense (hachage + SVD)`
> **Emplacement** : `ai-eng/embeddings/with-embedding/`
> **Temps de lecture** : ~15 min · **Temps d'exécution complet** : < 5 min sur un laptop

---

## 1. À propos / Objectifs

Construction et exploitation d'un index d'embeddings sur un catalogue produit publié trois fois
(notice constructeur, fiche commerciale, note SAV) : hachage de n-grammes (uni+bigrammes) puis SVD
tronquée apprise sur le train, normalisation L2, matrice de vecteurs persistée avec ses métadonnées
et sa configuration. Le projet publie ce qu'un index a de particulier — dimension réellement
produite, fidélité de la projection, taille de l'artefact, débit de requêtes, taux de succès du
cache — et sert deux cas d'usage avec le même artefact : la recherche par question (rappel mesuré
par segment, abstention sur les questions hors corpus) et la détection de quasi-doublons
(`nearest_neighbours`), qui retrouve les deux autres fiches d'un même fait pour les 168 fiches du
corpus. Aucun poids pré-entraîné, aucun téléchargement, aucun appel réseau : l'index est appris sur
le corpus, à graine fixée.

Le catalogue est publié trois fois : la notice constructeur, la fiche commerciale et la note SAV
reprennent la même autonomie et la même durée de garantie, avec un cadrage et un vocabulaire
d'accompagnement différents. Le service client cherche dans les trois, l'équipe catalogue veut
retrouver les fiches qui écrivent le même fait, et la plateforme veut un index réutilisable —
dimension, taille, débit et fidélité connus — plutôt qu'un script de recherche par cas d'usage.

**Ce que cet exemple démontre**

1. Construire un index vectoriel déterministe et hors ligne : hachage de n-grammes, SVD tronquée apprise sur le train, normalisation L2.
2. Mesurer les propriétés d'un index au lieu de les supposer : dimension réellement produite, fidélité de la projection, taille de l'artefact.
3. Chiffrer ce que la compression apporte et ce qu'elle coûte : 128 dimensions contre 4096, 0,16 Mo de matrice contre 5,25 Mo, fidélité de projection de 1,00.
4. Servir deux cas d'usage avec un seul artefact : recherche par requête et détection de quasi-doublons (`nearest_neighbours`).
5. Instrumenter le service : cache de vecteurs de requêtes, taux de succès, latence p50/p95.
6. Comparer un index dense à sa référence lexicale sur les mêmes questions, et publier le résultat même quand il contredit l'intuition : ici le dense gagne le classement, le lexical la précision des citations.
7. Écrire un générateur dont la redondance est annotée : trois fiches disent le même fait, les trois sont pertinentes, le premier rang devient la difficulté et le dédoublonnage devient mesurable sans code supplémentaire.
8. Calibrer une décision d'abstention sur un score borné, et publier l'exactitude équilibrée avec la couverture.
9. Vérifier la stabilité : deux exécutions à graine fixée produisent la même matrice, donc les mêmes métriques.

**Pourquoi Index vectoriel dense (hachage + SVD) ici ?**

- Embeddings **déterministes et hors ligne** : n-grammes hachés (uni+bigrammes) puis SVD tronquée apprise sur le train — aucun poids pré-entraîné, aucun téléchargement.
- Conçu pour mesurer ce qu'un index d'embeddings a de particulier : dimension produite, fidélité de la projection, taille de l'index, débit de requêtes et taux de succès du cache.
- Sert la recherche (cosinus remappé sur `[0, 1]`) et la détection de quasi-doublons (`nearest_neighbours`), que l'index lexical ne rend pas.

---

## 2. Stack technologique

| Rôle | Outil | Version minimale | Pourquoi |
| --- | --- | --- | --- |
| Framework ML | **Index vectoriel dense (hachage + SVD)** | joblib>=1.3 | ai-engineering |
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

Le catalogue est publié trois fois : la notice constructeur, la fiche commerciale et la note SAV
reprennent la même autonomie et la même durée de garantie, avec un cadrage et un vocabulaire
d'accompagnement différents. Le service client cherche dans les trois, l'équipe catalogue veut
retrouver les fiches qui écrivent le même fait, et la plateforme veut un index réutilisable —
dimension, taille, débit et fidélité connus — plutôt qu'un script de recherche par cas d'usage.

| | |
| --- | --- |
| **Qui** (persona / consommateur) | Équipe plateforme données d'une enseigne de matériel électronique (400 personnes), avec un responsable catalogue qui publie le référentiel produit et un ingénieur IA qui outille le service client. |
| **Problème** | Produire des **vecteurs de documents** déterministes, hors ligne et mesurables, les servir comme index de recherche (recall@k), comme détecteur de quasi-doublons (les trois fiches d'un même fait) et comme artefact versionné dont on connaît la dimension, la fidélité, la taille et le débit — au lieu d'un index dont on ne sait rien sinon qu'il « marche ». |
| **Entrées** | Un corpus de fiches produit versionnées (titre, section, espace documentaire d'origine, date, texte) et des questions annotées avec leurs fiches pertinentes et leurs questions hors corpus. |
| **Sorties** | Un index vectoriel persisté (matrice de vecteurs, dimension, propriétés), les passages les plus proches d'une requête ou d'un document, une fiche de modèle qui publie la dimension, la fidélité de la projection et les statistiques de cache, et un rapport de mesure par segment. |
| **Valeur métier** | Un seul artefact pour trois usages (recherche, dédoublonnage, routage), des chiffres opposables à la baseline lexicale — mesurés sur les mêmes questions, y compris quand ils déplaisent — et une réponse chiffrée à la question « 128 dimensions suffisent-elles ? » : sur ce corpus la compression divise la taille de la matrice par trente-deux (0,16 Mo contre 5,25 Mo) et la fidélité de projection mesure ce qu'elle conserve (1,0000 : l'ordre des similarités est inchangé sur les paires testées). |
| **Cadence** | Index reconstruit à chaque publication du catalogue ; recherche en ligne, dédoublonnage en lot nocturne. |

**Objectif de modélisation** — Produire des **vecteurs de documents** déterministes, hors ligne et mesurables, les servir
comme index de recherche (recall@k), comme détecteur de quasi-doublons (les trois fiches
d'un même fait) et comme artefact versionné dont on connaît la dimension, la fidélité, la
taille et le débit — au lieu d'un index dont on ne sait rien sinon qu'il « marche ».

**Critères de réussite**

- [ ] recall@5 ≥ 0,90 sur les questions répondables du split de test : les fiches annotées du fait demandé sont retrouvées (mesuré : 0,94 pour les deux variantes ; les planchers triviaux de la même métrique valent 0,05 pour le tirage aléatoire et 0,00 pour l'ordre du corpus).
- [ ] Précision au premier rang ≥ 0,70 : le premier résultat est l'une des trois fiches du fait demandé, malgré deux quasi-doublons par fait (mesuré : 0,78 en dense, 0,73 en TF-IDF cosinus).
- [ ] Exact match ≥ 0,60 : la phrase produite est celle du corpus, mot pour mot (mesuré : 0,79 en dense, 0,76 en TF-IDF cosinus ; 1,00 sur les questions factuelles et 0,60 sur les paraphrases, pour les deux variantes).
- [ ] Précision des citations ≥ 0,90 : une réponse n'annexe pas la phrase d'une fiche non pertinente (mesuré : 0,94 en dense, 0,97 en TF-IDF cosinus — le lexical cite plus juste, le dense classe mieux, et les deux chiffres sont publiés).
- [ ] Le taux de réponse sur les questions hors corpus est **publié** à côté de la couverture, jamais caché derrière le rappel (mesuré : 4 des 7 questions du test refusées en dense, 3 en TF-IDF cosinus, pour une couverture de 0,94 et 0,88 — ces questions sont écrites avec le vocabulaire des fiches, c'est la limite mesurée de la famille).
- [ ] Fidélité de la projection ≥ 0,95 : la compression conserve l'ordre des similarités du corpus (mesuré : 1,0000 à 128 dimensions, pour une matrice de 0,16 Mo contre 5,25 Mo en 4096 dimensions, soit un index trente-deux fois plus petit).
- [ ] Dimension publiée : la fiche de modèle annonce la dimension **réellement** stockée, pas la dimension demandée — la SVD réduit ses composantes sur un corpus plus petit que la dimension visée.
- [ ] Dédoublonnage : les deux plus proches voisins d'une fiche sont ses deux autres écritures, vérifié par la suite de tests sur **toutes** les fiches du corpus de test, pas sur un échantillon (336 voisins attendus, deux par fiche).
- [ ] Débit : index de 168 fiches construit en moins de 30 s et requête servie en moins de 250 ms sur CPU (mesuré : 0,42 s de construction en dense, 0,07 s en lexical ; latence p95 de 4,64 ms en dense et 7,44 ms en lexical, cache armé).
- [ ] Déterminisme : deux exécutions à graine fixée produisent la même matrice et les mêmes métriques.
- [ ] Stabilité de la décision : l'exactitude équilibrée réponse/abstention est publiée avec la couverture, jamais l'une sans l'autre (mesuré : 0,76 d'exactitude équilibrée pour 0,94 de couverture en dense, 0,66 pour 0,88 en lexical).

**Contraintes**

- Aucun poids pré-entraîné, aucun téléchargement : l'index est appris sur le corpus, hors ligne.
- Le corpus est synthétique : aucune donnée client, aucune donnée personnelle.
- L'index doit rester rechargeable à l'identique : vecteurs, métadonnées et configuration du générateur sont persistés ensemble.

---

## 4. Données

Les données sont **synthétiques**, générées localement par
[`src/data/generators.py`](src/data/generators.py) : aucune donnée réelle, aucun téléchargement,
aucune information confidentielle. Elles sont néanmoins construites pour être **crédibles
métier** (corrélations réalistes, bruit, valeurs manquantes, outliers légitimes).

| Propriété | Valeur |
| --- | --- |
| Jeu de données | `equipment_catalogue_editions` |
| Volume par défaut | 168 lignes |
| Formats écrits | parquet, csv |
| Emplacement | `data/raw/equipment_catalogue_editions.parquet` (et `.csv`) |
| Cible | *aucune* (apprentissage non supervisé) |
| Identifiant | `doc_id` |
| Colonne temporelle | `published_at` |
| Graine | `42` (reproductible) |

**Schéma**

| Colonne | Type | Rôle | Signification métier |
| --- | --- | --- | --- |
| `doc_id` | str | identifier | Identifiant stable de la fiche (clé de jointure, jamais une feature). |
| `title` | str | metadata | Titre de la fiche : style éditorial, fait et référence concernée. |
| `section` | category | feature | Section éditoriale : procedure (notice), definition (fiche commerciale), contact (note SAV). |
| `source` | category | feature | Espace documentaire d'origine : wiki_it, wiki_ops, wiki_support. |
| `published_at` | datetime | timestamp | Date de publication de la fiche (les trois exemplaires d'un fait ne sont pas datés pareil). |
| `n_tokens` | int | metadata | Nombre de tokens du texte, calculé avec le tokenizer du projet. |
| `text` | str | feature | Texte intégral de la fiche : la phrase du fait y est recopiée mot pour mot. |

Détail complet (distributions attendues, remarques) : [`data/README.md`](data/README.md).

---

## 5. Arborescence

```text
ai-eng/embeddings/with-embedding/
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
│   ├── models/                  # BaseModel (ABC) + implémentation Index vectoriel dense (hachage + SVD)
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
   ├── EmbeddingModel(BaseModel)   ← implémentation Index vectoriel dense (hachage + SVD)
   │        fit / predict / save / load
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
                     → EmbeddingModel.fit (callbacks)
                     → Evaluator.evaluate → ReportBuilder → artifacts/{models,metrics,reports,figures}
mode=evaluate      → chargement des artefacts → nouvelle évaluation + rapports
mode=predict       → InferenceDataSchema → Predictor.predict → artifacts/reports/predictions.csv
```

### 6.4 Comparaison avec les autres stacks

Le **même cas d'usage** est implémenté avec d'autres frameworks dans les dossiers voisins.
La structure, les contrats de données et les métriques étant identiques, la comparaison est
directe (qualité, temps d'entraînement, lisibilité du code) :

- `../../with-tfidf/` — tfidf
- `../../with-embedding/` — embedding

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
cd ai-eng/embeddings/with-embedding
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-dev.txt   # runtime + tests + lint + notebooks
```

**Option B — uv (plus rapide)**

```bash
cd ai-eng/embeddings/with-embedding
uv venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

**Option C — installation éditable du projet**

```bash
cd ai-eng/embeddings/with-embedding
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

Résultat : `data/raw/equipment_catalogue_editions.parquet` (+ `.csv` pour la lecture humaine)
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
python scripts/predict.py predict.input=data/raw/equipment_catalogue_editions.csv predict.n_samples=10
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
from src.models.model import EmbeddingModel
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
| `data/raw/equipment_catalogue_editions.parquet` (+ `csv`) | jeu de données synthétique, 168 lignes |
| `data/raw/generation_metadata.json` | recette de génération : graine, options, empreinte du jeu, fichiers écrits |
| `data/processed/chunks.parquet` (+ `csv`) | passages indexés : le texte découpé, ses métadonnées et les annotations conservées |
| `artifacts/models/vector_index.joblib` | modèle entraîné |
| `artifacts/models/preprocessing.joblib` | pipeline de preprocessing ajusté (aucune fuite) |
| `artifacts/models/model_card.json` | carte du modèle (params, métriques, features, date) |
| `artifacts/models/resolved_config.json` | configuration Hydra résolue : l'artefact entraîné porte sa recette exacte |
| `artifacts/metrics/training_metrics.json` | métriques d'entraînement et de validation |
| `artifacts/metrics/evaluation_metrics.json` | métriques sur le split de test + verdict des seuils |
| `artifacts/reports/evaluation_report.md` | rapport lisible (métriques, analyse d'erreurs, recommandations) |
| `artifacts/reports/per_question.csv` | métriques par question : rappel@k, MRR, citations, abstention, latence |
| `artifacts/reports/retrieved_passages.csv` | passages servis à chaque question, avec leur rang et leur score |
| `artifacts/reports/segment_metrics.csv` | ventilation par difficulté, intention et type de réponse |
| `artifacts/reports/predictions.csv` | prédictions sur l'échantillon de démonstration |
| `artifacts/figures/*.png` | figures spécifiques à la tâche |
| `outputs/<date>/<heure>/` | configuration composée + logs Hydra |

Métrique principale : **`recall_at_5`** (sens `maximize`, seuil de smoke test : ≥ 0.9).
Métriques secondaires : recall_at_1, recall_at_3, recall_at_10, precision_at_1, mrr, ndcg_at_10, answer_f1, answer_exact_match, citation_precision, abstention_balanced_accuracy, answer_coverage, false_answer_rate.

Résultats de l'exécution de référence (`make all`, graine 42) :

| Indicateur | Valeur mesurée |
| --- | --- |
| recall@5 (test, 41 questions dont 34 répondables) | **0,9412** |
| Précision au premier rang (le premier résultat est l'une des trois fiches du fait) | **0,7805** |
| MRR / nDCG@10 | **0,9608 / 0,9682** |
| Exact match de la réponse (1,00 sur les questions factuelles, 0,60 sur les paraphrases) | **0,7941** |
| F1 de la réponse | **0,8802** |
| Précision des citations | **0,9375** |
| Exactitude équilibrée réponse / abstention (couverture 0,94) | **0,7563** |
| Questions hors corpus refusées (7 questions du test) | **4** |
| Fidélité de la projection (128 dimensions, matrice de 0,16 Mo contre 5,25 Mo en 4096) | **1,0000** |
| Construction de l'index / latence p95 d'une requête (cache armé) | **0,42 s / 4,64 ms** |

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
- **Classe abstraite `BaseModel`** : contrat commun `fit / predict / save / load`, ce qui permet
  de changer de framework sans toucher au reste du code (DIP).
- **Type hints partout** + `mypy` configuré ; docstrings Google sur toutes les entités publiques.
- **Hydra** pour toute la configuration, **Pydantic** pour la valider au démarrage.
- **Pandera** à trois endroits critiques : données brutes, données transformées, données d'inférence.
- **Aucune fuite de données** : le preprocessing est appris sur le train seul et persisté.
- **Parquet** pour les données intermédiaires (typé, compressé, lecture partielle), CSV pour l'humain.
- **Artefacts traçables** : model card JSON, métriques JSON, rapport Markdown, figures PNG.
- **Reproductibilité** : graine propagée à Python/NumPy/Index vectoriel dense (hachage + SVD), données régénérables à l'identique.
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
| `src/models/` | `BaseModel` (ABC) + implémentation Index vectoriel dense (hachage + SVD) | Aucune logique de training loop ici : le modèle expose un contrat. |
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
| `ModuleNotFoundError: No module named 'src'` | Exécution depuis un autre répertoire | `cd ai-eng/embeddings/with-embedding` puis `python -m src.main …` (ou `export PYTHONPATH=.`) |
| `FileNotFoundError: data/raw/...` | Données non générées | `make data` (ou `python scripts/generate_data.py`) |
| `SchemaError` Pandera au chargement | Dataset corrompu / régénéré avec un autre schéma | `make clean-artifacts && make data` |
| `Could not find a version that satisfies the requirement index vectoriel dense (hachage + svd)` | Index PyPI inaccessible / Python trop ancien | Python ≥ 3.10, `pip install --upgrade pip`, vérifier le proxy |
| Erreur `Key 'X' not in ...` sur un override Hydra | Clé absente du schéma | Préfixer l'override avec `++` (ex. `++train.epochs=5`) |
| Les chemins pointent vers `outputs/...` | Un outil a changé le CWD | `hydra.job.chdir=false` est déjà actif ; les chemins viennent de `src/utils/paths.py` |
| Tests lents | Entraînement complet dans les tests | Les fixtures utilisent un petit dataset ; `pytest -m "not slow"` |

Pour rejouer une configuration exacte : Hydra sauvegarde la config composée dans
`outputs/<date>/<heure>/.hydra/config.yaml` — copiez-la avec `--config-path`/`--config-name`.

---

## 18. Références

- [scikit-learn : TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html)
- [scikit-learn : HashingVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html)
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

*Dernière génération : 2026-10-05 · projet `ai-eng-embeddings-embedding`*
