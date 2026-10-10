# Extraction d'entités nommées par spaCy — le tagger appris, corroboré par des règles

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-nlp-orange)
![Hydra](https://img.shields.io/badge/config-Hydra-89b482)
![Pandera](https://img.shields.io/badge/validation-Pandera-16a34a)
![pytest](https://img.shields.io/badge/tests-pytest-2ea44f)
![Licence](https://img.shields.io/badge/license-MIT-lightgrey)

> **Domaine** : `ai-eng` · **Problématique** : `named-entity-recognition` · **Stack** : `spaCy`
> **Emplacement** : `ai-eng/named-entity-recognition/with-spacy/`
> **Temps de lecture** : ~15 min · **Temps d'exécution complet** : < 5 min sur un laptop

---

## 1. À propos / Objectifs

Cinq types d'entités (produit, référence de commande, montant, date, transporteur), 1 200 messages
de service client synthétiques annotés au caractère, écrits dans deux styles rédactionnels — 55 %
rédigés (« CMD-1234 », « 12 mars 2025 ») et 45 % abrégés (« cmd 1234 », « 12/03/2025 »). Le pipeline
spaCy est entraîné **sur le corpus** (`spacy.blank`, aucun modèle pré-entraîné téléchargé) et servi
en **hybride** : un tagger à transitions décide, une couche de règles déclarées (motifs de
référence, de montant et de date + index de surfaces appris sur le train) corrobore et comble les
trous. Cette couche de règles est la référence explicable du projet : elle est chiffrée sur le même
split de test, et l'écart avec le tagger est publié plutôt que supposé. Les surfaces de produits et
de transporteurs des splits d'évaluation sont **réservées** — jamais vues à l'entraînement — donc le
projet mesure ce qu'un modèle apprend de la *forme* d'un nom au lieu de récompenser une liste.
Chaque mention publiée porte sa provenance et une confiance dont la précision est mesurée par
niveau.

Le service client reçoit 1 200 messages par semaine par formulaire, courriel, chat et courrier.
Chaque message raconte la même histoire en mots différents : une référence de commande, un produit,
un transporteur, un montant et une date. Aujourd'hui, un conseiller relit le message pour recopier
ces cinq informations dans le dossier — un travail mécanique, fastidieux, et qui conditionne tout le
reste : sans la référence de commande correcte, aucune relance n'est possible, et un montant mal
recopié se retrouve dans la comptabilité.

**Ce que cet exemple démontre**

1. Structurer un projet named-entity-recognition prêt pour la production (OOP, typage, tests, configuration déclarative).
2. Valider les données avec des contrats exécutables **Pandera** à chaque étape critique.
3. Configurer l'intégralité du projet avec **Hydra** (aucune valeur codée en dur).
4. Rendre le résultat **reproductible** : graine fixée, artefacts horodatés, rapports générés.

**Pourquoi spaCy ici ?**

- Pipeline `spacy.blank(...)` entraîné from scratch : aucune dépendance à un modèle pré-entraîné téléchargeable.
- Thinc (le moteur ML de spaCy) fournit les couches et l'optimisation ; `Example` alimente l'entraînement.
- Trois algorithmes servis : `gazetteer` (couche de règles déclarées + surfaces apprises sur le train, aucun apprentissage de poids — le plancher explicable), `tagger` (tok2vec + classifieur à transitions, entraîné sur le corpus) et `hybrid` (le tagger décide, les règles corroborent et comblent les trous — servi par défaut).
- Un tagger à transitions n'expose pas de probabilité par mention : la confiance publiée est le **taux de recouvrement** entre le span prédit et la couche de règles, et sa précision est mesurée par niveau dans le rapport d'évaluation (`confidence_gap`). Publier une fausse probabilité serait pire que publier une mesure de corroboration documentée.
- Les frontières de mentions comptent autant que le type : la métrique principale est une F1 micro au niveau entité (type **et** bornes exactes), complétée par une F1 partielle qui mesure le coût de cette exigence.

---

## 2. Stack technologique

| Rôle | Outil | Version minimale | Pourquoi |
| --- | --- | --- | --- |
| Framework ML | **spaCy** | thinc>=8.2 | nlp |
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

Le service client reçoit 1 200 messages par semaine par formulaire, courriel, chat et courrier.
Chaque message raconte la même histoire en mots différents : une référence de commande, un produit,
un transporteur, un montant et une date. Aujourd'hui, un conseiller relit le message pour recopier
ces cinq informations dans le dossier — un travail mécanique, fastidieux, et qui conditionne tout le
reste : sans la référence de commande correcte, aucune relance n'est possible, et un montant mal
recopié se retrouve dans la comptabilité.

| | |
| --- | --- |
| **Qui** (persona / consommateur) | Responsable du back-office d'un distributeur de matériel informatique (300 personnes), avec un ingénieur IA qui outille le service client et une équipe qualité qui mesure les dossiers traités en retard. |
| **Problème** | Extraire automatiquement les entités d'un message (produit, référence de commande, montant, date, transporteur), avec leurs positions exactes dans le texte, en publiant pour chaque mention sa **provenance** (règle déclenchée ou modèle) et sa **confiance** — et en mesurant la part de mentions qui portent sur des noms jamais vus à l'entraînement, parce que c'est là que le système se trompe. |
| **Entrées** | Un message de client en texte libre, avec son canal d'arrivée, son style rédactionnel et sa date de réception. Aucune mise en forme, aucune structure : le texte tel qu'il a été écrit. |
| **Sorties** | Une table de mentions : identifiant du message, décalages de début et de fin, type, surface, provenance, confiance, et le verdict (correct ou non) quand la vérité terrain est connue. Plus la fiche de modèle, le rapport d'évaluation et les figures de lecture (F1 par type, erreurs de bornes, précision par niveau de confiance). |
| **Valeur métier** | Le conseiller relit une extraction au lieu de recopier cinq champs, et chaque erreur devient diagnosticable : le rapport dit si le système s'est trompé sur le *type* (un transporteur pris pour un produit) ou sur les *bornes* (une mention trop courte), et la ventilation par style montre que les messages abrégés sont plus difficiles que les messages rédigés — un résultat publié, pas une excuse. |
| **Cadence** | Entraînement mensuel sur les messages étiquetés ; extraction en ligne à la réception du message. |

**Objectif de modélisation** — Extraire automatiquement les entités d'un message (produit, référence de commande, montant,
date, transporteur), avec leurs positions exactes dans le texte, en publiant pour chaque
mention sa **provenance** (règle déclenchée ou modèle) et sa **confiance** — et en mesurant
la part de mentions qui portent sur des noms jamais vus à l'entraînement, parce que c'est là
que le système se trompe.

**Critères de réussite**

- [ ] F1 au niveau entité ≥ 0,70 sur le split de test : la métrique est micro (toutes les mentions comptent pareil), et le plancher trivial vaut 0,0 — un système qui n'annote rien n'a aucune précision.
- [ ] F1 macro également publiée : sur cinq types de fréquence inégale, la F1 micro peut cacher un type sacrifié, et le transporteur est le type difficile du corpus.
- [ ] Les surfaces réservées aux splits d'évaluation (produits et transporteurs jamais vus à l'entraînement) sont mesurées **à part** : la dégradation entre le train et le test est un résultat du projet, pas une surprise.
- [ ] Les distracteurs déclarés (familles de produits, villes, numéros de facture, années seules) ne sont **jamais** annotés par le système, et la couche de règles le vérifie par test.
- [ ] Chaque mention publiée porte sa provenance (`regle` ou `modele`) et une confiance dont la précision est mesurée par niveau : une confiance de 0,9 qui ne vaut que 0,6 en précision est publiée comme telle.
- [ ] Déterminisme : deux entraînements à graine fixée produisent des prédictions identiques, et la suite de tests le vérifie.
- [ ] Aucune dépendance réseau à l'exécution : pipeline `spacy.blank(...)` entraîné sur le corpus, vocabulaire appris sur le train, aucun modèle pré-entraîné téléchargé.

**Contraintes**

- Aucune donnée personnelle : le corpus est synthétique et les textes sont des gabarits paramétrés.
- Une mention sans provenance n'est pas exploitable par un conseiller : la sortie doit dire d'où elle vient.
- Les bornes comptent : une mention décalée est refusée par le back-office, donc comptée comme fausse.

---

## 4. Données

Les données sont **synthétiques**, générées localement par
[`src/data/generators.py`](src/data/generators.py) : aucune donnée réelle, aucun téléchargement,
aucune information confidentielle. Elles sont néanmoins construites pour être **crédibles
métier** (corrélations réalistes, bruit, valeurs manquantes, outliers légitimes).

| Propriété | Valeur |
| --- | --- |
| Jeu de données | `sav_messages` |
| Volume par défaut | 1 200 lignes |
| Formats écrits | parquet, csv |
| Emplacement | `data/raw/sav_messages.parquet` (et `.csv`) |
| Cible | *aucune* (apprentissage non supervisé) |
| Identifiant | `msg_id` |
| Colonne temporelle | `received_at` |
| Graine | `42` (reproductible) |

**Schéma**

| Colonne | Type | Rôle | Signification métier |
| --- | --- | --- | --- |
| `msg_id` | str | identifier | Identifiant stable du message (clé de jointure avec la table d'annotations). |
| `text` | str | feature | Texte du message tel qu'il a été écrit, avec ses surfaces rédigées ou abrégées. |
| `canal` | category | feature | Canal d'arrivée du message : formulaire, courriel, chat ou courrier. |
| `style` | category | metadata | Style rédactionnel : redige (surfaces complètes) ou abrege (surfaces courtes, ponctuation réduite). |
| `n_entities` | int | metadata | Nombre d'entités annotées dans le message (calculé à la génération, jamais prédit). |
| `n_tokens` | int | metadata | Nombre de tokens du texte, calculé avec le tokenizer du projet (pour les figures de longueur). |
| `received_at` | datetime | timestamp | Date de réception du message (six mois d'historique). |
| `split` | category | group | Découpage du corpus (train / val / calibration / test), stratifié par style au moment de la génération. |

Détail complet (distributions attendues, remarques) : [`data/README.md`](data/README.md).

---

## 5. Arborescence

```text
ai-eng/named-entity-recognition/with-spacy/
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
│   ├── preprocessing/           # outils de la modalité texte (tokénisation, découpage)
│   ├── features/                # traits de forme des mentions (signature, chiffres, casse)
│   ├── models/                  # BaseModel (ABC) + implémentation spaCy
│   ├── training/                # EntityTrainer, callbacks, métriques
│   ├── evaluation/              # EntityEvaluator + génération de rapports
│   ├── inference/               # EntityPredictor (batch + enregistrement unitaire)
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
├── tests/                       # pytest (corpus, schémas, moteur, évaluation, pipelines)
│   ├── conftest.py
│   ├── test_generator.py
│   ├── test_data_schemas.py
│   ├── test_loaders.py
│   ├── test_features.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   └── test_pipeline.py
│
└── artifacts/                   # produits générés (ignorés par git)
    ├── models/                  # modèle spaCy (répertoire) + fiche modèle
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
   ├── SyntheticEntityCorpusGenerator ← produit data/raw/*.parquet : les messages **et** les mentions
   ├── EntityCorpusLoader            ← lecture + validation Pandera des deux tables, splits, prédictions
   ├── GazetteerIndex                ← surfaces vues au train → index de règles (motif le plus long d'abord)
   │
   ├── SpacyEntityTagger(BaseModel)   ← implémentation spaCy
   │        fit / predict / model_card / save / load
   ├── EntityTrainer                 ← alignement des offsets, callbacks, métriques au niveau entité
   ├── EntityEvaluator               ← F1 entité, ventilation, surfaces réservées, familles d'erreurs
   ├── ReportBuilder                 ← Markdown + figures dans artifacts/
   └── EntityPredictor               ← inférence validée (batch + 1 enregistrement)
```

### 6.3 Flux de bout en bout

```text
mode=generate-data → SyntheticEntityCorpusGenerator → data/raw/{messages,spans}.parquet
mode=train         → EntityCorpusLoader (validation Pandera des deux tables)
                     → GazetteerIndex.from_spans (surfaces du **train** uniquement)
                     → SpacyEntityTagger.fit (offsets alignés sur les tokens, callbacks)
                     → EntityEvaluator.evaluate → ReportBuilder → artifacts/{models,metrics,reports,figures}
mode=evaluate      → chargement des artefacts → nouvelle évaluation + rapports
mode=predict       → MessagesSchema → EntityPredictor.predict → artifacts/reports/predictions.csv
```

### 6.4 Comparaison avec les autres stacks

Le **même cas d'usage** est implémenté avec d'autres frameworks dans les dossiers voisins.
La structure, les contrats de données et les métriques étant identiques, la comparaison est
directe (qualité, temps d'entraînement, lisibilité du code) :

- `../../with-spacy/` — spacy

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
cd ai-eng/named-entity-recognition/with-spacy
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-dev.txt   # runtime + tests + lint + notebooks
```

**Option B — uv (plus rapide)**

```bash
cd ai-eng/named-entity-recognition/with-spacy
uv venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

**Option C — installation éditable du projet**

```bash
cd ai-eng/named-entity-recognition/with-spacy
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

Résultat : `data/raw/sav_messages.parquet` (+ `.csv` pour la lecture humaine)
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
python -m src.main mode=train model.params.training={'dropout': 0.2, 'shuffle': True}

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
python scripts/predict.py predict.input=data/raw/sav_messages.csv predict.n_samples=10
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
from src.data.generators import SyntheticEntityCorpusGenerator
from src.data.loaders import EntityCorpusLoader
from src.models import build_model, load_model
from src.utils.paths import ProjectPaths

corpus = SyntheticEntityCorpusGenerator(n_documents=200, seed=42).generate()
loader = EntityCorpusLoader(paths=ProjectPaths.from_root())
train_documents = loader.split("train", frame=corpus.documents)
train_spans = loader.split_annotations("train", documents=corpus.documents, spans=corpus.queries)
settings = {"model": {"algorithm": "hybrid"}, "seed": 42}
model = build_model(settings)  # algorithme explicite, défauts du registre
model.fit(train_documents, train_spans)
mentions = model.predict(train_documents["text"].tolist())
print(model.summary())
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
| `conf/model/default.yaml` | algorithme et hyperparamètres | `model.params.training=…` |
| `conf/train/default.yaml` | split, epochs, callbacks, noms d'artefacts | `++train.epochs=20`, `train.split.test_size=0.25` |
| `conf/preprocessing/default.yaml` | imputation, scaling, encodage, features dérivées | `preprocessing.numeric.scaler=robust`, `preprocessing.categorical.encoder=ordinal` |
| `conf/hydra/local.yaml` | répertoires de sortie Hydra (`outputs/`, `multirun/`) | `hydra.run.dir=outputs/debug` |
| `conf/config.yaml` -> bloc `train` | Réglages métier de la famille, au même niveau que `mode` et `seed` : 1 clés (artifacts), chacune commentée dans le fichier — c'est là qu'on change le métier sans toucher au code. | `train.artifacts={'model_file': 'tagger'}` |

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
| `data/raw/sav_messages.parquet` (+ `csv`) | jeu de données synthétique, 1 200 lignes |
| `data/raw/generation_metadata.json` | recette de génération : graine, options, empreinte du jeu, fichiers écrits |
| `data/raw/spans.parquet` (+ `csv`) | mentions annotées : offsets, type, surface, et le drapeau `holdout` (surface réservée à l'évaluation) |
| `artifacts/models/tagger` | modèle entraîné |
| `artifacts/models/model_card.json` | carte du modèle (params, métriques, features, date) |
| `artifacts/models/resolved_config.json` | configuration Hydra résolue : l'artefact entraîné porte sa recette exacte |
| `artifacts/metrics/training_metrics.json` | métriques d'entraînement et de validation |
| `artifacts/metrics/evaluation_metrics.json` | métriques sur le split de test + verdict des seuils |
| `artifacts/reports/evaluation_report.md` | rapport lisible (métriques, analyse d'erreurs, recommandations) |
| `artifacts/reports/per_label.csv` | précision, rappel, F1 et support **par type d'entité**, ligne `micro` incluse |
| `artifacts/reports/segment_metrics.csv` | F1 au niveau entité par style rédactionnel et par canal |
| `artifacts/reports/confidence.csv` | mentions publiées, correctes et précision observée par niveau de confiance |
| `artifacts/reports/holdout.csv` | rappel sur les surfaces réservées, à côté des surfaces vues à l'entraînement |
| `artifacts/reports/errors.csv` | erreurs classées (inventée, bornes, manquée), avec la phrase qui les entoure |
| `artifacts/reports/predictions.csv` | prédictions sur l'échantillon de démonstration |
| `artifacts/figures/*.png` | mentions par type et par split, F1 par type, rappel sur les surfaces réservées, longueurs par style, précision par niveau de confiance |
| `outputs/<date>/<heure>/` | configuration composée + logs Hydra |

Métrique principale : **`entity_f1`** (sens `maximize`, seuil de smoke test : ≥ 0.7).
Métriques secondaires : entity_precision, entity_recall, macro_f1, partial_f1, type_accuracy, boundary_accuracy, mean_confidence, confidence_gap, predicted_entities, latency_p50_ms, latency_p95_ms.

Résultats de l'exécution de référence (`make all`, graine 42) :

| Indicateur | Valeur mesurée |
| --- | --- |
| F1 au niveau entité, micro, sur le split de test (seuil 0,7 ; plancher trivial 0,0) | **0,9340** |
| Précision / rappel au niveau entité (bornes exactes exigées) | **0,9521 / 0,9167** |
| F1 macro (les cinq types comptent pareil) / F1 partielle (bornes tolérées) | **0,9239 / 0,9735** |
| Référence « couche de règles » sur le même test (surfaces du train uniquement) | **0,8710** |
| Rappel sur les surfaces réservées aux splits d'évaluation / déjà vues au train | **0,6056 / 1,0000** |
| Confiance moyenne publiée (écart sur les mentions correctes) | **0,8423 (0,1396)** |
| Erreurs classées au test (inventée / bornes / manquée) | **5 / 20 / 0** |
| Type le plus difficile (F1 au niveau entité) / le plus facile | **transporteur 0,713 / commande, date, montant 1,000** |
| Surfaces réservées au test : produits et transporteurs jamais vus à l'entraînement | **12 / 4** |

Ces valeurs sont reproductibles à l'identique ; elles proviennent du split de test, jamais
du split d'entraînement, et le détail complet est dans `artifacts/reports/evaluation_report.md`.

---

## 13. Notebooks

Tous les notebooks sont **exécutables de bout en bout** (`make notebooks`) et documentés
cellule par cellule, comme un support de formation pour juniors.


| Notebook | Ce qu'on y apprend |
| --- | --- |
| `01_eda.ipynb` | Cartographie d'un corpus **annoté au caractère** : mentions par type et par split, longueurs, styles rédactionnels, canaux, surfaces réservées et distracteurs déclarés — et la mesure de ce qu'une **règle de forme** retrouverait déjà, type par type. |
| `02_validation.ipynb` | Les contrats Pandera des **deux** tables (messages et annotations) et leur jointure — `surface == text[start:end]`, pas de chevauchement, mention rattachée au split de son message —, puis **corruption volontaire** pour lire le message d'échec. |
| `03_preprocessing.ipynb` | Traits de forme et séparabilité des signatures, tokénisation `spacy.blank('fr')` et **alignement des offsets** sur les tokens (le taux de mentions ignorées est publié), puis index de règles appris sur le train : ce qu'il sait, et la surface réservée qu'il ne peut pas connaître. |
| `04_model_exploration.ipynb` | Les trois algorithmes de la stack — `gazetteer` (règles), `tagger` (appris), `hybrid` (servi) — comparés sur la validation, l'architecture **lue** sur le pipeline plutôt que redéclarée, et la confiance comme corroboration, avec un seuil d'automatisation arbitré sur la calibration. |
| `05_training.ipynb` | *Entraînement dans les conditions de production* — Le pipeline réel (`TrainPipeline`) exécuté dans un bac à sable, ses artefacts (répertoire spaCy, fiche modèle, métriques, configuration résolue), le **rechargement à l'identique** et la preuve de déterminisme. |
| `06_error_analysis.ipynb` | *Analyse d'erreurs et recommandations* — Verdict contractuel sur le test, ventilation par type, par style et par canal, comparatif **surface réservée / surface vue**, trois familles d'erreurs (inventée, bornes, manquée) expliquées sur des exemples, figures et recommandations reliées à un chiffre. |

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
| `tests/test_generator.py` | Le générateur synthétique : reproductibilité à graine fixée, trois à cinq entités par message, mentions jamais chevauchantes, surfaces réservées absentes du train, distracteurs présents dans les textes et **jamais** annotés. |
| `tests/test_data_schemas.py` | Les contrats Pandera des deux tables : `surface == text[start:end]`, surface décalée ou chevauchement refusés, type inconnu et identifiant hors format rejetés, confiance impossible refusée dans la table de prédictions. |
| `tests/test_loaders.py` | Chargement Parquet/CSV du corpus et des annotations, `split` / `split_annotations` alignés, distribution d'entités par split et par type, écriture des prédictions validée, refus d'une table non conforme avant écriture. |
| `tests/test_features.py` | Traits de forme dérivés de la surface (signature stable, casse ignorée), résumé par type, et **raccourci de forme** appris sur le train puis mesuré ailleurs — refusé sur son propre split d'apprentissage. |
| `tests/test_preprocessing.py` | Les utilitaires de la modalité texte livrés avec le projet (normalisation, tokénisation, découpage en passages, mots vides, vectoriseur lexical, plongement par hachage) : ils sont testés pour eux-mêmes, même si le corpus d'entités ne les traverse pas. |
| `tests/test_models.py` | Le contrat du modèle : registre d'algorithmes de la stack, contrat des mentions (bornes, type, provenance, confiance), couche de règles qui ne connaît que le train, artefact sauvegardé qui prédit à l'identique, fiche modèle et architecture réellement construite. |
| `tests/test_training.py` | La métrique au niveau entité (F1 nul pour un système muet, 1 pour une prédiction parfaite, borne décalée comptée fausse), la table par type, les trois familles d'erreurs, la confiance par niveau, la déterminisme d'une époque et le refus d'un corpus sans annotation. |
| `tests/test_evaluation.py` | L'évaluateur : métriques au niveau entité et références, couche de règles apprise **sur le train seulement**, ventilation par segment et surfaces réservées, sérialisation, verdict lu sur la métrique principale et refus de conclure sans métrique. |
| `tests/test_pipeline.py` | Bout en bout : les quatre pipelines s'enchaînent sur une configuration réduite, écrivent leurs artefacts (les deux tables, la recette du run, les mentions validées) et refusent d'évaluer sans artefact. |

Les tests utilisent des **fixtures légères** (`tests/conftest.py`) : petit dataset synthétique
et configuration réduite, donc exécution en quelques secondes.

---

## 15. Bonnes pratiques appliquées

- **OOP systématique** : générateur, loaders, preprocessing, modèle, trainer, évaluateur,
  predictor et pipelines sont des classes à responsabilité unique.
- **Classe abstraite `BaseEntityTagger`** : contrat commun `fit / predict / save / load`, ce qui permet
  de changer de framework sans toucher au reste du code (DIP).
- **Type hints partout** + `mypy` configuré ; docstrings Google sur toutes les entités publiques.
- **Hydra** pour toute la configuration, **Pydantic** pour la valider au démarrage.
- **Pandera** à trois endroits critiques : données brutes, données transformées, données d'inférence.
- **Aucune fuite de données** : le preprocessing est appris sur le train seul et persisté.
- **Parquet** pour les données intermédiaires (typé, compressé, lecture partielle), CSV pour l'humain.
- **Artefacts traçables** : model card JSON, métriques JSON, rapport Markdown, figures PNG.
- **Reproductibilité** : graine propagée à Python/NumPy/spaCy, données régénérables à l'identique.
- **Chemins robustes** : `pathlib.Path` uniquement, racine résolue depuis `src/`, jamais depuis le CWD.
- **Logs structurés** avec loguru (interception du `logging` stdlib des librairies tierces).
- **Tests unitaires concrets** (comportements, pas `assert True`) + smoke test de bout en bout.
- **Notebooks documentés** : chaque bloc de code est précédé d'une intention et suivi d'une lecture du résultat.

---

## 16. Structure du code expliquée

| Dossier | Responsabilité | Points d'attention |
| --- | --- | --- |
| `src/data/` | Génération synthétique, chargement, **contrats Pandera** | Le générateur est déterministe ; les schémas sont la documentation exécutable des données. |
| `src/preprocessing/` | Utilitaires de la modalité texte livrés avec le projet | Le corpus d'entités ne les traverse pas ; ils sont testés pour eux-mêmes. |
| `src/features/` | Traits de **forme** d'une mention (signature, chiffres, casse, séparateurs) | Le raccourci « signature ⇒ type » est appris sur le train puis mesuré sur un autre split. |
| `src/models/` | `BaseModel` (ABC) + implémentation spaCy | Aucune logique de training loop ici : le modèle expose un contrat. |
| `src/training/` | `EntityTrainer`, callbacks, métriques au niveau entité | Les effets de bord (logs, early stopping) sont des callbacks, pas du code inline. |
| `src/evaluation/` | `EntityEvaluator` + `ReportBuilder` | Les métriques sont calculées une seule fois puis sérialisées. |
| `src/inference/` | `EntityPredictor` | Valide l'entrée avec `MessagesSchema` et publie `PredictedMentionsSchema`. |
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
| `ModuleNotFoundError: No module named 'src'` | Exécution depuis un autre répertoire | `cd ai-eng/named-entity-recognition/with-spacy` puis `python -m src.main …` (ou `export PYTHONPATH=.`) |
| `FileNotFoundError: data/raw/...` | Données non générées | `make data` (ou `python scripts/generate_data.py`) |
| `SchemaError` Pandera au chargement | Dataset corrompu / régénéré avec un autre schéma | `make clean-artifacts && make data` |
| `Could not find a version that satisfies the requirement spacy` | Index PyPI inaccessible / Python trop ancien | Python ≥ 3.10, `pip install --upgrade pip`, vérifier le proxy |
| Erreur `Key 'X' not in ...` sur un override Hydra | Clé absente du schéma | Préfixer l'override avec `++` (ex. `++train.epochs=5`) |
| Les chemins pointent vers `outputs/...` | Un outil a changé le CWD | `hydra.job.chdir=false` est déjà actif ; les chemins viennent de `src/utils/paths.py` |
| Tests lents | Entraînement complet dans les tests | Les fixtures utilisent un petit dataset ; `pytest -m "not slow"` |

Pour rejouer une configuration exacte : Hydra sauvegarde la config composée dans
`outputs/<date>/<heure>/.hydra/config.yaml` — copiez-la avec `--config-path`/`--config-name`.

---

## 18. Références

- [spaCy Usage Guides](https://spacy.io/usage)
- [spaCy API](https://spacy.io/api)
- [spaCy : entraîner un composant d'entités nommées](https://spacy.io/usage/training#ner)
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

*Dernière génération : 2026-10-06 · projet `ai-eng-named-entity-recognition-spacy`*
