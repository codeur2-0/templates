# Résumé de comptes-rendus d'intervention — encodeur-décodeur appris, baseline extractive publiée

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![Encodeur-décodeur entraîné sur le corpus (résumé)](https://img.shields.io/badge/Encodeur--décodeur%20entraîné%20sur%20le%20corpus%20(résumé)-deep-learning-orange)
![Hydra](https://img.shields.io/badge/config-Hydra-89b482)
![Pandera](https://img.shields.io/badge/validation-Pandera-16a34a)
![pytest](https://img.shields.io/badge/tests-pytest-2ea44f)
![Licence](https://img.shields.io/badge/license-MIT-lightgrey)

> **Domaine** : `nlp` · **Problématique** : `summarization` · **Stack** : `Encodeur-décodeur entraîné sur le corpus (résumé)`
> **Emplacement** : `nlp/summarization/with-seq2seq/`
> **Temps de lecture** : ~15 min · **Temps d'exécution complet** : < 5 min sur un laptop

---

## 1. À propos / Objectifs

Résumer un compte-rendu d'intervention en quelques phrases **fidèles** : 600 documents synthétiques,
leurs résumés de référence écrits à partir des faits saillants, et la table des faits annotés
(équipement, symptôme, cause, action, pièce, durée, statut). Un encodeur-décodeur minuscule est
entraîné **sur le corpus** — vocabulaire WordPiece et poids appris sur le split d'entraînement,
aucun modèle pré-entraîné téléchargé —, puis servi avec un décodage glouton et un budget de longueur
déclaré. La baseline extractive et le résumé vide sont mesurés sur les **mêmes lignes de test**,
donc chaque score se lit par différence. La couverture des faits saillants et les valeurs non
supportées sont publiées à côté du ROUGE : un résumé fluide qui oublie la durée d'une intervention
est un mauvais résumé, et il est visible.

Chaque intervention donne lieu à un compte-rendu d'une dizaine de phrases : ce qui a été constaté,
pourquoi, ce qui a été fait, avec quelle pièce, en combien de temps et dans quel état le matériel
est laissé. Les responsables de site les reçoivent par dizaines chaque semaine et les lisent en
diagonale ; la synthèse est faite à la main, quand elle est faite.

**Ce que cet exemple démontre**

1. Structurer un projet summarization prêt pour la production (OOP, typage, tests, configuration déclarative).
2. Valider les données avec des contrats exécutables **Pandera** à chaque étape critique.
3. Configurer l'intégralité du projet avec **Hydra** (aucune valeur codée en dur).
4. Rendre le résultat **reproductible** : graine fixée, artefacts horodatés, rapports générés.

**Pourquoi Encodeur-décodeur entraîné sur le corpus (résumé) ici ?**

- Aucun poids pré-entraîné n'est téléchargé : le vocabulaire WordPiece partagé par l'encodeur et le décodeur, puis les poids, sont appris sur le split d'entraînement — le seul réglage possible hors ligne, et il est publié dans la fiche de modèle.
- Trois étapes dans `fit` : pré-entraînement dénoising (le résumé de référence est tronqué et bruité, le modèle apprend à le reconstruire), affinage supervisé enseignant-forcé, puis étalonnage de la longueur de sortie sur le split de validation.
- Le décodage est glouton et déterministe (`temperature` n'existe pas ici) : un résumé doit être reproductible ligne à ligne, et le rapport publie le taux de résumés qui atteignent la longueur maximale, signe d'un décodeur qui ne sait pas s'arrêter.
- Deux architectures servies : `transformer_tiny` (2 couches, 128 unités) et une baseline extractive `textrank` qui ne dépend d'aucun apprentissage et sert de plancher publié ; la comparaison est mesurée sur les mêmes lignes de test.
- La troncature, le taux de jetons `[UNK]` et le taux de résumés coupés à `max_output_length` sont publiés : trois raisons de ne pas lire le ROUGE seul.

---

## 2. Stack technologique

| Rôle | Outil | Version minimale | Pourquoi |
| --- | --- | --- | --- |
| Framework ML | **Encodeur-décodeur entraîné sur le corpus (résumé)** | tokenizers>=0.19 | deep-learning |
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

Chaque intervention donne lieu à un compte-rendu d'une dizaine de phrases : ce qui a été constaté,
pourquoi, ce qui a été fait, avec quelle pièce, en combien de temps et dans quel état le matériel
est laissé. Les responsables de site les reçoivent par dizaines chaque semaine et les lisent en
diagonale ; la synthèse est faite à la main, quand elle est faite.

| | |
| --- | --- |
| **Qui** (persona / consommateur) | Responsable d'un service de maintenance industrielle (60 techniciens), avec un ingénieur données qui outille le retour d'expérience et un responsable qualité qui relit les comptes-rendus avant de les archiver. |
| **Problème** | Produire automatiquement le résumé d'un compte-rendu — ses faits saillants, en quelques phrases fidèles — sans jamais inventer une valeur : une durée, une référence de pièce ou un statut absent du document se paie sur le terrain, alors qu'un résumé un peu moins fluide se lit très bien. |
| **Entrées** | Un compte-rendu d'intervention en texte libre, avec son type d'intervention, son urgence déclarée, son site et sa date de rédaction. Aucune mise en forme, aucun champ structuré. |
| **Sorties** | Un résumé de quelques phrases, sa longueur, sa compression, la part des faits saillants du document qu'il rapporte (globalement et par type de fait) et la liste des valeurs qu'il contient sans qu'elles soient dans le document. Plus la fiche de modèle, le rapport d'évaluation et les figures de lecture. |
| **Valeur métier** | Un responsable de site lit cinq phrases au lieu de dix, et chaque résumé est *vérifiable* : la couverture dit ce qui a été oublié, les valeurs non supportées disent ce qui a été inventé. Le projet publie aussi ce qu'une baseline extractive obtient sur les mêmes lignes, donc personne ne confond « le modèle résume » avec « le modèle recopie le début du document ». |
| **Cadence** | Entraînement mensuel sur les comptes-rendus validés ; résumé produit à la clôture de chaque intervention. |

**Objectif de modélisation** — Produire automatiquement le résumé d'un compte-rendu — ses faits saillants, en quelques
phrases fidèles — sans jamais inventer une valeur : une durée, une référence de pièce ou un
statut absent du document se paie sur le terrain, alors qu'un résumé un peu moins fluide se
lit très bien.

**Critères de réussite**

- [ ] ROUGE-1 F1 au-dessus du seuil déclaré sur le split de test, mesuré face à la meilleure des références de chaque document.
- [ ] La couverture des faits saillants est publiée à côté du ROUGE : gagner du ROUGE en perdant des faits est un résultat visible, pas un succès.
- [ ] Les valeurs non supportées sont comptées et listées : une valeur du résumé absente du document est une hallucination détectable, et elle est publiée telle quelle.
- [ ] La baseline extractive et le résumé vide sont mesurés sur les mêmes lignes de test, donc chaque score se lit par différence.
- [ ] La ventilation par type d'intervention, par urgence déclarée et par site est publiée : une moyenne qui cache un effondrement sur une catégorie n'est pas un résultat.
- [ ] Déterminisme : deux entraînements à graine fixée produisent les mêmes résumés, et la suite de tests le vérifie.
- [ ] Aucune dépendance réseau à l'exécution : vocabulaire et poids appris sur le split d'entraînement, aucun modèle pré-entraîné téléchargé.

**Contraintes**

- Aucune donnée personnelle : le corpus est synthétique, les sites et les techniciens sont anonymes.
- Une valeur inventée est plus grave qu'une phrase mal tournée : la fidélité est mesurée, publiée, et prioritaire dans les arbitrages.
- Le résumé tient dans un budget de longueur déclaré : une synthèse qui atteint sa borne signale un décodeur qui ne sait pas conclure.

---

## 4. Données

Les données sont **synthétiques**, générées localement par
[`src/data/generators.py`](src/data/generators.py) : aucune donnée réelle, aucun téléchargement,
aucune information confidentielle. Elles sont néanmoins construites pour être **crédibles
métier** (corrélations réalistes, bruit, valeurs manquantes, outliers légitimes).

| Propriété | Valeur |
| --- | --- |
| Jeu de données | `intervention_reports` |
| Volume par défaut | 600 lignes |
| Formats écrits | parquet, csv |
| Emplacement | `data/raw/intervention_reports.parquet` (et `.csv`) |
| Cible | *aucune* (apprentissage non supervisé) |
| Identifiant | `doc_id` |
| Colonne temporelle | `published_at` |
| Graine | `42` (reproductible) |

**Schéma**

| Colonne | Type | Rôle | Signification métier |
| --- | --- | --- | --- |
| `doc_id` | str | identifier | Identifiant stable du compte-rendu (clé de jointure avec les résumés et les faits). |
| `text` | str | feature | Compte-rendu complet : contexte, symptôme, cause, actions, pièce, durée et conclusion. |
| `intervention_type` | category | feature | Nature de l'intervention : dépannage, maintenance, installation ou expertise. |
| `urgency` | category | metadata | Urgence déclarée à l'ouverture : un distracteur pour le modèle, un segment d'analyse pour le rapport. |
| `site` | category | metadata | Site d'intervention (anonymisé) : six sites, un segment d'analyse. |
| `n_sentences` | int | metadata | Nombre de phrases du document (calculé à la génération, jamais prédit). |
| `n_tokens` | int | metadata | Nombre de tokens du document : c'est le dénominateur de la compression. |
| `published_at` | datetime | timestamp | Date de rédaction du compte-rendu. |
| `split` | category | group | Découpage du corpus (train / val / calibration / test), écrit dans le corpus au moment de la génération. |

Détail complet (distributions attendues, remarques) : [`data/README.md`](data/README.md).

---

## 5. Arborescence

```text
nlp/summarization/with-seq2seq/
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
│   ├── models/                  # BaseModel (ABC) + implémentation Encodeur-décodeur entraîné sur le corpus (résumé)
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
   ├── EncoderDecoderSummarizer(BaseModel)   ← implémentation Encodeur-décodeur entraîné sur le corpus (résumé)
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
                     → EncoderDecoderSummarizer.fit (callbacks)
                     → Evaluator.evaluate → ReportBuilder → artifacts/{models,metrics,reports,figures}
mode=evaluate      → chargement des artefacts → nouvelle évaluation + rapports
mode=predict       → InferenceDataSchema → Predictor.predict → artifacts/reports/predictions.csv
```

### 6.4 Comparaison avec les autres stacks

Le **même cas d'usage** est implémenté avec d'autres frameworks dans les dossiers voisins.
La structure, les contrats de données et les métriques étant identiques, la comparaison est
directe (qualité, temps d'entraînement, lisibilité du code) :

- `../../with-seq2seq/` — seq2seq

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
cd nlp/summarization/with-seq2seq
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-dev.txt   # runtime + tests + lint + notebooks
```

**Option B — uv (plus rapide)**

```bash
cd nlp/summarization/with-seq2seq
uv venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

**Option C — installation éditable du projet**

```bash
cd nlp/summarization/with-seq2seq
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

Résultat : `data/raw/intervention_reports.parquet` (+ `.csv` pour la lecture humaine)
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
python -m src.main mode=train model.params.layers=2

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
python scripts/predict.py predict.input=data/raw/intervention_reports.csv predict.n_samples=10
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
from src.models.model import EncoderDecoderSummarizer
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
| `conf/model/default.yaml` | algorithme et hyperparamètres | `model.params.layers=…` |
| `conf/train/default.yaml` | split, epochs, callbacks, noms d'artefacts | `++train.epochs=20`, `train.split.test_size=0.25` |
| `conf/preprocessing/default.yaml` | imputation, scaling, encodage, features dérivées | `preprocessing.numeric.scaler=robust`, `preprocessing.categorical.encoder=ordinal` |
| `conf/hydra/local.yaml` | répertoires de sortie Hydra (`outputs/`, `multirun/`) | `hydra.run.dir=outputs/debug` |
| `conf/config.yaml` -> bloc `train` | Réglages métier de la famille, au même niveau que `mode` et `seed` : 1 clés (artifacts), chacune commentée dans le fichier — c'est là qu'on change le métier sans toucher au code. | `train.artifacts={'model_file': 'summarizer.pt'}` |

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
| `data/raw/intervention_reports.parquet` (+ `csv`) | jeu de données synthétique, 600 lignes |
| `data/raw/generation_metadata.json` | recette de génération : graine, options, empreinte du jeu, fichiers écrits |
| `data/processed/chunks.parquet` (+ `csv`) | passages indexés : le texte découpé, ses métadonnées et les annotations conservées |
| `artifacts/models/summarizer.pt` | modèle entraîné |
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

Métrique principale : **`rouge1_f`** (sens `maximize`, seuil de smoke test : ≥ 0.45).
Métriques secondaires : rouge2_f, rouge_l_f, fact_coverage, fact_precision, unsupported_share, compression_mean, hit_max_length_share, n_words_mean, latency_p50_ms, latency_p95_ms.

Résultats de l'exécution de référence (`make all`, graine 42) :

| Indicateur | Valeur mesurée |
| --- | --- |
| Documents / phrases par document / tokens par document | **600 / 11,2 / 130,9** |
| Faits annotés / saillants / par document | **5 036 / 3 573 / 8,4 et 6,0** |
| Taux de réécriture des résumés de référence / registres | **≈ 50 % / deux formulations par type de fait** |
| Part des résumés annotés couverts par la référence elle-même | **1,0000** |
| Référence « résumé vide » (plancher défini, publié pour être lu) | **0,0000** |

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
- **Reproductibilité** : graine propagée à Python/NumPy/Encodeur-décodeur entraîné sur le corpus (résumé), données régénérables à l'identique.
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
| `src/models/` | `BaseModel` (ABC) + implémentation Encodeur-décodeur entraîné sur le corpus (résumé) | Aucune logique de training loop ici : le modèle expose un contrat. |
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
| `ModuleNotFoundError: No module named 'src'` | Exécution depuis un autre répertoire | `cd nlp/summarization/with-seq2seq` puis `python -m src.main …` (ou `export PYTHONPATH=.`) |
| `FileNotFoundError: data/raw/...` | Données non générées | `make data` (ou `python scripts/generate_data.py`) |
| `SchemaError` Pandera au chargement | Dataset corrompu / régénéré avec un autre schéma | `make clean-artifacts && make data` |
| `Could not find a version that satisfies the requirement encodeur-décodeur entraîné sur le corpus (résumé)` | Index PyPI inaccessible / Python trop ancien | Python ≥ 3.10, `pip install --upgrade pip`, vérifier le proxy |
| Erreur `Key 'X' not in ...` sur un override Hydra | Clé absente du schéma | Préfixer l'override avec `++` (ex. `++train.epochs=5`) |
| Les chemins pointent vers `outputs/...` | Un outil a changé le CWD | `hydra.job.chdir=false` est déjà actif ; les chemins viennent de `src/utils/paths.py` |
| Tests lents | Entraînement complet dans les tests | Les fixtures utilisent un petit dataset ; `pytest -m "not slow"` |

Pour rejouer une configuration exacte : Hydra sauvegarde la config composée dans
`outputs/<date>/<heure>/.hydra/config.yaml` — copiez-la avec `--config-path`/`--config-name`.

---

## 18. Références

- [Vaswani et al. (2017), Attention is all you need](https://arxiv.org/abs/1706.03762)
- [Lewis et al. (2019), BART : dénoising seq2seq pre-training](https://arxiv.org/abs/1910.13461)
- [Lin (2004), ROUGE : a package for automatic evaluation of summaries](https://aclanthology.org/W04-1013/)
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

*Dernière génération : 2026-10-06 · projet `nlp-summarization-seq2seq`*
