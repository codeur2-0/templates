# templates — projets data de référence, exécutable de bout en bout

Dépôt de **mini-projets de référence** : chacun est exécutable en cinq commandes, entièrement
configuré par Hydra, validé par des contrats Pandera, testé par pytest, documenté en français et
accompagné de six notebooks pédagogiques. Le même cas d'usage est décliné sur plusieurs stacks
technologiques afin de rendre les choix d'implémentation **comparables à données et métriques
constantes**.

> Tout ce qui se trouve ici s'exécute. Il n'y a ni `TODO`, ni pseudo-code, ni exemple décoratif :
> chaque projet passe par `tools/verify.py` (lint, formatage, typage, tests, notebooks exécutés,
> pipeline complet) avant d'être considéré comme livré.

---

## 1. Ce qui est livré aujourd'hui

`data-science/classification` — **prédiction d'attrition client (churn télécom)**, six stacks :

| Projet | Stack | Modèle | ROC AUC (test) | PR AUC | Accuracy | F1 | Log loss | Durée `fit` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`with-sklearn`](data-science/classification/with-sklearn) | scikit-learn | `random_forest` | 0,8654 | 0,7240 | 0,8175 | 0,6636 | 0,4190 | 0,73 s |
| [`with-xgboost`](data-science/classification/with-xgboost) | XGBoost | `xgboost` | 0,8586 | 0,7140 | 0,7963 | 0,6418 | 0,4242 | 0,09 s |
| [`with-lightgbm`](data-science/classification/with-lightgbm) | LightGBM | `lightgbm` | 0,8574 | 0,7167 | 0,7887 | 0,6318 | 0,4259 | 0,10 s |
| [`with-pytorch`](data-science/classification/with-pytorch) | PyTorch | `mlp` (4 161 paramètres) | **0,8726** | **0,7352** | 0,7925 | 0,6570 | 0,4416 | 0,60 s |
| [`with-keras`](data-science/classification/with-keras) | Keras (API fonctionnelle) | `mlp` (4 161 paramètres) | 0,8675 | 0,7306 | 0,8137 | **0,6740** | 0,4216 | 5,11 s |
| [`with-tensorflow`](data-science/classification/with-tensorflow) | TensorFlow (`GradientTape`) | `mlp` (4 161 paramètres) | 0,8604 | 0,7230 | 0,7975 | 0,6463 | 0,4340 | 3,11 s |

Chiffres mesurés sur le même jeu synthétique (4 000 clients, 15 colonnes, 25,5 % de churn),
mêmes graines, mêmes 31 features après pré-traitement, mêmes définitions de métriques, split
65/15/20 stratifié. Les réseaux sont arrêtés par early stopping (12 à 24 époques sur un budget de
40) et restaurés sur leurs meilleurs poids.

**Lecture honnête de ce tableau** : sur des données tabulaires de cette taille, les trois stacks
classiques tiennent la dragée haute aux réseaux de neurones, pour un coût d'entraînement bien
inférieur. L'écart d'AUC entre la meilleure et la moins bonne stack est de 0,015 — c'est-à-dire du
bruit au regard de la variance d'un split. Le choix d'une stack se justifie donc ici par
l'**écosystème** (serving, GPU, compétences de l'équipe) et par l'**apprentissage**, pas par la
performance brute. C'est exactement la conclusion que le dépôt veut rendre vérifiable plutôt que
de l'asséner.

Chaque projet expose aussi : les six notebooks exécutés par la CI locale, les artefacts
(`artifacts/models`, `artifacts/metrics`, `artifacts/reports`, `artifacts/figures`), une fiche
modèle JSON (`model_card.json` : métriques, hyperparamètres, versions des librairies, empreinte
des données) et un rapport d'évaluation Markdown avec analyse d'erreurs.

---

## 2. Démarrage rapide (cinq commandes)

```bash
git clone https://github.com/codeur2-0/templates.git && cd templates/data-science/classification/with-sklearn
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
make data          # génère le jeu synthétique + l'exporte en Parquet/CSV
make train         # entraîne, évalue, écrit les artefacts et la fiche modèle
make quality       # ruff + mypy + pytest (le projet doit rester vert)
```

Variantes utiles :

```bash
make all           # data -> train -> evaluate -> predict -> quality
make notebooks     # ouvre Jupyter sur les six notebooks
python -m src.main mode=all ++train.epochs=3      # surcharge Hydra à la volée
python -m src.main mode=train model.algorithm=linear   # changer de modèle sans toucher au code
```

Pour les stacks deep learning, `pip install -r requirements.txt` installe `torch` ou
`tensorflow-cpu` : prévoyez 1 à 2 Go. Rien ne télécharge de jeu de données ni de poids pré-entraînés
— tout est synthétique et hors ligne.

---

## 3. Anatomie d'un projet

Tous les projets partagent la même arborescence, générée par `tools/scaffold` :

```
with-<stack>/
├── README.md                     # 19 sections : cas métier, arborescence, commandes, limites…
├── pyproject.toml                # ruff + mypy + pytest configurés (mêmes règles pour tous)
├── requirements.txt              # dépendances bornées, issues du registre de la stack
├── Makefile                      # install/data/train/evaluate/predict/quality/notebooks/clean
├── conf/                         # Hydra : config.yaml + groupes data/model/train/preprocessing
│   ├── config.yaml               #   assemblage des groupes, seed global, chemins
│   ├── data/default.yaml         #   générateur, splits, colonnes et leurs rôles
│   ├── model/default.yaml        #   algorithme, params, validation croisée, seuil de décision
│   ├── train/default.yaml        #   époques, batch, taux d'apprentissage, early stopping, artefacts
│   ├── preprocessing/default.yaml#   imputation, encodage, scaling, features dérivées
│   └── hydra/local.yaml          #   comportement d'Hydra (répertoire de run, journalisation)
├── data/{raw,processed,external} # + README expliquant ce qui est versionné ou non
├── notebooks/                    # 01 EDA · 02 validation Pandera · 03 pré-traitement
│                                 # 04 exploration de modèles · 05 entraînement · 06 analyse d'erreurs
├── scripts/                      # generate_data.py · train.py · evaluate.py · predict.py
├── src/
│   ├── data/                     # schémas Pandera, loaders, split, générateurs synthétiques
│   ├── preprocessing/            # pipelines appris sur le train uniquement (aucune fuite)
│   ├── features/                 # feature engineering déclaré par configuration
│   ├── models/                   # base.py (contrat BaseModel ABC) · model.py (stack) · factory.py
│   ├── training/                 # trainer.py · callbacks.py · losses_metrics.py
│   ├── evaluation/               # métriques, rapport Markdown, analyse d'erreurs
│   ├── inference/                # chargement du modèle, prédiction par lot, schéma de sortie
│   ├── pipelines/                # orchestration par mode (generate-data/train/evaluate/predict)
│   ├── schemas/                  # modèles pydantic de toute la configuration (AppConfig)
│   ├── utils/                    # io, logging (loguru), chemins, graines, minuterie
│   ├── visualization/            # figures (matplotlib/seaborn) écrites dans artifacts/figures
│   └── main.py                   # point d'entrée Hydra
├── tests/                        # conftest.py + pytest : contrats, pipelines, modèles, données
└── artifacts/{models,metrics,reports,figures}
```

Règles non négociables, appliquées partout :

- **OOP** : tout modèle implémente `BaseModel` (ABC) avec `fit` / `predict` / `save` / `load`,
  `model_card`, `check_is_fitted`, `align_features` — le reste du code ne connaît que ce contrat.
- **Hydra pour toute la configuration** : aucun hyperparamètre, chemin ou seuil codé en dur ;
  chaque valeur est surchargeable en CLI (`++train.epochs=3`, `model.algorithm=linear`).
- **Pandera** : contrats `DataFrameModel` sur les données brutes, traitées et d'inférence ; une
  colonne inattendue ou une valeur hors domaine échoue bruyamment, jamais silencieusement.
- **pydantic** : `AppConfig` valide la configuration au démarrage (types, bornes, littéraux).
- **Parquet/PyArrow** pour les données, **pytest** pour les contrats, **ruff + mypy stricts**
  pour le code, annotations de type et `pathlib` partout, docstrings Google.
- **Zéro placeholder** : pas de `TODO`, pas de fonction vide, pas d'exemple non exécutable.

---

## 4. Les six stacks, ce que chacune montre

Même contrat `BaseModel`, mêmes callbacks de projet, mêmes noms de métriques — seule
l'implémentation change. C'est ce qui rend la comparaison possible.

| Stack | Ce qu'elle met en évidence | Persistance |
| --- | --- | --- |
| **scikit-learn** | `ColumnTransformer` appris sur le train uniquement, plusieurs algorithmes interchangeables, validation croisée | `model.joblib` |
| **XGBoost** | Early stopping passé au **constructeur** (armé seulement si un split de validation existe), export natif `.json`, gestion des paramètres inutilisés selon l'algorithm | `model.joblib` + `.json` natif |
| **LightGBM** | Early stopping par **callbacks de `fit`** (`eval_X`/`eval_y`), export natif `.txt` | `model.joblib` + `.txt` natif |
| **PyTorch** | Boucle d'époques écrite à la main : `DataLoader` semé, `forward`/`backward`/`step`, instantané et restauration du meilleur `state_dict` | checkpoint `model.pt` (poids + méta, rechargé en `weights_only=True`) |
| **Keras** | API fonctionnelle : graphe déclaré puis `compile`/`fit`, callbacks natifs (`EarlyStopping(restore_best_weights)`, `ReduceLROnPlateau`, `TerminateOnNaN`) pontés vers les callbacks du projet, `class_weight` pour le déséquilibre | archive native `model.keras` + `model.meta.json` |
| **TensorFlow** | Niveau le plus bas : modèle **subclassé**, couches maison (`DenseBlock`, `ResidualBlock`), pipeline `tf.data`, `GradientTape` + `tf.function`, pertes sur logits, écrêtage du gradient, pondération d'échantillons | poids `model.weights.h5` + `model.meta.json` |

Pièges documentés dans le code (et résolus) que ces stacks partagent :

- **Les schedules Keras/TensorFlow se paramètrent en pas d'optimiseur, pas en époques.** Avec
  `decay_steps=epochs`, le taux d'apprentissage tombe à zéro dès la première époque et
  l'apprentissage se fige silencieusement. `steps_per_epoch_` est calculé avant compilation,
  journalisé, persisté et restauré au chargement.
- **Un modèle subclassé n'a pas de graphe sérialisable** : on persiste les poids et on reconstruit
  l'architecture depuis la configuration, d'où le sidecar JSON obligatoire.
- **On n'importe jamais un framework lourd seulement pour le semer ou lire sa version.** Faire
  cohabiter PyTorch et TensorFlow dans un même processus peut déclencher un conflit de runtime
  OpenMP (segfault) : `set_seed` et les collectes de versions se limitent aux modules déjà chargés.
- **`BatchNormalization` casse sur un lot de taille 1** : le dernier lot est écarté quand
  `n_samples % batch_size == 1`.
- **`torch.load` est en `weights_only=True` par défaut depuis torch 2.x** : les tableaux NumPy ne
  passent pas dans le checkpoint, les classes sont donc sérialisées en listes.
- **`ReduceLROnPlateau` n'accepte plus `verbose=`** (supprimé depuis torch 2.7).

---

## 5. Le générateur : `tools/scaffold`

Les projets ne sont pas écrits à la main puis recopiés : ils sont **générés** à partir de
templates Jinja2 et de registres, ce qui garantit l'homogénéité et permet de régénérer toute la
famille en une commande.

```
tools/
├── scaffold/
│   ├── build.py                  # python -m tools.scaffold.build --manifest <yaml> [--all]
│   ├── config.py                 # couches de configuration : global -> stack -> famille -> manifeste
│   ├── registry.py               # modèles pydantic des registres (StackSpec, FamilySpec…)
│   ├── engine.py                 # rendu Jinja2 + post-traitement (ruff format / ruff --fix)
│   ├── context.py                # contexte de rendu (spec, stack, family, helpers comme wrap())
│   ├── registry/
│   │   ├── stacks.yaml           # 20 stacks : dépendances, classe, format du fichier modèle, docs
│   │   └── families.yaml         # familles de problèmes : modalité, tâche, builder de notebooks
│   ├── defaults/
│   │   ├── global.yaml           # valeurs par défaut de tous les projets (train, preprocessing…)
│   │   └── family/<famille>.yaml # cas d'usage : données, métriques, seuils de qualité
│   ├── manifests/*.yaml          # UN manifeste = UN projet livré
│   ├── notebooks/tabular.py      # générateur des six notebooks (prose française + code)
│   └── templates/
│       ├── base/                 # partagé par tous : Makefile, conf/, utils, schemas, models/base.py
│       ├── modality/tabular/     # data, preprocessing, features, training, evaluation, inference…
│       ├── task/classification/  # métriques, rapport, visualisations propres à la tâche
│       └── stack/<stack>/        # src/models/{model.py.j2,factory.py.j2} de la stack
└── verify.py                     # harnais de vérification (lint, types, tests, notebooks, pipeline)
```

Un manifeste ne décrit **que ce qui distingue le projet** (identité, stack, famille, budget
d'entraînement, modèle, objectifs pédagogiques) : le reste vient des couches de défauts. Ajouter
une stack revient à écrire `templates/stack/<clé>/src/models/{model,factory}.py.j2` et une entrée
dans `registry/stacks.yaml`.

Vérification — c'est la définition de « livré » dans ce dépôt :

```bash
python -m tools.verify --all --quiet                 # lint + types + tests + pipeline
python -m tools.verify --all --notebooks-inplace     # … et les 6 notebooks exécutés de chaque projet
python -m tools.verify data-science/classification/with-pytorch
```

État au dernier passage : **6/6 projets conformes** (ruff check, ruff format, mypy strict,
163 tests par projet, 6 notebooks exécutés, `python -m src.main mode=all`).

Les notebooks sont versionnés **sans outputs** : ils sont rejoués par `tools/verify.py`, le dépôt
reste léger et leur exécution reste une preuve vérifiable plutôt qu'une capture d'écran.

---

## 6. Conventions de travail

- **Langue** : documentation, README et notebooks en français ; docstrings et identifiants en
  anglais (le code se lit dans les deux langues, la documentation s'adresse à l'équipe).
- **Qualité avant vitesse** : chaque projet est vérifié avant d'être commité ; la dette de lint
  est soldée plutôt que contournée (les seules exemptions sont commentées dans `pyproject.toml`).
- **Déterminisme** : graines partout, opérations TensorFlow déterministes, CPU par défaut,
  mélange des lots semé. Deux runs identiques produisent les mêmes métriques.
- **Aucune dépendance réseau à l'exécution** : données synthétiques, pas de poids pré-entraînés
  téléchargés, pas de clé API requise.

---

## 7. Feuille de route

La famille `data-science/classification` est complète (six stacks). Suite du plan, par ordre de
valeur pédagogique :

1. `data-science/regression`, `clustering`, `time-series`, `recommendation`, `anomaly-detection`
   — les contrats `BaseModel`, les registres et le générateur de notebooks les acceptent déjà ;
   les tâches sont déclarées dans `ARCHITECTURES_BY_TASK` de chaque stack.
2. `data-eng/` (pandas + PyArrow, DuckDB, Prefect), `mlops/` (MLflow, GitHub Actions + tox),
   `analytics/` (rapports Jinja2, monitoring de drift SciPy).
3. `ai-eng/` (LangChain, Transformers) et serving (FastAPI).
4. `computer-vision/` et `nlp/` (spaCy, Transformers), avec des générateurs de données
   synthétiques adaptés à chaque modalité.

Chaque ajout suit la même procédure : entrée de registre -> templates de stack ou de famille ->
manifeste -> `build` -> `verify` -> commit.
