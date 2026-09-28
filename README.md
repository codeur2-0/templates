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

**15 projets**, tous verts dans `tools/verify.py` (lint, formatage, typage, tests, six notebooks
exécutés, pipeline complet `data → train → evaluate → predict`).

### 1.1 `data-science/classification` — prédiction d'attrition client (churn télécom), six stacks

| Projet | Stack | Modèle | ROC AUC (test) | PR AUC | Accuracy | F1 | Log loss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [`with-sklearn`](data-science/classification/with-sklearn) | scikit-learn | `random_forest` | 0,8654 | 0,7240 | **0,8175** | 0,6636 | 0,4190 |
| [`with-xgboost`](data-science/classification/with-xgboost) | XGBoost | `xgboost` (47 rounds) | 0,8580 | 0,7049 | **0,8237** | 0,5889 | **0,3882** |
| [`with-lightgbm`](data-science/classification/with-lightgbm) | LightGBM | `lightgbm` (72 rounds) | 0,8544 | 0,6925 | 0,8150 | 0,5843 | 0,3947 |
| [`with-pytorch`](data-science/classification/with-pytorch) | PyTorch | `mlp` (4 161 paramètres) | **0,8727** | **0,7365** | 0,7937 | 0,6570 | 0,4388 |
| [`with-keras`](data-science/classification/with-keras) | Keras (API fonctionnelle) | `mlp` (4 161 paramètres) | 0,8655 | 0,7282 | 0,8137 | **0,6711** | 0,4251 |
| [`with-tensorflow`](data-science/classification/with-tensorflow) | TensorFlow (`GradientTape`) | `mlp` (4 161 paramètres) | 0,8709 | 0,7341 | 0,7863 | 0,6517 | 0,4515 |

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

### 1.2 `data-science/regression` — estimation de prix immobilier (AVM), six stacks

Même cas d'usage, mêmes données, mêmes métriques : seul le modèle change. Cible `price_eur`
(8 000 biens, 33 features après pré-traitement, split 65/15/20), seuil de conformité
`RMSE ≤ 90 000 EUR`.

| Projet | Stack | Modèle | RMSE (EUR) | MAE | R² | MAPE % | max_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [`with-sklearn`](data-science/regression/with-sklearn) | scikit-learn | `hist_gradient_boosting` | 34 805 | 22 823 | 0,9496 | 7,51 | 279 811 |
| [`with-xgboost`](data-science/regression/with-xgboost) | XGBoost | `xgboost` (374 rounds) | 33 674 | **22 013** | 0,9528 | **7,28** | 327 510 |
| [`with-lightgbm`](data-science/regression/with-lightgbm) | LightGBM | `lightgbm` (387 rounds) | 33 353 | **21 897** | 0,9537 | **7,24** | 297 304 |
| [`with-pytorch`](data-science/regression/with-pytorch) | PyTorch | `mlp` (13 057 paramètres) | **32 212** | 21 974 | **0,9568** | 7,46 | **273 692** |
| [`with-keras`](data-science/regression/with-keras) | Keras (API fonctionnelle) | `mlp` (13 441 paramètres) | 34 127 | 22 465 | 0,9516 | 7,55 | 354 971 |
| [`with-tensorflow`](data-science/regression/with-tensorflow) | TensorFlow (`GradientTape`) | `mlp` (13 441 paramètres) | 33 050 | 22 266 | 0,9546 | 7,70 | 348 765 |

Lecture : ici les réseaux s'en sortent **mieux** que les boosters (normalisation interne de la
cible continue + early stopping sur la validation), à l'inverse du cas churn — la comparaison des
deux familles est précisément ce qui rend le choix de stack argumentable plutôt que dogmatique.

### 1.3 `data-science/clustering` — segmentation d'une base clients retail, scikit-learn

| Projet | Stack | Modèle | Silhouette | Calinski-Harabasz | Davies-Bouldin | ARI latent | Stabilité (ARI) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [`with-sklearn`](data-science/clustering/with-sklearn) | scikit-learn | `kmeans` (k=6, 36 features) | 0,2057 | 232,89 | 1,6612 | 0,418 | 0,978 |

Rapport de conformité **7/7** : qualité de structure, équilibre des tailles, stabilité par
bootstrap, validité externe contre les segments latents du générateur, et profilage métier de
chaque segment (le rapport nomme les segments et propose une action par segment).

### 1.4 `data-science/anomaly-detection` — détection de fraude sur paiements, scikit-learn

| Projet | Stack | Modèle | PR AUC | ROC AUC | Rappel au budget | Précision au budget | Lift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [`with-sklearn`](data-science/anomaly-detection/with-sklearn) | scikit-learn | `isolation_forest` (500 arbres, 2 variables par coupe) | **0,6661** | 0,9468 | 0,6071 | 0,7083 | **30,4x** |

12 000 transactions, prévalence 2,3 % sur le split de test (56 fraudes), budget
d'investigation fixé à 2 % des lignes — la capacité réelle d'une équipe d'analystes. Une PR AUC
aléatoire vaudrait 0,0306 ici : le gain est donc de 0,636 en valeur absolue, et le classement
concentre 30 fois mieux la fraude qu'un tirage au sort.

Couverture par mode opératoire : identité synthétique 0,85 | carte absente 0,65 | prise de
compte 0,60 | **fraude amicale 0,00**. Ce dernier zéro n'est pas un bug à masquer : la fraude
amicale est légitime en apparence (bon appareil, bon historique, bon montant) et aucun détecteur
transactionnel non supervisé ne peut la voir. C'est un plafond structurel, documenté comme tel
dans le rapport et le notebook 06.

Le registre sklearn de la tâche `anomaly` sert quatre détecteurs (`isolation_forest`,
`one_class_svm`, `elliptic_envelope`, `local_outlier_factor` en mode `novelty=True`), ce qui permet
au notebook 04 de comparer réellement trois hypothèses — densité à noyau RBF, écart de
Mahalanobis robuste, rareté relative au voisinage. Le LOF y obtient la meilleure PR AUC (0,624
contre 0,666 pour la forêt d'isolation selon la graine) mais reste écarté : il doit conserver
l'intégralité du jeu d'entraînement en mémoire pour scorer une ligne nouvelle, ce qui est
rédhibitoire sur un flux de millions de paiements.

### 1.5 `data-science/time-series-forecasting` — prévision de consommation électrique, scikit-learn

| Projet | Stack | Modèle | MAPE | MAE | MASE | R² | Biais | Gain sur naif | Couverture 90 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`with-sklearn`](data-science/time-series-forecasting/with-sklearn) | scikit-learn | `hist_gradient_boosting` (41 features) | **3,692 %** | 99,4 MW | 0,420 | 0,953 | +0,89 % | **+59,1 %** | **85,4 %** |

4 800 lignes (1 200 origines x 4 horizons), du 2021-02-10 au 2024-05-24, avec 80 vagues de
froid, 72 canicules et 36 arrêts industriels reproduits fidèlement (3,92 % des jours). Le split
est chronologique : jamais de mélange aléatoire, jamais de validation croisée aléatoire.

| Horizon | MAPE | MAE | Biais | Couverture |
| --- | --- | --- | --- | --- |
| J+1 | 3,555 % | 96,5 MW | +0,84 % | 87,5 % |
| J+2 | 3,733 % | 99,9 MW | +0,86 % | 84,6 % |
| J+3 | 3,596 % | 96,8 MW | +0,78 % | 81,7 % |
| J+7 | 3,886 % | 104,5 MW | +1,07 % | 87,9 % |

L'écart J+1 → J+7 n'est que de 0,33 point, ce qui justifie le choix livré : un modèle unique
portant l'horizon en variable, plutôt que quatre modèles (la comparaison est chiffrée dans le
notebook 04). Le plancher structurel du bruit multiplicatif injecté par le générateur est
d'environ 2,8 % de MAPE — les 0,9 point restants sont l'erreur apprise.

Quatre méthodes d'intervalle sont calibrées sur la validation puis comparées sur le test
(niveau nominal 90 %) : quantiles de résidus 66,5 % | conformal 73,2 % | conformal adaptatif
79,9 % | **conformal normalisé 85,4 %** (méthode livrée). Le bruit étant multiplicatif
(écart-type des résidus 109 MW en validation, 154 MW en test), la normalisation par l'échelle
locale est précisément ce qui redresse la couverture. La variante normalisée + adaptative monte
à 86,7 % mais tombe à 0 % sur les vagues de froid : elle est documentée comme limite connue
plutôt que livrée.

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
| **XGBoost** | Early stopping natif passé au **constructeur** (armé seulement si un split de validation existe), registres `xgboost` / `xgboost_dart` / `xgboost_linear`, allow-list de paramètres par algorithme, pont `TrainingCallback` vers les callbacks du projet | `model.joblib` |
| **LightGBM** | Early stopping par **callback de `fit`** (`eval_set` + `lightgbm.early_stopping`), croissance leaf-wise, `subsample` désactivé pour GOSS, early stopping indisponible pour DART, `best_iteration` réutilisé en prédiction | `model.joblib` |
| **PyTorch** | Boucle d'époques écrite à la main : `DataLoader` semé, `forward`/`backward`/`step`, instantané et restauration du meilleur `state_dict` | checkpoint `model.pt` (poids + méta, rechargé en `weights_only=True`) |
| **Keras** | API fonctionnelle : graphe déclaré puis `compile`/`fit`, callbacks natifs (`EarlyStopping(restore_best_weights)`, `ReduceLROnPlateau`, `TerminateOnNaN`) pontés vers les callbacks du projet, `class_weight` pour le déséquilibre | archive native `model.keras` + sidecar `model.config.json` |
| **TensorFlow** | Niveau le plus bas : modèle **subclassé**, couches maison (`DenseBlock`, `ResidualBlock`), pipeline `tf.data`, `GradientTape` + `tf.function`, pertes sur logits, écrêtage du gradient, pondération d'échantillons | poids `model.weights.h5` + sidecar `model.config.json` |

Pièges documentés dans le code (et résolus) que ces stacks partagent :

- **Les schedules Keras se paramètrent en pas d'optimiseur, pas en époques.** Avec
  `decay_steps=epochs`, le taux d'apprentissage tombe à zéro dès la première époque et
  l'apprentissage se fige silencieusement. `steps_per_epoch` est calculé avant compilation et
  journalisé dans le contexte d'entraînement.
- **Un modèle subclassé n'a pas de graphe sérialisable** : on persiste les poids et on reconstruit
  l'architecture depuis la configuration, d'où le sidecar JSON obligatoire (et les **noms de
  couches explicites**, puisque `load_weights` apparie les variables par nom et que Keras les
  numérote sinon depuis un compteur global au processus).
- **`tf.function` capture les variables du réseau clos** : un pas d'entraînement compilé au niveau
  module réutiliserait le graphe du premier réseau et échouerait au deuxième run
  (*« only supports singleton tf.Variables »*). Le pas compilé est donc lié à un réseau précis.
- **XGBoost 3.2 : jamais de `set_params` après construction.** L'appel reconfigure le booster C++
  et sérialise mal un `eval_metric` en liste (*« Unknown metric function ['logloss', 'auc'] »*
  remonté plus tard, à la prédiction). Les callbacks passent au constructeur, puis sont détachés
  par affectation directe.
- **Rien de picklable ne doit être défini localement** : ponts de callbacks (`RoundBridge`) et
  constructeurs du registre d'algorithmes sont des objets de **niveau module** — une lambda dans
  `AlgorithmSpec.builder` casse `joblib.dump` de tout le modèle.
- **`BatchNorm1d` casse sur un lot de taille 1** à l'entraînement : le dernier lot est écarté quand
  il est singleton (la régression du dépôt l'active, `batch_norm: true`).
- **`torch.load` est en `weights_only=True` par défaut depuis torch 2.x** : ni tableaux NumPy, ni
  `TorchVersion` ne passent dans le checkpoint — classes sérialisées en listes, version en `str`.

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

État au dernier passage : **15/15 projets conformes** (ruff check, ruff format, mypy strict,
165 tests par projet soit 2 475 au total, 90 notebooks exécutés, `python -m src.main mode=all`
de bout en bout). Chaque projet est rejoué intégralement — lint, typage, tests, exécution des
six notebooks et pipeline complet — avant d'être considéré comme livré.

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

État du générateur : `registry/families.yaml` déclare **30 familles** et `registry/stacks.yaml`
**18 stacks** ; les couches `base/`, `modality/tabular/`, `task/{classification,regression,clustering,anomaly,forecasting}/`,
`family/{binary_classification,regression,clustering,anomaly_detection,time_series_forecasting}/`
et `stack/{sklearn,xgboost,lightgbm,pytorch,tensorflow,keras}/` sont écrites et vérifiées. Ce qui reste, par ordre de valeur pédagogique :

1. **Familles tabulaires restantes** — `multiclass_classification` et `recommendation`.
   `anomaly_detection` et `time_series_forecasting` sont livrées : leurs couches `task/`
   (évaluateur, rapport, prédicteur, figures) et `family/` (générateur + valeurs par défaut)
   servent de modèle pour les deux suivantes. Les métriques existent déjà dans
   `losses_metrics.py` (macro-F1, MCC, recall@top-k, Precision@K / NDCG@K / MAP@K) et le
   registre sklearn déclare quatre détecteurs pour la tâche `anomaly`. Pour `recommendation`
   il reste la couche `task/` (évaluation par utilisateur et par catalogue, coupure temporelle
   leave-one-out), la couche `family/` (générateur d'interactions creuses) et la branche du
   générateur de notebooks.
2. **Autres modalités** — `data-eng/` (pandas + PyArrow, DuckDB, Prefect), `mlops/` (MLflow,
   GitHub Actions + tox), `analytics/` (rapports Jinja2, monitoring de drift SciPy),
   `ai-eng/` (LangChain, Transformers, serving FastAPI), `computer-vision/` et `nlp/`
   (spaCy, Transformers). Chacune demande une nouvelle couche `modality/` (loaders,
   pré-traitement, pipelines, tests) en plus des couches `family/` et `task/`.
3. **Stacks déjà déclarées, non implémentées** — `spacy`, `transformers`, `langchain`, `duckdb`,
   `pandas`, `prefect`, `mlflow`, `fastapi`, `scipy`, `pandera` : l'entrée de registre existe,
   le dossier `templates/stack/<clé>/src/models/` reste à écrire.

Chaque ajout suit la même procédure : entrée de registre -> templates de stack ou de famille ->
manifeste -> `build` -> `verify` -> commit.
