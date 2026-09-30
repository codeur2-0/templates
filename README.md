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

**28 projets** sur sept familles tabulaires, tous verts dans `tools/verify.py` (lint, formatage,
typage, tests, six notebooks exécutés, pipeline complet `data → train → evaluate → predict`).

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

### 1.4 `data-science/anomaly-detection` — détection de fraude sur paiements, trois stacks

| Projet | Stack | Modèle | PR AUC | ROC AUC | Rappel au budget | Précision au budget | Lift |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [`with-sklearn`](data-science/anomaly-detection/with-sklearn) | scikit-learn | `isolation_forest` (500 arbres, 2 variables par coupe) | 0,6661 | 0,9468 | 0,6071 | 0,7083 | 30,4x |
| [`with-pytorch`](data-science/anomaly-detection/with-pytorch) | PyTorch | `autoencoder` 51-16-**2**-16-51 (1 781 paramètres) | **0,6915** | **0,9745** | **0,6250** | **0,7292** | **31,2x** |
| [`with-tensorflow`](data-science/anomaly-detection/with-tensorflow) | TensorFlow (`GradientTape`) | `autoencoder` 51-8-**3**-8-51 (934 paramètres) | 0,6459 | 0,9672 | **0,6250** | **0,7292** | **31,2x** |

12 000 transactions, prévalence 2,3 % sur le split de test (56 fraudes), budget
d'investigation fixé à 2 % des lignes — la capacité réelle d'une équipe d'analystes. Une PR AUC
aléatoire vaudrait 0,0306 ici : le gain est donc de 0,62 à 0,66 en valeur absolue, et le
classement concentre 30 fois mieux la fraude qu'un tirage au sort. Aucun des trois détecteurs
n'a vu une étiquette : ils sont entraînés sur le flux brut et les étiquettes ne servent qu'à mesurer.

| Rappel au budget par mode opératoire | Forêt d'isolation | Auto-encodeur PyTorch | Auto-encodeur TensorFlow |
| --- | --- | --- | --- |
| Prise de compte (20 fraudes) | 0,60 | 0,80 | **0,85** |
| Carte absente (17) | **0,65** | 0,59 | 0,53 |
| Identité synthétique (13) | **0,85** | 0,54 | 0,54 |
| Fraude amicale (6) | 0,00 | 0,33 | 0,33 |

**Lecture honnête** : à PR AUC comparable, les deux familles ne voient pas la même fraude. La
forêt d'isolation répartit son attention (sa première variable ne pèse que 5,9 % de l'importance
par permutation) et excelle sur l'identité synthétique ; les auto-encodeurs concentrent la leur
sur l'historique de contestations, l'appareil neuf et les échecs récents (44 à 45 % à eux trois) et
dominent sur la prise de compte. C'est aussi ce qui leur fait capturer 2 fraudes amicales sur 6 :
un effectif trop petit pour conclure, mais qui montre que la fraude amicale — légitime en
apparence, trahie seulement par l'historique de contestations — reste un plafond structurel du
détecteur transactionnel, pas un zéro absolu. Deux détecteurs aussi complémentaires sont un
argument pour un score combiné, à mesurer plutôt qu'à supposer.

Le levier décisif d'un auto-encodeur de détection est sa **capacité** : sur la validation, deux
couches de 64 et 32 neurones plafonnent entre 0,21 et 0,45 de PR AUC (le réseau apprend aussi à
reconstruire la fraude), une seule couche de 8 ou 16 neurones avec un goulot de 2 ou 3 dimensions
monte à 0,55–0,68. Les réglages sont choisis sur 3 graines ; quand l'avance du meilleur point est
inférieure à deux fois sa dispersion, c'est le plus stable qui est retenu. Limite documentée : la
file d'alertes de l'auto-encodeur TensorFlow change de 6 à 8 alertes sur 36 d'une graine à l'autre
(recouvrement 0,78–0,83 au notebook 04), sous le seuil de 0,85 visé avant production ; celle de
PyTorch est à 0,94.

Le registre sklearn de la tâche `anomaly` sert quatre détecteurs (`isolation_forest`,
`one_class_svm`, `elliptic_envelope`, `local_outlier_factor` en mode `novelty=True`), ce qui permet
au notebook 04 de comparer réellement trois hypothèses — densité à noyau RBF, écart de
Mahalanobis robuste, rareté relative au voisinage. Le LOF y obtient la meilleure PR AUC (0,624
contre 0,666 pour la forêt d'isolation selon la graine) mais reste écarté : il doit conserver
l'intégralité du jeu d'entraînement en mémoire pour scorer une ligne nouvelle, ce qui est
rédhibitoire sur un flux de millions de paiements.

### 1.5 `data-science/time-series-forecasting` — prévision de consommation électrique, cinq stacks

| Projet | Stack | Modèle | MAPE | MAE | MASE | R² | Biais | Gain sur naif | Couverture 90 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`with-sklearn`](data-science/time-series-forecasting/with-sklearn) | scikit-learn | `hist_gradient_boosting` (15 feuilles) | 3,692 % | 99,4 MW | 0,420 | 0,953 | +0,89 % | +59,1 % | 85,4 % |
| [`with-xgboost`](data-science/time-series-forecasting/with-xgboost) | XGBoost | `xgboost` (profondeur 3, arrêt anticipé) | 3,441 % | 93,3 MW | 0,394 | 0,957 | +0,33 % | +61,9 % | **87,2 %** |
| [`with-lightgbm`](data-science/time-series-forecasting/with-lightgbm) | LightGBM | `lightgbm` (7 feuilles, arrêt anticipé) | 3,521 % | 94,0 MW | 0,397 | 0,960 | +0,69 % | +61,0 % | **87,2 %** |
| [`with-pytorch`](data-science/time-series-forecasting/with-pytorch) | PyTorch | `mlp` 128-64 (dropout 0,1) | **3,386 %** | **90,4 MW** | **0,382** | **0,962** | −0,76 % | **+62,5 %** | 84,8 % |
| [`with-keras`](data-science/time-series-forecasting/with-keras) | Keras (API fonctionnelle) | `mlp` 128-64 `gelu` (dropout 0,1) | 3,422 % | 90,5 MW | **0,382** | 0,961 | **−0,22 %** | +62,1 % | 84,0 % |

4 800 lignes (1 200 origines x 4 horizons), du 2021-02-10 au 2024-05-24, avec 80 vagues de
froid, 72 canicules et 36 arrêts industriels reproduits fidèlement (3,92 % des jours). Le split
est chronologique : jamais de mélange aléatoire, jamais de validation croisée aléatoire. Les
cinq projets passent les sept critères de la famille.

**Lecture honnête de ce tableau** : les cinq stacks tiennent dans 0,31 point de MAPE, et ce qui
les départage n'est pas le MAPE. Les boosters XGBoost et LightGBM ont la meilleure couverture
d'intervalle (87,2 %) et un biais positif faible. Les réseaux ont le meilleur MAPE, mais sous-prévoient
en mars (biais mensuel de −3,3 à −3,4 %) et leur couverture reste sous la cible métier de 85 %
(au-dessus du seuil bloquant de 82 %). Pour un acheteur d'énergie, une sous-prévision coûte un achat spot : le
choix se discute sur ces deux critères, pas sur le troisième chiffre après la virgule.

Chaque notebook commente **sa** stack : les réglages et les commentaires chiffrés propres à une
stack (familles comparées, perte, levier de complexité) se déclarent dans
`extras.notebook_forecasting` du manifeste, avec les valeurs scikit-learn par défaut. Trois
lectures qui en sortent : la perte quadratique l'emporte pour les trois boosters ; en Keras, un
notebook raccourci (12 époques, sans arrêt anticipé) donnait l'avantage à la perte absolue, et
le protocole de production l'inverse ; en PyTorch, la perte absolue garde une avance de 0,07 point
de validation, sous la dispersion des replis, donc sans changement de configuration. Enfin
`gelu` bat `relu` d'un point de MAPE en Keras, pas en PyTorch.

Détail du projet scikit-learn :

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

### 1.6 `data-science/recommendation` — classement de catalogue e-commerce, deux stacks

| Projet | Stack | Modèle | NDCG@10 | Precision@10 | Recall@10 | MAP@10 | Hit-rate@10 | Couverture catalogue |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`with-sklearn`](data-science/recommendation/with-sklearn) | scikit-learn | `hist_gradient_boosting` (15 feuilles) | 0,602 | **0,299** | **0,733** | 0,582 | **0,951** | **65,3 %** |
| [`with-pytorch`](data-science/recommendation/with-pytorch) | PyTorch | `mlp` 32 pointwise, sigmoid (1 953 paramètres) | **0,614** | 0,297 | 0,725 | **0,606** | 0,942 | **65,4 %** |

36 000 couples (1 200 utilisateurs x 30 candidats, catalogue de 900 références, ~14 % de
pertinence), split chronologique par session, évaluation **par utilisateur** (226 utilisateurs de
test avec au moins une intention observable) et jamais ligne à ligne. Verdict **conforme, 8/8
objectifs** pour les deux stacks.

| Référence (même test) | NDCG@10 | Couverture catalogue |
| --- | --- | --- |
| Tirage aléatoire (plancher) | 0,235 | 82,8 % |
| **Tri par popularité** (le moteur historique à battre) | 0,300 | 15,4 % |
| Intention récente (ne recommander que le déjà-vu) | 0,343 | 83,6 % |
| Plafond oracle (probabilité de pertinence avant bruit, jeu complet) | 0,726 | — |

Le modèle double le NDCG du tri par popularité (+100 %) et capte 83 % du plafond atteignable, en
exposant quatre fois plus de catalogue. Les utilisateurs froids (une commande ou moins sur 12 mois)
sont servis presque aussi bien que les autres (rapport de rappel 0,96), et le NDCG varie de 0,033
seulement entre plis chronologiques. Le tri par popularité échoue précisément au critère de
couverture : c'est le mécanisme par lequel la longue traîne meurt, que le NDCG seul ne voit pas.

Le MLP PyTorch fait jeu égal avec le boosting : +0,012 de NDCG@10, du même ordre que l'amplitude
entre plis chronologiques (0,029 pour le réseau), pour une couverture identique et un rapport de
rappel des utilisateurs froids de 0,99. Il score les mêmes features, sans identifiant appris : ce
n'est **pas** un modèle à deux tours, qui exigerait des embeddings d'identifiants et abandonnerait
les utilisateurs froids que ce jeu met au contrat. Sa capacité (une couche de 32, dropout 0,3) est
choisie sur 3 graines : un réseau 128-64 fait 0,001 de mieux en validation, soit moins de deux fois
sa dispersion, pour huit fois plus de paramètres. L'arrêt anticipé tombe entre les époques 3 et 6.

### 1.7 `data-science/multiclass-classification` — diagnostic du mode de défaillance machine, cinq stacks

Une alarme d'automate arrête une fraiseuse CNC ; il faut envoyer la bonne équipe. Six modes à
distinguer à partir des capteurs (`false_alarm`, `tool_wear`, `heat_dissipation`,
`power_failure`, `overstrain`, `random_failure`, de 30 % à 6 %), métrique de pilotage
**macro-F1**, et une décision qui ne se réduit pas à l'argmax : acquitter une panne réelle coûte
vingt fois plus cher que déranger une équipe pour rien.

| Projet | Stack | Modèle | Macro-F1 | Log loss | ECE | Coût / alarme (coût min.) | Pannes acquittées |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [`with-sklearn`](data-science/multiclass-classification/with-sklearn) | scikit-learn | `hist_gradient_boosting` (100 x 8 feuilles) | 0,671 | 0,694 | 0,026 | 172 EUR | 0,0 % |
| [`with-xgboost`](data-science/multiclass-classification/with-xgboost) | XGBoost | `xgboost` (106 rounds, profondeur 4) | **0,676** | **0,669** | 0,027 | **167 EUR** | 0,0 % |
| [`with-lightgbm`](data-science/multiclass-classification/with-lightgbm) | LightGBM | `lightgbm` (62 rounds, 15 feuilles) | 0,674 | 0,684 | 0,039 | 168 EUR | 0,0 % |
| [`with-pytorch`](data-science/multiclass-classification/with-pytorch) | PyTorch | `mlp` softmax (4 134 paramètres) | 0,657 | 0,731 | 0,030 | 168 EUR | 0,0 % |
| [`with-keras`](data-science/multiclass-classification/with-keras) | Keras (API fonctionnelle) | `mlp` softmax (4 134 paramètres) | 0,659 | 0,726 | **0,020** | 168 EUR | 0,0 % |

6 000 alarmes, split stratifié 65/15/20, mêmes graines et mêmes 28 features pour les cinq stacks.
Les cinq rapports rendent un verdict **conforme (7/7 objectifs)**.

Trois références sont rejouées sur le même test, et c'est leur écart qui donne un sens aux
chiffres :

| Référence | Macro-F1 | Coût / alarme | Pannes acquittées |
| --- | --- | --- | --- |
| Classe majoritaire (plancher) | 0,077 | 1 755 EUR | 100 % |
| **Routage actuel par code automate** (la règle à battre) | 0,375 | 428 EUR | 14,4 % |
| Plafond oracle (argmax des vraies probabilités du générateur) | 0,679 | — | — |

**Lecture honnête de ce tableau** :

- Les boosters atteignent le plafond (97 à 99 % du chemin règle → plafond), les réseaux en sont
  à 93-94 %. Les écarts entre boosters sont dans le bruit : le signal est épuisé, et la suite se
  joue ailleurs.
- **La décision compte plus que le modèle.** À l'argmax, chaque modèle coûte 455 à 498 EUR par
  alarme et acquitte 21 à 24 % des pannes réelles — plus cher que la règle actuelle. La
  décision à coût minimal (probabilités x matrice de coûts `diagnosis.costs`) ramène ce coût à
  167-172 EUR (**−60 %** face au routage actuel) et n'acquitte plus aucune panne.
- `random_failure` a un rappel nul pour tout modèle, oracle compris : aucun capteur ne l'annonce.
  C'est un plafond structurel documenté, pas un défaut de réglage.
- Un boosting profond (300 x 31 feuilles) n'améliore pas le macro-F1 de validation mais devient
  sur-confiant (ECE 0,13 contre 0,06) : le modèle livré est volontairement petit, parce que la
  décision multiplie ses probabilités par des euros.

La couche `task/multiclass/` apporte l'évaluateur (références, verdict, confusions, décision à coût
minimal, revue experte, calibration top-label), le rapport, le predictor (mode prédit, probabilité
par mode, décision, équipe, drapeau de revue, justification) et dix figures ; les réglages métier
vivent dans le bloc `diagnosis` de `conf/config.yaml`.

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
- **XGBoost n'accepte que des classes codées 0..K-1** : des libellés (`"tool_wear"`) lèvent au
  `fit`. Le modèle encode les labels à l'entraînement (et dans l'`eval_set`) puis décode à la
  prédiction ; en binaire 0/1 le code est le label lui-même, l'entraînement est inchangé.
- **`sklearn.metrics.log_loss` trie ses `labels` en interne** et lit les colonnes de probabilités
  dans cet ordre trié — sans la moindre erreur si elles sont rangées autrement ; `roc_auc_score`,
  lui, refuse des labels non triés. Le registre de métriques réaligne colonnes et labels avant
  l'appel (mesuré : une log loss de 3,4 au lieu de 0,71 avec des colonnes dans l'ordre métier).
- **Les stubs de numpy ≥ 2.4 utilisent la syntaxe `type X = …` de Python 3.12** : mypy, qui cible
  Python 3.11 dans les projets, s'arrête sur `numpy/__init__.pyi` avant de lire une ligne de code.
  Les dépendances de développement bornent donc `numpy<2.4`.
- **Un `HistGradientBoostingClassifier` multi-classes construit K arbres par itération** : 300
  itérations x 6 classes = 1 800 arbres, soit 30 s d'entraînement sous Windows et des probabilités
  sur-confiantes. Le projet multi-classes livre 100 itérations x 8 feuilles.
- **Une étude de stabilité entre graines peut mentir par construction** : écrire
  `params={"random_state": seed}` ne change rien à un réseau, dont la graine est un attribut du
  modèle et non un hyperparamètre. Trois entraînements identiques donnent une stabilité parfaite
  et fausse. Le notebook d'anomalie pose la graine sur le modèle (`candidate.random_state = seed`)
  et ne l'écrit dans les paramètres que si la stack l'y déclare.
- **`FitResult.duration_seconds` n'était renseigné que par le `Trainer`** : un appel direct à
  `model.fit` (notebooks, études) rendait 0 s, et les colonnes de coût des notebooks affichaient
  0,0 pour tous les modèles. `BaseModel.fit` chronomètre désormais lui-même (horloge monotone).
- **Le classement pointwise de la stack PyTorch était appris en régression** : `ranking` était
  déclaré tête binaire à sigmoïde, mais la perte retombait sur la MSE et la sortie sur le logit
  brut, sans `predict_proba` (tests et notebooks 05/06 en échec). `ranking` partage désormais la
  tête binaire (`BCEWithLogitsLoss`, sigmoïde, probabilités).

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
│   │   ├── stacks.yaml           # 18 stacks : dépendances, classe, format du fichier modèle, docs
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

État au dernier passage : **28/28 projets conformes** (ruff check, ruff format, mypy strict,
4 730 tests au total — 165 par projet, 187 pour les projets multi-classes qui ajoutent les tests
de leur couche tâche —, 168 notebooks exécutés, `python -m src.main mode=all` de bout en bout). Chaque projet est rejoué intégralement — lint, typage, tests, exécution des
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

État du générateur : `registry/families.yaml` déclare **29 familles** et `registry/stacks.yaml`
**18 stacks**. Les couches `base/`, `modality/tabular/`,
`task/{classification,multiclass,regression,clustering,anomaly,forecasting,ranking}/`,
`family/{binary_classification,multiclass_classification,regression,clustering,anomaly_detection,time_series_forecasting,recommendation}/`
et `stack/{sklearn,xgboost,lightgbm,pytorch,tensorflow,keras}/` sont écrites et vérifiées : les
**sept familles tabulaires** sont livrées. Ce qui reste, par ordre de valeur pédagogique :

1. **Stacks tabulaires déclarées, pas encore livrées** — `binary_classification` avec MLflow,
   `recommendation` en TensorFlow et à deux tours. `clustering` en PyTorch / TensorFlow est
   **écarté après mesure** : sur les 36 features standardisées du jeu clients, un k-means dans
   l'espace latent d'un auto-encodeur ne bat le k-means direct sur aucune métrique (silhouette
   0,12–0,20 contre 0,206, ARI latent 0,22–0,39 contre 0,418) et sa stabilité entre graines
   (ARI 0,54–0,83, 250 époques comprises) reste sous le critère de 0,90 — le k-means direct est à
   0,999. Livrer ce projet obligerait à baisser un seuil. Pour les stacks restantes, les couches
   `task/` et `family/` existent ; il reste un manifeste par stack et, comme pour la prévision et la détection d'anomalies, des
   notebooks paramétrés par stack (`extras.notebook_<famille>`) : les commentaires chiffrés d'un
   notebook doivent parler de la stack réellement mesurée.
2. **Autres modalités** — `data-eng/` (pandas + PyArrow, DuckDB, Prefect), `mlops/` (MLflow,
   GitHub Actions + tox), `analytics/` (rapports Jinja2, monitoring de drift SciPy),
   `ai-eng/` (LangChain, Deepagents, Strands, CrewAI, RAG, Transformers, serving FastAPI), `computer-vision/` et `nlp/`
   (spaCy, Transformers). Chacune demande une nouvelle couche `modality/` (loaders,
   pré-traitement, pipelines, tests) en plus des couches `family/` et `task/`.
3. **Stacks déjà déclarées, non implémentées** — `spacy`, `transformers`, `langchain`, `duckdb`,
   `pandas`, `prefect`, `mlflow`, `fastapi`, `scipy`, `pandera` : l'entrée de registre existe,
   le dossier `templates/stack/<clé>/src/models/` reste à écrire.

Chaque ajout suit la même procédure : entrée de registre -> templates de stack ou de famille ->
manifeste -> `build` -> `verify` -> commit.
