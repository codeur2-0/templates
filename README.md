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

**35 projets** — 25 sur sept familles tabulaires (sections 1.1 à 1.7) et 10 sur les six familles
texte `ai-eng/rag` (section 1.8), `ai-eng/question-answering` (section 1.9), `ai-eng/embeddings`
(section 1.10), `ai-eng/text-classification` (section 1.11), `ai-eng/named-entity-recognition`
(section 1.12) et `nlp/summarization` (section 1.13) — tous verts dans
`tools/verify.py` (lint, formatage, typage, tests, six notebooks exécutés, pipeline complet
`data → train → evaluate → predict`).

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

### 1.6 `data-science/recommendation` — classement de catalogue e-commerce, scikit-learn

| Projet | Stack | Modèle | NDCG@10 | Precision@10 | Recall@10 | MAP@10 | Hit-rate@10 | Couverture catalogue |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`with-sklearn`](data-science/recommendation/with-sklearn) | scikit-learn | `hist_gradient_boosting` (15 feuilles) | **0,602** | 0,299 | 0,733 | 0,582 | 0,951 | **65,3 %** |

36 000 couples (1 200 utilisateurs x 30 candidats, catalogue de 900 références, ~14 % de
pertinence), split chronologique par session, évaluation **par utilisateur** (226 utilisateurs de
test avec au moins une intention observable) et jamais ligne à ligne. Verdict **conforme, 8/8
objectifs**.

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

### 1.8 `ai-eng/rag` — assistant documentaire interne sourcé, deux stacks

Le premier cas d'usage **texte** du dépôt, et le premier où le modèle ne produit pas un label mais
une réponse : répondre à une question interne uniquement à partir d'un corpus de procédures, citer
le passage qui fonde la réponse, et s'abstenir explicitement quand le corpus ne la contient pas.
128 documents synthétiques (huit thèmes d'entreprise x huit périmètres x deux éditions) découpés en
passages de 110 tokens (chevauchement 30), 319 questions annotées dont 34 hors corpus, quatre
difficultés mesurées séparément — facile, paraphrase, multi-document, hors corpus. La pertinence
d'un passage est jugée par le chevauchement avec l'extrait annoté (seuil 50 % des mots de contenu),
pas par la seule appartenance au bon document : un bon document n'est pas une bonne réponse.

| Projet | Stack | Retrieval | recall@5 (test) | MRR | Précision des citations | F1 réponse | Abstention (exactitude) | Latence p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`with-tfidf`](ai-eng/rag/with-tfidf) | scikit-learn | BM25 (Okapi) | **0,7194** | 0,7038 | 0,6688 | 0,3210 | 0,8073 | **5,4 ms** |
| [`with-langchain`](ai-eng/rag/with-langchain) | LangChain (LCEL) | BM25 + chaîne prompt/LLM/ancrage | **0,7194** | 0,7038 | 0,6688 | 0,3210 | 0,8073 | 12,9 ms |

Les deux projets partagent le corpus, les questions, le découpage et l'index BM25 : **les chiffres de
qualité sont identiques parce que le retrieval est identique** — c'est voulu, et c'est ce qui rend
la comparaison lisible. Ce que la variante LangChain apporte ne se lit donc pas dans ces colonnes :

- le **prompt** est un objet de configuration (`model.prompt.template`, variables `{context}` et
  `{question}` vérifiées au montage de la chaîne), plutôt qu'une chaîne de caractères enfouie dans le
  code ;
- l'**adaptateur LLM** est une couture : générateur extractif hors ligne par défaut (aucune clé,
  aucun réseau), client OpenAI-compatible optionnel qui reçoit **le prompt rendu par la chaîne** ;
- les **citations traversent tout le graphe** : le nœud d'ancrage conserve le texte, les identifiants
  des passages réellement utilisés, le prompt rendu et les passages, là où un `StrOutputParser`
  rendrait une chaîne nue ;
- la chaîne **n'est pas sérialisée** : l'état (passages, vocabulaire, embeddings) part en joblib et
  le graphe est reconstruit au chargement — un test vérifie que la reconstruction répond à l'identique ;
- le coût de l'orchestration est mesuré, pas caché : **+6,3 ms de médiane** (p50 4,8 -> 11,1 ms) et
  **+7,5 ms de p95** (5,4 -> 12,9 ms) par question, sur un index de 128 passages.

La même stack sert **trois stratégies de retrieval** interchangeables par `model.algorithm` :
`langchain_lexical` (BM25, celle du rapport ci-dessus), `langchain_dense` (n-grammes hachés
compressés par une SVD tronquée apprise sur le train) et `langchain_hybrid` (fusion par rangs
réciproques des deux classements). Le notebook 04 les compare sur le **split de validation** et
publie les intervalles de confiance ; le test qui refait le calcul de la fusion RRF à la main fait
partie de la suite.

Trois références donnent un sens aux 0,7194 : tirage aléatoire 0,0332, ordre du corpus 0,0357, et
98 des 109 questions de test ont une réponse dans le corpus — un rappel de 1,0 est hors d'atteinte.

| Segment (test) | Questions | recall@5 | MRR | F1 réponse | Abstention |
| --- | --- | --- | --- | --- | --- |
| `facile` | 43 | **0,9186** | 0,9193 | 0,3883 | 0,0930 |
| `multi_document` (deux sources) | 13 | 0,7692 | 0,7956 | **0,5218** | 0,0000 |
| `paraphrase` (aucun mot en commun) | 42 | **0,5000** | 0,4549 | 0,1900 | 0,4048 |
| `hors_corpus` | 11 | n/a | n/a | 0,0000 | **1,0000** |

**Lecture honnête** : le segment « paraphrase » divise le rappel par deux et concentre les échecs —
c'est la limite structurelle d'un retriever lexical, mesurée plutôt que supposée, et c'est
exactement ce que la variante dense puis hybride de la même stack est là pour chiffrer. Les
questions hors corpus sont toutes refusées (0 réponse inventée sur 11) et le seuil d'abstention est
calibré sur un split dédié, jamais sur le test. Les deux projets rendent un verdict **conforme**
(`recall_at_5 = 0,7194 >= 0,60`) et exposent le même jeu de métriques par question : recall@k, MRR,
nDCG@k, précision/rappel des citations, F1 de réponse, exactitude et rappel d'abstention, latences
p50/p95, plus les planchers triviaux et les segments.

---

### 1.9 `ai-eng/question-answering` — support client sur fiches produit, deux stacks

Deuxième cas d'usage texte, et deuxième contrat de réponse : ici la réponse attendue n'est pas
« une réponse appuyée sur un passage », c'est **la phrase exacte de la fiche** qui fait foi. Le
corpus synthétique compte 120 fiches produit (vingt produits x six articles : garantie, retour,
frais de retour, prise en charge SAV, compatibilité, mise à jour du firmware) et 200 questions
annotées dont 14 hors corpus. Chaque fiche porte une phrase canonique qui contient la valeur citée
et chaque question est écrite à partir d'elle : l'exact match mesure donc la sélection de la bonne
phrase, pas l'élégance d'une reformulation. Le vocabulaire est réparti par article — délai et
réception pour le retour, montant et frais pour les frais, prise en charge et panne pour le SAV —
car sans cette séparation une question aurait plusieurs bonnes fiches et la métrique ne dirait plus
rien du modèle. Les questions mono-document portent le nom du produit (le seul discriminant lexical
entre vingt fiches qui se ressemblent), les paraphrases gardent ce nom et s'éloignent du vocabulaire
de la fiche, les questions multi-document demandent deux valeurs à la fois (une garantie *et* un
délai d'intervention) et les questions hors corpus portent sur un fait voisin mais absent.

| Projet | Stack | Réponse | Exact match (test) | F1 réponse | recall@1 | Précision des citations | Abstention (équilibrée) | Latence p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`with-tfidf`](ai-eng/question-answering/with-tfidf) | scikit-learn | phrase copiée du meilleur passage | **0,6212** | **0,7381** | **0,8258** | **0,9808** | **0,8106** | **5,3 ms** |
| [`with-langchain`](ai-eng/question-answering/with-langchain) | LangChain (LCEL) | chaîne prompt + retriever + ancrage | **0,6212** | **0,7381** | **0,8258** | **0,9808** | **0,8106** | 11,0 ms |

Les deux projets partagent le corpus, les questions, le découpage et l'index BM25 : les chiffres de
qualité sont identiques par construction, et le coût de l'orchestration est la seule différence
mesurable (+5,3 ms en médiane, +5,6 ms en p95). Les deux verdicts sont **conformes**
(`recall_at_1 = 0,8258 >= 0,70`, la métrique principale de la famille : une réponse ne peut être
exacte que si le *meilleur* passage est le bon).

Deux garde-fous du générateur de réponse valent d'être nommés, parce qu'ils expliquent l'essentiel
des chiffres. `min_overlap = 2` interdit de citer une phrase qui ne partage qu'un mot avec la
question — sans lui, la phrase d'ouverture de chaque fiche (« Cette fiche est publiée par le service
client… ») suffisait à compléter une réponse exacte. `support_ratio = 0,8` écarte les passages dont
le score tombe sous 80 % du meilleur : sans lui, une question sur les *frais de retour* recevait la
phrase du *délai de retour*, deux fois moins bien classée, et perdait son exact match. Les deux
réglages sont mesurés dans le projet (exact match sur le segment facile 0,60 → 0,81, précision des
citations 0,89 → 0,98) et publiés dans la fiche de modèle avec le seuil d'abstention calibré
(41,36, appris sur le split de calibration, jamais sur le test).

| Segment (test) | Questions | Exact match | F1 réponse | recall@1 | Rang moyen du 1er pertinent | Abstention |
| --- | --- | --- | --- | --- | --- | --- |
| `facile` (phrase canonique) | 43 | **0,8140** | **0,8140** | **1,0000** | 1,00 | 0,1860 |
| `paraphrase` (autre formulation) | 12 | 0,5000 | 0,5323 | 0,5000 | 2,17 | 0,4167 |
| `multi_document` (deux fiches) | 11 | 0,0000 | 0,6664 | 0,5000¹ | 1,00 | 0,0909 |
| `hors_corpus` | 6 | n/a | 0,0139 | n/a | n/a | **0,8333** |

¹ Deux fiches d'or : le recall@1 est mécaniquement plafonné à 0,5 sur ce segment — le rang moyen du
premier document pertinent, lui, vaut 1,00.

**Lecture honnête** : sur les questions factuelles, 4 réponses sur 5 sont la phrase exacte de la
bonne fiche, et les citations ne se trompent jamais (précision 0,98) — la part du contrat que le
projet sait tenir. Trois limites sont mesurées plutôt que cachées. D'abord, 19 questions sur 72 sont
refusées, dont 5 des 6 questions hors corpus : le seuil calibré préfère le refus, et la couverture
(0,79) est publiée à côté de l'abstention. Ensuite, la paraphrase divise le recall@1 par deux
(1,00 → 0,50) : c'est la limite structurelle du lexical, mesurée ici sur un contrat où elle coûte
vraiment des points. Enfin, une question multi-document reçoit la phrase du mieux classé des deux
fiches (F1 0,67, abstention 0,09) : le `support_ratio` qui protège les questions factuelles est
exactement ce qui empêche d'aller chercher la seconde fiche, et le rapport le dit au lieu de le
taire. Trois références donnent un sens au 0,8258 : tirage aléatoire 0,0000, ordre du corpus 0,0076,
et 93 % des questions ont une réponse dans le corpus (plafond annoté publié dans les métadonnées de
génération).

Comme pour la famille RAG, la variante LangChain ne change pas le retrieval : elle change ce qui est
**explicite et testable** — le prompt de copie est un objet de configuration, l'adaptateur LLM reste
une couture (extractif hors ligne par défaut, client OpenAI-compatible optionnel), et les citations
traversent le graphe jusqu'au rapport. La même stack sert trois stratégies interchangeables
(`langchain_lexical`, `langchain_dense`, `langchain_hybrid`) que le notebook 04 compare sur le split
de validation.

### 1.10 `ai-eng/embeddings` — index vectoriel d'un catalogue publié trois fois, deux stacks

Troisième cas d'usage texte, et troisième objet mesuré : ici le sujet n'est pas la réponse mais
**l'index** — sa dimension, sa fidélité, sa taille, son débit et le service qu'il rend, dédoublonnage
inclus. Le corpus synthétique compte 168 fiches produites par une recette combinatoire :
vingt-huit références x deux faits (autonomie, garantie légale) x trois styles éditoriaux (notice
constructeur, fiche commerciale, note SAV), et 115 questions annotées dont 19 hors corpus. Trois
équipes publient le même référentiel : les deux phrases du fait — celle qui porte la valeur et celle
qui la contextualise — sont **recopiées** d'une fiche à l'autre, seules la phrase de style, la
section, l'espace documentaire d'origine et le titre changent. Les trois fiches sont donc annotées
pertinentes pour la question, ce qui plafonne le recall@1 au tiers et en fait une lecture de
précision ; 56 questions factuelles, 28 paraphrases, 12 multi-document (six fiches pertinentes :
deux faits x trois écritures) et 19 hors corpus couvrent quatre difficultés. Les paraphrases
partagent deux fois moins de mots pleins avec la phrase source que les questions factuelles
(34 % contre 71 %) sans jamais retirer la référence qui identifie la réponse.

| Projet | Stack | Index | recall@5 (test) | MRR | Exact match | F1 réponse | Précision des citations | Abstention (équilibrée) | Latence p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`with-embedding`](ai-eng/embeddings/with-embedding) | scikit-learn (hachage + SVD) | dense, 128 dimensions (matrice de 0,16 Mo) | **0,9412** | **0,9608** | **0,7941** | **0,8802** | **0,9375** | **0,7563** | **4,6 ms** |
| [`with-tfidf`](ai-eng/embeddings/with-tfidf) | scikit-learn (TF-IDF cosinus) | creux, 138 termes | **0,9412** | **0,9118** | **0,7647** | **0,8413** | **0,9667** | **0,6555** | 7,4 ms |

Les deux projets partagent le corpus, les questions, le découpage (110 tokens, 30 de recouvrement),
la réponse extractive et le seuil d'abstention calibré sur le split de calibration ; seul l'index
change. Les deux verdicts sont **conformes** (`recall_at_5 = 0,9412 >= 0,90`, la métrique principale
de la famille) et les deux publient leurs planchers : tirage aléatoire 0,0539, ordre du corpus
0,0000. L'index dense est un hachage de n-grammes (uni+bigrammes, 4096 dimensions) projeté par une
SVD tronquée apprise sur le train en 128 dimensions, vecteurs normalisés en norme L2 : la
similarité cosinus devient un produit matriciel et le score, remappé dans `[0, 1]`, reste
calibrable. Ce que la famille ajoute au rappel, c'est la **mesure de l'index lui-même** : matrice de
0,16 Mo contre 5,25 Mo pour l'espace de hachage brut (trente-deux fois plus petit), fidélité de
projection 1,0000 (l'ordre des similarités est inchangé sur les paires testées), index construit en
0,42 s en dense et 0,07 s en lexical, latence p95 sous 8 ms dans les deux cas.

**Lecture honnête** : l'index dense gagne le classement (MRR 0,96 contre 0,91) et l'exact match
(0,79 contre 0,76) ; l'index lexical garde la précision des citations (0,97 contre 0,94), coûte
sept fois moins cher à construire et refuse 3 des 7 questions hors corpus du test contre 4 — chaque
projet publie ses refus avec sa couverture (0,94 en dense, 0,88 en lexical), jamais l'un sans
l'autre. Trois limites sont mesurées plutôt que tues. D'abord, la compression ne crée pas de
sémantique : un hachage de n-grammes reste un index de mots, et les deux variantes tombent à 0,60
d'exact match sur les paraphrases contre 1,00 sur les questions factuelles. Ensuite, le recall@1 est
plafonné par construction (trois fiches pertinentes par fait, six pour le multi-document) : la
famille le publie en métrique secondaire et lit la précision au premier rang à sa place. Enfin, le
multi-document reste le point faible du dense (exact match 0,25 pour un F1 de 0,82) : la première
phrase est trouvée, la seconde est écartée par le garde-fou `support_ratio` qui protège les
questions factuelles — le rapport le dit au lieu de le taire.

| Segment (test) | Questions | Exact match | F1 réponse | recall@5 | Rang moyen du 1er pertinent | Abstention |
| --- | --- | --- | --- | --- | --- | --- |
| `facile` (phrase canonique) | 20 | **1,0000** | **1,0000** | **1,0000** | 1,00 | 0,0000 |
| `paraphrase` (autre formulation) | 10 | 0,6000 | 0,6645 | 0,8667 | 1,40 | 0,2000 |
| `multi_document` (deux faits) | 4 | 0,2500 | 0,8205 | 0,8333 | 1,00 | 0,0000 |
| `hors_corpus` | 7 | n/a | 0,0000 | n/a | n/a | **0,5714** |

Le même artefact sert un second usage, lui aussi mesuré : `nearest_neighbours` retrouve les deux
autres écritures d'un fait, et la suite de tests l'affirme pour **toutes** les fiches du corpus
(336 voisins attendus, deux par fiche) — le notebook 04 en fait sa quatrième section. Aucun poids
pré-entraîné, aucun téléchargement, aucun appel réseau : l'index est appris sur le corpus, à graine
fixée, et le `n_components` publié est celui que la SVD a réellement produit, pas celui demandé.

### 1.11 `ai-eng/text-classification` — routage de tickets de support, deux stacks

Quatrième cas d'usage texte, et le plus proche d'un besoin de service : **attribuer** un ticket à son
catégorie. Le corpus synthétique compte 1 200 tickets, six catégories (facturation, livraison,
produit défectueux, remboursement, compte client et une classe « autre » sans vocabulaire propre,
12 % du corpus) et trois styles rédactionnels : 50 % des tickets emploient le vocabulaire de leur
catégorie, 30 % reformulent la même demande sans ce vocabulaire, 20 % ajoutent une politesse et une
signature partagées par toutes les classes. La priorité déclarée par le client est tirée
**indépendamment** du libellé : c'est un distracteur déclaré, mesuré par un test de raccourci. Enfin
chaque pool de formulations est coupé en deux, deux tournures par couple (classe, style) étant
réservées aux splits d'évaluation : une F1 élevée ne peut pas venir d'un gabarit mémorisé.

| Projet | Stack | Modèle | F1 macro (test) | Exactitude | Kappa | ECE | F1 canonique / paraphrase | Latence p95 | Ajustement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`with-tfidf_classifier`](ai-eng/text-classification/with-tfidf_classifier) | scikit-learn (TF-IDF, classification de texte) | régression logistique multinomiale sur TF-IDF | **0,8408** | **0,8531** | **0,8217** | 0,2363 | 0,9262 / 0,7352 | **0,78 ms** | 0,4 s |
| [`with-transformers`](ai-eng/text-classification/with-transformers) | Hugging Face Transformers (entraîné sur le corpus) | encodeur BERT minuscule (2 couches, 344 582 paramètres) | 0,7942 | 0,7853 | 0,7426 | **0,2099** | 0,7892 / **0,8285** | 2,22 ms | ≈ 150 s |

Les deux projets partagent le corpus, les six classes, le découpage (719 / 243 / 61 / 177 tickets) et
le seuil contractuel (`macro_f1 >= 0,70`) : l'écart entre les deux colonnes est donc un résultat, pas
un artefact de protocole. Le classifieur lexical apprend un poids par terme et par classe, expose ses
poids — chaque décision se relit terme à terme — et reste la meilleure moyenne du dépôt sur ce
corpus (0,8408). La variante neuronale apprend son vocabulaire WordPiece et ses poids **sur le
train** : pré-entraînement masqué sans aucun libellé, puis affinage supervisé avec
`class_weight='balanced'` ; aucun poids pré-entraîné n'est téléchargé, l'artefact est un `model.pt`
(joblib + JSON, aucun pickle Hugging Face) et l'explication d'une décision est une occlusion (le
terme affiché est retiré du texte, la baisse de probabilité est publiée).

**Lecture honnête** : la moyenne reste au lexical, mais le partage est instructif. Sur les tickets
écrits avec le vocabulaire de leur catégorie, le TF-IDF écrase l'encodeur (0,9262 contre 0,7892) ;
sur les **paraphrases**, l'encodeur gagne de 0,09 (0,8285 contre 0,7352) — un modèle contextuel ne
dépend plus des mots de la catégorie, et c'est précisément ce que la famille mesure. La même grille
d'architectures, mesurée sur le split de validation (719 tickets d'entraînement), départage les trois
modèles neuronaux : encodeur à deux couches **0,8022**, moyenne des vecteurs de termes 0,7455,
encodeur à quatre couches 0,0559 — cette dernière ne décolle pas, sur aucun des réglages testés
(zéro à cinq époques de masquage, taux d'apprentissage de 0,005 à 0,03), et le projet le publie au
lieu de le retirer. Le facteur décisif n'est donc pas la profondeur mais le **régime
d'entraînement** : un encodeur initialisé au hasard diverge au taux d'apprentissage qui fait
converger un sac d'embeddings, et les réglages gagnants de la grille sont devenus les défauts de la
stack. Coût assumé : trois secondes d'ajustement pour le sac d'embeddings (~18 000 paramètres)
contre environ deux minutes et demie pour l'encodeur servi (344 582 paramètres, mesuré entre 136 s et
150 s selon la charge de la machine).

### 1.12 `ai-eng/named-entity-recognition` — extraction d'entités dans les messages clients, spaCy

Cinquième cas d'usage texte, et le premier où le modèle ne classe pas un document : il **localise**
des éléments dans le texte. Cinq types d'entités (produit, référence de commande, montant, date,
transporteur) doivent être retrouvés avec leurs **bornes exactes** dans 1 200 messages de service
client synthétiques (4 454 entités, 21 tokens par message en moyenne), écrits dans deux styles
rédactionnels — 55 % rédigés (« CMD-1234 », « 12 mars 2025 ») et 45 % abrégés (« cmd 1234 », «
12/03/2025 ») — et arrivés par quatre canaux. Le découpage est 720 / 240 / 60 / 180 messages. Deux
difficultés sont construites : les surfaces de produits et de transporteurs des splits d'évaluation
sont **réservées** (12 produits et 4 transporteurs jamais vus à l'entraînement : une liste apprise
ne peut pas les retrouver), et les distracteurs sont **déclarés** (familles de produits, villes,
numéros de facture, années seules — présents dans les textes, jamais annotés).

| Projet | Stack | Modèle | F1 entité (test) | Précision / rappel | F1 macro | F1 partielle | Surfaces réservées / vues | Ajustement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`with-spacy`](ai-eng/named-entity-recognition/with-spacy) | spaCy (tok2vec + transitions, entraîné sur le corpus) | **hybride** : tagger appris, corroboré par une couche de règles | **0,9340** | 0,9521 / 0,9167 | **0,9239** | 0,9735 | 0,6056 / 1,0000 | ≈ 95-103 s |

Trois références mesurées sur le même split de test (180 messages, 672 entités) donnent un sens aux
0,9340 : le plancher trivial (n'annoter aucune entité) vaut **0,0**, la couche de règles du projet —
motifs de référence, de montant et de date plus un index de surfaces appris **sur le train
uniquement** — atteint **0,8710** (précision 0,9725, rappel 0,7887), et le raccourci de forme («
telle signature de caractères veut dire tel type », appris sur le train) atteint 1,0 sur les trois
types à motif, 0,9265 sur `produit` et 0,9351 sur `transporteur` — mais seulement quand les bornes
lui sont données. Le modèle apporte **+0,0630** de F1 sur la couche de règles, et il **généralise**
là où l'index s'arrête — c'est la raison pour laquelle le tagger est servi ; la couche de règles,
elle, reste dans le rendu hybride pour ce qu'elle apporte en plus du score : une **provenance** et
un taux de corroboration par mention.

**Lecture honnête** : trois chiffres racontent le projet. (1) Le tagger retrouve **100 %** des
mentions dont la surface a été vue à l'entraînement et **60,6 %** de celles dont la surface était
réservée — cet écart de 0,3944 est la seule mesure du dépôt qui distingue « apprendre la forme d'un
nom » de « recopier un vocabulaire », et il est publié dans les deux sens. (2) Le type difficile est
`transporteur` (F1 0,713 : précision 0,828, rappel 0,626) parce que ses surfaces n'ont pas de
préfixe reconnaissable — à l'inverse `commande`, `date` et `montant` sont lus au sans-faute par les
deux couches (F1 1,0). La ventilation le montre au passage : les messages abrégés ne sont pas les
plus durs (0,943 contre 0,926 pour les rédigés), et le canal le plus difficile est le formulaire
(0,923) — un résultat publié, pas une intuition. (3) La confiance publiée n'est **pas** une
probabilité : un tagger à transitions n'en expose aucune, donc chaque mention porte son **taux de
corroboration** par les règles, dont la précision est mesurée par niveau — 0,973 pour les mentions
corroborées (545 sur 647), 0,843 pour les autres, écart de 0,1396 sur les mentions correctes.
Publier une fausse probabilité aurait été plus simple et faux ; le projet publie une mesure, avec
son mode d'emploi (seuil d'automatisation arbitré sur le split de calibration et relu sur la
validation, jamais sur le test).

La stack spaCy sert **trois algorithmes** interchangeables par `model.algorithm` : `gazetteer` (la
couche de règles seule, sans apprentissage de poids), `tagger` (tok2vec + classifieur à transitions,
`spacy.blank("fr")`, aucun poids pré-entraîné téléchargé) et `hybrid` (servi par défaut). Le projet
est structurellement le plus riche du dépôt : l'annotation est un **couple de tables** validé par
Pandera (`surface == text[start:end]`, pas de chevauchement, mention rattachée au split de son
message), l'entraînement commence par l'**alignement des offsets** sur les tokens (100 % des
annotations apprennent, 0 ignorée — un corpus décalé produirait un score moyen sans que personne ne
sache pourquoi), et l'artefact est un **répertoire** (`nlp.to_disk`) rechargé à l'identique, ce que
la suite de tests vérifie. Coût assumé : environ une minute et demie d'entraînement (30 époques sur
720 messages) et 3 à 5 s d'évaluation, pour une médiane de 2,5 à 3,1 ms par message en inférence
(p95 de 3,0 à 4,4 ms selon la charge — la latence est une mesure, pas un score, et le projet ne la
fige pas) ; les six notebooks tournent sur un corpus réduit (320 messages) et re-dérivent leurs
propres chiffres.

### 1.13 `nlp/summarization` — résumé de comptes-rendus d'intervention, encodeur-décodeur appris

Sixième famille texte, et la première du répertoire `nlp/` : il ne s'agit plus de **retrouver**
l'information (RAG, questions-réponses, entités) mais de la **produire**. 600 comptes-rendus
d'intervention techniques synthétiques (126,8 tokens par document pour 52,5 par résumé de référence,
sept types de faits annotés : équipement, symptôme, cause, action, pièce, durée, statut — 5 036 faits
dont 3 573 saillants) doivent être résumés en quelques phrases **fidèles** : un résumé fluide qui
oublie la durée d'une intervention est un mauvais résumé, et il est compté comme tel. Les résumés de
référence sont écrits à partir des faits saillants avec **50 % de réécriture** — recopier les
phrases du document ne suffit donc pas — et le split est 360 / 120 / 60 / 60.

| Projet | Stack | Modèle | ROUGE-1 F1 (test) | ROUGE-2 | ROUGE-L | Couverture des faits | Précision | Compression | Ajustement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`with-seq2seq`](nlp/summarization/with-seq2seq) | `seq2seq` (PyTorch, encodeur-décodeur écrit dans le projet) | **encodeur-décodeur transformer** : 2 couches, 128 unités, 4 têtes, vocabulaire WordPiece appris sur le train | **0,6958** | 0,5496 | 0,4616 | **0,6145** | 0,9794 | 0,4304 | 170 s (12 époques) |

Trois références mesurées sur les **mêmes 60 documents de test** donnent son sens au 0,6958 : le
résumé vide vaut **0,0** (un score défini, pas une absence), la baseline extractive publiée `lead`
(les premières phrases du document, même budget) atteint **0,4351** avec une couverture de 0,4915, et
`textrank` (graphe de phrases + diversification MMR) **0,3888** pour 0,3862. Le modèle apporte donc
**+0,2607** de ROUGE-1 sur la référence publiée, et **+0,1230** de couverture des faits sur la
meilleure extractive — deux gains qui ne vont pas de soi : un générateur peut améliorer la forme
sans améliorer la fidélité, et le rapport publie les deux colonnes côte à côte pour que ce cas soit
visible.

**Lecture honnête.** (1) La fidélité est mesurée, pas espérée : **0,6145** de couverture en moyenne,
et seulement **8,3 %** des résumés contiennent une valeur absente du document (précision 0,9794 —
c'est la seule métrique du projet qui compte les inventions). (2) Le type de fait qui résiste est
`piece` (couverture **0,0000**), suivi de `action` (0,1667) : ce ne sont pas des types marginaux
(37,2 % des documents portent un fait de type `piece`, 94,5 % un fait de type `action`), mais les
seuls dont la valeur est un identifiant consommé (`CRR-241`) ou une opération parmi douze — le modèle
a appris les gabarits fréquents (`cause` 0,9167, `statut` 0,8167) et pas encore la recopie d'un
identifiant rare. C'est le premier poste à travailler,
et il est publié tel quel. (3) Le budget de longueur se lit avec le ROUGE : **58,3 %** des résumés
atteignent leur borne (90 tokens), pour une compression moyenne de 0,4304 — la marge de progression
est dans le critère d'arrêt du décodeur autant que dans le modèle. Le projet publie aussi la
**latence** (p50 181,1 ms, p95 223,4 ms par document sur la machine de référence) : une mesure, pas un
score — elle dépend de la machine, jamais figée, et c'est écrit dans le manifeste.

La stack `seq2seq` sert **trois algorithmes** interchangeables par `model.algorithm` :
`transformer_tiny` (appris, servi), `textrank` et `lead` (extractifs, publiés comme références).
Rien n'est téléchargé : le vocabulaire WordPiece est **appris sur le split d'entraînement par le
projet** (879 pièces, 797 fusions, taux de jetons inconnus 0,0000) et les poids sont initialisés au
hasard, pré-entraînés par débruitage (2 époques) puis affinés en enseignant-forcé (12 époques,
loss 0,810) — 1 151 599 paramètres. Le décodage est **glouton** : un résumé doit être reproductible
ligne à ligne, ce que la suite de tests vérifie en réentraînant deux fois le même modèle, et un test
dédié vérifie que deux vocabulaires appris sur le même corpus sont identiques (identifiants
compris) — l'apprenti de la bibliothèque `tokenizers` départageait les fréquences égales par une
table de hachage interne, donc deux exécutions produisaient deux vocabulaires, et le projet a écrit
le sien pour rendre le résultat reproductible.

**Ce que la tranche a appris (et corrigé).** Le premier run de référence mesurait 0,3358 de
couverture : les identifiants du corpus (`P-12`, `CBL-045`) étaient découpés en `[UNK]` par le
tokenizer, donc le modèle ne pouvait pas recopier une pièce ou un équipement même quand il le
voulait. Le vocabulaire a été appris sur la ponctuation comme **segments à part entière**
(`BertPreTokenizer` côté encodage, recollage explicite au décodage) : le même modèle, le même corpus,
seuls le vocabulaire et le décodage ont changé, et la couverture est passée de 0,3358 à **0,6145**
(ROUGE-1 de 0,6876 à 0,6958). C'est l'unique chiffre du dépôt qui montre le prix d'un
pré-traitement : un caractère mal découpé coûte la moitié d'une métrique de fidélité.

Les six notebooks tournent sur un corpus réduit (200 comptes-rendus) et un bac à sable complet
(`outputs/notebooks/` : le corpus **et** les artefacts du notebook), donc ils ne peuvent pas écraser
les chiffres publiés par `make train` / `make evaluate` ; ils re-dérivent leurs propres scores et
publient leurs recommandations chiffrées. Coût assumé : environ deux minutes d'entraînement sur le
corpus complet.

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

Cinq stacks texte complètent la liste, documentées avec leurs projets : **scikit-learn
(TF-IDF / BM25)**, **LangChain (LCEL)**, l'**index vectoriel dense** (hachage de n-grammes puis SVD
tronquée, stack `embedding`), **spaCy** (tagger à transitions entraîné sur le corpus, corroboré par
une couche de règles) et le **encodeur-décodeur `seq2seq`** (transformer écrit dans le projet,
vocabulaire WordPiece appris et décodage glouton) — même corpus annoté d'un côté, corpus annoté au
caractère de l'autre, texte résumé de bout en bout du troisième (sections 1.8 à 1.13).

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
│   │   ├── stacks.yaml           # 22 stacks : dépendances, classe, format du fichier modèle, docs
│   │   └── families.yaml         # familles de problèmes : modalité, tâche, builder de notebooks
│   ├── defaults/
│   │   ├── global.yaml           # valeurs par défaut de tous les projets (train, preprocessing…)
│   │   └── family/<famille>.yaml # cas d'usage : données, métriques, seuils de qualité
│   ├── manifests/*.yaml          # UN manifeste = UN projet livré
│   ├── notebooks/
│   │   ├── tabular.py            # six notebooks des familles tabulaires
│   │   ├── text.py               # six notebooks des familles texte (RAG, questions-réponses)
│   │   ├── nel.py                # six notebooks de l'extraction d'entités nommées
│   │   └── summarization.py      # six notebooks du résumé (bac à sable et stratégies)
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

État au dernier passage : **35 projets**, dont **29 rejoués de bout en bout dans l'environnement de
cette branche** (ruff check, ruff format, mypy strict, pytest, six notebooks exécutés,
`python -m src.main mode=all`) — la barrière de qualité a d'ailleurs tenu sur les 35 : `ruff check`,
`ruff format` et `mypy` sont verts partout, y compris pour les six projets que l'environnement ne
peut pas exécuter faute de TensorFlow.

Les **dix projets texte** sont conformes : `rag/with-tfidf` (118,4 s, 107 tests),
`rag/with-langchain` (203,4 s, 112 tests), `question-answering/with-tfidf` (69,2 s, 107 tests),
`question-answering/with-langchain` (105,0 s, 112 tests), `embeddings/with-embedding` (61,3 s,
109 tests), `embeddings/with-tfidf` (59,2 s, 107 tests), `text-classification/with-tfidf_classifier`
(48,5 s, 121 tests), `text-classification/with-transformers` (893,0 s, 127 tests — l'encodeur
s'entraîne dans les notebooks, d'où la durée), `named-entity-recognition/with-spacy` (201,6 s,
101 tests) et `summarization/with-seq2seq` (50 tests, six notebooks en 108 s, `mode=all` à ROUGE-1
0,6958 — le vocabulaire et les poids s'apprennent sur le corpus, d'où la durée) — mêmes six
notebooks et même pipeline complet. Les **19 projets tabulaires**
scikit-learn, XGBoost, LightGBM et PyTorch rejoués dans le même environnement sont conformes eux
aussi (la référence multi-stacks complète, 4 235 tests, reste celle du dernier passage intégral) ;
les quatre projets **Keras** et les deux **TensorFlow** n'ont pas été rejoués ici — TensorFlow n'est
pas installé — mais leurs sources sont inchangées (seuls le README et `pyproject.toml` ont été
régénérés, et la barrière de qualité y passe). Chaque projet est rejoué intégralement — lint,
typage, tests, exécution des six notebooks et pipeline complet — avant d'être considéré comme
livré.

Les notebooks sont versionnés **sans outputs** : ils sont rejoués par `tools/verify.py`, le dépôt
reste léger et leur exécution reste une preuve vérifiable plutôt qu'une capture d'écran.

**Trou connu du crible de lint (mesuré).** Les projets déclarent
`extend-exclude = ["outputs", "multirun", "artifacts", "data", ".venv"]` : le motif `data` n'est pas
ancré, il exclut donc aussi `src/data/`. Conséquence : `ruff check .` et `ruff format .` sont muets
sur les générateurs, les schémas Pandera et les chargeurs — le dossier le plus dense de chaque
projet. Après `ruff format` + `ruff check --fix` appliqués à ces fichiers, il reste **388 erreurs**
sur les 35 projets (dont **294 E501**), concentrées dans `src/data/{schemas,generators,loaders}.py`.
Réparer les gabarits `task/*`/`family/*`/`modality/*` de `src/data/`, puis ancrer les motifs
(`/data`), doit se faire **en une seule tranche** : ancrer d'abord ferait passer 35 projets au rouge
d'un coup. Le contrôle des 35 projets porte donc, aujourd'hui, sur tout sauf `src/data/`.

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
**22 stacks**. Les couches `base/`, `modality/{tabular,text}/`,
`task/{classification,multiclass,regression,clustering,anomaly,forecasting,ranking,retrieval,text_multiclass,named_entity_recognition,summarization}/`,
`family/{binary_classification,multiclass_classification,regression,clustering,anomaly_detection,time_series_forecasting,recommendation,retrieval_augmented_generation,question_answering,embedding_pipeline,text_classification,named_entity_recognition,summarization}/`
et `stack/{sklearn,xgboost,lightgbm,pytorch,tensorflow,keras,tfidf,tfidf_classifier,langchain,embedding,transformers,spacy,seq2seq}/`
sont écrites et vérifiées : les **sept familles tabulaires**, la **famille RAG** (section 1.8), la
**famille questions-réponses** (section 1.9), la **famille d'index d'embeddings** (section 1.10), la
**famille de classification de texte** (section 1.11), la **famille d'extraction d'entités nommées**
(section 1.12) et la **famille de résumé** (section 1.13) sont livrées — neuf projets `ai-eng` et un
projet `nlp`, tous mesurés sur un corpus synthétique et un split de test dédié. Ce qui reste, par
ordre de valeur pédagogique :

1. **Stacks tabulaires déclarées, pas encore livrées** — `clustering` et `anomaly_detection` en
   PyTorch et TensorFlow (auto-encodeurs), `recommendation` en PyTorch (modèle à deux tours),
   `binary_classification` avec MLflow. Leurs couches `task/` et `family/` existent ; il reste un
   manifeste par stack et, comme pour la prévision, des notebooks paramétrés par stack
   (`extras.notebook_<famille>`) : les commentaires chiffrés d'un notebook doivent parler de la
   stack réellement mesurée.
2. **Suite du périmètre ai-eng et nlp** — la modalité texte, les tâches `retrieval`,
   `text_multiclass`, `named_entity_recognition` et `summarization` et les familles
   `retrieval_augmented_generation`, `question_answering`, `embedding_pipeline`,
   `text_classification`, `named_entity_recognition` et `summarization` sont en place (sections 1.8
   à 1.13) : la suite réutilise ces couches. Restent `ai-eng/` (agents : Deepagents, Strands, CrewAI ;
   serving FastAPI), `nlp/` (fine-tuning LLM) et les familles `llm_finetuning` et `agent_tools` —
   cette dernière réutilise `modality/text/` + `task/retrieval/` comme l'a fait
   `embedding_pipeline`.
3. **Autres modalités** — `data-eng/` (pandas + PyArrow, DuckDB, Prefect), `mlops/` (MLflow,
   GitHub Actions + tox), `analytics/` (rapports Jinja2, monitoring de drift SciPy),
   `computer-vision/`. Chacune demande une nouvelle couche `modality/` (loaders, pré-traitement,
   pipelines, tests) en plus des couches `family/` et `task/`.
4. **Stacks déjà déclarées, non implémentées** — `duckdb`, `pandas`, `prefect`, `mlflow`,
   `fastapi`, `jinja2`, `scipy`, `github-actions`, `pandera` : l'entrée de registre existe, le dossier
   `templates/stack/<clé>/` reste à écrire ; elles serviront `data-eng/`, `mlops/` et le serving
   (`spacy` est livrée avec la section 1.12).

Chaque ajout suit la même procédure : entrée de registre -> templates de stack ou de famille ->
manifeste -> `build` -> `verify` -> commit.

**5. Réparer le crible de lint sur `src/data/`** — 388 erreurs mesurées (294 E501, 62 RUF046,
25 RUF012, 6 F841, le reste en pièces détachées) après `ruff format` + `ruff check --fix`, dans les
gabarits `src/data/{schemas,generators,loaders}.py` des couches `task/`, `family/` et `modality/`.
Trois gestes, dans cet ordre : replier la prose des gabarits (`python -m tools.wrapdoc`, livré),
réparer les longues lignes de code, puis **ancrer** les motifs d'exclusion (`/data`) et rebâtir les
35 projets. L'ancrage seul transforme un contrôle incomplet en 35 projets rouges ; c'est pourquoi il
va avec la réparation, pas avant elle.
