# Famille `multiclass_classification` — diagnostic de mode de défaillance (sous-projet 1 : sklearn)

- **Date** : 2026-09-28
- **Statut** : design validé section par section, en attente de relecture de la spec
- **Branche** : `feat/multiclass-classification`
- **Périmètre** : sous-projet 1 sur 2. Le sous-projet 2 (stacks XGBoost, LightGBM, PyTorch, Keras
  de la même famille) fera l'objet de sa propre spec.

## 1. Objectif

Livrer le point 1 de la feuille de route du README racine : la famille tabulaire
`multiclass_classification`, déclarée dans `tools/scaffold/registry/families.yaml` mais sans
implémentation. Le livrable est un 17e projet, `data-science/multiclass-classification/with-sklearn`,
qui satisfait la définition de « livré » du dépôt (`tools/verify.py` : ruff, ruff format, mypy,
pytest, six notebooks exécutés, `python -m src.main mode=all`).

Le README racine est resynchronisé au passage : il annonce 15 projets alors que 16 sont livrés
(`recommendation/with-sklearn` n'y figure pas).

### Critères de réussite du sous-projet

1. `python -m tools.scaffold.build --manifest tools/scaffold/manifests/ds-multiclass-sklearn.yaml`
   génère le projet sans avertissement.
2. `python -m tools.verify data-science/multiclass-classification/with-sklearn --notebooks-inplace`
   est vert sur les six étapes.
3. Les 16 projets existants sont régénérés ; leur diff ne touche que les fichiers partagés listés
   en §6. `tools.verify` complet est vert sur `classification/with-sklearn` (témoin binaire), et
   `tools.verify --quick` est vert sur les 15 autres (stacks deep learning comprises, installées
   dans le venv local).
4. Le rapport d'évaluation du nouveau projet rend un verdict **conforme** sur ses objectifs
   contractuels, avec des seuils calibrés sur des mesures et non choisis a priori.
5. Le README racine décrit 17 projets avec des chiffres mesurés, dont ceux de la recommandation
   obtenus en rejouant son pipeline.

### Constat de départ (audit du générateur)

- Le mapping `multiclass → task/classification` existe (`engine.py:_task_dir`), mais un manifeste
  multiclasse génère aujourd'hui **un projet cassé sans erreur** : la couche `family/` absente est
  ignorée silencieusement, donc aucun `src/data/generators.py` n'est rendu. Ensuite
  `Evaluator._curves` plante (`roc_curve` sur labels multi-classes).
- `task/classification/` est binaire presque partout : `probabilities[:, 1]`, `astype(int)`,
  colonnes churn codées en dur, seuil et segments de risque dans le predictor, conseils
  « classe positive » dans le rapport.
- Stacks : sklearn, PyTorch et Keras gèrent déjà le multi-classes. LightGBM demande des métriques
  `multi_logloss`. XGBoost n'encode pas les labels texte (sous-projet 2).
- `losses_metrics.py` : `roc_auc_ovr` absente ; `_log_loss` utilise `np.unique(y_true)` comme
  labels, donc NaN dès qu'un split manque une classe ; `mcc` absent de
  `METRICS_BY_TASK["multiclass"]`.

## 2. Décisions d'architecture

| Décision | Choix | Raison |
| --- | --- | --- |
| Couche tâche | **nouvelle `task/multiclass/`**, alias `multiclass → multiclass` dans `engine.py` | Suit le motif du dépôt (une couche par sémantique de tâche : `anomaly`, `ranking`, `forecasting`). Zéro risque sur les 6 projets binaires. Réutilisable par `text_classification` et `image_classification` (task `multiclass` au registre). |
| Labels | **texte** (`tool_wear`, `heat_dissipation`…) | Rapports et notebooks lisibles côté métier ; sklearn, LightGBM, PyTorch, Keras les gèrent nativement. L'encodage XGBoost relève du sous-projet 2. |
| Réglages métier | bloc `diagnosis:` injecté dans `conf/config.yaml` via `extras.root_conf` | Même mécanisme que le bloc `recommendation` ; `AppConfig` est en `extra="allow"`. |
| Références | `generation_metadata.json` écrit par le générateur dans `data/raw`, lu par l'évaluateur | Même mécanisme que `task/ranking` (`generation_reference()`). |
| `problem` du projet | `multiclass-classification` | Évite la collision de chemin avec `data-science/classification/with-*`. |

Options écartées : rendre `task/classification/` task-aware (branches Jinja partout, régénération
des 6 projets binaires avec preuve de diff nul, logique de décision différente : seuil contre
argmax/coût) ; surcharger dans `family/` (la logique tâche serait dupliquée par chaque future
famille multi-classes).

## 3. Cas d'usage et données

### Métier

Ligne d'usinage CNC (fraisage). Chaque alarme automate déclenche une intervention. L'équipe envoyée
(mécanique, thermique, électrique, outillage) est aujourd'hui choisie d'après le **code alarme**
de l'automate, qui se trompe souvent : mauvaise équipe, second déplacement, arrêt prolongé.
Objectif : prédire le **mode de défaillance** à partir des capteurs au moment de l'alarme, pour
router vers la bonne équipe ou demander une expertise humaine quand le modèle doute.

### Jeu `machine_failure_diagnosis`

6 000 alarmes, une ligne par alarme, split stratifié 65/15/20, graine 42.

| Colonne | Type | Rôle | Remarque |
| --- | --- | --- | --- |
| `alarm_id` | str | identifiant | `^ALM-[0-9]{5}$`, unique |
| `alarm_timestamp` | datetime | horodatage | |
| `product_quality` | category | feature | L / M / H (~50 / 30 / 20 %) |
| `production_line` | category | feature | signal faible |
| `shift` | category | feature | day / evening / night |
| `alarm_code` | category | feature | familles E1xx thermique, E2xx électrique, E3xx mécanique, E9xx générique |
| `air_temperature_k` | float | feature | |
| `process_temperature_k` | float | feature | ≈ air + 10 K |
| `rotational_speed_rpm` | float | feature | |
| `torque_nm` | float | feature | |
| `tool_wear_min` | int | feature | dépend de la gamme |
| `coolant_flow_l_min` | float | feature | |
| `vibration_mm_s` | float | feature | nullable, ~4 % de manquants (capteur absent) |
| `hours_since_maintenance` | float | feature | |
| `failure_mode` | str | **cible** | 6 classes |

Les bornes Pandera (`checks`) de chaque colonne sont fixées à l'écriture du générateur et
publiées dans `defaults/family/multiclass_classification.yaml`.

### Classes

| Classe | Part visée | Mécanisme latent |
| --- | --- | --- |
| `false_alarm` | ~30 % | capteurs en zone normale, code souvent E9xx |
| `tool_wear` | ~20 % | usure outil élevée |
| `heat_dissipation` | ~18 % | Δ température < ~8,6 K, vitesse basse, débit de refroidissement faible |
| `power_failure` | ~14 % | puissance ∝ couple × vitesse hors plage |
| `overstrain` | ~12 % | usure × couple au-delà d'un seuil qui dépend de la gamme L/M/H |
| `random_failure` | ~6 % | aucun signal : plafond structurel |

Paires confusables voulues : `tool_wear` ↔ `overstrain` (usure), `heat_dissipation` ↔
`power_failure` (vitesse basse).

### Génération

1. Conditions d'exploitation tirées d'un mélange « normal » + « stressé sur un axe caché »
   (refroidissement, puissance, usure, couple).
2. Classe tirée d'un **softmax de logits physiques connus**. Les intercepts sont calibrés
   itérativement (`intercept += log(part visée / part observée)`) pour tenir les parts visées.
   Le bruit est irréductible et le générateur connaît les vraies probabilités.
3. `alarm_code` est dérivé de la vraie classe avec une matrice de confusion bruitée : c'est la
   règle de routage actuelle, référence métier à battre.
4. Valeurs manquantes injectées sur `vibration_mm_s`.

`generation_metadata.json` publie :

- les parts de classes observées ;
- le **plafond oracle** : accuracy, macro-F1 et log loss de l'argmax des vraies probabilités. L'oracle
  ne prédit jamais `random_failure` (F1 = 0), d'où un macro-F1 plafonné sous 1, discuté
  explicitement dans le rapport et le notebook 06 ;
- le **plancher** de la classe majoritaire ;
- la **référence métier** : accuracy et macro-F1 du routage par `alarm_code`.

Contrat du générateur, identique aux autres familles :
`SyntheticDataGenerator(n_samples, seed, dataset_name, output_dir, formats, **options)`,
`SUPPORTED_OPTIONS`, `from_config`, `generate`, `export`, `run`, `sample(n, with_target=False)`,
`metadata`, `DEFAULT_DATASET_NAME`. La sortie satisfait `RawDataSchema`.

### Features déclarées en configuration

Types existants de `build_features.py` : `difference` pour `temp_delta` (process − air), `product`
pour `power_proxy` (couple × vitesse) et `wear_torque` (usure × couple). Le prétraitement garde
l'encodage `onehot`.

## 4. Couche `task/multiclass/`

### Contrat imposé par les pipelines partagés (inchangés)

- `Evaluator(model, metrics_config, paths, task).evaluate(X, y, split, context) -> EvaluationResult`
  (`metrics`, `n_samples`, `extras`) ;
- `ReportBuilder(paths, config).build(evaluation, model=..., thresholds=None) -> dict[str, Path]` ;
- `Predictor.from_config(config, paths)`, `predict(frame)`, `load_inputs(path)`,
  `sample_inputs(n)` ;
- une classe de figures exportée par `src/visualization`.

`evaluation_pipeline.py` n'appelle `threshold_analysis` que pour `task == "binary"` : aucun
changement requis, `thresholds` vaut `None` en multi-classes.

### Bloc de configuration `diagnosis`

| Clé | Contenu |
| --- | --- |
| `class_order` | ordre canonique des 6 classes (tables, figures) |
| `failure_classes` | classes « panne réelle » (toutes sauf `false_alarm`) |
| `team_by_class` | classe → équipe envoyée |
| `cost_matrix` | vraie classe × classe prédite → coût en EUR |
| `review_threshold` | confiance sous laquelle l'alarme part en revue experte |
| `alarm_code_rule` | famille de code → classe (routage actuel) |
| `objectives` | seuils des objectifs contractuels |

Un bloc invalide (classe inconnue, matrice non carrée, coût négatif) lève une erreur explicite au
démarrage.

### `src/evaluation/evaluator.py`

- Métriques globales via `MetricCalculator` : `f1_macro` (primaire), `balanced_accuracy`,
  `accuracy`, `mcc`, `log_loss`, `roc_auc_ovr`, `f1_weighted`, `precision_macro`, `recall_macro`.
- Table par classe : précision, rappel, F1, support, AUC one-vs-rest.
- Matrice de confusion brute et normalisée par ligne, ordonnée par `model.classes_` (jamais par
  tri de chaînes). Paires confondues classées par volume et par part de la vraie classe.
- Références sur le même test : classe majoritaire, routage `alarm_code` (calculé sur le contexte
  du test), plafond oracle lu dans `generation_metadata.json`.
- Décision : coût attendu par alarme en argmax contre la règle de Bayes à coût minimal
  (`argmin_j Σ_i p_i · C[i, j]`), et taux de pannes réelles classées `false_alarm`.
- Abstention : courbe couverture × exactitude selon le seuil de confiance ; au
  `review_threshold`, part envoyée en revue et exactitude du flux automatisé.
- Calibration top-label : diagramme de fiabilité et ECE.
- Verdict : chaque objectif contractuel avec valeur mesurée, seuil et statut.

Garde-fous : un label de test inconnu du modèle lève une erreur explicite ; une classe absente du
split ne produit pas de NaN silencieux (les métriques reçoivent `classes_`).

### `src/evaluation/reports.py.j2`

Rapport Markdown : verdict ; métriques globales face aux trois références ; table par classe ;
lecture de la matrice (quelles confusions, pourquoi physiquement) ; décision coût-sensible ;
abstention ; calibration ; erreurs par segment (`product_quality`, `alarm_code`) ; limite
structurelle de `random_failure` ; recommandations.

### `src/inference/predictor.py.j2`

Colonnes de sortie, une ligne par alarme : `alarm_id`, `predicted_mode` (argmax),
`proba_<classe>` pour chaque classe, `confidence` (probabilité maximale), `margin` (écart entre
les deux meilleures), `second_choice`, `decision` (classe à coût minimal), `needs_review`
(`confidence < review_threshold`), `routed_team`, `decision_reason` (phrase en français).
`predict_one` retourne le même contenu pour un dict. Pas de seuil binaire, pas de segments de
risque. Les entrées de fichier sont validées par `InferenceDataSchema`.

### `src/visualization/plots.py`

`MulticlassPlots` : matrice normalisée (sûre si une classe n'a aucune ligne), F1 et rappel par
classe face aux références, ROC one-vs-rest, diagramme de fiabilité, histogramme de confiance
correct/erroné, courbe couverture-exactitude, barres de métriques (axe qui tolère un MCC négatif).
Figures écrites dans `artifacts/figures`.

## 5. Notebooks

Nouveau module `tools/scaffold/notebooks/multiclass.py`, sur le modèle de `notebooks/ranking.py`
(réutilise `LOAD_RAW`, `PREPARE`, `SETUP`, `_code`, `_md`, `_insight`, `_objectives`,
`_replace_placeholder`), branché dans `tabular.py` par un prédicat `_is_multiclass`.

| Notebook | Contenu |
| --- | --- |
| 01 §5 | Distribution des classes ; classe × gamme ; capteurs par classe ; nuage Δ température × vitesse coloré par mode ; table `alarm_code` × vrai mode (pourquoi le routage actuel se trompe). |
| 02, 03, 05 | Génériques, repris tels quels ; toute cellule binaire rencontrée à l'exécution est gardée par le prédicat. |
| 04 | Références d'abord (majoritaire, `alarm_code`, oracle) ; modèle linéaire avec et sans features physiques ; forêt ; HGB ; effet de `class_weight` sur macro-F1 contre accuracy ; stabilité entre graines ; petite grille. |
| 06 | Verdict de l'évaluateur de production ; paires confondues lues physiquement ; argmax contre coût minimal ; choix du seuil de revue sur la courbe d'abstention ; calibration ; plafond de `random_failure` ; plan d'action. |

## 6. Changements partagés (impact sur les 16 projets existants)

Tous additifs, comportement binaire inchangé.

| Fichier | Changement |
| --- | --- |
| `tools/scaffold/engine.py` | alias `multiclass → multiclass` |
| `modality/tabular/src/training/losses_metrics.py` | `roc_auc_ovr` ; `_log_loss` et `roc_auc_ovr` reçoivent `extra["classes"]` (repli `np.unique(y_true)`) ; `mcc` dans `METRICS_BY_TASK["multiclass"]` |
| `modality/tabular/tests/test_models.py` | casts `float64` limités aux tâches numériques |
| `modality/tabular/tests/test_loaders.py` | stratification comparée par proportion de chaque classe |
| `tools/scaffold/notebooks/tabular.py` | `_is_multiclass` et aiguillages 01 / 04 / 06 |
| `base/README.md.j2` | leçons 04 / 05 / 06 et figures propres au multi-classes |

Le point d'entrée des classes du modèle vers `MetricInputs.extra["classes"]` est tracé pendant le
plan (Trainer, `BaseModel.training_metrics`, `Evaluator`), en réutilisant
`metric_extra_from_config` quand c'est possible.

## 7. Nouveaux fichiers

- `tools/scaffold/defaults/family/multiclass_classification.yaml`
- `tools/scaffold/templates/family/multiclass_classification/src/data/generators.py.j2`
- `tools/scaffold/templates/task/multiclass/src/evaluation/{__init__.py,evaluator.py,reports.py.j2}`
- `tools/scaffold/templates/task/multiclass/src/inference/{__init__.py,predictor.py.j2}`
- `tools/scaffold/templates/task/multiclass/src/visualization/{__init__.py,plots.py}`
- `tools/scaffold/notebooks/multiclass.py`
- `tools/scaffold/manifests/ds-multiclass-sklearn.yaml`
- `data-science/multiclass-classification/with-sklearn/` (projet généré)

Tests spécifiques à la tâche ajoutés dans le projet généré : évaluateur (références, décision à
coût minimal, abstention, classe absente), predictor (colonnes, probabilités sommant à 1,
`needs_review`), générateur (parts de classes, métadonnées, déterminisme), bloc `diagnosis`
invalide.

## 8. Critères métier du projet livré

Seuils fixés **après** la première mesure (jamais inventés), inscrits dans le bloc
`diagnosis.objectives` et dans les `success_criteria` de la famille :

1. Macro-F1 au-dessus d'un seuil fixé sous le plafond oracle.
2. Battre la classe majoritaire et le routage par `alarm_code` en macro-F1.
3. Rappel ≥ seuil sur chaque mode réel prédictible (tous sauf `random_failure`).
4. Taux de pannes réelles classées `false_alarm` sous un plafond.
5. ECE top-label < 0,08.
6. Coût attendu par alarme inférieur à celui du routage actuel.
7. Inférence < 50 ms pour 1 000 alarmes.
8. Reproductibilité : même graine, mêmes métriques.

L'algorithme livré (HGB ou forêt aléatoire) est choisi sur les résultats du notebook 04.

## 9. Vérification et livraison

1. Venv local via `uv` (`.venv/`, déjà ignoré par git) : dépendances du dépôt, de la stack
   sklearn, puis xgboost, lightgbm, torch CPU, tensorflow-cpu pour vérifier les 16 projets.
2. Mesure de référence **avant** tout changement : `tools.verify --quick` sur les 16 projets, pour
   distinguer une régression d'un état déjà rouge.
3. Implémentation, génération du nouveau projet, `tools.verify` complet.
4. Régénération des 16 projets, lecture du diff, `tools.verify` complet sur
   `classification/with-sklearn`, `--quick` sur les 15 autres.
5. Rejeu du pipeline recommandation pour les chiffres du README.
6. README racine : 17 projets, section 1.6 recommandation, section 1.7 multi-classes, feuille de
   route à jour (la modification locale de la ligne `ai-eng` est conservée).
7. Commits locaux sur la branche. **Aucun push sans accord explicite.**

## 10. Risques

| Risque | Mitigation |
| --- | --- |
| Régression silencieuse d'un projet existant par les changements partagés | Mesure de référence avant changement ; diff de régénération lu fichier par fichier ; verify sur les 16. |
| Plafond oracle trop bas ou trop haut (jeu trivial ou impossible) | Calibration des logits sur mesures ; cible : écart oracle − majoritaire assez large pour que les modèles se distinguent. |
| Macro-F1 pénalisé par `random_failure` jugé comme un défaut | Plafond publié et expliqué ; objectif de rappel par classe excluant `random_failure`. |
| Notebooks génériques 02/03/05 contenant du code binaire non repéré | Exécution réelle par `tools.verify` ; garde par `_is_multiclass` au besoin. |
| Durée d'installation (≈ 2 Go de dépendances deep learning) | Installation en arrière-plan pendant le travail sur le générateur. |
| Temps d'exécution des notebooks | Grilles petites, volume maîtrisé ; mesure du temps dans verify. |
