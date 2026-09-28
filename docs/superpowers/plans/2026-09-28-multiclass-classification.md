# Famille multiclass_classification — plan d'implémentation

> Exécution native (l'utilisateur a demandé une exécution autonome). Cases à cocher suivies au fil de l'eau.

**Goal:** livrer `data-science/multiclass-classification/with-sklearn` conforme à `tools/verify.py`,
sans régression sur les 16 projets existants, puis enchaîner les autres stacks de la famille.

**Architecture:** nouvelle couche `templates/task/multiclass/` (alias `multiclass → multiclass`),
couche `family/multiclass_classification/` (générateur), module `notebooks/multiclass.py`,
changements additifs dans `modality/tabular`.

**Tech Stack:** Jinja2, pydantic, Hydra, Pandera, scikit-learn, nbformat/nbclient, ruff, mypy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-28-multiclass-classification-design.md`

## Global Constraints

- Code, identifiants, docstrings : anglais ; prose, README, notebooks, rapports : français.
- ruff (line-length 100) + mypy strict + pytest verts dans chaque projet généré.
- Aucune valeur métier codée en dur : tout passe par la config Hydra (`diagnosis:`, `data:`, `model:`).
- Déterminisme : graines partout ; fins de ligne LF quel que soit l'OS.
- Commandes lancées avec `PATH=.venv/Scripts:$PATH` et `PYTHONUTF8=1` (console Windows cp1252).
- Aucun push sans accord explicite.

## Review Focus

1. Classe absente d'un split de test → métriques calculées sur `classes_`, pas de NaN silencieux (test évaluateur).
2. Label d'inférence inconnu / payload sans `alarm_code` → erreur explicite ou colonne neutre (test predictor).
3. Bloc `diagnosis` incohérent (classe inconnue dans `cost_matrix`) → ValueError au démarrage (test config).
4. Probabilités du predictor : somme à 1, ordre des colonnes = `classes_` (test predictor).
5. Régénération des 16 projets : diff limité aux fichiers partagés modifiés (lecture du diff).

---

### Task 1: Portabilité du générateur (fins de ligne)

- [x] `engine.py` et `utils_notebooks.py` écrivent en `newline="\n"`.
- [x] Vérifié : régénération de `classification/with-sklearn` = diff nul hors date du README.
- [ ] Commit.

### Task 2: Métriques et tests partagés (modality/tabular)

**Files:** `templates/modality/tabular/src/training/losses_metrics.py`,
`templates/modality/tabular/tests/test_models.py`, `templates/modality/tabular/tests/test_loaders.py`

- [ ] `_log_loss` : `labels = extra["classes"]` si fourni, sinon `np.unique(y_true)`.
- [ ] `_roc_auc_ovr` : `roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes)` ;
  NaN + warning si < 2 classes ; enregistrée pour `multiclass`, `requires={y_true, y_proba}`.
- [ ] `METRICS_BY_TASK["multiclass"]` += `mcc`, `roc_auc_ovr`.
- [ ] `test_models.py` : casts `float64` des prédictions limités aux tâches numériques.
- [ ] `test_loaders.py` : stratification comparée par proportion de chaque classe.
- [ ] Régénérer `classification/with-sklearn`, pytest vert. Commit.

### Task 3: Famille, générateur, manifeste

**Files:** `defaults/family/multiclass_classification.yaml`,
`templates/family/multiclass_classification/src/data/generators.py.j2`,
`manifests/ds-multiclass-sklearn.yaml`, `engine.py` (alias)

- [ ] Défauts de famille : business, data (15 colonnes, checks), metrics (`task: multiclass`,
  primary `f1_macro`), preprocessing (features `difference`/`product`), train split stratifié,
  `extras.root_conf.diagnosis`.
- [ ] Générateur : conditions mélange normal/stressé, logits physiques, calibration itérative
  des intercepts, `alarm_code` bruité, manquants, `generation_metadata.json` avec oracle,
  majoritaire, règle `alarm_code`.
- [ ] Générer, `make data` équivalent, vérifier parts de classes et plafond oracle. Ajuster logits.
- [ ] Commit.

### Task 4: Couche task/multiclass

**Files:** `templates/task/multiclass/src/{evaluation,inference,visualization}/…`,
`templates/task/multiclass/tests/test_multiclass.py`

- [ ] `evaluator.py` : `EvaluationResult` compatible, métriques avec `extra.classes`, table par
  classe (+ AUC OvR), confusion ordonnée `classes_`, paires confondues, références
  (majoritaire, `alarm_code`, oracle), décision coût minimal, abstention, calibration top-label,
  verdict. `compare_to_baseline`, `feature_importance`.
- [ ] `reports.py.j2` : rapport Markdown + JSON + CSV.
- [ ] `predictor.py.j2` : colonnes spec §4.
- [ ] `plots.py` : `MulticlassPlots.save_all`.
- [ ] Tests dédiés (Review Focus 1-4). Pipeline `mode=all` vert. Commit.

### Task 5: Notebooks et README de projet

**Files:** `notebooks/multiclass.py`, `notebooks/tabular.py`, `templates/base/README.md.j2`

- [ ] `structure_cells`, `build_04_model_exploration`, `build_06_error_analysis`.
- [ ] `_is_multiclass` + aiguillages 01/04/06 + `__all__`.
- [ ] Leçons 04/05/06 multiclasses dans le README de projet.
- [ ] `tools.verify` complet vert sur le nouveau projet. Calibrer les seuils d'objectifs sur mesure. Commit.

### Task 6: Non-régression et README racine

- [ ] Régénérer les 16 projets ; relire `git diff -I 'Dernière génération'`.
- [ ] `tools.verify` complet sur `classification/with-sklearn`, `--quick` sur les 15 autres.
- [ ] Rejouer le pipeline recommandation, README racine (17 projets, §1.6, §1.7, feuille de route).
- [ ] Commit.

### Task 7+: Autres stacks de la famille (sous-projet 2)

- [ ] LightGBM (métriques `multi_logloss`), PyTorch, Keras (`class_weight: auto`), XGBoost (LabelEncoder).
- [ ] Un manifeste + verify complet par stack ; README racine mis à jour.
