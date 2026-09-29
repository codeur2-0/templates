# Données — Alarmes machine et mode de défaillance diagnostiqué

> **Jeu de données synthétique** généré localement par `src/data/generators.py`.
> Aucune donnée réelle, aucun téléchargement, aucune information confidentielle.
> Objectif : disposer d'un dataset **crédible métier**, petit et reproductible, pour
> démontrer tout le cycle de vie (EDA → validation → preprocessing → entraînement → évaluation).

---

## 1. Description métier

Une ligne par alarme levée par une fraiseuse CNC : contexte de production, code émis par l'automate,
relevés capteurs à l'instant de l'alarme, et mode de défaillance constaté par le technicien à la
clôture de l'intervention. Le mode est tiré d'un modèle physique (usure, écart thermique à basse
vitesse, puissance hors plage, effort au-delà d'un seuil dépendant de la gamme) avec un bruit
irréductible : le signal est réel mais imparfait, et une classe (`random_failure`) n'a
volontairement aucun signal observable.

**Contexte** : Chaque alarme automate arrête la machine et déclenche une intervention. Aujourd'hui,
l'équipe envoyée est choisie d'après le code d'alarme émis par l'automate, calculé par des
règles à seuil fixes qui ignorent la gamme de produit et les zones de bordure : une fois sur
deux environ, la mauvaise spécialité se déplace, la machine reste arrêtée pendant qu'on
envoie la bonne, et une panne réelle classée « fausse alarme » est acquittée alors que la
machine continue de se dégrader.

**Problème adressé** : Prédire, au moment de l'alarme et à partir des seuls capteurs disponibles, le mode de
défaillance parmi six (fausse alarme, usure outil, dissipation thermique, défaut de
puissance, surcharge mécanique, défaillance aléatoire), afin d'envoyer directement la bonne
équipe — ou de demander un avis expert quand le modèle doute.

**Consommateur principal** : Responsable maintenance d'un atelier d'usinage (fraiseuses CNC sur quatre lignes), avec une équipe de techniciens répartis par spécialité (outillage, thermique, électrique, mécanique) et un data scientist qui industrialise le modèle ; le superviseur de production consomme le diagnostic dans son outil de GMAO.

---

## 2. Fiche d'identité

| Propriété | Valeur |
| --- | --- |
| Nom du dataset | `machine_failure_diagnosis` |
| Granularité | une ligne = Identifiant unique de l'alarme |
| Nombre d'échantillons (par défaut) | 6 000 |
| Nombre de colonnes | 15 |
| Cible | `failure_mode` || Clé | `alarm_id` (unique) || Formats | parquet, csv |
| Emplacement | `data/raw/machine_failure_diagnosis.parquet` |
| Graine de génération | `42` |
| Valeurs manquantes | oui, en proportion maîtrisée (voir schéma) |
| Outliers | oui, légitimes métier (bornés par le schéma Pandera) |
| Source | **générée synthétiquement** (`SyntheticDataGenerator`) |

---

## 3. Schéma des colonnes

| # | Colonne | Type | Rôle | Unité | Signification métier | Distribution attendue |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `alarm_id` | `str` | identifier | - | Identifiant unique de l'alarme | identifiants séquentiels uniques |
| 2 | `alarm_timestamp` | `datetime` | timestamp | - | Date et heure de levée de l'alarme | uniforme sur ~18 mois, heure cohérente avec l'équipe |
| 3 | `product_quality` | `category` | feature | - | Gamme de la pièce usinée (L = standard, M = intermédiaire, H = haute précision) | ~50 % L, ~30 % M, ~20 % H |
| 4 | `production_line` | `category` | feature | - | Ligne de production de la machine | 30 / 28 / 24 / 18 % ; line_d est la plus ancienne (refroidissement plus faible) |
| 5 | `shift` | `category` | feature | - | Équipe en poste au moment de l'alarme | ~45 % jour, ~32 % soir, ~23 % nuit |
| 6 | `alarm_code` | `category` | feature | - | Code d'alarme émis par l'automate (règles à seuil fixes, une part de codes corrompus) | E1xx thermique, E2xx électrique, E3xx mécanique, E901 générique |
| 7 | `air_temperature_k` | `float` | feature | K | Température de l'air ambiant | normale ~300 K, écart-type ~1,8 K |
| 8 | `process_temperature_k` | `float` | feature | K | Température du process (zone de coupe) | ≈ air + 10 K ; l'écart se resserre quand le refroidissement faiblit |
| 9 | `rotational_speed_rpm` | `float` | feature | tr/min | Vitesse de rotation de la broche | normale ~1 540 tr/min, anti-corrélée au couple |
| 10 | `torque_nm` | `float` | feature | N.m | Couple de coupe | normale ~40 N.m, étirée par les régimes de surcharge |
| 11 | `tool_wear_min` | `int` | feature | min | Temps d'utilisation cumulé de l'outil en place | uniforme 0-205 min, étirée au-delà de 200 min par les outils en fin de vie |
| 12 | `coolant_flow_l_min` | `float` | feature | L/min | Débit du circuit de refroidissement | normale ~12,5 L/min, plus faible sur line_d |
| 13 | `vibration_mm_s` | `float` | feature | mm/s | Vitesse vibratoire efficace de la broche | gamma ~3,6 mm/s, monte avec l'usure et l'effort ; ~4 % de manquants (capteur absent) |
| 14 | `hours_since_maintenance` | `float` | feature | h | Heures de fonctionnement depuis la dernière maintenance préventive | gamma ~320 h, légèrement plus élevée sur les machines stressées |
| 15 | `failure_mode` | `str` | target | - | Mode de défaillance constaté à la clôture de l'intervention | ~30 / 20 / 18 / 14 / 12 / 6 % |

### Contraintes de validation (contrat exécutable)

Les contraintes ci-dessous ne sont pas de la documentation : elles sont **exécutées** à chaque
chargement par les schémas Pandera de `src/data/schemas.py` (`RawDataSchema`,
`ProcessedDataSchema`, `InferenceDataSchema`).

| Colonne | Contraintes |
| --- | --- |
| `alarm_id` | type `str`, unique, non nul, motif ^ALM-[0-9]{5}$ |
| `alarm_timestamp` | type `datetime`, non nul |
| `product_quality` | type `category`, non nul, dans L, M, H |
| `production_line` | type `category`, non nul, dans line_a, line_b, line_c, line_d |
| `shift` | type `category`, non nul, dans day, evening, night |
| `alarm_code` | type `category`, non nul, dans E101, E102, E201, E202, E301, E302, E901 |
| `air_temperature_k` | type `float`, non nul, >= 290.0 · <= 310.0 |
| `process_temperature_k` | type `float`, non nul, >= 295.0 · <= 322.0 |
| `rotational_speed_rpm` | type `float`, non nul, >= 900.0 · <= 2600.0 |
| `torque_nm` | type `float`, non nul, >= 2.0 · <= 90.0 |
| `tool_wear_min` | type `int`, non nul, >= 0 · <= 260 |
| `coolant_flow_l_min` | type `float`, non nul, >= 1.0 · <= 22.0 |
| `vibration_mm_s` | type `float`, nullable, >= 0.0 · <= 30.0 |
| `hours_since_maintenance` | type `float`, non nul, >= 0.0 · <= 3000.0 |
| `failure_mode` | type `str`, non nul, dans false_alarm, tool_wear, heat_dissipation, power_failure, overstrain, random_failure |

---

## 4. Générer / régénérer les données

```bash
# depuis la racine du projet
make data
# équivalents :
python scripts/generate_data.py
python -m src.main mode=generate-data

# variantes
python scripts/generate_data.py data.n_samples=20000 seed=7
python scripts/generate_data.py data.formats=[parquet]
```

Le générateur écrit :

| Fichier | Contenu |
| --- | --- |
| `data/raw/machine_failure_diagnosis.parquet` | données brutes (format machine, typé, compressé) |
| `data/raw/machine_failure_diagnosis.csv` | mêmes données, lisibles par un humain |
| `data/raw/generation_metadata.json` | empreinte du tirage : seed, shape, dtypes, taux de manquants, distribution de la cible |

**Déterminisme** : à `seed` et `n_samples` fixés, les fichiers sont identiques d'une exécution à
l'autre — condition indispensable pour comparer deux stacks ou rejouer un incident.

---

## 5. Cycle de vie des données

```text
SyntheticDataGenerator  →  data/raw/*.parquet            (immuable, jamais modifié en place)
        │
        ▼  RawDataLoader + RawDataSchema (validation)
FeatureBuilder          →  features dérivées
        │
        ▼  PreprocessingPipeline (fit sur train uniquement)
data/processed/*.parquet                                 (splits + matrice modélisable)
        │
        ▼  ProcessedDataSchema (validation avant entraînement)
Trainer / Evaluator / Predictor
```

Règles appliquées :

1. **`data/raw/` est immuable** : on ne nettoie jamais une donnée brute en place.
2. **`data/processed/` est jetable** : régénérable à tout moment depuis `raw/`.
3. **Parquet par défaut** pour les étapes intermédiaires (types stricts, compression, pushdown).
4. **Aucune donnée générée n'est committée** (`.gitignore`) : seuls le générateur et sa
   documentation le sont.

---

## 6. Ce que les données contiennent volontairement

Pour que l'exemple soit pédagogique, le générateur injecte des difficultés **réalistes** :

- Valeurs manquantes volontaires sur `vibration_mm_s` (~4 %) pour exercer l'imputation.
- Les classes sont tirées d'un softmax de logits physiques calculés sur les seules observables ; les ordonnées sont calibrées pour tenir les parts visées.
- Le code d'alarme dérive des capteurs par des règles à seuil fixes avec ~30 % de codes corrompus : il n'apporte aucune information sur la classe au-delà des capteurs.
- Plafond oracle, plancher majoritaire et référence de routage sont publiés dans `data/raw/generation_metadata.json`.

---

## 7. Observations attendues (EDA)

Points à retrouver dans `notebooks/01_exploratory_analysis.ipynb` :

1. Six classes déséquilibrées (de ~30 % à ~6 %) : l'accuracy récompense un modèle qui ignore les modes rares, d'où le **macro-F1** comme métrique de pilotage.
2. Aucun capteur seul ne sépare les modes : ce sont des **combinaisons physiques** qui le font — l'écart process-air à basse vitesse (dissipation thermique), le produit couple × vitesse (puissance), le produit usure × couple (surcharge).
3. Le seuil de surcharge dépend de la gamme (11 000 / 12 000 / 13 000 min.N.m pour L / M / H) : une règle unique, comme celle de l'automate, se trompe structurellement sur les pièces haute précision.
4. `tool_wear` et `overstrain` se confondent sur les outils usés soumis à un fort couple : la paire la plus difficile du jeu, qui se lit dans la matrice de confusion.
5. `heat_dissipation` et `power_failure` partagent la vitesse basse : un sous-régime de puissance ressemble à un défaut thermique tant qu'on ne regarde pas l'écart de température.
6. Le code d'alarme de l'automate est un résumé bruité des capteurs : utile, mais une fois sur deux environ il désigne la mauvaise spécialité.
7. `random_failure` n'a aucun signal observable par construction : son rappel reste proche de zéro pour tout modèle, oracle compris. C'est un plafond structurel, pas un défaut de réglage.
8. La vibration manque sur ~4 % des alarmes (machines non équipées) : l'imputation doit être apprise sur le train et le manque lui-même n'est pas informatif.
9. Les équipes de nuit acquittent un peu plus de fausses alarmes : effet faible, réel, à ne pas sur-interpréter.
10. Un score parfait signerait une fuite : le générateur publie un plafond oracle calculé avec le bruit, qu'aucun modèle ne peut dépasser.

---

## 8. Passage à l'échelle (données réelles)

Pour brancher ce projet sur des données de production :

1. Remplacer `SyntheticDataGenerator` par un `RawDataLoader` qui lit la source réelle
   (entrepôt, objet storage, API) — l'interface ne change pas.
2. **Conserver les schémas Pandera** : ils deviennent le contrat avec le fournisseur de données
   (et un test de non-régression à chaque évolution du schéma amont).
3. Versionner les datasets (DVC / Delta / snapshots horodatés) pour garantir la reproductibilité.
4. Ajouter une surveillance de dérive (voir `mlops/model-monitoring/*`).
