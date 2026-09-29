# Données — Consommation électrique régionale et prévision multi-horizons

> **Jeu de données synthétique** généré localement par `src/data/generators.py`.
> Aucune donnée réelle, aucun téléchargement, aucune information confidentielle.
> Objectif : disposer d'un dataset **crédible métier**, petit et reproductible, pour
> démontrer tout le cycle de vie (EDA → validation → preprocessing → entraînement → évaluation).

---

## 1. Description métier

Jeu d'apprentissage supervisé construit par **expansion temporelle** d'une série quotidienne de
consommation régionale (~3,4 ans) : chaque ligne correspond à un couple (origine, horizon) et porte
la consommation réellement observée `horizon_days` jours après l'origine, ainsi que les seules
informations disponibles au matin de cette origine. La série sous-jacente est générée par un modèle
explicite et réaliste : niveau de base, tendance lente, saisonnalité annuelle (pointe hivernale),
saisonnalité hebdomadaire (creux du week-end), thermo-sensibilité asymétrique (3,5 % par degré sous
15 °C, trois fois moins au-dessus de 24 °C car le parc français est peu climatisé), jours fériés et
vacances scolaires, épisodes extrêmes (vague de froid, canicule, arrêt industriel) et bruit
autocorrélé AR(1). La prévision de température fournie au modèle est la vérité **entachée d'une
erreur de prévision croissante avec l'horizon**, comme dans la réalité : c'est ce qui rend la
dégradation du J+1 au J+7 incompressible.

**Contexte** : Le réseau doit annoncer chaque matin sa consommation prévisionnelle pour les sept prochains
jours : c'est cette annonce qui pilote les achats sur les marchés de gros, le programme
d'appel des groupes de production et les alertes « équilibre menace » en cas d'écart.
Aujourd'hui la prévision est construite à la main à partir d'un profil hebdomadaire moyen et
d'un ajustement météo au jugement de l'analyste. Résultat : un écart moyen de 4 à 6 % qui
double pendant les vagues de froid, alors que chaque point d'erreur sur 2 500 MW représente
environ 25 MW à compenser en urgence — au prix spot du jour, souvent le plus élevé de
l'année.

**Problème adressé** : Prédire la consommation électrique quotidienne régionale (en MW) pour les horizons J+1, J+2,
J+3 et J+7, à partir de l'histoire de consommation disponible **au matin de l'origine** et
de la prévision météo connue à ce même instant, puis publier une prévision assortie d'un
intervalle et d'un indicateur de confiance.

**Consommateur principal** : Équipe Prévisions d'un gestionnaire de réseau de distribution électrique (data scientist qui industrialise le modèle, analyste marché qui engage les achats d'énergie, et astreinte conduite qui équilibre l'offre et la demande en J+1).

---

## 2. Fiche d'identité

| Propriété | Valeur |
| --- | --- |
| Nom du dataset | `regional_electricity_load` |
| Granularité | une ligne = Identifiant unique du couple (origine, horizon) |
| Nombre d'échantillons (par défaut) | 4 800 |
| Nombre de colonnes | 32 |
| Cible | `load_mw` || Clé | `sample_id` (unique) || Dimension temporelle | `origin_date` || Formats | parquet, csv |
| Emplacement | `data/raw/regional_electricity_load.parquet` |
| Graine de génération | `42` |
| Valeurs manquantes | oui, en proportion maîtrisée (voir schéma) |
| Outliers | oui, légitimes métier (bornés par le schéma Pandera) |
| Source | **générée synthétiquement** (`SyntheticDataGenerator`) |

---

## 3. Schéma des colonnes

| # | Colonne | Type | Rôle | Unité | Signification métier | Distribution attendue |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `sample_id` | `str` | identifier | - | Identifiant unique du couple (origine, horizon) | identifiants séquentiels déterministes |
| 2 | `origin_date` | `datetime` | timestamp | - | Date de l'origine : dernier jour dont la consommation est connue au moment de la prévision | étalement chronologique sur ~1 200 origines consécutives |
| 3 | `horizon_days` | `int` | feature | jours | Horizon visé en jours (1, 2, 3 ou 7) — le modèle apprend la dégradation avec l'horizon | quatre horizons équirépartis (25 % chacun) |
| 4 | `target_date` | `datetime` | metadata | - | Date prévue (origine + horizon) — sert à l'analyse d'erreur, jamais au modèle | - |
| 5 | `target_month` | `str` | metadata | - | Mois du jour cible, pour la ventilation saisonnière des erreurs | - |
| 6 | `event_type` | `str` | metadata | - | Nature de l'épisode touchant le jour cible (aucun, vague de froid, canicule, arrêt industriel) | ~96 % none, 1,7 % cold_snap, 1,5 % heatwave, 0,7 % industrial_shutdown |
| 7 | `is_extreme_event` | `bool` | metadata | - | Vrai si le jour cible appartient à un épisode extrême — cible prioritaire de l'analyse d'erreur | ~4 % de vrais |
| 8 | `load_mw` | `float` | target | MW | Consommation électrique régionale observée le jour cible | médiane ~2 190 MW, pointe de vague de froid ~4 900 MW, creux estival ~1 050 MW |
| 9 | `load_last_observed` | `float` | feature | MW | Dernière consommation connue : le jour de l'origine lui-même (ancre de persistance) | - |
| 10 | `load_lag_1d` | `float` | feature | MW | Consommation un jour avant l'origine | - |
| 11 | `load_lag_2d` | `float` | feature | MW | Consommation deux jours avant l'origine | - |
| 12 | `load_lag_7d` | `float` | feature | MW | Consommation sept jours avant l'origine (même jour de la semaine que l'origine) | - |
| 13 | `load_lag_14d` | `float` | feature | MW | Consommation quatorze jours avant l'origine | - |
| 14 | `load_lag_28d` | `float` | feature | MW | Consommation vingt-huit jours avant l'origine (niveau de référence mensuel) | - |
| 15 | `load_rolling_mean_7d` | `float` | feature | MW | Moyenne glissante 7 jours de la consommation, terminée à l'origine | - |
| 16 | `load_rolling_std_7d` | `float` | feature | MW | Écart-type glissant 7 jours : volatilité récente du réseau | - |
| 17 | `load_rolling_min_7d` | `float` | feature | MW | Minimum glissant 7 jours : plancher récent (détecte un creux de vacances) | - |
| 18 | `load_rolling_mean_28d` | `float` | feature | MW | Moyenne glissante 28 jours : niveau de fond mensuel | - |
| 19 | `load_seasonal_naive` | `float` | feature | MW | Ancrage saisonnier : consommation du même jour de la semaine, une semaine avant le jour cible (toujours ≤ origine, donc licite) | - |
| 20 | `temperature_forecast_c` | `float` | feature | °C | Prévision de température moyenne pour le jour cible, disponible à l'origine — entachée d'une erreur croissante avec l'horizon | erreur de prévision N(0, 0,35 °C) à J+1 jusqu'à N(0, 1,2 °C) à J+7 |
| 21 | `temperature_forecast_prev_c` | `float` | feature | °C | Prévision de température de la veille du jour cible (observation à J+1) — permet au modèle de reconstruire l'inertie thermique du bâti | erreur de prévision d'une échéance plus courte d'un jour ; manquante quand le capteur de l'origine est en panne (à J+1 uniquement) |
| 22 | `temperature_anomaly_c` | `float` | feature | °C | Écart de la prévision de température à la normale saisonnière du jour cible | - |
| 23 | `temperature_lag_1d` | `float` | feature | °C | Température observée à l'origine | - |
| 24 | `temperature_rolling_mean_7d` | `float` | feature | °C | Moyenne glissante 7 jours de la température observée, terminée à l'origine | - |
| 25 | `hdd_target` | `float` | feature | °C | Degrés-jours de chauffe du jour cible : max(0, 18 °C - température prévue) | - |
| 26 | `cdd_target` | `float` | feature | °C | Degrés-jours de froid du jour cible : max(0, température prévue - 24 °C) | - |
| 27 | `target_weekday` | `str` | feature | - | Jour de la semaine du jour cible (connu par avance) | ~14,3 % par jour, léger déséquilibre dû au découpage des origines |
| 28 | `target_is_weekend` | `bool` | feature | - | Vrai si le jour cible tombe un samedi ou un dimanche | ~28,6 % de vrais |
| 29 | `target_is_holiday` | `bool` | feature | - | Vrai si le jour cible est un jour férié national français | ~3,2 % de vrais |
| 30 | `target_school_holiday` | `bool` | feature | - | Vrai si le jour cible tombe pendant les vacances scolaires (effet résidentiel marqué) | ~33 % de vrais |
| 31 | `target_doy_sin` | `float` | feature | - | Composante sinus du jour de l'année cible (encodage cyclique de la saisonnalité annuelle) | - |
| 32 | `target_doy_cos` | `float` | feature | - | Composante cosinus du jour de l'année cible | - |

### Contraintes de validation (contrat exécutable)

Les contraintes ci-dessous ne sont pas de la documentation : elles sont **exécutées** à chaque
chargement par les schémas Pandera de `src/data/schemas.py` (`RawDataSchema`,
`ProcessedDataSchema`, `InferenceDataSchema`).

| Colonne | Contraintes |
| --- | --- |
| `sample_id` | type `str`, unique, non nul, motif ^FC-[0-9]{5}-H[0-9]{1,2}$ |
| `origin_date` | type `datetime`, non nul |
| `horizon_days` | type `int`, non nul, dans 1, 2, 3, 7 |
| `target_date` | type `datetime`, non nul |
| `target_month` | type `str`, non nul, dans jan, feb, mar, apr, may, jun, jul, aug, sep, oct, nov, dec |
| `event_type` | type `str`, non nul, dans none, cold_snap, heatwave, industrial_shutdown |
| `is_extreme_event` | type `bool`, non nul |
| `load_mw` | type `float`, non nul, >= 200.0 · <= 8000.0 |
| `load_last_observed` | type `float`, non nul, >= 150.0 · <= 8500.0 |
| `load_lag_1d` | type `float`, non nul, >= 150.0 · <= 8500.0 |
| `load_lag_2d` | type `float`, non nul, >= 150.0 · <= 8500.0 |
| `load_lag_7d` | type `float`, non nul, >= 150.0 · <= 8500.0 |
| `load_lag_14d` | type `float`, non nul, >= 150.0 · <= 8500.0 |
| `load_lag_28d` | type `float`, non nul, >= 150.0 · <= 8500.0 |
| `load_rolling_mean_7d` | type `float`, non nul, >= 150.0 · <= 8500.0 |
| `load_rolling_std_7d` | type `float`, non nul, >= 0.0 · <= 2000.0 |
| `load_rolling_min_7d` | type `float`, non nul, >= 150.0 · <= 8500.0 |
| `load_rolling_mean_28d` | type `float`, non nul, >= 150.0 · <= 8500.0 |
| `load_seasonal_naive` | type `float`, non nul, >= 150.0 · <= 8500.0 |
| `temperature_forecast_c` | type `float`, non nul, >= -25.0 · <= 48.0 |
| `temperature_forecast_prev_c` | type `float`, nullable, >= -25.0 · <= 48.0 |
| `temperature_anomaly_c` | type `float`, non nul, >= -25.0 · <= 25.0 |
| `temperature_lag_1d` | type `float`, nullable, >= -25.0 · <= 48.0 |
| `temperature_rolling_mean_7d` | type `float`, nullable, >= -25.0 · <= 48.0 |
| `hdd_target` | type `float`, non nul, >= 0.0 · <= 40.0 |
| `cdd_target` | type `float`, non nul, >= 0.0 · <= 25.0 |
| `target_weekday` | type `str`, non nul, dans mon, tue, wed, thu, fri, sat, sun |
| `target_is_weekend` | type `bool`, non nul |
| `target_is_holiday` | type `bool`, non nul |
| `target_school_holiday` | type `bool`, non nul |
| `target_doy_sin` | type `float`, non nul, >= -1.0 · <= 1.0 |
| `target_doy_cos` | type `float`, non nul, >= -1.0 · <= 1.0 |

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
| `data/raw/regional_electricity_load.parquet` | données brutes (format machine, typé, compressé) |
| `data/raw/regional_electricity_load.csv` | mêmes données, lisibles par un humain |
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

- Construction par expansion temporelle : 1 200 origines x 4 horizons = 4 800 lignes. Les quatre lignes d'une même origine partagent leur histoire et diffèrent par le jour cible.
- Contract d'antériorité : `load_lag_*`, `load_rolling_*` et `temperature_lag/rolling` sont calculés **jusqu'à l'origine incluse** ; `target_*` (calendrier) et `temperature_forecast_c` sont connus par avance ; `load_mw`, `target_date`, `event_type` et `is_extreme_event` sont des métadonnées de vérité terrain.
- `load_seasonal_naive` est licite bien qu'il regarde vers le jour cible : il s'agit de la consommation du jour cible **moins sept jours**, toujours antérieure ou égale à l'origine pour un horizon ≤ 7.
- Valeurs manquantes volontaires sur `temperature_lag_1d` (~3 %) et `temperature_rolling_mean_7d` (~3 %), avec un biais hivernal (pannes de capteur par temps de gel).
- Bruit irréductible AR(1) (σ ≈ 2,2 %) plus erreur de prévision météo : un MAPE proche de 0 signerait une fuite, pas un bon modèle. Le plancher atteignable est d'environ 1,5 %.
- Épisodes extrêmes sur les 3,3 ans simulés : ~18 jours de vague de froid (température déplacée de -7,5 °C, consommation +30 %), ~18 jours de canicule (+6 °C, consommation +12 %) et ~9 jours d'arrêt industriel (-24 %, aucune signature météo).

---

## 7. Observations attendues (EDA)

Points à retrouver dans `notebooks/01_exploratory_analysis.ipynb` :

1. Le bâti réagit à une température lissée sur deux jours, pas à la seule valeur du jour cible : fournir la prévision de la veille (`temperature_forecast_prev_c`) et sa variation est ce qui permet au modèle de reconstruire cette inertie, et c'est un des rares gains disponibles au-delà du réglage des hyperparamètres.
2. La consommation est pilotée à ~65 % par le calendrier et la température : un modèle qui ne connaît que les décalages temporels rate les épisodes de froid, un modèle qui ne connaît que la météo rate le creux du week-end. Les deux familles de features sont complémentaires, et le notebook 04 mesure leur apport séparé.
3. L'erreur croît avec l'horizon (MAPE J+1 ≈ moitié du MAPE J+7) pour deux raisons distinctes : l'erreur de prévision météo augmente, et l'information de court terme (J-1) devient moins pertinente. Publier un MAPE unique sans ventilation par horizon masque cette structure.
4. Le bruit est autocorrélé (AR(1) de coefficient ~0,6) : les résidus ne sont pas indépendants, donc les intervalles de confiance gaussiens naïfs sont trop étroits et le backtest par origine glissante est obligatoire — la validation croisée aléatoire produirait un score optimiste et faux.
5. `temperature_lag_1d` et `temperature_rolling_mean_7d` comportent des manquants volontaires (~3 %, panne de capteur, surreprésentés en hiver) : l'imputation doit être explicite et un indicateur de manquant est rentable.
6. Les jours fériés et les ponts produisent des creux de 10 à 20 % que le seul jour de la semaine ne prédit pas : sans feature calendaire explicite, le modèle sur-prévoit systématiquement ces jours-là.
7. Le générateur injecte trois régimes de rupture (vague de froid, canicule, arrêt industriel) qui représentent ~4 % des jours cibles mais une part disproportionnée de l'erreur totale : c'est le cœur de l'analyse d'erreur du notebook 06. Les deux premiers déplacent réellement la température — un front lissé sur trois jours — donc le modèle peut les anticiper ; l'arrêt industriel est un choc de demande sans aucune signature météo, volontairement imprévisible.

---

## 8. Passage à l'échelle (données réelles)

Pour brancher ce projet sur des données de production :

1. Remplacer `SyntheticDataGenerator` par un `RawDataLoader` qui lit la source réelle
   (entrepôt, objet storage, API) — l'interface ne change pas.
2. **Conserver les schémas Pandera** : ils deviennent le contrat avec le fournisseur de données
   (et un test de non-régression à chaque évolution du schéma amont).
3. Versionner les datasets (DVC / Delta / snapshots horodatés) pour garantir la reproductibilité.
4. Ajouter une surveillance de dérive (voir `mlops/model-monitoring/*`).
