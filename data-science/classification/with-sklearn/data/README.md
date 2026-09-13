# Données — Abonnés télécom et attrition (churn)

> **Jeu de données synthétique** généré localement par `src/data/generators.py`.
> Aucune donnée réelle, aucun téléchargement, aucune information confidentielle.
> Objectif : disposer d'un dataset **crédible métier**, petit et reproductible, pour
> démontrer tout le cycle de vie (EDA → validation → preprocessing → entraînement → évaluation).

---

## 1. Description métier

Population d'abonnés d'un opérateur télécom, avec leurs caractéristiques contractuelles, d'usage et
de relation client, et l'étiquette d'attrition observée. Les données sont générées par un modèle
logistique latent avec interactions : le signal est réel mais bruité, ce qui reproduit fidèlement la
difficulté d'un cas churn industriel.

**Contexte** : L'opérateur perd chaque mois une part significative de ses abonnés au profit de concurrents.
Acquérir un client coûte 5 à 7 fois plus cher que d'en conserver un, et les campagnes de
rétention actuelles sont déclenchées au feeling, trop tard et sur trop de monde (coût
inutile, fatigue commerciale).

**Problème adressé** : Identifier, avant la résiliation, les abonnés les plus susceptibles de partir dans les 30
prochains jours, afin de cibler une action de rétention (offre, geste commercial, appel
sortant) sur une population réduite et pertinente.

**Consommateur principal** : Équipe Customer Success / Marketing d'un opérateur télécom (abonnement B2C), avec un data scientist qui industrialise le modèle et un CRM qui consomme les scores.

---

## 2. Fiche d'identité

| Propriété | Valeur |
| --- | --- |
| Nom du dataset | `telecom_churn` |
| Granularité | une ligne = Identifiant unique de l'abonné |
| Nombre d'échantillons (par défaut) | 4,000 |
| Nombre de colonnes | 15 |
| Cible | `churned` || Taux de classe positive | ~26.0 % || Clé | `customer_id` (unique) || Formats | parquet, csv |
| Emplacement | `data/raw/telecom_churn.parquet` |
| Graine de génération | `42` |
| Valeurs manquantes | oui, en proportion maîtrisée (voir schéma) |
| Outliers | oui, légitimes métier (bornés par le schéma Pandera) |
| Source | **générée synthétiquement** (`SyntheticDataGenerator`) |

---

## 3. Schéma des colonnes

| # | Colonne | Type | Rôle | Unité | Signification métier | Distribution attendue |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `customer_id` | `str` | identifier | - | Identifiant unique de l'abonné | identifiants séquentiels uniques |
| 2 | `signup_date` | `datetime` | timestamp | - | Date de souscription de l'abonnement | étalement sur ~6 ans, cohérent avec l'ancienneté |
| 3 | `tenure_months` | `int` | feature | mois | Ancienneté de l'abonné | gamma bimodale selon le segment (jeunes abonnés vs fidèles) |
| 4 | `contract_type` | `category` | feature | - | Type de contrat souscrit | ~35 % mensuel, ~29 % 1 an, ~36 % 2 ans |
| 5 | `internet_service` | `category` | feature | - | Service internet associé à l'abonnement | ~55 % fibre, ~30 % DSL, ~15 % sans internet |
| 6 | `payment_method` | `category` | feature | - | Moyen de paiement utilisé | ~34 % electronic_check, ~16 % mailed_check, ~28 % virement, ~22 % carte |
| 7 | `region` | `category` | feature | - | Région commerciale de rattachement | quasi uniforme (22-28 %) |
| 8 | `monthly_charges` | `float` | feature | EUR | Montant mensuel facturé | log-normale centrée ~65 EUR, dépend du service internet |
| 9 | `total_charges` | `float` | feature | EUR | Cumul facturé depuis la souscription | ≈ monthly_charges × tenure (fortement corrélée, colinéarité volontaire) |
| 10 | `support_tickets_6m` | `int` | feature | tickets | Nombre de tickets support ouverts sur 6 mois | Poisson(λ ≈ 0.6-1.2) selon le contrat et le service |
| 11 | `avg_monthly_data_gb` | `float` | feature | Go | Consommation data mensuelle moyenne | gamma, ~3 % de manquants (capteur absent) |
| 12 | `num_products` | `int` | feature | produits | Nombre de services/produits souscrits (bundle) | 1 à 5, mode sur 2-3 |
| 13 | `has_promotion` | `bool` | feature | - | Bénéficie d'une promotion active (1 = oui) | ~25 % de promotions actives |
| 14 | `satisfaction_score` | `float` | feature | /10 | Score de satisfaction déclaré (enquête CSAT) | normale tronquée ~6.5, ~5 % de manquants (non-réponse) |
| 15 | `churned` | `int` | target | - | Attrition observée dans les 30 jours (1 = parti) | ~26 % de classe positive |

### Contraintes de validation (contrat exécutable)

Les contraintes ci-dessous ne sont pas de la documentation : elles sont **exécutées** à chaque
chargement par les schémas Pandera de `src/data/schemas.py` (`RawDataSchema`,
`ProcessedDataSchema`, `InferenceDataSchema`).

| Colonne | Contraintes |
| --- | --- |
| `customer_id` | type `str`, unique, non nul, motif ^CUS-[0-9]{5}$ |
| `signup_date` | type `datetime`, non nul |
| `tenure_months` | type `int`, non nul, >= 0 · <= 72 |
| `contract_type` | type `category`, non nul, dans month_to_month, one_year, two_year |
| `internet_service` | type `category`, non nul, dans fiber, dsl, none |
| `payment_method` | type `category`, non nul, dans electronic_check, mailed_check, bank_transfer, credit_card |
| `region` | type `category`, non nul, dans north, south, east, west |
| `monthly_charges` | type `float`, non nul, >= 18.0 · <= 480.0 |
| `total_charges` | type `float`, non nul, >= 18.0 · <= 20000.0 |
| `support_tickets_6m` | type `int`, non nul, >= 0 · <= 14 |
| `avg_monthly_data_gb` | type `float`, nullable, >= 0.0 · <= 1000.0 |
| `num_products` | type `int`, non nul, >= 1 · <= 5 |
| `has_promotion` | type `bool`, non nul, dans 0, 1 |
| `satisfaction_score` | type `float`, nullable, >= 1.0 · <= 10.0 |
| `churned` | type `int`, non nul, dans 0, 1 |

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
| `data/raw/telecom_churn.parquet` | données brutes (format machine, typé, compressé) |
| `data/raw/telecom_churn.csv` | mêmes données, lisibles par un humain |
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

- Valeurs manquantes volontaires sur `satisfaction_score` (~5 %) et `avg_monthly_data_gb` (~3 %) pour exercer l'imputation.
- Outliers légitimes (~1 % des lignes) : clients entreprise avec charges élevées et accumulation de tickets.
- Colinéarité volontaire entre `total_charges`, `monthly_charges` et `tenure_months`.
- Cible générée par un modèle logistique latent avec une interaction (contrat mensuel × jeune abonné × satisfaction basse).
- Bruit irréductible : la probabilité de churn est tirée, pas déterministe — un score parfait indiquerait une fuite.

---

## 7. Observations attendues (EDA)

Points à retrouver dans `notebooks/01_exploratory_analysis.ipynb` :

1. La classe cible est déséquilibrée (~26 % de churn) : l'accuracy seule est trompeuse, il faut piloter ROC/PR AUC, rappel et précision.
2. `contract_type = month_to_month` est de loin le premier facteur de risque : sans engagement, la résiliation est structurellement plus facile.
3. L'ancienneté (`tenure_months`) est fortement protectrice : le risque décroît régulièrement au-delà de 12 mois, d'où l'intérêt des features de cohorte.
4. `total_charges` et `monthly_charges × tenure` sont quasi redondants : colinéarité attendue, à surveiller pour les modèles linéaires (pas gênant pour les arbres).
5. Le nombre de tickets support sur 6 mois est un signal d'alerte précoce : au-delà de 4 tickets, la probabilité de départ progresse nettement.
6. `satisfaction_score` est anti-corrélé au churn mais comporte ~5 % de manquants : l'imputation doit être explicite et documentée (médiane par contrat).
7. Le paiement par `electronic_check` est associé à un churn supérieur : proxy d'un client peu engagé dans la relation, pas d'une cause directe.
8. La fibre (`internet_service = fiber`) présente des charges plus élevées et un churn plus fort : le prix perçu est un levier de rétention.
9. Quelques outliers légitimes existent (charges > 250 EUR, > 8 tickets) : les supprimer perdrait de l'information, le winsorising (1-99 %) est préférable.
10. Les distributions numériques sont hétérogènes (charges en EUR, consommation en Go, tickets en unités) : le scaling est indispensable pour les modèles sensibles à l'échelle (SVM, k-NN, réseaux).
11. La variable `region` n'apporte presque aucun signal : candidate à l'élimination pour simplifier le modèle (test de permutation importance).
12. Les interactions comptent : contrat mensuel + faible ancienneté + satisfaction basse concentre l'essentiel du risque, ce qu'un modèle linéaire sans terme d'interaction rate.
13. Le bruit injecté (probabilité de churn non déterministe) fixe un plafond de performance : un AUC de 1.0 signerait une fuite de données, pas un bon modèle.

---

## 8. Passage à l'échelle (données réelles)

Pour brancher ce projet sur des données de production :

1. Remplacer `SyntheticDataGenerator` par un `RawDataLoader` qui lit la source réelle
   (entrepôt, objet storage, API) — l'interface ne change pas.
2. **Conserver les schémas Pandera** : ils deviennent le contrat avec le fournisseur de données
   (et un test de non-régression à chaque évolution du schéma amont).
3. Versionner les datasets (DVC / Delta / snapshots horodatés) pour garantir la reproductibilité.
4. Ajouter une surveillance de dérive (voir `mlops/model-monitoring/*`).
