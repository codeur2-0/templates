# Données — Transactions de paiement et fraude observée

> **Jeu de données synthétique** généré localement par `src/data/generators.py`.
> Aucune donnée réelle, aucun téléchargement, aucune information confidentielle.
> Objectif : disposer d'un dataset **crédible métier**, petit et reproductible, pour
> démontrer tout le cycle de vie (EDA → validation → preprocessing → entraînement → évaluation).

---

## 1. Description métier

Flux de transactions d'un PSP français sur 90 jours, avec le contexte marchand, porteur et technique
de chaque paiement. Les données sont générées par un modèle latent : chaque transaction légitime
suit des distributions réalistes par catégorie et par canal, tandis que quatre **modes opératoires**
de fraude (card-not-present, prise de compte, identité synthétique, fraude amicale) déforment des
sous-ensembles précis de variables (vélocité, montant, empreinte d'appareil, authentification,
géographie). Du bruit irréductible, des valeurs manquantes et des outliers **légitimes** (achats de
luxe, paiements professionnels) sont injectés : un détecteur qui ne fait que repérer les gros
montants se fait piéger. Deux colonnes de métadonnées (`is_fraud`, `fraud_scheme`) permettent de
mesurer la performance sans jamais entrer dans les features.

**Contexte** : La détection repose aujourd'hui sur ~240 règles métier écrites au fil des incidents («
montant > 2 000 EUR ET pays différent du pays de facturation », etc.). Elles capturent 38 %
de la fraude confirmée, génèrent 11 000 alertes par jour dont 96 % sont levées sans suite
par les analystes, et sont aveugles aux schémas inédits : une règle n'existe que lorsque la
fraude a déjà été vue. La fraude évolue plus vite que le catalogue de règles (card testing
automatisé, prise de compte, identités synthétiques).

**Problème adressé** : Détecter **sans étiquette fiable et à jour** les transactions anormales — la fraude
confirmée n'est connue qu'après 30 à 90 jours (chargeback), donc un modèle supervisé apprend
toujours sur une vérité partielle et biaisée. Objectif : produire un score de risque
continu, ordonner les transactions et transmettre aux analystes un volume d'alertes
compatible avec leur capacité, en maximisant la fraude capturée.

**Consommateur principal** : Équipe Risques & Fraude d'un prestataire de services de paiement (PSP) français traitant ~1,2 million de transactions par jour, avec un data scientist qui construit le détecteur, des analystes fraude qui investiguent les alertes (capacité limitée : ~900 dossiers/jour) et un moteur de règles temps réel qui consomme les scores pour bloquer ou laisser passer.

---

## 2. Fiche d'identité

| Propriété | Valeur |
| --- | --- |
| Nom du dataset | `payment_transactions` |
| Granularité | une ligne = Identifiant unique de la transaction |
| Nombre d'échantillons (par défaut) | 12,000 |
| Nombre de colonnes | 22 |
| Taux de classe positive | ~1.8 % || Clé | `transaction_id` (unique) || Dimension temporelle | `occurred_at` || Formats | parquet, csv |
| Emplacement | `data/raw/payment_transactions.parquet` |
| Graine de génération | `42` |
| Valeurs manquantes | oui, en proportion maîtrisée (voir schéma) |
| Outliers | oui, légitimes métier (bornés par le schéma Pandera) |
| Source | **générée synthétiquement** (`SyntheticDataGenerator`) |

---

## 3. Schéma des colonnes

| # | Colonne | Type | Rôle | Unité | Signification métier | Distribution attendue |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `transaction_id` | `str` | identifier | - | Identifiant unique de la transaction | identifiants séquentiels uniques |
| 2 | `occurred_at` | `datetime` | timestamp | - | Horodatage de la tentative de paiement | 90 jours, profil horaire réaliste (creux nocturne, pics 12 h et 19 h) ; la fraude se concentre entre 1 h et 5 h |
| 3 | `amount_eur` | `float` | feature | EUR | Montant de la transaction | log-normale par catégorie marchand (médiane ~48 EUR) ; queue lourde pour la fraude et pour le luxe légitime |
| 4 | `merchant_category` | `category` | feature | - | Catégorie du marchand (MCC agrégé) | ~31 % grocery, 17 % restaurant, 14 % fashion, 12 % utilities, 10 % electronics, 7 % travel, 6 % gaming, 3 % jewelry |
| 5 | `channel` | `category` | feature | - | Canal de saisie du paiement | ~44 % mobile_app, 33 % web, 19 % in_store, 4 % phone ; la fraude est très majoritairement à distance |
| 6 | `shopper_country` | `category` | feature | - | Pays de la session d'achat | ~78 % FR, puis BE/DE/ES/IT/LU/PT et ~3 % other |
| 7 | `billing_country` | `category` | feature | - | Pays de facturation de la carte | ~86 % FR ; la divergence session/facturation est un signal fort mais pas suffisant |
| 8 | `card_age_months` | `int` | feature | mois | Ancienneté de la carte utilisée | ~gamma, médiane 26 mois ; cartes très récentes surreprésentées dans la fraude |
| 9 | `account_tenure_months` | `int` | feature | mois | Ancienneté du compte client chez le PSP | ~gamma, médiane 19 mois ; les prises de compte visent des comptes anciens |
| 10 | `transactions_24h` | `int` | feature | transactions | Nombre de tentatives de paiement sur la carte en 24 h | Poisson(lambda ~1,8) en légitime ; 8 à 40 en card testing |
| 11 | `distinct_merchants_24h` | `int` | feature | marchands | Marchands distincts contactés par la carte en 24 h | 1 à 3 en légitime ; élevé quand la carte est testée sur plusieurs marchands |
| 12 | `distinct_countries_7d` | `int` | feature | pays | Pays distincts observés sur la carte en 7 jours | 1 à 2 en légitime ; >= 3 en prise de compte (VPN, proxy) |
| 13 | `failed_attempts_1h` | `int` | feature | tentatives | Tentatives refusées dans l'heure précédente | 0 dans 88 % des cas légitimes ; 2 à 12 en card testing et identité synthétique |
| 14 | `device_age_days` | `float` | feature | jours | Âge de l'empreinte d'appareil (première fois vue) — flottant car la colonne est nullable | ~log-normale ; 4 % de manquants (empreinte bloquée) ; appareil neuf = risque accru |
| 15 | `session_duration_sec` | `float` | feature | secondes | Durée de la session avant paiement | ~log-normale (médiane 96 s) ; 6 % de manquants (paiement one-click) ; très courte en automatisé |
| 16 | `billing_shipping_distance_km` | `float` | feature | km | Distance entre adresses de facturation et de livraison | 0 km dans 71 % des cas ; ~5 % de manquants (retrait magasin) ; très élevée en prise de compte |
| 17 | `amount_to_customer_avg_ratio` | `float` | feature | ratio | Rapport du montant au panier moyen historique du porteur | ~log-normale centrée sur 1,0 ; > 6 en fraude et en achat exceptionnel légitime |
| 18 | `is_night` | `bool` | feature | - | Transaction entre 1 h et 5 h (heure locale) | ~9 % en légitime, ~46 % en fraude automatisée |
| 19 | `three_ds_authenticated` | `bool` | feature | - | Authentification forte 3-D Secure aboutie | ~72 % en légitime, ~19 % en fraude (contournement ou exemption) |
| 20 | `previous_chargebacks_12m` | `int` | feature | chargebacks | Chargebacks confirmés sur la carte dans les 12 derniers mois | 0 dans 97 % des cas légitimes ; 1 à 4 en fraude amicale et identité synthétique |
| 21 | `is_fraud` | `int` | metadata | - | Fraude confirmée par chargeback (étiquette de diagnostic, JAMAIS une feature) | ~1,8 % de la population |
| 22 | `fraud_scheme` | `str` | metadata | - | Mode opératoire de la fraude confirmée (diagnostic pédagogique et explication) | ~41 % card_not_present, 27 % account_takeover, 19 % synthetic_identity, 13 % friendly_fraud ; null si légitime |

### Contraintes de validation (contrat exécutable)

Les contraintes ci-dessous ne sont pas de la documentation : elles sont **exécutées** à chaque
chargement par les schémas Pandera de `src/data/schemas.py` (`RawDataSchema`,
`ProcessedDataSchema`, `InferenceDataSchema`).

| Colonne | Contraintes |
| --- | --- |
| `transaction_id` | type `str`, unique, non nul, motif ^TXN-[0-9]{7}$ |
| `occurred_at` | type `datetime`, non nul |
| `amount_eur` | type `float`, non nul, >= 0.5 · <= 12000.0 |
| `merchant_category` | type `category`, non nul, dans grocery, electronics, travel, gaming, jewelry, utilities, fashion, restaurant |
| `channel` | type `category`, non nul, dans web, mobile_app, in_store, phone |
| `shopper_country` | type `category`, non nul, dans FR, BE, DE, ES, IT, LU, PT, other |
| `billing_country` | type `category`, non nul, dans FR, BE, DE, ES, IT, LU, PT, other |
| `card_age_months` | type `int`, non nul, >= 0 · <= 120 |
| `account_tenure_months` | type `int`, non nul, >= 0 · <= 108 |
| `transactions_24h` | type `int`, non nul, >= 1 · <= 60 |
| `distinct_merchants_24h` | type `int`, non nul, >= 1 · <= 40 |
| `distinct_countries_7d` | type `int`, non nul, >= 1 · <= 9 |
| `failed_attempts_1h` | type `int`, non nul, >= 0 · <= 25 |
| `device_age_days` | type `float`, nullable, >= 0 · <= 1080 |
| `session_duration_sec` | type `float`, nullable, >= 2.0 · <= 3600.0 |
| `billing_shipping_distance_km` | type `float`, nullable, >= 0.0 · <= 14000.0 |
| `amount_to_customer_avg_ratio` | type `float`, non nul, >= 0.02 · <= 60.0 |
| `is_night` | type `bool`, non nul, dans 0, 1 |
| `three_ds_authenticated` | type `bool`, non nul, dans 0, 1 |
| `previous_chargebacks_12m` | type `int`, non nul, >= 0 · <= 6 |
| `is_fraud` | type `int`, non nul, dans 0, 1 |
| `fraud_scheme` | type `str`, nullable, dans card_not_present, account_takeover, synthetic_identity, friendly_fraud |

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
| `data/raw/payment_transactions.parquet` | données brutes (format machine, typé, compressé) |
| `data/raw/payment_transactions.csv` | mêmes données, lisibles par un humain |
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

- Valeurs manquantes volontaires sur `device_age_days` (~4 %), `session_duration_sec` (~6 %) et `billing_shipping_distance_km` (~5 %).
- Outliers légitimes (~2,5 % des lignes) : achats de luxe, paiements professionnels de fin de mois, voyageurs fréquents.
- Colinéarité volontaire entre `transactions_24h` et `distinct_merchants_24h`, et entre `amount_eur` et `amount_to_customer_avg_ratio`.
- Étiquettes générées par un processus latent par mode opératoire, avec bruit : une partie des fraudes est structurellement indétectable (plafond de performance).
- `is_fraud` et `fraud_scheme` sont déclarées `role: metadata` : exclues des features par `drop_columns`, utilisées uniquement par l'évaluateur.

---

## 7. Observations attendues (EDA)

Points à retrouver dans `notebooks/01_exploratory_analysis.ipynb` :

1. La prévalence est très faible (~1,8 %) : l'accuracy est inutile (98,2 % en prédisant « légitime »), il faut piloter PR AUC, rappel et précision **à budget fixé**.
2. Aucune variable ne suffit : `amount_eur` élevé est aussi le signe d'achats de luxe parfaitement légitimes (3 % du flux), d'où l'intérêt des ratios (montant / panier moyen) et de la vélocité.
3. La vélocité (`transactions_24h`, `distinct_merchants_24h`, `failed_attempts_1h`) est le signal le plus discriminant du card testing : un porteur légitime ne tente pas 25 paiements en une heure.
4. `three_ds_authenticated` est fortement protecteur mais **non absolu** : 19 % des fraudes passent l'authentification forte (prise de compte sur un appareil déjà connu).
5. La divergence `shopper_country` / `billing_country` est un signal classique, mais 14 % des transactions légitimes sont en déplacement professionnel : la règle brute génère des faux positifs massifs.
6. `device_age_days` et `session_duration_sec` comportent des manquants **informatifs** (empreinte bloquée, paiement one-click) : l'imputation doit être explicite et un indicateur de manquants est souvent plus utile que la valeur imputée.
7. Les quatre modes opératoires n'utilisent pas les mêmes variables : un détecteur global doit couvrir des signatures hétérogènes, ce qu'un jeu de règles figé rate.
8. L'étiquette arrive avec 30 à 90 jours de retard (chargeback) : c'est la raison structurelle de l'approche non supervisée, et non un choix par défaut.
9. `billing_shipping_distance_km` a une distribution bimodale (0 km ou très loin) : la traiter comme une variable continue gaussienne masque le signal.
10. Quelques transactions légitimes ressemblent exactement à de la fraude (achat de luxe nocturne à l'étranger) : le bruit irréductible fixe un plafond — un PR AUC de 1,0 signerait une fuite.
11. Le budget d'investigation (2 % du flux) est la vraie contrainte métier : le seuil se choisit sur la courbe rappel/précision en fonction du volume, pas sur un F1 abstrait.
12. Les features de vélocité sont calculées **avant** la transaction (fenêtre glissante strictement passée) : toute agrégation incluant la transaction courante créerait une fuite temporelle.

---

## 8. Passage à l'échelle (données réelles)

Pour brancher ce projet sur des données de production :

1. Remplacer `SyntheticDataGenerator` par un `RawDataLoader` qui lit la source réelle
   (entrepôt, objet storage, API) — l'interface ne change pas.
2. **Conserver les schémas Pandera** : ils deviennent le contrat avec le fournisseur de données
   (et un test de non-régression à chaque évolution du schéma amont).
3. Versionner les datasets (DVC / Delta / snapshots horodatés) pour garantir la reproductibilité.
4. Ajouter une surveillance de dérive (voir `mlops/model-monitoring/*`).
