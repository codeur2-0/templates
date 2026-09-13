# Données — Biens immobiliers résidentiels et prix de vente

> **Jeu de données synthétique** généré localement par `src/data/generators.py`.
> Aucune donnée réelle, aucun téléchargement, aucune information confidentielle.
> Objectif : disposer d'un dataset **crédible métier**, petit et reproductible, pour
> démontrer tout le cycle de vie (EDA → validation → preprocessing → entraînement → évaluation).

---

## 1. Description métier

Population de biens résidentiels mis en annonce sur une métropole française, avec leurs
caractéristiques physiques, énergétiques, géographiques et administratives, et le prix de vente net
vendeur observé. Les données sont générées par un modèle multiplicatif en log-prix : chaque
caractéristique contribue par un facteur explicite (prime de quartier, effet surface en log, malus
énergétique, effet étage), puis un bruit log-normal irréductible (sigma = 7 %) est ajouté. Le signal
est réaliste — hétéroscédastique, non linéaire, avec colinéarités — et plafonne la performance
atteignable.

**Contexte** : La plateforme promet une estimation de prix en moins de trente secondes. Aujourd'hui, cette
estimation repose sur un prix au m² communal publié par trimestre et sur le regard d'un
négociateur : le résultat est lent à rafraîchir, hétérogène d'une agence à l'autre, et
systématiquement aveugle à l'état du bien, à son étage ou à sa performance énergétique. Les
biens sur-évalués stagnent en portefeuille (quatre-vingt-dix jours en moyenne), les biens
sous-évalués partent en quelques jours et laissent de la valeur sur la table.

**Problème adressé** : Prédire le prix de vente net vendeur d'un bien résidentiel à partir de sa fiche descriptive,
et publier une **fourchette** assortie d'un niveau de confiance, plutôt qu'un prix unique
présenté comme exact.

**Consommateur principal** : Équipe Data d'une plateforme immobilière en ligne (estimation grand public + réseau d'agences partenaires), avec un data scientist qui industrialise le modèle, un produit web qui consomme l'estimation en temps réel et des négociateurs qui l'utilisent en rendez-vous de rentrée de mandat.

---

## 2. Fiche d'identité

| Propriété | Valeur |
| --- | --- |
| Nom du dataset | `real_estate_prices` |
| Granularité | une ligne = Identifiant unique du bien |
| Nombre d'échantillons (par défaut) | 8,000 |
| Nombre de colonnes | 16 |
| Cible | `price_eur` || Clé | `property_id` (unique) || Formats | parquet, csv |
| Emplacement | `data/raw/real_estate_prices.parquet` |
| Graine de génération | `42` |
| Valeurs manquantes | oui, en proportion maîtrisée (voir schéma) |
| Outliers | oui, légitimes métier (bornés par le schéma Pandera) |
| Source | **générée synthétiquement** (`SyntheticDataGenerator`) |

---

## 3. Schéma des colonnes

| # | Colonne | Type | Rôle | Unité | Signification métier | Distribution attendue |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `property_id` | `str` | identifier | - | Identifiant unique du bien | identifiants séquentiels uniques |
| 2 | `listing_date` | `datetime` | timestamp | - | Date de mise en annonce du bien | étalement sur ~3 ans, avec une saisonnalité printanière |
| 3 | `surface_m2` | `float` | feature | m² | Surface habitable | log-normale centrée ~63 m², queue droite étirée (grandes maisons) |
| 4 | `rooms` | `int` | feature | pièces | Nombre de pièces principales | ≈ surface / 22 + bruit (fortement corrélée à la surface, colinéarité volontaire) |
| 5 | `floor_level` | `int` | feature | étage | Étage du bien (0 = rez-de-chaussée) | ~38 % au niveau 0-1, décroissance jusqu'au 12e |
| 6 | `has_elevator` | `bool` | feature | - | Présence d'un ascenseur dans l'immeuble (1 = oui) | ~52 % avec ascenseur, quasi systématique au-delà du 4e étage |
| 7 | `has_outdoor_space` | `bool` | feature | - | Balcon, terrasse ou jardin (1 = oui) | ~34 % des biens |
| 8 | `building_year` | `int` | feature | année | Année de construction de l'immeuble | mélange d'époques : haussmannien (1850-1914), entre-deux-guerres, Trente Glorieuses, récent |
| 9 | `energy_rating` | `category` | feature | - | Classe énergétique du diagnostic de performance énergétique (DPE) | ~2 % A, ~12 % B, ~24 % C, ~32 % D, ~18 % E, ~9 % F, ~3 % G |
| 10 | `district` | `category` | feature | - | Quartier (segment géographique) de rattachement du bien | parts inégales : 14 % centre, 12 % bords, 18 % université, 11 % affaires, 23 % nord, 22 % sud |
| 11 | `transport_walk_min` | `int` | feature | minutes | Temps d'accès piéton au transport structurant le plus proche | gamma centrée ~9 min en centre, ~24 min en périphérie |
| 12 | `condo_fees_eur` | `float` | feature | EUR/mois | Charges de copropriété mensuelles | proportionnelle à la surface et aux services ; ~4 % de manquants (bien non renseigné) |
| 13 | `property_tax_eur` | `float` | feature | EUR/an | Taxe foncière annuelle | ≈ fonction de la surface et du quartier (colinéarité volontaire avec surface_m2) |
| 14 | `condition_score` | `float` | feature | /10 | État déclaré par le vendeur (1 = à rénover entièrement, 10 = haut de gamme rénové) | normale tronquée ~6.2 ; ~6 % de manquants (diagnostic non rempli) |
| 15 | `recent_sales_1km` | `int` | feature | ventes | Nombre de ventes comparables enregistrées à moins d'un kilomètre sur 12 mois | Poisson selon le quartier : marché dense en centre, marché fin en périphérie |
| 16 | `price_eur` | `float` | target | EUR | Prix de vente net vendeur observé | log-normale : médiane ~265 000 EUR, moyenne ~330 000 EUR, queue au-delà de 1 M EUR |

### Contraintes de validation (contrat exécutable)

Les contraintes ci-dessous ne sont pas de la documentation : elles sont **exécutées** à chaque
chargement par les schémas Pandera de `src/data/schemas.py` (`RawDataSchema`,
`ProcessedDataSchema`, `InferenceDataSchema`).

| Colonne | Contraintes |
| --- | --- |
| `property_id` | type `str`, unique, non nul, motif ^PRP-[0-9]{6}$ |
| `listing_date` | type `datetime`, non nul |
| `surface_m2` | type `float`, non nul, >= 12.0 · <= 320.0 |
| `rooms` | type `int`, non nul, >= 1 · <= 8 |
| `floor_level` | type `int`, non nul, >= 0 · <= 12 |
| `has_elevator` | type `bool`, non nul, dans 0, 1 |
| `has_outdoor_space` | type `bool`, non nul, dans 0, 1 |
| `building_year` | type `int`, non nul, >= 1850 · <= 2025 |
| `energy_rating` | type `category`, non nul, dans A, B, C, D, E, F, G |
| `district` | type `category`, non nul, dans centre_historique, bords_de_loire, quartier_universitaire, quartier_affaires, peripherie_nord, peripherie_sud |
| `transport_walk_min` | type `int`, non nul, >= 1 · <= 45 |
| `condo_fees_eur` | type `float`, nullable, >= 0.0 · <= 900.0 |
| `property_tax_eur` | type `float`, non nul, >= 120.0 · <= 6500.0 |
| `condition_score` | type `float`, nullable, >= 1.0 · <= 10.0 |
| `recent_sales_1km` | type `int`, non nul, >= 0 · <= 140 |
| `price_eur` | type `float`, non nul, >= 35000.0 · <= 2500000.0 |

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
| `data/raw/real_estate_prices.parquet` | données brutes (format machine, typé, compressé) |
| `data/raw/real_estate_prices.csv` | mêmes données, lisibles par un humain |
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

- Valeurs manquantes volontaires sur `condition_score` (~6 %) et `condo_fees_eur` (~4 %) pour exercer l'imputation.
- Outliers légitimes (~1 %) : biens d'exception (grande surface, centre historique, état haut de gamme).
- Colinéarité volontaire entre `surface_m2`, `rooms` et `property_tax_eur`.
- Cible générée par un modèle multiplicatif en log-prix avec interactions (étage x ascenseur, surface concave, malus DPE non linéaire).
- Bruit irréductible log-normal (σ ≈ 0.07) : un prix parfait indiquerait une fuite de données.

---

## 7. Observations attendues (EDA)

Points à retrouver dans `notebooks/01_exploratory_analysis.ipynb` :

1. La cible est **log-normale** : travailler en log-prix (ou transformer la cible) stabilise la variance et rend les erreurs comparables entre un studio et une grande maison.
2. Le quartier est le premier facteur de prix (écart d'un facteur ~2 entre centre historique et périphérie nord) : c'est aussi un proxy socio-économique, à documenter sous l'angle équité.
3. L'effet de la surface est **concave** : le prix au m² décroît avec la taille. Un modèle linéaire en surface sur-évalue systématiquement les grands biens et sous-évalue les studios.
4. `rooms` et `property_tax_eur` sont quasi redondants avec `surface_m2` : colinéarité attendue, pénalisante pour un modèle linéaire régularisé, neutre pour les arbres.
5. Le DPE pèse de plus en plus : une classe F ou G entraîne une décote explicite (interdiction progressive de location des passoires thermiques), non linéaire entre E et G.
6. L'étage ne vaut que **si** l'immeuble dispose d'un ascenseur : l'interaction étage x ascenseur change le signe de l'effet (un 6e sans ascenseur est décoté, un 6e avec ascenseur est primé).
7. L'année de construction a un effet en U : l'haussmannien et le très récent sont primés, les constructions des années 1960-1975 sont décotées.
8. `transport_walk_min` a un effet décroissant : chaque minute compte beaucoup jusqu'à 10 minutes, presque plus au-delà de 25.
9. Les erreurs sont **hétéroscédastiques** : l'erreur absolue croît avec le prix. Piloter uniquement la RMSE conduit à sacrifier les petits biens ; le MAPE et l'erreur médiane sont plus justes.
10. `recent_sales_1km` n'explique pas le prix mais la **fiabilité** de l'estimation : un marché fin (< 10 ventes) doit produire une fourchette plus large, pas un prix plus précis.
11. Quelques biens d'exception existent légitimement (> 1,2 M EUR) : les supprimer appauvrirait le modèle, le winsorising (0,5-99,5 %) et une analyse d'erreur dédiée sont préférables.
12. `condition_score` et `condo_fees_eur` comportent des manquants volontaires (6 % et 4 %) : l'imputation doit être explicite, et un indicateur de manquant est souvent rentable.
13. Le bruit injecté (multiplicateur log-normal d'environ 7 %) fixe un plafond : un MAPE proche de 0 signerait une fuite de données, pas un bon modèle. Même un modèle parfait ne couvre que ~82 % des biens à ±10 %.

---

## 8. Passage à l'échelle (données réelles)

Pour brancher ce projet sur des données de production :

1. Remplacer `SyntheticDataGenerator` par un `RawDataLoader` qui lit la source réelle
   (entrepôt, objet storage, API) — l'interface ne change pas.
2. **Conserver les schémas Pandera** : ils deviennent le contrat avec le fournisseur de données
   (et un test de non-régression à chaque évolution du schéma amont).
3. Versionner les datasets (DVC / Delta / snapshots horodatés) pour garantir la reproductibilité.
4. Ajouter une surveillance de dérive (voir `mlops/model-monitoring/*`).
