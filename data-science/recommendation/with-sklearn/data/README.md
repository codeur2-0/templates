# Données — Journal de candidats scorés d'un catalogue e-commerce

> **Jeu de données synthétique** généré localement par `src/data/generators.py`.
> Aucune donnée réelle, aucun téléchargement, aucune information confidentielle.
> Objectif : disposer d'un dataset **crédible métier**, petit et reproductible, pour
> démontrer tout le cycle de vie (EDA → validation → preprocessing → entraînement → évaluation).

---

## 1. Description métier

Jeu d'apprentissage **par couple (utilisateur, article candidat)** : chaque ligne décrit un candidat
proposé à un utilisateur lors d'une session, avec les features disponibles à cet instant et la
pertinence observée ensuite. 1 200 utilisateurs x 30 candidats = 36 000 lignes. La pertinence est
construite à partir d'une affinité latente utilisateur-article (goût par catégorie, sensibilité au
prix, appétence à la nouveauté) bruitée, de sorte qu'aucun modèle ne puisse atteindre 1,0 : la part
irréductible est mesurée et annoncée. Les utilisateurs froids (peu d'historique) et la longue traîne
du catalogue sont volontairement surreprésentés par rapport à un journal réel, parce que c'est là
que se joue la qualité d'un moteur de recommandation.

**Contexte** : Le site expose chaque visiteur à un catalogue de plusieurs milliers de références. Le
classement historique est un tri par popularité : il écrase la longue traîne, ignore
l'affinité individuelle et recommande des articles indisponibles. Une refonte du moteur de
recommandation doit prouver son apport **hors échantillon et par utilisateur**, pas
seulement sur une moyenne globale.

**Problème adressé** : Classer les articles candidats d'un utilisateur — les scorer puis publier un top-K — à
partir des interactions passées, de la fiche article et du profil utilisateur, sans jamais
utiliser d'information postérieure à la session scorée.

**Consommateur principal** : Équipe Personnalisation d'un e-commerçant : data scientist qui industrialise le modèle de classement, responsable CRM qui pilote le taux de conversion et le panier moyen, et équipe catalogue qui surveille l'exposition de la longue traîne et la disponibilité des références.

---

## 2. Fiche d'identité

| Propriété | Valeur |
| --- | --- |
| Nom du dataset | `ecommerce_candidate_impressions` |
| Granularité | une ligne = Identifiant unique du couple (utilisateur, article candidat) |
| Nombre d'échantillons (par défaut) | 36 000 |
| Nombre de colonnes | 33 |
| Cible | `relevance` || Clé | `sample_id` (unique) || Dimension temporelle | `session_date` || Formats | parquet, csv |
| Emplacement | `data/raw/ecommerce_candidate_impressions.parquet` |
| Graine de génération | `42` |
| Valeurs manquantes | oui, en proportion maîtrisée (voir schéma) |
| Outliers | oui, légitimes métier (bornés par le schéma Pandera) |
| Source | **générée synthétiquement** (`SyntheticDataGenerator`) |

---

## 3. Schéma des colonnes

| # | Colonne | Type | Rôle | Unité | Signification métier | Distribution attendue |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `sample_id` | `str` | identifier | - | Identifiant unique du couple (utilisateur, article candidat) | identifiants séquentiels déterministes |
| 2 | `user_id` | `str` | group | - | Identifiant utilisateur — clé de regroupement des métriques de classement, jamais une feature | 1 200 utilisateurs, 30 candidats chacun ; fréquence uniforme par construction |
| 3 | `item_id` | `str` | identifier | - | Identifiant article du catalogue — jamais une feature, pour les mêmes raisons | catalogue de 900 références, tirage des candidats pondéré par l'affinité et la popularité |
| 4 | `session_date` | `datetime` | timestamp | - | Date de la session au cours de laquelle le candidat a été scoré — clé du split chronologique | étalement sur ~14 mois, avec une saisonnalité hebdomadaire et un pic de fin d'année |
| 5 | `user_tenure_days` | `int` | feature | jours | Ancienneté du compte utilisateur en jours | loi puissance tronquée : beaucoup de comptes récents, peu de comptes anciens |
| 6 | `user_orders_12m` | `int` | feature | commandes | Nombre de commandes de l'utilisateur sur les 12 derniers mois — pilote le segment froid / tiède / chaud | Poisson de moyenne ~4,5, fortement asymétrique |
| 7 | `user_spend_eur_12m` | `float` | feature | EUR | Dépense cumulée de l'utilisateur sur 12 mois | log-normale corrélée au nombre de commandes |
| 8 | `user_sessions_30d` | `int` | feature | sessions | Sessions ouvertes sur les 30 derniers jours — intensité d'usage récente | Poisson de moyenne ~9 |
| 9 | `user_distinct_categories_12m` | `int` | feature | catégories | Nombre de catégories distinctes achetées sur 12 mois — amplitude d'exploration | binomiale tronquée, corrélée positivement aux commandes |
| 10 | `user_avg_basket_eur` | `float` | feature | EUR | Panier moyen de l'utilisateur | log-normale de médiane ~65 EUR |
| 11 | `user_typical_price_eur` | `float` | feature | EUR | Prix médian des articles achetés par l'utilisateur — référence pour mesurer l'écart de prix d'un candidat | log-normale, proche du panier moyen mais moins dispersée |
| 12 | `user_return_rate` | `float` | feature | ratio | Part des commandes retournées sur 12 mois — un retour fréquent signale une pertinence mal calibrée | bêta centrée sur ~0,08, masse en 0 pour les utilisateurs sans retour |
| 13 | `user_price_band_pref` | `category` | feature | - | Gamme de prix préférée de l'utilisateur | 55 % milieu, 30 % entrée, 15 % premium |
| 14 | `user_channel` | `category` | feature | - | Canal principal d'achat (le classement mobile privilégie les fiches courtes) | 45 % app, 33 % mobile, 22 % web |
| 15 | `user_region` | `str` | feature | - | Région de livraison — effet sur la disponibilité et les délais | 34 % IDF, 16 à 18 % par autre région |
| 16 | `user_is_member` | `bool` | feature | - | Adhésion au programme de fidélité | 38 % de membres, plus fréquents chez les utilisateurs chauds |
| 17 | `item_category` | `category` | feature | - | Catégorie catalogue de l'article | 8 catégories, de 8 % à 19 % du catalogue |
| 18 | `item_price_eur` | `float` | feature | EUR | Prix de vente de l'article | log-normale par catégorie (médiane ~45 EUR, high_tech plus cher) |
| 19 | `item_rating_avg` | `float` | feature | /5 | Note moyenne de l'article | bêta recentrée sur ~4,1, peu d'articles sous 2,5 |
| 20 | `item_reviews_count` | `int` | feature | avis | Nombre d'avis publiés — preuve sociale, mais aussi proxy de l'âge de l'article | loi puissance : quelques articles très commentés, une longue traîne à 0 avis |
| 21 | `item_stock_units` | `int` | feature | unités | Stock disponible au moment du scoring — un stock nul interdit la publication | 7 % des candidats en rupture, plus fréquent sur les articles très demandés |
| 22 | `item_age_days` | `int` | feature | jours | Ancienneté de l'article au catalogue | exponentielle de moyenne ~300 jours |
| 23 | `item_margin_pct` | `float` | feature | % | Marge brute de l'article — permet d'arbitrer explicitement pertinence contre valeur | normale de moyenne ~34 %, plus faible en high_tech, plus forte en beauté |
| 24 | `item_views_7d` | `int` | feature | vues | Vues de l'article sur les 7 jours **précédant** la session — fenêtre strictement antérieure, donc sans fuite | loi puissance ; sert aussi de score à la baseline popularité |
| 25 | `item_conversion_rate_30d` | `float` | feature | ratio | Taux de conversion de la fiche sur 30 jours glissants antérieurs | bêta centrée sur ~0,045, plus élevée sur les articles bien notés |
| 26 | `item_is_promoted` | `bool` | feature | - | Article poussé par une opération marketing au moment de la session | 33,4 % de candidats en promotion : 18 % de références poussées en fond de catalogue, plus un calendrier d'animation chargé en juin, novembre et décembre |
| 27 | `user_category_affinity` | `float` | feature | ratio | Part des dépenses de l'utilisateur dans la catégorie de l'article — le signal croisé le plus fort | bimodale : 62,6 % de zéros (catégories jamais achetées) et 35,0 % au-dessus de 0,3 |
| 28 | `user_brand_affinity` | `float` | feature | ratio | Affinité de l'utilisateur à la marque de l'article (historique d'achats et de consultations) | bêta de mode ~0,15, masse importante en 0 |
| 29 | `price_gap_pct` | `float` | feature | % | Écart relatif entre le prix du candidat et le prix habituel de l'utilisateur — un écart fort fait chuter la pertinence | centrée sur ~0, asymétrique à droite (les candidats plus chers sont plus fréquents) |
| 30 | `days_since_last_view` | `int` | feature | jours | Jours écoulés depuis la dernière consultation de cet article par cet utilisateur | 62,1 % de candidats jamais consultés (valeur manquante), le reste en décroissance rapide de médiane 18 jours |
| 31 | `user_item_views_30d` | `int` | feature | vues | Consultations de ce candidat par cet utilisateur sur 30 jours glissants antérieurs | 72,5 % de zéros, queue fine — signal d'intention rare mais très prédictif |
| 32 | `similar_users_buy_rate` | `float` | feature | ratio | Taux d'achat de ce candidat par les utilisateurs les plus proches (signal collaboratif précalculé sur fenêtre antérieure) | bêta centrée sur ~0,06, corrélée à l'affinité catégorie |
| 33 | `relevance` | `int` | target | - | Pertinence observée : 1 si l'utilisateur a ajouté au panier ou acheté, 0 sinon | 13,9 % de positifs au global, 14,6 % sur la fenêtre d'entraînement et 11,5 % sur la fenêtre de test (creux d'été) — la saisonnalité se voit dans la prévalence |

### Contraintes de validation (contrat exécutable)

Les contraintes ci-dessous ne sont pas de la documentation : elles sont **exécutées** à chaque
chargement par les schémas Pandera de `src/data/schemas.py` (`RawDataSchema`,
`ProcessedDataSchema`, `InferenceDataSchema`).

| Colonne | Contraintes |
| --- | --- |
| `sample_id` | type `str`, unique, non nul, motif ^RC-[0-9]{6}$ |
| `user_id` | type `str`, non nul, motif ^U-[0-9]{4}$ |
| `item_id` | type `str`, non nul, motif ^I-[0-9]{5}$ |
| `session_date` | type `datetime`, non nul |
| `user_tenure_days` | type `int`, non nul, >= 0 · <= 3000 |
| `user_orders_12m` | type `int`, non nul, >= 0 · <= 120 |
| `user_spend_eur_12m` | type `float`, non nul, >= 0.0 · <= 12000.0 |
| `user_sessions_30d` | type `int`, non nul, >= 0 · <= 200 |
| `user_distinct_categories_12m` | type `int`, non nul, >= 0 · <= 12 |
| `user_avg_basket_eur` | type `float`, non nul, >= 0.0 · <= 900.0 |
| `user_typical_price_eur` | type `float`, non nul, >= 0.0 · <= 900.0 |
| `user_return_rate` | type `float`, non nul, >= 0.0 · <= 1.0 |
| `user_price_band_pref` | type `category`, non nul, dans entree, milieu, premium |
| `user_channel` | type `category`, non nul, dans web, mobile, app |
| `user_region` | type `str`, non nul, dans nord, sud, est, ouest, idf |
| `user_is_member` | type `bool`, non nul |
| `item_category` | type `category`, non nul, dans mode, maison, high_tech, sport, beaute, alimentaire, jouet, jardin |
| `item_price_eur` | type `float`, non nul, >= 1.0 · <= 2500.0 |
| `item_rating_avg` | type `float`, non nul, >= 1.0 · <= 5.0 |
| `item_reviews_count` | type `int`, non nul, >= 0 · <= 6000 |
| `item_stock_units` | type `int`, non nul, >= 0 · <= 900 |
| `item_age_days` | type `int`, non nul, >= 0 · <= 2200 |
| `item_margin_pct` | type `float`, non nul, >= 0.0 · <= 70.0 |
| `item_views_7d` | type `int`, non nul, >= 0 · <= 40000 |
| `item_conversion_rate_30d` | type `float`, non nul, >= 0.0 · <= 0.6 |
| `item_is_promoted` | type `bool`, non nul |
| `user_category_affinity` | type `float`, non nul, >= 0.0 · <= 1.0 |
| `user_brand_affinity` | type `float`, non nul, >= 0.0 · <= 1.0 |
| `price_gap_pct` | type `float`, non nul, >= -95.0 · <= 400.0 |
| `days_since_last_view` | type `int`, nullable, >= 0 · <= 420 |
| `user_item_views_30d` | type `int`, non nul, >= 0 · <= 40 |
| `similar_users_buy_rate` | type `float`, non nul, >= 0.0 · <= 0.5 |
| `relevance` | type `int`, non nul, dans 0, 1 |

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
| `data/raw/ecommerce_candidate_impressions.parquet` | données brutes (format machine, typé, compressé) |
| `data/raw/ecommerce_candidate_impressions.csv` | mêmes données, lisibles par un humain |
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

- Le jeu est un **journal de candidats scorés**, pas une table d'interactions brutes : l'échantillonnage des négatifs est fait en amont par le générateur (30 candidats par utilisateur, dont ~14 % de pertinents), ce qui est le protocole standard d'entraînement d'un modèle de classement.
- Les 30 candidats d'un utilisateur sont tirés à la même date de session : le classement se compare donc à nombre de candidats égal entre utilisateurs, ce qui rend Precision@K et Recall@K comparables. Trente candidats pour un top-10 est un minimum : à 12 candidats le top-10 couvre presque toute la liste, le NDCG sature et même un scoreur aléatoire semble correct (mesuré : NDCG@10 aléatoire de 0,47 à 12 candidats contre 0,23 à 30).
- La pertinence est générée à partir d'une affinité latente bruitée : le plafond atteignable est inférieur à 1,0 et le rapport d'évaluation l'annonce explicitement pour éviter de poursuivre un bruit.
- Toutes les statistiques article sont des fenêtres glissantes **antérieures** à la session ; la construction est vérifiée par un test de fuite qui recompose une statistique postérieure et montre qu'elle améliore artificiellement les métriques.

---

## 7. Observations attendues (EDA)

Points à retrouver dans `notebooks/01_exploratory_analysis.ipynb` :

1. La pertinence se joue presque entièrement dans les **signaux croisés** : `user_category_affinity`, `price_gap_pct` et `user_item_views_30d` portent l'essentiel du signal, alors que les features purement utilisateur ou purement article ne distinguent pas un bon candidat d'un mauvais.
2. `item_views_7d` suit une loi puissance très marquée : classer par cette seule colonne reproduit le moteur historique. C'est une **vraie** baseline — son NDCG@10 de 0,334 bat nettement l'aléatoire (0,230), parce que l'audience observe un attrait latent qui profite aussi à la pertinence — mais elle n'expose que 8,4 % du catalogue dans ses top-10, contre plus de la moitié pour un modèle qui combine attrait et affinité individuelle.
3. 72,5 % des candidats n'ont jamais été consultés par l'utilisateur (`user_item_views_30d = 0`) et 62,1 % n'ont jamais été vus du tout (`days_since_last_view` manquant) : le moteur travaille majoritairement en exploration, pas en exploitation. Le signal d'intention est rare mais très prédictif quand il existe.
4. Les utilisateurs froids (au plus 1 commande sur 12 mois) représentent ~29 % des lignes mais une part bien plus faible des positifs : sans traitement dédié, un modèle global les classe mal et le moteur perd précisément les clients à convertir.
5. `item_stock_units = 0` sur ~7 % des candidats, et davantage sur les articles les plus populaires : filtrer après scoring coûte donc de la pertinence mesurée, ce qui doit être chiffré plutôt que supposé nul.
6. `price_gap_pct` a un effet en cloche inversée : un candidat très en dessous du prix habituel n'est pas mieux classé qu'un candidat aligné, ce qu'un modèle linéaire ne capture pas et qu'un arbre capture sans feature supplémentaire.

---

## 8. Passage à l'échelle (données réelles)

Pour brancher ce projet sur des données de production :

1. Remplacer `SyntheticDataGenerator` par un `RawDataLoader` qui lit la source réelle
   (entrepôt, objet storage, API) — l'interface ne change pas.
2. **Conserver les schémas Pandera** : ils deviennent le contrat avec le fournisseur de données
   (et un test de non-régression à chaque évolution du schéma amont).
3. Versionner les datasets (DVC / Delta / snapshots horodatés) pour garantir la reproductibilité.
4. Ajouter une surveillance de dérive (voir `mlops/model-monitoring/*`).
