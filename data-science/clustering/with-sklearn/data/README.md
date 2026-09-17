# Données — Base clients e-commerce et comportement d'achat sur 12 mois

> **Jeu de données synthétique** généré localement par `src/data/generators.py`.
> Aucune donnée réelle, aucun téléchargement, aucune information confidentielle.
> Objectif : disposer d'un dataset **crédible métier**, petit et reproductible, pour
> démontrer tout le cycle de vie (EDA → validation → preprocessing → entraînement → évaluation).

---

## 1. Description métier

Population de clients d'un e-commerçant (mode et maison) avec leurs agrégats comportementaux sur
douze mois glissants. Les données sont générées à partir de **six profils latents** (VIP fidèle,
chasseur de promotions, acheteur occasionnel, dormeur, nouveau curieux, client insatisfait) : chaque
profil définit des distributions propres (fréquence, panier, part promotionnelle, taux de retour,
engagement), puis un bruit individuel et des chevauchements volontaires sont injectés. La structure
est donc réelle mais **non triviale** : les groupes se recouvrent partiellement, ce qui rend le
choix du nombre de clusters et la lecture de la silhouette authentiquement difficiles. Deux colonnes
de métadonnées (segment latent et churn observé à 90 jours) permettent de mesurer la qualité de la
segmentation sans jamais entrer dans les features.

**Contexte** : La segmentation actuelle est artisanale : trois seuils RFM (récence, fréquence, montant)
posés à la main dans un tableur, revus une fois par an. Elle classe 62 % des clients dans un
seul groupe « standard », ignore complètement le comportement promotionnel, le taux de
retour et la pression sur le service client, et ne dit rien des nouveaux inscrits qui n'ont
pas encore commandé. Résultat : les campagnes sont indifférenciées, le coût d'acquisition
augmente, et le churn des clients à fort potentiel n'est détecté qu'après six mois
d'inactivité.

**Problème adressé** : Découvrir **sans étiquette préalable** des groupes de clients homogènes et actionnables à
partir de leur comportement d'achat et d'engagement sur douze mois glissants, choisir le
nombre de groupes de façon argumentée (et non arbitraire), profiler chaque groupe, puis
affecter tout nouveau client au groupe le plus proche avec un niveau de confiance.

**Consommateur principal** : Équipe CRM & Data d'un e-commerçant français (mode et maison, ~250 000 clients actifs), avec un data scientist qui construit la segmentation, un responsable CRM qui pilote les campagnes d'acquisition et de rétention, et un outil de marketing automation qui consomme le segment de chaque client pour choisir le message, le canal et la pression publicitaire.

---

## 2. Fiche d'identité

| Propriété | Valeur |
| --- | --- |
| Nom du dataset | `retail_customer_base` |
| Granularité | une ligne = Identifiant unique du client |
| Nombre d'échantillons (par défaut) | 6 000 |
| Nombre de colonnes | 22 |
| Clé | `customer_id` (unique) || Dimension temporelle | `signup_date` || Formats | parquet, csv |
| Emplacement | `data/raw/retail_customer_base.parquet` |
| Graine de génération | `42` |
| Valeurs manquantes | oui, en proportion maîtrisée (voir schéma) |
| Outliers | oui, légitimes métier (bornés par le schéma Pandera) |
| Source | **générée synthétiquement** (`SyntheticDataGenerator`) |

---

## 3. Schéma des colonnes

| # | Colonne | Type | Rôle | Unité | Signification métier | Distribution attendue |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `customer_id` | `str` | identifier | - | Identifiant unique du client | identifiants séquentiels uniques |
| 2 | `signup_date` | `datetime` | timestamp | - | Date d'inscription (première création de compte) | étalement sur ~5 ans, accéléré par les campagnes d'acquisition |
| 3 | `last_order_date` | `datetime` | timestamp | - | Date de la dernière commande | découlée de la récence ; jamais postérieure à la date de référence |
| 4 | `tenure_months` | `int` | feature | mois | Ancienneté du compte en mois | ≈ (date de référence - signup_date) ; forte densité sur les 24 derniers mois |
| 5 | `recency_days` | `int` | feature | jours | Nombre de jours depuis la dernière commande | bimodale : acheteurs actifs (< 90 j) et dormeurs (> 250 j) |
| 6 | `orders_12m` | `int` | feature | commandes | Nombre de commandes sur 12 mois glissants | Poisson selon le profil latent : ~0-1 pour un dormeur, ~12-20 pour un VIP |
| 7 | `revenue_12m_eur` | `float` | feature | EUR | Chiffre d'affaires réalisé sur 12 mois glissants | log-normale : médiane ~480 EUR, queue droite (VIP et clients d'exception) |
| 8 | `avg_basket_eur` | `float` | feature | EUR | Panier moyen (chiffre d'affaires / nombre de commandes) | ≈ revenue / orders avec bruit ; manquant quand orders_12m = 0 |
| 9 | `distinct_categories_12m` | `int` | feature | catégories | Nombre de catégories de produits achetées (largeur du catalogue) | croît avec la fréquence ; ~1-2 pour un occasionnel, 6-10 pour un VIP |
| 10 | `discount_share` | `float` | feature | % | Part du chiffre d'affaires réalisée avec une remise | bêta : ~0.15 pour un VIP, ~0.75 pour un chasseur de promotions |
| 11 | `return_rate` | `float` | feature | % | Taux de retour (articles retournés / articles commandés) | bêta centrée ~0.12 ; élevée (> 0.35) pour le profil insatisfait |
| 12 | `support_tickets_12m` | `int` | feature | tickets | Nombre de contacts au service client (réclamation, SAV, question livraison) | Poisson ~0.4 en général, ~3-5 pour le profil insatisfait |
| 13 | `newsletter_opens_12m` | `int` | feature | ouvertures | Nombre d'ouvertures de newsletter sur 12 mois | binomiale selon le consentement et le profil : très élevée chez les chasseurs de promo |
| 14 | `web_sessions_12m` | `int` | feature | sessions | Nombre de sessions web ou application sur 12 mois | Poisson : élevé chez les nouveaux curieux (beaucoup de visites, peu d'achats) |
| 15 | `mobile_share` | `float` | feature | % | Part des sessions réalisées sur mobile | bêta centrée ~0.62 ; plus élevée chez les inscrits récents (dérive d'usage) |
| 16 | `loyalty_tier` | `category` | feature | - | Palier du programme de fidélité | ~48 % none, ~28 % silver, ~18 % gold, ~6 % platinum (dépend du CA cumulé) |
| 17 | `acquisition_channel` | `category` | feature | - | Canal d'acquisition du client | ~26 % organic, ~22 % paid_search, ~19 % social, ~14 % marketplace, ~11 % referral, ~8 % email |
| 18 | `region` | `category` | feature | - | Région de livraison principale | ~24 % IDF, ~16 % nord, ~17 % ouest, ~13 % sud_ouest, ~18 % sud_est, ~12 % est |
| 19 | `nps_score` | `int` | feature | /10 | Note de recommandation déclarée (Net Promoter Score individuel) | ~8 % de manquants (enquête non répondue) ; bimodale : détracteurs et promoteurs |
| 20 | `opt_in_marketing` | `bool` | feature | - | Consentement aux communications marketing (1 = oui) | ~71 % de consentants ; quasi systématique chez les chasseurs de promotions |
| 21 | `latent_segment` | `category` | metadata | - | Profil latent injecté par le générateur — diagnostic pédagogique uniquement | ~8 % VIP, ~18 % promo, ~32 % occasionnel, ~20 % dormeur, ~14 % nouveau, ~8 % insatisfait |
| 22 | `churned_next_90d` | `bool` | metadata | - | Aucune commande dans les 90 jours suivant la date de référence (validité externe) | ~4 % chez les VIP, ~55 % chez les dormeurs ; écart > 20 points entre segments |

### Contraintes de validation (contrat exécutable)

Les contraintes ci-dessous ne sont pas de la documentation : elles sont **exécutées** à chaque
chargement par les schémas Pandera de `src/data/schemas.py` (`RawDataSchema`,
`ProcessedDataSchema`, `InferenceDataSchema`).

| Colonne | Contraintes |
| --- | --- |
| `customer_id` | type `str`, unique, non nul, motif ^CUS-[0-9]{6}$ |
| `signup_date` | type `datetime`, non nul |
| `last_order_date` | type `datetime`, non nul |
| `tenure_months` | type `int`, non nul, >= 0 · <= 62 |
| `recency_days` | type `int`, non nul, >= 1 · <= 730 |
| `orders_12m` | type `int`, non nul, >= 0 · <= 60 |
| `revenue_12m_eur` | type `float`, non nul, >= 0.0 · <= 15000.0 |
| `avg_basket_eur` | type `float`, nullable, >= 0.0 · <= 950.0 |
| `distinct_categories_12m` | type `int`, non nul, >= 0 · <= 12 |
| `discount_share` | type `float`, non nul, >= 0.0 · <= 1.0 |
| `return_rate` | type `float`, non nul, >= 0.0 · <= 1.0 |
| `support_tickets_12m` | type `int`, non nul, >= 0 · <= 15 |
| `newsletter_opens_12m` | type `int`, non nul, >= 0 · <= 120 |
| `web_sessions_12m` | type `int`, non nul, >= 0 · <= 400 |
| `mobile_share` | type `float`, non nul, >= 0.0 · <= 1.0 |
| `loyalty_tier` | type `category`, non nul, dans none, silver, gold, platinum |
| `acquisition_channel` | type `category`, non nul, dans organic, paid_search, social, marketplace, referral, email |
| `region` | type `category`, non nul, dans ile_de_france, nord, ouest, sud_ouest, sud_est, est |
| `nps_score` | type `int`, nullable, >= 0 · <= 10 |
| `opt_in_marketing` | type `bool`, non nul, dans 0, 1 |
| `latent_segment` | type `category`, non nul, dans vip_fidele, chasseur_promo, acheteur_occasionnel, dormeur, nouveau_curieux, client_insatisfait |
| `churned_next_90d` | type `bool`, non nul, dans 0, 1 |

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
| `data/raw/retail_customer_base.parquet` | données brutes (format machine, typé, compressé) |
| `data/raw/retail_customer_base.csv` | mêmes données, lisibles par un humain |
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

- Valeurs manquantes volontaires sur `nps_score` (~8 %) et `avg_basket_eur` (clients sans commande).
- Outliers légitimes (~1 %) : clients d'exception (gros volumes, paniers très élevés).
- Chevauchement volontaire entre profils : la structure latente est réelle mais floue.
- `latent_segment` et `churned_next_90d` sont des **métadonnées** : exclues des features, utilisées uniquement pour le diagnostic et la validité externe.
- Dérive temporelle volontaire : `mobile_share` augmente avec la date d'inscription.

---

## 7. Observations attendues (EDA)

Points à retrouver dans `notebooks/01_exploratory_analysis.ipynb` :

1. Il n'y a **pas de cible** : la qualité se juge sur des critères internes (silhouette, Davies-Bouldin, Calinski-Harabasz), externes (accord avec le segment latent, écart de churn) et **métier** (taille minimale, stabilité, actionnabilité).
2. La silhouette est très sensible à l'échelle des variables : `revenue_12m_eur` (euros) et `discount_share` (0-1) ne peuvent pas cohabiter sans standardisation. C'est la première raison d'échec d'un clustering en production.
3. `revenue_12m_eur`, `web_sessions_12m` et `orders_12m` sont fortement asymétriques (queue droite) : un passage en log et un winsorising stabilisent les centroïdes, sinon quelques clients d'exception tirent un groupe entier.
4. `avg_basket_eur` est quasi redondant avec `revenue_12m_eur / orders_12m` : la redondance gonfle artificiellement la séparation et biaise la silhouette au profit des variables les plus lourdes.
5. Les groupes se **chevauchent** volontairement : un acheteur occasionnel qui profite d'une promotion ressemble à un chasseur de promo. Une silhouette de 0.25-0.40 est donc un résultat honnête, pas un échec.
6. `recency_days` est bimodal (actifs vs dormeurs) : c'est la variable la plus discriminante, mais elle ne suffit pas — deux clients à 300 jours d'inactivité peuvent être un dormeur et un VIP en pause.
7. `discount_share` et `newsletter_opens_12m` séparent le profil promotionnel du profil fidèle plein tarif : c'est le levier d'arbitrage du budget remises.
8. `return_rate` et `support_tickets_12m` signalent l'insatisfaction avant le churn : un groupe à fort taux de retour est un gisement de rétention, pas un groupe à solliciter davantage.
9. `nps_score` comporte ~8 % de manquants (biais de non-réponse : les détracteurs répondent davantage) ; l'imputer par la médiane masque ce biais, un indicateur de manquant est préférable.
10. `loyalty_tier` est **calculé par le métier** à partir du chiffre d'affaires : l'utiliser comme feature introduit une circularité (on redécouvre le palier). À documenter, voire à exclure.
11. `region` et `acquisition_channel` sont des proxies socio-économiques : une segmentation qui sépare principalement par région poserait un problème d'équité et de conformité.
12. Le nombre de groupes est un **choix** : k trop petit fusionne des comportements distincts, k trop grand produit des micro-groupes inexploitables par le CRM (coût de campagne, lisibilité).

---

## 8. Passage à l'échelle (données réelles)

Pour brancher ce projet sur des données de production :

1. Remplacer `SyntheticDataGenerator` par un `RawDataLoader` qui lit la source réelle
   (entrepôt, objet storage, API) — l'interface ne change pas.
2. **Conserver les schémas Pandera** : ils deviennent le contrat avec le fournisseur de données
   (et un test de non-régression à chaque évolution du schéma amont).
3. Versionner les datasets (DVC / Delta / snapshots horodatés) pour garantir la reproductibilité.
4. Ajouter une surveillance de dérive (voir `mlops/model-monitoring/*`).
