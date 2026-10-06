# Données — Tickets de support multiclasse, écrits trois fois (canonique, paraphrase, bruité)

> **Jeu de données synthétique** généré localement par `src/data/generators.py`.
> Aucune donnée réelle, aucun téléchargement, aucune information confidentielle.
> Objectif : disposer d'un dataset **crédible métier**, petit et reproductible, pour
> démontrer tout le cycle de vie (EDA → validation → preprocessing → entraînement → évaluation).

---

## 1. Description métier

Un corpus synthétique de 1 200 tickets répartis sur six catégories de support — facturation,
livraison, produit défectueux, remboursement, compte client et « autre » — et trois styles
rédactionnels : cinquante pour cent des tickets réutilisent le vocabulaire de leur catégorie, trente
pour cent reformulent la même demande sans ce vocabulaire, vingt pour cent ajoutent une formule de
politesse et une signature partagées par toutes les classes. Chaque catégorie a son vocabulaire
propre, sauf « autre » : cette classe n'a aucun mot commun par construction, elle est minoritaire
(12 %) et c'est pour elle que la métrique principale est une F1 macro. La priorité déclarée par le
client est tirée indépendamment du libellé, et le canal d'arrivée n'est pas non plus un signal : ce
sont deux distracteurs déclarés, mesurés par un test de raccourci. Enfin, chaque pool de
formulations est coupé en deux : quelques tournures par couple (classe, style) sont **réservées aux
splits d'évaluation** et n'apparaissent jamais à l'entraînement, sans quoi le score publié
mesurerait une recopie de gabarits plutôt qu'une généralisation.

**Contexte** : Le support reçoit 1 200 tickets par semaine par quatre canaux (formulaire, courriel, chat,
courrier). Aujourd'hui, chaque ticket est lu par un conseiller qui le réaffecte à la main au
bon service : facturation, livraison, produit défectueux, remboursement, compte client — ou
« autre » quand la demande ne rentre dans aucune case. Le réaiguillage coûte une demi-
journée par semaine et, surtout, il se voit : un ticket de remboursement routé vers la
livraison repart pour un tour et le client attend deux jours de plus.

**Problème adressé** : Attribuer automatiquement à chaque ticket sa catégorie de service, **avec une décision
explicable** (les termes qui ont pesé), sans jamais sacrifier la classe minoritaire « autre
» au profit de l'exactitude globale — et sans dépendre d'un vocabulaire que le client
n'emploie pas forcément.

**Consommateur principal** : Responsable du support de niveau 1 d'un distributeur de matériel électronique (300 personnes), avec un ingénieur IA qui outille le routage des tickets et une équipe qualité qui mesure les réclamations mal orientées.

---

## 2. Fiche d'identité

| Propriété | Valeur |
| --- | --- |
| Nom du dataset | `support_tickets` |
| Granularité | une ligne = Identifiant stable du ticket (clé de jointure, jamais une feature). |
| Nombre d'échantillons (par défaut) | 1 200 |
| Nombre de colonnes | 9 |
| Cible | `label` || Clé | `doc_id` (unique) || Dimension temporelle | `published_at` || Formats | parquet, csv |
| Emplacement | `data/raw/support_tickets.parquet` |
| Graine de génération | `42` |
| Valeurs manquantes | oui, en proportion maîtrisée (voir schéma) |
| Outliers | oui, légitimes métier (bornés par le schéma Pandera) |
| Source | **générée synthétiquement** (`SyntheticDataGenerator`) |

---

## 3. Schéma des colonnes

| # | Colonne | Type | Rôle | Unité | Signification métier | Distribution attendue |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `doc_id` | `str` | identifier | - | Identifiant stable du ticket (clé de jointure, jamais une feature). | - |
| 2 | `text` | `str` | feature | - | Texte du ticket tel qu'il a été reçu, gabarit paramétré par produit et référence de commande. | - |
| 3 | `label` | `category` | target | - | Catégorie du ticket : la cible du modèle, celle que le conseiller aurait choisie. | ≈ 20 / 20 / 18 / 15 / 15 / 12 % : « autre » est la classe minoritaire |
| 4 | `source` | `category` | feature | - | Canal d'arrivée du ticket : formulaire, courriel, chat ou courrier. | ≈ 45 / 30 / 15 / 10 % |
| 5 | `style` | `category` | metadata | - | Style rédactionnel : canonique (vocabulaire de la catégorie), paraphrase (la même demande sans ce vocabulaire), bruite (formule de politesse et signature partagées). | ≈ 50 / 30 / 20 % |
| 6 | `priority` | `category` | metadata | - | Priorité déclarée par le client : un distracteur, tiré indépendamment du libellé. | ≈ 35 / 45 / 20 % |
| 7 | `published_at` | `datetime` | timestamp | - | Date de réception du ticket (les dates remontent sur six mois). | - |
| 8 | `n_tokens` | `int` | metadata | - | Nombre de tokens du texte, calculé avec le tokenizer du projet. | - |
| 9 | `split` | `category` | group | - | Découpage du corpus (train / val / calibration / test), stratifié par classe et par style au moment de la génération. | 60 / 20 / 5 / 15 % : le test n'est mesuré qu'une fois |

### Contraintes de validation (contrat exécutable)

Les contraintes ci-dessous ne sont pas de la documentation : elles sont **exécutées** à chaque
chargement par les schémas Pandera de `src/data/schemas.py` (`RawDataSchema`,
`ProcessedDataSchema`, `InferenceDataSchema`).

| Colonne | Contraintes |
| --- | --- |
| `doc_id` | type `str`, unique, non nul, motif ^TKT-[0-9]{4}$ |
| `text` | type `str`, non nul, longueur min 40 · longueur max 1200 |
| `label` | type `category`, non nul, dans facturation, livraison, produit_defectueux, remboursement, compte_client, autre |
| `source` | type `category`, non nul, dans formulaire, email, chat, courrier |
| `style` | type `category`, non nul, dans canonique, paraphrase, bruite |
| `priority` | type `category`, non nul, dans basse, normale, haute |
| `published_at` | type `datetime`, non nul |
| `n_tokens` | type `int`, non nul, >= 8 · <= 400 |
| `split` | type `category`, non nul, dans train, val, calibration, test |

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
| `data/raw/support_tickets.parquet` | données brutes (format machine, typé, compressé) |
| `data/raw/support_tickets.csv` | mêmes données, lisibles par un humain |
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

- Le corpus est déterministe : même graine, mêmes textes, mêmes découpages, mêmes métriques.
- Les tickets sont générés par gabarits paramétrés (produit, référence de commande) : aucune donnée personnelle, aucun texte recopié.

---

## 7. Observations attendues (EDA)

Points à retrouver dans `notebooks/01_exploratory_analysis.ipynb` :

1. Chaque catégorie possède un vocabulaire propre et une seule n'en a pas : la classe « autre » est donc structurellement la plus difficile, et c'est elle qu'une métrique macro protège d'un modèle qui l'ignorerait.
2. Un tiers des tickets est écrit en paraphrase : sans ce tiers, un modèle de mots-clés suffirait et le projet ne mesurerait rien.
3. La priorité déclarée est tirée indépendamment du libellé : un modèle qui l'utiliserait gagnerait de la variance, pas de l'information — le test de raccourci le chiffre.
4. Le bruit (formule de politesse, signature) est identique dans toutes les classes : il ne peut pas servir de signal, mais il dégrade les modèles qui le suivent.
5. Le découpage est écrit dans le corpus, pas tiré à l'entraînement : deux exécutions mesurent les mêmes lignes de test.
6. Les formulations d'évaluation sont réservées : par couple (classe, style), les tournures des splits val, calibration et test sont disjointes de celles du train — un score élevé ne peut donc pas venir d'un gabarit mémorisé.

---

## 8. Passage à l'échelle (données réelles)

Pour brancher ce projet sur des données de production :

1. Remplacer `SyntheticDataGenerator` par un `RawDataLoader` qui lit la source réelle
   (entrepôt, objet storage, API) — l'interface ne change pas.
2. **Conserver les schémas Pandera** : ils deviennent le contrat avec le fournisseur de données
   (et un test de non-régression à chaque évolution du schéma amont).
3. Versionner les datasets (DVC / Delta / snapshots horodatés) pour garantir la reproductibilité.
4. Ajouter une surveillance de dérive (voir `mlops/model-monitoring/*`).
