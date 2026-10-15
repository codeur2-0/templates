# Données — Catalogue produit publié trois fois (notice, fiche commerciale, note SAV)

> **Jeu de données synthétique** généré localement par `src/data/generators.py`.
> Aucune donnée réelle, aucun téléchargement, aucune information confidentielle.
> Objectif : disposer d'un dataset **crédible métier**, petit et reproductible, pour
> démontrer tout le cycle de vie (EDA → validation → preprocessing → entraînement → évaluation).

---

## 1. Description métier

Un corpus synthétique de 168 fiches — vingt-huit références x deux faits (autonomie, garantie
légale) x trois styles éditoriaux — et 115 questions annotées dont 19 hors corpus. Trois équipes
publient le même référentiel : les deux phrases du fait (celle qui porte la valeur et celle qui la
contextualise) sont **recopiées** dans les trois fiches, tandis que la phrase de style, la section,
l'espace d'origine et le titre changent. Les trois fiches sont annotées pertinentes pour la
question, ce qui plafonne recall@1 au tiers et fait de cette métrique une lecture de précision ; 56
questions factuelles, 28 paraphrases, 12 multi-document (six fiches pertinentes : deux faits x trois
écritures) et 19 hors corpus couvrent quatre difficultés, et la paraphrase change le vocabulaire de
la question — 34 % des mots pleins partagés avec la phrase source contre 71 % pour la question
factuelle — sans jamais retirer la référence qui identifie la réponse.

**Contexte** : Le catalogue est publié trois fois : la notice constructeur, la fiche commerciale et la note
SAV reprennent la même autonomie et la même durée de garantie, avec un cadrage et un
vocabulaire d'accompagnement différents. Le service client cherche dans les trois, l'équipe
catalogue veut retrouver les fiches qui écrivent le même fait, et la plateforme veut un
index réutilisable — dimension, taille, débit et fidélité connus — plutôt qu'un script de
recherche par cas d'usage.

**Problème adressé** : Produire des **vecteurs de documents** déterministes, hors ligne et mesurables, les servir
comme index de recherche (recall@k), comme détecteur de quasi-doublons (les trois fiches
d'un même fait) et comme artefact versionné dont on connaît la dimension, la fidélité, la
taille et le débit — au lieu d'un index dont on ne sait rien sinon qu'il « marche ».

**Consommateur principal** : Équipe plateforme données d'une enseigne de matériel électronique (400 personnes), avec un responsable catalogue qui publie le référentiel produit et un ingénieur IA qui outille le service client.

---

## 2. Fiche d'identité

| Propriété | Valeur |
| --- | --- |
| Nom du dataset | `equipment_catalogue_editions` |
| Granularité | une ligne = Identifiant stable de la fiche (clé de jointure, jamais une feature). |
| Nombre d'échantillons (par défaut) | 168 |
| Nombre de colonnes | 7 |
| Clé | `doc_id` (unique) || Dimension temporelle | `published_at` || Formats | parquet, csv |
| Emplacement | `data/raw/equipment_catalogue_editions.parquet` |
| Graine de génération | `42` |
| Valeurs manquantes | oui, en proportion maîtrisée (voir schéma) |
| Outliers | oui, légitimes métier (bornés par le schéma Pandera) |
| Source | **générée synthétiquement** (`SyntheticDataGenerator`) |

---

## 3. Schéma des colonnes

| # | Colonne | Type | Rôle | Unité | Signification métier | Distribution attendue |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `doc_id` | `str` | identifier | - | Identifiant stable de la fiche (clé de jointure, jamais une feature). | - |
| 2 | `title` | `str` | metadata | - | Titre de la fiche : style éditorial, fait et référence concernée. | un titre par fiche, construit sur le style, le fait et le code référence |
| 3 | `section` | `category` | feature | - | Section éditoriale : procedure (notice), definition (fiche commerciale), contact (note SAV). | - |
| 4 | `source` | `category` | feature | - | Espace documentaire d'origine : wiki_it, wiki_ops, wiki_support. | - |
| 5 | `published_at` | `datetime` | timestamp | - | Date de publication de la fiche (les trois exemplaires d'un fait ne sont pas datés pareil). | - |
| 6 | `n_tokens` | `int` | metadata | - | Nombre de tokens du texte, calculé avec le tokenizer du projet. | - |
| 7 | `text` | `str` | feature | - | Texte intégral de la fiche : la phrase du fait y est recopiée mot pour mot. | - |

### Contraintes de validation (contrat exécutable)

Les contraintes ci-dessous ne sont pas de la documentation : elles sont **exécutées** à chaque
chargement par les schémas Pandera de `src/data/schemas.py` (`RawDataSchema`,
`ProcessedDataSchema`, `InferenceDataSchema`).

| Colonne | Contraintes |
| --- | --- |
| `doc_id` | type `str`, unique, non nul, motif ^DOC-[0-9]{4}$ |
| `title` | type `str`, non nul, longueur min 6 · longueur max 160 |
| `section` | type `category`, non nul, dans politique, procedure, definition, contact, calcul, conformite |
| `source` | type `category`, non nul, dans wiki_rh, wiki_it, wiki_finance, wiki_juridique, wiki_ops, wiki_support |
| `published_at` | type `datetime`, non nul |
| `n_tokens` | type `int`, non nul, >= 40 · <= 1200 |
| `text` | type `str`, non nul, longueur min 200 |

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
| `data/raw/equipment_catalogue_editions.parquet` | données brutes (format machine, typé, compressé) |
| `data/raw/equipment_catalogue_editions.csv` | mêmes données, lisibles par un humain |
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

- Le corpus est synthétique, mais sa structure imite un référentiel publié par plusieurs équipes : même fait, trois plumes, aucune autorité.
- Les paraphrases ne partagent jamais zéro mot avec la fiche (17 % au minimum, 34 % en médiane) : c'est mesuré, et c'est la raison pour laquelle le segment reste difficile sans devenir insoluble.
- Le multi-document est plafonné par construction : la réponse de référence concatène deux phrases venues de deux fiches, et le garde-fou `support_ratio` écarte la seconde quand elle est trop mal classée (exact match 0,25, F1 0,82 en dense).
- Le recall@1 ne peut pas dépasser 1/3 sur les questions mono-document et 1/6 sur les multi-document : la famille le publie en métrique secondaire et lit la précision au premier rang à sa place.

---

## 7. Observations attendues (EDA)

Points à retrouver dans `notebooks/01_exploratory_analysis.ipynb` :

1. Chaque fait est écrit trois fois et les deux phrases du fait sont identiques dans les trois fiches : le même contenu est annoté pertinent trois fois, donc un rappel faible ne peut pas venir d'un manque de candidats.
2. Les trois écritures d'un fait se ressemblent plus entre elles qu'aux fiches d'un autre produit : la suite de tests vérifie, sur toutes les fiches du corpus de test, que les deux plus proches voisins d'une fiche sont ses deux autres écritures — c'est ce qui rend le dédoublonnage mesurable.
3. Le découpage (110 tokens, 30 de recouvrement) laisse une fiche entière — une cinquantaine de tokens (53 en moyenne) — dans un seul passage : la recherche et le dédoublonnage portent sur les fiches, pas sur des fragments.
4. Les paraphrases partagent deux fois moins de mots pleins avec la phrase source que les questions factuelles (34 % contre 71 %) sans retirer la référence qui identifie la réponse : c'est le segment qui sépare un index de mots d'un index sémantique, et où l'exact match tombe à 0,60.
5. Les questions hors corpus portent sur un fait voisin que le catalogue n'écrit nulle part (réparation hors garantie, livraison le samedi, paiement en plusieurs fois) mais elles sont écrites avec le vocabulaire des fiches : l'abstention y est mesurée à 0,57 en dense, et la famille publie ce chiffre au lieu de le cacher.

---

## 8. Passage à l'échelle (données réelles)

Pour brancher ce projet sur des données de production :

1. Remplacer `SyntheticDataGenerator` par un `RawDataLoader` qui lit la source réelle
   (entrepôt, objet storage, API) — l'interface ne change pas.
2. **Conserver les schémas Pandera** : ils deviennent le contrat avec le fournisseur de données
   (et un test de non-régression à chaque évolution du schéma amont).
3. Versionner les datasets (DVC / Delta / snapshots horodatés) pour garantir la reproductibilité.
4. Ajouter une surveillance de dérive (voir `mlops/model-monitoring/*`).
