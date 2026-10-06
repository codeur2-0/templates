# Données — Messages de service client annotés en entités, rédigés et abrégés

> **Jeu de données synthétique** généré localement par `src/data/generators.py`.
> Aucune donnée réelle, aucun téléchargement, aucune information confidentielle.
> Objectif : disposer d'un dataset **crédible métier**, petit et reproductible, pour
> démontrer tout le cycle de vie (EDA → validation → preprocessing → entraînement → évaluation).

---

## 1. Description métier

Un corpus synthétique de 1 200 messages de service client annotés au niveau du caractère : cinq
types d'entités (produit, référence de commande, montant, date, transporteur), 4 454 mentions, soit
3,7 mentions par message, aucun message sans entité par construction. Chaque message est écrit par
un gabarit à slots : les surfaces annotées sont enregistrées au moment de l'écriture, elles ne sont
pas retrouvées après coup. Deux styles rédactionnels se partagent le corpus — 55 % de messages
rédigés (« CMD-1234 », « 89,90 € », « 12 mars 2025 ») et 45 % de messages abrégés (« cmd 1234 », «
89.90 EUR », « 12/03/2025 »). Les surfaces de produits et de transporteurs des splits d'évaluation
sont **réservées** : 7,8 % des mentions portent un nom de produit ou de transporteur jamais vu à
l'entraînement, et cette part est publiée dans les métadonnées. Les distracteurs sont **déclarés** :
familles de produits, villes, numéros de facture, années seules apparaissent dans les textes et ne
sont jamais annotés. Le découpage (train / val / calibration / test) est stratifié par style et
écrit dans le corpus.

**Contexte** : Le service client reçoit 1 200 messages par semaine par formulaire, courriel, chat et
courrier. Chaque message raconte la même histoire en mots différents : une référence de
commande, un produit, un transporteur, un montant et une date. Aujourd'hui, un conseiller
relit le message pour recopier ces cinq informations dans le dossier — un travail mécanique,
fastidieux, et qui conditionne tout le reste : sans la référence de commande correcte,
aucune relance n'est possible, et un montant mal recopié se retrouve dans la comptabilité.

**Problème adressé** : Extraire automatiquement les entités d'un message (produit, référence de commande, montant,
date, transporteur), avec leurs positions exactes dans le texte, en publiant pour chaque
mention sa **provenance** (règle déclenchée ou modèle) et sa **confiance** — et en mesurant
la part de mentions qui portent sur des noms jamais vus à l'entraînement, parce que c'est là
que le système se trompe.

**Consommateur principal** : Responsable du back-office d'un distributeur de matériel informatique (300 personnes), avec un ingénieur IA qui outille le service client et une équipe qualité qui mesure les dossiers traités en retard.

---

## 2. Fiche d'identité

| Propriété | Valeur |
| --- | --- |
| Nom du dataset | `sav_messages` |
| Granularité | une ligne = Identifiant stable du message (clé de jointure avec la table d'annotations). |
| Nombre d'échantillons (par défaut) | 1 200 |
| Nombre de colonnes | 8 |
| Clé | `msg_id` (unique) || Dimension temporelle | `received_at` || Formats | parquet, csv |
| Emplacement | `data/raw/sav_messages.parquet` |
| Graine de génération | `42` |
| Valeurs manquantes | oui, en proportion maîtrisée (voir schéma) |
| Outliers | oui, légitimes métier (bornés par le schéma Pandera) |
| Source | **générée synthétiquement** (`SyntheticDataGenerator`) |

---

## 3. Schéma des colonnes

| # | Colonne | Type | Rôle | Unité | Signification métier | Distribution attendue |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `msg_id` | `str` | identifier | - | Identifiant stable du message (clé de jointure avec la table d'annotations). | - |
| 2 | `text` | `str` | feature | - | Texte du message tel qu'il a été écrit, avec ses surfaces rédigées ou abrégées. | - |
| 3 | `canal` | `category` | feature | - | Canal d'arrivée du message : formulaire, courriel, chat ou courrier. | ≈ 44 / 13 / 32 / 11 % |
| 4 | `style` | `category` | metadata | - | Style rédactionnel : redige (surfaces complètes) ou abrege (surfaces courtes, ponctuation réduite). | ≈ 55 / 45 % |
| 5 | `n_entities` | `int` | metadata | - | Nombre d'entités annotées dans le message (calculé à la génération, jamais prédit). | - |
| 6 | `n_tokens` | `int` | metadata | - | Nombre de tokens du texte, calculé avec le tokenizer du projet (pour les figures de longueur). | - |
| 7 | `received_at` | `datetime` | timestamp | - | Date de réception du message (six mois d'historique). | - |
| 8 | `split` | `category` | group | - | Découpage du corpus (train / val / calibration / test), stratifié par style au moment de la génération. | 60 / 20 / 5 / 15 % : le test n'est mesuré qu'une fois |

### Contraintes de validation (contrat exécutable)

Les contraintes ci-dessous ne sont pas de la documentation : elles sont **exécutées** à chaque
chargement par les schémas Pandera de `src/data/schemas.py` (`RawDataSchema`,
`ProcessedDataSchema`, `InferenceDataSchema`).

| Colonne | Contraintes |
| --- | --- |
| `msg_id` | type `str`, unique, non nul, motif ^MSG-[0-9]{4}$ |
| `text` | type `str`, non nul, longueur min 40 · longueur max 400 |
| `canal` | type `category`, non nul, dans formulaire, email, chat, courrier |
| `style` | type `category`, non nul, dans redige, abrege |
| `n_entities` | type `int`, non nul, >= 3 · <= 5 |
| `n_tokens` | type `int`, non nul, >= 8 · <= 120 |
| `received_at` | type `datetime`, non nul |
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
| `data/raw/sav_messages.parquet` | données brutes (format machine, typé, compressé) |
| `data/raw/sav_messages.csv` | mêmes données, lisibles par un humain |
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

- Le corpus est déterministe : même graine, mêmes textes, mêmes annotations, mêmes découpages, mêmes métriques.
- Les messages sont générés par gabarits paramétrés : aucune donnée personnelle, aucun texte recopié.
- Les annotations sont produites à l'écriture du texte, jamais reconstruites après coup : `surface == text[start:end]` est un contrat testé.

---

## 7. Observations attendues (EDA)

Points à retrouver dans `notebooks/01_exploratory_analysis.ipynb` :

1. Trois entités sur quatre sont reconnaissables par leur forme (référence de commande, montant, date) : elles ne dépendent pas du vocabulaire, et un système à base de motifs les trouve quasi parfaitement.
2. Les produits et les transporteurs, eux, sont des **noms** : leur reconnaissance dépend d'un vocabulaire appris, donc un nom réservé aux splits d'évaluation est hors de portée d'une liste — c'est l'écart que le projet mesure au lieu de le supposer.
3. Les villes ressemblent à des entités et n'en sont pas : sans distracteur, un modèle apprendrait à surligner tous les noms propres et la précision serait trompeuse.
4. Les deux styles d'écriture portent exactement les mêmes entités : un système qui ne sait lire que le style rédigé perd la moitié du corpus, et le rapport le chiffre.
5. Les bornes sont mesurées séparément du type : une mention bien typée mais décalée est fausse, et le rapport publie la F1 partielle pour dire ce que cette exigence coûte.
6. Le découpage est écrit dans le corpus, pas tiré à l'entraînement : deux exécutions mesurent les mêmes messages de test.

---

## 8. Passage à l'échelle (données réelles)

Pour brancher ce projet sur des données de production :

1. Remplacer `SyntheticDataGenerator` par un `RawDataLoader` qui lit la source réelle
   (entrepôt, objet storage, API) — l'interface ne change pas.
2. **Conserver les schémas Pandera** : ils deviennent le contrat avec le fournisseur de données
   (et un test de non-régression à chaque évolution du schéma amont).
3. Versionner les datasets (DVC / Delta / snapshots horodatés) pour garantir la reproductibilité.
4. Ajouter une surveillance de dérive (voir `mlops/model-monitoring/*`).
