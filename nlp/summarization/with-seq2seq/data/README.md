# Données — Comptes-rendus d'intervention : documents, résumés de référence et faits annotés

> **Jeu de données synthétique** généré localement par `src/data/generators.py`.
> Aucune donnée réelle, aucun téléchargement, aucune information confidentielle.
> Objectif : disposer d'un dataset **crédible métier**, petit et reproductible, pour
> démontrer tout le cycle de vie (EDA → validation → preprocessing → entraînement → évaluation).

---

## 1. Description métier

Un corpus synthétique de 600 comptes-rendus d'intervention (131 tokens en moyenne, 11 phrases),
leurs **résumés de référence** (53 tokens, 44 % de la longueur du document) et la table des
**faits** annotés : 8,4 faits par document en sept types (équipement, symptôme, cause, action,
pièce, durée, statut), dont 6 sont saillants — c'est-à-dire que le résumé de référence doit les
rapporter. Chaque document est écrit autour d'un **profil d'équipement** (pompe, convoyeur,
automate, variateur…), donc le symptôme, la cause, l'action et la pièce sont cohérents entre eux.
Les résumés réécrivent les faits : les valeurs sont reprises à l'identique (« 45 minutes », «
JNT-204 »), les phrases non, et la moitié des résumés utilise un second registre de rédaction. Les
distracteurs sont **déclarés** : numéros de ticket, de téléphone, de parking et horaires de présence
apparaissent dans les documents et jamais dans les résumés. Le découpage (train / val / calibration
/ test) est écrit dans le corpus.

**Contexte** : Chaque intervention donne lieu à un compte-rendu d'une dizaine de phrases : ce qui a été
constaté, pourquoi, ce qui a été fait, avec quelle pièce, en combien de temps et dans quel
état le matériel est laissé. Les responsables de site les reçoivent par dizaines chaque
semaine et les lisent en diagonale ; la synthèse est faite à la main, quand elle est faite.

**Problème adressé** : Produire automatiquement le résumé d'un compte-rendu — ses faits saillants, en quelques
phrases fidèles — sans jamais inventer une valeur : une durée, une référence de pièce ou un
statut absent du document se paie sur le terrain, alors qu'un résumé un peu moins fluide se
lit très bien.

**Consommateur principal** : Responsable d'un service de maintenance industrielle (60 techniciens), avec un ingénieur données qui outille le retour d'expérience et un responsable qualité qui relit les comptes-rendus avant de les archiver.

---

## 2. Fiche d'identité

| Propriété | Valeur |
| --- | --- |
| Nom du dataset | `intervention_reports` |
| Granularité | une ligne = Identifiant stable du compte-rendu (clé de jointure avec les résumés et les faits). |
| Nombre d'échantillons (par défaut) | 600 |
| Nombre de colonnes | 9 |
| Clé | `doc_id` (unique) || Dimension temporelle | `published_at` || Formats | parquet, csv |
| Emplacement | `data/raw/intervention_reports.parquet` |
| Graine de génération | `42` |
| Valeurs manquantes | oui, en proportion maîtrisée (voir schéma) |
| Outliers | oui, légitimes métier (bornés par le schéma Pandera) |
| Source | **générée synthétiquement** (`SyntheticDataGenerator`) |

---

## 3. Schéma des colonnes

| # | Colonne | Type | Rôle | Unité | Signification métier | Distribution attendue |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `doc_id` | `str` | identifier | - | Identifiant stable du compte-rendu (clé de jointure avec les résumés et les faits). | - |
| 2 | `text` | `str` | feature | - | Compte-rendu complet : contexte, symptôme, cause, actions, pièce, durée et conclusion. | - |
| 3 | `intervention_type` | `category` | feature | - | Nature de l'intervention : dépannage, maintenance, installation ou expertise. | ≈ 25 % par type |
| 4 | `urgency` | `category` | metadata | - | Urgence déclarée à l'ouverture : un distracteur pour le modèle, un segment d'analyse pour le rapport. | ≈ 20 / 50 / 30 % |
| 5 | `site` | `category` | metadata | - | Site d'intervention (anonymisé) : six sites, un segment d'analyse. | - |
| 6 | `n_sentences` | `int` | metadata | - | Nombre de phrases du document (calculé à la génération, jamais prédit). | - |
| 7 | `n_tokens` | `int` | metadata | - | Nombre de tokens du document : c'est le dénominateur de la compression. | - |
| 8 | `published_at` | `datetime` | timestamp | - | Date de rédaction du compte-rendu. | - |
| 9 | `split` | `category` | group | - | Découpage du corpus (train / val / calibration / test), écrit dans le corpus au moment de la génération. | 60 / 20 / 10 / 10 % : le test n'est mesuré qu'une fois |

### Contraintes de validation (contrat exécutable)

Les contraintes ci-dessous ne sont pas de la documentation : elles sont **exécutées** à chaque
chargement par les schémas Pandera de `src/data/schemas.py` (`RawDataSchema`,
`ProcessedDataSchema`, `InferenceDataSchema`).

| Colonne | Contraintes |
| --- | --- |
| `doc_id` | type `str`, unique, non nul, motif ^CR-[0-9]{4}$ |
| `text` | type `str`, non nul, longueur min 300 · longueur max 2400 |
| `intervention_type` | type `category`, non nul, dans depannage, maintenance, installation, expertise |
| `urgency` | type `category`, non nul, dans basse, normale, haute |
| `site` | type `category`, non nul, dans SITE-A, SITE-B, SITE-C, SITE-D, SITE-E, SITE-F |
| `n_sentences` | type `int`, non nul, >= 6 · <= 40 |
| `n_tokens` | type `int`, non nul, >= 60 · <= 600 |
| `published_at` | type `datetime`, non nul |
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
| `data/raw/intervention_reports.parquet` | données brutes (format machine, typé, compressé) |
| `data/raw/intervention_reports.csv` | mêmes données, lisibles par un humain |
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

- Le corpus est déterministe : même graine, mêmes documents, mêmes résumés de référence, mêmes faits, mêmes découpages.
- Les textes sont produits par gabarits paramétrés : aucune donnée personnelle, aucun texte recopié, aucune dépendance réseau.
- Chaque fait est enregistré **pendant** l'écriture du document, avec sa surface et l'index de sa phrase : rien n'est retrouvé après coup par une expression régulière.

---

## 7. Observations attendues (EDA)

Points à retrouver dans `notebooks/01_exploratory_analysis.ipynb` :

1. Tous les faits d'un compte-rendu ne méritent pas d'être résumés : le corpus en porte 8,4 en moyenne quand le résumé de référence n'en rapporte que 6 — les autres sont exacts et volontairement écartés.
2. Les résumés réécrivent les faits au lieu de recopier des phrases : un modèle extractif plafonne donc, et ce plafond est mesuré (baseline `lead`, publiée sur les mêmes lignes).
3. La moitié des résumés change de registre de rédaction : un modèle qui apprend une seule formulation perd la moitié du corpus, et la ventilation le montre.
4. Le décor (tickets, parkings, horaires) n'est jamais résumé : un modèle qui recopie tout le document gagne du ROUGE en précision et en perd en fidélité — les deux chiffres sont publiés ensemble.
5. Chaque document est cohérent : l'équipement, son symptôme, sa cause, l'action et la pièce forment un profil unique. Un corpus incohérent produirait des résumés absurdes et n'apprendrait rien.
6. La compression de référence (44 %) est un budget publié : un résumé plus long n'est pas un meilleur résumé, il est simplement plus long.

---

## 8. Passage à l'échelle (données réelles)

Pour brancher ce projet sur des données de production :

1. Remplacer `SyntheticDataGenerator` par un `RawDataLoader` qui lit la source réelle
   (entrepôt, objet storage, API) — l'interface ne change pas.
2. **Conserver les schémas Pandera** : ils deviennent le contrat avec le fournisseur de données
   (et un test de non-régression à chaque évolution du schéma amont).
3. Versionner les datasets (DVC / Delta / snapshots horodatés) pour garantir la reproductibilité.
4. Ajouter une surveillance de dérive (voir `mlops/model-monitoring/*`).
