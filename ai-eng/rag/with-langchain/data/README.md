# Données — Base de connaissances interne (procédures d'entreprise)

> **Jeu de données synthétique** généré localement par `src/data/generators.py`.
> Aucune donnée réelle, aucun téléchargement, aucune information confidentielle.
> Objectif : disposer d'un dataset **crédible métier**, petit et reproductible, pour
> démontrer tout le cycle de vie (EDA → validation → preprocessing → entraînement → évaluation).

---

## 1. Description métier

Un corpus documentaire synthétique de 128 documents (huit thèmes d'entreprise x huit périmètres de
population x deux éditions), construit à partir de faits plantés : chaque fait génère un document
qui le contient et une question qui l'interroge. Les questions couvrent quatre difficultés — facile,
paraphrase, multi-document, hors corpus — de sorte que le rappel mesure de vraies capacités
distinctes : retrouver une formulation identique, retrouver un sens sans les mêmes mots, agréger
deux sources, et savoir ne pas répondre.

**Contexte** : Les procédures internes sont empilées dans six espaces wiki (RH, IT, finance, juridique,
opérations, support). Personne ne retrouve une règle : le support répond quarante fois par
semaine aux mêmes questions, et chaque réponse donnée de mémoire est une source de risque
(une règle inventée coûte plus cher qu'une règle non trouvée).

**Problème adressé** : Répondre aux questions internes en s'appuyant uniquement sur les documents de référence,
citer le passage exact qui fonde la réponse, et s'abstenir explicitement quand le corpus ne
contient pas la réponse.

**Consommateur principal** : Équipe Knowledge / Support d'une scale-up de 400 personnes, avec un ingénieur IA qui industrialise l'assistant et une équipe support qui répond aujourd'hui à la main.

---

## 2. Fiche d'identité

| Propriété | Valeur |
| --- | --- |
| Nom du dataset | `internal_knowledge_base` |
| Granularité | une ligne = Identifiant stable du document (sert de clé de jointure, jamais de feature). |
| Nombre d'échantillons (par défaut) | 128 |
| Nombre de colonnes | 7 |
| Clé | `doc_id` (unique) || Dimension temporelle | `published_at` || Formats | parquet, csv |
| Emplacement | `data/raw/internal_knowledge_base.parquet` |
| Graine de génération | `42` |
| Valeurs manquantes | oui, en proportion maîtrisée (voir schéma) |
| Outliers | oui, légitimes métier (bornés par le schéma Pandera) |
| Source | **générée synthétiquement** (`SyntheticDataGenerator`) |

---

## 3. Schéma des colonnes

| # | Colonne | Type | Rôle | Unité | Signification métier | Distribution attendue |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `doc_id` | `str` | identifier | - | Identifiant stable du document (sert de clé de jointure, jamais de feature). | - |
| 2 | `title` | `str` | metadata | - | Titre éditorial du document, affiché avec les citations. | un titre par document, versionné par un suffixe d'édition |
| 3 | `section` | `category` | feature | - | Section éditoriale : politique, procedure, definition, contact, calcul, conformite. | - |
| 4 | `source` | `category` | feature | - | Espace wiki d'origine : wiki_rh, wiki_it, wiki_finance, wiki_juridique, wiki_ops, wiki_support. | - |
| 5 | `published_at` | `datetime` | timestamp | - | Date de publication du document (sert à calculer l'ancienneté d'une source). | - |
| 6 | `n_tokens` | `int` | metadata | - | Nombre de tokens du texte, calculé avec le tokenizer du projet. | - |
| 7 | `text` | `str` | feature | - | Texte intégral du document : c'est lui qui est découpé en passages puis indexé. | - |

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
| `data/raw/internal_knowledge_base.parquet` | données brutes (format machine, typé, compressé) |
| `data/raw/internal_knowledge_base.csv` | mêmes données, lisibles par un humain |
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

- Le corpus est synthétique, mais sa structure imite un vrai wiki : sections, espaces, dates, éditions successives.
- Le plafond annoté (part des questions répondables) est publié dans les métadonnées de génération : il évite de poursuivre un rappel de 1,0 inaccessible.

---

## 7. Observations attendues (EDA)

Points à retrouver dans `notebooks/01_exploratory_analysis.ipynb` :

1. Les documents ne contiennent pas la réponse « en clair » : plusieurs formulations voisines coexistent (par exemple plusieurs valeurs de solde de congés selon l'édition), ce qui oblige le classement à départager des passages proches.
2. Les questions paraphrases ne partagent presque aucun mot de contenu avec le passage qui répond : un retriever purement lexical y décroche mécaniquement, et c'est mesuré par segment.
3. Les questions multi-document exigent deux sources : tout système qui publie le meilleur passage unique est plafonné sur ce segment.
4. Les questions hors corpus annotent le document le plus proche sans extrait répondant : elles ne récompensent que l'abstention.
5. Le corpus contient volontairement des documents quasi dupliqués (même règle, éditions différentes) : la redondance est un piège classique des wikis d'entreprise.

---

## 8. Passage à l'échelle (données réelles)

Pour brancher ce projet sur des données de production :

1. Remplacer `SyntheticDataGenerator` par un `RawDataLoader` qui lit la source réelle
   (entrepôt, objet storage, API) — l'interface ne change pas.
2. **Conserver les schémas Pandera** : ils deviennent le contrat avec le fournisseur de données
   (et un test de non-régression à chaque évolution du schéma amont).
3. Versionner les datasets (DVC / Delta / snapshots horodatés) pour garantir la reproductibilité.
4. Ajouter une surveillance de dérive (voir `mlops/model-monitoring/*`).
