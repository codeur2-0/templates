# Données — Fiches produit et questions du support client (garantie, retour, SAV)

> **Jeu de données synthétique** généré localement par `src/data/generators.py`.
> Aucune donnée réelle, aucun téléchargement, aucune information confidentielle.
> Objectif : disposer d'un dataset **crédible métier**, petit et reproductible, pour
> démontrer tout le cycle de vie (EDA → validation → preprocessing → entraînement → évaluation).

---

## 1. Description métier

Un corpus synthétique de 120 fiches produit — vingt produits x six articles (garantie, retour, frais
de retour, prise en charge SAV, compatibilité, mise à jour) — et les questions du support qui s'y
rapportent. Chaque fiche contient une **phrase canonique** qui porte la valeur (une durée, un
montant, une compatibilité) et les questions sont écrites à partir d'elle : la réponse de référence
est donc la phrase du corpus, mot pour mot, et l'exact match mesure la sélection de la bonne phrase
plutôt que l'élégance d'une reformulation. Les questions couvrent quatre difficultés — factuelle,
paraphrase, multi-document, hors corpus — et la difficulté d'une paraphrase tient à la question,
jamais à la réponse.

**Contexte** : Le service client répond chaque jour à des questions factuelles — durée de garantie, délai
de retour, coût des frais, prise en charge SAV — dont la réponse existe déjà, écrite noir
sur blanc, dans les fiches produit internes. Les conseillers répondent de mémoire : une
durée de garantie annoncée de 24 mois au lieu de 36 coûte un litige, et chaque réponse
approximative nourrit un contentieux.

**Problème adressé** : Répondre à une question client par la **phrase exacte** de la fiche produit qui fait foi, en
citant la fiche, et refuser explicitement de répondre quand aucune fiche ne contient la
réponse — plutôt que de produire une phrase plausible.

**Consommateur principal** : Support client de niveau 1 d'un e-commerçant de 300 personnes, avec un responsable qualité qui répond aux litiges et un ingénieur IA chargé d'outiller l'équipe.

---

## 2. Fiche d'identité

| Propriété | Valeur |
| --- | --- |
| Nom du dataset | `product_support_faq` |
| Granularité | une ligne = Identifiant stable de la fiche (clé de jointure, jamais une feature). |
| Nombre d'échantillons (par défaut) | 120 |
| Nombre de colonnes | 7 |
| Clé | `doc_id` (unique) || Dimension temporelle | `published_at` || Formats | parquet, csv |
| Emplacement | `data/raw/product_support_faq.parquet` |
| Graine de génération | `42` |
| Valeurs manquantes | oui, en proportion maîtrisée (voir schéma) |
| Outliers | oui, légitimes métier (bornés par le schéma Pandera) |
| Source | **générée synthétiquement** (`SyntheticDataGenerator`) |

---

## 3. Schéma des colonnes

| # | Colonne | Type | Rôle | Unité | Signification métier | Distribution attendue |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `doc_id` | `str` | identifier | - | Identifiant stable de la fiche (clé de jointure, jamais une feature). | - |
| 2 | `title` | `str` | metadata | - | Titre de la fiche : le type d'article et le produit concerné. | un titre par fiche, construit sur le type d'article et le code produit |
| 3 | `section` | `category` | feature | - | Section éditoriale : definition, politique, calcul, procedure, contact. | - |
| 4 | `source` | `category` | feature | - | Espace wiki d'origine : wiki_juridique, wiki_support, wiki_finance, wiki_it. | - |
| 5 | `published_at` | `datetime` | timestamp | - | Date de publication de la fiche (sert à calculer son ancienneté). | - |
| 6 | `n_tokens` | `int` | metadata | - | Nombre de tokens du texte, calculé avec le tokenizer du projet. | - |
| 7 | `text` | `str` | feature | - | Texte intégral de la fiche : c'est lui qui contient la phrase canonique. | - |

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
| `data/raw/product_support_faq.parquet` | données brutes (format machine, typé, compressé) |
| `data/raw/product_support_faq.csv` | mêmes données, lisibles par un humain |
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

- Le corpus est synthétique, mais sa structure imite un vrai référentiel produit : articles répétés, valeurs par produit, phrases canoniques citables.
- Le plafond annoté (part des questions répondables) est publié dans les métadonnées de génération : il évite de poursuivre un rappel de 1,0 inatteignable.

---

## 7. Observations attendues (EDA)

Points à retrouver dans `notebooks/01_exploratory_analysis.ipynb` :

1. Chaque fiche contient une phrase canonique et trois à quatre phrases de contexte : le modèle extractif doit choisir la bonne, et l'exact match le dit sans ambiguïté.
2. Six types d'articles se répètent sur vingt produits : le nom du produit est le seul discriminant lexical entre des fiches qui se ressemblent presque mot pour mot.
3. Les paraphrases conservent le nom du produit : sans lui la question serait ambiguë (plusieurs fiches y répondent) et la mesurer ne dirait rien.
4. Les questions multi-document demandent deux valeurs à la fois (garantie et délai d'intervention) : la réponse de référence concatène deux phrases, et tout système qui s'arrête au premier passage est plafonné.
5. Les questions hors corpus portent sur des faits voisins mais absents (extension de garantie, SAV le dimanche) : seule l'abstention y est correcte.

---

## 8. Passage à l'échelle (données réelles)

Pour brancher ce projet sur des données de production :

1. Remplacer `SyntheticDataGenerator` par un `RawDataLoader` qui lit la source réelle
   (entrepôt, objet storage, API) — l'interface ne change pas.
2. **Conserver les schémas Pandera** : ils deviennent le contrat avec le fournisseur de données
   (et un test de non-régression à chaque évolution du schéma amont).
3. Versionner les datasets (DVC / Delta / snapshots horodatés) pour garantir la reproductibilité.
4. Ajouter une surveillance de dérive (voir `mlops/model-monitoring/*`).
