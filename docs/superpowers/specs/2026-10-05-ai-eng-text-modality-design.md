# Modalité `text` et familles `ai-eng` — RAG, QA, embeddings, agents

- **Date** : 2026-10-05
- **Statut** : design validé, tranches 1, 2, 2b et 3 livrées et vérifiées (`ai-eng/rag/with-{tfidf,langchain}`,
  `ai-eng/question-answering/with-{tfidf,langchain}`, `ai-eng/embeddings/with-{embedding,tfidf}`,
  `ai-eng/text-classification/with-{tfidf_classifier,transformers}` et
  `ai-eng/named-entity-recognition/with-spacy` conformes dans `tools.verify`), tranches 4 et 5 à venir
- **Branche** : `arena/3402658e-templates`
- **Périmètre** : point 2 de la feuille de route du README racine — `ai-eng/` (LangChain, RAG,
  Transformers, serving FastAPI) et `nlp/` (spaCy, Transformers). Ce document couvre la couche
  `modality/text/` et les quatre familles ai-eng de la première tranche.

## 1. Objectif

Le dépôt déclare **29 familles** et **21 stacks** (18 au moment de la rédaction, plus
`tfidf_classifier`, `embedding` et `spacy`), mais une seule modalité était implémentée au départ
(`modality/tabular/`). Les huit familles texte du registre — `text_classification`,
`named_entity_recognition`, `summarization`, `question_answering`,
`retrieval_augmented_generation`, `agent_tools`, `embedding_pipeline`, `llm_finetuning` — n'ont
ni couche de modalité, ni couche de tâche, ni couche de famille : un manifeste qui les
référencerait produirait un projet cassé sans erreur (mêmes symptômes que la famille
`multiclass_classification` avant sa livraison).

Livrer la modalité `text` et les projets ai-eng, c'est :

1. écrire `templates/modality/text/` (données, pré-traitement, features, pipelines, tests),
2. écrire les couches de tâche `task/retrieval/` et `task/generation/`,
3. écrire les couches de famille (`retrieval_augmented_generation`, `question_answering`,
   `embedding_pipeline`, `agent_tools`, puis `text_classification`, `named_entity_recognition`,
   `summarization`, `llm_finetuning`),
4. écrire les stacks `tfidf`, `langchain`, `transformers`, `spacy`, `fastapi`,
5. écrire le constructeur de notebooks `tools/scaffold/notebooks/text.py`,
6. générer les projets et les rendre verts dans `tools/verify.py`.

### Critères de réussite

1. `python -m tools.scaffold.build --manifest <manifeste ai-eng>` génère le projet sans
   avertissement.
2. `python -m tools.verify ai-eng/<projet> --notebooks-inplace` est vert sur les six étapes
   (ruff check, ruff format, mypy, pytest, six notebooks, `mode=all`).
3. **Aucune dépendance réseau à l'exécution** : corpus synthétique, embeddings déterministes,
   LLM local déterministe. Un adaptateur OpenAI-compatible existe mais reste optionnel et
   désactivé par défaut.
4. Les projets existants ne régressent pas : les couches `modality/text/` et `task/` étant
   additives, la régénération des 25 projets data-science doit produire un diff nul.

## 2. Décisions d'architecture

| Décision | Choix | Raison |
| --- | --- | --- |
| Contrat modèle | `BaseModel` réécrit dans `modality/text/src/models/base.py` : `fit(documents)`, `retrieve(query, k)`, `answer(question, k)`, `save`/`load`, `model_card` | Le contrat tabulaire (`X`, `y`, `predict_proba`) n'a aucun sens pour un retriever. La couche `modality` est la seule à définir le contrat, les stacks ne font que l'implémenter. |
| Représentation | Embeddings **hachés + SVD tronquée** appris sur le corpus (scikit-learn), L2-normalisés | Déterministe, hors ligne, aucun modèle pré-entraîné téléchargé, rejouable sur un laptop. La SVD est apprise sur le **train** uniquement et persistée avec le modèle. |
| Génération | `BaseLLM` avec deux implémentations : `ExtractiveLLM` (sélection de la meilleure phrase des passages, déterministe) et `OpenAICompatibleLLM` (client HTTP optionnel, jamais appelé dans les tests) | Un RAG pédagogique doit tourner sans clé API. L'interface montre où se branche un vrai LLM sans en faire une dépendance dure. |
| Chaîne LangChain | LCEL : retrieval (`BaseRetriever`) -> `RunnableParallel` -> `PromptTemplate` -> adaptateur LLM -> **nœud d'ancrage** projet, avec `Document`, `RunnableAssign` et `RunnableParallel` de `langchain-core` | Montre le vrai geste LangChain (composition de runnables, prompt de configuration, extension points) sans dépendre d'une API. `StrOutputParser` seul rendrait une chaîne nue : les citations — une métrique de la famille — seraient perdues. |
| Persistance LangChain | L'état (documents, chunks, embeddings, configuration) est sérialisé en joblib ; la chaîne est **reconstruite** au chargement | Un `Runnable` LCEL n'est pas garanti picklable ; persister l'état et reconstruire est explicite et testable. |
| Décision de réponse | Seuil de similarité → réponse ou **abstention** | Un RAG qui répond toujours invente : l'abstention mesurée est une métrique de premier plan. |
| Métriques | recall@k, precision@k, MRR, nDCG@k, MAP@k, hit-rate, couverture + fidélité des citations, EM/F1 de réponse, taux d'abstention, latence p50/p95 | Mêmes définitions que `task/ranking` quand elles coïncident, pour que les familles restent comparables. |
| Évaluation | Par question, jamais agrégée ligne à ligne ; comparaison systématique à deux références triviales (aléatoire, premier-k / popularité) | Un recall@5 absolu ne veut rien dire. |
| Domaines | `ai-eng/rag`, `ai-eng/question-answering`, `ai-eng/embeddings`, `ai-eng/agents`, `nlp/text-classification`, `nlp/ner`, `nlp/summarization`, `nlp/llm-finetuning` | Suit la feuille de route du README racine. |

Options écartées :

- **Réutiliser `task/ranking/`** : le classement y est pointwise par couple (utilisateur, item)
  avec un score supervisé ; la recherche documentaire évalue un classement par requête avec des
  cibles binaires par document et une notion de citations. Les métriques se ressemblent, la
  sémantique diffère.
- **Faire de `langchain` la seule stack ai-eng** : sans baseline lexicale (`tfidf`), les chiffres
  du RAG seraient ininterprétables ; la référence BM25 est le concurrent à battre.
- **Appeler un vrai LLM dans la CI** : aucune clé, aucun réseau, aucun coût. L'adaptateur distant
  existe, il est testé par un faux transport HTTP.

## 3. Cas d'usage

### 3.1 `retrieval_augmented_generation` — assistant documentaire interne

Une scale-up de 400 personnes a empilé ses procédures dans un wiki : politique de congés,
procédure d'incident, parcours d'onboarding, règles d'achat, conformité RGPD. Personne ne
retrouve rien ; le support répond 40 fois par semaine aux mêmes questions. Objectif : un
assistant qui **retrouve les passages** et rédige une réponse **sourcée**, ou s'abstient.

Corpus synthétique : ~420 documents, 6 sources, ~3 100 chunks, 240 questions annotées avec
leurs passages de référence et 4 niveaux de difficulté (facile, paraphrase, multi-document,
hors corpus).

### 3.2 `question_answering` — même corpus, réponse extractive

Même corpus, questions à réponse courte : la réponse existe **littéralement** dans un passage.
Evaluation : exact match, F1 token, recall du passage doré.

### 3.3 `embedding_pipeline` — industrialiser les représentations

Corpus de 5 000 textes, batchs, cache, versionnage des embeddings, dérive entre deux versions du
corpus, recall@k de la recherche vectorielle, débit et coût par million de textes.

### 3.4 `agent_tools` — agent avec outils et garde-fous

Un agent qui doit répondre à des questions de support en appelant des outils (recherche
documentaire, calculatrice, requête SQL sur un entrepôt de commandes, escalade). Mesures : taux
de réussite des tâches, nombre d'étapes, coût, taux d'appel d'outil injustifié, respect du budget
d'étapes.

## 4. Arborescence et couches

```
templates/
├── modality/text/
│   ├── src/data/{__init__,schemas.py.j2,loaders.py}
│   ├── src/preprocessing/{__init__,transformers,pipelines}.py
│   ├── src/features/{__init__,build_features}.py
│   ├── src/models/base.py                  # contrat texte : fit / retrieve / answer
│   ├── src/training/{__init__,losses_metrics,trainer}.py
│   ├── src/pipelines/{__init__,data_pipeline,train_pipeline,evaluation_pipeline,inference_pipeline}.py
│   └── tests/{conftest.py.j2,test_*.py}
├── task/retrieval/                         # evaluator, reports, predictor, plots
├── task/text_multiclass/                   # classification de texte (tranche 3)
├── task/generation/                        # résumé / fine-tuning (tranche 4)
├── task/named_entity_recognition/          # extraction d'entités (tranche 3)
├── family/{retrieval_augmented_generation,question_answering,embedding_pipeline,agent_tools,text_classification,named_entity_recognition}
└── stack/{tfidf,tfidf_classifier,langchain,transformers,spacy,fastapi}
```

## 5. Conventions spécifiques à la modalité texte

- **Une question = une ligne**, comme un utilisateur = une liste de candidats en ranking : toutes
  les métriques sont calculées par requête puis moyennées.
- **Aucun identifiant en feature** : le retriever ne voit jamais `doc_id` ni `query_id`.
- **Le corpus est séparé en trois** : documents de référence (train), questions de validation,
  questions de test. Les questions de test ne servent **jamais** à régler un paramètre.
- **Toute réponse porte ses citations** ; une réponse sans citation est comptée comme non
  fondée, même si elle est juste.
- **Fidélité mesurée** : proportion des citations qui pointent réellement vers un passage doré.

## 6. Plan de livraison

| Tranche | Contenu | État |
| --- | --- | --- |
| 1 | `stacks` tfidf + langchain, `modality/text`, `task/retrieval`, famille `retrieval_augmented_generation`, notebooks texte, deux manifests, vérification verte | **livrée** : `ai-eng/rag/with-tfidf` (104 tests, 81,5 s) et `ai-eng/rag/with-langchain` (110 tests, 165,1 s), recall@5 de test 0,7194 dans les deux cas |
| 2 | `question_answering` (mutualise `modality/text`, `task/retrieval`, `stack/{tfidf,langchain}`) | **livrée** : `ai-eng/question-answering/with-tfidf` (106 tests, 48,8 s) et `with-langchain` (111 tests, 85,4 s), exact match 0,6212 et F1 0,7381 dans les deux cas |
| 2b | `embedding_pipeline` (stack `embedding` : hachage + SVD, voisins, fidélité) et `agent_tools` (familles restantes de la modalité texte) | **livrée pour `embedding_pipeline`** : `ai-eng/embeddings/with-embedding` (109 tests, 46,0 s) et `with-tfidf` (107 tests, 39,2 s), recall@5 0,9412 dans les deux cas ; `agent_tools` à venir |
| 3 | `ai-eng/` : `text_classification` (tfidf, transformers), `named_entity_recognition` (spacy) | **livrée** : `ai-eng/text-classification/with-tfidf_classifier` (121 tests, 41,8 s) et `with-transformers` (127 tests, 1 056,1 s), F1 macro 0,8408 / 0,7942 ; `ai-eng/named-entity-recognition/with-spacy` (101 tests, 153,6 s), F1 entité 0,9340 |
| 4 | `nlp/` : `summarization`, `llm_finetuning` (transformers, architecture minuscule, poids aléatoires) | à venir |
| 5 | `mlops/model_serving` (fastapi) au-dessus d'un modèle entraîné | à venir |

## 7. Tranche 2 — famille `question_answering`

Le contrat de cette famille est la **phrase exacte** de la fiche, pas une réponse appuyée sur un
passage : `answer_exact_match` devient donc une métrique de pilotage, et le corpus est écrit pour la
rendre atteignable (chaque question est rédigée à partir de la phrase canonique de sa fiche, qui est
sa réponse de référence mot pour mot). Les questions mono-document portent le nom du produit — seul
discriminant lexical entre vingt fiches qui se ressemblent —, les paraphrases le gardent et
s'éloignent du vocabulaire de la fiche, les questions multi-document demandent deux valeurs à la
fois et les questions hors corpus portent sur un fait voisin mais absent.

### Ce que la tranche a appris (et corrigé)

1. **Un artefact qui oublie son générateur répond autrement que le modèle entraîné.** La
   configuration `model.llm` (`min_overlap`, `support_ratio`) décide du nombre de phrases citées :
   au rechargement, le modèle retombait sur les défauts et annexait des phrases parasites. Les deux
   stacks texte archivent désormais le nœud `model.llm` (jamais la clé API) et le fusionnent à la
   lecture ; `load_model(path)` sans configuration répond exactement comme le modèle ajusté.
2. **`min_overlap = 2` : une phrase ne complète une réponse que si elle partage deux mots de
   contenu avec la question.** La phrase d'ouverture commune à toutes les fiches ne peut plus être
   citée « au titre d'un mot » (l'exact match du segment facile passe de 0,6047 à 0,8140).
3. **`support_ratio = 0,8` : les passages dont le score tombe sous 80 % du meilleur sont écartés du
   choix de phrase.** Sans ce garde-fou, une question sur les *frais de retour* recevait la phrase
   du *délai de retour* (ratio 0,68) et perdait son exact match ; avec lui, la précision des
   citations passe de 0,8854 à 0,9808. Le coût est mesuré et publié : la seconde fiche d'une
   question multi-document (ratio 0,69-0,74) est écartée elle aussi, donc le modèle répond avec la
   fiche la mieux classée (F1 0,6664 contre 0,9090 sans le garde-fou) — le rapport l'explique au
   lieu de le taire.
4. **Le seuil d'abstention calibré et la couverture se lisent ensemble.** Le seuil appris sur le
   split de calibration (41,36) refuse 19 des 72 questions de test, dont 5 des 6 hors corpus : la
   couverture (0,7879) et l'exactitude équilibrée (0,8106) sont publiées côte à côte, et la fiche
   de modèle expose désormais le seuil **effectif**, pas le placeholder de la configuration.

### Mesures finales (split de test, identiques sur les deux stacks)

| Métrique | Valeur | Lecture |
| --- | --- | --- |
| `recall_at_1` (principale) | 0,8258 | la bonne fiche est première ; plancher de non-régression 0,70 |
| `answer_exact_match` | 0,6212 | 4 réponses faciles sur 5 sont la phrase exacte de la fiche |
| `answer_f1` | 0,7381 | le F1 pardonne la troncature, l'exact match non |
| `citation_precision` | 0,9808 | une citation ne pointe presque jamais à côté |
| `abstention_balanced_accuracy` | 0,8106 | refuser reste une décision mesurée, pas un aveu |
| Latence p95 | 5,3 ms (tfidf) / 11,0 ms (langchain) | l'orchestration coûte ~6 ms, hors réseau |

## 8. Tranche 2b — famille `embedding_pipeline`

Troisième contrat de la modalité texte, et le seul où l'objet mesuré n'est pas la réponse mais
**l'index** : dimension réellement stockée, fidélité de la projection, taille de la matrice, débit de
requêtes, taux de succès du cache — plus le second usage du même artefact, la détection de
quasi-doublons. Le corpus est écrit trois fois (notice constructeur, fiche commerciale, note SAV) :
les deux phrases du fait sont recopiées d'une fiche à l'autre, et les trois fiches sont annotées
pertinentes, ce qui plafonne `recall_at_1` au tiers (un sixième sur les questions multi-document) et
impose une métrique principale qui ne dépende pas du rang du premier candidat (`recall_at_5`, plancher
0,90).

### Ce que la tranche a appris (et corrigé)

1. **La dimension publiée est celle qui est stockée, pas celle qui est demandée.** Une SVD tronquée
   réduit ses composantes quand le corpus est plus petit que la dimension visée ; la fiche de modèle
   publiait le `n_components` de la configuration. Elle expose désormais la dimension de l'embedder
   ajusté, la taille de la matrice (`matrix_bytes`, `matrix_mb`), la fidélité de projection et le
   nombre de passages — les propriétés d'index remontent par `_extra_metadata` → `FitResult.extra` →
   carte, jamais par une clé inventée au moment du rapport.
2. **Un paramètre déclaré mais non lu est un mensonge de configuration.** `HashingEmbedder`
   acceptait `ngram_range` mais `from_config` l'ignorait : une configuration `[1, 1]` produisait des
   bigrammes en silence. Le paramètre est maintenant lu (avec contrôle du nombre de bornes) et un
   test de non-régression verrouille le comportement.
3. **Compresser n'est pas sématiser.** La projection en 128 dimensions conserve l'ordre des
   similarités de l'espace de hachage brut (fidélité 1,0000, matrice de 0,16 Mo contre 5,25 Mo,
   trente-deux fois plus petite) mais ne rapproche pas deux formulations : les deux variantes tombent
   à 0,60 d'exact match sur les paraphrases, contre 1,00 sur les questions factuelles (34 % de mots
   pleins partagés contre 71 %). C'est la limite mesurée de la famille, publiée comme les autres.
4. **Le même index sert deux cas d'usage, et le second se teste.** `nearest_neighbours(exclude_doc_id=…)`
   retrouve les deux autres écritures d'un fait : la suite de la stack vérifie la propriété pour
   **toutes** les fiches du corpus (deux voisins chacune), et le notebook 04 en fait sa quatrième
   section pour la seule variante dense — la variante lexicale garde trois sections, sa stack
   n'exposant pas l'API.
5. **Le dense ne gagne pas partout, et le projet le dit.** Sur ce corpus (phrases recopiées mot pour
   mot), l'index dense gagne le classement (MRR 0,9608 contre 0,9118) et l'exact match (0,7941 contre
   0,7647) ; l'index creux garde la précision des citations (0,9667 contre 0,9375), coûte 0,07 s au
   lieu de 0,42 s et refuse 3 des 7 questions hors corpus contre 4. Chaque variante publie ses
   propres chiffres (tableau §12 de son README) plutôt qu'un tableau de famille unique.

### Mesures finales (split de test, 41 questions dont 34 répondables)

| Métrique | `with-embedding` | `with-tfidf` | Lecture |
| --- | --- | --- | --- |
| `recall_at_5` (principale) | 0,9412 | 0,9412 | les trois fiches du fait sont retrouvées ; plancher 0,90 |
| `mrr` / `ndcg_at_10` | 0,9608 / 0,9682 | 0,9118 / 0,9462 | le dense classe mieux, le lexical suit |
| `answer_exact_match` | 0,7941 | 0,7647 | 1,00 sur les questions factuelles, 0,60 sur les paraphrases |
| `citation_precision` | 0,9375 | 0,9667 | l'index creux cite plus juste |
| `abstention_balanced_accuracy` | 0,7563 | 0,6555 | publiée avec la couverture (0,94 / 0,88) |
| Index | 128 dimensions, 0,16 Mo, fidélité 1,0000 | 138 termes, index creux | le sujet de la famille |
| Latence p95 | 4,64 ms | 7,44 ms | cache de requêtes armé (512 entrées) |
