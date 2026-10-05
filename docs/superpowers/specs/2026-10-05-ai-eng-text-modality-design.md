# Modalité `text` et familles `ai-eng` — RAG, QA, embeddings, agents

- **Date** : 2026-10-05
- **Statut** : design validé, tranche 1 livrée et vérifiée (`ai-eng/rag/with-tfidf` et
  `ai-eng/rag/with-langchain` conformes dans `tools.verify --all`), tranches 2 à 5 à venir
- **Branche** : `arena/3402658e-templates`
- **Périmètre** : point 2 de la feuille de route du README racine — `ai-eng/` (LangChain, RAG,
  Transformers, serving FastAPI) et `nlp/` (spaCy, Transformers). Ce document couvre la couche
  `modality/text/` et les quatre familles ai-eng de la première tranche.

## 1. Objectif

Le dépôt déclare **29 familles** et **18 stacks**, mais une seule modalité est implémentée
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
├── task/generation/                        # résumé / fine-tuning (tranche 2)
├── family/{retrieval_augmented_generation,question_answering,embedding_pipeline,agent_tools}
└── stack/{tfidf,langchain,transformers,spacy,fastapi}
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
| 2 | `embedding_pipeline`, `agent_tools`, `question_answering` (familles restantes de la modalité texte) | à venir |
| 3 | `nlp/` : `text_classification` (tfidf, transformers), `named_entity_recognition` (spacy) | à venir |
| 4 | `nlp/` : `summarization`, `llm_finetuning` (transformers, architecture minuscule, poids aléatoires) | à venir |
| 5 | `mlops/model_serving` (fastapi) au-dessus d'un modèle entraîné | à venir |
