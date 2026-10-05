"""Tests propres à la stack **LangChain (LCEL)**.

Ce module vit dans la couche `stack/langchain` : il a le droit de connaître `langchain-core`, les
noms d'algorithmes et la forme du graphe. Les tests partagés de la modalité texte ne les
connaissent pas, ce qui permet à une seconde stack de servir la même famille sans les réécrire.

Ce qui est vérifié ici n'est pas le contrat (la modalité le teste déjà) mais ce que la stack
apporte : le retrieval passe par un `BaseRetriever`, la chaîne compose contexte, prompt, adaptateur
et ancrage, la fusion hybride combine réellement les deux classements, et la chaîne est
**reconstruite** au chargement puisqu'elle n'est pas sérialisée.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

from src.models import available_algorithms, build_model, load_model
from src.models.base import BaseModel
from src.models.chain import (
    GroundedAnswer,
    LangChainGroundedLLM,
    build_prompt,
    format_documents,
)


@pytest.fixture(scope="session")
def corpus_question(queries: pd.DataFrame) -> str:
    """Une question factuelle du corpus du projet, lue dans les données et jamais codée en dur.

    La même suite de tests sert plusieurs familles (RAG, questions-réponses, ...) : une question
    littérale ne testerait que le corpus de l'une d'elles. La question est tirée des questions
    annotées par la famille — la première « facile », dont la réponse est un fait planté — pour que
    le test vérifie bien ce qu'il annonce : une question que le corpus peut trancher ne doit pas
    être une abstention.
    """
    factual = queries.loc[queries["difficulty"] == "facile"]
    if factual.empty:
        factual = queries.loc[queries["answer_type"] != "unanswerable"]
    if factual.empty:  # pragma: no cover - le contrat des questions interdit ce cas
        pytest.skip("le corpus du projet n'annote aucune question factuelle")
    return str(factual.iloc[0]["question"])


def test_the_stack_serves_three_retrieval_strategies() -> None:
    """Les trois stratégies de la stack sont déclarées par le registre, pas par ce test."""
    assert {"langchain_lexical", "langchain_dense", "langchain_hybrid"} <= set(
        available_algorithms()
    )


def test_the_retriever_is_a_langchain_base_retriever(
    fitted_model: BaseModel, corpus_question: str
) -> None:
    """Le retrieval passe par le point d'extension du framework, pas par une méthode maison."""
    retriever = fitted_model.retriever  # type: ignore[attr-defined]
    assert isinstance(retriever, BaseRetriever)
    documents = retriever.invoke(corpus_question)
    assert documents, "le retriever doit rendre au moins un passage"
    assert all(isinstance(document, Document) for document in documents)
    first = documents[0]
    assert set(first.metadata) >= {"chunk_id", "doc_id", "rank", "score"}
    assert first.metadata["rank"] == 1
    assert first.page_content == fitted_model.retrieve(corpus_question, 1)[0].text


def test_the_chain_answers_with_its_citations(
    fitted_model: BaseModel, corpus_question: str
) -> None:
    """Chaîne de bout en bout : une question, une réponse fondée sur les passages retrouvés."""
    answer: GroundedAnswer = fitted_model.chain.invoke(corpus_question)  # type: ignore[attr-defined]
    assert isinstance(answer, GroundedAnswer)
    assert not answer.abstained, "la question porte sur un fait planté du corpus"
    assert answer.citations, "une réponse sans citation n'est pas traçable"
    known = {str(document.metadata["chunk_id"]) for document in answer.documents}
    assert set(answer.citations) <= known
    assert corpus_question in answer.prompt
    assert answer.context in answer.prompt
    assert answer.to_generation().cited_chunk_ids == answer.citations


def test_the_prompt_comes_from_the_configuration() -> None:
    """Le prompt est un objet de configuration : le code ne doit pas en figer le texte."""
    prompt, budget = build_prompt(
        {
            "model": {
                "prompt": {"template": "C: {context} | Q: {question}", "max_context_chars": 120}
            }
        }
    )
    assert isinstance(prompt, PromptTemplate)
    assert prompt.input_variables == ["context", "question"]
    assert budget == 120
    rendered = prompt.invoke({"context": "un contexte", "question": "une question"}).to_string()
    assert rendered == "C: un contexte | Q: une question"


def test_a_prompt_without_variables_is_refused() -> None:
    """Un prompt qui ne reçoit pas les passages répondrait de mémoire : il est refusé au montage."""
    with pytest.raises(ValueError, match="context"):
        build_prompt({"model": {"prompt": {"template": "Réponds à la question."}}})


def test_the_context_budget_drops_the_excess_passages() -> None:
    """Le contexte est borné : les passages qui ne tiennent pas sont écartés, jamais tronqués."""
    documents = [
        Document(page_content="passage court", metadata={"rank": index, "chunk_id": f"C{index}"})
        for index in range(1, 6)
    ]
    context = format_documents(documents, max_chars=60)
    assert "passage court" in context
    assert context.count("[") < len(documents)


def test_the_hybrid_arm_fuses_ranks_of_both_arms(
    documents: pd.DataFrame, corpus_question: str
) -> None:
    """RRF combine les *rangs* des deux bras : le score recalculé ici doit être celui du modèle."""
    model = build_model({"model": {"algorithm": "langchain_hybrid"}})
    model.fit(documents)
    retriever = model.retriever  # type: ignore[attr-defined]

    # Les deux bras sont lus *sur le retriever* plutôt que recalculés : un ordre d'opérations
    # différent suffit à permuter deux passages à égalité stricte (des corpus très templés en
    # produisent beaucoup), et la fusion compare des rangs, pas des valeurs.
    lexical = np.asarray(
        retriever.lexical.score(corpus_question, retriever.lexical_matrix), dtype="float64"
    )
    arms = (lexical, retriever._dense_scores(corpus_question))
    expected = np.zeros(len(retriever.chunks))
    for scores in arms:
        for rank, index in enumerate(np.argsort(-scores, kind="stable"), start=1):
            expected[int(index)] += 1.0 / (retriever.rrf_constant + rank)

    np.testing.assert_allclose(retriever.score_query(corpus_question), expected, rtol=1e-12)
    fused = [document.metadata["chunk_id"] for document in retriever.invoke(corpus_question)]
    assert len(fused) == len(set(fused))


def test_the_dense_arm_does_not_rank_like_the_lexical_one(
    documents: pd.DataFrame, corpus_question: str
) -> None:
    """Deux représentations distinctes, deux classements distincts : sinon l'une des deux ment."""
    lexical = build_model({"model": {"algorithm": "langchain_lexical"}})
    lexical.fit(documents)
    dense = build_model({"model": {"algorithm": "langchain_dense"}})
    dense.fit(documents)
    lexical_ids = [passage.chunk_id for passage in lexical.retrieve(corpus_question, 10)]
    dense_ids = [passage.chunk_id for passage in dense.retrieve(corpus_question, 10)]
    assert lexical_ids != dense_ids


def test_the_chain_is_rebuilt_at_load_time(
    fitted_model: BaseModel, corpus_question: str, tmp_path: Path
) -> None:
    """L'artefact ne contient pas le graphe : il est reconstruit, et répond à l'identique.

    Sérialiser un `Runnable` LCEL serait fragile (fermetures, versions) : l'état est persisté et la
    chaîne remontée depuis la configuration. Le test vérifie que la reconstruction est fidèle.
    """
    path = fitted_model.save(tmp_path / "rag_chain.joblib")
    reloaded = load_model(path)
    assert isinstance(reloaded.llm, LangChainGroundedLLM)
    assert isinstance(reloaded.retriever, BaseRetriever)  # type: ignore[attr-defined]
    # Le générateur est de la *configuration*, pas de l'état : il doit être reconstruit à
    # l'identique, sinon l'artefact recharge répondrait sous une autre règle que celle entraînée.
    assert reloaded.llm.describe()["adapter"] == fitted_model.llm.describe()["adapter"]
    before = fitted_model.chain.invoke(corpus_question)  # type: ignore[attr-defined]
    after = reloaded.chain.invoke(corpus_question)  # type: ignore[attr-defined]
    assert after.citations == before.citations
    assert after.text == before.text
    assert [document.metadata["chunk_id"] for document in after.documents] == [
        document.metadata["chunk_id"] for document in before.documents
    ]


def test_the_adapter_receives_the_prompt_rendered_by_the_chain(
    fitted_model: BaseModel, corpus_question: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L'adaptateur LLM est un Runnable branché sur le prompt : il reçoit le texte rendu."""
    captured: dict[str, str] = {}
    adapter = fitted_model._adapter  # type: ignore[attr-defined]
    original = adapter.generate

    def spy(question: str, passages: object, *, prompt: str | None = None):  # type: ignore[no-untyped-def]
        captured["prompt"] = str(prompt)
        return original(question, passages, prompt=prompt)  # type: ignore[arg-type]

    monkeypatch.setattr(adapter, "generate", spy)
    answer = fitted_model.chain.invoke(corpus_question)  # type: ignore[attr-defined]
    # Le texte du prompt appartient à la configuration du projet : le test vérifie que ce que la
    # chaîne a rendu contient bien le contexte assemblé et la question, pas un libellé littéral.
    assert corpus_question in captured["prompt"]
    assert answer.context in captured["prompt"]
