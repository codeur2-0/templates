"""Language models used to turn retrieved passages into an answer.

The project needs a *generation* step, but it must run without any API key, any network access
and any cost. The resolution is an explicit interface with two implementations:

* :class:`ExtractiveLLM` — the default. It selects the sentences of the retrieved passages that
  best answer the question and stitches them into a short answer, always with citations. It is
  deterministic, instant, and honest about what it is: no fluency, no paraphrase, no world
  knowledge.
* :class:`OpenAICompatibleLLM` — an optional HTTP client for any OpenAI-compatible endpoint
  (vLLM, Ollama, Azure, OpenAI). It is never instantiated unless ``model.llm.provider`` is set
  to ``openai`` **and** an API key is present; the tests use a stub transport, never the network.

Showing the seam matters pedagogically: the retrieval quality (measured by recall@k) is what
the project can actually control, and swapping the generator changes the wording, not the
grounding. The evaluation therefore scores *grounding* (citations, abstention) separately from
*fluency*.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.preprocessing.transformers import split_sentences, tokenize
from src.utils.config_access import as_mapping
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class Passage:
    """A passage handed to a generator.

    Attributes:
        chunk_id: Identifier of the passage (used for citations).
        doc_id: Identifier of the source document.
        text: Passage text.
        score: Retrieval score of the passage.
        rank: Rank of the passage in the retrieved list (1-based).
        title: Title of the source document, when known.
        section: Editorial section of the source document, when known.
    """

    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    title: str = ""
    section: str = ""


@dataclass(slots=True)
class Generation:
    """What a generator produced.

    Attributes:
        text: The answer (empty string when the generator refuses to answer).
        cited_chunk_ids: Passages actually used to build the answer.
        rationale: Short human readable explanation of the choice (used in the report).
    """

    text: str
    cited_chunk_ids: list[str] = field(default_factory=list)
    rationale: str = ""


class BaseLLM(ABC):
    """Interface implemented by every generator."""

    #: Identifier used in the model card and the configuration.
    provider: str = "base"

    @abstractmethod
    def generate(
        self, question: str, passages: Sequence[Passage], *, prompt: str | None = None
    ) -> Generation:
        """Answer a question from retrieved passages.

        Args:
            question: User question.
            passages: Retrieved passages, best first.
            prompt: Prompt already rendered by the caller (an orchestrator such as the LangChain
                chain owns the prompt and passes it down). Adapters that build their own prompt
                ignore it; adapters that can follow one use it instead of rebuilding it.

        Returns:
            The generation (possibly an explicit refusal).
        """

    def describe(self) -> dict[str, Any]:
        """Return a JSON-friendly description of the generator (model card)."""
        return {"provider": self.provider}


@dataclass
class ExtractiveLLM(BaseLLM):
    """Deterministic extractive generator: no parameter, no download, no hallucination.

    The algorithm is a scoring of the sentences of the retrieved passages:

    1. each sentence is scored by the share of the question's content words it contains,
       weighted by the passage's retrieval score and penalised by its position in the passage;
    2. the best sentence is kept, then a second one is appended only if it adds new question
       words (answers that need two facts are common in procedures);
    3. the answer is the concatenation of the kept sentences, and the cited passages are those
       the sentences come from.

    When no sentence shares a minimum number of question words, the generator refuses to
    answer (empty text, no citation) — the abstention is decided here *and* by the retriever's
    score threshold, and both are measured.

    Attributes:
        max_sentences: Maximum number of sentences in the answer.
        min_overlap: Minimum number of shared content words for a sentence to be usable.
        max_chars: Hard cap on the answer length.
    """

    max_sentences: int = 2
    min_overlap: int = 1
    max_chars: int = 480
    provider = "extractive"

    def generate(
        self, question: str, passages: Sequence[Passage], *, prompt: str | None = None
    ) -> Generation:
        """Build the answer from the best sentences of the retrieved passages.

        Args:
            question: User question.
            passages: Retrieved passages, best first.
            prompt: Rendered prompt of the caller, **ignored by construction**: an extractive
                generator selects sentences, it does not follow instructions. The argument exists
                so that every adapter of the project exposes the same interface.

        Returns:
            The extractive answer with its citations.
        """
        del prompt
        question_tokens = [token for token in tokenize(question) if len(token) > 2]
        if not question_tokens or not passages:
            return Generation(text="", rationale="no passage or no content word in the question")

        scored: list[tuple[float, int, str, str]] = []
        for passage in passages:
            sentences = split_sentences(passage.text) or [passage.text]
            for position, sentence in enumerate(sentences):
                sentence_tokens = set(tokenize(sentence))
                overlap = len(sentence_tokens & set(question_tokens))
                if overlap < self.min_overlap:
                    continue
                score = overlap + 0.25 * passage.score - 0.01 * position
                scored.append((score, passage.rank, sentence, passage.chunk_id))
        if not scored:
            return Generation(text="", rationale="no sentence overlaps the question")

        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        kept: list[str] = []
        cited: list[str] = []
        covered: set[str] = set()
        for _, _, sentence, chunk_id in scored:
            if len(kept) >= self.max_sentences:
                break
            new_words = set(tokenize(sentence)) & set(question_tokens) - covered
            if kept and not new_words:
                continue
            kept.append(sentence.strip())
            cited.append(chunk_id)
            covered |= new_words
        answer = " ".join(kept)
        if len(answer) > self.max_chars:
            answer = answer[: self.max_chars].rsplit(" ", 1)[0] + "..."
        return Generation(
            text=answer,
            cited_chunk_ids=list(dict.fromkeys(cited)),
            rationale=f"{len(kept)} sentence(s) kept out of {len(scored)} candidate(s)",
        )

    def describe(self) -> dict[str, Any]:
        """Describe the generator for the model card."""
        return {
            "provider": self.provider,
            "max_sentences": self.max_sentences,
            "min_overlap": self.min_overlap,
        }


@dataclass
class OpenAICompatibleLLM(BaseLLM):
    """Optional client for an OpenAI-compatible chat completion endpoint.

    The client uses :mod:`urllib` from the standard library: adding an HTTP dependency for the
    fifteen lines of a chat completion call would be a cost without a benefit. It is only
    instantiated when the configuration selects it **and** the key is available in the
    environment, so a missing key degrades to the extractive generator instead of crashing a
    pipeline.

    Attributes:
        model: Model name sent to the endpoint.
        base_url: Base URL of the API (``https://api.openai.com/v1`` by default).
        api_key: Secret, read from the environment by :meth:`from_config`.
        temperature: Sampling temperature (0 by default, for reproducibility).
        max_tokens: Maximum number of generated tokens.
        timeout_seconds: HTTP timeout.
    """

    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 320
    timeout_seconds: float = 30.0
    provider = "openai"

    @classmethod
    def from_config(
        cls, config: Any, *, env: dict[str, str] | None = None
    ) -> OpenAICompatibleLLM | None:
        """Build the client when the configuration selects it and a key is available.

        Args:
            config: ``model.llm`` configuration node.
            env: Environment mapping (defaults to ``os.environ``), injectable for the tests.

        Returns:
            The configured client, or ``None`` when the provider is not selected or the key is
            missing (the caller then falls back on the extractive generator).
        """
        environment = os.environ if env is None else env
        settings = as_mapping(config)
        provider = str(settings.get("provider", "extractive"))
        if provider != "openai":
            return None
        key = str(environment.get("OPENAI_API_KEY", "")).strip()
        if not key:
            logger.warning(
                "model.llm.provider=openai but OPENAI_API_KEY is not set: falling back on the "
                "extractive generator (no network call is attempted)"
            )
            return None
        return cls(
            model=str(settings.get("model", "gpt-4o-mini")),
            base_url=str(settings.get("base_url", "https://api.openai.com/v1")).rstrip("/"),
            api_key=key,
            temperature=float(settings.get("temperature", 0.0)),
            max_tokens=int(settings.get("max_tokens", 320)),
        )

    def build_prompt(self, question: str, passages: Sequence[Passage]) -> str:
        """Render the grounded prompt sent to the model.

        Args:
            question: User question.
            passages: Retrieved passages.

        Returns:
            The prompt, passages numbered so that the answer can cite them.
        """
        context = "\n\n".join(
            f"[{passage.rank}] ({passage.chunk_id}) {passage.text}" for passage in passages
        )
        return (
            "Tu réponds uniquement à partir du contexte fourni. "
            "Si le contexte ne contient pas la réponse, réponds exactement 'ABSTENTION'. "
            "Termine ta réponse par 'SOURCES: ' suivi des identifiants de passages utilisés.\n\n"
            f"Contexte:\n{context}\n\nQuestion: {question}\nRéponse:"
        )

    def generate(
        self, question: str, passages: Sequence[Passage], *, prompt: str | None = None
    ) -> Generation:
        """Call the endpoint and parse the answer.

        Args:
            question: User question.
            passages: Retrieved passages.
            prompt: Rendered prompt of the caller. When an orchestrator (the LangChain chain)
                owns the prompt, its version is sent as-is; otherwise the adapter builds the
                grounded prompt from the passages.

        Returns:
            The generated answer with the passages it cites.

        Raises:
            RuntimeError: When the endpoint answers with an error or an unexpected payload.
        """
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": "Assistant documentaire strictement fondé sur le contexte.",
                },
                {
                    "role": "user",
                    "content": prompt or self.build_prompt(question, passages),
                },
            ],
        }
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            msg = f"LLM endpoint call failed: {type(error).__name__}: {error}"
            raise RuntimeError(msg) from error
        try:
            text = str(body["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as error:
            msg = f"Unexpected LLM response payload: {str(body)[:200]}"
            raise RuntimeError(msg) from error
        return self._parse(text, passages)

    @staticmethod
    def _parse(text: str, passages: Sequence[Passage]) -> Generation:
        """Split the model answer from its ``SOURCES:`` citation line."""
        if "ABSTENTION" in text.upper():
            return Generation(text="", rationale="the model abstained on the given context")
        answer, _, sources = text.partition("SOURCES:")
        cited = [
            identifier.strip()
            for identifier in sources.replace(",", " ").split()
            if identifier.strip()
        ]
        known = {passage.chunk_id for passage in passages}
        cited = [identifier for identifier in cited if identifier in known]
        return Generation(text=answer.strip(), cited_chunk_ids=cited, rationale="remote completion")

    def describe(self) -> dict[str, Any]:
        """Describe the generator for the model card (never the key)."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
        }


def build_llm(config: Any, *, env: dict[str, str] | None = None) -> BaseLLM:
    """Instantiate the configured generator, falling back on the extractive one.

    Args:
        config: ``model.llm`` configuration node (or the whole ``model`` node; the ``llm``
            sub-node is used when present).
        env: Environment mapping, injectable for the tests.

    Returns:
        The generator to use.
    """
    settings = as_mapping(config)
    remote = OpenAICompatibleLLM.from_config(settings.get("llm", settings), env=env)
    if remote is not None:
        return remote
    return ExtractiveLLM(
        max_sentences=int(settings.get("max_sentences", 2)),
        min_overlap=int(settings.get("min_overlap", 1)),
    )


__all__ = [
    "BaseLLM",
    "ExtractiveLLM",
    "Generation",
    "OpenAICompatibleLLM",
    "Passage",
    "build_llm",
]
