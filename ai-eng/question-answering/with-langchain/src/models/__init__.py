"""Model layer of the **LangChain (LCEL)** stack.

Public surface, imported everywhere else in the project::

    from src.models import build_model, load_model          # fabrique + rechargement
    from src.models.base import BaseModel, RetrievedChunk   # contrat et objets de retour
    from src.models.chain import build_retrieval_chain      # graphe LCEL de bout en bout
    from src.models.factory import available_algorithms     # stratégies servies par la stack

Nothing outside this package imports a ``langchain_core`` class: the framework stays an
implementation detail behind :class:`~src.models.base.BaseModel` (dependency inversion). The one
exception is deliberate — :class:`~src.models.chain.ChunkRetriever` *is* a ``BaseRetriever``, and
that is the point of the stack: the chain composes it with a prompt and an LLM adapter.
"""

from src.models.base import Answer, BaseModel, FitResult, ModelCard, RetrievedChunk
from src.models.chain import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_PROMPT_TEMPLATE,
    ChunkRetriever,
    GroundedAnswer,
    LangChainGroundedLLM,
    build_answer_chain,
    build_prompt,
    build_retrieval_chain,
    format_documents,
)
from src.models.factory import (
    ALGORITHMS,
    AlgorithmSpec,
    available_algorithms,
    build_model,
    describe_algorithm,
    load_model,
    supported_tasks,
)
from src.models.model import DOCUMENT_COLUMNS, LangChainModel

__all__ = [
    "ALGORITHMS",
    "AlgorithmSpec",
    "Answer",
    "BaseModel",
    "DOCUMENT_COLUMNS",
    "DEFAULT_MAX_CONTEXT_CHARS",
    "DEFAULT_PROMPT_TEMPLATE",
    "ChunkRetriever",
    "FitResult",
    "GroundedAnswer",
    "LangChainGroundedLLM",
    "LangChainModel",
    "ModelCard",
    "RetrievedChunk",
    "available_algorithms",
    "build_answer_chain",
    "build_model",
    "build_prompt",
    "build_retrieval_chain",
    "describe_algorithm",
    "format_documents",
    "load_model",
    "supported_tasks",
]
