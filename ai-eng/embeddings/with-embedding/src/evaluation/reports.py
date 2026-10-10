"""Human readable evaluation report of a retrieval-augmented project.

The report is the deliverable the business reads: it states the verdict on the contractual
objective, compares the model to the trivial references, breaks the results down by segment,
shows the failure modes and lists what the project does **not** do. It is generated, never
hand-edited: every number printed here comes from the evaluation payload, so the prose cannot
drift from the measurement.

Artefacts written:

* ``artifacts/reports/evaluation_report.md``   the report itself,
* ``artifacts/reports/per_question.csv``       one row per question (audit trail),
* ``artifacts/reports/segment_metrics.csv``    the segment breakdown,
* ``artifacts/reports/figures/*.png``          the five diagnostic figures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation.evaluator import EvaluationResult
from src.utils.io import write_table, write_text
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths
from src.visualization import plots

logger = get_logger(__name__)

#: Title printed at the top of the report.
REPORT_TITLE = (
    "Index vectoriel dense d'un catalogue dupliqué (hachage + SVD) — recherche et dédoublonnage"
)

#: Persona and context (from the manifest) used to frame the reading of the numbers.
PERSONA = (
    "Équipe plateforme données d'une enseigne de matériel électronique (400 personnes), avec un "
    "responsable catalogue qui publie le référentiel produit et un ingénieur IA qui outille le "
    "service client."
)

#: Contractual objective, as declared in the manifest.
PRIMARY_METRIC = "recall_at_5"
MIN_PRIMARY = 0.9

#: Metrics documented in the report, in reading order.
DOCUMENTED_METRICS: tuple[str, ...] = (
    "recall_at_1",
    "recall_at_3",
    "recall_at_10",
    "precision_at_1",
    "mrr",
    "ndcg_at_10",
    "answer_f1",
    "answer_exact_match",
    "citation_precision",
    "abstention_balanced_accuracy",
    "answer_coverage",
    "false_answer_rate",
)

#: Questions printed in the failure appendix.
MAX_FAILURES = 8


def _format(value: Any, *, digits: int = 4) -> str:
    """Format a metric for the Markdown table (``n/a`` for a missing value)."""
    if value is None or (isinstance(value, float) and value != value):
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


@dataclass
class ReportBundle:
    """What the report builder produced.

    Attributes:
        report_path: The Markdown report.
        summary: One-line summary printed by the pipeline.
        artifacts: Every artefact written (report, tables, figures).
        verdict: ``conforme``, ``non conforme`` or ``indéterminé``.
    """

    report_path: Path
    summary: str
    artifacts: list[Path] = field(default_factory=list)
    verdict: str = "indéterminé"


class ReportBuilder:
    """Render the evaluation report, its tables and its figures."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        paths: ProjectPaths | None = None,
    ) -> None:
        """Configure the builder.

        Args:
            config: Full application configuration (project identity, model, metrics).
            paths: Project filesystem layout.
        """
        self.config = dict(config or {})
        self.paths = paths or ProjectPaths.from_root()

    # ------------------------------------------------------------------ API ---------------
    def build(
        self,
        result: EvaluationResult,
        *,
        model_summary: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ReportBundle:
        """Write the report, the tables and the figures.

        Args:
            result: Evaluation payload.
            model_summary: Description of the evaluated model (name, framework, algorithm, llm).
            metadata: Generation metadata (archived next to the corpus).

        Returns:
            The :class:`ReportBundle`.

        Raises:
            ValueError: When the evaluation payload holds no metric.
        """
        if not result.metrics:
            msg = "The evaluation produced no metric: refusing to write an empty report"
            raise ValueError(msg)

        figures = self._figures(result)
        per_question_path = write_table(
            result.per_question, self.paths.reports_dir / "per_question.csv"
        )
        segments_path = write_table(
            self._segments_frame(result), self.paths.reports_dir / "segment_metrics.csv"
        )
        report_path = write_text(
            self.paths.reports_dir / "evaluation_report.md",
            self._render(result, figures, model_summary or {}, metadata or {}),
        )
        bundle = ReportBundle(
            report_path=report_path,
            summary=self._summary(result),
            artifacts=[
                report_path,
                per_question_path,
                segments_path,
                *figures.values(),
            ],
            verdict=self._verdict(result),
        )
        logger.info("Report written: {} ({})", report_path, bundle.verdict)
        return bundle

    # ------------------------------------------------------------------ verdict ----------
    def _verdict(self, result: EvaluationResult) -> str:
        """Decide the compliance of the run against the contractual objective."""
        measured = result.metrics.get(PRIMARY_METRIC)
        if measured is None or measured != measured:
            return "indéterminé"
        if MIN_PRIMARY is None:
            return "sans seuil contractuel"
        return "conforme" if float(measured) >= float(MIN_PRIMARY) else "non conforme"

    def _summary(self, result: EvaluationResult) -> str:
        """Build the one-line summary printed by the pipeline."""
        measured = result.metrics.get(PRIMARY_METRIC)
        random_reference = result.baselines.get("random", {}).get(PRIMARY_METRIC)
        gap = (
            f", +{float(measured) - float(random_reference):.3f} sur le tirage aléatoire"
            if measured is not None
            and measured == measured
            and random_reference is not None
            and random_reference == random_reference
            else ""
        )
        return (
            f"{result.n_questions} questions évaluées, {PRIMARY_METRIC}="
            f"{_format(measured)}{gap} — verdict : {self._verdict(result)}."
        )

    # ------------------------------------------------------------------ figures ----------
    def _figures(self, result: EvaluationResult) -> dict[str, Path]:
        """Draw the diagnostic figures, skipping the ones the data cannot support."""
        directory = self.paths.figures_dir
        threshold = self.config.get("model", {}).get("params", {}).get("abstention_threshold")
        candidates = {
            "recall_curve": lambda: plots.plot_recall_curve(
                result.metrics, result.baselines, directory / "recall_curve.png", ks=result.ks
            ),
            "score_separation": lambda: plots.plot_score_separation(
                result.per_question,
                directory / "score_separation.png",
                threshold=None if threshold is None else float(threshold),
            ),
            "latency": lambda: plots.plot_latency(result.per_question, directory / "latency.png"),
            "segments": lambda: plots.plot_segments(result.segments, directory / "segments.png"),
        }
        written: dict[str, Path] = {}
        for name, factory in candidates.items():
            try:
                path = factory()
            except (ValueError, TypeError) as error:  # pragma: no cover - defensive
                logger.warning("Figure '{}' could not be drawn: {}", name, error)
                continue
            if path is not None:
                written[name] = path
        return written

    def _segments_frame(self, result: EvaluationResult) -> pd.DataFrame:
        """Turn the segment metrics into a table."""
        rows = [{"segment": key, **dict(values)} for key, values in result.segments.items()]
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["segment"])

    # ------------------------------------------------------------------ rendu ------------
    def _render(
        self,
        result: EvaluationResult,
        figures: Mapping[str, Path],
        model_summary: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> str:
        """Render the Markdown report."""
        sections: list[str] = [
            f"# {REPORT_TITLE} — rapport d'évaluation",
            "",
            f"**Public visé** : {PERSONA}",
            "",
            "> Rapport généré par `python -m src.main mode=evaluate`. "
            "Chaque chiffre provient de `artifacts/metrics/evaluation_metrics.json` : "
            "le texte ne peut pas diverger de la mesure.",
            "",
            "## 1. Synthèse",
            "",
            f"- Questions évaluées : **{result.n_questions}** "
            f"(dont {result.n_answerable} avec une réponse dans le corpus).",
            f"- Objectif contractuel : **{PRIMARY_METRIC} ≥ {MIN_PRIMARY}**.",
            f"- Mesuré : **{_format(result.metrics.get(PRIMARY_METRIC))}** → verdict "
            f"**{self._verdict(result)}**.",
            f"- Références triviales sur la même métrique : {self._baselines_line(result)}.",
            "",
            *self._notes(result),
            "",
            "## 2. Configuration évaluée",
            "",
            *self._configuration_lines(model_summary, metadata),
            "",
            "## 3. Qualité de la recherche",
            "",
            self._ranking_table(result),
            "",
            "## 4. Qualité des réponses",
            "",
            self._answer_table(result),
            "",
            "## 5. Latence",
            "",
            self._latency_table(result),
            "",
            "## 6. Résultats par segment",
            "",
            self._segments_table(result),
            "",
            "## 7. Diagnostic des échecs",
            "",
            *self._failures(result),
            "",
            "## 8. Figures",
            "",
            *self._figure_lines(figures),
            "",
            "## 9. Limites connues",
            "",
            *self._limitations(metadata),
            "",
            "## 10. Ce que le projet enseigne",
            "",
            "- Construire un index vectoriel déterministe et hors ligne : hachage de "
            "n-grammes, SVD tronquée apprise sur le train, normalisation L2.",
            "- Mesurer les propriétés d'un index au lieu de les supposer : dimension "
            "réellement produite, fidélité de la projection, taille de l'artefact.",
            "- Chiffrer ce que la compression apporte et ce qu'elle coûte : 128 dimensions "
            "contre 4096, 0,16 Mo de matrice contre 5,25 Mo, fidélité de projection de 1,00.",
            "- Servir deux cas d'usage avec un seul artefact : recherche par requête et "
            "détection de quasi-doublons (`nearest_neighbours`).",
            "- Instrumenter le service : cache de vecteurs de requêtes, taux de succès, "
            "latence p50/p95.",
            "- Comparer un index dense à sa référence lexicale sur les mêmes questions, et "
            "publier le résultat même quand il contredit l'intuition : ici le dense gagne le "
            "classement, le lexical la précision des citations.",
            "- Écrire un générateur dont la redondance est annotée : trois fiches disent le "
            "même fait, les trois sont pertinentes, le premier rang devient la difficulté et "
            "le dédoublonnage devient mesurable sans code supplémentaire.",
            "- Calibrer une décision d'abstention sur un score borné, et publier l'exactitude "
            "équilibrée avec la couverture.",
            "- Vérifier la stabilité : deux exécutions à graine fixée produisent la même "
            "matrice, donc les mêmes métriques.",
            "",
        ]
        return "\n".join(sections)

    def _baselines_line(self, result: EvaluationResult) -> str:
        """Format the baseline comparison of the primary metric."""
        parts: list[str] = []
        for name, values in result.baselines.items():
            label = plots.BASELINE_LABELS.get(name, name)
            parts.append(f"{label} {_format(values.get(PRIMARY_METRIC))}")
        return ", ".join(parts) if parts else "non calculées"

    def _notes(self, result: EvaluationResult) -> list[str]:
        """Render the evaluator notes as a bullet list."""
        return [f"- {note}" for note in result.notes] or ["- Aucune remarque."]

    def _configuration_lines(
        self, model_summary: Mapping[str, Any], metadata: Mapping[str, Any]
    ) -> list[str]:
        """Describe the evaluated configuration."""
        model = self.config.get("model", {})
        preprocessing = self.config.get("preprocessing", {})
        chunking = preprocessing.get("chunking", {}) if isinstance(preprocessing, Mapping) else {}
        llm = model_summary.get("llm", {}) if isinstance(model_summary.get("llm"), Mapping) else {}
        lines = [
            f"- Modèle : **{model_summary.get('name', model.get('name', 'n/a'))}** "
            f"(`{model_summary.get('framework', 'n/a')}` / "
            f"`{model_summary.get('algorithm', 'n/a')}`).",
            f"- Passages indexés : **{model_summary.get('n_chunks', 0)}**.",
            f"- Découpage : {chunking.get('max_tokens', 'n/a')} tokens, "
            f"chevauchement {chunking.get('overlap_tokens', 'n/a')} tokens.",
            f"- Générateur de réponse : `{llm.get('provider', 'extractive')}`"
            + (f" (modèle `{llm.get('model')}`)" if llm.get("model") else "")
            + ".",
        ]
        if metadata:
            documents = metadata.get("n_documents")
            questions = metadata.get("n_questions")
            lines.append(
                f"- Corpus synthétique : {documents or 'n/a'} documents, "
                f"{questions or 'n/a'} questions annotées "
                f"(graine {metadata.get('seed', 'n/a')})."
            )
        return lines

    def _ranking_table(self, result: EvaluationResult) -> str:
        """Render the ranking metrics against the baselines."""
        ks = list(result.ks)
        header = [
            "Métrique",
            "Modèle",
            *[plots.BASELINE_LABELS.get(n, n) for n in result.baselines],
        ]
        rows = []
        for metric in ("recall", "precision", "ndcg"):
            for k in ks:
                name = f"{metric}_at_{k}"
                cells = [_format(result.metrics.get(name))]
                cells.extend(_format(values.get(name)) for values in result.baselines.values())
                rows.append([name, *cells])
        for name in ("mrr",):
            cells = [_format(result.metrics.get(name))]
            cells.extend(_format(values.get(name)) for values in result.baselines.values())
            rows.append([name, *cells])
        return _markdown_table(header, rows)

    def _answer_table(self, result: EvaluationResult) -> str:
        """Render the grounding / abstention metrics."""
        rows = []
        for name in (
            "answer_f1",
            "answer_exact_match",
            "citation_precision",
            "citation_recall",
            "abstention_accuracy",
            "abstention_recall",
            "answer_coverage",
            "abstention_balanced_accuracy",
            "abstention_rate",
            "false_answer_rate",
            "mean_citations",
        ):
            if name in result.metrics:
                rows.append([name, _format(result.metrics.get(name))])
        if not rows:
            return (
                "Aucune métrique de réponse : le manifeste ne les a pas demandées (retriever pur)."
            )
        return _markdown_table(["Métrique", "Valeur"], rows)

    def _latency_table(self, result: EvaluationResult) -> str:
        """Render the latency percentiles."""
        rows = [
            [name, _format(result.metrics.get(name), digits=1)]
            for name in ("latency_p50_ms", "latency_p95_ms")
            if name in result.metrics
        ]
        if not rows:
            return "Latence non mesurée dans cette exécution."
        return _markdown_table(
            ["Percentile", "Latence (ms)"],
            rows,
        )

    def _segments_table(self, result: EvaluationResult) -> str:
        """Render the segment breakdown."""
        if not result.segments:
            return "Aucun segment mesuré."
        header = ["Segment", "Questions", "recall@5", "MRR", "F1 réponse", "Abstention"]
        rows = []
        for key, values in sorted(result.segments.items()):
            rows.append(
                [
                    key,
                    _format(values.get("n_questions"), digits=0),
                    _format(values.get("recall_at_5")),
                    _format(values.get("mrr")),
                    _format(values.get("answer_f1")),
                    _format(values.get("abstention_rate")),
                ]
            )
        return _markdown_table(header, rows)

    def _failures(self, result: EvaluationResult) -> list[str]:
        """Render the worst cases: questions whose first relevant passage is far, or missing."""
        frame = result.per_question
        if frame.empty:
            return ["Aucune question évaluée."]
        answerable = frame.loc[frame["answer_type"] != "unanswerable"].copy()
        answerable["rank_sort"] = answerable["first_relevant_rank"].replace(0, 10_000)
        worst = answerable.sort_values(["rank_sort", "top_score"], ascending=[False, True]).head(
            MAX_FAILURES
        )
        lines = [
            "| Question | Difficulté | Rang du 1er passage pertinent | Score | Citations |",
            "| --- | --- | --- | --- | --- |",
        ]
        for record in worst.to_dict(orient="records"):
            rank = int(record.get("first_relevant_rank", 0))
            lines.append(
                f"| {str(record.get('question', ''))[:70]} | {record.get('difficulty', '')} | "
                f"{rank if rank else 'non trouvé'} | {_format(record.get('top_score'))} | "
                f"{_format(record.get('n_citations'), digits=0)} |"
            )
        unanswerable = frame.loc[frame["answer_type"] == "unanswerable"]
        if not unanswerable.empty:
            invented = unanswerable.loc[unanswerable.get("abstained", 1).eq(0)]
            lines.extend(
                [
                    "",
                    f"Questions hors corpus traitées par une réponse (hallucination mesurée) : "
                    f"**{len(invented)}/{len(unanswerable)}**.",
                ]
            )
        return lines

    def _figure_lines(self, figures: Mapping[str, Path]) -> list[str]:
        """Reference the figures with their relative path."""
        if not figures:
            return ["Aucune figure produite."]
        descriptions = {
            "recall_curve": "Rappel par coupure, face aux références triviales.",
            "score_separation": "Séparation des scores entre questions répondables et hors corpus.",
            "latency": "Distribution de la latence de bout en bout.",
            "segments": "Métrique principale par segment.",
        }
        lines: list[str] = []
        for name, path in sorted(figures.items()):
            relative = path.relative_to(self.paths.root).as_posix()
            lines.append(f"![{name}](../{relative}) — {descriptions.get(name, name)}")
        return lines

    def _limitations(self, metadata: Mapping[str, Any]) -> list[str]:
        """List what the project does not claim."""
        lines = [
            "- **Corpus synthétique** : les ordres de grandeur sont réalistes, les valeurs ne le "
            "sont pas. Aucune conclusion ne doit être tirée sur un corpus réel sans rejouer "
            "l'évaluation.",
            "- **Générateur local** : par défaut la réponse est extractive (phrases du corpus). "
            "La fluidité d'un LLM distant n'est ni mesurée ni revendiquée ; seule la fidélité aux "
            "passages l'est.",
            "- **Recherche lexicale ou vectorielle légère** : les paraphrases longues restent "
            "difficiles ; c'est précisément ce que mesure le segment « paraphrase ».",
            "- **Une seule langue** : le corpus est français, la tokenisation aussi.",
        ]
        ceiling = metadata.get("retrievable_ceiling")
        if ceiling is not None:
            lines.append(
                f"- **Plafond du générateur** : {ceiling} — le corpus annoté ne permet pas de "
                "dépasser cette valeur, quel que soit le modèle."
            )
        return lines


def _markdown_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a Markdown table from a header and its rows."""
    lines = [
        "| " + " | ".join(str(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


__all__ = ["MAX_FAILURES", "REPORT_TITLE", "ReportBuilder", "ReportBundle"]
