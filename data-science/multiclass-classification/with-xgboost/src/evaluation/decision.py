"""Decision policy of the diagnosis: settings, cost matrix, minimum-cost decision, expert review.

A multiclass classifier does not decide anything by itself: ``argmax`` only answers *which class
is the most probable*, while the business asks *which action costs the least*. The two differ as
soon as errors do not cost the same — and here they never do: sending the wrong team costs a
second trip, while acknowledging a real failure as a false alarm lets the machine run until it
breaks.

This module owns that policy, so that the evaluator, the report and the predictor apply
**exactly** the same rules:

* :class:`DiagnosisSettings` — the ``diagnosis`` block of ``conf/config.yaml``, validated;
* :meth:`DiagnosisSettings.cost_matrix` — cost of deciding class *j* when the truth is *i*;
* :func:`expected_costs` / :func:`minimum_cost_decision` — the Bayes decision under that matrix;
* :func:`realised_costs` — what each decision actually cost, expert reviews included.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

#: ``diagnosis`` block shipped with the project (``conf/config.yaml``). It is used when no
#: configuration is given (tests, ad-hoc scripts); a configuration always takes precedence. The
#: prose notes of the block (``confusion_notes``) only serve the report and are not repeated here.
DEFAULT_DIAGNOSIS: dict[str, Any] = {
    "class_order": [
        "false_alarm",
        "tool_wear",
        "heat_dissipation",
        "power_failure",
        "overstrain",
        "random_failure",
    ],
    "no_failure_class": "false_alarm",
    "structural_classes": ["random_failure"],
    "alarm_code_column": "alarm_code",
    "alarm_code_rule": {
        "E101": "heat_dissipation",
        "E102": "heat_dissipation",
        "E201": "power_failure",
        "E202": "power_failure",
        "E301": "overstrain",
        "E302": "tool_wear",
        "E901": "false_alarm",
    },
    "team_by_class": {
        "false_alarm": "acquittement opérateur",
        "tool_wear": "outillage",
        "heat_dissipation": "thermique",
        "power_failure": "électrique",
        "overstrain": "mécanique",
        "random_failure": "expertise maintenance",
    },
    "costs": {
        "correct_intervention": 120.0,
        "correct_acknowledgement": 10.0,
        "wrong_team": 420.0,
        "missed_failure": 2500.0,
        "useless_dispatch": 150.0,
        "expert_review": 90.0,
    },
    "review_threshold": 0.5,
    "objectives": {
        "f1_macro_min": 0.6,
        "gain_vs_alarm_code_min": 0.2,
        "per_class_recall_min": 0.55,
        "missed_failure_rate_max": 0.05,
        "ece_max": 0.08,
        "cost_reduction_vs_alarm_code_min": 0.4,
        "latency_ms_per_1000_max": 50.0,
    },
}

#: Objectives evaluated when the configuration does not declare them.
DEFAULT_OBJECTIVES: dict[str, float] = {
    "f1_macro_min": 0.60,
    "gain_vs_alarm_code_min": 0.20,
    "per_class_recall_min": 0.55,
    "missed_failure_rate_max": 0.05,
    "ece_max": 0.08,
    "cost_reduction_vs_alarm_code_min": 0.40,
    "latency_ms_per_1000_max": 50.0,
}


@dataclass(frozen=True, slots=True)
class DiagnosisCosts:
    """Unit costs (EUR per alarm) from which the cost matrix is derived.

    Attributes:
        correct_intervention: Right team sent on a real failure (the nominal intervention).
        correct_acknowledgement: False alarm correctly acknowledged by the operator.
        wrong_team: Real failure, wrong specialty sent (second trip, longer downtime).
        missed_failure: Real failure acknowledged as a false alarm (the machine keeps running).
        useless_dispatch: A team sent on a false alarm.
        expert_review: Expert review of an uncertain alarm, on top of the correct intervention.
    """

    correct_intervention: float = 120.0
    correct_acknowledgement: float = 10.0
    wrong_team: float = 420.0
    missed_failure: float = 2500.0
    useless_dispatch: float = 150.0
    expert_review: float = 90.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> DiagnosisCosts:
        """Build the costs from a configuration mapping.

        Args:
            payload: ``diagnosis.costs`` node (``None`` keeps the defaults).

        Returns:
            The validated costs.

        Raises:
            ValueError: On an unknown key or a negative cost.
        """
        values = dict(payload or {})
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            msg = f"Unknown diagnosis cost(s) {unknown}. Allowed: {sorted(known)}"
            raise ValueError(msg)
        costs = {key: float(value) for key, value in values.items()}
        negative = {key: value for key, value in costs.items() if value < 0.0}
        if negative:
            msg = f"Diagnosis costs must be >= 0, got {negative}"
            raise ValueError(msg)
        return cls(**costs)


@dataclass(frozen=True, slots=True)
class DiagnosisSettings:
    """Validated ``diagnosis`` block: classes, routing rule, teams, costs, review threshold.

    Attributes:
        class_order: Canonical order of the classes (tables, figures, probability columns).
        no_failure_class: The "nothing to repair" class (``None`` when the task has none).
        structural_classes: Classes without any observable signal (excluded from recall goals).
        alarm_code_column: Raw column holding the current routing signal (``None`` if absent).
        alarm_code_rule: Current routing rule, code -> class (the business reference).
        team_by_class: Team dispatched for each diagnosed class.
        costs: Unit costs of the decisions.
        review_threshold: Confidence under which an alarm is sent to an expert.
        objectives: Contractual objectives evaluated by the evaluator.
    """

    class_order: tuple[str, ...]
    no_failure_class: str | None
    structural_classes: tuple[str, ...]
    alarm_code_column: str | None
    alarm_code_rule: Mapping[str, str]
    team_by_class: Mapping[str, str]
    costs: DiagnosisCosts
    review_threshold: float
    objectives: Mapping[str, float]

    @classmethod
    def resolve(
        cls,
        config: Mapping[str, Any] | None = None,
        *,
        classes: Sequence[Any] | None = None,
    ) -> DiagnosisSettings:
        """Build the settings from the root configuration (or the shipped defaults).

        Args:
            config: Root configuration mapping; its ``diagnosis`` block wins over
                :data:`DEFAULT_DIAGNOSIS`.
            classes: Classes known by the fitted model, checked against ``class_order``.

        Returns:
            The validated settings.

        Raises:
            ValueError: When the block is inconsistent (unknown class, bad threshold, ...).
        """
        node = dict((config or {}).get("diagnosis") or DEFAULT_DIAGNOSIS or {})
        model_classes = [str(value) for value in classes] if classes is not None else []
        class_order = tuple(
            str(value) for value in (node.get("class_order") or sorted(model_classes))
        )
        if not class_order:
            msg = "diagnosis.class_order is empty and no model classes were given"
            raise ValueError(msg)
        if len(set(class_order)) != len(class_order):
            msg = f"diagnosis.class_order contains duplicates: {list(class_order)}"
            raise ValueError(msg)
        known = set(class_order)
        unexpected = sorted(set(model_classes) - known)
        if unexpected:
            msg = f"The model predicts class(es) {unexpected} absent from diagnosis.class_order"
            raise ValueError(msg)

        no_failure = node.get("no_failure_class")
        if no_failure is not None and str(no_failure) not in known:
            msg = f"diagnosis.no_failure_class '{no_failure}' is not in class_order"
            raise ValueError(msg)
        structural = tuple(str(value) for value in node.get("structural_classes") or ())
        _check_subset("diagnosis.structural_classes", structural, known)

        rule = {
            str(code): str(label) for code, label in dict(node.get("alarm_code_rule") or {}).items()
        }
        _check_subset("diagnosis.alarm_code_rule (targets)", rule.values(), known)
        teams = {
            str(label): str(team) for label, team in dict(node.get("team_by_class") or {}).items()
        }
        _check_subset("diagnosis.team_by_class (keys)", teams, known)

        threshold = float(node.get("review_threshold", 0.5))
        if not 0.0 <= threshold < 1.0:
            msg = f"diagnosis.review_threshold must be in [0, 1), got {threshold}"
            raise ValueError(msg)

        objectives = {
            **DEFAULT_OBJECTIVES,
            **{str(k): float(v) for k, v in dict(node.get("objectives") or {}).items()},
        }
        column = node.get("alarm_code_column")
        return cls(
            class_order=class_order,
            no_failure_class=None if no_failure is None else str(no_failure),
            structural_classes=structural,
            alarm_code_column=None if column is None else str(column),
            alarm_code_rule=rule,
            team_by_class=teams,
            costs=DiagnosisCosts.from_mapping(node.get("costs")),
            review_threshold=threshold,
            objectives=objectives,
        )

    # ------------------------------------------------------------------ helpers ---------
    def ordered(self, labels: Sequence[Any]) -> list[str]:
        """Sort labels in the canonical order (unknown labels last, alphabetically).

        Args:
            labels: Labels to sort.

        Returns:
            The ordered labels, as strings.
        """
        rank = {label: position for position, label in enumerate(self.class_order)}
        return sorted(
            {str(label) for label in labels}, key=lambda label: (rank.get(label, len(rank)), label)
        )

    def predictable_classes(self, labels: Sequence[Any]) -> list[str]:
        """Return the labels that carry an observable signal (recall objectives apply)."""
        return [
            label for label in self.ordered(labels) if label not in set(self.structural_classes)
        ]

    def team_for(self, label: Any) -> str:
        """Return the team dispatched for a diagnosed class."""
        return self.team_by_class.get(str(label), "expertise maintenance")

    def cost_matrix(self, labels: Sequence[Any]) -> np.ndarray:
        """Return the cost matrix: ``C[i, j]`` = cost of deciding ``labels[j]`` on ``labels[i]``.

        Args:
            labels: Class labels, in the order of the probability columns.

        Returns:
            The ``(K, K)`` cost matrix, in EUR per alarm.
        """
        names = [str(label) for label in labels]
        costs = self.costs
        matrix = np.zeros((len(names), len(names)), dtype="float64")
        for row, truth in enumerate(names):
            for column, decided in enumerate(names):
                if truth == self.no_failure_class:
                    matrix[row, column] = (
                        costs.correct_acknowledgement
                        if decided == truth
                        else costs.useless_dispatch
                    )
                elif decided == truth:
                    matrix[row, column] = costs.correct_intervention
                elif decided == self.no_failure_class:
                    matrix[row, column] = costs.missed_failure
                else:
                    matrix[row, column] = costs.wrong_team
        return matrix

    def route_alarm_codes(self, codes: Sequence[Any] | np.ndarray) -> np.ndarray:
        """Apply the current routing rule to alarm codes (unknown codes -> no-failure class).

        Args:
            codes: Alarm codes, one per row.

        Returns:
            The class chosen by the current rule for every row.
        """
        fallback = self.no_failure_class or self.class_order[0]
        return np.asarray(
            [self.alarm_code_rule.get(str(code), fallback) for code in codes], dtype=object
        )


def expected_costs(
    probabilities: np.ndarray, labels: Sequence[Any], settings: DiagnosisSettings
) -> np.ndarray:
    """Expected cost of every possible decision, row by row.

    ``E[cost | decide j] = sum_i p_i * C[i, j]``: the probability of each truth weighted by what it
    would cost to answer ``j``.

    Args:
        probabilities: ``(n, K)`` class probabilities, columns aligned with ``labels``.
        labels: Class labels of the probability columns.
        settings: Decision settings.

    Returns:
        The ``(n, K)`` expected costs.
    """
    return np.asarray(probabilities, dtype="float64") @ settings.cost_matrix(labels)


def minimum_cost_decision(
    probabilities: np.ndarray, labels: Sequence[Any], settings: DiagnosisSettings
) -> np.ndarray:
    """Return the decision minimising the expected cost (the Bayes decision under ``C``).

    Args:
        probabilities: ``(n, K)`` class probabilities, columns aligned with ``labels``.
        labels: Class labels of the probability columns.
        settings: Decision settings.

    Returns:
        The decided label of every row.
    """
    names = np.asarray([str(label) for label in labels], dtype=object)
    return names[expected_costs(probabilities, labels, settings).argmin(axis=1)]


def realised_costs(
    truth: Sequence[Any] | np.ndarray,
    decided: Sequence[Any] | np.ndarray,
    labels: Sequence[Any],
    settings: DiagnosisSettings,
    *,
    reviewed: np.ndarray | None = None,
) -> np.ndarray:
    """Cost actually incurred by each decision.

    A reviewed alarm costs the expert review plus the correct action (the expert finds the right
    cause); every other alarm costs ``C[truth, decided]``.

    Args:
        truth: True labels.
        decided: Decided labels.
        labels: Label universe of the cost matrix.
        settings: Decision settings.
        reviewed: Boolean mask of the alarms sent to an expert (``None``: none).

    Returns:
        The cost of every row, in EUR.
    """
    names = [str(label) for label in labels]
    index = {label: position for position, label in enumerate(names)}
    matrix = settings.cost_matrix(names)
    true_index = np.asarray([index[str(value)] for value in truth], dtype="int64")
    decided_index = np.asarray([index[str(value)] for value in decided], dtype="int64")
    costs = matrix[true_index, decided_index]
    if reviewed is not None:
        mask = np.asarray(reviewed, dtype=bool)
        costs = np.where(mask, matrix[true_index, true_index] + settings.costs.expert_review, costs)
    return costs


def _check_subset(name: str, values: Any, known: set[str]) -> None:
    """Raise when ``values`` references a class absent from ``known``."""
    unknown = sorted({str(value) for value in values} - known)
    if unknown:
        msg = f"{name} references unknown class(es) {unknown}; known: {sorted(known)}"
        raise ValueError(msg)
