"""Task-utility matching and scoring."""

from buse_uav.utility.matching import BoxMatch, fuse_reference, hungarian_match
from buse_uav.utility.task_utility import (
    CandidateChoice,
    CandidateEvaluation,
    choose_best_candidate,
    evaluate_candidate,
    identity_evaluation,
    should_early_stop,
)

__all__ = [
    "BoxMatch",
    "CandidateChoice",
    "CandidateEvaluation",
    "choose_best_candidate",
    "evaluate_candidate",
    "fuse_reference",
    "hungarian_match",
    "identity_evaluation",
    "should_early_stop",
]
