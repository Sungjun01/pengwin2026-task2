"""PENGWIN Task 1 evaluation package.

Public API:
    evaluate_case        - Six metrics for a single (pred, gt) file pair.
    evaluate_submission  - Per-case + aggregate metrics for a submission dir.
    rank_submissions     - PENGWIN ranking scheme across multiple methods.
    rank_dict            - Independent per-metric ranking helper.
"""
from evaluation.pengwin_eval import evaluate_case, evaluate_submission
from evaluation.ranking import rank_submissions, rank_dict

__all__ = [
    "evaluate_case", "evaluate_submission",
    "rank_submissions", "rank_dict",
]
