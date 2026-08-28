"""PENGWIN multi-method ranking scheme.

Per metric, rank submissions independently (lower-is-better metrics use ascending
sort, higher-is-better use descending). Final rank = mean across all 8 metric
ranks (3 anatomy + 5 fracture). Tie-breaker: lower execution time wins.

PENGWIN 2026 metric list (challenge spec).
    Anatomy : IoU, HD95, ASSD                      (3)
    Fracture: IoU, HD95, ASSD, LDSC, RQ            (5)
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from evaluation.pengwin_eval import evaluate_submission


METRICS_HIGHER_BETTER = {
    "IoU_F": True,  "HD95_F_mm": False, "ASSD_F_mm": False,
    "LDSC_F": True, "RQ_F": True,
    "IoU_A": True,  "HD95_A_mm": False, "ASSD_A_mm": False,
}


def rank_dict(scores: dict[str, float], higher_better: bool) -> dict[str, int]:
    """Rank submissions 1..N by their score. Ties share the lowest rank."""
    items = sorted(
        scores.items(),
        key=lambda kv: kv[1],
        reverse=higher_better,
    )
    ranks: dict[str, int] = {}
    prev_score = None
    prev_rank = 0
    for i, (name, score) in enumerate(items, start=1):
        if prev_score is not None and score == prev_score:
            ranks[name] = prev_rank
        else:
            ranks[name] = i
            prev_rank = i
            prev_score = score
    return ranks


def rank_submissions(
    submissions: dict[str, Path],
    gt_dir: Path | str,
    execution_times: dict[str, float] | None = None,
    output_path: Path | str | None = None,
    matching_method: str = "hungarian",
) -> dict:
    """Apply PENGWIN ranking scheme across multiple submissions."""
    submission_aggregates: dict[str, dict[str, float]] = {}
    for name, pred_dir in submissions.items():
        r = evaluate_submission(pred_dir, gt_dir, matching_method=matching_method)
        submission_aggregates[name] = r["aggregate"]

    per_metric_ranks: dict[str, dict[str, int]] = {}
    for metric, higher_better in METRICS_HIGHER_BETTER.items():
        scores = {
            name: submission_aggregates[name].get(f"{metric}_mean", float("nan"))
            for name in submissions
        }
        per_metric_ranks[metric] = rank_dict(scores, higher_better=higher_better)

    final_rank: dict[str, float] = {
        name: mean(per_metric_ranks[m][name] for m in METRICS_HIGHER_BETTER)
        for name in submissions
    }

    def sort_key(name: str) -> tuple[float, float]:
        primary = final_rank[name]
        tb = execution_times.get(name, 0.0) if execution_times else 0.0
        return (primary, tb)

    ordering = sorted(submissions.keys(), key=sort_key)

    result = {
        "submissions": list(submissions.keys()),
        "submission_aggregates": submission_aggregates,
        "per_metric_ranks": per_metric_ranks,
        "final_rank": final_rank,
        "ordering": ordering,
        "execution_times": execution_times or {},
    }
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    return result


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Rank multiple PENGWIN submissions by the 8-metric scheme (3 anatomy + 5 fracture).",
    )
    parser.add_argument(
        "-s", "--submission", action="append", required=True,
        help="Submission spec 'name:path' (repeat for multiple methods)",
    )
    parser.add_argument("-g", "--gt_dir", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument(
        "--times", action="append", default=[],
        help="Execution-time spec 'name:seconds' (repeat per method)",
    )
    parser.add_argument("--matching", choices=["hungarian", "greedy"],
                        default="hungarian")
    args = parser.parse_args()

    submissions: dict[str, Path] = {}
    for spec in args.submission:
        name, path = spec.split(":", 1)
        submissions[name] = Path(path)
    times: dict[str, float] = {}
    for spec in args.times:
        name, secs = spec.split(":", 1)
        times[name] = float(secs)

    result = rank_submissions(
        submissions=submissions,
        gt_dir=args.gt_dir,
        execution_times=times or None,
        output_path=args.output,
        matching_method=args.matching,
    )
    print("Ranking:")
    for i, name in enumerate(result["ordering"], start=1):
        print(f"  {i}. {name}  (final rank score: {result['final_rank'][name]:.2f})")


if __name__ == "__main__":
    _main()
