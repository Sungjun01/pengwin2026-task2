"""Airtight, no-GPU verification of the Task2 click postprocessing.

Uses the GT label as a stand-in for the v19 prediction. Because each click is one
seed inside its GT fragment, relabel_from_clicks(GT, clicks) must reproduce GT's
fragments almost exactly. Also checks the axis self-correction by feeding the same
clicks with their coordinates reversed (simulating a GC (x,y,z) re-serialization).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.click_guided_postproc import (  # noqa: E402
    ANATOMY_RANGES,
    parse_clicks,
    relabel_from_clicks,
)

TRAINSET = Path("/home/schoaq/taehee/PENGWIN_Challenge_datasets/Trainset")
CLICKS = Path("/home/schoaq/taehee/PENGWIN_Challenge_datasets/task2/PENGWIN26_task2_train_clicks")
STRATEGIES = ["center_of_mass", "euclidean_distance_transform", "uniformly_sampled", "boundary_internal_margin"]


def find_label(case: str) -> Path | None:
    for part in sorted(TRAINSET.glob("PENGWIN26_task1_2_train_part*")):
        p = part / case / "label.mha"
        if p.is_file():
            return p
    return None


def instance_f1(gt: np.ndarray, pred: np.ndarray, lo: int, hi: int, iou_thr: float = 0.5):
    gt_ids = [i for i in np.unique(gt) if lo <= i <= hi]
    pr_ids = [i for i in np.unique(pred) if lo <= i <= hi]
    if not gt_ids and not pr_ids:
        return None
    gt_masks = {i: (gt == i) for i in gt_ids}
    pr_masks = {i: (pred == i) for i in pr_ids}
    matched_gt = 0
    used_pr = set()
    for gi, gm in gt_masks.items():
        best, best_iou = None, 0.0
        for pi, pm in pr_masks.items():
            if pi in used_pr:
                continue
            inter = np.logical_and(gm, pm).sum()
            if inter == 0:
                continue
            iou = inter / np.logical_or(gm, pm).sum()
            if iou > best_iou:
                best, best_iou = pi, iou
        if best is not None and best_iou >= iou_thr:
            matched_gt += 1
            used_pr.add(best)
    recall = matched_gt / len(gt_ids) if gt_ids else 0.0
    precision = matched_gt / len(pr_ids) if pr_ids else 0.0
    f1 = 0.0 if (recall + precision) == 0 else 2 * recall * precision / (recall + precision)
    return dict(gt=len(gt_ids), pred=len(pr_ids), matched=matched_gt, recall=recall, precision=precision, f1=f1)


def coord_landing_rate(gt: np.ndarray, clicks) -> tuple[int, int]:
    """How many clicks land on a nonzero voxel of their named anatomy at (z,y,x)?"""
    ok = 0
    for c in clicks:
        z, y, x = (int(round(v)) for v in c.point)
        z = np.clip(z, 0, gt.shape[0] - 1); y = np.clip(y, 0, gt.shape[1] - 1); x = np.clip(x, 0, gt.shape[2] - 1)
        lbl = int(gt[z, y, x])
        if c.anatomy in ANATOMY_RANGES:
            lo, hi = ANATOMY_RANGES[c.anatomy]
            ok += int(lo <= lbl <= hi)
        else:
            ok += int(lbl > 0)
    return ok, len(clicks)


def main(cases: list[str]):
    print(f"{'case':>5} {'strat':>26} {'land':>7} {'F1':>6} {'rec':>5} {'prec':>5} {'gt/pr':>7}  axisflip-F1")
    for case in cases:
        lab = find_label(case)
        if lab is None:
            print(f"{case:>5}  (no label.mha)"); continue
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(lab)))
        for strat in STRATEGIES:
            cj = CLICKS / strat / case / "peripelvic-fragment-clicks.json"
            if not cj.is_file():
                continue
            clicks = parse_clicks(cj)
            ok, tot = coord_landing_rate(gt, clicks)
            out = relabel_from_clicks(gt, clicks)
            f1s = [instance_f1(gt, out, lo, hi) for _, (lo, hi) in ANATOMY_RANGES.items()]
            f1s = [r for r in f1s if r]
            f1 = np.mean([r["f1"] for r in f1s]) if f1s else float("nan")
            rec = np.mean([r["recall"] for r in f1s]) if f1s else float("nan")
            prec = np.mean([r["precision"] for r in f1s]) if f1s else float("nan")
            gtn = sum(r["gt"] for r in f1s); prn = sum(r["pred"] for r in f1s)

            # axis-flip robustness: reverse every stored coord, parser must self-correct
            flipped = [type(c)(c.anatomy, tuple(c.point[::-1]), c.name, c.fragment_id) for c in clicks]
            out_f = relabel_from_clicks(gt, flipped)
            f1f = [instance_f1(gt, out_f, lo, hi) for _, (lo, hi) in ANATOMY_RANGES.items()]
            f1f = [r for r in f1f if r]
            f1_flip = np.mean([r["f1"] for r in f1f]) if f1f else float("nan")

            print(f"{case:>5} {strat:>26} {ok:>3}/{tot:<3} {f1:>6.3f} {rec:>5.2f} {prec:>5.2f} {gtn:>3}/{prn:<3}  {f1_flip:>6.3f}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["099", "186", "157", "119", "092", "097"]
    main(args)
