"""Training stability analysis — epoch-별 per-class Dice 시계열로 LH/RH balance 정량화.

motivation. progress_report_21 의 진단 — Task A (D007 vanilla cascade) 의 EMA mean
Dice 가 안정적으로 보이지만 실제로는 epoch 별 anti-correlated oscillation (한 epoch
LH 우세, 다음 RH 우세) 이 그대로 지속. EMA 평균이 instability 를 위장한다. paper §5
의 진짜 contribution 으로 가려면 epoch-별 balance stability 를 정량화해야 한다.

지표.
    balance_epoch_ratio : LH 와 RH 가 *둘 다* dice > threshold 인 epoch 비율
                          (threshold default 0.3, 학습 후반 window).
                          1.0 = 모든 epoch 에서 balance, 0.0 = 항상 한쪽 collapse.
    ratio_std           : min(LH, RH) / max(LH, RH) 의 epoch-내 표준편차.
                          0.0 = 매 epoch 비율 일정, 큰 값 = oscillation.
    mode_flip_frequency : 연속 epoch 사이 (LH > RH ↔ RH > LH) swap 비율.
                          0.0 = 한 방향으로 안정, 0.5 = 매 epoch swap (가장 unstable).
    mean_lh, mean_rh    : window 내 평균 dice (참고용).

CLI.
    python -m evaluation.training_stability \\
        --log <training_log_*.txt> \\
        --window 500 1000 \\
        --threshold 0.3 \\
        --output <out.json>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np


EPOCH_RE = re.compile(r"^[0-9\- :.]+:\s*Epoch (\d+)\s*$")
PSEUDO_DICE_RE = re.compile(r"Pseudo dice \[(.*?)\]")
FLOAT_RE = re.compile(
    r"np\.float32\((-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)\)|(-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?)"
)


def parse_training_log(log_path: Path) -> dict[int, list[float]]:
    """Return {epoch_int: [dice_class_0, dice_class_1, ...]}.

    nnUNet training log format:
        2026-05-17 ...: Epoch 122
        2026-05-17 ...: Pseudo dice [np.float32(0.98), np.float32(0.0), np.float32(0.6)]
    """
    series: dict[int, list[float]] = {}
    current_epoch: Optional[int] = None
    with open(log_path) as f:
        for line in f:
            m_ep = EPOCH_RE.match(line.rstrip())
            if m_ep:
                current_epoch = int(m_ep.group(1))
                continue
            m_pd = PSEUDO_DICE_RE.search(line)
            if m_pd and current_epoch is not None:
                inner = m_pd.group(1)
                vals = []
                for nm in FLOAT_RE.finditer(inner):
                    g = nm.group(1) if nm.group(1) is not None else nm.group(2)
                    vals.append(float(g))
                if vals:
                    series[current_epoch] = vals
                current_epoch = None
    return series


def compute_stability(
    series: dict[int, list[float]],
    lh_idx: int = 1,
    rh_idx: int = 2,
    window: tuple[int, int] = (500, 1000),
    threshold: float = 0.3,
) -> dict[str, float]:
    """Compute balance stability indicators within an epoch window."""
    epochs = sorted(e for e in series.keys() if window[0] <= e <= window[1])
    if len(epochs) < 2:
        return {
            "balance_epoch_ratio": float("nan"),
            "ratio_std": float("nan"),
            "mode_flip_frequency": float("nan"),
            "mean_lh": float("nan"),
            "mean_rh": float("nan"),
            "n_epochs_in_window": len(epochs),
        }

    lh_vals = np.array([series[e][lh_idx] for e in epochs])
    rh_vals = np.array([series[e][rh_idx] for e in epochs])

    balance_mask = (lh_vals > threshold) & (rh_vals > threshold)
    balance_epoch_ratio = float(balance_mask.mean())

    max_pair = np.maximum(lh_vals, rh_vals)
    min_pair = np.minimum(lh_vals, rh_vals)
    safe_max = np.where(max_pair > 1e-8, max_pair, 1e-8)
    ratio = min_pair / safe_max
    ratio_std = float(ratio.std(ddof=0))

    lh_wins = lh_vals > rh_vals
    flips = int(np.sum(lh_wins[1:] != lh_wins[:-1]))
    mode_flip_frequency = float(flips / max(1, len(epochs) - 1))

    return {
        "balance_epoch_ratio": balance_epoch_ratio,
        "ratio_std": ratio_std,
        "mode_flip_frequency": mode_flip_frequency,
        "mean_lh": float(lh_vals.mean()),
        "mean_rh": float(rh_vals.mean()),
        "n_epochs_in_window": int(len(epochs)),
    }


def export_series_csv(
    series: dict[int, list[float]],
    out_csv: Path,
    class_names: list[str],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("epoch," + ",".join(class_names) + "\n")
        for e in sorted(series.keys()):
            vals = series[e]
            row = [str(e)] + [f"{v:.4f}" for v in vals]
            f.write(",".join(row) + "\n")


def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Training stability analysis from nnUNet training_log.",
    )
    parser.add_argument("-l", "--log", required=True, type=Path)
    parser.add_argument("--lh_idx", type=int, default=1)
    parser.add_argument("--rh_idx", type=int, default=2)
    parser.add_argument("--window", nargs=2, type=int, default=[500, 1000])
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument(
        "--class_names",
        nargs="+",
        default=["sacrum", "leftHip", "rightHip"],
    )
    args = parser.parse_args()

    if not args.log.exists():
        sys.exit(f"log not found: {args.log}")

    series = parse_training_log(args.log)
    if not series:
        sys.exit("no Pseudo dice entries found in log")

    stats = compute_stability(
        series,
        lh_idx=args.lh_idx,
        rh_idx=args.rh_idx,
        window=tuple(args.window),
        threshold=args.threshold,
    )

    result = {
        "log_path": str(args.log),
        "window": args.window,
        "threshold": args.threshold,
        "total_epochs_in_log": len(series),
        "stats": stats,
    }

    print(f"[stability] log              = {args.log.name}")
    print(f"[stability] total epochs     = {len(series)}")
    print(f"[stability] window           = {args.window}")
    print(f"[stability] n in window      = {stats['n_epochs_in_window']}")
    print(
        f"[stability] balance ratio    = {stats['balance_epoch_ratio']:.3f}  "
        f"(둘 다 dice > {args.threshold} 인 epoch 비율)"
    )
    print(
        f"[stability] ratio std        = {stats['ratio_std']:.3f}  "
        f"(min/max LH-RH 비의 표준편차)"
    )
    print(
        f"[stability] mode flip freq   = {stats['mode_flip_frequency']:.3f}  "
        f"(연속 epoch 사이 우세 swap 비율, 0.5 = chaotic)"
    )
    print(
        f"[stability] mean LH / RH     = {stats['mean_lh']:.3f} / {stats['mean_rh']:.3f}"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[stability] wrote {args.output}")

    if args.csv is not None:
        export_series_csv(series, args.csv, args.class_names)
        print(f"[stability] wrote {args.csv}")


if __name__ == "__main__":
    _main()
