"""OOF PPI Radar Discriminator — 실패 모드 자동 분류 + Threat Report.

각 케이스를 5 신호로 분류:
  🟢 CLEAR     : n_pred == gtN, HD95 낮음 (정상)
  🔴 TARGET    : 골절(gtN>1) 정확 분리 (RQ≥0.8)
  ⚠️  CLUTTER   : Dice 높은데 HD95 튐 (배경 false-positive 섬)
  ❌ JAM-OVER  : n_pred ≫ gtN (watershed 과분할)
  ❌ JAM-UNDER : n_pred < gtN (d013 양측/조각 붕괴)

⚠️ tmux 백그라운드 로그 친화 — \\r 애니메이션 없음, 정적 한 줄씩.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Iterable

HD95_CLEAR = 5.0
HD95_CLUTTER = 20.0
DICE_HIGH = 0.95
RQ_TARGET = 0.80
TINY_ISLAND_VOLUME_MM3 = 500.0


@dataclass(frozen=True)
class CaseTelemetry:
    case: str
    gt_count: int
    pred_count: int
    anatomy: str = "femur"
    cc_count: int | None = None
    ws_count: int | None = None
    rq: float = 0.0
    hd95: float = 0.0
    dice: float | None = None
    assd: float | None = None
    local_dice: float | None = None
    component_volumes_mm3: tuple[float, ...] = ()
    decoder_mode: str = ""
    env_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SignalResult:
    code: str
    label: str
    severity: int
    reason: str


@dataclass(frozen=True)
class FixHint:
    signal: str
    action: str
    source_area: str


@dataclass(frozen=True)
class RadarTransition:
    case: str
    before_signal: str
    after_signal: str
    transition: str
    delta_rq: float
    delta_hd95: float
    delta_dice: float | None


def normalize_telemetry(row: dict | CaseTelemetry) -> CaseTelemetry:
    """Normalize legacy loose OOF rows into the v2 telemetry schema."""
    if isinstance(row, CaseTelemetry):
        return row
    gt = int(row.get("gt_count", row.get("gtN", row.get("gt", 0))))
    pred = int(row.get("pred_count", row.get("n_pred", row.get("n_final", 0))))
    cc = row.get("cc_count", row.get("n_cc"))
    ws = row.get("ws_count", row.get("n_ws"))
    return CaseTelemetry(
        case=str(row.get("case", row.get("case_id", ""))),
        anatomy=str(row.get("anatomy", "femur")),
        gt_count=gt,
        pred_count=pred,
        cc_count=int(cc) if cc is not None else pred,
        ws_count=int(ws) if ws is not None else pred,
        rq=float(row.get("rq", row.get("RQ_F", row.get("ins_f1", 0.0)))),
        hd95=float(row.get("hd95", row.get("HD95_F_mm", row.get("fracture_hd95", 0.0)))),
        dice=(
            float(row["dice"])
            if "dice" in row
            else float(row["IoU_F"]) if "IoU_F" in row
            else float(row["fracture_dice"]) if "fracture_dice" in row
            else None
        ),
        assd=float(row["assd"]) if "assd" in row else float(row["ASSD_F_mm"]) if "ASSD_F_mm" in row else None,
        local_dice=float(row["local_dice"]) if "local_dice" in row else float(row["LDSC_F"]) if "LDSC_F" in row else None,
        component_volumes_mm3=tuple(float(v) for v in row.get("component_volumes_mm3", ())),
        decoder_mode=str(row.get("decoder_mode", "")),
        env_flags=tuple(str(v) for v in row.get("env_flags", ())),
    )


def _tiny_island_count(tel: CaseTelemetry) -> int:
    return sum(1 for v in tel.component_volumes_mm3 if 0 < v < TINY_ISLAND_VOLUME_MM3)


def classify_case(row: dict | CaseTelemetry) -> SignalResult:
    tel = normalize_telemetry(row)
    cc = tel.cc_count if tel.cc_count is not None else tel.pred_count
    ws = tel.ws_count if tel.ws_count is not None else tel.pred_count
    dice = tel.dice

    if tel.pred_count < tel.gt_count:
        if cc >= tel.gt_count or ws >= tel.gt_count:
            return SignalResult(
                "JAM_UNDER_DECODE",
                f"❌ [JAM-UNDER-DECODE] decode/merge collapsed ({tel.gt_count}→{tel.pred_count})",
                5,
                "mask/decode had enough pieces but final instances collapsed",
            )
        return SignalResult(
            "JAM_UNDER_MODEL",
            f"❌ [JAM-UNDER-MODEL] model mask already fused ({tel.gt_count}→{tel.pred_count})",
            5,
            "foreground lacks separable components before decode",
        )
    if tel.pred_count > tel.gt_count + 1:
        if cc <= tel.gt_count + 1 and ws > tel.gt_count + 1:
            return SignalResult(
                "JAM_OVER_DECODE",
                f"❌ [JAM-OVER-DECODE] watershed/decode split ({tel.gt_count}→{tel.pred_count})",
                4,
                "binary topology is plausible but decoder oversplit it",
            )
        return SignalResult(
            "JAM_OVER_MASK",
            f"❌ [JAM-OVER-MASK] foreground fragmented ({tel.gt_count}→{tel.pred_count})",
            4,
            "model foreground already has too many components",
        )
    if (dice is not None and dice >= DICE_HIGH and tel.hd95 >= HD95_CLUTTER) or _tiny_island_count(tel) > 0:
        return SignalResult(
            "CLUTTER",
            f"⚠️ [CLUTTER] high overlap but HD95/noise spike (HD95 {tel.hd95:.1f})",
            3,
            "small far-away islands or boundary dust likely dominate surface distance",
        )
    if tel.gt_count > 1 and tel.pred_count == tel.gt_count and tel.rq >= RQ_TARGET:
        return SignalResult(
            "TARGET",
            f"🔴 [TARGET] fracture acquired (RQ {tel.rq:.2f})",
            1,
            "true fracture topology is separated with high instance quality",
        )
    if tel.pred_count == tel.gt_count and tel.hd95 <= HD95_CLEAR:
        return SignalResult(
            "CLEAR",
            "🟢 [CLEAR] secured airspace",
            0,
            "topology and boundary are within safe limits",
        )
    return SignalResult(
        "WATCH",
        f"🟡 [WATCH] borderline signal (RQ {tel.rq:.2f}, HD95 {tel.hd95:.1f})",
        2,
        "not catastrophic but should be inspected in candidate diffs",
    )


def classify_signal(gtN: int, n_pred: int, rq: float, hd95: float, dice: float = None) -> tuple[str, str]:
    """Legacy wrapper: returns the old coarse code strings."""
    result = classify_case(
        CaseTelemetry(
            case="",
            gt_count=gtN,
            pred_count=n_pred,
            cc_count=n_pred,
            ws_count=n_pred,
            rq=rq,
            hd95=hd95,
            dice=dice,
        )
    )
    coarse = {
        "JAM_UNDER_DECODE": "JAM-UNDER",
        "JAM_UNDER_MODEL": "JAM-UNDER",
        "JAM_OVER_DECODE": "JAM-OVER",
        "JAM_OVER_MASK": "JAM-OVER",
    }.get(result.code, result.code)
    return coarse, result.label


def fix_hint_for_signal(signal: str) -> FixHint:
    hints = {
        "JAM_OVER_DECODE": FixHint(
            signal,
            "Tune watershed peaks or route candidate through affinity/MWS decode.",
            "scripts/inference_v7_d015.py, scripts/mutex_watershed_decode.py",
        ),
        "JAM_OVER_MASK": FixHint(
            signal,
            "Improve foreground/boundary model; post-processing alone is unlikely to rescue it.",
            "scripts/trainers/, Dataset004/D016 model training",
        ),
        "JAM_UNDER_DECODE": FixHint(
            signal,
            "Disable aggressive merge; inspect PIVOT-Assembly chunk/global-cc gates.",
            "scripts/pivot_assembly.py, scripts/inference_v7_d015.py",
        ),
        "JAM_UNDER_MODEL": FixHint(
            signal,
            "Train seam/affinity model; the model mask is fused before decode.",
            "scripts/trainers/nnUNetTrainerFemurAffinity.py, scripts/affinity_loss.py",
        ),
        "CLUTTER": FixHint(
            signal,
            "Apply connected-component dust filtering or topology-locked trim.",
            "scripts/surface_refine.py, scripts/femur_boundary_refine.py",
        ),
        "WATCH": FixHint(
            signal,
            "Inspect per-case metric deltas before changing decode rules.",
            "scripts/oof_femur_8metric.py",
        ),
    }
    return hints.get(signal, FixHint(signal, "No action required.", "n/a"))


def _threat_codes() -> set[str]:
    return {"JAM_OVER_DECODE", "JAM_OVER_MASK", "JAM_UNDER_DECODE", "JAM_UNDER_MODEL", "CLUTTER"}


def print_radar_report(rows: list[dict], mode: str = "OOF_SCAN") -> None:
    """rows: legacy dicts or CaseTelemetry. Static, CI/tmux-friendly output."""
    W = 70
    print("=" * W)
    print(f"📡 PENGWIN-2026 OOF PPI RADAR | MODE: {mode}")
    print("=" * W)
    counts: dict[str, int] = {}
    threats: list[tuple[str, CaseTelemetry, SignalResult]] = []
    telemetry = [normalize_telemetry(r) for r in rows]
    for r in sorted(telemetry, key=lambda x: x.case):
        result = classify_case(r)
        code, label = result.code, result.label
        counts[code] = counts.get(code, 0) + 1
        print(
            f"📦 case_{str(r.case):>4} | {label:<48} | "
            f"RQ {r.rq:.2f} | HD95 {r.hd95:5.1f} | "
            f"cc/ws/final {r.cc_count}/{r.ws_count}/{r.pred_count}"
        )
        if code in _threat_codes():
            threats.append((code, r, result))
    print("-" * W)
    print("📊 SIGNAL TALLY: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("-" * W)
    print("🚨 UNRESOLVED THREAT REPORT")
    if not threats:
        print("   (없음 — 전 케이스 CLEAR/TARGET, 녹색 영공 확보)")
    else:
        for code, r, result in sorted(threats, key=lambda item: (-item[2].severity, item[1].case)):
            hint = fix_hint_for_signal(code)
            print(f"   [!] case_{r.case} [{code}] → {hint.action}")
            print(f"       source: {hint.source_area}")
            print(f"       reason: {result.reason}")
    print("=" * W)


def compare_radar_reports(
    before: Iterable[dict | CaseTelemetry],
    after: Iterable[dict | CaseTelemetry],
) -> list[RadarTransition]:
    before_by_case = {normalize_telemetry(row).case: normalize_telemetry(row) for row in before}
    after_by_case = {normalize_telemetry(row).case: normalize_telemetry(row) for row in after}
    transitions: list[RadarTransition] = []
    for case in sorted(before_by_case.keys() & after_by_case.keys()):
        b = before_by_case[case]
        a = after_by_case[case]
        b_sig = classify_case(b)
        a_sig = classify_case(a)
        delta_rq = a.rq - b.rq
        delta_hd95 = a.hd95 - b.hd95
        delta_dice = None if a.dice is None or b.dice is None else a.dice - b.dice
        if b_sig.code in _threat_codes() and a_sig.code not in _threat_codes():
            transition = "KILL"
        elif b_sig.code not in _threat_codes() and a_sig.code in _threat_codes():
            if delta_rq > 0.01 or (delta_dice is not None and delta_dice > 0.005):
                transition = "TRADEOFF"
            else:
                transition = "REGRESSION"
        elif a_sig.code == b_sig.code:
            transition = "UNCHANGED"
        else:
            transition = "TRADEOFF"
        transitions.append(
            RadarTransition(
                case=case,
                before_signal=b_sig.code,
                after_signal=a_sig.code,
                transition=transition,
                delta_rq=delta_rq,
                delta_hd95=delta_hd95,
                delta_dice=delta_dice,
            )
        )
    return transitions


def print_radar_diff(before: Iterable[dict | CaseTelemetry], after: Iterable[dict | CaseTelemetry]) -> None:
    print("=" * 70)
    print("📡 PENGWIN-2026 OOF RADAR DIFF")
    print("=" * 70)
    for t in compare_radar_reports(before, after):
        dice = "n/a" if t.delta_dice is None else f"{t.delta_dice:+.3f}"
        print(
            f"case_{t.case:>4} | {t.before_signal:16s} -> {t.after_signal:16s} | "
            f"{t.transition:10s} | ΔRQ {t.delta_rq:+.3f} | ΔHD95 {t.delta_hd95:+.1f} | ΔDice {dice}"
        )
    print("=" * 70)


def render_ppi_frame(azimuth_deg: int, active_case: str = "") -> str:
    az = int(azimuth_deg) % 360
    active = active_case or "----"
    return "\n".join(
        [
            "======================================================================",
            "📡 PENGWIN-2026 OOF DIAGNOSTIC PPI-SCOPE",
            f"[ANTENNA AZIMUTH: {az:03d}°] [ACTIVE CASE: {active}]",
            "----------------------------------------------------------------------",
            "     \\     |     /",
            "       \\   |   /",
            "         \\ | /",
            "  ---------•---------",
            "         / | \\",
            "       /   |   \\",
            "     /     |     \\",
            "----------------------------------------------------------------------",
        ]
    )


def animate_ppi(rows: Iterable[dict | CaseTelemetry], delay_s: float = 0.05) -> None:
    telemetry = [normalize_telemetry(row) for row in rows]
    if not sys.stdout.isatty():
        print_radar_report(telemetry, mode="OOF_SCAN_STATIC")
        return
    for idx, row in enumerate(telemetry):
        print("\033[2J\033[H", end="")
        print(render_ppi_frame((idx * 37) % 360, active_case=row.case))
        print_radar_report([row], mode="PPI_SWEEP")
        time.sleep(delay_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppi", action="store_true", help="show optional TTY-only sweep animation")
    args = parser.parse_args()
    demo = [
        {"case": "002", "gtN": 0, "n_pred": 0, "rq": 1.0, "hd95": 0.0},
        {"case": "004", "gtN": 2, "n_pred": 1, "n_cc": 2, "n_ws": 2, "rq": 0.667, "hd95": 43.5},
        {"case": "005", "gtN": 1, "n_pred": 1, "rq": 1.0, "hd95": 0.8},
        {"case": "284", "gtN": 2, "n_cc": 2, "n_ws": 7, "n_pred": 7, "rq": 0.20, "hd95": 67.9},
        {"case": "291", "gtN": 2, "n_pred": 2, "rq": 0.91, "hd95": 26.3, "dice": 0.977},
    ]
    if args.ppi:
        animate_ppi(demo)
    else:
        print_radar_report(demo, mode="DEMO_BEFORE_FIX")


if __name__ == "__main__":
    main()
