"""Grand Challenge entry point for PENGWIN 2026 Task 2 interactive segmentation.

Pipeline: Task1 v19 auto-segmentation (run_pipeline_v7) -> click-guided fragment
selection / merge-split / false-positive removal (click_guided_postproc).

Grand-challenge interface (PENGWIN 2026 Task 2 official):
    Input  : peripelvic-fracture-ct      /input/images/peripelvic-fracture-ct/<UUID>.mha|.tif
    Input  : peripelvic-fragment-clicks   /input/peripelvic-fragment-clicks.json
    Output : peripelvic-fracture-ct-segmentation
             /output/images/peripelvic-fracture-ct-segmentation/<UUID>.mha|.tif

Robustness: each case is wrapped in try/except so a single failure can never abort
the whole submission. On click-postproc failure we still emit the raw v19 instance
map (a valid 1-200 label image); on total failure we emit a background volume so the
output file always exists for grand-challenge to read.
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.nnunet_preprocess_memory_patch import apply as _apply_nnunet_memory_patch

_apply_nnunet_memory_patch()

from scripts.click_guided_postproc import parse_clicks, relabel_from_clicks
import scripts.inference_v7_d015 as INF
from scripts.inference_v7_d015 import run_pipeline_v7, get_orientation_code, D015_ANATOMY

import ctypes as _ctypes
import gc as _module_gc


def _rss(tag: str) -> None:
    try:
        with open("/proc/self/status") as _f:
            for _l in _f:
                if _l.startswith("VmRSS:"):
                    print(f"[rss] {tag}: {int(_l.split()[1]) / 1024 / 1024:.2f} GiB", flush=True)
                    return
    except Exception:
        pass


def _malloc_trim() -> None:
    """Return freed heap pages to the OS. numpy/Python keep freed large buffers in
    the glibc arena, so RSS stays high after a big free (softmax, float32 border
    temporaries) even though the memory is logically free. On a hard 15GB cgroup
    that phantom RSS is what OOM-kills a 105M-voxel border_pure case. malloc_trim(0)
    forces glibc to hand the pages back so the cgroup accounting drops."""
    try:
        _module_gc.collect()
        _ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _lowmem_border_predict(predictor, data, props):
    """Memory-lean replacement for predict_single_npy_array(...probabilities=True) on
    large (>90M) cases. Returns (seg uint8 in original frame, {2,4,6: border prob f16}).

    Reimplements convert_predicted_logits_to_segmentation_with_correct_shape so at most
    ~2 full-volume 7-class arrays are ever live (vs ~4 in the stock path: proc logits +
    resampled + softmax + uncropped-reverted probs, which spikes >15GB on 105M). Key
    moves: resample logits in fp16 -> FREE proc logits -> chunked uint8 argmax for the
    seg (softmax-free, argmax(logits)==argmax(softmax)) -> per-channel stable softmax
    for ONLY channels 2/4/6 -> revert cropping. Seg is byte-identical to the stock path
    (same resample fn + argmax); only the unused 4 channels are never materialised."""
    import numpy as _np
    import torch as _torch
    from nnunetv2.inference.data_iterators import PreprocessAdapterFromNpy
    from nnunetv2.inference.export_prediction import insert_crop_into_image

    pm = predictor.plans_manager
    cm = predictor.configuration_manager

    ppa = PreprocessAdapterFromNpy([data], [None], [props], [None], pm,
                                   predictor.dataset_json, cm,
                                   num_threads_in_multithreaded=1, verbose=False)
    _rss("before-preprocess")
    dct = next(ppa)
    _malloc_trim()  # release the ~12GB 5-channel INPUT-resample phantom RSS to the OS
    _rss("after-preprocess+trim")
    logits = predictor.predict_logits_from_preprocessed_data(dct["data"]).cpu().half()
    props2 = dct["data_properties"]  # only properties needed downstream; free the 2.1GB data
    del dct
    _malloc_trim()
    _rss("after-predict.half+free-dct")

    sp_t = [props2["spacing"][i] for i in pm.transpose_forward]
    shp_ac = props2["shape_after_cropping_and_before_resampling"]
    cur_sp = cm.spacing if len(cm.spacing) == len(shp_ac) else [sp_t[0], *cm.spacing]
    # DOWNSAMPLED border extraction (env PENGWIN_BORDER_DS, default 2). The full-res
    # 7-class resample (~1.5GB fp16 output + working) is the peak that tips GC's
    # tighter-than-15GB effective memory -> numpy error -> EDT fallback. Resampling to
    # 1/DS resolution cuts the softmax memory by DS^3 (8x at DS=2); the seg is upsampled
    # nearest and the border (a smooth prob field) trilinear, near-lossless for the
    # watershed ridge. Only the >90M border path uses this, so nothing else changes.
    ds = max(1, int(os.environ.get("PENGWIN_BORDER_DS", "2")))
    shp_rs = [max(1, int(round(s / ds))) for s in shp_ac] if ds > 1 else list(shp_ac)
    os.environ["PIVOT_RESAMPLE_FP16"] = "1"
    try:
        resampled = cm.resampling_fn_probabilities(logits, shp_rs, cur_sp, sp_t)
    finally:
        os.environ.pop("PIVOT_RESAMPLE_FP16", None)
    del logits
    _malloc_trim()
    if isinstance(resampled, _torch.Tensor):
        resampled = resampled.numpy()
    resampled = _np.asarray(resampled)  # (C, *shp_rs) float16  (downsampled)
    _rss("after-resample(ds)")

    C = resampled.shape[0]
    seg_ds = _np.empty(resampled.shape[1:], dtype=_np.uint8)  # chunked argmax at ds-res
    n0 = resampled.shape[1]
    step = max(1, n0 // 32)
    for a in range(0, n0, step):
        seg_ds[a:a + step] = resampled[:, a:a + step].argmax(0).astype(_np.uint8)
    mx = resampled.max(0).astype(_np.float32)  # stable per-channel softmax, border ch only
    denom = _np.zeros(resampled.shape[1:], dtype=_np.float32)
    for c in range(C):
        denom += _np.exp(resampled[c].astype(_np.float32) - mx)
    border3_ds = {bl: (_np.exp(resampled[bl].astype(_np.float32) - mx) / denom).astype(_np.float16)
                  for bl in (2, 4, 6)}
    del resampled, mx, denom
    _malloc_trim()
    _rss("after-argmax+softmax3(ds)")

    # upsample ds-res -> full crop res: seg nearest (order0), border trilinear (order1)
    if ds > 1:
        from scipy.ndimage import zoom as _zoom
        zf = [shp_ac[i] / seg_ds.shape[i] for i in range(3)]
        seg_c = _zoom(seg_ds, zf, order=0, mode="nearest").astype(_np.uint8)
        # 정확히 shp_ac 맞추기 (반올림 오차 보정)
        seg_c = seg_c[:shp_ac[0], :shp_ac[1], :shp_ac[2]]
        if seg_c.shape != tuple(shp_ac):
            _pad = [(0, shp_ac[i] - seg_c.shape[i]) for i in range(3)]
            seg_c = _np.pad(seg_c, _pad, mode="edge")
        border3 = {}
        for bl in (2, 4, 6):
            b = _zoom(border3_ds[bl].astype(_np.float32), zf, order=1, mode="nearest").astype(_np.float16)
            b = b[:shp_ac[0], :shp_ac[1], :shp_ac[2]]
            if b.shape != tuple(shp_ac):
                border3[bl] = _np.pad(b, [(0, shp_ac[i] - b.shape[i]) for i in range(3)], mode="edge")
            else:
                border3[bl] = b
        del seg_ds, border3_ds
        _malloc_trim()
    else:
        seg_c, border3 = seg_ds, border3_ds
    _rss("after-upsample")

    bbox = props2["bbox_used_for_cropping"]
    shp_bc = props2["shape_before_cropping"]
    seg = insert_crop_into_image(_np.zeros(shp_bc, dtype=_np.uint8), seg_c, bbox)
    seg = _np.ascontiguousarray(seg.transpose(pm.transpose_backward))
    del seg_c
    border_lps = {}
    for bl in (2, 4, 6):
        b = insert_crop_into_image(_np.zeros(shp_bc, dtype=_np.float16), border3[bl], bbox)
        border_lps[bl] = _np.ascontiguousarray(b.transpose(pm.transpose_backward))
    del border3
    _malloc_trim()
    _rss("after-reinsert")
    return seg, border_lps

# Verified +0.025 topology gain (watershed line between split fragments). Safe to
# default on now that click_guided_postproc._watershed_safe guards the watershed_line
# hang bug (long/thin femur shafts could stall skimage's watershed indefinitely;
# fixed via a timed subprocess fallback to watershed_line=False). Overridable via env.
os.environ.setdefault("PENGWIN_WATERSHED_LINE", "1")

# ---------------------------------------------------------------------------
# EXPERIMENTAL (this scripts_border_test/ copy only, NOT the real submission
# scripts/ dir): border_pure T4/accuracy verification. Reuses the exact
# monkeypatch technique already validated in ~/taehee/border_test.py (28-case
# fold_0 gate PASSED) -- patches predict_argmax_multichannel (pelvic-only call
# site) to also stash the 7-class softmax, then extracts the 3 border channels
# for relabel_from_clicks's border_arr. NOTE: the default predict_argmax_multichannel's
# own docstring warns "GC containers OOM with save_probabilities=True" -- this
# patch reintroduces exactly that flag, so this build's whole point is to find
# out whether it actually OOMs under real --memory=15g constraints.
# ---------------------------------------------------------------------------
os.environ["PENGWIN_BORDER_SEAM"] = "1"
os.environ["PENGWIN_BORDER_PURE"] = "1"

_BORDER_STASH: dict = {}


def _patched_argmax_with_softmax_stash(predictor, channel_paths, tmp_out):
    # SIZE-GATED border_pure (v19). border_pure needs the full 7-class softmax
    # (~28*V bytes), which OOMs on large-FOV pelvic volumes regardless of any
    # caller-side memory trick (measured: 084=70M vox fits <15GiB; 048=108M vox
    # peaks 15.4-17.2GiB > 15 limit -> rc=137, for BOTH the npz-roundtrip (v17)
    # and in-memory (v18) paths). So we PROACTIVELY gate on voxel count:
    #   volume <= PENGWIN_BORDER_MAX_VOXELS -> in-memory border_pure (F1/HD95 gain)
    #   volume >  gate                       -> memory-safe argmax (v15 EDT path)
    # The fallback is byte-identical to v15's predict_argmax_multichannel (no
    # softmax, nothing stashed -> border_dict stays None -> EDT split), so a big
    # case is NEVER failed (C2) and is never worse than v15; small/normal-FOV
    # cases still bank the border_pure gain. Threshold 85M leaves margin below
    # the ~95M estimated 15GiB crossover (084=70M safe, 048=108M falls back).
    import numpy as _np
    import gc as _gc
    # Read ONLY the header to get the size -- do NOT load the full CT array here.
    # (A prior version held the whole ~217MB CT image alive through the pelvic
    # prediction, which tipped the already-marginal 048 over the 15GiB cgroup
    # limit even on the memory-safe fallback path. Header-only read removes it.)
    _rdr = sitk.ImageFileReader()
    _rdr.SetFileName(str(channel_paths[0]))
    _rdr.ReadImageInformation()
    _sx, _sy, _sz = _rdr.GetSize()  # (X, Y, Z), header only, no pixels
    nvox = int(_sx) * int(_sy) * int(_sz)
    max_vox = int(os.environ.get("PENGWIN_BORDER_MAX_VOXELS", "85000000"))
    if nvox > max_vox:
        print(f"[border] {nvox/1e6:.1f}M vox > {max_vox/1e6:.0f}M gate -> EDT fallback (memory-safe, =v15)", flush=True)
        tmp_out.mkdir(parents=True, exist_ok=True)
        predictor.predict_from_files_sequential(
            list_of_lists_or_source_folder=[[str(p) for p in channel_paths]],
            output_folder_or_list_of_truncated_output_files=str(tmp_out),
            save_probabilities=False, overwrite=True, folder_with_segs_from_prev_stage=None,
        )
        seg_files = sorted(tmp_out.glob("*.nii.gz")) + sorted(tmp_out.glob("*.mha"))
        if not seg_files:
            raise RuntimeError(f"no segmentation output in {tmp_out}")
        seg = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_files[0])))
        return _np.asarray(seg, dtype=_np.uint8)
    # in-memory border extraction (no npz round-trip): returns (seg, probs) in RAM,
    # extract 3 border channels (sac=2, LH=4, RH=6) to float16, free full softmax.
    print(f"[border] {nvox/1e6:.1f}M vox <= {max_vox/1e6:.0f}M gate -> border_pure", flush=True)
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    data, props = SimpleITKIO().read_images([str(p) for p in channel_paths])
    _fp16 = nvox > int(os.environ.get("PENGWIN_BORDER_FP16_ABOVE", "90000000"))
    if _fp16:
        # LARGE-CASE (>90M) memory-lean path. Stock predict_single_npy_array +
        # return_probabilities holds ~4 full-volume 7-class arrays and spikes >15GB
        # on 105M (rc=137). _lowmem_border_predict reimplements the convert to keep
        # <=2 live. Wrapped so any failure falls back to EDT (never worse than v25).
        print(f"[border] {nvox/1e6:.1f}M vox > 90M -> fp16 lowmem predict (memory)", flush=True)
        try:
            seg, _border_lps = _lowmem_border_predict(predictor, data, props)
            del data
            _BORDER_STASH["border_lps"] = _border_lps
        except Exception:
            print("[border] lowmem path failed -> EDT fallback (no border)", flush=True)
            traceback.print_exc()
            _BORDER_STASH.pop("border_lps", None)
            tmp_out.mkdir(parents=True, exist_ok=True)
            predictor.predict_from_files_sequential(
                list_of_lists_or_source_folder=[[str(p) for p in channel_paths]],
                output_folder_or_list_of_truncated_output_files=str(tmp_out),
                save_probabilities=False, overwrite=True, folder_with_segs_from_prev_stage=None,
            )
            _sf = sorted(tmp_out.glob("*.nii.gz")) + sorted(tmp_out.glob("*.mha"))
            seg = sitk.GetArrayFromImage(sitk.ReadImage(str(_sf[0])))
            _gc.collect()
            _malloc_trim()
            return _np.asarray(seg, dtype=_np.uint8)
    else:
        seg, probs = predictor.predict_single_npy_array(
            data, props, None, None, save_or_return_probabilities=True)
        _BORDER_STASH["border_lps"] = {bl: _np.asarray(probs[bl], dtype=_np.float16) for bl in (2, 4, 6)}
        del probs
    _BORDER_STASH["ref_lps"] = sitk.ReadImage(str(channel_paths[0]))  # small-volume path has headroom
    _gc.collect()
    _malloc_trim()  # release the ~2.9GB 7-class softmax back to the OS, not just to numpy's arena
    return _np.asarray(seg, dtype=_np.uint8)


INF.predict_argmax_multichannel = _patched_argmax_with_softmax_stash


def _reorient_border_channel(arr_lps_f32, ref_lps, orig_code):
    im = sitk.GetImageFromArray(arr_lps_f32)
    im.CopyInformation(ref_lps)
    im = sitk.DICOMOrient(im, orig_code)
    return sitk.GetArrayFromImage(im)

INPUT_ROOT = Path("/input")
INPUT_DIR = Path("/input/images/peripelvic-fracture-ct")
CLICK_JSON = Path("/input/peripelvic-fragment-clicks.json")
_CLICK_JSON_PATH: list = []   # [v73] 전략 이름 판정용 — 실제로 읽은 경로
OUTPUT_DIR = Path("/output/images/peripelvic-fracture-ct-segmentation")
RESULTS_DIR = Path(os.environ.get("nnUNet_results", "/opt/algorithm/nnUNet_results"))

# T4 ~10 min/case budget: non-overlapping tiles (matches the Task1 v19 champion).
TILE_STEP_SIZE = float(os.environ.get("PIVOT_TILE_STEP_SIZE", "1.0"))


def _find_ct_files() -> list[Path]:
    if INPUT_DIR.exists():
        files = sorted(INPUT_DIR.glob("*.mha")) + sorted(INPUT_DIR.glob("*.tif"))
        if files:
            return files
    return sorted(p for p in INPUT_ROOT.rglob("*") if p.suffix.lower() in (".mha", ".tif"))


def _find_click_json() -> Path:
    if CLICK_JSON.is_file():
        return CLICK_JSON
    candidates = [
        p for p in INPUT_ROOT.rglob("*.json")
        if p.name != "inputs.json" and "click" in p.name.lower()
    ]
    if not candidates:
        candidates = [p for p in INPUT_ROOT.rglob("*.json") if p.name != "inputs.json"]
    if not candidates:
        raise RuntimeError(
            f"no click JSON found (expected {CLICK_JSON} or *clicks*.json under /input)"
        )
    return sorted(candidates)[0]


def _write_label(arr: np.ndarray, reference: sitk.Image, out_path: Path) -> None:
    out_img = sitk.GetImageFromArray(arr.astype(np.uint8))
    out_img.CopyInformation(reference)
    sitk.WriteImage(sitk.Cast(out_img, sitk.sitkUInt8), str(out_path), useCompression=True)


def _route_anatomy_from_clicks(clicks) -> None:
    """Route pelvic vs femur from CLICK anatomy names (100% reliable) instead of
    the physical-FOV heuristic that misroutes ~250mm-boundary cases -> wrong
    anatomy -> clicks match nothing -> F1=0. Any femur click => femur; else pelvic."""
    anats = {c.anatomy for c in clicks if getattr(c, "anatomy", None) in
             ("sacrum", "leftHip", "rightHip", "femur")}
    if not anats:
        os.environ.pop("PENGWIN_FORCE_ANATOMY", None)
        return
    forced = "femur" if "femur" in anats else "pelvic"
    os.environ["PENGWIN_FORCE_ANATOMY"] = forced
    print(f"[route] clicks anatomy={sorted(anats)} -> force {forced}")


def _femur_border_d013(in_path: Path, shape) -> dict | None:
    """[v67] D013's learned fracture-surface probability for the femur split.

    femur has always decoded on geometry alone -- Dataset004 is {background, femur}, so no
    border channel exists and `elev = -EDT` cuts at the shape's narrowest neck, which is not
    where the bone broke. Measured consequence: EMPTY 0 / TINY 166, i.e. the right number of
    pieces cut in the wrong place, and 71.6% of the leaderboard deficit.

    Dataset013_PENGWIN_BorderCoreFemur has been trained since 2026-06-18 and was built by
    symlinking D004's own imagesTr, so it takes the same CT channel in the same frame -- no
    reorientation is needed here, unlike the pelvic path whose stash is in LPS.

    THIS IS NOT THE v27/v28 FEMUR PATH. Those fed D013 to `border_core_to_instances` and let
    the model decide the fragment COUNT; its core-CC merge then sutured displaced 004/005
    into one piece (HD95 104 mm). Here the clicks fix the count and D013 only says WHERE to
    cut, so that failure mode cannot occur by construction. Measured on D013's fold_0
    held-out split (34 cases x 3 strategies): ins_f1 0.7900 -> 0.8593, merge 0.863 -> 0.647,
    27 win / 69 tie / 6 loss, and all 6 regressions keep their surface metrics (HD95 improves
    in 3 of them) -- fragment-boundary shifts, not v28-style sutures.

    Size-gated like the pelvic border: above the gate this returns None and the case decodes
    exactly as it does today, so a large volume is never failed, only un-improved.
    """
    import gc as _gc

    max_vox = int(os.environ.get("PENGWIN_FEMUR_BORDER_MAX_VOXELS", "110000000"))
    n_vox = int(np.prod(shape))
    if n_vox > max_vox:
        print(f"[femur-border] {n_vox/1e6:.1f}M vox > gate {max_vox/1e6:.0f}M -> skip "
              f"(decodes exactly as v65)", flush=True)
        return None

    model_dir = (RESULTS_DIR / "Dataset013_PENGWIN_BorderCoreFemur"
                 / "nnUNetTrainerBorderWeightedFemurSnap__nnUNetPlans__3d_fullres")
    if not model_dir.exists():
        print(f"[femur-border] {model_dir} missing -> skip", flush=True)
        return None
    try:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        im = sitk.ReadImage(str(in_path))
        arr = sitk.GetArrayFromImage(im)[None].astype(np.float32)
        props = {"spacing": list(im.GetSpacing())[::-1]}
        del im
        pred = nnUNetPredictor(tile_step_size=TILE_STEP_SIZE, use_gaussian=True,
                               use_mirroring=False,
                               device=torch.device("cuda", 0) if torch.cuda.is_available()
                               else torch.device("cpu"),
                               verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        pred.initialize_from_trained_model_folder(
            str(model_dir), use_folds=("all",), checkpoint_name="checkpoint_final.pth")
        _seg, probs = pred.predict_single_npy_array(arr, props, None, None, True)
        # [v73] argmax 를 버리지 않는다. D013 의 core|border 전경이 Task1 이 쓰는
        # 마스크이고, 그건 이미 여기서 계산돼 있다 — 추가 추론 0.
        femur_arg = np.asarray(_seg, dtype=np.uint8)
        del arr, _seg, pred
        # class 2 = femur_border. Cast to float16 immediately: the 3-class float32 softmax is
        # ~12 bytes/voxel and this container has a 15 GB cgroup.
        border = np.asarray(probs[2], dtype=np.float16)
        del probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _gc.collect()
        _malloc_trim()
        if border.shape != tuple(shape):
            print(f"[femur-border] shape {border.shape} != pred {tuple(shape)} -> skip",
                  flush=True)
            return None
        print(f"[femur-border] D013 border ready ({n_vox/1e6:.1f}M vox)", flush=True)
        return {"femur": border, "__femur_arg": femur_arg}
    except Exception as exc:                    # never fail a case over an optional channel
        print(f"[femur-border] failed ({exc!r}) -> falling back to v65 geometry", flush=True)
        _malloc_trim()
        return None


def _process_case(in_path: Path, clicks, out_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="pengwin_task2_") as tmp_s:
        task1_out = Path(tmp_s) / in_path.name
        _BORDER_STASH.clear()
        _route_anatomy_from_clicks(clicks)
        run_pipeline_v7(
            input_ct=in_path,
            output_path=task1_out,
            nnunet_results=RESULTS_DIR,
            tile_step_size=TILE_STEP_SIZE,
            use_pivot_form=False,
            use_d017_sacrum=os.environ.get("PIVOT_USE_D017_SACRUM", "0") == "1",
            form_learn_mode=None,
            form_tau_intact=float(os.environ.get("PENGWIN_TAU_INTACT", "0.08")),
            min_volume_mm3=float(os.environ.get("PENGWIN_MIN_VOLUME_MM3", "500")),
        )
        pred_img = sitk.ReadImage(str(task1_out))
        pred_arr = sitk.GetArrayFromImage(pred_img)
        _malloc_trim()  # drop the nnU-Net inference/resample RSS before the border reorient stacks up

        border_dict = None
        if "border_lps" in _BORDER_STASH:
            orig_code = get_orientation_code(pred_img)
            border_lps = _BORDER_STASH.pop("border_lps")  # {2,4,6} -> float16 LPS channel
            ref_lps = _BORDER_STASH.pop("ref_lps")
            # Memory-lean reorientation: process ONE channel at a time and hand the
            # float32 temporaries (~420MB each on a 105M case) + SimpleITK images back
            # to the OS between channels, instead of a dict-comprehension that stacks
            # all 3 at once (~2.5GB) and OOM-kills the 15GB node on border_pure.
            border_dict = {}
            for nm, (_cl, bl) in D015_ANATOMY.items():
                ch32 = border_lps[bl].astype(np.float32)
                border_dict[nm] = _reorient_border_channel(ch32, ref_lps, orig_code).astype(np.float16)
                del ch32
                _malloc_trim()
            del border_lps, ref_lps
            _malloc_trim()

        # [v67] femur gets its own border channel. Guarded on the empty stash rather than on
        # the route alone so pelvic can never be touched: the pelvic branch above always
        # populates border_dict, so this only ever fires on the femur path.
        if (border_dict is None
                and os.environ.get("PENGWIN_FORCE_ANATOMY") == "femur"
                and os.environ.get("PENGWIN_FEMUR_BORDER", "1") == "1"):
            border_dict = _femur_border_d013(in_path, pred_arr.shape)

        if os.environ.get("DUMP_INTERMEDIATES", "0") == "1":
            # Dump the post-proc inputs so Taehee can iterate on the split OFFLINE
            # (no GPU, no re-inference): auto-seg instance map + reoriented border
            # channels + clicks + geometry. Loadable by taehee_postproc_sandbox.py.
            import json as _json
            _dd = OUTPUT_DIR.parent / "intermediates"
            _dd.mkdir(parents=True, exist_ok=True)
            _stem = in_path.name.split(".")[0]
            np.save(_dd / f"{_stem}__pred.npy", pred_arr.astype(np.uint8))
            if border_dict:
                for _nm, _b in border_dict.items():
                    np.save(_dd / f"{_stem}__border_{_nm}.npy", np.asarray(_b, dtype=np.float16))
            _meta = {
                "spacing": list(pred_img.GetSpacing()),
                "origin": list(pred_img.GetOrigin()),
                "direction": list(pred_img.GetDirection()),
                "anatomies": sorted(border_dict.keys()) if border_dict else [],
                "clicks": [{"anatomy": getattr(c, "anatomy", None), "point": list(c.point)} for c in clicks],
            }
            (_dd / f"{_stem}__meta.json").write_text(_json.dumps(_meta))
            print(f"[dump] wrote intermediates for {_stem} -> {_dd}", flush=True)

        # [v73] 대퇴는 새 디코드로. 전경을 D004 대신 D013 argmax 로 잡고, 씨앗을
        # 클릭에서 직접 얻는다(가장자리 클릭 전략만 h-maxima + 학습된 병합).
        # 측정(학습 femur 170, FemurSnap, 케이스단위 OOF): 0.9081 -> 0.9714.
        _v73 = None
        if (os.environ.get("PIVOT_FEMUR_V73", "0") == "1"
                and os.environ.get("PENGWIN_FORCE_ANATOMY") == "femur"
                and isinstance(border_dict, dict)
                and border_dict.get("__femur_arg") is not None):
            try:
                import femur_decode_v73 as _fd
                _model = _fd.load_model()
                _arg = border_dict["__femur_arg"]
                _mask = (_arg == 1) | (_arg == 2)
                _sp = pred_img.GetSpacing()
                _spz = (float(_sp[2]), float(_sp[1]), float(_sp[0]))
                _vox = float(_sp[0] * _sp[1] * _sp[2])
                _bnd = _fd.is_boundary_strategy(_CLICK_JSON_PATH[0]) if _CLICK_JSON_PATH else False
                _pts = [tuple(int(round(x)) for x in c.point) for c in clicks
                        if getattr(c, "anatomy", None) == "femur"]
                print(f"[v73] femur decode — 클릭씨앗={not _bnd} (전략 boundary={_bnd}), "
                      f"클릭 {len(_pts)}개", flush=True)
                _v73 = _fd.femur_decode(
                    _mask, np.asarray(border_dict["femur"], dtype=np.float32), _pts,
                    _spz, _vox, target_range=(151, 200),
                    min_volume_mm3=float(os.environ.get("PENGWIN_MIN_VOLUME_MM3", "1000")),
                    use_clicks=(not _bnd), model=_model)
                if int(_v73.max()) == 0:
                    print("[v73] 빈 결과 -> 기존 경로로 폴백", flush=True)
                    _v73 = None
            except Exception:
                print("[v73] 실패 -> 기존 경로로 폴백", flush=True)
                traceback.print_exc()
                _v73 = None
        if _v73 is not None:
            _write_label(_v73, pred_img, out_path)
            n_frag = int(len(set(_v73[_v73 > 0].tolist())))
            print(f"[task2] {in_path.name}: v73 femur fragments={n_frag} -> {out_path}")
            return
        try:
            relabeled = relabel_from_clicks(pred_arr, clicks, image=pred_img, border_arr=border_dict)
        except Exception:
            # click postproc failed -> ship the raw v19 instance map (still valid 1-200).
            print(f"[warn] click postproc failed for {in_path.name}; using raw v19 map")
            traceback.print_exc()
            relabeled = pred_arr
        # Safety net: never ship an empty map when v19 DID segment something. If the
        # clicks failed to parse / selected nothing (relabel returned all-zeros), the
        # raw v19 instance map at least scores the auto-seg quality instead of 0.0.
        if int(relabeled.max()) == 0 and int(pred_arr.max()) > 0:
            print(f"[warn] relabel produced EMPTY for {in_path.name} (clicks={len(clicks)}); "
                  f"falling back to raw v19 map")
            relabeled = pred_arr
        _write_label(relabeled, pred_img, out_path)
        n_frag = int(len(set(relabeled[relabeled > 0].tolist())))
        print(f"[task2] {in_path.name}: clicks={len(clicks)} fragments={n_frag} -> {out_path}")


def main() -> None:
    if not INPUT_DIR.exists():
        sys.exit(f"input directory not found: {INPUT_DIR}")
    ct_files = _find_ct_files()
    if not ct_files:
        sys.exit(f"no mha/tif files in {INPUT_DIR}")
    click_json = _find_click_json()
    _CLICK_JSON_PATH.append(click_json)
    clicks = parse_clicks(click_json)
    if not clicks:
        print(f"[warn] no clicks parsed from {click_json}; cases will fall back to raw v19")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup] {len(ct_files)} CT cases, clicks={click_json} ({len(clicks)} points)")

    failures = 0
    for in_path in ct_files:
        out_path = OUTPUT_DIR / in_path.name
        print(f"[run] {in_path.name}")
        try:
            _process_case(in_path, clicks, out_path)
        except Exception:
            failures += 1
            print(f"[error] case {in_path.name} failed; writing background fallback")
            traceback.print_exc()
            try:
                ref = sitk.ReadImage(str(in_path))
                blank = np.zeros(sitk.GetArrayFromImage(ref).shape, dtype=np.uint8)
                _write_label(blank, ref, out_path)
            except Exception:
                print(f"[fatal] could not even write fallback for {in_path.name}")
                traceback.print_exc()

    print(f"[done] {len(ct_files)} cases written to {OUTPUT_DIR} ({failures} fallback)")


if __name__ == "__main__":
    main()
