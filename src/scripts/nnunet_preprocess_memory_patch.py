"""Reduce nnUNet preprocessing peak RAM during 5-channel pelvic resampling.

GC Task2 containers OOM when nnUNet does ``data.astype(float)`` (float64) on
(c, x, y, z) — ~4 GiB for large CT + 5 channels. The old patch only converted
input to float32; the original function still promoted to float64.

This module replaces ``resample_data_or_seg`` with a float32 + per-channel path.
Import and call ``apply()`` before any nnUNet inference.

DEPLOY-ONLY COPY -- do NOT promote this file over scripts/nnunet_preprocess_memory_patch.py.
apply() here also installs the probability stash (PENGWIN_PROBS_STASH, default on),
which makes nnUNet's export skip the .npz. Every caller that still does
``predict_from_files_sequential(save_probabilities=True)`` and then globs for the
.npz -- scripts/inference_full_pipeline.py, scripts/inference_d006_sacrum.py,
scripts/export_d015_probs.py -- would raise "no .npz from inference". The images
that ship this file pair it with a predict_softmax that pops the stash first.
"""
from __future__ import annotations

import gc
from collections import OrderedDict
from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage import map_coordinates
from skimage.transform import resize


def _resample_data_or_seg_float32(
    data: np.ndarray,
    new_shape: Union[Tuple[float, ...], List[float], np.ndarray],
    is_seg: bool = False,
    axis: Union[None, int] = None,
    order: int = 3,
    do_separate_z: bool = False,
    order_z: int = 0,
    dtype_out=None,
) -> np.ndarray:
    """nnUNet ``resample_data_or_seg`` with float32 working buffers (not float64)."""
    assert data.ndim == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == data.ndim - 1

    if is_seg:
        resize_fn = resize_segmentation
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {"mode": "edge", "anti_aliasing": False}

    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if dtype_out is None:
        dtype_out = np.float32 if not is_seg else data.dtype
    reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
    if np.any(shape != new_shape):
        if not is_seg:
            data = np.asarray(data, dtype=np.float32)
        else:
            data = np.asarray(data, copy=False)

        if do_separate_z:
            assert axis is not None, "If do_separate_z, we need to know what axis is anisotropic"
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            for c in range(data.shape[0]):
                tmp = deepcopy(new_shape)
                tmp[axis] = shape[axis]
                reshaped_here = np.zeros(tmp, dtype=dtype_out)
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_here[slice_id] = resize_fn(
                            data[c, slice_id], new_shape_2d, order, **kwargs
                        )
                    elif axis == 1:
                        reshaped_here[:, slice_id] = resize_fn(
                            data[c, :, slice_id], new_shape_2d, order, **kwargs
                        )
                    else:
                        reshaped_here[:, :, slice_id] = resize_fn(
                            data[c, :, :, slice_id], new_shape_2d, order, **kwargs
                        )
                if shape[axis] != new_shape[axis]:
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_here.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final[c] = map_coordinates(
                            reshaped_here, coord_map, order=order_z, mode="nearest"
                        )[None]
                    else:
                        unique_labels = np.sort(pd.unique(reshaped_here.ravel()))
                        for cl in unique_labels:
                            mask = (reshaped_here == cl).astype(np.float32)
                            mapped = map_coordinates(
                                mask, coord_map, order=order_z, mode="nearest"
                            )
                            reshaped_final[c][np.round(mapped) > 0.5] = cl
                else:
                    reshaped_final[c] = reshaped_here
        else:
            for c in range(data.shape[0]):
                reshaped_final[c] = resize_fn(data[c], new_shape, order, **kwargs)
        return reshaped_final
    return data


def _resample_multichannel_lowmem(
    data: np.ndarray,
    new_shape: Union[Tuple[float, ...], List[float], np.ndarray],
    is_seg: bool,
    axis: Union[None, int],
    order: int,
    do_separate_z: bool,
    order_z: int,
    dtype_out,
) -> np.ndarray:
    """Resample each channel separately, in parallel, memory-safe on a 16 GiB node.

    scipy's order>=2 (cubic) resampling internally allocates a *float64* spline
    prefilter at the INPUT resolution (~1 GiB per channel for a large pelvic CT);
    running several of those in parallel OOMs. Two guards:

    * The 5-channel pelvic INPUT is [CT, offset_x, offset_y, offset_z, side_mask].
      Channels 1..4 are analytic (clipped-)linear ramps, so order-1 resampling is
      ~exact for them and skips the float64 spline entirely. Only CT (channel 0)
      keeps the requested high order, so at most ONE float64 prefilter is ever live.
    * The worker cap is sized from the real float64 spline cost at the INPUT
      resolution (not the smaller output), so it can't under-count memory again.

    Output is written into a preallocated buffer; the thread pool is exact (skimage/
    scipy release the GIL during the C spline work).
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    import os as _os

    n_ch = data.shape[0]
    new_shape = np.asarray(new_shape)
    if dtype_out is None:
        dtype_out = np.float32 if not is_seg else data.dtype
    # BORDER_PURE FP16 (opt-in via env, set only around a >90M-voxel border_pure
    # predict). The 7-class softmax resample holds float32 OUTPUT (2.94GB @105M) at
    # the SAME time as the float32 INPUT logits (2.94GB) -> 5.9GB tips the 15GB node.
    # float16 output halves both the output buffer here and the returned probs the
    # border channels are read from; argmax over float16 is identical for our
    # purposes. Only enabled for large cases that otherwise get NO border (EDT
    # fallback), so the proven <=90M float32 path is untouched -> can only help.
    if (not is_seg) and _os.environ.get("PIVOT_RESAMPLE_FP16", "0") == "1":
        dtype_out = np.float16
    out = np.zeros((n_ch, *new_shape), dtype=dtype_out)

    # LARGE-VOLUME OOM GUARD. scipy's order>=2 (cubic) resampling builds a
    # *float64* spline prefilter at the OUTPUT resolution (~932 MiB for a large
    # pelvic CT, shape ~536x425x536). On a real 15 GiB node this single
    # allocation tips a big case over -> numpy ArrayMemoryError -> the whole
    # case falls back to an EMPTY output -> score 0 (confirmed root cause of the
    # leaderboard's large-case failures). When the output is large we drop to
    # order-1 (linear), which scipy does WITHOUT any spline_filter, so the 932MB
    # float64 alloc never happens. Small cases keep the requested high order
    # (byte-identical to before), so nothing that currently works regresses.
    _big = float(np.prod(new_shape)) > float(
        _os.environ.get("PIVOT_RESAMPLE_LINEAR_ABOVE_VOX", "80000000"))
    heavy = (order >= 2) and (not is_seg) and (not _big)

    def _ch_order(c: int) -> int:
        # CT (channel 0) keeps the high order; analytic linear side channels use order-1.
        o = order if (c == 0 or not heavy) else 1
        return min(o, 1) if _big else o

    def _do(c: int) -> None:
        out[c : c + 1] = _resample_data_or_seg_float32(
            np.asarray(data[c : c + 1], copy=False),
            new_shape,
            is_seg=is_seg,
            axis=axis,
            order=_ch_order(c),
            do_separate_z=do_separate_z,
            order_z=order_z,
            dtype_out=dtype_out,
        )

    try:
        base = int(os.environ.get("PIVOT_RESAMPLE_WORKERS", "8"))
    except ValueError:
        base = 8
    try:
        mem_budget_gb = float(os.environ.get("PIVOT_RESAMPLE_MEM_GB", "4.0"))
    except ValueError:
        mem_budget_gb = 4.0
    in_vox = float(np.prod(data.shape[1:]))
    out_vox = float(np.prod(new_shape))
    # heavy: float64 spline at INPUT res + skimage copies (~1.6x). light: output only.
    per_ch_bytes = (max(in_vox, out_vox) * 8.0 * 1.6) if heavy else (out_vox * 4.0 * 1.5)
    mem_cap = max(1, int((mem_budget_gb * 1e9) / max(1.0, per_ch_bytes)))
    workers = max(1, min(base, mem_cap, n_ch))
    if workers == 1:
        for c in range(n_ch):
            _do(c)
            gc.collect()
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_do, range(n_ch)))
        gc.collect()
    return out


def _install_frugal_argmax() -> None:
    """Replace LabelManager.convert_probabilities_to_segmentation with a chunked,
    uint8-output argmax.

    The stock implementation does ``predicted_probabilities.argmax(0)`` which numpy
    materializes as an int64 full-volume array (8 bytes/voxel). For a large pelvic
    7-class volume that single allocation is ~2.7 GiB and OOMs the GC node at export.
    The result only needs <=255 labels, so we argmax slab-by-slab into a uint8 buffer
    (8x smaller, plus only a 1/32-volume int64 temporary at a time).
    """
    import nnunetv2.utilities.label_handling.label_handling as labelmod

    label_manager_cls = labelmod.LabelManager
    if getattr(label_manager_cls, "_pengwin_frugal_argmax", False):
        return
    _orig_convert = label_manager_cls.convert_probabilities_to_segmentation

    def convert_probabilities_to_segmentation(self, predicted_probabilities):
        # region-based (sigmoid) training keeps the original threshold logic.
        if getattr(self, "has_regions", False):
            return _orig_convert(self, predicted_probabilities)
        is_numpy = isinstance(predicted_probabilities, np.ndarray)
        probs = predicted_probabilities if is_numpy else predicted_probabilities.cpu().numpy()
        n_classes = probs.shape[0]
        out_dtype = np.uint8 if n_classes <= 255 else np.uint16
        seg = np.empty(probs.shape[1:], dtype=out_dtype)
        n0 = probs.shape[1]
        step = max(1, n0 // 32)
        for a in range(0, n0, step):
            seg[a : a + step] = probs[:, a : a + step].argmax(0).astype(out_dtype, copy=False)
        gc.collect()
        return seg if is_numpy else torch.from_numpy(seg)

    label_manager_cls.convert_probabilities_to_segmentation = convert_probabilities_to_segmentation
    label_manager_cls._pengwin_frugal_argmax = True  # type: ignore[attr-defined]


def _install_inplace_softmax() -> None:
    """Make LabelManager.apply_inference_nonlin write the softmax back over its own
    input instead of allocating a second full-volume copy.

    convert_predicted_logits_to_segmentation_with_correct_shape (return_probabilities
    path) does::

        predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits)
        segmentation = label_manager.convert_probabilities_to_segmentation(...)
        del predicted_logits            # <- only AFTER both exist

    so the resampled logits and the softmax are live at the same time: 2 x 7-class
    full-res float32, ~6.1 GiB together on a 108M-voxel pelvic case. That, not the
    argmax, is the true peak of the export.

    softmax along dim 0 is independent per spatial position, so computing it slab by
    slab along dim 1 and storing each slab back in place is bit-identical to the
    stock call while holding only a 1/32-volume temporary. The returned tensor
    shares storage with the input, which makes nnUNet's later ``del
    predicted_logits`` a no-op instead of a missed free.

    Kill switch: PENGWIN_INPLACE_SOFTMAX=0.
    """
    import os as _os

    if _os.environ.get("PENGWIN_INPLACE_SOFTMAX", "1") != "1":
        return

    import nnunetv2.utilities.label_handling.label_handling as labelmod

    label_manager_cls = labelmod.LabelManager
    if getattr(label_manager_cls, "_pengwin_inplace_softmax", False):
        return
    _orig_nonlin = label_manager_cls.apply_inference_nonlin

    def apply_inference_nonlin(self, logits):
        # region-based (sigmoid) training keeps the original path.
        if getattr(self, "has_regions", False):
            return _orig_nonlin(self, logits)
        t = torch.from_numpy(logits) if isinstance(logits, np.ndarray) else logits
        # .float() in the stock version would copy anything that is not already
        # float32; let it, rather than silently changing dtype semantics.
        if t.dtype != torch.float32:
            return _orig_nonlin(self, logits)
        with torch.no_grad():
            n0 = t.shape[1]
            step = max(1, n0 // 32)
            for a in range(0, n0, step):
                # self.inference_nonlin (not a hardcoded softmax) so a configuration
                # with a different nonlinearity still gets its own function.
                t[:, a : a + step] = self.inference_nonlin(t[:, a : a + step])
        gc.collect()
        return t

    label_manager_cls.apply_inference_nonlin = apply_inference_nonlin
    label_manager_cls._pengwin_inplace_softmax = True  # type: ignore[attr-defined]


def _malloc_trim() -> None:
    """Return freed glibc-arena pages to the OS.

    ``del`` + ``gc.collect()`` hands a numpy buffer back to glibc, but glibc
    keeps multi-GiB arenas mapped, so container RSS -- which is what the GC
    cgroup kills on -- never comes down. Measured on 048 (108.5M voxels): the
    12g run OOM-killed at 11.94 GiB *during stage-2 inference*, i.e. ~2 GiB of
    stage-1 arrays freed by v57's dels were still resident. Same call the
    Task 2 harness (predict_task2.py) has shipped since v65.
    """
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass  # non-glibc platform: RSS stays as-is, correctness unaffected


# --------------------------------------------------------------- probs stash ---

_PROBS_STASH: dict = {}


def pop_stashed_probabilities():
    """Return and clear the probabilities stashed by the last export, or None."""
    if not _PROBS_STASH:
        return None
    _, probs = _PROBS_STASH.popitem()
    return probs


def _install_probs_stash() -> None:
    """Hand the export's probabilities to the caller in memory instead of
    round-tripping them through a compressed .npz.

    Stock flow with save_probabilities=True (measured on 048, 108.5M voxels):
    export builds the float32 (7,Z,Y,X) probabilities (2.96 GiB), zlib-compresses
    them into a ~2 GiB npz under /tmp, frees them -- and predict_softmax*()
    immediately np.load()s the same 2.96 GiB back. The RSS probe shows exactly
    this as the run's two peaks: 12.0->14.0 GiB at the stage-2 reload (t=412s)
    and 14.43 GiB at the voter reload (t=770s). The stash removes the reload
    copy at both peaks, the ~2 GiB tmp file (host RAM if GC mounts /tmp as
    tmpfs), and the compress+decompress wall time, while returning the very
    same array the npz would have carried.

    Patches export_prediction_from_logits in BOTH namespaces:
    nnunetv2.inference.export_prediction (definition) and
    nnunetv2.inference.predict_from_raw_data (from-import used by
    predict_from_files_sequential at its L772 call site). The sequential
    predictor calls it in-process -- no worker pools -- so the monkeypatch is
    seen. save_probabilities=False calls delegate to the original untouched.
    The .npz fallback in predict_softmax*() stays, so PENGWIN_PROBS_STASH=0
    restores v59 behavior byte-for-byte.
    """
    import os as _os

    if _os.environ.get("PENGWIN_PROBS_STASH", "1") != "1":
        return

    import nnunetv2.inference.export_prediction as ep
    import nnunetv2.inference.predict_from_raw_data as pr

    if getattr(ep, "_pengwin_probs_stash", False):
        return
    _orig_export = ep.export_prediction_from_logits

    def export_prediction_from_logits(predicted_array_or_file, properties_dict,
                                      configuration_manager, plans_manager,
                                      dataset_json_dict_or_file, output_file_truncated,
                                      save_probabilities=False,
                                      num_threads_torch=ep.default_num_processes):
        if not save_probabilities:
            return _orig_export(predicted_array_or_file, properties_dict,
                                configuration_manager, plans_manager,
                                dataset_json_dict_or_file, output_file_truncated,
                                save_probabilities, num_threads_torch)

        if isinstance(dataset_json_dict_or_file, str):
            dataset_json_dict_or_file = ep.load_json(dataset_json_dict_or_file)
        label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
        segmentation_final, probabilities_final = \
            ep.convert_predicted_logits_to_segmentation_with_correct_shape(
                predicted_array_or_file, plans_manager, configuration_manager,
                label_manager, properties_dict,
                return_probabilities=True, num_threads_torch=num_threads_torch,
            )
        del predicted_array_or_file

        _PROBS_STASH.clear()
        _PROBS_STASH[output_file_truncated] = probabilities_final
        # stock export also writes the properties pickle next to the npz; keep it
        # for anything downstream that globs for it.
        ep.save_pickle(properties_dict, output_file_truncated + '.pkl')

        rw = plans_manager.image_reader_writer_class()
        rw.write_seg(segmentation_final,
                     output_file_truncated + dataset_json_dict_or_file['file_ending'],
                     properties_dict)

    ep.export_prediction_from_logits = export_prediction_from_logits
    pr.export_prediction_from_logits = export_prediction_from_logits
    ep._pengwin_probs_stash = True  # type: ignore[attr-defined]


def apply() -> None:
    _install_frugal_argmax()
    _install_inplace_softmax()
    _install_probs_stash()

    import nnunetv2.preprocessing.resampling.default_resampling as resampling

    if getattr(resampling, "_pengwin_float32_patch", False):
        return

    _orig = resampling.resample_data_or_seg

    def resample_data_or_seg(
        data: np.ndarray,
        new_shape: Union[Tuple[float, ...], List[float], np.ndarray],
        is_seg: bool = False,
        axis: Union[None, int] = None,
        order: int = 3,
        do_separate_z: bool = False,
        order_z: int = 0,
        dtype_out=None,
    ) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        assert data.ndim == 4

        if is_seg or data.shape[0] <= 1:
            return _resample_data_or_seg_float32(
                data,
                new_shape,
                is_seg=is_seg,
                axis=axis,
                order=order,
                do_separate_z=do_separate_z,
                order_z=order_z,
                dtype_out=dtype_out,
            )

        return _resample_multichannel_lowmem(
            data,
            new_shape,
            is_seg=is_seg,
            axis=axis,
            order=order,
            do_separate_z=do_separate_z,
            order_z=order_z,
            dtype_out=dtype_out,
        )

    resample_data_or_seg._pengwin_float32_patch = True  # type: ignore[attr-defined]
    resampling.resample_data_or_seg = resample_data_or_seg
