"""Spacing-aware CT-derived signal channels for SAFT experiments."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def _safe_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if float(np.max(np.abs(values))) <= 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values / (1.0 + np.abs(values))).astype(np.float32)


def build_signal_channels(
    image_zyx: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    sigma_mm: float = 0.75,
) -> np.ndarray:
    """Return gradient, structure-coherence, and Hessian-sheet CT channels."""
    image = np.asarray(image_zyx, dtype=np.float32)
    spacing = np.asarray(spacing_zyx, dtype=np.float32)
    if image.ndim != 3:
        raise ValueError(f"image_zyx must be 3D, got shape {image.shape}")
    if spacing.shape != (3,) or np.any(spacing <= 0):
        raise ValueError(f"spacing_zyx must contain three positive values, got {spacing_zyx}")
    if float(sigma_mm) <= 0:
        raise ValueError(f"sigma_mm must be positive, got {sigma_mm}")

    sigma_vox = tuple(float(sigma_mm / axis_spacing) for axis_spacing in spacing)
    smoothed = gaussian_filter(image, sigma=sigma_vox, mode="nearest")
    grad_z, grad_y, grad_x = np.gradient(smoothed, *spacing.tolist(), edge_order=1)
    gradient_magnitude = np.sqrt(grad_z * grad_z + grad_y * grad_y + grad_x * grad_x)

    tensor_zz = gaussian_filter(grad_z * grad_z, sigma=sigma_vox, mode="nearest")
    tensor_yy = gaussian_filter(grad_y * grad_y, sigma=sigma_vox, mode="nearest")
    tensor_xx = gaussian_filter(grad_x * grad_x, sigma=sigma_vox, mode="nearest")
    tensor_trace = tensor_zz + tensor_yy + tensor_xx
    dominant_axis_energy = np.maximum(np.maximum(tensor_zz, tensor_yy), tensor_xx)
    coherence = np.divide(
        dominant_axis_energy,
        tensor_trace + 1e-6,
        out=np.zeros_like(tensor_trace, dtype=np.float32),
        where=tensor_trace > 0,
    )

    hessian_zz = np.gradient(grad_z, float(spacing[0]), axis=0, edge_order=1)
    hessian_yy = np.gradient(grad_y, float(spacing[1]), axis=1, edge_order=1)
    hessian_xx = np.gradient(grad_x, float(spacing[2]), axis=2, edge_order=1)
    hessian_sheet = np.abs(hessian_zz) + np.abs(hessian_yy) + np.abs(hessian_xx)

    return np.stack(
        (
            _safe_scale(gradient_magnitude),
            _safe_scale(coherence),
            _safe_scale(hessian_sheet),
        ),
        axis=0,
    ).astype(np.float32)
