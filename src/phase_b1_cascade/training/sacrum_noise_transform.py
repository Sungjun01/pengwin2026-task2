"""Sacrum channel noise injection transform (mechanism C2).

nnUNet 2.7 의 batchgeneratorsv2.BasicTransform API 에 맞춰진 transform.
Model_B 의 sacrum input channel (channel 1, P_sacrum) 에 학습 시 morphological
변형 + regional dropout 노이즈를 주입해 Model_A 의 imperfect prediction 에 대한
covariate shift 강건성을 키운다.

데이터 분석. progress_report_16 §2 의 sacrum CC 분포 — Intact 70% / Moderate 19%
/ Severe 11% — 가 default noise parameter 의 근거. radius 1-3 voxel 의 dilation
/ erosion 이 inference 시 흔히 보는 boundary 흔들림을 simulate.

Reference:
    progress_report_16 §3 [권장] C2 Noise-Injected Anchor Training
"""
from __future__ import annotations

import numpy as np
import torch
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from scipy.ndimage import binary_dilation, binary_erosion


class SacrumNoiseTransform(BasicTransform):
    """Image tensor 의 sacrum channel 에 morphological + regional dropout 노이즈.

    Args:
        sacrum_channel_idx: Image tensor 의 sacrum 채널 인덱스. Dataset007 기준 1.
        p_apply: 샘플당 전체 transform 적용 확률.
        p_morph: apply 가 True 일 때 morphological 변형을 적용할 조건부 확률.
        p_dropout: apply 가 True 일 때 regional dropout 을 적용할 조건부 확률.
        max_radius: morphological dilation/erosion 의 최대 반경 (voxel).
        dropout_bbox_only: True 면 sacrum bounding box 안에서만 dropout, False 면
            전체 volume 범위에서.
        preserve_soft_channel: True 면 노이즈 적용 후 원본 sigmoid 확률을 binary
            mask 에 곱해 보존, False 면 binary float 으로 저장 (0/1).
        prob_threshold: sigmoid 확률맵을 binary mask 로 만들 때의 threshold.
    """

    def __init__(
        self,
        sacrum_channel_idx: int = 1,
        p_apply: float = 0.5,
        p_morph: float = 0.5,
        p_dropout: float = 0.5,
        max_radius: int = 3,
        dropout_bbox_only: bool = True,
        preserve_soft_channel: bool = False,
        prob_threshold: float = 0.5,
        dropout_voxel_p: float = 0.3,
    ):
        super().__init__()
        self.sacrum_channel_idx = sacrum_channel_idx
        self.p_apply = p_apply
        self.p_morph = p_morph
        self.p_dropout = p_dropout
        self.max_radius = max_radius
        self.dropout_bbox_only = dropout_bbox_only
        self.preserve_soft_channel = preserve_soft_channel
        self.prob_threshold = prob_threshold
        self.dropout_voxel_p = dropout_voxel_p

    def get_parameters(self, **data_dict) -> dict:
        if np.random.uniform() > self.p_apply:
            return {"apply": False}
        return {
            "apply": True,
            "do_morph": np.random.uniform() < self.p_morph,
            "do_dropout": np.random.uniform() < self.p_dropout,
            "morph_is_dilation": np.random.uniform() < 0.5,
            "radius": int(np.random.randint(1, max(2, self.max_radius + 1))),
        }

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        if not params.get("apply"):
            return img
        c = self.sacrum_channel_idx
        if c >= img.shape[0]:
            return img

        original = img[c].detach().cpu().numpy()
        binary = original > self.prob_threshold
        if not binary.any():
            return img

        modified = binary.copy()

        if params["do_morph"]:
            struct = np.ones((3, 3, 3), dtype=bool)
            radius = params["radius"]
            if params["morph_is_dilation"]:
                for _ in range(radius):
                    modified = binary_dilation(modified, structure=struct)
            else:
                for _ in range(radius):
                    modified = binary_erosion(modified, structure=struct)

        if params["do_dropout"]:
            modified = self._apply_dropout(modified, binary)

        if self.preserve_soft_channel:
            new_channel = original * modified.astype(np.float32)
        else:
            new_channel = modified.astype(np.float32)

        img = img.clone()
        img[c] = torch.from_numpy(new_channel).to(dtype=img.dtype, device=img.device)
        return img

    def _apply_to_segmentation(self, seg, **params):
        return seg

    def _apply_to_regr_target(self, regression_target, **params):
        return regression_target

    def _apply_to_keypoints(self, keypoints, **params):
        return keypoints

    def _apply_to_bbox(self, bbox, **params):
        return bbox

    def _apply_dropout(self, modified: np.ndarray, original_binary: np.ndarray) -> np.ndarray:
        target = self._bbox_slice(original_binary) if self.dropout_bbox_only else tuple(slice(None) for _ in modified.shape)
        if target is None:
            return modified
        sub = modified[target]
        if sub.size == 0 or not sub.any():
            return modified
        dropout_mask = np.random.uniform(size=sub.shape) < self.dropout_voxel_p
        new_sub = sub & ~dropout_mask
        modified = modified.copy()
        modified[target] = new_sub
        return modified

    @staticmethod
    def _bbox_slice(mask: np.ndarray):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0) + 1
        return tuple(slice(int(mn), int(mx)) for mn, mx in zip(mins, maxs))
