"""Custom nnUNet trainer for Phase B-1 sacrum cascade Model_B.

Phase B-1 sacrum cascade 의 Model_B (Dataset007: bg/sacrum/LH/RH with sacrum
input channel) 학습 시 사용. SacrumNoiseTransform 을 학습 augmentation pipeline
의 끝에 통합하여 Model_A 의 imperfect prediction 에 대한 covariate shift 강건성을
학습한다.

사용법.
    PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH nnUNetv2_train 7 3d_fullres all \\
        -tr nnUNetTrainerSacrumCascade

주의.
    Sacrum channel 인덱스는 Dataset007 의 channel_names 순서로 결정된다.
    Dataset007 의 channel_names = {"0": "CT", "1": "P_sacrum"} 이라면 idx=1.

    nnUNet 2.7 의 batchgeneratorsv2 API 기준으로 작성. nnUNet major version 이
    오르면 get_training_transforms 시그니처 + ComposeTransforms return type 을
    다시 점검.

Reference.
    progress_report_16 §3 [권장] C2 Noise-Injected Anchor Training
    docs/specs/2026-05-16-phase-b1-sacrum-cascade-design.md §3-4, §6-2
"""
from __future__ import annotations

from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from phase_b1_cascade.training.sacrum_noise_transform import SacrumNoiseTransform


class nnUNetTrainerSacrumCascade(nnUNetTrainer):
    """Sacrum channel noise injection 을 추가한 Model_B 학습용 trainer.

    Class attribute 로 noise 강도를 조정 가능. 학습 시 subclass override 가능.
    """

    SACRUM_CHANNEL_IDX: int = 1
    NOISE_P_APPLY: float = 0.5
    NOISE_P_MORPH: float = 0.5
    NOISE_P_DROPOUT: float = 0.5
    NOISE_MAX_RADIUS: int = 3
    NOISE_DROPOUT_BBOX_ONLY: bool = True
    NOISE_PRESERVE_SOFT: bool = False

    @staticmethod
    def get_training_transforms(
        patch_size,
        rotation_for_DA,
        deep_supervision_scales,
        mirror_axes,
        do_dummy_2d_data_aug,
        use_mask_for_norm=None,
        is_cascaded: bool = False,
        foreground_labels=None,
        regions=None,
        ignore_label=None,
    ) -> BasicTransform:
        base = nnUNetTrainer.get_training_transforms(
            patch_size=patch_size,
            rotation_for_DA=rotation_for_DA,
            deep_supervision_scales=deep_supervision_scales,
            mirror_axes=mirror_axes,
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm,
            is_cascaded=is_cascaded,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label,
        )

        noise = SacrumNoiseTransform(
            sacrum_channel_idx=nnUNetTrainerSacrumCascade.SACRUM_CHANNEL_IDX,
            p_apply=nnUNetTrainerSacrumCascade.NOISE_P_APPLY,
            p_morph=nnUNetTrainerSacrumCascade.NOISE_P_MORPH,
            p_dropout=nnUNetTrainerSacrumCascade.NOISE_P_DROPOUT,
            max_radius=nnUNetTrainerSacrumCascade.NOISE_MAX_RADIUS,
            dropout_bbox_only=nnUNetTrainerSacrumCascade.NOISE_DROPOUT_BBOX_ONLY,
            preserve_soft_channel=nnUNetTrainerSacrumCascade.NOISE_PRESERVE_SOFT,
        )

        if isinstance(base, ComposeTransforms):
            base.transforms.append(noise)
            return base
        return ComposeTransforms([base, noise])
