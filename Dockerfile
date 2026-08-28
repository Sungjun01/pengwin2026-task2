# PENGWIN 2026 — Task 2 (Interactive Fragment Segmentation)
# Final submission: pengwin-task2-v75  (build-id d308ad623da4fc1f62176aa7e529d649)
#
# This is a FLATTENED, self-contained rebuild of the container that was submitted to
# Grand Challenge. The submitted image was produced as a chain of thin incremental
# layers (v63 -> v65 -> v67 -> v70 -> v71 -> v72 -> v73 -> v75); every file, every
# environment variable and the entrypoint below were read back out of that final
# image, so a build from this single Dockerfile is equivalent to it.
#
# Build:  ./build.sh
# Run  :  ./run_case.sh <input_dir> <output_dir>     (Grand Challenge interface)

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

LABEL maintainer="schoaq <sungjunc01@gmail.com>"
LABEL description="PENGWIN 2026 Task 2 — final submission (v75), flattened self-contained build"
LABEL version="v75"

# --------------------------------------------------------------- system deps --
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

WORKDIR /opt/algorithm

# ------------------------------------------------------------------ python ----
# Pins are the versions actually present in the submitted image
# (full `pip freeze` of that image: docs/pip-freeze-v75.txt).
COPY requirements.txt /opt/algorithm/requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r /opt/algorithm/requirements.txt

# ------------------------------------------------------------------- source ---
# src/ mirrors /opt/algorithm of the submitted image (scripts, evaluation,
# phase_b1_cascade, model/femur_merge_model.json).
COPY src/ /opt/algorithm/

ENV nnUNet_raw="/opt/algorithm/nnUNet_raw" \
    nnUNet_preprocessed="/opt/algorithm/nnUNet_preprocessed" \
    nnUNet_results="/opt/algorithm/nnUNet_results" \
    PYTHONPATH="/opt/algorithm"

# ---------------------------------------------------------------- trainers ----
# nnU-Net resolves a checkpoint's trainer class by walking the FILESYSTEM under
# nnunetv2/training/nnUNetTrainer, so the classes have to exist as .py files inside
# that package. trainers/*.py are thin modules; the implementations live in
# src/scripts/trainers/ and are imported from there (no second copy that can drift).
COPY trainers/ /tmp/trainers/
RUN NNT=$(python -c "import nnunetv2, os; print(os.path.join(os.path.dirname(nnunetv2.__file__),'training/nnUNetTrainer'))") && \
    cp /tmp/trainers/nnUNetTrainerBorderWeighted.py          "$NNT/variants/" && \
    cp /tmp/trainers/nnUNetTrainerBorderWeightedAseg.py      "$NNT/variants/" && \
    cp /tmp/trainers/nnUNetTrainerBorderWeightedAseg_w40.py  "$NNT/variants/" && \
    cp /tmp/trainers/_pengwin_femur_trainer.py               "$NNT/" && \
    rm -rf /tmp/trainers && \
    ls -1 "$NNT/variants/" | grep BorderWeighted

# ----------------------------------------------------------------- weights ----
# Not in git (≈942 MB). Run ./fetch_weights.sh first — see README.md.
COPY models/ /opt/algorithm/nnUNet_results/

# ------------------------------------------------------------- pipeline env ---
# Model selection
ENV PIVOT_PELVIC_DATASET="Dataset016_PENGWIN_BorderCore_MedialOrient" \
    PIVOT_PELVIC_TRAINER_PLANS="nnUNetTrainerBorderWeightedAseg_w40__nnUNetPlans__3d_fullres" \
    PIVOT_PELVIC_FOLDS="all" \
    PIVOT_FEMUR_DATASET="Dataset004_PENGWIN_FemurAnatomy" \
    PIVOT_FEMUR_TRAINER_PLANS="nnUNetTrainer__nnUNetPlans__3d_fullres" \
    PIVOT_FEMUR_INSTANCE_METHOD="watershed"

# Instance assembly / decode
ENV PIVOT_ASSEMBLY="1" \
    PIVOT_ASSEMBLY_FEMUR_CHUNKS="1" \
    PIVOT_ASSEMBLY_MAX_SLIVER_MM3="2000" \
    PIVOT_ASSEMBLY_MIN_VOL_RATIO="5" \
    PIVOT_ASSEMBLY_DILATION_ITERS="1" \
    PIVOT_ASSEMBLY_MAX_DIST_MM="30" \
    PENGWIN_MIN_VOLUME_MM3="500" \
    PIVOT_V19_DECODE="0"

# Inference / memory budget (tuned for the Grand Challenge 16 GB host-RAM limit)
ENV PIVOT_TILE_STEP_SIZE="1.0" \
    PENGWIN_NNUNET_ON_DEVICE="1" \
    PIVOT_RESAMPLE_WORKERS="1" \
    PIVOT_RESAMPLE_MEM_GB="3.0" \
    PENGWIN_PROBS_STASH="1" \
    PIVOT_RESAMPLE_LINEAR_ABOVE_VOX="999000000000"

# Click-guided post-processing / border-seam split
ENV PENGWIN_CLICK_RECENTER="1" \
    PENGWIN_SEED_ANCHOR_R="3" \
    PENGWIN_STARVE_FLOOR="26" \
    PENGWIN_SEAM_ANATS="sacrum,leftHip,rightHip,femur" \
    PENGWIN_BORDER_PURE_SACRUM="1" \
    PENGWIN_BORDER_PURE_HIP="0" \
    PENGWIN_BORDER_PURE_FEMUR="1" \
    PENGWIN_BORDER_LAM="3.0" \
    PENGWIN_BORDER_MAX_VOXELS="150000000" \
    PENGWIN_BORDER_FP16_ABOVE="90000000" \
    PENGWIN_BORDER_DS="2" \
    PENGWIN_FEMUR_BORDER="1" \
    PENGWIN_FEMUR_BORDER_MAX_VOXELS="110000000"

# Anatomy router + v73 click-seeded femur decode
ENV PENGWIN_ROUTER="hybrid" \
    PIVOT_FEMUR_V73="1" \
    PIVOT_FEMUR_MERGE_MODEL="/opt/algorithm/model/femur_merge_model.json"

# ------------------------------------------------------------------ runtime ---
RUN mkdir -p /input/images/peripelvic-fracture-ct \
             /output/images/peripelvic-fracture-ct-segmentation \
             /opt/ml/model && \
    chown -R algorithm:algorithm /opt/algorithm /opt/ml /input /output

USER algorithm
ENTRYPOINT ["python", "/opt/algorithm/scripts/predict_task2.py"]
