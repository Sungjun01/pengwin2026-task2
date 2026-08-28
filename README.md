# PENGWIN 2026 — Task 2 (Interactive Fragment Segmentation)

Final submission of team **schoaq**, container `pengwin-task2-v75`
(build-id `d308ad623da4fc1f62176aa7e529d649`).

This repository contains everything needed to rebuild and run that container:
a single self-contained `Dockerfile`, a `build.sh`, the inference source, the
custom nnU-Net trainer classes, and a download link for the model weights.

> **Note on `v75`.** The image submitted to Grand Challenge was built as a chain of
> thin incremental layers (`v63 → v65 → v67 → v70 → v71 → v72 → v73 → v75`), each
> `FROM` the previous one. `v75` itself is a *content-identical re-issue* of `v73`
> — it adds a single `BUILD_ID.txt` file and nothing else, because Grand Challenge
> refuses an upload whose image digest is already registered. The `Dockerfile`
> here is that whole chain **flattened into one file**: every source file, every
> environment variable, the user and the entrypoint were read back out of the
> submitted image, so a build from this repository is equivalent to it.

---

## Quick start

```bash
git clone <this-repo> && cd pengwin2026-task2-release

./fetch_weights.sh            # downloads + verifies ./models/ (≈900 MB)
./build.sh                    # -> pengwin-task2-v75:latest
./run_case.sh  test/input  test/output
```

`./build.sh --save` additionally writes `pengwin-task2-v75.tar.gz`, the artifact
that is uploaded to Grand Challenge.

Requirements: Docker with the NVIDIA container runtime, one GPU, ~30 GB free disk
(the built image is ≈7.2 GB).

---

## Grand Challenge interface

The container reads and writes exactly the paths used by the official Task 2
interface, and does so with no network access:

| | path |
|---|---|
| Input — CT | `/input/images/peripelvic-fracture-ct/<UUID>.mha` (or `.tif`) |
| Input — clicks | `/input/peripelvic-fragment-clicks.json` |
| Output — segmentation | `/output/images/peripelvic-fracture-ct-segmentation/<UUID>.mha` |

Entrypoint: `python /opt/algorithm/scripts/predict_task2.py`. It processes every
case found under `/input`, so it works both for the single-case Grand Challenge
job and for a local batch.

Click JSON: points are accepted in **either voxel index or physical (mm) world
coordinates** — the resolver tries both and keeps the interpretation that lands
inside the bone mask. Grand Challenge was observed to send world coordinates, and
this fallback is what makes the submission robust to that.

Running it by hand, without the helper script:

```bash
docker run --rm --gpus all --network none --shm-size 4g --memory 16g \
    -v /abs/path/to/input:/input:ro \
    -v /abs/path/to/output:/output \
    pengwin-task2-v75:latest
```

`--memory 16g` mirrors the Grand Challenge host-RAM budget; the pipeline is tuned
to stay inside it (see *Memory* below).

---

## Model weights

Not in git. Archive `pengwin-task2-v75-weights.tar.gz` (873 MB compressed,
939 MB unpacked):

- **Download:** `<PUT DOWNLOAD LINK HERE>`
- **SHA256:** `a1b2eaa6ce058cd0959135bb6ee0cc2d4bb937b94a88267e7051352420f62bbf`

```bash
# place the archive next to fetch_weights.sh, or:
WEIGHTS_URL=<direct-link> ./fetch_weights.sh
```

`fetch_weights.sh` unpacks into `./models/` and verifies every file against
`models.md5` (16 files), which was generated from the submitted image itself.

### Or: skip the build entirely

If it is more convenient to run the submitted container directly rather than
rebuild it, the exact submitted image is also available as a `docker load`
archive:

- **Download:** `<PUT IMAGE DOWNLOAD LINK HERE>`
- `pengwin-task2-v75-image.tar.xz` — 2.6 GB
- **SHA256:** `8de65602f4c840a40c7caa3c01d6953d335250dfdf508b5047aeffdfcc330d21`

```bash
docker load -i pengwin-task2-v75-image.tar.xz     # -> pengwin-task2-v75:latest
./run_case.sh <input_dir> <output_dir>
```

The image already contains the weights, so `fetch_weights.sh` and `build.sh` are
not needed on this path. Its config digest is
`sha256:2813fb3341856a6b50e38d09f03b49be0fe845525ce8abc81aef17cfacdc9815`, which
is the image registered on Grand Challenge as our final submission.

Four nnU-Net models, all `3d_fullres`, all `fold_all`:

| dataset | trainer | role |
|---|---|---|
| `Dataset016_PENGWIN_BorderCore_MedialOrient` | `nnUNetTrainerBorderWeightedAseg_w40` | pelvic border-core segmentation (sacrum + both hips) |
| `Dataset006_PENGWIN_SacrumOnly` | `nnUNetTrainer` | sacrum binary stage-1 (LPS frame) |
| `Dataset004_PENGWIN_FemurAnatomy` | `nnUNetTrainer` | femur anatomy |
| `Dataset013_PENGWIN_BorderCoreFemur` | `nnUNetTrainerBorderWeightedFemurSnap` | femur border-core / fracture-surface |

---

## Method

Two stages: an automatic instance segmentation, then a click-guided correction.

**1. Routing.** A hybrid router decides pelvic vs. femur from the CT field of view,
with the organizers' decision tree as a fallback (`scripts/classify_case_type.py`).
The FOV rule was added because the tree alone misroutes a non-trivial share of
cases, and a misrouted case scores zero.

**2. Automatic segmentation** (`scripts/inference_v7_d015.py`). Anatomy-wise
border-core segmentation: the network predicts, per bone, a *core* and a *border*
class; instances are recovered by seeded watershed on the cores with the predicted
border acting as the fracture surface. Pelvic and femur run separate model paths,
with a sacrum-specific stage-1 in the LPS frame. Sliver fragments below
500 mm³ are absorbed into their neighbour.

**3. Click-guided post-processing** (`scripts/click_guided_postproc.py`). The
clicks give the true number and location of fragments, and are used to
(a) select and relabel the fragments the user marked, (b) **split** a predicted
component that carries two clicks, cutting along the learned border surface rather
than a Euclidean-distance-transform plane, and (c) drop components no click
supports. Splitting along the predicted fracture surface, rather than a geometric
plane, is the single largest gain in the pipeline: it directly attacks the
merge errors that dominate the error budget.

**4. Femur decode, click-seeded** (`scripts/femur_decode_v73.py`). The femur
foreground is taken from the `D013` argmax (already computed for the seam, so no
extra inference), and the watershed seeds come **directly from the clicks**. This
replaced an older automatic seeding that was, measurably, worse than the
click-free Task 1 pipeline — the clicks were masking how weak the underlying
segmentation was. Measured on the 170 training femur cases, case-wise 5-fold OOF
with the model actually deployed here, instance F1 went **0.9081 → 0.9714**.
One exception is handled explicitly: for the `boundary_internal_margin` click
strategy the seeds sit near the watershed ridge and leak into the neighbouring
fragment, so that strategy alone keeps h-maxima seeding. The strategy name is
given in the input JSON, so this branch uses provided information, not ground truth.
Fragment merging uses a 15-coefficient logistic regression stored as JSON
(`src/model/femur_merge_model.json`, AUC 0.9489) — no extra dependency.

**Memory.** Grand Challenge caps host RAM at 16 GB, and large cases
(>100 M voxels) overran it. The pipeline uses a frugal argmax over the softmax
volumes, an in-memory probability stash instead of round-tripping through `.npz`,
fp16 border probabilities above 90 M voxels, single-worker resampling, and
`malloc_trim(0)` to hand freed glibc arena pages back to the cgroup
(`scripts/nnunet_preprocess_memory_patch.py`, and `_malloc_trim` in
`scripts/predict_task2.py`).

**Robustness.** Every case is wrapped in `try/except`. If click post-processing
fails, the raw automatic instance map is emitted; if everything fails, a
background volume is written, so an output file always exists for Grand Challenge
to read.

---

## Verifying this rebuild against the submitted image

The flattened build was checked against the actual submitted image
`pengwin-task2-v75` on a femur case (`t2_277`, `boundary_internal_margin` click
strategy, i.e. the branch that exercises the v73 femur decode), run under the
Grand Challenge constraints (`--network none --memory 16g`, one GPU):

| comparison | differing voxels (of 29,753,344) |
|---|---|
| submitted image, run 1 vs. run 2 | 3 |
| submitted image run 1 vs. this build | 3 |
| submitted image run 2 vs. this build | **0** |

Both produce the same 5 fragments with identical labels, geometry and per-label
voxel counts. The 3-voxel band is GPU run-to-run nondeterminism — the submitted
image differs from *itself* by exactly the same amount — not a difference between
the two builds.

Model weights were verified separately: all 16 files under `models/` match the
files inside the submitted image bit for bit (`models.md5`).

An example click JSON in the Grand Challenge format is at
`docs/example-peripelvic-fragment-clicks.json`.

---

## Repository layout

```
Dockerfile              self-contained build of the submitted image
build.sh                build (and optionally `docker save`) the container
run_case.sh             run one case through the Grand Challenge interface
fetch_weights.sh        download + checksum the model weights
requirements.txt        pinned python deps (as installed in the submitted image)
models.md5              checksums of the 16 weight files
src/                    -> /opt/algorithm  in the image
  scripts/              inference code (entrypoint: scripts/predict_task2.py)
  scripts/trainers/     custom nnU-Net trainer implementations
  evaluation/           PENGWIN metrics (instance F1, IoU, HD95) used for our own gating
  phase_b1_cascade/     sacrum cascade trainer + transforms
  model/                femur_merge_model.json (logistic merge classifier)
trainers/               thin re-export modules injected into nnunetv2's package
docs/pip-freeze-v75.txt full `pip freeze` of the submitted image
```

`src/scripts/` is our full research tree; only part of it runs at inference time.
The deployed import closure of `scripts/predict_task2.py` is:

```
predict_task2.py, inference_v7_d015.py, click_guided_postproc.py,
femur_decode_v73.py, classify_case_type.py, pivot_assembly.py,
pivot_form_postproc.py, v19_decode.py, nnunet_preprocess_memory_patch.py,
affinity_labels.py, femur_boundary_refine.py, form_learn_edge.py,
form_learn_features.py, form_learn_postproc.py,
measure_border_core_recovery.py, mutex_watershed_decode.py,
public003_sacrum_completion.py, surface_completion_decode.py,
surface_refine.py, topology_locked_refine.py,
evaluation/{io_utils,label_schema,matching,metrics,pengwin_eval}.py
```

Everything else under `src/scripts/` is training, evaluation and ablation code,
kept so the results above can be reproduced.

### One thing worth stating plainly

`scripts/inference_v7_d015.py` contains two experimental branches keyed on the
SHA256 of specific *public* validation cases (`PIVOT_PUBLIC001_V20_SLIVER`,
`PIVOT_PUBLIC003_SACRUM_COMPLETION`). They were used during development to probe
two known failure modes. **Both default to off and neither is enabled in the
submitted container** — you can confirm this in the `ENV` block of the
`Dockerfile`, which is a faithful transcription of the submitted image's
configuration. They are dead code in the submission and are left in place only so
that this repository matches the submitted image byte for byte.

---

## Custom nnU-Net trainers

`Dataset016` and `Dataset013` were trained with border-weighted trainers that are
not part of nnU-Net. Because nnU-Net resolves a checkpoint's trainer class by
walking the *filesystem* under `nnunetv2/training/nnUNetTrainer/`, the classes
must exist as `.py` files inside that installed package — the `Dockerfile` copies
`trainers/*.py` there. Those are thin modules; the implementations live in
`src/scripts/trainers/` and are imported from there, so there is no second copy
that can drift.

---

## Environment

Base image `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` (Python 3.11.10,
torch 2.5.1+cu121, numpy 2.1.2), plus `nnunetv2==2.7.0`, `monai==1.5.0`,
`SimpleITK==2.5.5`, `scipy==1.17.1`, `scikit-image==0.26.0`,
`connected-components-3d==3.28.0`. Full list in `docs/pip-freeze-v75.txt`.

## Contact

schoaq — <schoaq@connect.ust.hk>
tjeong — <tjeong@connect.ust.hk>
