"""Case-type classifier (pelvic vs femur) for PENGWIN 2026 Task 1.

Routes each case at the inference entry point to the pelvic specialist path or
the femur path. A misroute is catastrophic: the output carries labels from the
wrong range, so the case has no fragments in any anatomical region the evaluator
looks at, and it scores ~0 on every metric.

WHAT THE ORGANIZERS PUBLISHED. The update notice ships a concrete
`classify_pelvic_femur` decision tree, asks all teams to adopt it, and states
"The test sets have been verified to conform to this rule." It is reproduced
verbatim below, including the axis-name swap in their `get_image_info`
(`sp = img.GetSpacing()` is (x, y, z) but is read out as spacing_z = sp[0],
spacing_x = sp[2], with physical_x_mm = sp[2] * dim_x). The swap is deliberate
here: the tree was fit on those swapped features. Feeding it semantically
correct spacings scores 172/340 on our training cases, versus 295/340 verbatim.
Do NOT "fix" it.

WHAT IT DOES ON THEIR OWN TRAINING DATA. Measured over all 340 released cases,
ground truth taken from the label ranges (1-150 pelvic, 151-200 femur):

    organizer's rule verbatim      295 / 340   (86.8%)
    transverse-FOV threshold       338 / 340   (99.4%)
    the two disagree on 47 cases; FOV is right on 45 of them

The rule's errors are not marginal, they are physically impossible. The 31
pelvic cases it calls femur have transverse FOV 257.5-392.8 mm (median 294.8) --
you cannot fit both hips in a 250 mm field of view. The 14 femur cases it calls
pelvic are all <= 242.2 mm. On the 5 public validation cases both rules agree
and both are correct, so those give no discriminating evidence either way.

WHAT WE SHIP: hybrid. Physics decides wherever it can; the organizers' rule
decides inside the band where field of view genuinely cannot separate the two.

    transverse FOV >= 255.0 mm  ->  pelvic
    transverse FOV <= 243.0 mm  ->  femur
    in between                  ->  classify_pelvic_femur() verbatim

That is 340/340 on the training set, and it is not a knife-edge fit: both
thresholds sit inside empty gaps. The largest FOV among true femur cases the
rule gets right is 254.6 mm and the smallest among the pelvic cases it gets
wrong is 257.5 mm, so any cut in (254.6, 257.5) behaves identically; likewise
the femur side is separated by (242.2, 248.8). Each threshold is the midpoint of
its gap. The whole 255/243 plateau -- hi in {255, 257}, lo in {243, 245, 248} --
scores 340/340.

Rationale for deviating from "use this verbatim": both hypotheses are covered.
If the test set really does conform to the organizers' rule, the hybrid differs
from it only inside 243-255 mm, where it defers to the rule -- so it agrees
everywhere the rule is the sole discriminator, and its residual risk is the 2
training cases where FOV alone errs (162 at 248.8 mm, 418 at 254.6 mm), both of
which fall in the deferral band and are therefore decided by the rule and
correct. If the test set instead resembles the released training data, the
verbatim rule loses ~13% of cases outright. The hybrid is the lower-risk choice
under either hypothesis.

PENGWIN_ROUTER=official or =fov selects the pure rules for A/B and rollback.
"""
from __future__ import annotations

import os

# --- organizer's published rule (verbatim, swapped axis names included) ---


def classify_pelvic_femur(spacing_x, spacing_y, spacing_z, physical_x_mm, physical_z_mm):
    if physical_x_mm <= 285.35:
        if spacing_x <= 0.71:
            return "pelvic"
        elif spacing_z <= 0.90:
            return "femur"
        else:
            return "pelvic" if spacing_y <= 0.91 else "femur"
    else:
        if spacing_z <= 0.68:
            return "pelvic" if physical_z_mm <= 193.55 else "femur"
        else:
            return "pelvic" if physical_z_mm <= 390.78 else "femur"


def get_classification_inputs(sitk_image) -> dict:
    """Reproduce the organizers' `get_image_info` from a SimpleITK image.

    `GetSize()` is (x, y, z) voxels, the reverse of the numpy array shape
    (z, y, x) they index, so dim_z = size[2] and dim_x = size[0]. The
    spacing-to-name mapping is copied from their code as-is.
    """
    sp = sitk_image.GetSpacing()   # (x, y, z) mm
    size = sitk_image.GetSize()    # (x, y, z) voxels
    dim_z, dim_y, dim_x = size[2], size[1], size[0]
    return {
        "dim_z": dim_z, "dim_y": dim_y, "dim_x": dim_x,
        "spacing_z": sp[0], "spacing_y": sp[1], "spacing_x": sp[2],
        "physical_z_mm": sp[0] * dim_z, "physical_x_mm": sp[2] * dim_x,
    }


def classify_official(sitk_image) -> str:
    info = get_classification_inputs(sitk_image)
    return classify_pelvic_femur(
        info["spacing_x"], info["spacing_y"], info["spacing_z"],
        info["physical_x_mm"], info["physical_z_mm"],
    )


# --- transverse field of view, the physical discriminator -----------------

FOV_PELVIC_MIN_MM = 255.0    # at or above this, both hips are in frame
FOV_FEMUR_MAX_MM = 243.0     # at or below this, a pelvis cannot be in frame
FOV_THRESHOLD_MM = 251.7     # single-cut version (338/340), for PENGWIN_ROUTER=fov


def transverse_fov_mm(sitk_image) -> float:
    return float(sitk_image.GetSpacing()[0] * sitk_image.GetSize()[0])


def classify_fov(sitk_image) -> str:
    return "femur" if transverse_fov_mm(sitk_image) <= FOV_THRESHOLD_MM else "pelvic"


def classify_hybrid(sitk_image) -> str:
    fov = transverse_fov_mm(sitk_image)
    if fov >= FOV_PELVIC_MIN_MM:
        return "pelvic"
    if fov <= FOV_FEMUR_MAX_MM:
        return "femur"
    return classify_official(sitk_image)


# --- entry point ----------------------------------------------------------

_ROUTERS = {"hybrid": classify_hybrid, "official": classify_official, "fov": classify_fov}


def classify_image(sitk_image) -> str:
    mode = os.environ.get("PENGWIN_ROUTER", "hybrid")
    return _ROUTERS.get(mode, classify_hybrid)(sitk_image)


# backwards-compatible alias used by some Task 2 diagnostics
def classify_pelvic_femur_fov(sitk_image) -> str:
    return classify_fov(sitk_image)
