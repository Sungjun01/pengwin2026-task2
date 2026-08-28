import os, sys; sys.path.insert(0, ".")
import numpy as np, SimpleITK as sitk
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scripts.border_core_conversion import convert_to_border_core, PELVIC_ANATOMY

NRAW="/home/schoaq/taehee/nnUNet_data/raw/Dataset012_PENGWIN_BorderCorePelvic/imagesTr"
ANAT={"sacrum":(1,50),"LH":(51,100),"RH":(101,150)}

def bone_window(ct):
    lo,hi=-200.0,1100.0
    return np.clip((ct-lo)/(hi-lo),0,1)

for cid in ["001","003","006"]:
    ct=sitk.GetArrayFromImage(sitk.ReadImage(f"{NRAW}/PENGWIN_{cid}_0000.nii.gz")).astype(np.float32)
    gt=sitk.GetArrayFromImage(sitk.ReadImage(f"predictions/gt_instance/PENGWIN_{cid}.nii.gz"))
    bc=convert_to_border_core(gt, PELVIC_ANATOMY, t=2)
    # anatomy with most fragments
    name,(lo,hi)=max(ANAT.items(), key=lambda kv:len([v for v in np.unique(gt) if kv[1][0]<=v<=kv[1][1]]))
    counts=[len([v for v in np.unique(gt[z]) if lo<=v<=hi]) for z in range(gt.shape[0])]
    z=int(np.argmax(counts))
    m=(gt[z]>=lo)&(gt[z]<=hi); ys,xs=np.where(m)
    pad=25; y0,y1=max(ys.min()-pad,0),min(ys.max()+pad,gt.shape[1]); x0,x1=max(xs.min()-pad,0),min(xs.max()+pad,gt.shape[2])
    ctw=bone_window(ct[z,y0:y1,x0:x1])
    gts=np.where((gt[z]>=lo)&(gt[z]<=hi),gt[z],0)[y0:y1,x0:x1]
    bcs=bc[z,y0:y1,x0:x1]
    core=np.isin(bcs,[1,3,5]); bord=np.isin(bcs,[2,4,6])

    fig,ax=plt.subplots(1,4,figsize=(22,6))
    ax[0].imshow(ctw,cmap="gray"); ax[0].set_title(f"PENGWIN_{cid}  CT (bone window)  z={z}")
    # GT fragments colored overlay
    ax[1].imshow(ctw,cmap="gray")
    gm=np.ma.masked_where(gts==0, gts)
    ax[1].imshow(gm,cmap="tab20",alpha=0.55); ax[1].set_title(f"CT + GT fragments ({name}, {counts[z]} pieces)")
    # cores
    ax[2].imshow(ctw,cmap="gray")
    ax[2].imshow(np.ma.masked_where(~core,core),cmap=ListedColormap(["red"]),alpha=0.45); ax[2].set_title("CT + cores")
    # seam = fracture line
    ax[3].imshow(ctw,cmap="gray")
    ax[3].imshow(np.ma.masked_where(~bord,bord),cmap=ListedColormap(["cyan"]),alpha=0.95); ax[3].set_title("CT + seam = FRACTURE lines")
    for a in ax: a.axis("off")
    plt.tight_layout(); out=f"viz/detailed_{cid}.png"; plt.savefig(out,dpi=95,bbox_inches="tight"); plt.close()
    print(f"saved {out}  (z={z}, {name} {counts[z]} fragments)")
print("DONE")
