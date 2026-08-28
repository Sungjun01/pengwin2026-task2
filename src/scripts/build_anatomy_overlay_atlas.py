import numpy as np, SimpleITK as sitk, glob
from scipy import ndimage as ndi
SH=64; R=128.0; B=2*R/SH  # 4mm bins, +-128mm
ANAT=[(1,50),(51,100),(101,150)]
frac=np.zeros((SH,SH,SH)); anat=np.zeros((SH,SH,SH)); nc=0
files=sorted(glob.glob("predictions/gt_instance/*.nii.gz"))[:120]
def to_idx(coords,cen,sp):
    off=(coords-cen)*sp
    idx=np.clip(((off+R)/B).astype(int),0,SH-1)
    return idx
for f in files:
    img=sitk.ReadImage(f); arr=sitk.GetArrayFromImage(img)
    sx,sy,sz=img.GetSpacing(); sp=np.array([sz,sy,sx])
    sac=(arr>=1)&(arr<=50)
    if not sac.any(): continue
    cen=np.array(ndi.center_of_mass(sac))
    fg=(arr>=1)&(arr<=150)
    zz,yy,xx=np.where(fg)
    # anatomy presence (subsample stride 3 for speed)
    s=slice(None,None,3); ai=to_idx(np.stack([zz[s],yy[s],xx[s]],1).astype(float),cen,sp)
    av=np.zeros((SH,SH,SH),bool); av[ai[:,0],ai[:,1],ai[:,2]]=True; anat+=av
    # fracture seam (inter-fragment, same anatomy)
    seam=np.zeros(arr.shape,bool)
    for lo,hi in ANAT:
        a=np.where((arr>=lo)&(arr<=hi),arr,0)
        for sh in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nb=np.roll(a,sh,axis=(0,1,2)); seam|=(a>0)&(nb>0)&(a!=nb)
    sz_,sy_,sx_=np.where(seam)
    if len(sz_)==0: nc+=1; continue
    fi=to_idx(np.stack([sz_,sy_,sx_],1).astype(float),cen,sp)
    fv=np.zeros((SH,SH,SH),bool); fv[fi[:,0],fi[:,1],fi[:,2]]=True; frac+=fv
    nc+=1
anat/=nc; frac/=nc
print(f"cases={nc}  anat max={anat.max():.2f} frac max={frac.max():.2f}")
np.save("/tmp/anat_atlas.npy",anat); np.save("/tmp/frac_atlas64.npy",frac)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,3,figsize=(18,6))
views=[(0,"Axial (top-down): L-R vs A-P"),(2,"Sagittal (side): A-P vs S-I"),(1,"Coronal (front): L-R vs S-I")]
for k,(axis,t) in enumerate(views):
    ax[k].imshow(anat.max(axis=axis),cmap="gray",origin="lower",alpha=0.85)
    fm=frac.max(axis=axis); im=ax[k].imshow(np.ma.masked_where(fm<0.03,fm),cmap="hot",origin="lower",alpha=0.85,vmin=0,vmax=frac.max())
    ax[k].set_title(t); ax[k].axis("off")
fig.suptitle(f"Fracture hotspot (hot) over average pelvic shape (gray) — {nc} cases, anatomy-relative",fontsize=13)
fig.colorbar(im,ax=ax,fraction=0.025,label="P(fracture)")
plt.savefig("viz/atlas_anatomy_overlay.png",dpi=120,bbox_inches="tight")
print("saved viz/atlas_anatomy_overlay.png")
