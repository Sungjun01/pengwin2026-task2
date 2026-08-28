import numpy as np, SimpleITK as sitk, glob, os
from scipy import ndimage as ndi
ANAT=[(1,50),(51,100),(101,150)]
R=150.0; BIN=5.0; N=int(2*R/BIN)  # 60^3 grid, +-150mm, 5mm bins
hist=np.zeros((N,N,N),dtype=np.float64)
ncase=0
files=sorted(glob.glob("predictions/gt_instance/*.nii.gz"))  # all 170 pelvic (sacrum-anchored atlas)
for f in files:
    img=sitk.ReadImage(f); arr=sitk.GetArrayFromImage(img)
    sx,sy,sz=img.GetSpacing(); sp=np.array([sz,sy,sx])  # zyx mm
    sac=(arr>=1)&(arr<=50)
    if not sac.any(): continue
    cz,cy,cx=ndi.center_of_mass(sac)  # sacrum centroid (zyx voxel)
    cen=np.array([cz,cy,cx])
    # seam voxels (inter-fragment, same anatomy)
    seam=np.zeros(arr.shape,bool)
    for lo,hi in ANAT:
        a=np.where((arr>=lo)&(arr<=hi),arr,0)
        for shift in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            nb=np.roll(a,shift,axis=(0,1,2)); seam|=(a>0)&(nb>0)&(a!=nb)
    if not seam.any(): continue
    zz,yy,xx=np.where(seam)
    coords=np.stack([zz,yy,xx],1).astype(np.float64)
    off_mm=(coords-cen)*sp  # offset in mm (zyx)
    idx=((off_mm+R)/BIN).astype(int)
    ok=np.all((idx>=0)&(idx<N),axis=1)
    idx=idx[ok]
    np.add.at(hist,(idx[:,0],idx[:,1],idx[:,2]),1.0)
    ncase+=1
print(f"cases accumulated: {ncase}, total seam voxels binned: {hist.sum():.0f}")
hp=hist/max(ncase,1)
# hotspot clarity metrics
flat=hp[hp>0]
print(f"occupied bins: {len(flat)}  max/mean ratio: {hp.max()/hp[hp>0].mean():.1f}")
top10=np.percentile(flat,90)
mass_top=hp[hp>=top10].sum()/hp.sum()
print(f"fraction of seam-mass in top-10% bins: {100*mass_top:.1f}%  (uniform≈10%, peaky≫10%)")
np.save("/tmp/fracture_atlas_hist.npy", hp)
# 2D projections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,3,figsize=(18,6))
ext=[-R,R,-R,R]
ax[0].imshow(hp.sum(0),cmap="hot",extent=ext,origin="lower"); ax[0].set_title("axial proj (x vs y) — fracture density rel. sacrum centroid")
ax[1].imshow(hp.sum(1),cmap="hot",extent=ext,origin="lower"); ax[1].set_title("coronal proj (x vs z)")
ax[2].imshow(hp.sum(2),cmap="hot",extent=ext,origin="lower"); ax[2].set_title("sagittal proj (y vs z)")
for a in ax: a.set_xlabel("mm"); a.set_ylabel("mm")
plt.tight_layout(); plt.savefig("viz/fracture_atlas.png",dpi=110,bbox_inches="tight")
print("saved viz/fracture_atlas.png")
