import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
np.random.seed(0)
SH=64; BIN=4.0
frac=np.load("/tmp/frac_atlas64.npy"); anat=np.load("/tmp/anat_atlas.npy")
flat=frac.flatten().astype(float); flat=flat/flat.sum()
idx=np.random.choice(len(flat),size=60000,p=flat)
zyx=np.array(np.unravel_index(idx,frac.shape)).T.astype(float)+np.random.uniform(0,1,(60000,3))
pts=(zyx-SH/2)*BIN   # mm, (z=SI, y=AP, x=LR)
tr,te=pts[:48000],pts[48000:]
# BIC sweep
Ks=range(2,13); bic=[]
for K in Ks:
    gm=GaussianMixture(K,covariance_type='full',random_state=0,reg_covar=1e-3).fit(tr)
    bic.append(gm.bic(tr))
bestK=list(Ks)[int(np.argmin(bic))]
print("BIC by K:", {k:int(b) for k,b in zip(Ks,bic)})
print("best K (min BIC):", bestK)
# GMM vs KDE test log-likelihood
gm=GaussianMixture(bestK,covariance_type='full',random_state=0,reg_covar=1e-3).fit(tr)
gm_ll=gm.score_samples(te).mean()
kde=KernelDensity(bandwidth=8.0).fit(tr); kde_ll=kde.score_samples(te[:8000]).mean()
print(f"test avg log-lik:  GMM(K={bestK}) = {gm_ll:.3f}   KDE(bw=8mm) = {kde_ll:.3f}")
print(f"-> {'GMM 충분(comparable/better)' if gm_ll>=kde_ll-0.1 else 'KDE 우세 (surface 구조가 few-Gaussian 초과)'}")
# component weights
print("component weights:", np.round(np.sort(gm.weights_)[::-1],3))
# viz: atlas MIP + GMM means per view
m=gm.means_  # (K,3) in mm (z,y,x)
fig,ax=plt.subplots(1,3,figsize=(18,6))
def mm2bin(v): return v/BIN+SH/2
views=[(0,1,2,"Axial: LR(x) vs AP(y)"),(2,1,0,"Sagittal: AP(y) vs SI(z)"),(1,2,0,"Coronal: LR(x) vs SI(z)")]
# proj axis: which atlas axis to MIP-collapse, and which two dims to plot (x-axis dim, y-axis dim)
proj=[(0,2,1),(2,1,0),(1,2,0)]  # (collapse_axis, plot_x_dim, plot_y_dim)
titles=["Axial (collapse SI): x=LR y=AP","Sagittal (collapse LR): x=AP y=SI","Coronal (collapse AP): x=LR y=SI"]
for k,(cax,dx,dy) in enumerate(proj):
    ax[k].imshow(anat.max(axis=cax),cmap="gray",origin="lower")
    fm=frac.max(axis=cax); ax[k].imshow(np.ma.masked_where(fm<0.03,fm),cmap="hot",origin="lower",alpha=0.7)
    # GMM means -> bin coords on this projection
    bx=mm2bin(m[:,dx]); by=mm2bin(m[:,dy])
    ax[k].scatter(bx,by,s=120,facecolors='none',edgecolors='cyan',linewidths=2.5)
    for j in range(bestK): ax[k].text(bx[j],by[j],str(j),color='cyan',fontsize=9)
    ax[k].set_title(titles[k]); ax[k].axis("off")
fig.suptitle(f"GMM (K={bestK}) components (cyan) over fracture atlas — anatomical clusters?",fontsize=13)
plt.savefig("viz/gmm_components.png",dpi=110,bbox_inches="tight")
# BIC curve
plt.figure(figsize=(6,4)); plt.plot(list(Ks),bic,'o-'); plt.xlabel("K"); plt.ylabel("BIC"); plt.title("GMM BIC vs K"); plt.tight_layout(); plt.savefig("viz/gmm_bic.png",dpi=110)
print("saved viz/gmm_components.png, viz/gmm_bic.png")
