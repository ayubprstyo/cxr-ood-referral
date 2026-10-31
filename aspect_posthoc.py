# POST-HOC exploratory analysis (motivated by Grad-CAM attention on padding bands; not pre-specified)
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
rng=np.random.default_rng(20261028); B=2000
m=pd.read_csv("outputs/notebook01/mooney_manifest_final.csv"); m["aspect"]=m.width/m.height
m["padding_fraction"]=1-np.minimum(m.width,m.height)/np.maximum(m.width,m.height)
P=pd.read_csv("outputs/notebook03/predictions/predictions_long.csv.gz")
def boot(d, f):
    codes,u=pd.factorize(d.patient_id); out=[]
    for _ in range(B):
        w=np.bincount(rng.integers(0,len(u),len(u)),minlength=len(u))[codes]; idx=np.repeat(np.arange(len(d)),w)
        try: out.append(f(d.iloc[idx]))
        except ValueError: pass
    return np.percentile(out,[2.5,97.5])
rows=[]
for s,lab in [("train","Training"),("val","Validation"),("test_id","ID test")]:
    d=m[m.final_split==s].reset_index(drop=True)
    f=lambda x: roc_auc_score(x.y,x.aspect)
    lo,hi=boot(d,f)
    rows.append(dict(split=lab,n=len(d),aspect_median_normal=round(d[d.y==0].aspect.median(),3),aspect_median_pneumonia=round(d[d.y==1].aspect.median(),3),
                     auroc_aspect_only=round(f(d),3),ci_low=round(lo,3),ci_high=round(hi,3)))
A=pd.DataFrame(rows); A.to_csv("results/supp/tableS_aspect_ratio_auroc.csv",index=False); print(A.to_string(index=False))
rows=[]
for mod in ["efficientnet_b4","vit_b16"]:
    for c in ["MOONEY_VAL","MOONEY_TEST"]:
        d=P[(P.model==mod)&(P.cohort==c)&(P.y==0)].merge(m[["filepath","aspect"]],on="filepath").reset_index(drop=True)
        q=pd.qcut(d.aspect,4,labels=["Q1 (narrowest)","Q2","Q3","Q4 (widest)"]); edges=pd.qcut(d.aspect,4,retbins=True)[1]
        for k,lab in enumerate(q.cat.categories):
            dd=d[q==lab].reset_index(drop=True); fp=dd.pred_A_primary.mean()
            lo,hi=boot(dd,lambda x: x.pred_A_primary.mean())
            rows.append(dict(model=mod,cohort=c,aspect_quartile=lab,aspect_range=f"{edges[k]:.2f}–{edges[k+1]:.2f}",n_normal=len(dd),
                             false_positive_rate=round(fp,3),ci_low=round(lo,3),ci_high=round(hi,3)))
        r=spearmanr(d.aspect,d.logit)[0]
        rows.append(dict(model=mod,cohort=c,aspect_quartile="Spearman rho(aspect, logit), normals",aspect_range="",n_normal=len(d),false_positive_rate=round(r,3),ci_low=np.nan,ci_high=np.nan))
Q=pd.DataFrame(rows); Q.to_csv("results/supp/tableS_fp_rate_by_aspect_quartile.csv",index=False); print(Q.to_string(index=False))
