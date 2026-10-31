"""Additional (not pre-specified) analyses, all from existing outputs:
 (A) NIH paediatric subgroup (age < 18, from the all-age cohorts), primary models;
 (B) 3-seed deep ensembles per architecture (Notebook 06 letterbox runs) vs MC Dropout;
 (C) NIH subgroups by sex and age band, primary models.
Patient-level bootstrap with fixed seeds."""
import numpy as np, pandas as pd, json
from pathlib import Path
from scipy.special import expit, xlogy
F = Path("results"); S = F / "supp"
B_, V_ = "efficientnet_b4", "vit_b16"; NICE = {B_: "EfficientNet-B4", V_: "ViT-B/16"}
NB = 2000; NB_ENS = 1000

def ranks(s): u, inv = np.unique(s, return_inverse=True); return inv, len(u)
def w_auc(rk, lab, w):
    inv, K = rk; pos = np.bincount(inv, w * lab, K); neg = np.bincount(inv, w * (1 - lab), K); P, N = pos.sum(), neg.sum()
    return float(np.sum(pos * (np.cumsum(neg) - 0.5 * neg)) / (P * N)) if P > 0 and N > 0 else np.nan
def sd(a, b): return a / b if b > 0 else np.nan
def metrics(d, rk, w):
    y, pr, e, a = d.y.values, d.pred.values, d.error.values, d.accepted.values
    out = dict(auroc=w_auc(rk["s"], y, w), sens_A=sd((w * (pr == 1) * (y == 1)).sum(), (w * (y == 1)).sum()),
               spec_A=sd((w * (pr == 0) * (y == 0)).sum(), (w * (y == 0)).sum()), errdet_auroc=w_auc(rk["u"], e, w),
               error_rate=sd((w * e).sum(), w.sum()), selective_risk=sd((w * a * e).sum(), (w * a).sum()),
               coverage=sd((w * a).sum(), w.sum()), g4_pct_of_errors=100 * sd((w * a * e).sum(), (w * e).sum()))
    out["risk_change"] = out["selective_risk"] - out["error_rate"]
    if "u2" in rk: out["errdet_auroc_mi"] = w_auc(rk["u2"], e, w)
    return out
def boot(d, nb, seed, extra=None):
    d = d.reset_index(drop=True); rk = {"s": ranks(d.score.values), "u": ranks(d.unc.values)}
    if "unc2" in d: rk["u2"] = ranks(d.unc2.values)
    pt = metrics(d, rk, np.ones(len(d)))
    codes, u = pd.factorize(d.patient_id.values); rng = np.random.default_rng(seed)
    bs = pd.DataFrame([metrics(d, rk, np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[codes].astype(float)) for _ in range(nb)])
    rows = []
    for k, v in pt.items():
        lo, hi = np.nanpercentile(bs[k], [2.5, 97.5]) if bs[k].notna().any() else (np.nan, np.nan)
        rows.append(dict(metric=k, estimate=v, ci_low=lo, ci_high=hi))
    base = dict(n_images=len(d), n_pneumonia=int(d.y.sum()), n_patients=len(u)); base.update(extra or {})
    return [dict(**base, **r) for r in rows]

# ---------------- primary per-image data (version A, Notebook 04 MC Dropout)
U = pd.read_csv("outputs/notebook04/predictions/uncertainty_per_image.csv.gz",
                usecols=["model", "cohort", "filename", "filepath", "patient_id", "y", "logit_det", "entropy_mc", "pred", "error", "low_uncertainty"])
U = U.rename(columns={"logit_det": "score", "entropy_mc": "unc", "low_uncertainty": "accepted"})
DEMO = pd.concat([pd.read_csv(f"outputs/notebook01b/locked_cohorts/{c}.csv", usecols=["cohort", "filename", "age", "sex"])
                  for c in ["OOD_AP_ALLAGE", "OOD_PA_ALLAGE", "OOD_HARD_AP_ALLAGE", "OOD_AP_ADULT", "OOD_PA_ADULT", "OOD_HARD_AP_ADULT"]])
U = U.merge(DEMO, on=["cohort", "filename"], how="left")

# ---------------- (A) paediatric NIH subgroup
rows = []
for m in [B_, V_]:
    for base, lab in [("AP", "NIH AP"), ("PA", "NIH PA"), ("HARD_AP", "NIH AP-Hard")]:
        d = U[(U.model == m) & (U.cohort == f"OOD_{base}_ALLAGE")]
        for grp, sub in [("Children (<18 y)", d[d.age < 18]), ("Adults (18–100 y)", d[(d.age >= 18) & (d.age <= 100)])]:
            rows += boot(sub, NB, 7100 + len(rows), dict(model=NICE[m], cohort=lab, group=grp, age_median=float(sub.age.median())))
    d = U[(U.model == m) & (U.cohort == "MOONEY_TEST")]
    rows += boot(d, NB, 7999 + len(rows), dict(model=NICE[m], cohort="Mooney ID test", group="Children (Mooney)", age_median=np.nan))
PED = pd.DataFrame(rows); PED.to_csv(S / "_paediatric_long.csv", index=False)

# ---------------- (C) sex and age bands, adult primary cohorts
rows = []
bands = [("18–39 y", 18, 39), ("40–59 y", 40, 59), ("60–100 y", 60, 100)]
for m in [B_, V_]:
    for c, lab in [("OOD_AP_ADULT", "NIH AP"), ("OOD_PA_ADULT", "NIH PA"), ("OOD_HARD_AP_ADULT", "NIH AP-Hard")]:
        d = U[(U.model == m) & (U.cohort == c)]
        groups = [("Sex", "Female", d[d.sex == "F"]), ("Sex", "Male", d[d.sex == "M"])] + [("Age", n, d[(d.age >= lo) & (d.age <= hi)]) for n, lo, hi in bands]
        for kind, g, sub in groups:
            rows += boot(sub, NB, 8100 + len(rows), dict(model=NICE[m], cohort=lab, kind=kind, group=g))
SUB = pd.DataFrame(rows); SUB.to_csv(S / "_subgroups_long.csv", index=False)

# ---------------- (B) deep ensembles (Notebook 06 letterbox seeds 42/43/44)
P6 = pd.read_csv("outputs/notebook06/predictions/robustness_per_image.csv.gz", usecols=["run", "cohort", "filepath", "patient_id", "y", "logit", "entropy_mc"])
def H2(p): return -(xlogy(p, p) + xlogy(1 - p, 1 - p)) / np.log(2)
def thr_A(z, y):
    best = None
    for t in np.r_[np.unique(z), z.max() + 1e-6]:
        pr = z >= t; sens = (pr & (y == 1)).sum() / (y == 1).sum(); spec = (~pr & (y == 0)).sum() / (y == 0).sum()
        if spec >= 0.95 and (best is None or (sens, spec, t) > best): best = (sens, spec, t)
    return float(best[2])
rows, ENS_OPS = [], {}
for m in [B_, V_]:
    runs = [f"{m}__lb_s{s}" for s in (42, 43, 44)]
    W = P6[P6.run.isin(runs)].pivot_table(index=["cohort", "filepath", "patient_id", "y"], columns="run", values="logit").reset_index()
    p = expit(W[runs].values.astype(np.float64)); pbar = p.mean(1)
    W["score"] = np.log(np.clip(pbar, 1e-12, 1)) - np.log(np.clip(1 - pbar, 1e-12, 1))
    W["unc"] = H2(pbar); W["unc2"] = np.clip(H2(pbar) - H2(p).mean(1), 0, None)
    v = W[W.cohort == "MOONEY_VAL"]; tA = thr_A(v.score.values, v.y.values); tau = float(np.quantile(v.unc, 0.90, method="higher"))
    ENS_OPS[m] = dict(thrA=tA, tau90=tau)
    W["pred"] = (W.score >= tA).astype(int); W["error"] = (W.pred != W.y).astype(int); W["accepted"] = (W.unc <= tau).astype(int)
    for c, lab in [("MOONEY_TEST", "Mooney ID test"), ("OOD_AP_ADULT", "NIH AP"), ("OOD_PA_ADULT", "NIH PA"), ("OOD_HARD_AP_ADULT", "NIH AP-Hard")]:
        rows += boot(W[W.cohort == c], NB_ENS, 9100 + len(rows), dict(model=NICE[m], method="3-seed deep ensemble", cohort=lab))
    # MC Dropout of the primary model, same pipeline for comparison
    for c, lab in [("MOONEY_TEST", "Mooney ID test"), ("OOD_AP_ADULT", "NIH AP"), ("OOD_PA_ADULT", "NIH PA"), ("OOD_HARD_AP_ADULT", "NIH AP-Hard")]:
        d = U[(U.model == m) & (U.cohort == c)]
        rows += boot(d, NB_ENS, 9500 + len(rows), dict(model=NICE[m], method="MC Dropout, primary model", cohort=lab))
    # OOD detection AUROC for the ensemble (ID test = 0 vs NIH = 1)
    I = W[W.cohort == "MOONEY_TEST"]
    for c, lab in [("OOD_AP_ADULT", "NIH AP"), ("OOD_PA_ADULT", "NIH PA"), ("OOD_HARD_AP_ADULT", "NIH AP-Hard")]:
        O = W[W.cohort == c]; sc = np.r_[I.unc.values, O.unc.values]; lb = np.r_[np.zeros(len(I)), np.ones(len(O))]
        ENS_OPS[m][f"ooddet_{c}"] = w_auc(ranks(sc), lb, np.ones(len(sc)))
ENS = pd.DataFrame(rows); ENS.to_csv(S / "_ensemble_long.csv", index=False)
json.dump(ENS_OPS, open(S / "_ensemble_ops.json", "w"), indent=1)

# ---------------- publication tables
fm = lambda e, l, h: ("—" if np.isnan(e) else (f"{e:.3f} ({l:.3f}–{h:.3f})" if min(l, h) >= 0 else f"{e:.3f} ({l:.3f} to {h:.3f})").replace("-", "−"))
def wide(L, keys, metrics_):
    out = []
    for k, g in L.groupby(keys, sort=False):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r.update({"Images": int(g.n_images.iloc[0]), "Pneumonia": int(g.n_pneumonia.iloc[0])})
        for mm, nm in metrics_:
            x = g[g.metric == mm]; r[nm] = fm(x.estimate.iloc[0], x.ci_low.iloc[0], x.ci_high.iloc[0]) if len(x) else "—"
        out.append(r)
    return pd.DataFrame(out)
MT = [("auroc", "AUROC (95% CI)"), ("spec_A", "Specificity at A"), ("sens_A", "Sensitivity at A"), ("errdet_auroc", "Error-detection AUROC"),
      ("error_rate", "Error rate, no referral"), ("selective_risk", "Selective risk at τ90")]
wide(PED, ["model", "cohort", "group"], MT).rename(columns={"model": "Model", "cohort": "Cohort", "group": "Group"}).to_csv(S / "tableS13_nih_paediatric.csv", index=False)
wide(SUB, ["model", "cohort", "kind", "group"], MT[:4] + [("selective_risk", "Selective risk at τ90")]).drop(columns=["kind"]).rename(
    columns={"model": "Model", "cohort": "Cohort", "group": "Subgroup"}).to_csv(S / "tableS14_nih_subgroups.csv", index=False)
wide(ENS, ["model", "method", "cohort"], [("auroc", "AUROC (95% CI)"), ("spec_A", "Specificity at A"), ("errdet_auroc", "Error-detection AUROC (entropy)"),
      ("errdet_auroc_mi", "Error-detection AUROC (disagreement)"), ("error_rate", "Error rate, no referral"), ("selective_risk", "Selective risk at τ90"),
      ("g4_pct_of_errors", "Confidently incorrect (% of errors)")]).rename(columns={"model": "Model", "method": "Uncertainty method", "cohort": "Cohort"}).to_csv(S / "tableS15_deep_ensemble.csv", index=False)
print("done")
