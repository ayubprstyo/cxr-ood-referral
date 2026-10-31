from pathlib import Path
INPUT_ROOT = Path("/kaggle/input")
NIH_ROOT   = Path("/kaggle/input/datasets/nih-chest-xrays/data")
PROJECT    = Path("outputs/notebook03")
MODELS     = ["efficientnet_b4", "vit_b16"]
N_BOOT     = 2000
BOOT_SEED  = 20261005
ECE_BINS   = 15
BATCH      = 64
SMOKE_TEST = False

LOCKED_THRESHOLDS = {
    "efficientnet_b4": {"A_primary": 0.4689939068127956, "B_secondary": 3.875000849286004, "C_default": 0.0},
    "vit_b16":         {"A_primary": -0.28979498040711366, "B_secondary": 3.5078123195824205, "C_default": 0.0},
}   # threshold_logit values
THR_LABEL = {"A_primary": "A (primary): max sens | spec≥0.95", "B_secondary": "B: max spec | sens≥0.95", "C_default": "C: 0.50"}
EXPECTED_FP = {
    "MOONEY_VAL": "13d0556c2de25782967c5cd20ceb54508ca4b8061cdca9e0c9c6ae0bff2f139c",
    "MOONEY_TEST": "723c6c2e962c6704b4e61e693ea96f831104cd9f937acbbf309b46b0edc55101",
    "OOD_AP_ALLAGE": "2f209001f1183bcddc265cd93a36eaf00d2151d6d0088b0b44c2cd1d03c70f80",
    "OOD_AP_ADULT": "b13bc8c6413b3a5725e08184ade4a34305288d0daff717533589399c83e0d567",
    "OOD_PA_ALLAGE": "f1a6c96829c1b91d1221cd971c2de053be878f49fea846ff5fd36ba3e36b8474",
    "OOD_PA_ADULT": "ede8eeb671334525f5d4d0edce8a8a025bcf7992bb19d9060e8a17c6bd116cb0",
    "OOD_HARD_AP_ALLAGE": "9647dd49e598ef43d1321fe31e4401786c8293e15ba71956d8b6c77f2c922372",
    "OOD_HARD_AP_ADULT": "b8cc02779f56de273cd8ffcca8bc5915451e279f97f9b42f2211c1f1c544fb60",
}
ID_COHORTS  = ["MOONEY_TEST", "MOONEY_TEST_NONCOLLIDING", "MOONEY_TEST_DEDUP"]
OOD_PRIMARY = ["OOD_AP_ADULT", "OOD_PA_ADULT", "OOD_HARD_AP_ADULT"]
OOD_SENS    = ["OOD_AP_ALLAGE", "OOD_PA_ALLAGE", "OOD_HARD_AP_ALLAGE"]
EVAL_COHORTS = ID_COHORTS + OOD_PRIMARY + OOD_SENS

import os, re, sys, json, time, math, hashlib, shutil, platform, datetime
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DIRS = {k: PROJECT / k for k in ["predictions", "metrics", "figures", "tables", "logs", "configs", "code"]}
for d in DIRS.values(): d.mkdir(parents=True, exist_ok=True)
ENV = dict(note="Offline recompute of Notebook 03 statistics from Notebook 04 deterministic logits (locked Notebook 02 version A)",
           source_logits="phase5_6 ZIP 2026-10-24 predictions/image_logits_locked_version.csv",
           python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
           timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(DIRS["logs"] / "environment_03_offline.json").write_text(json.dumps(ENV, indent=2))
def fingerprint(paths): return hashlib.sha256("\n".join(sorted(paths)).encode()).hexdigest()
man = pd.read_csv("outputs/notebook01/mooney_manifest_final.csv")
dev_pids = set(man.loc[man.final_split.isin(["train", "val"]), "patient_id"].astype(str))
COH = {}
for name, split in [("MOONEY_VAL", "val"), ("MOONEY_TEST", "test_id")]:
    d = man[man.final_split == split].sort_values("filepath").reset_index(drop=True)
    COH[name] = pd.DataFrame({"dataset": "MOONEY_KERMANY", "cohort": name, "filename": d.filename, "filepath": d.filepath,
                              "patient_id": "M:" + d.patient_id.astype(str), "y": d.y.astype(int), "sha256": d.sha256})
t = COH["MOONEY_TEST"]
COH["MOONEY_TEST_NONCOLLIDING"] = t[~t.patient_id.str[2:].isin(dev_pids)].assign(cohort="MOONEY_TEST_NONCOLLIDING")
COH["MOONEY_TEST_DEDUP"] = t[~t.sha256.duplicated(keep="first")].assign(cohort="MOONEY_TEST_DEDUP")
for c in OOD_PRIMARY + OOD_SENS:
    d = pd.read_csv(f"outputs/notebook01b/locked_cohorts/{c}.csv").sort_values("filename").reset_index(drop=True)
    COH[c] = pd.DataFrame({"dataset": "NIH_CHESTXRAY14", "cohort": c, "filename": d.filename, "filepath": d.filepath,
                           "patient_id": "N:" + d.patient_id.astype(str), "y": d.y.astype(int), "age": d.age, "sex": d.sex,
                           "analysis_role": d.analysis_role})
checks = []
for c in ["MOONEY_VAL", "MOONEY_TEST"]: checks.append((c, fingerprint(COH[c].filepath) == EXPECTED_FP[c]))
for c in OOD_PRIMARY + OOD_SENS: checks.append((c, hashlib.sha256("\n".join(COH[c].filename).encode()).hexdigest() == EXPECTED_FP[c]))
checks += [("MOONEY_TEST_NONCOLLIDING n=304", len(COH["MOONEY_TEST_NONCOLLIDING"]) == 304), ("MOONEY_TEST_DEDUP n=618", len(COH["MOONEY_TEST_DEDUP"]) == 618)]
for c, ok in checks: print(f"[{'PASS' if ok else 'FAIL'}] cohort {c}")
assert all(ok for _, ok in checks)
LOGITS = pd.read_csv("outputs/notebook04/predictions/image_logits_locked_version.csv").drop_duplicates("filepath").set_index("filepath")[MODELS].astype(np.float64)
need = set(pd.concat([c.filepath for c in COH.values()]))
assert need <= set(LOGITS.index), f"missing {len(need - set(LOGITS.index))}"
# reproduction check against the locked Notebook 02 version A validation predictions
for m in MODELS:
    ref = pd.read_csv(f"outputs/notebook02/predictions/{m}_val_predictions.csv").set_index("filepath")["logit"]
    dd = np.abs(LOGITS.loc[COH["MOONEY_VAL"].filepath, m].values - ref.loc[COH["MOONEY_VAL"].filepath].values)
    print(f"{m}: vs Notebook 02 (version A) validation logits — median {np.median(dd):.4f}, max {dd.max():.4f}")
    assert dd.max() < 0.05

#
# ## 4. Temperature scaling — fitted on Mooney validation only, then frozen

#
def sigmoid(z): return 1 / (1 + np.exp(-z))
def nll(z, y):
    p = np.clip(sigmoid(z), 1e-12, 1 - 1e-12); return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
TEMP = {}
yv = COH["MOONEY_VAL"].y.values
for m in MODELS:
    zv = LOGITS.loc[COH["MOONEY_VAL"].filepath, m].values
    r = minimize_scalar(lambda lt: nll(zv / np.exp(lt), yv), bounds=(-3, 3), method="bounded")
    TEMP[m] = float(np.exp(r.x))
    if abs(r.x) > 2.95: print(f"WARNING: {m} temperature at search bound (T={TEMP[m]:.3f}) — report and review before interpreting calibration.")
    print(f"{m}: T = {TEMP[m]:.4f} | val NLL {nll(zv, yv):.4f} → {nll(zv / TEMP[m], yv):.4f}")
(DIRS["configs"] / "thresholds_locked.json").write_text(json.dumps(dict(threshold_logits=LOCKED_THRESHOLDS, labels=THR_LABEL,
    locked="2026-10-05, from Notebook 02 Mooney validation predictions only, before any ID-test or NIH output existed"), indent=2))
(DIRS["configs"] / "temperature_scaling.json").write_text(json.dumps(dict(
    temperatures=TEMP, fitted_on="MOONEY_VAL (738 images) only", objective="NLL, bounded search on log T in [-3, 3]",
    note="Applied unchanged to all cohorts; predicted classes are unaffected because thresholds act on logits."), indent=2))

#
# ## 5. Long-format predictions (one row per model × cohort × image)

#
rows = []
for m in MODELS:
    for c in ["MOONEY_VAL"] + EVAL_COHORTS:
        d = COH[c][["dataset", "cohort", "filename", "filepath", "patient_id", "y"]].copy()
        d.insert(0, "model", m)
        z = LOGITS.loc[d.filepath, m].values
        d["logit"], d["prob"], d["prob_ts"] = z, sigmoid(z), sigmoid(z / TEMP[m])
        for k, tl in LOCKED_THRESHOLDS[m].items(): d[f"pred_{k}"] = (z >= tl).astype(int)
        d["pred_class"] = d["pred_A_primary"]
        rows.append(d)
PRED = pd.concat(rows, ignore_index=True)
PRED.to_csv(DIRS["predictions"] / "predictions_long.csv.gz", index=False)
print(PRED.groupby(["model", "cohort"]).size().unstack(0))

#
# ## 6. Metrics with patient-level bootstrap

#
class Prepared:
    """Pre-binned scores so weighted AUROC / AP / calibration are cheap for each bootstrap draw."""
    def __init__(self, y, z, T, pids):
        self.y = y.astype(float); self.z = z; self.p = sigmoid(z); self.pts = sigmoid(z / T)
        self.u, self.inv = np.unique(z, return_inverse=True); self.K = len(self.u)
        self.pid_codes, self.pid_uni = pd.factorize(pids)
        self.bin_raw = np.minimum((self.p * ECE_BINS).astype(int), ECE_BINS - 1)
        self.bin_ts  = np.minimum((self.pts * ECE_BINS).astype(int), ECE_BINS - 1)

def weighted_metrics(P, w, thr_logits):
    y = P.y; wy, wn = w * y, w * (1 - y)
    Pw, Nw = wy.sum(), wn.sum()
    pos = np.bincount(P.inv, wy, P.K); neg = np.bincount(P.inv, wn, P.K)
    auroc = float(np.sum(pos * (np.cumsum(neg) - neg + 0.5 * neg)) / (Pw * Nw)) if Pw > 0 and Nw > 0 else np.nan
    tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = tp / Pw if Pw > 0 else np.full_like(tp, np.nan)
    auprc = float(np.sum(np.diff(np.r_[0, rec]) * prec)) if Pw > 0 else np.nan
    out = dict(auroc=auroc, auprc=auprc)
    W = w.sum()
    for tag, p, b in [("raw", P.p, P.bin_raw), ("ts", P.pts, P.bin_ts)]:
        pc = np.clip(p, 1e-12, 1 - 1e-12)
        out[f"brier_{tag}"] = float(np.sum(w * (p - y) ** 2) / W)
        out[f"nll_{tag}"] = float(-np.sum(w * (y * np.log(pc) + (1 - y) * np.log(1 - pc))) / W)
        out[f"ece_{tag}"] = float(np.abs(np.bincount(b, w * (y - p), ECE_BINS)).sum() / W)
    for k, tl in thr_logits.items():
        yhat = (P.z >= tl).astype(float)
        TP, FN = np.sum(wy * yhat), np.sum(wy * (1 - yhat)); FP, TN = np.sum(wn * yhat), np.sum(wn * (1 - yhat))
        sd = lambda a, b: a / b if b > 0 else np.nan
        sens, spec, ppv, npv = sd(TP, TP + FN), sd(TN, TN + FP), sd(TP, TP + FP), sd(TN, TN + FN)
        out.update({f"sens_{k}": sens, f"spec_{k}": spec, f"ppv_{k}": ppv, f"npv_{k}": npv,
                    f"f1_{k}": sd(2 * ppv * sens, ppv + sens) if not (np.isnan(ppv) or np.isnan(sens)) else np.nan,
                    f"acc_{k}": sd(TP + TN, W)})
    return out

def cohort_seed(c): return BOOT_SEED + int(hashlib.md5(c.encode()).hexdigest()[:8], 16) % 10**6

POINT, BOOT = {}, {}
t0 = time.time()
for c in EVAL_COHORTS:
    rng = np.random.default_rng(cohort_seed(c))            # identical draws for both models → paired comparisons
    base = COH[c]; npat = base.patient_id.nunique()
    draws = [np.bincount(rng.integers(0, npat, npat), minlength=npat).astype(np.int32) for _ in range(N_BOOT if not SMOKE_TEST else 50)]
    for m in MODELS:
        d = PRED[(PRED.model == m) & (PRED.cohort == c)]
        P = Prepared(d.y.values, d.logit.values, TEMP[m], d.patient_id.values)
        POINT[(m, c)] = weighted_metrics(P, np.ones(len(d)), LOCKED_THRESHOLDS[m])
        # pid factorisation order must match the draw vector: re-map draws through this cohort's patient codes
        codes = pd.Index(P.pid_uni)
        ref_codes = pd.Index(pd.factorize(base.patient_id.values)[1])
        remap = ref_codes.get_indexer(codes)
        BOOT[(m, c)] = pd.DataFrame([weighted_metrics(P, cnt[remap][P.pid_codes].astype(float), LOCKED_THRESHOLDS[m]) for cnt in draws])
    print(f"{c}: {len(base)} images, {npat} patients — bootstrapped ({time.time() - t0:.0f}s)")

def ci(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    return (np.percentile(a, 2.5), np.percentile(a, 97.5)) if len(a) else (np.nan, np.nan)
def fmt(v, lo, hi, d=3): return f"{v:.{d}f} ({lo:.{d}f}–{hi:.{d}f})" if not np.isnan(v) else "NA"

long = []
for (m, c), pt in POINT.items():
    for k, v in pt.items():
        lo, hi = ci(BOOT[(m, c)][k]); long.append(dict(model=m, cohort=c, metric=k, estimate=v, ci_low=lo, ci_high=hi))
MET = pd.DataFrame(long); MET.to_csv(DIRS["metrics"] / "metrics_with_ci_long.csv", index=False)

# confusion matrices (point estimates)
cm = []
for m in MODELS:
    for c in EVAL_COHORTS:
        d = PRED[(PRED.model == m) & (PRED.cohort == c)]
        for k in LOCKED_THRESHOLDS[m]:
            yh = d[f"pred_{k}"].values; y = d.y.values
            cm.append(dict(model=m, cohort=c, threshold=k, TP=int(((yh == 1) & (y == 1)).sum()), FN=int(((yh == 0) & (y == 1)).sum()),
                           FP=int(((yh == 1) & (y == 0)).sum()), TN=int(((yh == 0) & (y == 0)).sum())))
pd.DataFrame(cm).to_csv(DIRS["metrics"] / "confusion_matrices.csv", index=False)

#
# ## 7. ID → OOD deltas and architecture comparison

#
DELTA_METRICS = ["auroc", "auprc", "sens_A_primary", "spec_A_primary", "brier_raw", "brier_ts", "ece_raw", "ece_ts", "nll_raw", "nll_ts"]
drows = []
for m in MODELS:
    for c in OOD_PRIMARY + OOD_SENS:
        for k in DELTA_METRICS:
            v = POINT[(m, c)][k] - POINT[(m, "MOONEY_TEST")][k]
            lo, hi = ci(BOOT[(m, c)][k].values - BOOT[(m, "MOONEY_TEST")][k].values)
            drows.append(dict(model=m, ood_cohort=c, metric=k, delta_ood_minus_id=v, ci_low=lo, ci_high=hi))
DELTA = pd.DataFrame(drows); DELTA.to_csv(DIRS["metrics"] / "id_to_ood_deltas.csv", index=False)

def delong(y, s1, s2):
    """Fast DeLong (Sun & Xu 2014) for two correlated AUROCs on the same images."""
    pos, neg = s1[y == 1], s1[y == 0]; pos2, neg2 = s2[y == 1], s2[y == 0]
    def comps(p, n):
        allv = np.r_[p, n]; r = pd.Series(allv).rank().values
        rp, rn = pd.Series(p).rank().values, pd.Series(n).rank().values
        m_, n_ = len(p), len(n)
        auc = (r[:m_].sum() - m_ * (m_ + 1) / 2) / (m_ * n_)
        v10 = (r[:m_] - rp) / n_; v01 = 1 - (r[m_:] - rn) / m_
        return auc, v10, v01
    a1, v10a, v01a = comps(pos, neg); a2, v10b, v01b = comps(pos2, neg2)
    s10 = np.cov(np.vstack([v10a, v10b])); s01 = np.cov(np.vstack([v01a, v01b]))
    S = s10 / len(pos) + s01 / len(neg)
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    zst = (a1 - a2) / np.sqrt(var) if var > 0 else np.nan
    return a1 - a2, float(2 * norm.sf(abs(zst))) if not np.isnan(zst) else np.nan

arows = []
if len(MODELS) == 2:
    m1, m2 = MODELS
    for c in EVAL_COHORTS:
        d1 = PRED[(PRED.model == m1) & (PRED.cohort == c)].set_index("filepath")
        d2 = PRED[(PRED.model == m2) & (PRED.cohort == c)].set_index("filepath").loc[d1.index]
        dauc, p_delong = delong(d1.y.values, d1.logit.values, d2.logit.values)
        for k in ["auroc", "auprc", "sens_A_primary", "spec_A_primary", "brier_ts", "ece_ts"]:
            v = POINT[(m1, c)][k] - POINT[(m2, c)][k]
            lo, hi = ci(BOOT[(m1, c)][k].values - BOOT[(m2, c)][k].values)
            arows.append(dict(cohort=c, metric=k, difference=f"{m1} − {m2}", estimate=v, ci_low=lo, ci_high=hi,
                              delong_p=p_delong if k == "auroc" else np.nan))
ARCH = pd.DataFrame(arows); ARCH.to_csv(DIRS["metrics"] / "architecture_comparison.csv", index=False)

#
# ## 8. Publication tables (Tables 1–4)

#
def cell(m, c, k, d=3):
    r = MET[(MET.model == m) & (MET.cohort == c) & (MET.metric == k)].iloc[0]
    return fmt(r.estimate, r.ci_low, r.ci_high, d)

# Table 1 — cohort characteristics
t1 = []
for c in ["MOONEY_VAL"] + EVAL_COHORTS:
    d = COH[c]
    row = dict(cohort=c, role="calibration/threshold" if c == "MOONEY_VAL" else ("ID primary" if c == "MOONEY_TEST" else
               "ID sensitivity" if c in ID_COHORTS else "OOD primary" if c in OOD_PRIMARY else "OOD sensitivity"),
               images=len(d), positives=int(d.y.sum()), negatives=int((d.y == 0).sum()), prevalence_pct=round(100 * d.y.mean(), 2),
               patients=d.patient_id.nunique())
    if "age" in d:
        row.update(age_median_iqr=f"{d.age.median():.0f} ({d.age.quantile(.25):.0f}–{d.age.quantile(.75):.0f})",
                   female_pct=round(100 * (d.sex == "F").mean(), 1))
    t1.append(row)
pd.DataFrame(t1).to_csv(DIRS["tables"] / "table1_cohort_characteristics.csv", index=False)

PERF = [("auroc", "AUROC"), ("auprc", "AUPRC"), ("sens_A_primary", "Sensitivity"), ("spec_A_primary", "Specificity"),
        ("ppv_A_primary", "PPV"), ("npv_A_primary", "NPV"), ("f1_A_primary", "F1"), ("acc_A_primary", "Accuracy")]
def perf_table(cohorts):
    return pd.DataFrame([dict(model=m, cohort=c, **{lab: cell(m, c, k) for k, lab in PERF}) for m in MODELS for c in cohorts])
perf_table(ID_COHORTS).to_csv(DIRS["tables"] / "table2_id_performance.csv", index=False)

t3 = perf_table(OOD_PRIMARY + OOD_SENS)
for k, lab in [("auroc", "ΔAUROC"), ("auprc", "ΔAUPRC"), ("sens_A_primary", "ΔSensitivity"), ("spec_A_primary", "ΔSpecificity")]:
    t3[lab] = [fmt(*DELTA[(DELTA.model == r.model) & (DELTA.ood_cohort == r.cohort) & (DELTA.metric == k)]
                   [["delta_ood_minus_id", "ci_low", "ci_high"]].iloc[0].values) for r in t3.itertuples()]
t3.to_csv(DIRS["tables"] / "table3_ood_performance_and_degradation.csv", index=False)

t4 = []
for m in MODELS:
    for c in ID_COHORTS[:1] + OOD_PRIMARY + OOD_SENS:
        row = dict(model=m, cohort=c, temperature=round(TEMP[m], 4))
        for k, lab in [("ece", "ECE"), ("brier", "Brier"), ("nll", "NLL")]:
            row[f"{lab} uncalibrated"] = cell(m, c, f"{k}_raw", 4); row[f"{lab} temperature-scaled"] = cell(m, c, f"{k}_ts", 4)
        if c != "MOONEY_TEST":
            for k, lab in [("ece_ts", "ΔECE (TS)"), ("brier_ts", "ΔBrier (TS)")]:
                r = DELTA[(DELTA.model == m) & (DELTA.ood_cohort == c) & (DELTA.metric == k)].iloc[0]
                row[lab] = fmt(r.delta_ood_minus_id, r.ci_low, r.ci_high, 4)
        t4.append(row)
pd.DataFrame(t4).to_csv(DIRS["tables"] / "table4_calibration_id_vs_ood.csv", index=False)

thr_rows = []
for m in MODELS:
    for c in ID_COHORTS[:1] + OOD_PRIMARY:
        for k in LOCKED_THRESHOLDS[m]:
            thr_rows.append(dict(model=m, cohort=c, threshold=THR_LABEL[k], sensitivity=cell(m, c, f"sens_{k}"), specificity=cell(m, c, f"spec_{k}")))
pd.DataFrame(thr_rows).to_csv(DIRS["tables"] / "tableS_threshold_sensitivity_analysis.csv", index=False)
print(pd.read_csv(DIRS["tables"] / "table3_ood_performance_and_degradation.csv")[["model", "cohort", "AUROC", "ΔAUROC", "Sensitivity", "Specificity"]].to_string(index=False))

#
# ## 9. Figures (Figure 2: discrimination ID vs OOD; Figure 3: reliability; supplementary ROC curves)

#
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 600, "figure.dpi": 120})
COLORS = {"efficientnet_b4": "#1f5aa6", "vit_b16": "#c0562b"}
NICE = {"efficientnet_b4": "EfficientNet-B4", "vit_b16": "ViT-B/16"}
SHORT = {"MOONEY_TEST": "ID test", "OOD_AP_ADULT": "NIH AP", "OOD_PA_ADULT": "NIH PA", "OOD_HARD_AP_ADULT": "NIH AP-Hard"}
def save(fig, name):
    for ext in ["png", "pdf"]: fig.savefig(DIRS["figures"] / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, (k, lab) in zip(axes, [("auroc", "AUROC"), ("auprc", "AUPRC")]):
    cs = list(SHORT)
    for j, m in enumerate(MODELS):
        x = np.arange(len(cs)) + (j - 0.5) * 0.22
        r = [MET[(MET.model == m) & (MET.cohort == c) & (MET.metric == k)].iloc[0] for c in cs]
        est = np.array([q.estimate for q in r]); lo = np.array([q.ci_low for q in r]); hi = np.array([q.ci_high for q in r])
        ax.errorbar(x, est, yerr=[est - lo, hi - est], fmt="o", ms=4, capsize=2.5, color=COLORS[m], label=NICE[m])
    if k == "auprc":
        prev = [COH[c].y.mean() for c in cs]
        ax.scatter(np.arange(len(cs)), prev, marker="_", s=200, color="grey", label="Prevalence (chance AUPRC)")
    ax.set_xticks(range(len(cs))); ax.set_xticklabels([SHORT[c] for c in cs]); ax.set_ylabel(lab); ax.set_ylim(0, 1.02)
    ax.axvline(0.5, color="lightgrey", lw=0.8, ls=":")
axes[0].legend(frameon=False, fontsize=8, loc="lower left"); axes[1].legend(frameon=False, fontsize=7, loc="upper right")
fig.suptitle("Discrimination in-distribution vs cross-population OOD (95% patient-level bootstrap CI)", fontsize=9)
fig.tight_layout(); save(fig, "figure2_id_vs_ood_discrimination")

def reliability(ax, y, p, color, label, ls="-"):
    b = np.minimum((p * ECE_BINS).astype(int), ECE_BINS - 1)
    cnt = np.bincount(b, minlength=ECE_BINS); conf = np.bincount(b, p, ECE_BINS); acc = np.bincount(b, y, ECE_BINS)
    k = cnt > 0
    ax.plot(conf[k] / cnt[k], acc[k] / cnt[k], ls, marker="o", ms=3, color=color, label=label)
fig, axes = plt.subplots(len(MODELS), len(SHORT), figsize=(9.5, 2.5 * len(MODELS)), squeeze=False)
for i, m in enumerate(MODELS):
    for j, c in enumerate(SHORT):
        ax = axes[i, j]; d = PRED[(PRED.model == m) & (PRED.cohort == c)]
        ax.plot([0, 1], [0, 1], color="lightgrey", lw=0.8)
        reliability(ax, d.y.values, d.prob.values, "grey", f"Uncal. ECE {POINT[(m, c)]['ece_raw']:.3f}", "--")
        reliability(ax, d.y.values, d.prob_ts.values, COLORS[m], f"TS ECE {POINT[(m, c)]['ece_ts']:.3f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
        if i == 0: ax.set_title(SHORT[c], fontsize=9)
        if j == 0: ax.set_ylabel(f"{NICE[m]}\nObserved frequency")
        if i == len(MODELS) - 1: ax.set_xlabel("Predicted probability")
fig.suptitle(f"Reliability diagrams ({ECE_BINS} equal-width bins); temperature fitted on Mooney validation only", fontsize=9)
fig.tight_layout(); save(fig, "figure3_reliability_diagrams")

fig, axes = plt.subplots(1, len(MODELS), figsize=(3.4 * len(MODELS), 3.2), squeeze=False)
for i, m in enumerate(MODELS):
    ax = axes[0, i]
    for c, ls in zip(SHORT, ["-", "--", "-.", ":"]):
        d = PRED[(PRED.model == m) & (PRED.cohort == c)].sort_values("logit", ascending=False)
        y = d.y.values; tpr = np.r_[0, np.cumsum(y) / y.sum()]; fpr = np.r_[0, np.cumsum(1 - y) / (1 - y).sum()]
        ax.plot(fpr, tpr, ls, lw=1.2, label=f"{SHORT[c]} ({POINT[(m, c)]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], color="lightgrey", lw=0.8); ax.set(xlabel="1 − Specificity", ylabel="Sensitivity", title=NICE[m])
    ax.legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); save(fig, "figureS_roc_curves")

#
# ## 10. Run log and ZIP

#
summary = dict(temperatures=TEMP, n_boot=N_BOOT if not SMOKE_TEST else 50, boot_seed=BOOT_SEED, ece_bins=ECE_BINS,
               thresholds_logit=LOCKED_THRESHOLDS, cohort_checks=dict(checks), env=ENV,
               headline={f"{m}|{c}": {k: POINT[(m, c)][k] for k in ["auroc", "auprc", "sens_A_primary", "spec_A_primary", "ece_raw", "ece_ts", "brier_ts"]}
                         for m in MODELS for c in ID_COHORTS[:1] + OOD_PRIMARY})
(DIRS["metrics"] / "notebook03_summary.json").write_text(json.dumps(summary, indent=2, default=str))
zip_base = PROJECT / "phase3_evaluation_outputs_versionA"
tmp = PROJECT / "_zip03"; shutil.rmtree(tmp, ignore_errors=True)
for k in ["predictions", "metrics", "figures", "tables", "logs", "configs"]:
    shutil.copytree(DIRS[k], tmp / k, dirs_exist_ok=True)
shutil.make_archive(str(zip_base), "zip", tmp); shutil.rmtree(tmp)
print("ZIP →", f"{zip_base}.zip", f"({os.path.getsize(f'{zip_base}.zip') / 1e6:.0f} MB)")
print("Phases 3–4 complete. Review the outputs before Notebook 04 (MC Dropout).")
