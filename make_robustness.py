"""Robustness (Notebook 06) — supplementary tables S9–S11 and Figure 9, built only from the Notebook 06 output files."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
P6 = Path("outputs/notebook06"); F = Path("results")
R = pd.read_csv(P6 / "metrics/robustness_metrics_with_ci_long.csv")
NICE = {"efficientnet_b4": "EfficientNet-B4", "vit_b16": "ViT-B/16"}
CN = {"MOONEY_TEST": "Mooney ID test", "OOD_AP_ADULT": "NIH AP (adult)", "OOD_PA_ADULT": "NIH PA (adult)", "OOD_HARD_AP_ADULT": "NIH AP-Hard (adult)"}
EVAL = list(CN)
def lab(run):
    m, v = run.split("__"); pre, s = v.split("_s")
    return NICE[m], ("Letterbox" if pre == "lb" else "No padding") + f", seed {s}" + (" (primary)" if v == "lb_s42" else "")
def g(run, c, k): r = R[(R.run == run) & (R.cohort == c) & (R.metric == k)].iloc[0]; return r.estimate, r.ci_low, r.ci_high
fm = lambda e, l, h, d=3: (f"{e:.{d}f} ({l:.{d}f}–{h:.{d}f})" if min(l, h) >= 0 else f"{e:.{d}f} ({l:.{d}f} to {h:.{d}f})").replace("-", "−")
ORDER = [f"{m}__{v}" for m in NICE for v in ["lb_s42", "lb_s43", "lb_s44", "st_s42"]]
OPS = pd.read_csv(P6 / "tables/robustness_operating_points.csv", index_col=0)

rows = []
for run in ORDER:
    mo, va = lab(run)
    for c in EVAL:
        rows.append({"Model": mo, "Run": va, "Cohort": CN[c], "AUROC (95% CI)": fm(*g(run, c, "auroc")),
                     "Specificity at A": f"{g(run, c, 'spec_A')[0]:.3f}", "Error-detection AUROC (95% CI)": fm(*g(run, c, "errdet_auroc")),
                     "Error rate, no referral": f"{g(run, c, 'error_rate')[0]:.3f}", "Selective risk at τ90": f"{g(run, c, 'selective_risk')[0]:.3f}",
                     "Confidently incorrect (% of errors)": f"{g(run, c, 'g4_pct_of_errors')[0]:.1f}"})
pd.DataFrame(rows).to_csv(F / "supp/tableS9_robustness_per_run.csv", index=False)
op = OPS.loc[ORDER].reset_index().rename(columns={"index": "run"})
op.insert(0, "Run", [lab(r)[1] for r in op.run]); op.insert(0, "Model", [lab(r)[0] for r in op.run])
op = op[["Model", "Run", "epoch", "thrA", "T", "tau90"]].rename(columns={"epoch": "Selected epoch", "thrA": "Threshold A (logit)", "T": "Temperature", "tau90": "τ90 (bits)"})
op["Threshold A (logit)"] = op["Threshold A (logit)"].astype(float).round(3); op["Temperature"] = op["Temperature"].astype(float).round(3)
op["τ90 (bits)"] = op["τ90 (bits)"].astype(float).map(lambda v: f"{v:.4f}")
op.to_csv(F / "supp/tableS10_robustness_operating_points.csv", index=False)

PAD = pd.read_csv(P6 / "tables/R2_padding_ablation.csv")
MN = {"auroc": "AUROC", "spec_A": "Specificity at A", "sens_A": "Sensitivity at A", "ece_ts": "ECE (TS)", "errdet_auroc": "Error-detection AUROC",
      "risk_change": "Selective risk − error rate", "g4_pct_of_errors": "Confidently incorrect (% of errors)"}
PAD = PAD.assign(Model=PAD.model.map(NICE), Cohort=PAD.cohort.map(CN), Metric=PAD.metric.map(MN))
d = 1
PAD["Letterbox"] = PAD.letterbox.map(lambda v: f"{v:.3f}"); PAD["No padding"] = PAD.stretch.map(lambda v: f"{v:.3f}")
PAD["Difference (95% CI)"] = [fm(e, l, h) for e, l, h in zip(PAD.difference, PAD.ci_low, PAD.ci_high)]
PAD[["Model", "Cohort", "Metric", "Letterbox", "No padding", "Difference (95% CI)"]].to_csv(F / "supp/tableS11_padding_ablation.csv", index=False)

# aspect-ratio gradient: per-run quartiles + paired change of (Q4 − Q1), stretch − letterbox
P = pd.read_csv(P6 / "predictions/robustness_per_image.csv.gz", usecols=["run", "cohort", "filepath", "patient_id", "y", "aspect", "pred"])
P = P[P.cohort == "MOONEY_TEST"]
rng = np.random.default_rng(20261030); qrows = []; GRAD = {}
for m in NICE:
    a = P[P.run == f"{m}__lb_s42"].reset_index(drop=True); b = P[P.run == f"{m}__st_s42"].reset_index(drop=True)
    assert (a.filepath.values == b.filepath.values).all()
    nm = (a.y == 0).values; e = np.quantile(a.aspect[nm], [0, .25, .5, .75, 1]); e[-1] += 1e-9
    q = np.clip(np.searchsorted(e, a.aspect, side="right") - 1, 0, 3); codes, u = pd.factorize(a.patient_id)
    fp = lambda pr, w: np.array([(w * nm * (q == k) * pr).sum() / (w * nm * (q == k)).sum() for k in range(4)])
    W = [np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[codes].astype(float) for _ in range(2000)]
    BA = np.array([fp(a.pred.values, w) for w in W]); BB = np.array([fp(b.pred.values, w) for w in W])
    pa, pb = fp(a.pred.values, np.ones(len(a))), fp(b.pred.values, np.ones(len(a)))
    for nmv, pt, bs in [("Letterbox", pa, BA), ("No padding", pb, BB)]:
        r = {"Model": NICE[m], "Preprocessing": nmv}
        for k in range(4): r[f"Q{k + 1}"] = fm(pt[k], *np.percentile(bs[:, k], [2.5, 97.5]))
        r["Q4 − Q1"] = fm(pt[3] - pt[0], *np.percentile(bs[:, 3] - bs[:, 0], [2.5, 97.5])); qrows.append(r)
    ch = (pb[3] - pb[0]) - (pa[3] - pa[0]); chb = (BB[:, 3] - BB[:, 0]) - (BA[:, 3] - BA[:, 0])
    GRAD[m] = dict(lb=pa[3] - pa[0], st=pb[3] - pb[0], change=ch, lo=np.percentile(chb, 2.5), hi=np.percentile(chb, 97.5))
    qrows.append({"Model": NICE[m], "Preprocessing": "Change (no padding − letterbox)", "Q4 − Q1": fm(ch, *np.percentile(chb, [2.5, 97.5]))})
pd.DataFrame(qrows).fillna("").to_csv(F / "supp/tableS12_aspect_gradient_padding.csv", index=False)
pd.DataFrame(GRAD).T.to_csv(F / "supp/_aspect_gradient_change.csv")

# ---------------- Figure 9
COL = {"efficientnet_b4": "#2a78d6", "vit_b16": "#eb6834"}
MK = {"lb_s42": ("o", True), "lb_s43": ("s", True), "lb_s44": ("D", True), "st_s42": ("^", False)}
SH = {"MOONEY_TEST": "ID\ntest", "OOD_AP_ADULT": "NIH\nAP", "OOD_PA_ADULT": "NIH\nPA", "OOD_HARD_AP_ADULT": "NIH\nAP-Hard"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#555555", "xtick.color": "#333333", "ytick.color": "#333333"})
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
panels = [("auroc", "AUROC", 0.5), ("errdet_auroc", "Error-detection AUROC", 0.5), ("risk_change", "Selective risk − error rate", 0.0)]
for ax, (k, ylab, ref) in zip(axs, panels):
    ax.axhline(ref, color="#999999", lw=0.7, ls=":", zorder=0)
    for j, m in enumerate(NICE):
        for i, v in enumerate(MK):
            run = f"{m}__{v}"; mk, filled = MK[v]
            x = np.arange(4) + (j - 0.5) * 0.42 + (i - 1.5) * 0.08
            e = [g(run, c, k) for c in EVAL]; est = np.array([t[0] for t in e])
            ax.errorbar(x, est, yerr=[est - [t[1] for t in e], [t[2] for t in e] - est], fmt="none", ecolor=COL[m], elinewidth=0.7, zorder=1)
            ax.scatter(x, est, marker=mk, s=16, facecolor=COL[m] if filled else "white", edgecolor=COL[m] if filled else COL[m], linewidth=0.9, zorder=2)
    ax.set_xticks(range(4)); ax.set_xticklabels([SH[c] for c in EVAL], rotation=0, fontsize=7); ax.set_ylabel(ylab)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
for ax, t in zip(axs, "ABC"): ax.text(-0.18, 1.04, t, transform=ax.transAxes, fontweight="bold", fontsize=9)
h = [Line2D([], [], marker="o", ls="", color=COL["efficientnet_b4"], label="EfficientNet-B4"), Line2D([], [], marker="o", ls="", color=COL["vit_b16"], label="ViT-B/16"),
     Line2D([], [], marker="o", ls="", mfc="#555555", mec="#555555", label="seed 42 (primary)"), Line2D([], [], marker="s", ls="", mfc="#555555", mec="#555555", label="seed 43"),
     Line2D([], [], marker="D", ls="", mfc="#555555", mec="#555555", label="seed 44"), Line2D([], [], marker="^", ls="", mfc="white", mec="#555555", label="no padding, seed 42")]
fig.legend(handles=h, loc="lower center", ncol=6, frameon=False, fontsize=7, handletextpad=0.2, columnspacing=1.0)
fig.tight_layout(rect=(0, 0.08, 1, 1))
for ext, dpi in [("png", 600), ("pdf", None)]: fig.savefig(F / f"figures/figure9_robustness.{ext}", dpi=dpi)
print(pd.DataFrame(GRAD).T.round(3))
