"""Phase 9 — publication figures, rebuilt from the verified output files (no new modelling)."""
import json, shutil, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT = Path("results/figures"); SUP = Path("results/supp")
P3 = Path("outputs/notebook03"); P4 = Path("outputs/notebook04"); P5 = Path("outputs/notebook05")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#555555",
                     "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "legend.frameon": False, "legend.fontsize": 7, "lines.linewidth": 1.6, "savefig.dpi": 600,
                     "figure.dpi": 110, "pdf.fonttype": 42})
INK, MUTED = "#222222", "#888888"
MCOL = {"efficientnet_b4": "#2a78d6", "vit_b16": "#eb6834"}
NICE = {"efficientnet_b4": "EfficientNet-B4", "vit_b16": "ViT-B/16"}
MAIN = ["MOONEY_TEST", "OOD_AP_ADULT", "OOD_PA_ADULT", "OOD_HARD_AP_ADULT"]
SHORT = {"MOONEY_TEST": "ID test (Mooney)", "OOD_AP_ADULT": "NIH AP", "OOD_PA_ADULT": "NIH PA", "OOD_HARD_AP_ADULT": "NIH AP-Hard"}
CCOL = {"MOONEY_TEST": INK, "OOD_AP_ADULT": "#2a78d6", "OOD_PA_ADULT": "#eb6834", "OOD_HARD_AP_ADULT": "#1baf7a"}
CLS = {"MOONEY_TEST": "-", "OOD_AP_ADULT": "--", "OOD_PA_ADULT": "-.", "OOD_HARD_AP_ADULT": ":"}
MODELS = ["efficientnet_b4", "vit_b16"]
def save(fig, name, folder=OUT):
    for ext in ["png", "pdf"]: fig.savefig(folder / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)

MET = pd.read_csv(P3 / "metrics/metrics_with_ci_long.csv")
PRED = pd.read_csv(P3 / "predictions/predictions_long.csv.gz")
U = pd.read_csv(P4 / "predictions/uncertainty_per_image.csv.gz")
STAT = pd.read_csv(P4 / "metrics/uncertainty_metrics_with_ci_long.csv")
SEL = pd.read_csv(P4 / "metrics/selective_prediction_validation_thresholds.csv")
TAU = json.load(open(P4 / "metrics/notebook04_summary.json"))["referral_threshold"]
TEMP = json.load(open(P3 / "configs/temperature_scaling.json"))["temperatures"]
def met(m, c, k): return MET[(MET.model == m) & (MET.cohort == c) & (MET.metric == k)].iloc[0]
def stat(m, c, meas, k): return STAT[(STAT.model == m) & (STAT.cohort == c) & (STAT.measure == meas) & (STAT.metric == k)].iloc[0]

# ---------------------------------------------------------------- Figure 1: study workflow
fig, ax = plt.subplots(figsize=(7.2, 3.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off"); ax.grid(False)
def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec="#666666", lw=0.7))
    ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.2, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 6.3, body, ha="center", va="top", fontsize=6.0, color=INK, linespacing=1.35)
def arrow(x1, y1, x2, y2): ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.8))
box(1, 27, 22, 21, "Development data", "Mooney/Kermany paediatric CXR\ntrain + val → audited pool\npatient-grouped split\ntrain 4,468 · val 738", "#eef4fc")
box(1, 7, 22, 16, "Untouched ID test", "original Mooney test\n624 images\n427 heuristic patients\n+2 sensitivity subsets", "#eef4fc")
box(27, 27, 21, 21, "Training", "EfficientNet-B4 (384 px)\nViT-B/16 (224 px)\nImageNet-pretrained, BCE\ndropout head (MC layers)", "#f4f4f2")
box(27, 7, 21, 16, "Validation-only", "threshold A\n(max sens | spec ≥ 0.95)\ntemperature T\nreferral τ (90 % cov.)", "#f4f4f2")
box(52, 2, 21, 46, "Frozen evaluation", "ID test\n\nNIH ChestX-ray14\nadults 18–100\n(external only; never used\nfor fitting or selection)\n• AP controlled 20,588\n• PA controlled 38,038\n• AP-Hard 18,736\n\nall-age cohorts =\nsensitivity analyses", "#fdf1ec")
box(77, 2, 22, 46, "Analyses", "discrimination\n(AUROC, AUPRC, sens, spec)\n\ncalibration (ECE, Brier, NLL)\n\nMC Dropout uncertainty\nerror & OOD detection\n\nselective referral\n(risk–coverage, AURC)\n\nGrad-CAM + lung masks\n(Lung Attention Ratio)\n\npatient-level bootstrap CIs", "#eefaf4")
arrow(23.5, 37.5, 26.5, 37.5); arrow(37.5, 26.5, 37.5, 23.6)
ax.annotate("", xy=(51.5, 4.0), xytext=(12, 6.5), arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.8, connectionstyle="angle,angleA=-90,angleB=180,rad=0"))
arrow(48.5, 37.5, 51.5, 37.5); arrow(48.5, 15, 51.5, 15)
arrow(73.5, 25, 76.5, 25)
save(fig, "figure1_study_workflow")

# ---------------------------------------------------------------- Figure 2: discrimination ID vs OOD
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, (k, lab) in zip(axes, [("auroc", "AUROC"), ("auprc", "AUPRC")]):
    for j, m in enumerate(MODELS):
        x = np.arange(len(MAIN)) + (j - 0.5) * 0.24
        r = [met(m, c, k) for c in MAIN]
        est = np.array([q.estimate for q in r]); lo = np.array([q.ci_low for q in r]); hi = np.array([q.ci_high for q in r])
        ax.errorbar(x, est, yerr=[est - lo, hi - est], fmt="o" if j == 0 else "s", ms=5, capsize=2.5, elinewidth=1, color=MCOL[m], label=NICE[m])
    if k == "auroc": ax.axhline(0.5, color=MUTED, lw=0.8, ls="--"); ax.text(1.5, 0.465, "chance", fontsize=6.5, color=MUTED, ha="center")
    else:
        prev = [PRED[(PRED.model == MODELS[0]) & (PRED.cohort == c)].y.mean() for c in MAIN]
        ax.scatter(np.arange(len(MAIN)), prev, marker="_", s=260, color=MUTED, lw=1.5, label="prevalence (chance)", zorder=1)
    ax.set_xticks(range(len(MAIN))); ax.set_xticklabels([SHORT[c].replace(" (Mooney)", "") for c in MAIN]); ax.set_ylabel(lab); ax.set_ylim(0, 1.02)
    ax.axvspan(-0.5, 0.5, color="#f3f3f3", zorder=0, lw=0)
axes[0].legend(loc="lower left"); axes[1].legend(loc="upper right")
fig.tight_layout(); save(fig, "figure2_id_vs_ood_discrimination")

# ---------------------------------------------------------------- Figure 3: reliability with bin counts
NB = 15; edges = np.linspace(0, 1, NB + 1); mids = (edges[:-1] + edges[1:]) / 2
fig = plt.figure(figsize=(7.4, 5.0)); gs = fig.add_gridspec(4, 4, height_ratios=[3, 1, 3, 1], hspace=0.35, wspace=0.28)
for i, m in enumerate(MODELS):
    for j, c in enumerate(MAIN):
        ax = fig.add_subplot(gs[2 * i, j]); axh = fig.add_subplot(gs[2 * i + 1, j], sharex=ax)
        d = PRED[(PRED.model == m) & (PRED.cohort == c)]
        ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.8)
        for col, ls, mk, lab, colr in [("prob", "--", "o", "uncalibrated", MUTED), ("prob_ts", "-", "s", f"temperature-scaled (T={TEMP[m]:.2f})", MCOL[m])]:
            b = np.minimum((d[col].values * NB).astype(int), NB - 1)
            cnt = np.bincount(b, minlength=NB); conf = np.bincount(b, d[col].values, NB); acc = np.bincount(b, d.y.values, NB)
            ok = cnt >= 10
            ax.plot(conf[ok] / cnt[ok], acc[ok] / cnt[ok], ls, marker=mk, ms=3, color=colr, lw=1.2, label=lab)
            if col == "prob_ts": axh.bar(mids, cnt, width=1 / NB * 0.9, color=colr, alpha=0.8)
        e_raw, e_ts = met(m, c, "ece_raw").estimate, met(m, c, "ece_ts").estimate
        ax.text(0.03, 0.97, f"ECE {e_raw:.3f} → {e_ts:.3f}", transform=ax.transAxes, va="top", fontsize=6.5, color=INK)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); plt.setp(ax.get_xticklabels(), visible=False)
        axh.set_yscale("log"); axh.set_ylim(0.8, None); axh.tick_params(labelsize=6); axh.grid(False)
        if i == 0: ax.set_title(SHORT[c])
        if j == 0: ax.set_ylabel(f"{NICE[m]}\nobserved frequency"); axh.set_ylabel("n", fontsize=6.5)
        if i == len(MODELS) - 1: axh.set_xlabel("predicted probability")
        if i < len(MODELS) - 1: plt.setp(axh.get_xticklabels(), visible=False)
        if i == 0 and j == 0: LEG = ax.get_legend_handles_labels()
fig.legend([LEG[0][0], LEG[0][1]], ["uncalibrated", "temperature-scaled (T fitted on Mooney validation)"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.045), fontsize=6.5)
fig.text(0.5, -0.075, "Points shown only for bins with ≥ 10 images; histograms (log scale) show temperature-scaled probability counts. Temperature fitted on Mooney validation only.",
         ha="center", fontsize=6.3, color=INK)
save(fig, "figure3_reliability_diagrams")

# ---------------------------------------------------------------- Figures 4–5: uncertainty (log axis)
FLOOR = 1e-12
def ecdf(ax, x, **kw):
    x = np.sort(np.clip(x, FLOOR, None)); ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", **kw)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
for ax, m in zip(axes, MODELS):
    for c in MAIN:
        lab = SHORT[c] if c == "MOONEY_TEST" else f"{SHORT[c]} (OOD-AUROC {stat(m, c, 'entropy_mc', 'ooddet_auroc').estimate:.2f})"
        ecdf(ax, U[(U.model == m) & (U.cohort == c)].entropy_mc.values, color=CCOL[c], ls=CLS[c], label=lab)
    ax.axvline(TAU[m], color=MUTED, lw=0.8); ax.text(TAU[m] * 1.3, 0.04, "referral\nthreshold τ", fontsize=6, color=MUTED)
    ax.set_xscale("log"); ax.set_xlim(1e-10, 1.2); ax.set_xlabel("MC-Dropout predictive entropy (bits, log scale)"); ax.set_title(NICE[m])
    ax.legend(loc="upper left", fontsize=6.2)
axes[0].set_ylabel("cumulative fraction of images")
fig.tight_layout(); save(fig, "figure4_uncertainty_id_vs_ood")

fig, axes = plt.subplots(2, 4, figsize=(7.4, 4.0), sharex=True, sharey=True)
for i, m in enumerate(MODELS):
    for j, c in enumerate(MAIN):
        ax = axes[i, j]; d = U[(U.model == m) & (U.cohort == c)]
        for e, col, ls, lab in [(0, "#2a78d6", "-", "correct"), (1, "#eb6834", "--", "incorrect")]:
            x = d[d.error == e].entropy_mc.values
            if len(x): ecdf(ax, x, color=col, ls=ls, label=f"{lab} (n={len(x):,})")
        ax.axvline(TAU[m], color=MUTED, lw=0.8)
        r = stat(m, c, "entropy_mc", "errdet_auroc")
        ax.set_title(f"{SHORT[c].replace(' (Mooney)', '')}\nerror-det. AUROC {r.estimate:.2f} ({r.ci_low:.2f}–{r.ci_high:.2f})", fontsize=7)
        ax.set_xscale("log"); ax.set_xlim(1e-10, 1.2); ax.legend(loc="upper left", fontsize=5.6)
        if j == 0: ax.set_ylabel(f"{NICE[m]}\ncumulative fraction")
        if i == 1: ax.set_xlabel("entropy (bits, log)")
fig.tight_layout(); save(fig, "figure5_uncertainty_correct_vs_incorrect")

# ---------------------------------------------------------------- Figure 6: risk–coverage
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, m in zip(axes, MODELS):
    for c in MAIN:
        d = U[(U.model == m) & (U.cohort == c)].sort_values("entropy_mc", kind="mergesort")
        n = np.arange(1, len(d) + 1); a = stat(m, c, "entropy_mc", "aurc")
        ax.plot(n / len(d), np.cumsum(d.error.values) / n, color=CCOL[c], ls=CLS[c], lw=1.4, label=f"{SHORT[c].replace(' (Mooney)', '')}: AURC {a.estimate:.3f}")
        s = SEL[(SEL.model == m) & (SEL.cohort == c) & (SEL.measure == "entropy_mc") & (SEL.nominal_coverage_pct == 90)].iloc[0]
        ax.plot(s.coverage, s.selective_risk, "o", color=CCOL[c], ms=5.5, mec="white", mew=1, zorder=5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("coverage (fraction accepted by AI)"); ax.set_title(NICE[m])
    ax.legend(loc="center right", fontsize=6.2)
axes[0].set_ylabel("selective risk (error rate among accepted)")
fig.text(0.5, -0.03, "Curves rank images by MC-Dropout entropy within each cohort; dots mark the validation-derived referral threshold (nominal 90 % coverage).",
         ha="center", fontsize=6.3)
fig.tight_layout(); save(fig, "figure6_risk_coverage")

# ---------------------------------------------------------------- Figures 7–8 (Notebook 05) and Figure 8 rebuilt in house style
for m in MODELS:
    for ext in ["png", "pdf"]: shutil.copy(P5 / f"figures/figure7_gradcam_safety_cases_{m}.{ext}", OUT / f"figure7_gradcam_safety_cases_{m}.{ext}")
LAR = pd.read_csv(P5 / "metrics/gradcam_lar_per_image.csv"); SMP = pd.read_csv(P5 / "gradcam/quantitative_sample.csv")
Q = SMP.merge(LAR, on=["model", "filepath"]); Q = Q[Q.qc_pass & Q.lar.notna()]
GROUPS = ["G1_correct_low_unc", "G2_correct_high_unc", "G3_incorrect_high_unc", "G4_incorrect_low_unc"]
GLAB = ["G1 correct · low unc.", "G2 correct · high unc.", "G3 incorrect · high unc.", "G4 incorrect · low unc. (confidently wrong)"]
GCOL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), sharey=True)
for ax, m in zip(axes, MODELS):
    for j, c in enumerate(MAIN):
        for k, g in enumerate(GROUPS):
            v = Q[(Q.model == m) & (Q.cohort == c) & (Q.safety_group == g)].lar.values
            if len(v) >= 5:
                bp = ax.boxplot([v], positions=[j * 5 + k], widths=0.75, patch_artist=True, showfliers=False,
                                medianprops=dict(color=INK, lw=1), whiskerprops=dict(color="#666666", lw=0.7), capprops=dict(color="#666666", lw=0.7))
                bp["boxes"][0].set(facecolor=GCOL[k], alpha=0.85, edgecolor="#555555", lw=0.6, hatch="////" if k == 3 else None)
        lf = Q[(Q.model == m) & (Q.cohort == c)].lung_fraction.mean()
        ax.hlines(lf, j * 5 - 0.6, j * 5 + 3.6, colors=INK, linestyles=":", lw=1)
    ax.set_xticks([j * 5 + 1.5 for j in range(len(MAIN))]); ax.set_xticklabels([SHORT[c].replace(" (Mooney)", "") for c in MAIN])
    ax.set_ylim(0, 1); ax.set_title(NICE[m]); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("Lung Attention Ratio (Grad-CAM mass inside lungs)")
handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#555555", lw=0.6, hatch="////" if i == 3 else None) for i, c in enumerate(GCOL)] + [plt.Line2D([], [], color=INK, ls=":")]
fig.legend(handles, GLAB + ["lung fraction (LAR of a uniform map)"], loc="lower center", ncol=3, fontsize=6.3)
fig.tight_layout(rect=(0, 0.12, 1, 1)); save(fig, "figure8_lung_attention_by_safety_group")

# ---------------------------------------------------------------- Supplementary: ROC curves, aspect-ratio post-hoc, mask QC
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
for ax, m in zip(axes, MODELS):
    for c in MAIN:
        d = PRED[(PRED.model == m) & (PRED.cohort == c)].sort_values("logit", ascending=False)
        y = d.y.values; ax.plot(np.r_[0, np.cumsum(1 - y) / (1 - y).sum()], np.r_[0, np.cumsum(y) / y.sum()], color=CCOL[c], ls=CLS[c],
                                label=f"{SHORT[c]} (AUROC {met(m, c, 'auroc').estimate:.3f})")
    ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=0.8); ax.set_xlabel("1 − specificity"); ax.set_title(NICE[m]); ax.legend(loc="lower right", fontsize=6.3)
axes[0].set_ylabel("sensitivity"); fig.tight_layout(); save(fig, "figureS1_roc_curves", SUP)

man = pd.read_csv("outputs/notebook01/mooney_manifest_final.csv"); man["aspect"] = man.width / man.height
FPQ = pd.read_csv(SUP / "tableS_fp_rate_by_aspect_quartile.csv"); AUC = pd.read_csv(SUP / "tableS_aspect_ratio_auroc.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = axes[0]
for i, (s, lab) in enumerate([("train", "Training"), ("val", "Validation"), ("test_id", "ID test")]):
    for k, (yv, col) in enumerate([(0, "#2a78d6"), (1, "#eb6834")]):
        v = man[(man.final_split == s) & (man.y == yv)].aspect.values
        bp = ax.boxplot([v], positions=[i * 3 + k], widths=0.7, patch_artist=True, showfliers=False, medianprops=dict(color=INK))
        bp["boxes"][0].set(facecolor=col, alpha=0.85, edgecolor="#555555", lw=0.6)
    a = AUC[AUC.split == lab].iloc[0]
    ax.text(i * 3 + 0.5, 2.2, f"AUROC\n{a.auroc_aspect_only:.2f}\n({a.ci_low:.2f}–{a.ci_high:.2f})", ha="center", fontsize=6, color=INK)
ax.set_xticks([0.5, 3.5, 6.5]); ax.set_xticklabels(["Training", "Validation", "ID test"]); ax.set_ylabel("image aspect ratio (width / height)")
ax.set_ylim(0.6, 2.6); ax.set_title("A  Aspect ratio by label (Mooney)", loc="left")
ax.legend([plt.Rectangle((0, 0), 1, 1, fc="#2a78d6"), plt.Rectangle((0, 0), 1, 1, fc="#eb6834")], ["Normal", "Pneumonia"], loc="lower right", fontsize=6.3, ncol=2)
ax = axes[1]
for j, m in enumerate(MODELS):
    d = FPQ[(FPQ.model == m) & (FPQ.cohort == "MOONEY_TEST") & FPQ.aspect_quartile.str.startswith("Q")]
    x = np.arange(4) + (j - 0.5) * 0.25
    ax.errorbar(x, d.false_positive_rate, yerr=[d.false_positive_rate - d.ci_low, d.ci_high - d.false_positive_rate], fmt="o" if j == 0 else "s",
                ms=5, capsize=2.5, color=MCOL[m], label=NICE[m])
ax.set_xticks(range(4)); ax.set_xticklabels(["Q1\nnarrowest", "Q2", "Q3", "Q4\nwidest"]); ax.set_ylim(0, 1)
ax.set_ylabel("false-positive rate among NORMAL"); ax.set_title("B  ID-test normals by aspect quartile", loc="left"); ax.legend(loc="upper left", fontsize=6.3)
fig.text(0.5, -0.04, "Post-hoc exploratory analysis (not pre-specified), prompted by Grad-CAM attention on letterbox padding. Associative evidence only.",
         ha="center", fontsize=6.3)
fig.tight_layout(); save(fig, "figureS2_aspect_ratio_shortcut_posthoc", SUP)
shutil.copy(P5 / "figures/figureS_lung_mask_qc.png", SUP / "figureS3_lung_mask_qc.png")
print("figures done")
