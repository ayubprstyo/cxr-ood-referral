"""Phase 9 — publication tables built only from verified output files."""
import json, pandas as pd, numpy as np
from pathlib import Path
T = Path("results/tables"); S = Path("results/supp")
P3 = Path("outputs/notebook03"); P4 = Path("outputs/notebook04"); P5 = Path("outputs/notebook05")
NICE = {"efficientnet_b4": "EfficientNet-B4", "vit_b16": "ViT-B/16"}
CNAME = {"MOONEY_VAL": "Mooney validation", "MOONEY_TEST": "Mooney ID test", "MOONEY_TEST_NONCOLLIDING": "ID test, non-colliding IDs",
         "MOONEY_TEST_DEDUP": "ID test, de-duplicated", "OOD_AP_ADULT": "NIH AP (adult)", "OOD_PA_ADULT": "NIH PA (adult)",
         "OOD_HARD_AP_ADULT": "NIH AP-Hard (adult)", "OOD_AP_ALLAGE": "NIH AP (all ages)", "OOD_PA_ALLAGE": "NIH PA (all ages)",
         "OOD_HARD_AP_ALLAGE": "NIH AP-Hard (all ages)"}
MAIN = ["MOONEY_TEST", "OOD_AP_ADULT", "OOD_PA_ADULT", "OOD_HARD_AP_ADULT"]
def nm(df):
    if "model" in df: df["model"] = df.model.map(NICE).fillna(df.model)
    if "cohort" in df: df["cohort"] = df.cohort.map(CNAME).fillna(df.cohort)
    return df.rename(columns={"model": "Model", "cohort": "Cohort"})

# Table 1
t1 = pd.read_csv(P3 / "tables/table1_cohort_characteristics.csv")
man = pd.read_csv("outputs/notebook01/mooney_manifest_final.csv"); tr = man[man.final_split == "train"]
t1 = pd.concat([pd.DataFrame([dict(cohort="Mooney training", role="model fitting", images=len(tr), positives=int(tr.y.sum()),
                                   negatives=int((tr.y == 0).sum()), prevalence_pct=round(100 * tr.y.mean(), 2), patients=tr.patient_id.nunique())]), t1])
t1["cohort"] = t1.cohort.map(CNAME).fillna(t1.cohort)
t1 = t1.rename(columns=dict(cohort="Cohort", role="Role", images="Images", positives="Pneumonia", negatives="Negative",
                            prevalence_pct="Prevalence (%)", patients="Patients", age_median_iqr="Age, median (IQR)", female_pct="Female (%)"))
t1 = t1.fillna("—"); t1.to_csv(T / "table1_cohort_characteristics.csv", index=False)

# Table 2
t2 = nm(pd.read_csv(P3 / "tables/table2_id_performance.csv")); t2.to_csv(T / "table2_id_performance.csv", index=False)

# Table 3 (primary) + S (all ages)
t3 = pd.read_csv(P3 / "tables/table3_ood_performance_and_degradation.csv")
cols = ["model", "cohort", "AUROC", "AUPRC", "Sensitivity", "Specificity", "PPV", "NPV", "ΔAUROC", "ΔAUPRC", "ΔSensitivity", "ΔSpecificity"]
nm(t3[t3.cohort.str.endswith("_ADULT")][cols].copy()).to_csv(T / "table3_ood_performance_degradation.csv", index=False)
nm(t3[t3.cohort.str.endswith("_ALLAGE")][cols].copy()).to_csv(S / "tableS_ood_all_ages_sensitivity.csv", index=False)

# Table 4
t4 = pd.read_csv(P3 / "tables/table4_calibration_id_vs_ood.csv")
t4 = t4[t4.cohort.isin(MAIN)]; nm(t4).to_csv(T / "table4_calibration.csv", index=False)

# Table 5
t5 = pd.read_csv(P4 / "tables/table5_uncertainty_characteristics.csv"); nm(t5).to_csv(T / "table5_uncertainty_characteristics.csv", index=False)

# Table 6
t6 = pd.read_csv(P4 / "tables/table6_error_and_ood_detection.csv")
t6["measure"] = t6.measure.map({"entropy_mc": "MC predictive entropy (primary)", "mi_mc": "MC mutual information", "var_mc": "MC variance",
                                "entropy_det": "Deterministic entropy (baseline)"})
t6 = t6[t6.measure != "MC variance"].rename(columns={"measure": "Uncertainty score"})
nm(t6).to_csv(T / "table6_error_ood_detection.csv", index=False)

# Table 7 (main: 100/90/70 %) + supplementary full
SEL = pd.read_csv(P4 / "metrics/selective_prediction_validation_thresholds.csv")
SEL = SEL[(SEL.measure == "entropy_mc") & SEL.cohort.isin(MAIN)].copy()
stat = pd.read_csv(P4 / "metrics/uncertainty_metrics_with_ci_long.csv")
def aurc(m, c):
    a = stat[(stat.model == m) & (stat.cohort == c) & (stat.measure == "entropy_mc") & (stat.metric == "aurc")].iloc[0]
    o = stat[(stat.model == m) & (stat.cohort == c) & (stat.measure == "entropy_mc") & (stat.metric == "aurc_optimal")].iloc[0]
    return f"{a.estimate:.3f} ({a.ci_low:.3f}–{a.ci_high:.3f})", round(a.estimate - o.estimate, 3)
rows = []
for _, r in SEL.iterrows():
    a, e = aurc(r.model, r.cohort)
    rows.append({"model": r.model, "cohort": r.cohort, "Nominal coverage (%)": r.nominal_coverage_pct,
                 "Achieved coverage": f"{r.coverage:.3f} ({r.coverage_ci})", "Selective risk": f"{r.selective_risk:.3f} ({r.selective_risk_ci})",
                 "Sensitivity (accepted)": f"{r.sens_accepted:.3f}", "Specificity (accepted)": f"{r.spec_accepted:.3f}",
                 "Retained / referred": f"{r.n_retained:,} / {r.n_referred:,}", "Errors referred (%)": f"{r.errors_referred_pct:.1f}",
                 "AURC (95% CI)": a, "E-AURC": e})
t7 = nm(pd.DataFrame(rows))
t7.to_csv(S / "tableS_selective_prediction_all_coverages.csv", index=False)
t7[t7["Nominal coverage (%)"].isin([100, 90, 70])].to_csv(T / "table7_selective_prediction.csv", index=False)

# Table 8
t8 = pd.read_csv(P5 / "tables/table8_quantitative_xai_lung_attention.csv")
keep = ["model", "cohort", "n_images", "lung_fraction_mean", "LAR Correct", "LAR Incorrect", "LAR Low uncertainty", "LAR High uncertainty",
        "LAR Correct, low uncertainty (G1)", "LAR Confidently incorrect (G4)", "Δ LAR Incorrect − Correct",
        "Δ LAR Confidently incorrect (G4) − Correct, low uncertainty (G1)"]
t8 = nm(t8[keep].rename(columns={"n_images": "Images analysed", "lung_fraction_mean": "Lung fraction (uniform-map LAR)"}))
t8.to_csv(T / "table8_lung_attention.csv", index=False)

# Supplementary
nm(pd.read_csv(P3 / "tables/tableS_threshold_sensitivity_analysis.csv")).to_csv(S / "tableS_threshold_operating_points.csv", index=False)
a = pd.read_csv(P3 / "metrics/architecture_comparison.csv")
a["Difference (95% CI)"] = [f"{e:.3f} ({l:.3f}–{h:.3f})" for e, l, h in zip(a.estimate, a.ci_low, a.ci_high)]
a["DeLong p"] = a.delong_p.map(lambda p: "" if pd.isna(p) else ("<0.0001" if p < 1e-4 else f"{p:.4f}"))
nm(a[["cohort", "metric", "difference", "Difference (95% CI)", "DeLong p"]].copy()).to_csv(S / "tableS_architecture_comparison.csv", index=False)
nm(pd.read_csv(P4 / "tables/tableS_confidently_incorrect.csv")).to_csv(S / "tableS_confidently_incorrect.csv", index=False)
nm(pd.read_csv(P5 / "metrics/lung_mask_qc.csv")).to_csv(S / "tableS_lung_mask_qc.csv", index=False)
print("tables done")
