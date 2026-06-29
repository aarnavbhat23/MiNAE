from __future__ import annotations

from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


BASE = Path("/Users/aarnav/Documents/MiNAE_Global_Local")
ANN = BASE / "external_data" / "annapragada2025"
FODA = BASE / "external_data" / "foda2023"
OUT = BASE / "external_validation_outputs"

RANDOM_SEED = 42
N_BOOTSTRAPS = 1000
N_PERMUTATIONS = 300


def mkdirs() -> None:
    OUT.mkdir(exist_ok=True)


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def weighted_quantile_from_counts(freq: np.ndarray, widths: np.ndarray, q: float) -> np.ndarray:
    cdf = np.cumsum(freq, axis=1)
    idx = (cdf >= q).argmax(axis=1)
    return widths[idx]


def make_length_features(frag_lengths: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    pivot = (
        frag_lengths.pivot_table(index="id", columns="width", values="count", aggfunc="sum", fill_value=0)
        .sort_index(axis=1)
        .astype(float)
    )
    widths = pivot.columns.to_numpy(dtype=float)
    counts = pivot.to_numpy(dtype=float)
    totals = counts.sum(axis=1)
    freq = counts / np.maximum(totals[:, None], 1.0)

    freq_cols = [f"p_len_{int(w)}" for w in widths]
    features = pd.DataFrame(freq, index=pivot.index, columns=freq_cols)

    mean = (freq * widths[None, :]).sum(axis=1)
    sd = np.sqrt((freq * (widths[None, :] - mean[:, None]) ** 2).sum(axis=1))
    median = weighted_quantile_from_counts(freq, widths, 0.50)
    q1 = weighted_quantile_from_counts(freq, widths, 0.25)
    q3 = weighted_quantile_from_counts(freq, widths, 0.75)

    short = freq[:, (widths >= 105) & (widths <= 150)].sum(axis=1)
    sub_nucleosomal = freq[:, (widths >= 105) & (widths <= 145)].sum(axis=1)
    nucleosomal = freq[:, (widths >= 151) & (widths <= 180)].sum(axis=1)
    long_frag = freq[:, (widths >= 181) & (widths <= 220)].sum(axis=1)
    peak_160_170 = freq[:, (widths >= 160) & (widths <= 170)].sum(axis=1)
    peak_166 = freq[:, (widths >= 164) & (widths <= 168)].sum(axis=1)
    entropy = -(freq * np.log2(np.maximum(freq, 1e-12))).sum(axis=1)
    centered = widths[None, :] - mean[:, None]
    skew = (freq * centered**3).sum(axis=1) / np.maximum(sd**3, 1e-12)

    summary = pd.DataFrame(
        {
            "length_mean": mean,
            "length_sd": sd,
            "length_median": median,
            "length_iqr": q3 - q1,
            "frac_105_145": sub_nucleosomal,
            "frac_105_150": short,
            "frac_151_180": nucleosomal,
            "frac_181_220": long_frag,
            "frac_160_170": peak_160_170,
            "frac_164_168": peak_166,
            "short_long_ratio": short / np.maximum(long_frag, 1e-9),
            "subnuc_nuc_ratio": sub_nucleosomal / np.maximum(nucleosomal, 1e-9),
            "entropy": entropy,
            "skewness": skew,
            "log_total_fragments": np.log10(np.maximum(totals, 1.0)),
        },
        index=pivot.index,
    )

    all_features = pd.concat([features, summary], axis=1).reset_index().rename(columns={"index": "id"})
    primary_cols = freq_cols + [c for c in summary.columns if c != "log_total_fragments"]
    technical_cols = ["log_total_fragments"]
    return all_features, primary_cols, technical_cols


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    ix = np.argmax(tpr - fpr)
    return float(thresholds[ix])


def metric_dict(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "ROC_AUC": roc_auc_score(y_true, scores),
        "PR_AUC": average_precision_score(y_true, scores),
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, pred),
        "Sensitivity": recall_score(y_true, pred, zero_division=0),
        "Specificity": safe_div(tn, tn + fp),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def bootstrap_ci(y_true: np.ndarray, scores: np.ndarray, threshold: float, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    n = len(y_true)
    for _ in range(N_BOOTSTRAPS):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        rows.append(metric_dict(y_true[idx], scores[idx], threshold))
    boot = pd.DataFrame(rows)
    ci_rows = []
    for metric in ["ROC_AUC", "PR_AUC", "Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity", "Precision", "F1"]:
        ci_rows.append(
            {
                "Metric": metric,
                "Mean": boot[metric].mean(),
                "CI_2.5": boot[metric].quantile(0.025),
                "CI_97.5": boot[metric].quantile(0.975),
            }
        )
    return pd.DataFrame(ci_rows)


def linear_fragment_pipeline(n_features: int) -> Pipeline:
    k = min(60, max(5, n_features // 2), n_features)
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k)),
            ("clf", LinearSVC(C=0.1, class_weight="balanced", max_iter=50000, random_state=RANDOM_SEED)),
        ]
    )


def get_scores(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model[-1], "decision_function"):
        return model.decision_function(x)
    return model.predict_proba(x)[:, 1]


def fit_fragment_model(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], label_col: str) -> dict:
    model = linear_fragment_pipeline(len(feature_cols))
    model.fit(train[feature_cols], train[label_col])
    train_scores = get_scores(model, train[feature_cols])
    test_scores = get_scores(model, test[feature_cols])
    threshold = choose_threshold(train[label_col].to_numpy(), train_scores)
    metrics = metric_dict(test[label_col].to_numpy(), test_scores, threshold)
    return {"model": model, "threshold": threshold, "train_scores": train_scores, "test_scores": test_scores, "metrics": metrics}


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def technical_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_cols),
            (
                "cat",
                Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot_encoder())]),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_technical_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    label_col: str,
) -> dict:
    pipe = Pipeline(
        [
            ("prep", technical_preprocessor(numeric_cols, categorical_cols)),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear", random_state=RANDOM_SEED)),
        ]
    )
    pipe.fit(train[numeric_cols + categorical_cols], train[label_col])
    train_scores = pipe.predict_proba(train[numeric_cols + categorical_cols])[:, 1]
    test_scores = pipe.predict_proba(test[numeric_cols + categorical_cols])[:, 1]
    threshold = choose_threshold(train[label_col].to_numpy(), train_scores)
    metrics = metric_dict(test[label_col].to_numpy(), test_scores, threshold)
    return {"model": pipe, "threshold": threshold, "train_scores": train_scores, "test_scores": test_scores, "metrics": metrics}


def residualize_fragment_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prep = technical_preprocessor(numeric_cols, categorical_cols)
    z_train = prep.fit_transform(train[numeric_cols + categorical_cols])
    z_test = prep.transform(test[numeric_cols + categorical_cols])

    ridge = Ridge(alpha=10.0)
    ridge.fit(z_train, train[feature_cols].to_numpy())
    train_resid = train[feature_cols].to_numpy() - ridge.predict(z_train)
    test_resid = test[feature_cols].to_numpy() - ridge.predict(z_test)

    train_out = train.copy()
    test_out = test.copy()
    for i, col in enumerate(feature_cols):
        train_out[col] = train_resid[:, i]
        test_out[col] = test_resid[:, i]
    return train_out, test_out


def plot_roc(y_true: np.ndarray, score_map: dict[str, np.ndarray], out: Path, title: str) -> None:
    plt.figure(figsize=(6.4, 5.2), dpi=180)
    for label, scores in score_map.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        plt.plot(fpr, tpr, lw=2, label=f"{label} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_mean_profiles(df: pd.DataFrame, widths: list[int], label_col: str, out: Path) -> None:
    plt.figure(figsize=(7.2, 4.6), dpi=180)
    for label, group_name, color in [(0, "Non-cirrhotic controls", "#2E86AB"), (1, "Advanced fibrosis/cirrhosis", "#C73E1D")]:
        sub = df[df[label_col] == label]
        y = sub[[f"p_len_{w}" for w in widths]].mean().to_numpy()
        plt.plot(widths, y, lw=2, label=f"{group_name} (n={len(sub)})", color=color)
    plt.xlabel("Fragment length (bp)")
    plt.ylabel("Mean normalized frequency")
    plt.title("External fragment-length profiles: Annapragada liver disease data")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def annapragada_external_replication() -> dict:
    frag = pd.read_csv(ANN / "frag_lengths.csv")
    meta = pd.read_csv(ANN / "All_LCr_Clinical_Metadata_withscores.csv")
    tech = pd.read_csv(ANN / "TableS2.csv")

    features, primary_cols, technical_count_cols = make_length_features(frag)

    df = (
        features.merge(meta, on="id", how="inner", suffixes=("", "_meta"))
        .merge(tech, on="id", how="left", suffixes=("", "_tech"))
        .copy()
    )
    df["label_cirrhosis"] = (df["type3"] == "Advanced Fibrosis/Cirrhosis").astype(int)

    numeric_tech = [
        "Genomic Library Input (ng)",
        "high_qual_cov",
        "q20_rate",
        "q30_rate",
        "read_gc_perc",
        "log_total_fragments",
    ]
    categorical_tech = ["batch", "seq_date", "source", "type2"]
    for col in numeric_tech:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in categorical_tech:
        df[col] = df[col].astype("string").fillna("missing")

    train = df[df["training"] == "Discovery"].copy()
    test = df[df["training"] == "Validation"].copy()

    results = []
    score_map = {}
    rng = np.random.default_rng(RANDOM_SEED)

    frag_fit = fit_fragment_model(train, test, primary_cols, "label_cirrhosis")
    results.append({"Experiment": "Annapragada external Discovery->Validation fragmentomics", **frag_fit["metrics"]})
    score_map["Fragmentomics"] = frag_fit["test_scores"]
    bootstrap_ci(
        test["label_cirrhosis"].to_numpy(), frag_fit["test_scores"], frag_fit["threshold"], rng
    ).to_csv(OUT / "annapragada_fragmentomics_bootstrap_ci.csv", index=False)

    tech_fit = fit_technical_model(train, test, numeric_tech, categorical_tech, "label_cirrhosis")
    results.append({"Experiment": "Annapragada technical covariates only", **tech_fit["metrics"]})
    score_map["Technical covariates"] = tech_fit["test_scores"]

    train_resid, test_resid = residualize_fragment_features(train, test, primary_cols, numeric_tech, categorical_tech)
    resid_fit = fit_fragment_model(train_resid, test_resid, primary_cols, "label_cirrhosis")
    results.append({"Experiment": "Annapragada residualized fragmentomics", **resid_fit["metrics"]})
    score_map["Residualized fragmentomics"] = resid_fit["test_scores"]

    # Source-level holdout: a practical study-level validation proxy.
    source_rows = []
    for source, source_test in df.groupby("source"):
        if len(source_test) < 20 or source_test["label_cirrhosis"].nunique() < 2:
            continue
        source_train = df[df["source"] != source].copy()
        if source_train["label_cirrhosis"].nunique() < 2:
            continue
        try:
            fit = fit_fragment_model(source_train, source_test.copy(), primary_cols, "label_cirrhosis")
            source_rows.append({"Held_Out_Source": source, "N": len(source_test), **fit["metrics"]})
        except Exception as exc:
            source_rows.append({"Held_Out_Source": source, "N": len(source_test), "Error": str(exc)})

    source_df = pd.DataFrame(source_rows)
    source_df.to_csv(OUT / "annapragada_source_holdout_results.csv", index=False)

    # Permutation audit for the external fragmentomics split.
    observed_auc = frag_fit["metrics"]["ROC_AUC"]
    null_aucs = []
    for i in range(N_PERMUTATIONS):
        perm_train = train.copy()
        perm_train["label_cirrhosis"] = rng.permutation(perm_train["label_cirrhosis"].to_numpy())
        try:
            perm_fit = fit_fragment_model(perm_train, test, primary_cols, "label_cirrhosis")
            null_aucs.append(perm_fit["metrics"]["ROC_AUC"])
        except Exception:
            continue
    null_aucs = np.asarray(null_aucs)
    p_value = (1 + np.sum(null_aucs >= observed_auc)) / (1 + len(null_aucs))
    pd.DataFrame({"permutation_auc": null_aucs}).to_csv(OUT / "annapragada_permutation_null.csv", index=False)

    metrics = pd.DataFrame(results)
    metrics["Permutation_P_Value_vs_Fragmentomics"] = np.nan
    metrics.loc[metrics["Experiment"].str.contains("fragmentomics", case=False).idxmax(), "Permutation_P_Value_vs_Fragmentomics"] = p_value
    metrics.to_csv(OUT / "annapragada_external_metrics.csv", index=False)

    widths = sorted(frag["width"].unique().tolist())
    df.to_csv(OUT / "annapragada_external_feature_matrix.csv", index=False)
    plot_roc(
        test["label_cirrhosis"].to_numpy(),
        score_map,
        OUT / "annapragada_external_roc.png",
        "Annapragada external Discovery->Validation classification",
    )
    plot_mean_profiles(df, widths, "label_cirrhosis", OUT / "annapragada_mean_fragment_profiles.png")

    # Biological interpretation from the primary model coefficients after selection.
    model = frag_fit["model"]
    selected_mask = model.named_steps["select"].get_support()
    selected_cols = np.asarray(primary_cols)[selected_mask]
    coefs = model.named_steps["clf"].coef_.ravel()
    coef_df = pd.DataFrame({"feature": selected_cols, "coefficient": coefs, "abs_coefficient": np.abs(coefs)})
    coef_df.sort_values("abs_coefficient", ascending=False).to_csv(OUT / "annapragada_fragmentomics_top_features.csv", index=False)

    summary = {
        "n_total_with_fragment_lengths": int(len(df)),
        "n_discovery": int(len(train)),
        "n_validation": int(len(test)),
        "validation_class_counts": test["type3"].value_counts().to_dict(),
        "discovery_class_counts": train["type3"].value_counts().to_dict(),
        "n_source_holdouts": int(len(source_df)),
        "permutation_p_value": float(p_value),
    }
    return summary


def foda_delfi_benchmark() -> dict:
    clinical = pd.read_csv(FODA / "foda_supp_table_1.csv")
    wgs = pd.read_csv(FODA / "foda_supp_table_2.csv")
    library = pd.read_csv(FODA / "foda_supp_table_5.csv")
    df = clinical.merge(wgs, on="Patient", how="inner").merge(library, on="Patient", how="left")

    df["is_hcc"] = (df["Patient type"] == "HCC").astype(int)
    df["is_high_risk_non_cancer"] = df["Patient type"].isin(["Cirrhosis", "HBV", "HCV"]).astype(int)

    rows = []
    score_cols = ["DELFI-surveillance score", "DELFI-screening score", "ichor"]
    for cohort_name, cohort_df in {
        "HCC vs all non-HCC": df[df["Patient type"] != "Non-cancer"].copy(),
        "HCC vs cirrhosis/HBV/HCV": df[(df["Patient type"] == "HCC") | (df["is_high_risk_non_cancer"] == 1)].copy(),
        "HCC vs cirrhosis only": df[df["Patient type"].isin(["HCC", "Cirrhosis"])].copy(),
    }.items():
        for score_col in score_cols:
            sub = cohort_df[["is_hcc", score_col, "Patient type"]].dropna()
            if len(sub) == 0 or sub["is_hcc"].nunique() < 2:
                continue
            y = sub["is_hcc"].to_numpy()
            scores = pd.to_numeric(sub[score_col], errors="coerce").to_numpy()
            keep = ~np.isnan(scores)
            y = y[keep]
            scores = scores[keep]
            if len(np.unique(y)) < 2:
                continue
            threshold = choose_threshold(y, scores)
            rows.append(
                {
                    "Benchmark": cohort_name,
                    "Score": score_col,
                    "N": int(len(y)),
                    "N_HCC": int(y.sum()),
                    "N_Control": int((y == 0).sum()),
                    **metric_dict(y, scores, threshold),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "foda_delfi_benchmark_metrics.csv", index=False)

    # Plot the most relevant Foda benchmark.
    sub = df[(df["Patient type"] == "HCC") | (df["is_high_risk_non_cancer"] == 1)][
        ["is_hcc", "DELFI-surveillance score"]
    ].dropna()
    if len(sub) and sub["is_hcc"].nunique() == 2:
        plot_roc(
            sub["is_hcc"].to_numpy(),
            {"Published DELFI-surveillance score": sub["DELFI-surveillance score"].to_numpy()},
            OUT / "foda_delfi_surveillance_roc.png",
            "Foda et al. published score benchmark",
        )

    return {
        "n_foda_samples": int(len(df)),
        "patient_type_counts": df["Patient type"].value_counts(dropna=False).to_dict(),
        "benchmark_rows": int(len(out)),
    }


def write_report(ann_summary: dict, foda_summary: dict) -> None:
    metrics = pd.read_csv(OUT / "annapragada_external_metrics.csv")
    foda_metrics = pd.read_csv(OUT / "foda_delfi_benchmark_metrics.csv")
    source = pd.read_csv(OUT / "annapragada_source_holdout_results.csv")
    ci = pd.read_csv(OUT / "annapragada_fragmentomics_bootstrap_ci.csv")
    top = pd.read_csv(OUT / "annapragada_fragmentomics_top_features.csv").head(12)

    def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
        if df.empty:
            return "_No rows._"
        clean = df.copy()
        for col in clean.columns:
            if pd.api.types.is_float_dtype(clean[col]):
                clean[col] = clean[col].map(lambda x: "" if pd.isna(x) else format(float(x), floatfmt))
            else:
                clean[col] = clean[col].map(lambda x: "" if pd.isna(x) else str(x))
        header = "| " + " | ".join(clean.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(clean.columns)) + " |"
        body = ["| " + " | ".join(row) + " |" for row in clean.astype(str).to_numpy()]
        return "\n".join([header, sep] + body)

    lines = []
    lines.append("# MiNAE External Validation and Replication Audit\n")
    lines.append("## What Was Tested\n")
    lines.append(
        "The downloaded external data do not provide MiNAE-compatible HCC fragment-length profiles. "
        "Therefore, the analysis is reported honestly as an external fragmentomics replication/audit using "
        "Annapragada liver disease fragment-length profiles, plus a Foda et al. published-score HCC benchmark."
    )
    lines.append("\n## Annapragada External Fragment-Length Replication\n")
    lines.append(f"- Total samples with fragment-length profiles: {ann_summary['n_total_with_fragment_lengths']}")
    lines.append(f"- Discovery/train samples: {ann_summary['n_discovery']}")
    lines.append(f"- Validation/test samples: {ann_summary['n_validation']}")
    lines.append(f"- Validation class counts: {ann_summary['validation_class_counts']}")
    lines.append(f"- Permutation p-value for fragmentomics model: {ann_summary['permutation_p_value']:.4f}")
    lines.append("\n### Discovery-to-Validation Metrics\n")
    lines.append(md_table(metrics))
    lines.append("\n### Fragmentomics Bootstrap Confidence Intervals\n")
    lines.append(md_table(ci))
    lines.append("\n### Source-Level Holdout Results\n")
    if len(source):
        lines.append(md_table(source))
    else:
        lines.append("No held-out source contained enough samples from both classes for source-level AUC calculation.")
    lines.append("\n### Top Fragmentomic Features\n")
    lines.append(md_table(top, floatfmt=".4f"))
    lines.append("\n## Foda et al. HCC Benchmark Context\n")
    lines.append(f"- Foda workbook samples: {foda_summary['n_foda_samples']}")
    lines.append(f"- Patient-type counts: {foda_summary['patient_type_counts']}")
    lines.append(
        "These results use published DELFI/Foda scores from the supplementary workbook. "
        "They are not MiNAE predictions, but they contextualize the HCC task."
    )
    lines.append("\n### Published-Score Metrics\n")
    lines.append(md_table(foda_metrics))
    lines.append("\n## Interpretation\n")
    lines.append(
        "This audit strengthens MiNAE by adding external fragmentomics replication, confidence intervals, "
        "technical-covariate comparison, residualized fragmentomics, and source-level holdout testing. "
        "It does not yet provide a full independent HCC validation for MiNAE because the downloaded external "
        "tables do not include HCC sample-level fragment-length histograms compatible with MiNAE feature extraction."
    )
    (OUT / "MiNAE_external_validation_report.md").write_text("\n".join(lines))


def main() -> None:
    mkdirs()
    ann_summary = annapragada_external_replication()
    foda_summary = foda_delfi_benchmark()
    with open(OUT / "external_validation_summary.json", "w") as f:
        json.dump({"annapragada": ann_summary, "foda": foda_summary}, f, indent=2)
    write_report(ann_summary, foda_summary)
    print("External validation/audit complete.")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
