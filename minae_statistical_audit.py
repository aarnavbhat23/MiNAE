from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

PROJECT_DIR = Path.home() / "Documents" / "MiNAE_Global_Local"
FUSION_DIR = PROJECT_DIR / "dual_reference_fusion_outputs"
DUAL_DIR = PROJECT_DIR / "dual_reference_outputs"
AUDIT_DIR = PROJECT_DIR / "statistical_audit_outputs"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_PERMUTATIONS = 500


def selected_model(n_features):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=min(70, n_features))),
        ("model", LinearSVC(
            C=0.1,
            class_weight="balanced",
            dual="auto",
            random_state=RANDOM_STATE,
            max_iter=20_000,
        )),
    ])


def get_feature_columns(df):
    return [
        col for col in df.columns
        if col not in {"sample_id", "class_label", "class_name"}
        and not col.endswith("_Fragment_Count")
    ]


def specificity(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return tn / (tn + fp + 1e-12)


def metric_row(name, y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    return {
        "Experiment": name,
        "ROC_AUC": roc_auc_score(y_true, scores),
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, pred),
        "Sensitivity": recall_score(y_true, pred),
        "Specificity": specificity(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
    }


def choose_threshold(y_true, scores):
    best = None
    for threshold in np.linspace(np.min(scores), np.max(scores), 181):
        row = metric_row("threshold", y_true, scores, threshold)
        key = (row["Balanced_Accuracy"], row["Accuracy"], row["Precision"])
        if best is None or key > best[0]:
            best = (key, threshold)
    return float(best[1])


def load_current_split():
    train = pd.read_csv(FUSION_DIR / "fusion_train_split.csv")
    test = pd.read_csv(FUSION_DIR / "fusion_test_split.csv")
    final = pd.read_csv(FUSION_DIR / "fusion_final_validation_split.csv")
    return train, test, final


def permutation_test():
    train, test, final = load_current_split()
    feature_cols = get_feature_columns(train)
    development = pd.concat([train, test], ignore_index=True)
    X_dev = development[feature_cols]
    y_dev = development["class_label"].to_numpy()
    X_test = test[feature_cols]
    y_test = test["class_label"].to_numpy()
    X_final = final[feature_cols]
    y_final = final["class_label"].to_numpy()

    model = selected_model(len(feature_cols))
    fitted = clone(model).fit(X_dev, y_dev)
    true_scores = fitted.decision_function(X_final)
    true_test_scores = fitted.decision_function(X_test)
    true_threshold = choose_threshold(y_test, true_test_scores)
    observed = metric_row("Observed labels", y_final, true_scores, true_threshold)

    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for i in range(N_PERMUTATIONS):
        shuffled = rng.permutation(y_dev)
        perm_model = clone(model).fit(X_dev, shuffled)
        perm_scores = perm_model.decision_function(X_final)
        rows.append({
            "Permutation": i + 1,
            "ROC_AUC": roc_auc_score(y_final, perm_scores),
        })

    null = pd.DataFrame(rows)
    p_value = (1 + (null["ROC_AUC"] >= observed["ROC_AUC"]).sum()) / (N_PERMUTATIONS + 1)
    observed["Permutation_P_Value"] = p_value
    pd.DataFrame([observed]).to_csv(AUDIT_DIR / "permutation_observed_metrics.csv", index=False)
    null.to_csv(AUDIT_DIR / "permutation_null_distribution.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.histplot(null["ROC_AUC"], bins=30, kde=True, color="#7aa6c2")
    plt.axvline(observed["ROC_AUC"], color="crimson", linestyle="--", linewidth=2, label=f"Observed AUC = {observed['ROC_AUC']:.3f}")
    plt.title(f"Permutation Test Null Distribution (p = {p_value:.4f})")
    plt.xlabel("Final Validation ROC-AUC Under Shuffled Training Labels")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(AUDIT_DIR / "permutation_test_auc_null.png", dpi=300)
    plt.close()


def build_technical_feature_table():
    fusion = pd.read_csv(FUSION_DIR / "dual_reference_fusion_features.csv")[["sample_id", "class_label", "class_name"]]
    hg38 = pd.read_csv(DUAL_DIR / "dna_features_hg38.csv")[["sample_id", "Total_Used_Fragments"]].rename(
        columns={"Total_Used_Fragments": "Total_Used_Fragments_hg38"}
    )
    hg19 = pd.read_csv(DUAL_DIR / "dna_features_hg19.csv")[["sample_id", "Total_Used_Fragments"]].rename(
        columns={"Total_Used_Fragments": "Total_Used_Fragments_hg19"}
    )
    table = fusion.merge(hg38, on="sample_id").merge(hg19, on="sample_id")
    table["Fragment_Count_Delta"] = table["Total_Used_Fragments_hg38"] - table["Total_Used_Fragments_hg19"]
    table["Fragment_Count_Mean"] = (table["Total_Used_Fragments_hg38"] + table["Total_Used_Fragments_hg19"]) / 2
    table.to_csv(AUDIT_DIR / "technical_negative_control_features.csv", index=False)
    return table


def negative_control_model():
    technical = build_technical_feature_table()
    train_ids = set(pd.read_csv(FUSION_DIR / "fusion_train_split.csv")["sample_id"])
    test_ids = set(pd.read_csv(FUSION_DIR / "fusion_test_split.csv")["sample_id"])
    final_ids = set(pd.read_csv(FUSION_DIR / "fusion_final_validation_split.csv")["sample_id"])
    train = technical[technical["sample_id"].isin(train_ids)]
    test = technical[technical["sample_id"].isin(test_ids)]
    final = technical[technical["sample_id"].isin(final_ids)]

    feature_cols = ["Total_Used_Fragments_hg38", "Total_Used_Fragments_hg19", "Fragment_Count_Delta", "Fragment_Count_Mean"]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearSVC(C=0.1, class_weight="balanced", dual="auto", random_state=RANDOM_STATE, max_iter=20_000)),
    ])
    fitted = model.fit(pd.concat([train, test])[feature_cols], pd.concat([train, test])["class_label"])
    test_scores = fitted.decision_function(test[feature_cols])
    threshold = choose_threshold(test["class_label"], test_scores)
    final_scores = fitted.decision_function(final[feature_cols])
    row = metric_row("Technical feature negative control", final["class_label"], final_scores, threshold)
    pd.DataFrame([row]).to_csv(AUDIT_DIR / "technical_negative_control_metrics.csv", index=False)


def error_analysis():
    pred = pd.read_csv(FUSION_DIR / "fusion_final_validation_predictions.csv")
    pred["error_type"] = "correct"
    pred.loc[(pred["true_label"] == 0) & (pred["predicted_label"] == 1), "error_type"] = "false_positive"
    pred.loc[(pred["true_label"] == 1) & (pred["predicted_label"] == 0), "error_type"] = "false_negative"
    pred["distance_from_threshold"] = (pred["predicted_probability"] - 0.485).abs()
    pred.sort_values(["error_type", "distance_from_threshold"]).to_csv(AUDIT_DIR / "final_validation_error_analysis.csv", index=False)

    summary = pred.groupby(["class_name", "error_type"]).size().reset_index(name="count")
    summary.to_csv(AUDIT_DIR / "error_summary_by_disease.csv", index=False)


def main():
    permutation_test()
    negative_control_model()
    error_analysis()
    print(f"Statistical audit outputs saved in: {AUDIT_DIR}")


if __name__ == "__main__":
    main()
