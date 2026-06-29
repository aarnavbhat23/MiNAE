from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
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
ABLATION_DIR = PROJECT_DIR / "ablation_chromosome_outputs"
REPEATED_DIR = PROJECT_DIR / "repeated_validation_outputs"
REPEATED_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_REPEATS = 100
TEST_PER_CLASS = 20
FINAL_PER_CLASS = 20


def load_strategy_tables():
    paths = {
        "hg38_global": ABLATION_DIR / "strategy_1_hg38_global_train.csv",
        "hg19_global": ABLATION_DIR / "strategy_2_hg19_global_train.csv",
        "dual_global_fusion": FUSION_DIR / "dual_reference_fusion_features.csv",
        "chromosome_fusion": ABLATION_DIR / "chromosome_fusion_features.csv",
        "global_plus_chromosome": ABLATION_DIR / "global_plus_chromosome_features.csv",
    }

    # For hg38/hg19 global, the train CSV is only used to discover columns.
    # Load the complete cached feature files instead.
    complete_paths = {
        "hg38_global": PROJECT_DIR / "dual_reference_outputs" / "dna_features_hg38.csv",
        "hg19_global": PROJECT_DIR / "dual_reference_outputs" / "dna_features_hg19.csv",
        "dual_global_fusion": paths["dual_global_fusion"],
        "chromosome_fusion": paths["chromosome_fusion"],
        "global_plus_chromosome": paths["global_plus_chromosome"],
    }

    tables = {}
    for name, path in complete_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required feature table for {name}: {path}")
        table = pd.read_csv(path)
        tables[name] = table
        print(f"{name}: {table.shape}, class counts {table['class_label'].value_counts().sort_index().to_dict()}")
    return tables


def align_to_fusion_patients(tables):
    # Use the same 176-patient pool as the strongest fusion model.
    fusion_ids = set(tables["dual_global_fusion"]["sample_id"])
    aligned = {}
    for name, table in tables.items():
        subset = table[table["sample_id"].isin(fusion_ids)].copy()
        subset = subset.drop_duplicates("sample_id").reset_index(drop=True)
        aligned[name] = subset
        print(f"Aligned {name}: {subset.shape}, class counts {subset['class_label'].value_counts().sort_index().to_dict()}")
    return aligned


def get_feature_columns(table):
    excluded = {"sample_id", "class_label", "class_name"}
    return [
        col for col in table.columns
        if col not in excluded and not col.endswith("_Fragment_Count")
    ]


def make_split(table, seed):
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []
    final_parts = []

    for label in [0, 1]:
        subset = table[table["class_label"] == label].copy()
        order = rng.permutation(len(subset))
        subset = subset.iloc[order].reset_index(drop=True)

        final_parts.append(subset.iloc[:FINAL_PER_CLASS])
        test_parts.append(subset.iloc[FINAL_PER_CLASS:FINAL_PER_CLASS + TEST_PER_CLASS])
        train_parts.append(subset.iloc[FINAL_PER_CLASS + TEST_PER_CLASS:])

    train = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    final = pd.concat(final_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, test, final


def model_library(n_features):
    k = min(70, n_features)
    return {
        f"Locked_LinearSVC_k{k}": Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k)),
            ("model", LinearSVC(
                C=0.1,
                class_weight="balanced",
                dual="auto",
                random_state=RANDOM_STATE,
                max_iter=20_000,
            )),
        ])
    }


def model_scores(model, X):
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp + 1e-12)


def metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pred = (scores >= threshold).astype(int)
    return {
        "ROC_AUC": roc_auc_score(y_true, scores),
        "PR_AUC": average_precision_score(y_true, scores),
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, pred),
        "Sensitivity": recall_score(y_true, pred),
        "Specificity": specificity_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred),
    }


def choose_threshold(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    candidates = np.linspace(np.min(scores), np.max(scores), 181)
    best = None
    for threshold in candidates:
        row = metrics(y_true, scores, threshold)
        key = (
            row["Balanced_Accuracy"],
            row["Accuracy"],
            row["Precision"],
            row["F1"],
        )
        if best is None or key > best[0]:
            best = (key, threshold, row)
    return float(best[1])


def evaluate_strategy(table, strategy_name, seed):
    train, test, final = make_split(table, seed)
    feature_cols = get_feature_columns(table)

    X_train, y_train = train[feature_cols], train["class_label"]
    X_test, y_test = test[feature_cols], test["class_label"]
    X_final, y_final = final[feature_cols], final["class_label"]

    rows = []
    fitted = {}
    thresholds = {}
    for model_name, model in model_library(len(feature_cols)).items():
        fitted_model = clone(model).fit(X_train, y_train)
        test_scores = model_scores(fitted_model, X_test)
        threshold = choose_threshold(y_test, test_scores)
        row = metrics(y_test, test_scores, threshold)
        row.update({
            "Seed": seed,
            "Strategy": strategy_name,
            "Split": "test",
            "Model": model_name,
            "Threshold": threshold,
        })
        rows.append(row)
        fitted[model_name] = fitted_model
        thresholds[model_name] = threshold

    test_results = pd.DataFrame(rows).sort_values(
        ["Balanced_Accuracy", "ROC_AUC", "Accuracy"],
        ascending=False,
    )
    best_model_name = test_results.iloc[0]["Model"]
    threshold = float(test_results.iloc[0]["Threshold"])

    development = pd.concat([train, test], axis=0)
    X_dev, y_dev = development[feature_cols], development["class_label"]
    final_model = clone(model_library(len(feature_cols))[best_model_name]).fit(X_dev, y_dev)
    final_scores = model_scores(final_model, X_final)

    final_row = metrics(y_final, final_scores, threshold)
    final_row.update({
        "Seed": seed,
        "Strategy": strategy_name,
        "Split": "final_validation",
        "Model": best_model_name,
        "Threshold": threshold,
    })
    return test_results, final_row


def summarize(results):
    metric_cols = [
        "ROC_AUC",
        "PR_AUC",
        "Accuracy",
        "Balanced_Accuracy",
        "Sensitivity",
        "Specificity",
        "Precision",
        "F1",
    ]
    rows = []
    for strategy, group in results.groupby("Strategy"):
        row = {"Strategy": strategy, "N_Repeats": len(group)}
        for metric in metric_cols:
            row[f"{metric}_Mean"] = group[metric].mean()
            row[f"{metric}_SD"] = group[metric].std()
            row[f"{metric}_CI2.5"] = group[metric].quantile(0.025)
            row[f"{metric}_CI97.5"] = group[metric].quantile(0.975)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ROC_AUC_Mean", ascending=False)


def plot_repeated_summary(results, summary):
    key_metrics = ["ROC_AUC", "Accuracy", "Sensitivity", "Specificity", "Precision"]
    long = results.melt(
        id_vars=["Seed", "Strategy"],
        value_vars=key_metrics,
        var_name="Metric",
        value_name="Value",
    )
    plt.figure(figsize=(13, 6))
    sns.boxplot(data=long, x="Strategy", y="Value", hue="Metric")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0.45, 1.02)
    plt.title("Repeated Held-Out Validation Across Feature Strategies")
    plt.tight_layout()
    plt.savefig(REPEATED_DIR / "repeated_validation_boxplots.png", dpi=300)
    plt.close()

    heat_cols = [
        "ROC_AUC_Mean",
        "Accuracy_Mean",
        "Sensitivity_Mean",
        "Specificity_Mean",
        "Precision_Mean",
        "F1_Mean",
    ]
    heat = summary.set_index("Strategy")[heat_cols]
    plt.figure(figsize=(10, 5))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.5, vmax=1.0)
    plt.title("Mean Repeated-Validation Performance")
    plt.tight_layout()
    plt.savefig(REPEATED_DIR / "repeated_validation_mean_heatmap.png", dpi=300)
    plt.close()


def main():
    tables = align_to_fusion_patients(load_strategy_tables())
    final_rows = []
    test_rows = []

    for repeat in range(N_REPEATS):
        seed = RANDOM_STATE + repeat
        print(f"\nRepeat {repeat + 1}/{N_REPEATS} (seed={seed})")
        for strategy_name, table in tables.items():
            print(f"  {strategy_name}")
            test_result, final_row = evaluate_strategy(table, strategy_name, seed)
            test_rows.append(test_result)
            final_rows.append(final_row)

    test_results = pd.concat(test_rows, ignore_index=True)
    final_results = pd.DataFrame(final_rows)
    summary = summarize(final_results)

    test_results.to_csv(REPEATED_DIR / "repeated_test_model_selection.csv", index=False)
    final_results.to_csv(REPEATED_DIR / "repeated_final_validation_results.csv", index=False)
    summary.to_csv(REPEATED_DIR / "repeated_validation_summary.csv", index=False)
    plot_repeated_summary(final_results, summary)

    print("\nRepeated validation summary:")
    display_cols = ["Strategy", "N_Repeats", "ROC_AUC_Mean", "Accuracy_Mean", "Sensitivity_Mean", "Specificity_Mean", "Precision_Mean", "F1_Mean"]
    print(summary[display_cols].to_string(index=False))
    print(f"\nOutputs saved in: {REPEATED_DIR}")


if __name__ == "__main__":
    main()
