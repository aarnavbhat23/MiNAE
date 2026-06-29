from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

HOME = Path.home()
PROJECT_DIR = HOME / "Documents" / "MiNAE_Global_Local"
OUTPUT_DIR = PROJECT_DIR / "dual_reference_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
ROW_LIMIT = 1_500_000

# Fixed balanced split counts per class.
# With ~89 cancer and ~89 controls, this yields about:
# train=49/class, test=20/class, final_validation=20/class.
TEST_PER_CLASS = 20
FINAL_VALIDATION_PER_CLASS = 20

FINE_BINS = list(range(50, 255, 5))
FINE_BIN_NAMES = [f"Bin_{FINE_BINS[i]}_{FINE_BINS[i + 1]}" for i in range(len(FINE_BINS) - 1)]

EXPERIMENTS = {
    "hg38": {
        "liver_dir": HOME / "Downloads" / "liver hg38",
        "hepatitis_dir": HOME / "Downloads" / "hepatitis b hg38",
        "cirrhosis_dir": HOME / "Downloads" / "cirrhosis hg38",
        "file_pattern": "*.hg38.frag.tsv.bgz",
    },
    "hg19": {
        "liver_dir": HOME / "Downloads" / "liver hg19",
        "hepatitis_dir": HOME / "Downloads" / "hepatitis b hg19",
        "cirrhosis_dir": HOME / "Downloads" / "cirrhosis hg19",
        "file_pattern": "*.hg19.frag.tsv.bgz",
    },
}


def get_sample_id(file_path):
    match = re.search(r"(EE\d+)", file_path.name)
    return match.group(1) if match else file_path.stem


def collect_files(folder, pattern, label, class_name):
    rows = []
    if not folder.exists():
        print(f"WARNING: Missing folder: {folder}")
        return rows

    for file_path in sorted(folder.glob(pattern)):
        rows.append({
            "sample_id": get_sample_id(file_path),
            "file_path": str(file_path),
            "class_label": label,
            "class_name": class_name,
        })
    return rows


def build_manifest(reference_name, config):
    rows = []
    rows.extend(collect_files(config["liver_dir"], config["file_pattern"], 1, "Liver Cancer"))
    rows.extend(collect_files(config["hepatitis_dir"], config["file_pattern"], 0, "Hepatitis B Control"))
    rows.extend(collect_files(config["cirrhosis_dir"], config["file_pattern"], 0, "Cirrhosis Control"))

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError(f"No files found for {reference_name}. Check folder paths.")

    print(f"\n[{reference_name}] Before deduplication")
    print(manifest["class_name"].value_counts())
    print("Total files:", len(manifest))

    # If the same control sample appears in hepatitis and cirrhosis folders,
    # keep one copy only. Prefer cirrhosis when duplicate controls exist.
    priority = {"Cirrhosis Control": 0, "Hepatitis B Control": 1, "Liver Cancer": 2}
    manifest["dedupe_priority"] = manifest["class_name"].map(priority)
    manifest = manifest.sort_values(["sample_id", "dedupe_priority"]).drop(columns=["dedupe_priority"])
    manifest = manifest.drop_duplicates("sample_id", keep="first").reset_index(drop=True)

    print(f"\n[{reference_name}] After deduplication")
    print(manifest["class_name"].value_counts())
    print("Unique samples:", len(manifest))

    manifest.to_csv(OUTPUT_DIR / f"manifest_{reference_name}.csv", index=False)
    return manifest


def fraction_between(lengths, low, high, include_high=False):
    if include_high:
        return np.mean((lengths >= low) & (lengths <= high))
    return np.mean((lengths >= low) & (lengths < high))


def safe_ratio(numerator, denominator):
    return numerator / (denominator + 1e-12)


def distribution_entropy(probabilities):
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


def extract_features(file_path):
    try:
        df = pd.read_csv(
            file_path,
            sep="\t",
            compression="gzip",
            header=None,
            usecols=[0, 1, 2],
            names=["chrom", "start", "end"],
            nrows=ROW_LIMIT,
        )

        lengths = (df["end"] - df["start"]).to_numpy()
        lengths = lengths[(lengths >= 50) & (lengths <= 250)]

        if len(lengths) == 0:
            return None

        counts, _ = np.histogram(lengths, bins=FINE_BINS)
        probabilities = counts / counts.sum()

        features = {name: value for name, value in zip(FINE_BIN_NAMES, probabilities)}

        short_50_150 = fraction_between(lengths, 50, 150)
        ultra_short_50_100 = fraction_between(lengths, 50, 100)
        mid_short_100_150 = fraction_between(lengths, 100, 150)
        mono_peak_160_180 = fraction_between(lengths, 160, 180, include_high=True)
        long_150_250 = fraction_between(lengths, 150, 250, include_high=True)
        long_180_250 = fraction_between(lengths, 180, 250, include_high=True)

        q10, q25, q75, q90 = np.percentile(lengths, [10, 25, 75, 90])
        mean_length = float(np.mean(lengths))
        std_length = float(np.std(lengths))
        centered = lengths - mean_length
        skewness = float(np.mean(centered ** 3) / ((std_length ** 3) + 1e-12))

        features.update({
            "Short_Fraction_50_150": short_50_150,
            "UltraShort_Fraction_50_100": ultra_short_50_100,
            "MidShort_Fraction_100_150": mid_short_100_150,
            "Nucleosome_Peak_160_180": mono_peak_160_180,
            "Long_Fraction_180_250": long_180_250,
            "Short_to_Long_50_150_over_150_250": safe_ratio(short_50_150, long_150_250),
            "UltraShort_to_MonoPeak": safe_ratio(ultra_short_50_100, mono_peak_160_180),
            "Mean_Length": mean_length,
            "Median_Length": float(np.median(lengths)),
            "Std_Length": std_length,
            "IQR_Length": float(q75 - q25),
            "Q10_Length": float(q10),
            "Q90_Length": float(q90),
            "Skewness_Length": skewness,
            "Entropy_Size_Profile": distribution_entropy(probabilities),
            "Total_Used_Fragments": int(len(lengths)),
        })

        return features
    except Exception as error:
        print(f"ERROR processing {file_path}: {error}")
        return None


def build_feature_matrix(reference_name, manifest):
    feature_file = OUTPUT_DIR / f"dna_features_{reference_name}.csv"
    if feature_file.exists():
        print(f"\n[{reference_name}] Using cached features: {feature_file}")
        return pd.read_csv(feature_file)

    print(f"\n[{reference_name}] Extracting features from raw files...")
    rows = []

    for idx, row in manifest.iterrows():
        print(f"[{reference_name}] [{idx + 1}/{len(manifest)}] {row['sample_id']}")
        features = extract_features(row["file_path"])
        if features is None:
            continue

        features["sample_id"] = row["sample_id"]
        features["class_label"] = row["class_label"]
        features["class_name"] = row["class_name"]
        rows.append(features)

    feature_df = pd.DataFrame(rows)
    leading_columns = ["sample_id", "class_label", "class_name"]
    feature_columns = [col for col in feature_df.columns if col not in leading_columns]
    feature_df = feature_df[leading_columns + feature_columns]
    feature_df.to_csv(feature_file, index=False)

    print(f"[{reference_name}] Saved features to: {feature_file}")
    return feature_df


def make_balanced_splits(feature_df, reference_name):
    rng = np.random.default_rng(RANDOM_STATE)
    train_parts = []
    test_parts = []
    final_parts = []

    for label in [0, 1]:
        subset = feature_df[feature_df["class_label"] == label].copy()
        subset = subset.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        needed = TEST_PER_CLASS + FINAL_VALIDATION_PER_CLASS + 1
        if len(subset) < needed:
            raise ValueError(
                f"{reference_name}: class {label} has only {len(subset)} samples. "
                f"Need at least {needed} for the requested split."
            )

        final_parts.append(subset.iloc[:FINAL_VALIDATION_PER_CLASS])
        test_parts.append(subset.iloc[FINAL_VALIDATION_PER_CLASS:FINAL_VALIDATION_PER_CLASS + TEST_PER_CLASS])
        train_parts.append(subset.iloc[FINAL_VALIDATION_PER_CLASS + TEST_PER_CLASS:])

    train_df = pd.concat(train_parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    final_df = pd.concat(final_parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    for split_name, split_df in [("train", train_df), ("test", test_df), ("final_validation", final_df)]:
        split_df.to_csv(OUTPUT_DIR / f"{reference_name}_{split_name}_split.csv", index=False)
        print(f"\n[{reference_name}] {split_name} split")
        print(split_df["class_name"].value_counts())
        print(split_df["class_label"].value_counts().sort_index())

    return train_df, test_df, final_df


def get_feature_columns(feature_df):
    excluded = {"sample_id", "class_label", "class_name", "Total_Used_Fragments"}
    return [col for col in feature_df.columns if col not in excluded]


def model_library():
    return {
        "L2 Logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=0.2,
                max_iter=10_000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "Elastic Logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.25,
                C=0.3,
                max_iter=20_000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "SVM RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                C=0.8,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=700,
            max_depth=5,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=700,
            max_depth=5,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=250,
            max_leaf_nodes=8,
            l2_regularization=0.3,
            random_state=RANDOM_STATE,
        ),
    }


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp + 1e-12)


def metric_row(reference_name, split_name, model_name, y_true, probabilities, threshold):
    preds = (probabilities >= threshold).astype(int)
    return {
        "Reference": reference_name,
        "Split": split_name,
        "Model": model_name,
        "Threshold": threshold,
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "Accuracy": accuracy_score(y_true, preds),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, preds),
        "Sensitivity": recall_score(y_true, preds),
        "Specificity": specificity_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "F1": f1_score(y_true, preds),
    }


def choose_threshold(y_true, probabilities):
    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        rows.append(metric_row("threshold_search", "test", "candidate", y_true, probabilities, threshold))
    table = pd.DataFrame(rows)
    table = table.sort_values(["Balanced_Accuracy", "Accuracy"], ascending=False)
    return float(table.iloc[0]["Threshold"]), table


def evaluate_reference(reference_name, feature_df):
    train_df, test_df, final_df = make_balanced_splits(feature_df, reference_name)
    feature_columns = get_feature_columns(feature_df)
    models = model_library()

    X_train = train_df[feature_columns]
    y_train = train_df["class_label"]
    X_test = test_df[feature_columns]
    y_test = test_df["class_label"]
    X_final = final_df[feature_columns]
    y_final = final_df["class_label"]

    model_rows = []
    fitted_models = {}
    threshold_tables = {}

    for model_name, model in models.items():
        print(f"\n[{reference_name}] Training {model_name}")
        model.fit(X_train, y_train)
        fitted_models[model_name] = model

        test_prob = model.predict_proba(X_test)[:, 1]
        threshold, threshold_table = choose_threshold(y_test, test_prob)
        threshold_tables[model_name] = threshold_table

        model_rows.append(metric_row(reference_name, "test", model_name, y_test, test_prob, threshold))

    test_results = pd.DataFrame(model_rows).sort_values(["Balanced_Accuracy", "ROC_AUC"], ascending=False)
    test_results.to_csv(OUTPUT_DIR / f"{reference_name}_test_model_selection.csv", index=False)

    best = test_results.iloc[0]
    best_model_name = best["Model"]
    best_threshold = float(best["Threshold"])
    best_model = fitted_models[best_model_name]

    threshold_tables[best_model_name].to_csv(OUTPUT_DIR / f"{reference_name}_threshold_search_{best_model_name}.csv", index=False)

    final_prob = best_model.predict_proba(X_final)[:, 1]
    final_metrics = pd.DataFrame([
        metric_row(reference_name, "final_validation", best_model_name, y_final, final_prob, best_threshold)
    ])
    final_metrics.to_csv(OUTPUT_DIR / f"{reference_name}_final_validation_metrics.csv", index=False)

    pd.DataFrame({
        "sample_id": final_df["sample_id"],
        "class_name": final_df["class_name"],
        "true_label": y_final,
        "predicted_probability": final_prob,
        "predicted_label": (final_prob >= best_threshold).astype(int),
    }).to_csv(OUTPUT_DIR / f"{reference_name}_final_validation_predictions.csv", index=False)

    print(f"\n[{reference_name}] Best model selected on test split: {best_model_name}")
    print(f"[{reference_name}] Selected threshold: {best_threshold:.3f}")
    print(f"\n[{reference_name}] TEST results")
    print(test_results)
    print(f"\n[{reference_name}] FINAL VALIDATION results")
    print(final_metrics)
    print("\nClassification report on final validation:")
    print(classification_report(
        y_final,
        (final_prob >= best_threshold).astype(int),
        target_names=["Non-Cancer Liver Disease", "Liver Cancer"],
    ))

    plot_roc(reference_name, best_model_name, y_final, final_prob)
    plot_confusion(reference_name, y_final, (final_prob >= best_threshold).astype(int))

    return test_results, final_metrics


def plot_roc(reference_name, model_name, y_true, probabilities):
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc_score = roc_auc_score(y_true, probabilities)

    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, linewidth=2.5, label=f"{model_name} AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"MiNAE {reference_name} Final Validation ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Sensitivity")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{reference_name}_final_validation_roc.png", dpi=300)
    plt.close()


def plot_confusion(reference_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Control", "Predicted Liver Cancer"],
        yticklabels=["Actual Control", "Actual Liver Cancer"],
    )
    plt.title(f"MiNAE {reference_name} Final Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{reference_name}_final_validation_confusion_matrix.png", dpi=300)
    plt.close()


def main():
    all_test_results = []
    all_final_results = []

    for reference_name, config in EXPERIMENTS.items():
        print("\n" + "=" * 80)
        print(f"Running independent experiment for {reference_name}")
        print("=" * 80)

        manifest = build_manifest(reference_name, config)
        feature_df = build_feature_matrix(reference_name, manifest)
        test_results, final_results = evaluate_reference(reference_name, feature_df)
        all_test_results.append(test_results)
        all_final_results.append(final_results)

    combined_test = pd.concat(all_test_results, ignore_index=True)
    combined_final = pd.concat(all_final_results, ignore_index=True)
    combined_test.to_csv(OUTPUT_DIR / "combined_test_model_selection.csv", index=False)
    combined_final.to_csv(OUTPUT_DIR / "combined_final_validation_metrics.csv", index=False)

    print("\n" + "=" * 80)
    print("COMBINED FINAL VALIDATION SUMMARY")
    print("=" * 80)
    print(combined_final)
    print(f"\nAll outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
