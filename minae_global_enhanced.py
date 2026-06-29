from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

HOME = Path.home()
PROJECT_DIR = HOME / "Documents" / "MiNAE_Global_Local"
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

LIVER_DIR = HOME / "Downloads" / "liver hg38"
HEPATITIS_B_DIR = HOME / "Downloads" / "hepatitis b hg38"
CIRRHOSIS_DIR = HOME / "Downloads" / "cirrhosis hg38"

FEATURE_FILE = PROJECT_DIR / "dna_features_enhanced.csv"
MANIFEST_FILE = PROJECT_DIR / "manifest_enhanced.csv"

FINE_BINS = list(range(50, 255, 5))
FINE_BIN_NAMES = [f"Bin_{FINE_BINS[i]}_{FINE_BINS[i + 1]}" for i in range(len(FINE_BINS) - 1)]

ROW_LIMIT = 1_500_000
RANDOM_STATE = 42


def get_sample_id(file_path):
    match = re.search(r"(EE\d+)", file_path.name)
    return match.group(1) if match else file_path.stem


def collect_files(folder, label, class_name):
    rows = []
    if not folder.exists():
        print(f"WARNING: Folder not found: {folder}")
        return rows

    for file_path in sorted(folder.glob("*.hg38.frag.tsv.bgz")):
        rows.append({
            "sample_id": get_sample_id(file_path),
            "file_path": str(file_path),
            "class_label": label,
            "class_name": class_name,
        })
    return rows


def build_manifest():
    rows = []
    rows.extend(collect_files(LIVER_DIR, 1, "Liver Cancer"))
    rows.extend(collect_files(HEPATITIS_B_DIR, 0, "Hepatitis B Control"))
    rows.extend(collect_files(CIRRHOSIS_DIR, 0, "Cirrhosis Control"))

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("No hg38 .bgz files found. Check your folder paths.")

    print("\nBefore deduplication:")
    print(manifest["class_name"].value_counts())
    print("Total files:", len(manifest))

    # Prefer cirrhosis labels over duplicated hepatitis B labels in the control group.
    priority = {"Cirrhosis Control": 0, "Hepatitis B Control": 1, "Liver Cancer": 2}
    manifest["dedupe_priority"] = manifest["class_name"].map(priority)
    manifest = manifest.sort_values(["sample_id", "dedupe_priority"]).drop(columns=["dedupe_priority"])

    duplicates = manifest[manifest.duplicated("sample_id", keep=False)]
    if len(duplicates) > 0:
        print(f"\nDuplicate sample IDs found: {duplicates['sample_id'].nunique()}")
        print("Keeping one file per patient to prevent leakage.")

    manifest = manifest.drop_duplicates("sample_id", keep="first").reset_index(drop=True)

    print("\nAfter deduplication:")
    print(manifest["class_name"].value_counts())
    print("Total unique samples:", len(manifest))

    manifest.to_csv(MANIFEST_FILE, index=False)
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


def extract_fragment_fingerprint(file_path):
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
        long_180_250 = fraction_between(lengths, 180, 250, include_high=True)
        long_150_250 = fraction_between(lengths, 150, 250, include_high=True)

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


def build_feature_matrix(manifest):
    if FEATURE_FILE.exists():
        print(f"\nUsing cached enhanced feature file: {FEATURE_FILE}")
        return pd.read_csv(FEATURE_FILE)

    print("\nExtracting enhanced global fragmentomics features...")
    print(f"Reading up to {ROW_LIMIT:,} fragments per sample. This happens once.\n")

    rows = []
    for idx, row in manifest.iterrows():
        print(f"[{idx + 1}/{len(manifest)}] Processing {row['sample_id']}")
        features = extract_fragment_fingerprint(row["file_path"])
        if features is None:
            continue

        features["sample_id"] = row["sample_id"]
        features["class_label"] = row["class_label"]
        features["class_name"] = row["class_name"]
        rows.append(features)

    feature_df = pd.DataFrame(rows)
    leading = ["sample_id", "class_label", "class_name"]
    rest = [column for column in feature_df.columns if column not in leading]
    feature_df = feature_df[leading + rest]
    feature_df.to_csv(FEATURE_FILE, index=False)

    print(f"\nSaved enhanced feature matrix to: {FEATURE_FILE}")
    return feature_df


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp + 1e-12)


def threshold_table(y_true, probabilities):
    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        preds = (probabilities >= threshold).astype(int)
        rows.append({
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_true, preds),
            "Balanced_Accuracy": balanced_accuracy_score(y_true, preds),
            "Sensitivity": recall_score(y_true, preds),
            "Specificity": specificity_score(y_true, preds),
            "Precision": precision_score(y_true, preds, zero_division=0),
            "F1": f1_score(y_true, preds),
        })
    return pd.DataFrame(rows)


def model_library():
    return {
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
        "L2 Logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=0.2,
                max_iter=10_000,
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


def evaluate_models(feature_df):
    metadata_cols = {"sample_id", "class_label", "class_name", "Total_Used_Fragments"}
    feature_columns = [col for col in feature_df.columns if col not in metadata_cols]
    X = feature_df[feature_columns]
    y = feature_df["class_label"]

    print("\nDataset:")
    print(feature_df["class_name"].value_counts())
    print("Feature matrix shape:", X.shape)

    repeated_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    prediction_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
    }

    models = model_library()
    rows = []
    probability_store = {}

    for name, model in models.items():
        print(f"\nEvaluating {name} with repeated stratified CV...")
        cv_scores = cross_validate(model, X, y, cv=repeated_cv, scoring=scoring, n_jobs=-1)

        probs = cross_val_predict(model, X, y, cv=prediction_cv, method="predict_proba", n_jobs=-1)[:, 1]
        preds = (probs >= 0.5).astype(int)
        probability_store[name] = probs

        rows.append({
            "Model": name,
            "Repeated_ROC_AUC_Mean": cv_scores["test_roc_auc"].mean(),
            "Repeated_ROC_AUC_SD": cv_scores["test_roc_auc"].std(),
            "OOF_ROC_AUC": roc_auc_score(y, probs),
            "OOF_Accuracy": accuracy_score(y, preds),
            "OOF_Balanced_Accuracy": balanced_accuracy_score(y, preds),
            "OOF_Sensitivity": recall_score(y, preds),
            "OOF_Specificity": specificity_score(y, preds),
            "OOF_Precision": precision_score(y, preds),
            "OOF_F1": f1_score(y, preds),
        })

    results_df = pd.DataFrame(rows).sort_values("Repeated_ROC_AUC_Mean", ascending=False)
    results_df.to_csv(PROJECT_DIR / "enhanced_model_results.csv", index=False)

    print("\nEnhanced model comparison:")
    print(results_df)

    best_model_name = results_df.iloc[0]["Model"]
    best_probs = probability_store[best_model_name]
    thresholds = threshold_table(y, best_probs)
    thresholds.to_csv(PROJECT_DIR / "enhanced_threshold_analysis.csv", index=False)
    best_threshold_row = thresholds.sort_values(
        ["Balanced_Accuracy", "Accuracy"],
        ascending=False,
    ).iloc[0]
    best_threshold = float(best_threshold_row["Threshold"])
    best_preds = (best_probs >= best_threshold).astype(int)

    pd.DataFrame({
        "sample_id": feature_df["sample_id"],
        "class_name": feature_df["class_name"],
        "true_label": y,
        "predicted_probability": best_probs,
        "predicted_label": best_preds,
    }).to_csv(PROJECT_DIR / "enhanced_oof_predictions.csv", index=False)

    print(f"\nBest model by repeated CV ROC-AUC: {best_model_name}")
    print(f"OOF ROC-AUC for best model: {roc_auc_score(y, best_probs):.4f}")
    print("\nBest cross-validated threshold:")
    print(best_threshold_row)
    print("\nClassification report:")
    print(classification_report(
        y,
        best_preds,
        target_names=["Non-Cancer Liver Disease", "Liver Cancer"],
    ))

    plot_roc(y, best_probs, best_model_name)
    plot_confusion_matrix(y, best_preds)
    plot_permutation_importance(models[best_model_name], X, y, feature_columns)


def plot_roc(y, probs, model_name):
    fpr, tpr, _ = roc_curve(y, probs)
    auc_score = roc_auc_score(y, probs)

    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, linewidth=2.5, label=f"{model_name} OOF AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("MiNAE Enhanced Global ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Sensitivity")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "enhanced_roc_curve.png", dpi=300)
    plt.close()


def plot_confusion_matrix(y, preds):
    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Control", "Predicted Liver Cancer"],
        yticklabels=["Actual Control", "Actual Liver Cancer"],
    )
    plt.title("MiNAE Enhanced Confusion Matrix")
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "enhanced_confusion_matrix.png", dpi=300)
    plt.close()


def plot_permutation_importance(model, X, y, feature_columns):
    print("\nFitting best model once for permutation importance...")
    model.fit(X, y)
    result = permutation_importance(
        model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance_Mean": result.importances_mean,
        "Importance_SD": result.importances_std,
    }).sort_values("Importance_Mean", ascending=False)

    importance_df.to_csv(PROJECT_DIR / "enhanced_permutation_importance.csv", index=False)

    top = importance_df.head(20)
    plt.figure(figsize=(10, 7))
    sns.barplot(data=top, x="Importance_Mean", y="Feature")
    plt.title("Top Enhanced Global Features by Permutation Importance")
    plt.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    plt.savefig(PROJECT_DIR / "enhanced_permutation_importance.png", dpi=300)
    plt.close()

    print("\nTop permutation importance features:")
    print(top)


if __name__ == "__main__":
    print("Starting MiNAE Enhanced Global Model...")
    manifest_df = build_manifest()
    feature_df = build_feature_matrix(manifest_df)
    evaluate_models(feature_df)
    print("\nDone.")
    print(f"Outputs saved in: {PROJECT_DIR}")
