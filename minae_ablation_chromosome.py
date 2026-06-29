from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

PROJECT_DIR = Path.home() / "Documents" / "MiNAE_Global_Local"
DUAL_DIR = PROJECT_DIR / "dual_reference_outputs"
FUSION_DIR = PROJECT_DIR / "dual_reference_fusion_outputs"
ABLATION_DIR = PROJECT_DIR / "ablation_chromosome_outputs"
ABLATION_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
ROW_LIMIT = 1_500_000

FINE_BINS = list(range(50, 255, 5))
FINE_BIN_NAMES = [f"Bin_{FINE_BINS[i]}_{FINE_BINS[i + 1]}" for i in range(len(FINE_BINS) - 1)]

CHROMOSOME_TRACKS = {
    "gain_chr1_8": {"chr1", "chr8", "1", "8"},
    "loss_chr4_16": {"chr4", "chr16", "4", "16"},
    "control_chr2_3": {"chr2", "chr3", "2", "3"},
}

RAW_FOLDERS = {
    "hg38": {
        "liver": Path.home() / "Downloads" / "liver hg38",
        "hepatitis": Path.home() / "Downloads" / "hepatitis b hg38",
        "cirrhosis": Path.home() / "Downloads" / "cirrhosis hg38",
        "pattern": "*.hg38.frag.tsv.bgz",
    },
    "hg19": {
        "liver": Path.home() / "Downloads" / "liver hg19",
        "hepatitis": Path.home() / "Downloads" / "hepatitis b hg19",
        "cirrhosis": Path.home() / "Downloads" / "cirrhosis hg19",
        "pattern": "*.hg19.frag.tsv.bgz",
    },
}


def get_sample_id(file_path):
    match = re.search(r"(EE\d+)", file_path.name)
    return match.group(1) if match else file_path.stem


def build_file_index(reference):
    config = RAW_FOLDERS[reference]
    index = {}
    for folder_key in ["liver", "hepatitis", "cirrhosis"]:
        for file_path in sorted(config[folder_key].glob(config["pattern"])):
            index.setdefault(get_sample_id(file_path), file_path)
    return index


def load_existing_splits():
    required = [
        FUSION_DIR / "fusion_train_split.csv",
        FUSION_DIR / "fusion_test_split.csv",
        FUSION_DIR / "fusion_final_validation_split.csv",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing fusion split files. Run minae_dual_reference_fusion.py first.\n"
            + "\n".join(str(path) for path in missing)
        )

    train = pd.read_csv(required[0])
    test = pd.read_csv(required[1])
    final = pd.read_csv(required[2])
    return train, test, final


def global_feature_tables():
    hg38 = pd.read_csv(DUAL_DIR / "dna_features_hg38.csv")
    hg19 = pd.read_csv(DUAL_DIR / "dna_features_hg19.csv")
    return hg38, hg19


def normalize_chrom(chrom):
    chrom = str(chrom)
    if chrom.startswith("chr"):
        return chrom
    return f"chr{chrom}"


def fraction_between(lengths, low, high, include_high=False):
    if len(lengths) == 0:
        return 0.0
    if include_high:
        return float(np.mean((lengths >= low) & (lengths <= high)))
    return float(np.mean((lengths >= low) & (lengths < high)))


def safe_ratio(numerator, denominator):
    return float(numerator / (denominator + 1e-12))


def entropy(probabilities):
    probabilities = probabilities[probabilities > 0]
    if len(probabilities) == 0:
        return 0.0
    return float(-(probabilities * np.log2(probabilities)).sum())


def summarize_lengths(prefix, lengths):
    features = {}
    if len(lengths) == 0:
        for name in FINE_BIN_NAMES:
            features[f"{prefix}_{name}"] = 0.0
        for stat in [
            "Short_Fraction_50_150",
            "UltraShort_Fraction_50_100",
            "Nucleosome_Peak_160_180",
            "Long_Fraction_180_250",
            "Short_to_Long",
            "Mean_Length",
            "Median_Length",
            "IQR_Length",
            "Entropy_Size_Profile",
            "Fragment_Count",
        ]:
            features[f"{prefix}_{stat}"] = 0.0
        return features

    counts, _ = np.histogram(lengths, bins=FINE_BINS)
    probabilities = counts / counts.sum()
    for name, value in zip(FINE_BIN_NAMES, probabilities):
        features[f"{prefix}_{name}"] = float(value)

    short = fraction_between(lengths, 50, 150)
    long = fraction_between(lengths, 150, 250, include_high=True)
    q25, q75 = np.percentile(lengths, [25, 75])

    features.update({
        f"{prefix}_Short_Fraction_50_150": short,
        f"{prefix}_UltraShort_Fraction_50_100": fraction_between(lengths, 50, 100),
        f"{prefix}_Nucleosome_Peak_160_180": fraction_between(lengths, 160, 180, include_high=True),
        f"{prefix}_Long_Fraction_180_250": fraction_between(lengths, 180, 250, include_high=True),
        f"{prefix}_Short_to_Long": safe_ratio(short, long),
        f"{prefix}_Mean_Length": float(np.mean(lengths)),
        f"{prefix}_Median_Length": float(np.median(lengths)),
        f"{prefix}_IQR_Length": float(q75 - q25),
        f"{prefix}_Entropy_Size_Profile": entropy(probabilities),
        f"{prefix}_Fragment_Count": int(len(lengths)),
    })
    return features


def extract_chromosome_features(file_path):
    df = pd.read_csv(
        file_path,
        sep="\t",
        compression="gzip",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "start", "end"],
        nrows=ROW_LIMIT,
    )
    df["chrom"] = df["chrom"].map(normalize_chrom)
    df["length"] = df["end"] - df["start"]
    df = df[(df["length"] >= 50) & (df["length"] <= 250)]

    features = {}
    for track_name, chromosomes in CHROMOSOME_TRACKS.items():
        normalized = {normalize_chrom(chrom) for chrom in chromosomes}
        lengths = df.loc[df["chrom"].isin(normalized), "length"].to_numpy()
        features.update(summarize_lengths(track_name, lengths))

    gain_short = features["gain_chr1_8_Short_Fraction_50_150"]
    loss_short = features["loss_chr4_16_Short_Fraction_50_150"]
    control_short = features["control_chr2_3_Short_Fraction_50_150"]
    features["gain_short_minus_control_short"] = gain_short - control_short
    features["loss_short_minus_control_short"] = loss_short - control_short
    features["gain_loss_short_ratio"] = safe_ratio(gain_short, loss_short)
    return features


def build_chromosome_feature_table(reference, sample_ids):
    output = ABLATION_DIR / f"chromosome_features_{reference}.csv"
    if output.exists():
        return pd.read_csv(output)

    file_index = build_file_index(reference)
    rows = []
    for idx, sample_id in enumerate(sample_ids, start=1):
        if sample_id not in file_index:
            print(f"[{reference}] Missing raw file for {sample_id}; skipping")
            continue
        print(f"[{reference}] Chromosome features {idx}/{len(sample_ids)}: {sample_id}")
        features = extract_chromosome_features(file_index[sample_id])
        features["sample_id"] = sample_id
        rows.append(features)

    table = pd.DataFrame(rows)
    leading = ["sample_id"]
    table = table[leading + [col for col in table.columns if col not in leading]]
    table.to_csv(output, index=False)
    return table


def add_reference_suffix(df, suffix):
    metadata = {"sample_id", "class_label", "class_name"}
    rename = {col: f"{col}_{suffix}" for col in df.columns if col not in metadata}
    return df.rename(columns=rename)


def fusion_from_pair(hg38, hg19, label_df=None):
    metadata_cols = ["sample_id"]
    if label_df is not None:
        metadata = label_df[["sample_id", "class_label", "class_name"]]
    else:
        metadata = hg38[["sample_id", "class_label", "class_name"]]

    hg38_features = add_reference_suffix(hg38.drop(columns=[c for c in ["class_label", "class_name"] if c in hg38.columns]), "hg38")
    hg19_features = add_reference_suffix(hg19.drop(columns=[c for c in ["class_label", "class_name"] if c in hg19.columns]), "hg19")
    merged = metadata.merge(hg38_features, on="sample_id").merge(hg19_features, on="sample_id")

    shared = []
    for col in hg38.columns:
        if col in {"sample_id", "class_label", "class_name"}:
            continue
        if col in hg19.columns:
            shared.append(col)

    delta = {
        f"{col}_delta": merged[f"{col}_hg38"] - merged[f"{col}_hg19"]
        for col in shared
    }
    return pd.concat([merged, pd.DataFrame(delta)], axis=1)


def filter_to_split(df, split_df):
    return split_df[["sample_id", "class_label", "class_name"]].merge(
        df.drop(columns=[c for c in ["class_label", "class_name"] if c in df.columns]),
        on="sample_id",
        how="inner",
    )


def get_feature_columns(df):
    excluded = {"sample_id", "class_label", "class_name"}
    return [col for col in df.columns if col not in excluded and not col.endswith("_Fragment_Count")]


def model_library(n_features):
    k_values = sorted({min(40, n_features), min(70, n_features), min(100, n_features)})
    models = {}
    for k in k_values:
        models[f"Linear SVM k{k}"] = Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k)),
            ("model", SVC(
                kernel="linear",
                C=0.1,
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ])
    return models


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp + 1e-12)


def metric_row(strategy, split, model, y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "Strategy": strategy,
        "Split": split,
        "Model": model,
        "Threshold": threshold,
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "PR_AUC": average_precision_score(y_true, probabilities),
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, predictions),
        "Sensitivity": recall_score(y_true, predictions),
        "Specificity": specificity_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions),
    }


def choose_threshold(strategy, model_name, y_true, probabilities):
    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        rows.append(metric_row(strategy, "test", model_name, y_true, probabilities, threshold))
    table = pd.DataFrame(rows)
    table = table.sort_values(["Balanced_Accuracy", "Accuracy", "Precision"], ascending=False)
    return float(table.iloc[0]["Threshold"]), table


def evaluate_strategy(strategy_name, train_df, test_df, final_df):
    feature_cols = get_feature_columns(train_df)
    X_train, y_train = train_df[feature_cols], train_df["class_label"]
    X_test, y_test = test_df[feature_cols], test_df["class_label"]
    X_final, y_final = final_df[feature_cols], final_df["class_label"]

    rows = []
    fitted = {}
    thresholds = {}
    for model_name, model in model_library(len(feature_cols)).items():
        print(f"[{strategy_name}] training {model_name}")
        fitted_model = clone(model).fit(X_train, y_train)
        test_prob = fitted_model.predict_proba(X_test)[:, 1]
        threshold, threshold_table = choose_threshold(strategy_name, model_name, y_test, test_prob)
        threshold_table.to_csv(ABLATION_DIR / f"{strategy_name}_{model_name.replace(' ', '_')}_thresholds.csv", index=False)

        rows.append(metric_row(strategy_name, "test", model_name, y_test, test_prob, threshold))
        fitted[model_name] = fitted_model
        thresholds[model_name] = threshold

    test_results = pd.DataFrame(rows).sort_values(["Balanced_Accuracy", "ROC_AUC"], ascending=False)
    best_model_name = test_results.iloc[0]["Model"]
    best_threshold = float(test_results.iloc[0]["Threshold"])

    development = pd.concat([train_df, test_df], axis=0)
    X_dev, y_dev = development[feature_cols], development["class_label"]
    final_model = clone(model_library(len(feature_cols))[best_model_name]).fit(X_dev, y_dev)
    final_prob = final_model.predict_proba(X_final)[:, 1]
    final_pred = (final_prob >= best_threshold).astype(int)

    final_result = pd.DataFrame([
        metric_row(strategy_name, "final_validation", best_model_name, y_final, final_prob, best_threshold)
    ])
    prediction_df = pd.DataFrame({
        "sample_id": final_df["sample_id"],
        "class_name": final_df["class_name"],
        "true_label": y_final,
        "predicted_probability": final_prob,
        "predicted_label": final_pred,
    })
    prediction_df.to_csv(ABLATION_DIR / f"{strategy_name}_final_predictions.csv", index=False)

    plot_strategy_roc(strategy_name, y_final, final_prob, final_result.iloc[0]["ROC_AUC"])
    plot_strategy_confusion(strategy_name, y_final, final_pred)

    return test_results, final_result


def plot_strategy_roc(strategy_name, y_true, probabilities, auc_score):
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, linewidth=2.5, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"{strategy_name} Final Validation ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(ABLATION_DIR / f"{strategy_name}_final_roc.png", dpi=300)
    plt.close()


def plot_strategy_confusion(strategy_name, y_true, predictions):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        confusion_matrix(y_true, predictions),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Control", "Predicted HCC"],
        yticklabels=["Actual Control", "Actual HCC"],
    )
    plt.title(f"{strategy_name} Final Validation")
    plt.tight_layout()
    plt.savefig(ABLATION_DIR / f"{strategy_name}_final_confusion.png", dpi=300)
    plt.close()


def build_strategy_tables():
    train_split, test_split, final_split = load_existing_splits()
    all_sample_ids = sorted(set(pd.concat([train_split, test_split, final_split])["sample_id"]))

    global_hg38, global_hg19 = global_feature_tables()
    global_fusion = pd.read_csv(FUSION_DIR / "dual_reference_fusion_features.csv")

    chrom_hg38 = build_chromosome_feature_table("hg38", all_sample_ids)
    chrom_hg19 = build_chromosome_feature_table("hg19", all_sample_ids)
    label_df = global_fusion[["sample_id", "class_label", "class_name"]]
    chrom_fusion = fusion_from_pair(chrom_hg38, chrom_hg19, label_df=label_df)
    chrom_fusion.to_csv(ABLATION_DIR / "chromosome_fusion_features.csv", index=False)

    global_plus_chrom = label_df.merge(
        global_fusion.drop(columns=["class_label", "class_name"]),
        on="sample_id",
    ).merge(
        chrom_fusion.drop(columns=["class_label", "class_name"]),
        on="sample_id",
        suffixes=("", "_chrom"),
    )
    global_plus_chrom.to_csv(ABLATION_DIR / "global_plus_chromosome_features.csv", index=False)

    strategies = {
        "strategy_1_hg38_global": global_hg38,
        "strategy_2_hg19_global": global_hg19,
        "strategy_3_dual_global_fusion": global_fusion,
        "strategy_4_chromosome_fusion": chrom_fusion,
        "strategy_5_global_plus_chromosome": global_plus_chrom,
    }

    split_tables = {}
    for name, table in strategies.items():
        split_tables[name] = (
            filter_to_split(table, train_split),
            filter_to_split(table, test_split),
            filter_to_split(table, final_split),
        )
        for split_name, split_table in zip(["train", "test", "final_validation"], split_tables[name]):
            split_table.to_csv(ABLATION_DIR / f"{name}_{split_name}.csv", index=False)
    return split_tables


def plot_summary_table(results):
    metrics = ["ROC_AUC", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1"]
    plot_df = results.set_index("Strategy")[metrics]
    plt.figure(figsize=(11, 5))
    sns.heatmap(plot_df, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.5, vmax=1.0)
    plt.title("MiNAE Ablation Study: Final Validation Performance")
    plt.tight_layout()
    plt.savefig(ABLATION_DIR / "ablation_final_metric_heatmap.png", dpi=300)
    plt.close()


def main():
    split_tables = build_strategy_tables()
    all_test = []
    all_final = []

    for strategy_name, (train_df, test_df, final_df) in split_tables.items():
        print("\n" + "=" * 80)
        print(strategy_name)
        print("=" * 80)
        test_result, final_result = evaluate_strategy(strategy_name, train_df, test_df, final_df)
        test_result.to_csv(ABLATION_DIR / f"{strategy_name}_test_model_selection.csv", index=False)
        final_result.to_csv(ABLATION_DIR / f"{strategy_name}_final_metrics.csv", index=False)
        all_test.append(test_result)
        all_final.append(final_result)

    combined_test = pd.concat(all_test, ignore_index=True)
    combined_final = pd.concat(all_final, ignore_index=True)
    combined_test.to_csv(ABLATION_DIR / "combined_test_model_selection.csv", index=False)
    combined_final.to_csv(ABLATION_DIR / "combined_final_validation_metrics.csv", index=False)
    plot_summary_table(combined_final)

    print("\nFinal ablation ledger:")
    print(combined_final.to_string(index=False))
    print(f"\nOutputs saved in: {ABLATION_DIR}")


if __name__ == "__main__":
    main()
