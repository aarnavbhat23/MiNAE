from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

PROJECT_DIR = Path.home() / "Documents" / "MiNAE_Global_Local"
DUAL_DIR = PROJECT_DIR / "dual_reference_outputs"
FUSION_DIR = PROJECT_DIR / "dual_reference_fusion_outputs"
FUSION_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_PER_CLASS = 20
FINAL_PER_CLASS = 20


def load_dual_reference_features():
    hg38_path = DUAL_DIR / "dna_features_hg38.csv"
    hg19_path = DUAL_DIR / "dna_features_hg19.csv"

    if not hg38_path.exists() or not hg19_path.exists():
        raise FileNotFoundError(
            "Missing cached hg38/hg19 feature files. Run minae_dual_reference_split.py first."
        )

    hg38 = pd.read_csv(hg38_path)
    hg19 = pd.read_csv(hg19_path)

    exclude = {"sample_id", "class_label", "class_name", "Total_Used_Fragments"}
    feature_cols = [col for col in hg38.columns if col not in exclude]

    merged = hg38[["sample_id", "class_label", "class_name"] + feature_cols].merge(
        hg19[["sample_id"] + feature_cols],
        on="sample_id",
        suffixes=("_hg38", "_hg19"),
    )

    delta_features = {
        f"{col}_delta": merged[f"{col}_hg38"] - merged[f"{col}_hg19"]
        for col in feature_cols
    }
    merged = pd.concat([merged, pd.DataFrame(delta_features)], axis=1)
    merged.to_csv(FUSION_DIR / "dual_reference_fusion_features.csv", index=False)

    print("Dual-reference fusion table:")
    print(merged["class_label"].value_counts().sort_index())
    print("Shape:", merged.shape)

    return merged


def make_balanced_splits(df):
    train_parts = []
    test_parts = []
    final_parts = []

    for label in [0, 1]:
        subset = df[df["class_label"] == label].sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        if len(subset) < TEST_PER_CLASS + FINAL_PER_CLASS + 1:
            raise ValueError(f"Class {label} does not have enough samples for this split.")

        final_parts.append(subset.iloc[:FINAL_PER_CLASS])
        test_parts.append(subset.iloc[FINAL_PER_CLASS:FINAL_PER_CLASS + TEST_PER_CLASS])
        train_parts.append(subset.iloc[FINAL_PER_CLASS + TEST_PER_CLASS:])

    train = pd.concat(train_parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    final = pd.concat(final_parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    train.to_csv(FUSION_DIR / "fusion_train_split.csv", index=False)
    test.to_csv(FUSION_DIR / "fusion_test_split.csv", index=False)
    final.to_csv(FUSION_DIR / "fusion_final_validation_split.csv", index=False)

    print("\nSplit sizes:")
    print("Train:", train["class_label"].value_counts().sort_index().to_dict())
    print("Test:", test["class_label"].value_counts().sort_index().to_dict())
    print("Final:", final["class_label"].value_counts().sort_index().to_dict())

    return train, test, final


def get_feature_columns(df):
    excluded = {"sample_id", "class_label", "class_name"}
    return [col for col in df.columns if col not in excluded]


def model_library():
    return {
        "L2 Logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=20_000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Linear SVM k40": Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=40)),
            ("model", SVC(
                kernel="linear",
                C=0.1,
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "Linear SVM k70": Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=70)),
            ("model", SVC(
                kernel="linear",
                C=0.1,
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "RBF SVM k40": Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=40)),
            ("model", SVC(
                kernel="rbf",
                C=0.5,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=1_000,
            max_depth=4,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.02,
            max_iter=350,
            max_leaf_nodes=6,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp + 1e-12)


def metric_row(split, model, y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "Split": split,
        "Model": model,
        "Threshold": threshold,
        "ROC_AUC": roc_auc_score(y_true, probabilities),
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, predictions),
        "Sensitivity": recall_score(y_true, predictions),
        "Specificity": specificity_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions),
    }


def choose_threshold(y_true, probabilities):
    rows = []
    for threshold in np.linspace(0.05, 0.95, 181):
        rows.append(metric_row("test", "candidate", y_true, probabilities, threshold))
    table = pd.DataFrame(rows)
    table = table.sort_values(["Balanced_Accuracy", "Accuracy", "Precision"], ascending=False)
    return float(table.iloc[0]["Threshold"]), table


def evaluate():
    df = load_dual_reference_features()
    train, test, final = make_balanced_splits(df)
    feature_cols = get_feature_columns(df)

    X_train = train[feature_cols]
    y_train = train["class_label"]
    X_test = test[feature_cols]
    y_test = test["class_label"]
    X_final = final[feature_cols]
    y_final = final["class_label"]

    rows = []
    fitted = {}
    thresholds = {}

    for name, model in model_library().items():
        print(f"\nTraining {name}")
        fitted_model = clone(model).fit(X_train, y_train)
        test_prob = fitted_model.predict_proba(X_test)[:, 1]
        threshold, threshold_table = choose_threshold(y_test, test_prob)

        fitted[name] = fitted_model
        thresholds[name] = threshold
        threshold_table.to_csv(FUSION_DIR / f"threshold_search_{name.replace(' ', '_')}.csv", index=False)
        rows.append(metric_row("test", name, y_test, test_prob, threshold))

    test_results = pd.DataFrame(rows).sort_values(["Balanced_Accuracy", "ROC_AUC"], ascending=False)
    test_results.to_csv(FUSION_DIR / "fusion_test_model_selection.csv", index=False)

    best_name = test_results.iloc[0]["Model"]
    best_threshold = float(test_results.iloc[0]["Threshold"])

    print("\nTest-set model selection:")
    print(test_results)
    print(f"\nSelected model: {best_name}")
    print(f"Selected threshold: {best_threshold:.3f}")

    # After selecting the model family and threshold on the test set,
    # refit on train + test, then evaluate once on the untouched final set.
    X_development = pd.concat([X_train, X_test], axis=0)
    y_development = pd.concat([y_train, y_test], axis=0)
    final_model = clone(model_library()[best_name]).fit(X_development, y_development)
    final_prob = final_model.predict_proba(X_final)[:, 1]
    final_pred = (final_prob >= best_threshold).astype(int)

    final_metrics = pd.DataFrame([
        metric_row("final_validation", best_name, y_final, final_prob, best_threshold)
    ])
    final_metrics.to_csv(FUSION_DIR / "fusion_final_validation_metrics.csv", index=False)

    pd.DataFrame({
        "sample_id": final["sample_id"],
        "class_name": final["class_name"],
        "true_label": y_final,
        "predicted_probability": final_prob,
        "predicted_label": final_pred,
    }).to_csv(FUSION_DIR / "fusion_final_validation_predictions.csv", index=False)

    print("\nFinal validation metrics:")
    print(final_metrics)
    print("\nFinal validation classification report:")
    print(classification_report(
        y_final,
        final_pred,
        target_names=["Non-Cancer Liver Disease", "Liver Cancer"],
    ))

    plot_roc(y_final, final_prob, best_name)
    plot_precision_recall(y_final, final_prob, best_name)
    plot_confusion(y_final, final_pred)
    plot_probability_distribution(final, final_prob, best_threshold)
    plot_calibration(y_final, final_prob)
    plot_fragment_profile_means(final)
    plot_feature_heatmap(final_model, X_final, y_final, feature_cols)
    plot_model_explanation(final_model, X_development, X_final, y_final, feature_cols)
    save_bootstrap_intervals(y_final, final_prob, final_pred)


def plot_roc(y_true, probabilities, model_name):
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc_score = roc_auc_score(y_true, probabilities)

    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, linewidth=2.5, label=f"{model_name} AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("MiNAE Dual-Reference Fusion Final Validation ROC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Sensitivity")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FUSION_DIR / "fusion_final_validation_roc.png", dpi=300)
    plt.close()


def plot_precision_recall(y_true, probabilities, model_name):
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    ap_score = average_precision_score(y_true, probabilities)

    plt.figure(figsize=(7, 7))
    plt.plot(recall, precision, linewidth=2.5, label=f"{model_name} AP = {ap_score:.3f}")
    plt.title("MiNAE Dual-Reference Fusion Precision-Recall Curve")
    plt.xlabel("Recall / Sensitivity")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FUSION_DIR / "fusion_final_validation_precision_recall.png", dpi=300)
    plt.close()


def plot_confusion(y_true, predictions):
    cm = confusion_matrix(y_true, predictions)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Control", "Predicted Liver Cancer"],
        yticklabels=["Actual Control", "Actual Liver Cancer"],
    )
    plt.title("MiNAE Dual-Reference Fusion Final Validation")
    plt.tight_layout()
    plt.savefig(FUSION_DIR / "fusion_final_validation_confusion_matrix.png", dpi=300)
    plt.close()


def plot_probability_distribution(final_df, probabilities, threshold):
    plot_df = pd.DataFrame({
        "Class": final_df["class_name"].replace({
            "Hepatitis B Control": "Non-Cancer Liver Disease",
            "Cirrhosis Control": "Non-Cancer Liver Disease",
        }),
        "Predicted Probability": probabilities,
    })

    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=plot_df,
        x="Predicted Probability",
        hue="Class",
        bins=12,
        kde=True,
        stat="density",
        common_norm=False,
        alpha=0.45,
    )
    plt.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.3f}")
    plt.title("Final Validation Probability Separation")
    plt.xlabel("Predicted Probability of Liver Cancer")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FUSION_DIR / "fusion_probability_distribution.png", dpi=300)
    plt.close()


def plot_calibration(y_true, probabilities):
    frac_positive, mean_predicted = calibration_curve(y_true, probabilities, n_bins=5, strategy="uniform")

    plt.figure(figsize=(6, 6))
    plt.plot(mean_predicted, frac_positive, marker="o", linewidth=2, label="MiNAE")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.title("Final Validation Calibration Curve")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Cancer Fraction")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FUSION_DIR / "fusion_calibration_curve.png", dpi=300)
    plt.close()


def plot_fragment_profile_means(final_df):
    # Plot the original 5 bp bin profiles separately for hg38 and hg19.
    for reference in ["hg38", "hg19"]:
        bin_cols = [
            col for col in final_df.columns
            if col.startswith("Bin_") and col.endswith(f"_{reference}")
        ]
        if not bin_cols:
            continue

        display_df = final_df[["class_label"] + bin_cols].copy()
        display_df["Class"] = np.where(display_df["class_label"] == 1, "Liver Cancer", "Non-Cancer Liver Disease")
        means = display_df.groupby("Class")[bin_cols].mean().T
        means.index = [col.replace(f"_{reference}", "").replace("Bin_", "") for col in means.index]

        plt.figure(figsize=(12, 5))
        for class_name in means.columns:
            plt.plot(means.index, means[class_name], marker="o", linewidth=2, label=class_name)
        plt.title(f"Final Validation Mean Fragment-Size Profiles ({reference})")
        plt.xlabel("Fragment Size Bin")
        plt.ylabel("Normalized Frequency")
        plt.xticks(rotation=60)
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(FUSION_DIR / f"fusion_mean_fragment_profiles_{reference}.png", dpi=300)
        plt.close()


def selected_feature_names(model, feature_cols):
    if isinstance(model, Pipeline) and "select" in model.named_steps:
        mask = model.named_steps["select"].get_support()
        return [name for name, keep in zip(feature_cols, mask) if keep]
    return feature_cols


def transformed_feature_matrix(model, X):
    data = X.copy()
    if isinstance(model, Pipeline):
        if "scaler" in model.named_steps:
            data = model.named_steps["scaler"].transform(data)
        if "select" in model.named_steps:
            data = model.named_steps["select"].transform(data)
    return data


def plot_feature_heatmap(model, X_final, y_final, feature_cols):
    selected_names = selected_feature_names(model, feature_cols)
    transformed = transformed_feature_matrix(model, X_final)
    heatmap_df = pd.DataFrame(transformed, columns=selected_names)
    heatmap_df["Class"] = np.where(y_final.to_numpy() == 1, "Liver Cancer", "Control")
    heatmap_df = heatmap_df.sort_values("Class")

    # Keep the plot readable for a paper figure.
    variance_rank = heatmap_df.drop(columns=["Class"]).var().sort_values(ascending=False)
    top_features = variance_rank.head(30).index.tolist()

    plt.figure(figsize=(12, 7))
    sns.heatmap(
        heatmap_df[top_features],
        cmap="vlag",
        center=0,
        yticklabels=heatmap_df["Class"],
        xticklabels=True,
    )
    plt.title("Top Selected Feature Heatmap on Final Validation Set")
    plt.xlabel("Selected Fragmentomics Features")
    plt.ylabel("Final Validation Samples")
    plt.tight_layout()
    plt.savefig(FUSION_DIR / "fusion_selected_feature_heatmap.png", dpi=300)
    plt.close()


def plot_model_explanation(model, X_development, X_final, y_final, feature_cols):
    selected_names = selected_feature_names(model, feature_cols)
    transformed_final = transformed_feature_matrix(model, X_final)

    estimator = model.named_steps["model"] if isinstance(model, Pipeline) else model

    if hasattr(estimator, "coef_"):
        coefs = estimator.coef_.ravel()
        coefficient_df = pd.DataFrame({
            "Feature": selected_names,
            "Coefficient": coefs,
            "Abs_Coefficient": np.abs(coefs),
        }).sort_values("Abs_Coefficient", ascending=False)
        coefficient_df.to_csv(FUSION_DIR / "fusion_model_coefficients.csv", index=False)

        top = coefficient_df.head(25).sort_values("Coefficient")
        plt.figure(figsize=(9, 8))
        colors = ["#2b8cbe" if value > 0 else "#d95f0e" for value in top["Coefficient"]]
        plt.barh(top["Feature"], top["Coefficient"], color=colors)
        plt.title("Top Model Coefficients Driving Liver Cancer Prediction")
        plt.xlabel("Linear Model Coefficient")
        plt.tight_layout()
        plt.savefig(FUSION_DIR / "fusion_model_coefficients.png", dpi=300)
        plt.close()

    try:
        import shap
    except ImportError:
        (FUSION_DIR / "SHAP_NOT_GENERATED.txt").write_text(
            "SHAP is not installed. Run: python3 -m pip install shap\n"
            "Then rerun: python3 minae_dual_reference_fusion.py\n"
        )
        print("\nSHAP not installed. To generate SHAP plots, run:")
        print("python3 -m pip install shap")
        print("Then rerun: python3 minae_dual_reference_fusion.py")
        return

    if not hasattr(estimator, "coef_"):
        (FUSION_DIR / "SHAP_NOT_GENERATED.txt").write_text(
            "SHAP generation is currently implemented for the selected linear model only.\n"
        )
        return

    transformed_development = transformed_feature_matrix(model, X_development)
    explainer = shap.LinearExplainer(estimator, transformed_development)
    shap_values = explainer(transformed_final)

    shap_df = pd.DataFrame(shap_values.values, columns=selected_names)
    shap_df.insert(0, "sample_class", np.where(y_final.to_numpy() == 1, "Liver Cancer", "Control"))
    shap_df.to_csv(FUSION_DIR / "fusion_shap_values_final_validation.csv", index=False)

    plt.figure()
    shap.summary_plot(shap_values.values, transformed_final, feature_names=selected_names, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(FUSION_DIR / "fusion_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        shap_values.values,
        transformed_final,
        feature_names=selected_names,
        plot_type="bar",
        show=False,
        max_display=25,
    )
    plt.tight_layout()
    plt.savefig(FUSION_DIR / "fusion_shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_bootstrap_intervals(y_true, probabilities, predictions):
    rng = np.random.default_rng(RANDOM_STATE)
    values = []
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = np.asarray(predictions)

    for _ in range(2_000):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(y_true[idx], predictions[idx]).ravel()
        values.append({
            "ROC_AUC": roc_auc_score(y_true[idx], probabilities[idx]),
            "Accuracy": accuracy_score(y_true[idx], predictions[idx]),
            "Sensitivity": recall_score(y_true[idx], predictions[idx]),
            "Specificity": tn / (tn + fp + 1e-12),
            "Precision": precision_score(y_true[idx], predictions[idx], zero_division=0),
            "F1": f1_score(y_true[idx], predictions[idx]),
        })

    boot = pd.DataFrame(values)
    intervals = boot.quantile([0.025, 0.5, 0.975]).T
    intervals.columns = ["CI_2.5", "Median", "CI_97.5"]
    intervals.to_csv(FUSION_DIR / "fusion_bootstrap_confidence_intervals.csv")


if __name__ == "__main__":
    evaluate()
    print(f"\nOutputs saved in: {FUSION_DIR}")
