# MiNAE: A Dual-Reference Cell-Free DNA Fragmentomics Framework for Non-Invasive Detection of Hepatocellular Carcinoma

## Abstract

Early detection of hepatocellular carcinoma (HCC) remains difficult because existing surveillance tools, including ultrasound and alpha-fetoprotein testing, have imperfect sensitivity and specificity, especially in patients with chronic liver disease. This project presents MiNAE, a machine learning framework that uses plasma cell-free DNA (cfDNA) fragmentomics to distinguish HCC from non-cancer liver disease controls. Instead of detecting tumor mutations, MiNAE analyzes the physical size architecture of cfDNA fragments from FinaleDB fragment-coordinate files. The dataset included liver cancer, hepatitis B, and cirrhosis samples, with patient-level deduplication to prevent leakage across hg38 and hg19 reference builds. MiNAE evaluated global fragment-size profiles, dual-reference hg38/hg19 fusion, chromosome-targeted fragmentomics, and combined global-chromosome feature sets. In a single held-out final validation split, the dual-reference global fusion model achieved an ROC-AUC of 0.90, accuracy of 87.5%, sensitivity of 85.0%, specificity of 90.0%, and precision of 89.5%. Across 100 repeated held-out validations, hg38 global fragmentomics showed the most stable average performance, with mean ROC-AUC of 0.889 and mean accuracy of 0.816. A permutation test produced a p-value of 0.002, supporting that the observed classification signal was unlikely under shuffled training labels. However, a technical-feature negative control showed non-trivial signal, indicating that future external validation and batch-effect analysis are necessary. These results support cfDNA fragment-size architecture as a promising non-invasive signal for HCC screening research while emphasizing that MiNAE remains a research prototype rather than a clinically validated diagnostic.

## 1. Introduction

Hepatocellular carcinoma is a major cause of cancer-related mortality worldwide. Early detection is clinically important because treatment options and survival are much better when HCC is identified at an earlier stage. However, current screening approaches are imperfect. Ultrasound can miss small tumors, particularly in cirrhotic liver tissue, and alpha-fetoprotein can be elevated in non-malignant inflammatory liver disease. These limitations create a need for non-invasive biomarkers that can detect HCC while distinguishing it from chronic liver disease.

Liquid biopsy has become an important area of cancer diagnostics, but many approaches rely on detecting tumor-specific mutations in circulating tumor DNA. Mutation-based assays often require deep sequencing because tumor-derived DNA can represent a small fraction of total cfDNA. MiNAE explores a different approach: fragmentomics. Rather than reading the DNA sequence itself, fragmentomics examines the physical structure of cfDNA fragments, including their length distribution and genomic origin.

Healthy and non-malignant cells release cfDNA through relatively structured biological processes, often creating fragment-size distributions enriched near nucleosome-protected lengths. Cancer can disrupt this structure through abnormal cell death, chromatin organization, and nuclease activity. MiNAE tests whether these fragment-size patterns can distinguish HCC from non-cancer liver disease controls such as hepatitis B and cirrhosis.

The central research question is:

> Can cfDNA fragment-size architecture distinguish hepatocellular carcinoma from chronic liver disease controls using a reproducible machine learning pipeline?

## 2. Dataset

The project used publicly available cfDNA fragment-coordinate files from FinaleDB. Three disease groups were used:

- Liver cancer / HCC
- Hepatitis B
- Cirrhosis

The local raw data folders contained:

| Folder | Files | Unique sample IDs |
|---|---:|---:|
| liver hg38 | 89 | 89 |
| liver hg19 | 89 | 89 |
| hepatitis b hg38 | 61 | 61 |
| hepatitis b hg19 | 60 | 60 |
| cirrhosis hg38 | 67 | 67 |
| cirrhosis hg19 | 68 | 68 |

Because the same biological sample can appear in multiple reference builds or disease-control folders, all modeling was performed at the patient/sample ID level rather than at the file level. The final dual-reference fusion dataset contained 176 unique patients:

| Class | Patients |
|---|---:|
| Liver cancer | 89 |
| Non-cancer liver disease | 87 |
| Total | 176 |

The non-cancer group included:

| Control subtype | Patients |
|---|---:|
| Cirrhosis | 66 |
| Hepatitis B | 21 |

This control group is clinically important because cirrhosis and hepatitis B are not healthy states. They are high-risk liver diseases that can cause inflammation, tissue remodeling, and cfDNA shedding. Therefore, the classification problem is more realistic than a simple cancer-versus-healthy comparison.

## 3. Methods

### 3.1 Fragment Length Calculation

Each `.frag.tsv.bgz` file contains genomic coordinate information. MiNAE reads the chromosome, start coordinate, and end coordinate for each fragment. Fragment length is calculated as:

```text
fragment length = end coordinate - start coordinate
```

Fragments were filtered to retain lengths between:

```text
50 bp and 250 bp
```

For each file, up to 1,500,000 fragment rows were read. For dual-reference fusion, each patient could contribute up to 1,500,000 hg38 fragments and 1,500,000 hg19 fragments.

### 3.2 Global Fragmentomics Features

Global fragmentomics features were calculated across the whole genome. Fragment lengths were binned into 5 bp intervals from 50 to 250 bp. Summary features included short-fragment fraction, nucleosome peak fraction, long-fragment fraction, mean length, median length, interquartile range, skewness, and entropy.

### 3.3 Dual-Reference Fusion

The dual-reference feature set merged three categories of features for the same patient:

```text
hg38 features
hg19 features
hg38 - hg19 delta features
```

The purpose was to test whether reference-sensitive fragmentomic variation improved classification performance. Importantly, hg38 and hg19 were not treated as separate samples. They were combined into one row per patient.

### 3.4 Chromosome-Targeted Features

A chromosome-targeted feature set was also tested. The target groups were:

| Track | Chromosomes |
|---|---|
| Gain-associated | chr1, chr8 |
| Loss-associated | chr4, chr16 |
| Control | chr2, chr3 |

These tracks were chosen to test whether biologically informed chromosome-localized fragmentomics could improve HCC classification beyond global size profiles.

### 3.5 Machine Learning Models

The main model family used a linear support vector machine with feature selection:

```text
StandardScaler
SelectKBest feature selection
Linear SVM
```

This model type was chosen because the dataset has relatively few patients compared with the number of engineered features. Linear models are often more stable than complex tree-based models in small-sample, high-dimensional biomedical settings.

### 3.6 Validation Design

The primary split design used:

```text
Train set: model fitting
Test set: model and threshold selection
Final validation set: untouched final evaluation
```

The main held-out split contained:

| Split | Patients | Cancer | Controls |
|---|---:|---:|---:|
| Train | 96 | 49 | 47 |
| Test | 40 | 20 | 20 |
| Final validation | 40 | 20 | 20 |

There was no patient overlap between train, test, and final validation sets.

To test robustness, the split process was repeated 100 times using random patient-level splits.

## 4. Results

### 4.1 Single Held-Out Final Validation

The strongest single held-out validation result came from the dual-reference global fusion model:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.900 |
| Accuracy | 0.875 |
| Balanced accuracy | 0.875 |
| Sensitivity | 0.850 |
| Specificity | 0.900 |
| Precision | 0.895 |
| F1-score | 0.872 |

The model correctly classified 35 of 40 final validation patients:

```text
17 / 20 liver cancer cases detected
18 / 20 non-cancer liver disease controls correctly rejected
```

### 4.2 Ablation Study

The ablation study compared five feature strategies on the same held-out split:

| Strategy | ROC-AUC | Accuracy | Sensitivity | Specificity | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| hg38 global | 0.828 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 |
| hg19 global | 0.838 | 0.725 | 0.700 | 0.750 | 0.737 | 0.718 |
| dual-reference global fusion | 0.900 | 0.875 | 0.850 | 0.900 | 0.895 | 0.872 |
| chromosome-targeted fusion | 0.890 | 0.875 | 0.850 | 0.900 | 0.895 | 0.872 |
| global + chromosome fusion | 0.869 | 0.725 | 0.600 | 0.850 | 0.800 | 0.686 |

Dual-reference fusion achieved the strongest single-split AUC. Chromosome-targeted fusion performed nearly as well, suggesting that chromosome-localized fragmentomics carries independent signal. However, combining all global and chromosome-targeted features reduced performance, likely because the feature set became too large for the sample size.

### 4.3 Repeated Held-Out Validation

Across 100 repeated random held-out validations, hg38 global fragmentomics showed the strongest average stability:

| Strategy | Mean ROC-AUC | SD | Mean Accuracy | Mean Sensitivity | Mean Specificity | Mean Precision |
|---|---:|---:|---:|---:|---:|---:|
| hg38 global | 0.889 | 0.049 | 0.816 | 0.762 | 0.869 | 0.873 |
| hg19 global | 0.886 | 0.048 | 0.809 | 0.755 | 0.863 | 0.864 |
| chromosome fusion | 0.872 | 0.052 | 0.761 | 0.727 | 0.795 | 0.806 |
| dual global fusion | 0.868 | 0.050 | 0.766 | 0.737 | 0.795 | 0.804 |
| global + chromosome | 0.847 | 0.057 | 0.751 | 0.698 | 0.805 | 0.798 |

This result adds nuance. Although dual-reference fusion performed best in the single final validation split, hg38 global fragmentomics was the most stable strategy across repeated random splits.

### 4.4 Permutation Test

A permutation test was performed by shuffling training labels and evaluating final-validation AUC under the null hypothesis of no relationship between features and disease label. The observed model achieved:

| Metric | Value |
|---|---:|
| Decision-score ROC-AUC | 0.925 |
| Accuracy | 0.875 |
| Permutation p-value | 0.002 |

This supports that the observed signal is unlikely to arise by random label assignment.

### 4.5 Negative Control Model

A technical-feature negative control was trained using only fragment-count-derived technical features:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.835 |
| Accuracy | 0.700 |
| Sensitivity | 0.400 |
| Specificity | 1.000 |

This result is important. It shows that technical features contain non-trivial class-associated information. The final MiNAE models excluded fragment-count features from the primary feature set, but this negative-control result still suggests that batch effects, sequencing depth, or technical artifacts must be investigated in future work.

### 4.6 Error Analysis

In the final validation set, MiNAE made five errors:

| Error type | Count |
|---|---:|
| False positives | 2 |
| False negatives | 3 |

By disease group:

| Disease group | Correct | Error |
|---|---:|---:|
| Cirrhosis | 11 | 1 false positive |
| Hepatitis B | 7 | 1 false positive |
| Liver cancer | 17 | 3 false negatives |

This suggests that both types of non-cancer liver disease can occasionally resemble cancer-like fragmentomic profiles, and some HCC samples remain difficult to detect.

## 5. Discussion

MiNAE provides proof-of-concept evidence that cfDNA fragmentomics can distinguish HCC from chronic liver disease controls. The strongest single validation result achieved 0.90 ROC-AUC, and repeated validation showed that fragment-size features remain predictive across many random splits.

The ablation study was especially informative. It showed that dual-reference fusion and chromosome-targeted fusion can both perform strongly. However, simply adding more features did not improve performance. In fact, the global-plus-chromosome model performed worse, suggesting overfitting. This is important because it shows that MiNAE benefits from disciplined feature design rather than indiscriminate feature expansion.

The repeated-validation results also changed the interpretation. Although dual-reference fusion produced the best single held-out result, hg38 global fragmentomics was the most stable strategy on average. Therefore, the final scientific conclusion should not be that dual-reference fusion is universally superior. A more accurate conclusion is that dual-reference fusion can produce strong performance, while hg38 global fragmentomics appears more stable across random splits in the current dataset.

The statistical audit strengthened the project by adding a permutation test and negative control. The permutation test supported that the observed signal is unlikely under random labels. However, the technical negative control revealed that non-biological features may also carry class-associated information. This is a serious limitation and should be handled transparently. It does not invalidate the project, but it means future work should include stronger batch correction, external validation, and study-level splitting.

## 6. Limitations

This study has several limitations:

1. The data come from public retrospective datasets rather than a prospective clinical cohort.
2. The final validation set is small.
3. There is no independent external validation cohort.
4. Technical features showed non-trivial predictive signal, raising possible batch-effect concerns.
5. The model has not been compared directly against AFP, ultrasound, or clinical risk scores.
6. Early-stage HCC-specific performance could not be assessed without reliable tumor stage metadata.
7. MiNAE is not clinically validated and should not be used for medical decision-making.

## 7. Future Work

The next steps are:

1. External validation on an independent HCC cfDNA cohort.
2. Study-level validation, where one study is used for training and another for testing.
3. Batch-effect analysis and technical covariate adjustment.
4. Early-stage HCC analysis if tumor-stage metadata become available.
5. Comparison against AFP and clinical risk factors.
6. Expanded controls including HCV, NAFLD/NASH, benign liver nodules, and other cancers.
7. More biologically grounded chromosome-localized and nucleosome-aware features.

## 8. Conclusion

MiNAE demonstrates that cfDNA fragment-size architecture contains signal for distinguishing hepatocellular carcinoma from chronic liver disease controls. A dual-reference fusion model achieved 0.90 ROC-AUC on a held-out final validation split, while repeated validation showed stable performance near 0.89 ROC-AUC for hg38 global fragmentomics. These results support cfDNA fragmentomics as a promising non-invasive signal for HCC screening research. However, external validation and stronger control of technical confounding are required before any clinical claims can be made.

## Suggested Figures

1. MiNAE workflow schematic
2. Dataset composition table
3. Fragment-size distribution plot
4. ROC curve for final validation
5. Precision-recall curve
6. Confusion matrix
7. Ablation heatmap
8. Repeated-validation boxplot
9. SHAP beeswarm plot
10. Permutation-test null distribution
11. Error-analysis table

## Suggested Final Claim

> MiNAE is a reproducible research framework showing that cfDNA fragmentomics can distinguish HCC from cirrhosis and hepatitis B controls, with a peak held-out AUC of 0.90 and repeated-validation performance near 0.89 AUC. The strongest contribution is not a clinical diagnostic claim, but a transparent computational framework combining ablation, repeated validation, interpretability, and statistical auditing.

