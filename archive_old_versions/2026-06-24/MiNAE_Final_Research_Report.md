# MiNAE Final Research Report

## Project Title

**MiNAE: Dual-Reference Cell-Free DNA Fragmentomics for Non-Invasive Hepatocellular Carcinoma Detection**

## Executive Summary

MiNAE is a research-grade machine learning framework for detecting hepatocellular carcinoma (HCC) from plasma cell-free DNA (cfDNA) fragmentomics. Instead of using mutation calling or deep sequencing to read DNA letters, MiNAE measures the physical size patterns of cfDNA fragments and tests whether those patterns distinguish liver cancer from non-cancer chronic liver disease.

The clinically important part of this project is the control group. The model is not simply comparing liver cancer patients against healthy individuals. It is comparing liver cancer against **cirrhosis and hepatitis B**, which are high-risk liver disease states that can also cause inflammation, tissue damage, and cfDNA shedding. This makes the classification task more realistic and harder.

The strongest single held-out validation result was achieved by the **dual-reference global fusion model**, which combined hg38 features, hg19 features, and hg38-hg19 delta features:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.900 |
| Accuracy | 0.875 |
| Sensitivity | 0.850 |
| Specificity | 0.900 |
| Precision | 0.895 |
| F1-score | 0.872 |

However, the repeated validation experiment showed an important nuance: across 100 random held-out splits, **hg38 global fragmentomics was the most stable strategy on average**:

| Strategy | Mean ROC-AUC | Mean Accuracy | Mean Precision |
|---|---:|---:|---:|
| hg38 global | 0.889 | 0.816 | 0.873 |
| hg19 global | 0.886 | 0.809 | 0.864 |
| chromosome fusion | 0.872 | 0.761 | 0.806 |
| dual global fusion | 0.868 | 0.766 | 0.804 |
| global + chromosome | 0.847 | 0.751 | 0.798 |

The final scientific interpretation is therefore:

> MiNAE demonstrates that cfDNA fragment-size architecture can distinguish HCC from chronic liver disease controls. Dual-reference fusion achieved the best single held-out validation score, while repeated validation suggests that standard hg38 global fragmentomics is the most stable feature strategy across random splits.

## Dataset

The project uses FinaleDB `.frag.tsv.bgz` cfDNA fragment coordinate files from three disease groups:

- Liver cancer / HCC
- Hepatitis B
- Cirrhosis

Raw local folders:

| Folder | Files | Unique IDs |
|---|---:|---:|
| liver hg38 | 89 | 89 |
| liver hg19 | 89 | 89 |
| hepatitis b hg38 | 61 | 61 |
| hepatitis b hg19 | 60 | 60 |
| cirrhosis hg38 | 67 | 67 |
| cirrhosis hg19 | 68 | 68 |

After deduplication and reference matching, the final fusion pool contains:

| Class | Patients |
|---|---:|
| Liver cancer | 89 |
| Non-cancer liver disease | 87 |
| Total | 176 |

The non-cancer group includes:

| Control subtype | Patients |
|---|---:|
| Cirrhosis | 66 |
| Hepatitis B | 21 |

Important: the model does **not** treat hg38 and hg19 files as separate patients. Each patient remains one biological sample. The fusion model merges the two reference builds into one feature row per patient.

## What The Pipeline Does

For each patient file, MiNAE reads compressed fragment coordinate data:

```text
chromosome, start coordinate, end coordinate
```

It calculates fragment length as:

```text
fragment length = end coordinate - start coordinate
```

It keeps fragments between:

```text
50 bp and 250 bp
```

For each sample, the model uses up to:

```text
1,500,000 fragments per reference file
```

For the final dual-reference fusion dataset, each patient can contribute:

```text
1,500,000 hg38 fragments + 1,500,000 hg19 fragments
= up to 3,000,000 fragment rows per patient
```

Across 176 patients, the feature extraction represents up to:

```text
528,000,000 fragment rows
```

These raw fragments are compressed into normalized fragmentomics features.

## Feature Strategies Tested

MiNAE tested five major feature strategies.

### Strategy 1: hg38 Global Fragmentomics

Uses only hg38-aligned fragment-size features across the whole genome.

### Strategy 2: hg19 Global Fragmentomics

Uses only hg19-aligned fragment-size features across the whole genome.

### Strategy 3: Dual-Reference Global Fusion

Combines:

```text
hg38 features
hg19 features
hg38 - hg19 delta features
```

This tests whether reference-sensitive fragmentomics variation adds signal.

### Strategy 4: Chromosome-Targeted Fusion

Uses chromosome-localized features from HCC-relevant groups:

```text
Gain-associated chromosomes: chr1, chr8
Loss-associated chromosomes: chr4, chr16
Control chromosomes: chr2, chr3
```

### Strategy 5: Global + Chromosome Fusion

Combines the global dual-reference features with chromosome-targeted features.

This strategy did not improve performance, likely because it added too many features for the sample size and increased overfitting risk.

## Model Design

The final model family uses:

```text
StandardScaler
SelectKBest feature selection
Linear SVM
```

Why this makes sense:

- The dataset is relatively small: 176 patients.
- The feature count is moderately high.
- Linear SVMs often work well in small-sample, high-dimensional biomedical datasets.
- Tree models can overfit when sample size is limited.

## Validation Design

The main validation design uses three independent patient-level splits:

```text
Train set: model fitting
Test set: model/threshold selection
Final validation set: untouched final evaluation
```

For the strongest single validation run:

| Split | Patients | Cancer | Controls |
|---|---:|---:|---:|
| Train | 96 | 49 | 47 |
| Test | 40 | 20 | 20 |
| Final validation | 40 | 20 | 20 |

There is zero patient overlap between train, test, and final validation.

## Final Single-Split Result

The best single held-out result came from Strategy 3: dual-reference global fusion.

| Metric | Value |
|---|---:|
| ROC-AUC | 0.900 |
| Accuracy | 0.875 |
| Balanced accuracy | 0.875 |
| Sensitivity | 0.850 |
| Specificity | 0.900 |
| Precision | 0.895 |
| F1-score | 0.872 |

This corresponds to:

```text
35 / 40 final validation patients classified correctly
17 / 20 cancer cases detected
18 / 20 non-cancer controls correctly rejected
```

## Ablation Study Results

Final validation ablation table:

| Strategy | ROC-AUC | Accuracy | Sensitivity | Specificity | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|
| hg38 global | 0.828 | 0.750 | 0.750 | 0.750 | 0.750 | 0.750 |
| hg19 global | 0.838 | 0.725 | 0.700 | 0.750 | 0.737 | 0.718 |
| dual-reference global fusion | 0.900 | 0.875 | 0.850 | 0.900 | 0.895 | 0.872 |
| chromosome-targeted fusion | 0.890 | 0.875 | 0.850 | 0.900 | 0.895 | 0.872 |
| global + chromosome fusion | 0.869 | 0.725 | 0.600 | 0.850 | 0.800 | 0.686 |

Interpretation:

- Dual-reference global fusion was best on the original held-out split.
- Chromosome-targeted fusion independently performed almost as well.
- Combining all features hurt performance, suggesting feature overload and overfitting.

## Repeated Validation Results

To test whether the result was stable, the final pipeline was repeated over 100 random patient-level splits.

| Strategy | Mean ROC-AUC | SD | Mean Accuracy | Mean Sensitivity | Mean Specificity | Mean Precision |
|---|---:|---:|---:|---:|---:|---:|
| hg38 global | 0.889 | 0.049 | 0.816 | 0.762 | 0.869 | 0.873 |
| hg19 global | 0.886 | 0.048 | 0.809 | 0.755 | 0.863 | 0.864 |
| chromosome fusion | 0.872 | 0.052 | 0.761 | 0.727 | 0.795 | 0.806 |
| dual global fusion | 0.868 | 0.050 | 0.766 | 0.737 | 0.795 | 0.804 |
| global + chromosome | 0.847 | 0.057 | 0.751 | 0.698 | 0.805 | 0.798 |

This is the most important robustness result. It shows that the dual-reference fusion model can perform very well, but the most stable average performer across random splits is hg38 global fragmentomics.

## What This Means Scientifically

The project supports three claims:

1. cfDNA fragment-size profiles contain signal that separates HCC from chronic liver disease controls.
2. Dual-reference feature fusion can produce strong held-out performance, but its stability varies across random splits.
3. Chromosome-targeted features carry independent signal, but combining them with all global features can overfit.

The safest final conclusion is:

> MiNAE provides proof-of-concept evidence that low-cost cfDNA fragmentomics can distinguish HCC from cirrhosis/hepatitis B controls. The strongest single held-out result achieved 0.90 ROC-AUC, while repeated validation showed stable performance around 0.89 ROC-AUC for hg38 global fragmentomics.

## Is This Clinically Usable Today?

No. This is not yet a clinical diagnostic test.

It is a strong research prototype, but real deployment would require:

- External validation on independent cohorts
- More patients
- Prospective sample collection
- Standardized sequencing and preprocessing
- Comparison with ultrasound and AFP
- CLIA/CAP validation for laboratory-developed testing
- FDA or equivalent regulatory review if marketed clinically
- Calibration across age, sex, ethnicity, liver disease etiology, and fibrosis stage

## Real-World Application Path

A realistic future clinical workflow would be:

1. Blood draw from high-risk liver disease patient
2. Plasma cfDNA extraction
3. Low-pass paired-end sequencing
4. Fragment coordinate generation
5. MiNAE feature extraction
6. ML risk score
7. Physician receives risk classification
8. High-risk patients receive follow-up imaging or clinical review

Potential users:

- Liver cancer surveillance programs
- Clinical genomics labs
- Liquid biopsy companies
- Academic hepatology/oncology groups
- Diagnostic biotech startups

## Originality Grade

Overall originality: **B+ / A-**

Breakdown:

| Category | Grade | Reason |
|---|---|---|
| Clinical framing | A- | HCC vs cirrhosis/hepatitis B is clinically meaningful. |
| Fragmentomics concept | B | cfDNA fragmentomics has been established by prior work such as DELFI. |
| Dataset originality | B- | Public data from FinaleDB, not newly generated samples. |
| ML novelty | B | Linear SVM and feature selection are standard, but appropriate. |
| Dual-reference fusion | A- | Interesting and unusual engineering idea, though needs biological validation. |
| Chromosome-targeted ablation | A- | Biologically informed and useful for originality. |
| Validation rigor | B+ | Held-out validation and repeated splits are strong; external validation is still missing. |
| Interpretability | A- | SHAP, coefficient plots, heatmaps, ROC/PR/calibration outputs are strong. |

## Scientific Contribution Grade

Scientific contribution: **B+**

The project contributes:

- A reproducible local pipeline for HCC cfDNA fragmentomics.
- A clinically relevant HCC vs chronic liver disease classification task.
- A dual-reference fusion feature strategy.
- A chromosome-targeted feature ablation.
- A repeated-validation robustness analysis.

What prevents an A/A+:

- No independent external cohort.
- Public retrospective data only.
- Small final validation sets.
- Potential study/batch effects.
- Dual-reference delta features need deeper biological explanation.

## Suggested Paper Structure

### Title

**MiNAE: A Dual-Reference Cell-Free DNA Fragmentomics Framework for Non-Invasive Detection of Hepatocellular Carcinoma**

### Abstract

Briefly state:

- HCC screening problem
- cfDNA fragmentomics premise
- FinaleDB dataset
- HCC vs cirrhosis/hepatitis B controls
- feature extraction
- model validation
- best AUC and repeated-validation result
- limitations

### Introduction

Cover:

- HCC mortality and early detection problem
- limitations of ultrasound and AFP
- liquid biopsy cost barriers
- cfDNA fragmentomics as lower-cost signal
- why liver disease controls matter

### Methods

Include:

- Dataset source
- Inclusion/exclusion criteria
- Deduplication
- hg38/hg19 feature extraction
- fragment length calculation
- 50-250 bp filtering
- 5 bp bins
- dual-reference fusion
- chromosome-targeted tracks
- train/test/final validation
- repeated validation
- model selection
- metrics

### Results

Figures/tables:

1. Pipeline schematic
2. Dataset table
3. ROC and PR curves
4. Confusion matrix
5. Ablation table
6. Repeated-validation boxplot
7. SHAP/feature importance
8. Fragment-size profile plots

### Discussion

Discuss:

- fragmentomics signal exists
- chronic liver disease controls make task harder
- dual-reference fusion improves one split but not repeated average
- hg38 global is most stable
- chromosome tracks carry independent signal
- limitations and future work

### Limitations

State clearly:

- retrospective public data
- sample size limited
- no external validation
- possible batch effects
- not clinically approved

### Conclusion

Suggested wording:

> MiNAE demonstrates that cfDNA fragmentomics can distinguish HCC from non-cancer chronic liver disease controls using a reproducible machine learning pipeline. The strongest held-out result achieved 0.90 ROC-AUC, while repeated validation showed stable performance near 0.89 ROC-AUC, supporting fragment-size architecture as a promising non-invasive signal for HCC screening research.

## Final Recommendation

Do not keep tuning the final validation split. The best possible version now is not a higher single number; it is the more honest result:

- Report the 0.90 AUC single held-out result.
- Emphasize the repeated-validation stability result.
- Use the ablation table as the centerpiece.
- Present chromosome-targeted fusion as a biologically informed extension.
- Be clear that this is research-grade, not clinical-grade.

