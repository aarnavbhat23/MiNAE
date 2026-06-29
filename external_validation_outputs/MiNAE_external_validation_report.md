# MiNAE External Validation and Replication Audit

## What Was Tested

The downloaded external data do not provide MiNAE-compatible HCC fragment-length profiles. Therefore, the analysis is reported honestly as an external fragmentomics replication/audit using Annapragada liver disease fragment-length profiles, plus a Foda et al. published-score HCC benchmark.

## Annapragada External Fragment-Length Replication

- Total samples with fragment-length profiles: 553
- Discovery/train samples: 362
- Validation/test samples: 191
- Validation class counts: {'Non-Cirrhotic Controls': 106, 'Advanced Fibrosis/Cirrhosis': 85}
- Permutation p-value for fragmentomics model: 0.0033

### Discovery-to-Validation Metrics

| Experiment | ROC_AUC | PR_AUC | Accuracy | Balanced_Accuracy | Sensitivity | Specificity | Precision | F1 | TN | FP | FN | TP | Permutation_P_Value_vs_Fragmentomics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Annapragada external Discovery->Validation fragmentomics | 0.891 | 0.894 | 0.822 | 0.813 | 0.729 | 0.896 | 0.849 | 0.785 | 95 | 11 | 23 | 62 | 0.003 |
| Annapragada technical covariates only | 0.998 | 0.997 | 0.921 | 0.913 | 0.835 | 0.991 | 0.986 | 0.904 | 105 | 1 | 14 | 71 |  |
| Annapragada residualized fragmentomics | 0.799 | 0.801 | 0.733 | 0.722 | 0.624 | 0.821 | 0.736 | 0.675 | 87 | 19 | 32 | 53 |  |

### Fragmentomics Bootstrap Confidence Intervals

| Metric | Mean | CI_2.5 | CI_97.5 |
| --- | --- | --- | --- |
| ROC_AUC | 0.891 | 0.842 | 0.935 |
| PR_AUC | 0.893 | 0.837 | 0.937 |
| Accuracy | 0.821 | 0.764 | 0.869 |
| Balanced_Accuracy | 0.812 | 0.755 | 0.865 |
| Sensitivity | 0.728 | 0.628 | 0.819 |
| Specificity | 0.896 | 0.837 | 0.950 |
| Precision | 0.848 | 0.757 | 0.926 |
| F1 | 0.782 | 0.701 | 0.847 |

### Source-Level Holdout Results

| Held_Out_Source | N | ROC_AUC | PR_AUC | Accuracy | Balanced_Accuracy | Sensitivity | Specificity | Precision | F1 | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| INCAP (Guatemala City, Guatemala) | 66 | 0.699 | 0.887 | 0.667 | 0.580 | 0.771 | 0.389 | 0.771 | 0.771 | 7 | 11 | 11 | 37 |

### Top Fragmentomic Features

| feature | coefficient | abs_coefficient |
| --- | --- | --- |
| p_len_137 | 0.7156 | 0.7156 |
| p_len_150 | -0.6947 | 0.6947 |
| p_len_138 | 0.6755 | 0.6755 |
| p_len_128 | -0.5860 | 0.5860 |
| p_len_110 | 0.5579 | 0.5579 |
| p_len_115 | -0.5424 | 0.5424 |
| p_len_111 | 0.5234 | 0.5234 |
| p_len_136 | 0.4971 | 0.4971 |
| p_len_141 | -0.4702 | 0.4702 |
| p_len_116 | -0.4618 | 0.4618 |
| p_len_108 | 0.4517 | 0.4517 |
| p_len_139 | 0.4486 | 0.4486 |

## Foda et al. HCC Benchmark Context

- Foda workbook samples: 501
- Patient-type counts: {'Non-cancer': 293, 'Cirrhosis': 78, 'HCC': 75, 'HCV': 29, 'HBV': 26}
These results use published DELFI/Foda scores from the supplementary workbook. They are not MiNAE predictions, but they contextualize the HCC task.

### Published-Score Metrics

| Benchmark | Score | N | N_HCC | N_Control | ROC_AUC | PR_AUC | Accuracy | Balanced_Accuracy | Sensitivity | Specificity | Precision | F1 | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HCC vs all non-HCC | DELFI-surveillance score | 208 | 75 | 133 | 0.900 | 0.850 | 0.851 | 0.837 | 0.787 | 0.887 | 0.797 | 0.792 | 118 | 15 | 16 | 59 |
| HCC vs cirrhosis/HBV/HCV | DELFI-surveillance score | 208 | 75 | 133 | 0.900 | 0.850 | 0.851 | 0.837 | 0.787 | 0.887 | 0.797 | 0.792 | 118 | 15 | 16 | 59 |
| HCC vs cirrhosis only | DELFI-surveillance score | 153 | 75 | 78 | 0.889 | 0.879 | 0.830 | 0.829 | 0.787 | 0.872 | 0.855 | 0.819 | 68 | 10 | 16 | 59 |

## Interpretation

This audit strengthens MiNAE by adding external fragmentomics replication, confidence intervals, technical-covariate comparison, residualized fragmentomics, and source-level holdout testing. It does not yet provide a full independent HCC validation for MiNAE because the downloaded external tables do not include HCC sample-level fragment-length histograms compatible with MiNAE feature extraction.