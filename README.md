# MiNAE

**Mutation-independent Nucleosome Architecture Evaluation (MiNAE)** is a cfDNA fragmentomics research project for evaluating hepatocellular carcinoma (HCC) detection from public fragment-coordinate data.

MiNAE does not use mutation calls or raw DNA sequence letters. It calculates fragment length from genomic coordinates, summarizes fragment-size architecture into patient-level feature vectors, and evaluates whether those features distinguish HCC from chronic liver disease controls such as cirrhosis and hepatitis B.

## Current Paper Title

**Mutation-independent Nucleosome Architecture Evaluation (MiNAE): Patient-Level Validation of Public cfDNA Fragmentomics for Hepatocellular Carcinoma Detection**

## Repository Contents

- `minae_dual_reference_split.py`  
  Builds hg19 and hg38 global feature matrices and patient-level train/test/final validation splits.

- `minae_dual_reference_fusion.py`  
  Builds the dual-reference fusion model using hg38 features, hg19 features, and reference-delta features.

- `minae_ablation_chromosome.py`  
  Runs the feature-strategy ablation study comparing global, dual-reference, chromosome-targeted, and combined feature sets.

- `minae_repeated_validation.py`  
  Runs repeated patient-level validation to estimate split stability.

- `minae_statistical_audit.py`  
  Runs permutation testing, final-validation error analysis, and technical negative-control checks.

- `minae_external_validation_audit.py`  
  Runs external fragmentomics replication and benchmark analyses using public Annapragada and Foda data resources.

- `*_outputs/`  
  Processed feature matrices, metrics, predictions, plots, and audit outputs.

- `external_data/`  
  Small external public supplementary data used for the external audit. Large raw sequencing or fragment-coordinate files are not stored in this repository.

- `MiNAE_FINAL_OVERLEAF_SUBMISSION/`  
  Final Overleaf manuscript source, bibliography, and manuscript figures.

- `MiNAE_FINAL_OVERLEAF_SUBMISSION.zip`  
  Upload-ready Overleaf package.

## Data Note

The raw FinaleDB fragment-coordinate downloads are not committed because they are very large. They are stored locally outside this repository in the user's Downloads folder. The repository keeps the scripts, processed feature matrices, final metrics, plots, and manuscript files needed to preserve and reproduce the analysis workflow.

## Main Result Summary

On the locked final validation split, the dual-reference linear SVM model achieved:

- ROC-AUC: 0.900
- Accuracy: 0.875
- Sensitivity: 0.850
- Specificity: 0.900
- Precision: 0.895

Across 100 repeated held-out validations, the simpler hg38 global fragmentomics model was most stable:

- Mean ROC-AUC: 0.889
- Mean accuracy: 0.816

The external audit found that fragmentomics retained signal after technical adjustment, but also showed substantial technical confounding in public cfDNA datasets. MiNAE should therefore be interpreted as a retrospective validation and audit framework, not as a clinically validated diagnostic.

## Reproducibility Warning

This project uses public retrospective data. Results should not be interpreted as clinical performance estimates without prospective validation, independent HCC fragment-length cohorts, study-level holdout, and explicit batch-effect correction.
