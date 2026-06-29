# MiNAE Paper and Research Grade

## Overall Grade

Research project: 8.0 / 10

Conference-paper readiness for BIBM main track: 7.0 / 10

Conference-paper readiness for a student, workshop, poster, or special-session track: 8.0 / 10

## Research Strengths

- The project asks a clinically meaningful question: HCC versus chronic liver disease controls, not cancer versus healthy controls.
- The data processing pipeline is reproducible from public FinaleDB fragment-coordinate files.
- Patient-level splitting prevents the most obvious hg19/hg38 leakage problem.
- The paper includes ablation, repeated validation, permutation testing, and a technical negative control.
- The interpretation is honest: MiNAE is presented as a retrospective computational validation framework, not a clinical diagnostic.

## Research Weaknesses

- There is no external independent validation cohort.
- The sample size is modest: 176 unique patients.
- The final validation set is small: 40 patients.
- Technical negative-control performance is high enough to raise batch-effect concerns.
- Clinical metadata such as tumor stage, AFP, ultrasound findings, cirrhosis severity, and treatment history are not available.
- The chromosome-targeted analysis is biologically motivated but exploratory.

## Novelty Score

Novelty: 7.0 / 10

The broad idea of cfDNA fragmentomics for HCC has been done before. MiNAE is novel mainly as an open, public-data, patient-level validation and audit framework. Its originality is strongest in the combination of public FinaleDB processing, dual-reference leakage control, repeated validation, ablation, and technical negative-control analysis.

## Scientific Rigor Score

Rigor: 7.5 / 10

The validation is much stronger than a basic student ML project because it includes leakage prevention, locked validation, repeated validation, and negative controls. The reason it is not higher is that the study still lacks external validation and has possible residual technical confounding.

## Paper Readiness Score

Current manuscript: 7.0 / 10

The manuscript is now structurally credible for a conference submission. The strongest sections are Methods, Validation, Results, and Limitations. The paper still needs a final Overleaf compile check to ensure the added table does not break spacing or exceed the page limit.

## Acceptance Outlook

BIBM main-track full paper: possible but competitive.

Estimated chance after polishing: roughly 15-25%, depending on reviewer fit and whether BIBM has many stronger clinical-validation papers in the same track.

Student/workshop/poster/special-session track: substantially stronger fit.

Estimated chance: roughly 35-55%, assuming formatting is clean and claims remain honest.

## Best Submission Framing

"MiNAE is a reproducible public-data validation framework for evaluating cfDNA fragmentomics in hepatocellular carcinoma detection under patient-level splitting, ablation, repeated validation, and technical-confounding audit."

## Claims To Avoid

- "MiNAE is clinically validated."
- "MiNAE is ready for hospital deployment."
- "MiNAE proves fragmentomics detects liver cancer."
- "MiNAE is the first cfDNA fragmentomics method for HCC."

## Claims To Use

- "MiNAE supports measurable HCC-associated signal in public cfDNA fragment-coordinate data."
- "MiNAE demonstrates the importance of patient-level validation and technical negative controls in public cfDNA machine learning."
- "MiNAE provides a reproducible benchmark for HCC-vs-chronic-liver-disease fragmentomics."

## Revisions Made

- Added a study-design safeguards table.
- Strengthened the novelty boundary versus previous HCC fragmentomics studies.
- Expanded the clinical rationale for cirrhosis and hepatitis B controls.
- Clarified leakage prevention and threshold selection.
- Added reproducibility and availability language.
- Added explicit split-sensitivity interpretation.
- Preserved the technical-confounding caveat rather than hiding it.
