# IEEE BIBM 2026 Submission Checklist for MiNAE

## Venue Requirements

- Venue: 2026 IEEE International Conference on Bioinformatics and Biomedicine.
- Location/date: Dallas, TX, USA, Dec. 1-4, 2026.
- Full paper deadline: July 5, 2026.
- Notification: Sept. 25, 2026.
- Camera-ready deadline: Oct. 25, 2026.
- Format: IEEE two-column conference format.
- Length: up to 8 pages.
- Review mode: double blind.
- Submission file: PDF or PostScript.
- Submission system: CyberChair link from the BIBM 2026 website.

## What This Folder Contains

- `MiNAE_BIBM2026_double_blind.tex`: main double-blind IEEE manuscript source.
- `references.bib`: BibTeX references.
- `figures/`: figures copied or generated from the MiNAE pipeline.

## How To Build The Paper In Overleaf

1. Open Overleaf.
2. Create a new blank project.
3. Upload everything in this folder:
   - `MiNAE_BIBM2026_double_blind.tex`
   - `references.bib`
   - the full `figures/` folder
4. Rename `MiNAE_BIBM2026_double_blind.tex` to `main.tex` inside Overleaf, or set it as the main document.
5. Use `pdfLaTeX` as the compiler.
6. Recompile. If references show as question marks, run recompile again.
7. Check the PDF length. It must be 8 pages or fewer unless BIBM later publishes a different rule.

## Double-Blind Checks

Before submitting:

- Do not include your name, school, affiliation, email, mentor, or acknowledgments.
- Do not include a personal GitHub link unless it is anonymized.
- Do not mention "my project," "our school," or anything that identifies you.
- Inspect the PDF metadata if possible and remove author metadata.
- Make sure figure file names and captions do not identify you.
- Keep the statement "Acknowledgments are omitted for double-blind review."

## Scientific Framing

Use this framing:

"MiNAE is a reproducible public-data fragmentomics benchmark that evaluates whether cfDNA fragment-size architecture distinguishes HCC from chronic liver disease controls."

Avoid this framing:

"MiNAE is a clinically validated liver cancer diagnostic test."

## Strengths To Emphasize

- Uses disease controls instead of only healthy controls.
- Performs patient-level splitting to avoid leakage.
- Tests multiple feature strategies through ablation.
- Runs 100 repeated held-out validations.
- Includes permutation testing.
- Includes a technical negative control and reports the caveat honestly.

## Main Weaknesses To Be Honest About

- Public retrospective data only.
- No external independent cohort.
- Small final validation set.
- Missing clinical metadata such as AFP, ultrasound, tumor stage, and cirrhosis severity.
- Technical negative control suggests possible residual confounding.

## Best Submission Category

The paper fits best under:

- Next Generation Sequencing and High-throughput Methods
- Cross-Cutting Computational Methods and Bioinformatics of Disease
- Data Mining, Machine Learning, and Artificial Intelligence
- Biomarker Discovery
- Precision Medicine

## Final Pre-Submission To-Do

- Re-run all scripts once before final submission.
- Confirm every table value still matches the generated CSV files.
- Reduce text if the paper exceeds 8 pages.
- Consider adding an anonymized supplementary ZIP with scripts and processed feature tables if the submission system allows it.
- After acceptance, replace the anonymous author block with the correct author information and add acknowledgments.
