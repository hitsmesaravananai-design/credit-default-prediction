# Group Project Part 1 – Predicting Credit Card Default Risk

Python data analytics pipeline for the UCI **Default of Credit Card Clients** dataset
(30,000 clients, 23 predictors, binary target = default on next month's payment).

## Contents

| Path | Purpose |
|---|---|
| `credit_default_analysis.py` | Full pipeline: load → explore → preprocess → model → evaluate → fairness audit |
| `build_report.js` | Generates the Word report from `outputs/results.json` and `figures/` |
| `GroupProject_Part1_Report.docx` | The 3.5-page deliverable (Sections 1–4 + references) |
| `data/` | Raw UCI file (`default of credit card clients.xls`) and the original zip |
| `figures/` | fig1 class balance · fig2 distributions · fig3 default by pay status · fig4 correlation · fig5 ROC · fig6 confusion matrices · fig7 feature importance |
| `outputs/` | `cleaned_data.csv`, `summary_stats.csv`, `model_metrics.csv`, `feature_importance.csv`, `fairness_by_group.csv`, `results.json` |

## Reproduce

```bash
# from the parent "Project 2" folder, using its virtual environment
../.venv/bin/python credit_default_analysis.py
node build_report.js
```

Requirements: Python 3.9+, pandas, numpy, scikit-learn, matplotlib, seaborn, xlrd; Node.js with the `docx` package (for the report only).

## Results (test set, n = 6,000)

| Model | CV ROC-AUC | Test ROC-AUC | Accuracy | Recall (default) | F1 |
|---|---|---|---|---|---|
| Majority baseline | – | 0.500 | 77.9% | 0.000 | 0.000 |
| Logistic Regression | 0.760 | 0.747 | 74.5% | 0.601 | 0.510 |
| Random Forest | 0.785 | 0.777 | 77.1% | 0.601 | 0.538 |

## Before submitting

Replace the `[Member 1]` … `[Member 4]` placeholders in `build_report.js` (title block and Section 4) with real names, then re-run `node build_report.js`.
