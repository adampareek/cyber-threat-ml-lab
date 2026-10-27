# Cyber Threat ML Lab

An end-to-end defensive cybersecurity machine learning project for three detection tasks:

1. **Malware file/header classification** from structured file metadata.
2. **Network intrusion scoring** from flow-level network features.
3. **Phishing URL detection** using URL-only feature engineering.

The project uses reproducible synthetic datasets, reusable preprocessing pipelines, model training, evaluation, probability scoring, and tests.

> Data in this repository is synthetic. Performance on these datasets should not be interpreted as production detection accuracy.

## Why this project matters

Many cybersecurity tools assign risk scores to files, network flows, or URLs before an analyst reviews them. This project demonstrates the core workflow behind that type of system: feature engineering, preprocessing, model selection, probability scoring, and ROC AUC evaluation.

## Tech stack

- Python
- pandas
- NumPy
- scikit-learn
- pytest

## Repository layout

```text
cyber-threat-ml-lab/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── src/
│   └── cyber_threat_ml/
│       ├── __init__.py
│       ├── data.py
│       ├── features.py
│       ├── models.py
│       ├── evaluate.py
│       └── cli.py
├── scripts/
│   └── run_demo.py
├── tests/
│   ├── test_features.py
│   └── test_models.py
├── examples/
│   └── sample_predictions.csv
└── reports/
    └── model_results.md
```

## Quick start

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run the full demo:

```bash
python scripts/run_demo.py
```

Or run one task through the CLI:

```bash
python -m cyber_threat_ml.cli --task malware
python -m cyber_threat_ml.cli --task network
python -m cyber_threat_ml.cli --task phishing
python -m cyber_threat_ml.cli --task all
```

Run tests:

```bash
pytest -q
```

## What the demo produces

Each detector returns:

- Accuracy
- Precision
- Recall
- F1 score
- ROC AUC
- A prediction DataFrame with `index` and `prob_label_1`

Example output format:

```text
=== Malware File/Header Detector ===
ROC AUC: 0.9194
F1: 0.6364

=== Network Intrusion Detector ===
ROC AUC: 0.8857
F1: 0.5851

=== Phishing URL Detector ===
ROC AUC: 1.0000
F1: 1.0000
```

These scores are from the reproducible synthetic demo in `reports/model_results.md`. The perfect phishing result reflects the artificial dataset and does not establish real-world accuracy. Scores can vary slightly by environment; the models use fixed random seeds.

## Detection tasks

### Malware file/header detector

Uses structured features such as entropy, section count, imported functions, suspicious import count, signature status, packer score, and file size. The model is a tree-based classifier with median imputation for missing numeric values.

### Network intrusion detector

Uses synthetic flow-level features such as protocol, service, bytes, packets, flow duration, failed logins, TCP behavior, and HTTP/FTP indicators. The pipeline one-hot encodes categorical columns, imputes missing values, and trains a tree-based classifier.

### Phishing URL detector

Uses two feature sources:

1. URL statistics: URL length, digit count, number of dots, special character count, suspicious token count, IP-like host flag, entropy, and HTTPS flag.
2. Character-level TF-IDF n-grams from the raw URL string.

The model is a logistic regression classifier trained on a combined feature representation.

## Notes

The datasets generated here are synthetic and meant for reproducible demonstration. For production-grade research, replace the generators with licensed public datasets and document the source, schema, and preprocessing decisions.
