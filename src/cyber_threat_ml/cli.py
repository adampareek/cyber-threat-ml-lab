from __future__ import annotations

import argparse

from cyber_threat_ml.data import generate_malware_dataset, generate_network_dataset, generate_phishing_dataset
from cyber_threat_ml.evaluate import evaluate_model
from cyber_threat_ml.models import (
    build_malware_pipeline,
    build_network_pipeline,
    build_phishing_pipeline,
    prediction_frame,
    split_features_labels,
)


def run_malware_demo():
    df = generate_malware_dataset()
    train_x, test_x, train_y, test_y = split_features_labels(df)
    model = build_malware_pipeline(train_x)
    model.fit(train_x, train_y)
    return evaluate_model(model, test_x, test_y), prediction_frame(model, test_x)


def run_network_demo():
    df = generate_network_dataset()
    train_x, test_x, train_y, test_y = split_features_labels(df)
    model = build_network_pipeline(train_x)
    model.fit(train_x, train_y)
    return evaluate_model(model, test_x, test_y), prediction_frame(model, test_x)


def run_phishing_demo():
    df = generate_phishing_dataset()
    train_x, test_x, train_y, test_y = split_features_labels(df)
    train_urls = train_x["url"].fillna("").astype(str)
    test_urls = test_x["url"].fillna("").astype(str)
    model = build_phishing_pipeline()
    model.fit(train_urls, train_y)
    return evaluate_model(model, test_urls, test_y), prediction_frame(model, test_urls)


def _print_metrics(name: str, metrics: dict[str, float]):
    print(f"\n=== {name} ===")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run cybersecurity ML demos.")
    parser.add_argument(
        "--task",
        choices=["malware", "network", "phishing", "all"],
        default="all",
        help="Detection pipeline to run.",
    )
    args = parser.parse_args()

    if args.task in ("malware", "all"):
        metrics, _ = run_malware_demo()
        _print_metrics("Malware File/Header Detector", metrics)

    if args.task in ("network", "all"):
        metrics, _ = run_network_demo()
        _print_metrics("Network Intrusion Detector", metrics)

    if args.task in ("phishing", "all"):
        metrics, _ = run_phishing_demo()
        _print_metrics("Phishing URL Detector", metrics)


if __name__ == "__main__":
    main()
