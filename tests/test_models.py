from cyber_threat_ml.cli import run_malware_demo, run_network_demo, run_phishing_demo


def test_malware_demo_outputs_probabilities():
    metrics, predictions = run_malware_demo()

    assert set(predictions.columns) == {"index", "prob_label_1"}
    assert predictions["prob_label_1"].between(0, 1).all()
    assert metrics["roc_auc"] >= 0.80


def test_network_demo_outputs_probabilities():
    metrics, predictions = run_network_demo()

    assert set(predictions.columns) == {"index", "prob_label_1"}
    assert predictions["prob_label_1"].between(0, 1).all()
    assert metrics["roc_auc"] >= 0.75


def test_phishing_demo_outputs_probabilities():
    metrics, predictions = run_phishing_demo()

    assert set(predictions.columns) == {"index", "prob_label_1"}
    assert predictions["prob_label_1"].between(0, 1).all()
    assert metrics["roc_auc"] >= 0.85
