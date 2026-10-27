import pandas as pd

from cyber_threat_ml.features import URLFeatureExtractor, add_malware_features, add_network_features


def test_url_feature_extractor_returns_expected_columns():
    extractor = URLFeatureExtractor()
    features = extractor.transform(pd.Series(["https://example.com/login", "http://10.0.0.1/verify?id=123"]))

    assert len(features) == 2
    assert "url_length" in features.columns
    assert "suspicious_token_count" in features.columns
    assert "has_ip_host" in features.columns
    assert features.loc[1, "has_ip_host"] == 1.0


def test_network_feature_engineering_adds_columns():
    df = pd.DataFrame(
        {
            "bytes_in": [10],
            "bytes_out": [20],
            "packets_in": [2],
            "packets_out": [4],
            "duration": [0.1],
            "http_depth": [1],
            "ftp_cmd_count": [0],
            "tcp_rst_count": [1],
            "tcp_syn_count": [2],
        }
    )
    result = add_network_features(df)

    assert "bytes_total" in result.columns
    assert "bytes_ratio_out_in" in result.columns
    assert "short_flow" in result.columns


def test_malware_feature_engineering_adds_columns():
    df = pd.DataFrame(
        {
            "imports_count": [100],
            "section_count": [5],
            "suspicious_imports": [4],
            "has_signature": [0],
            "entropy": [7.2],
            "packer_score": [0.7],
            "executable_sections": [3],
            "writable_sections": [2],
        }
    )
    result = add_malware_features(df)

    assert "imports_per_section" in result.columns
    assert "unsigned_high_entropy" in result.columns
    assert "packed_or_high_entropy" in result.columns
