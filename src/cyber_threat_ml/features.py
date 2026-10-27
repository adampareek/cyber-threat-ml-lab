from __future__ import annotations

import math
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """Convert raw URL strings into numeric phishing-detection features."""

    suspicious_terms = (
        "login",
        "signin",
        "verify",
        "secure",
        "update",
        "confirm",
        "wallet",
        "bank",
        "invoice",
        "password",
    )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        urls = pd.Series(X).fillna("").astype(str)
        rows = [self._features_for_url(url) for url in urls]
        return pd.DataFrame(rows)

    def _features_for_url(self, url: str) -> dict[str, float]:
        parsed = urlparse(url if "://" in url else "http://" + url)
        host = parsed.netloc.lower()
        path = parsed.path.lower()
        query = parsed.query.lower()
        full = url.lower()

        return {
            "url_length": float(len(url)),
            "host_length": float(len(host)),
            "path_length": float(len(path)),
            "query_length": float(len(query)),
            "dot_count": float(url.count(".")),
            "dash_count": float(url.count("-")),
            "at_count": float(url.count("@")),
            "slash_count": float(url.count("/")),
            "digit_count": float(sum(ch.isdigit() for ch in url)),
            "special_count": float(sum(not ch.isalnum() for ch in url)),
            "uses_https": float(parsed.scheme == "https"),
            "has_ip_host": float(self._host_looks_like_ip(host)),
            "suspicious_token_count": float(sum(term in full for term in self.suspicious_terms)),
            "entropy": self._shannon_entropy(full),
        }

    @staticmethod
    def _host_looks_like_ip(host: str) -> bool:
        return bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host))

    @staticmethod
    def _shannon_entropy(text: str) -> float:
        if not text:
            return 0.0
        counts = pd.Series(list(text)).value_counts(normalize=True)
        return float(-sum(p * math.log2(p) for p in counts))


def add_network_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple flow-behavior features to a network dataframe."""
    result = df.copy()

    result["bytes_total"] = result["bytes_in"] + result["bytes_out"]
    result["packets_total"] = result["packets_in"] + result["packets_out"]
    result["bytes_ratio_out_in"] = result["bytes_out"] / (result["bytes_in"] + 1)
    result["packets_ratio_out_in"] = result["packets_out"] / (result["packets_in"] + 1)
    result["bytes_per_packet"] = result["bytes_total"] / (result["packets_total"] + 1)
    result["short_flow"] = (result["duration"] < 0.5).astype(int)
    result["http_activity"] = (result["http_depth"] > 0).astype(int)
    result["ftp_activity"] = (result["ftp_cmd_count"] > 0).astype(int)
    result["tcp_reset_ratio"] = result["tcp_rst_count"] / (result["tcp_syn_count"] + 1)

    return result


def add_malware_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived PE/header risk features."""
    result = df.copy()

    result["imports_per_section"] = result["imports_count"] / (result["section_count"] + 1)
    result["suspicious_import_ratio"] = result["suspicious_imports"] / (result["imports_count"] + 1)
    result["unsigned_high_entropy"] = ((result["has_signature"] == 0) & (result["entropy"] > 6.5)).astype(int)
    result["packed_or_high_entropy"] = ((result["packer_score"] > 0.5) | (result["entropy"] > 7.0)).astype(int)
    result["section_exec_write_gap"] = result["executable_sections"] - result["writable_sections"]

    return result
