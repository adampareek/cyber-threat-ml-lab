from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    """Train/test split plus metadata used by demos and tests."""

    train_features: pd.DataFrame
    test_features: pd.DataFrame
    train_labels: pd.Series
    test_labels: pd.Series


def generate_malware_dataset(n_rows: int = 1600, random_state: int = 42) -> pd.DataFrame:
    """Create a synthetic PE/header-style malware classification dataset.

    The dataset intentionally uses interpretable file attributes instead of
    private samples. The labels are generated from a noisy risk score so the
    task is suitable for testing the workflow without distributing malware.
    """
    rng = np.random.default_rng(random_state)

    entropy = rng.normal(5.8, 1.2, n_rows).clip(1.0, 8.0)
    file_size_kb = rng.lognormal(mean=6.1, sigma=0.75, size=n_rows)
    section_count = rng.poisson(5, n_rows) + 1
    imports_count = rng.poisson(85, n_rows) + 5
    suspicious_imports = rng.poisson(1.8, n_rows)
    has_signature = rng.binomial(1, 0.62, n_rows)
    packer_score = rng.beta(1.2, 3.0, n_rows)
    writable_sections = rng.poisson(1.0, n_rows)
    executable_sections = rng.poisson(1.4, n_rows) + 1
    overlay_size_kb = rng.lognormal(mean=3.5, sigma=1.1, size=n_rows)

    risk_score = (
        1.15 * (entropy > 6.8)
        + 1.00 * (suspicious_imports >= 3)
        + 0.85 * (has_signature == 0)
        + 1.25 * (packer_score > 0.55)
        + 0.55 * (writable_sections >= 2)
        + 0.45 * (overlay_size_kb > np.percentile(overlay_size_kb, 75))
        + rng.normal(0, 0.45, n_rows)
    )
    label = (risk_score > 2.10).astype(int)

    return pd.DataFrame(
        {
            "entropy": entropy,
            "file_size_kb": file_size_kb,
            "section_count": section_count,
            "imports_count": imports_count,
            "suspicious_imports": suspicious_imports,
            "has_signature": has_signature,
            "packer_score": packer_score,
            "writable_sections": writable_sections,
            "executable_sections": executable_sections,
            "overlay_size_kb": overlay_size_kb,
            "label": label,
        }
    )


def generate_network_dataset(n_rows: int = 1800, random_state: int = 43) -> pd.DataFrame:
    """Create a synthetic network-flow intrusion dataset."""
    rng = np.random.default_rng(random_state)
    protocols = rng.choice(["tcp", "udp", "icmp"], size=n_rows, p=[0.70, 0.24, 0.06])
    services = rng.choice(["http", "dns", "ftp", "ssh", "smtp", "other"], size=n_rows)

    duration = rng.exponential(scale=2.8, size=n_rows)
    bytes_in = rng.lognormal(mean=7.0, sigma=1.0, size=n_rows)
    bytes_out = rng.lognormal(mean=6.7, sigma=1.05, size=n_rows)
    packets_in = rng.poisson(35, n_rows) + 1
    packets_out = rng.poisson(31, n_rows) + 1
    failed_logins = rng.poisson(0.28, n_rows)
    tcp_syn_count = rng.poisson(2.0, n_rows)
    tcp_rst_count = rng.poisson(0.55, n_rows)
    unique_dst_ports = rng.poisson(1.4, n_rows) + 1
    http_depth = rng.poisson(1.2, n_rows)
    ftp_cmd_count = rng.poisson(0.18, n_rows)

    risk_score = (
        0.90 * (failed_logins >= 2)
        + 0.80 * (unique_dst_ports >= 4)
        + 0.75 * (tcp_syn_count >= 5)
        + 0.70 * (tcp_rst_count >= 2)
        + 0.55 * (services == "ssh")
        + 0.50 * (services == "ftp")
        + 0.45 * (bytes_out > np.percentile(bytes_out, 80))
        + 0.40 * ((bytes_out / (bytes_in + 1)) > 3.0)
        + 0.35 * (duration < 0.25)
        + rng.normal(0, 0.50, n_rows)
    )
    label = (risk_score > 1.10).astype(int)

    return pd.DataFrame(
        {
            "protocol": protocols,
            "service": services,
            "duration": duration,
            "bytes_in": bytes_in,
            "bytes_out": bytes_out,
            "packets_in": packets_in,
            "packets_out": packets_out,
            "failed_logins": failed_logins,
            "tcp_syn_count": tcp_syn_count,
            "tcp_rst_count": tcp_rst_count,
            "unique_dst_ports": unique_dst_ports,
            "http_depth": http_depth,
            "ftp_cmd_count": ftp_cmd_count,
            "label": label,
        }
    )


def _random_domain(rng: random.Random) -> str:
    names = ["example", "accounts", "portal", "secure", "login", "docs", "cloud", "pay"]
    tlds = ["com", "net", "org", "io", "co"]
    return f"{rng.choice(names)}-{rng.randint(10, 999)}.{rng.choice(tlds)}"


def generate_phishing_dataset(n_rows: int = 1800, random_state: int = 44) -> pd.DataFrame:
    """Create a synthetic URL dataset for phishing detection."""
    rng = random.Random(random_state)
    rows: list[dict[str, object]] = []

    benign_paths = ["/home", "/products", "/account", "/help", "/docs", "/blog/security"]
    phishing_tokens = ["verify", "update", "secure", "wallet", "signin", "confirm", "invoice", "bank"]

    for _ in range(n_rows):
        is_phish = rng.random() < 0.45
        if is_phish:
            scheme = rng.choice(["http", "https"])
            host = rng.choice(
                [
                    f"{rng.randint(10, 250)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}",
                    f"{rng.choice(phishing_tokens)}-{_random_domain(rng)}",
                    f"{_random_domain(rng)}.{rng.choice(['login', 'verify', 'secure'])}.com",
                ]
            )
            token = rng.choice(phishing_tokens)
            path = f"/{token}/{rng.randint(100000, 999999)}/session-{rng.randint(100, 999)}"
            query = f"?user={rng.randint(10000, 99999)}&token={rng.randint(10**8, 10**9 - 1)}"
            url = f"{scheme}://{host}{path}{query}"
        else:
            scheme = "https"
            host = _random_domain(rng)
            path = rng.choice(benign_paths)
            if rng.random() < 0.25:
                path += f"/{rng.choice(['faq', 'contact', 'pricing', 'download'])}"
            url = f"{scheme}://www.{host}{path}"

        rows.append({"url": url, "label": int(is_phish)})

    return pd.DataFrame(rows)
