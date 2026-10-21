# 🛡️ Production Network Anomaly Detector (ML-NIDS)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Security Benchmark](https://img.shields.io/badge/Dataset-NSL--KDD-green.svg)](https://www.unb.ca/cic/datasets/nsl.html)
[![HTB Module](https://img.shields.io/badge/HackTheBox-Module%20292%20Applied%20AI%20in%20InfoSec-red.svg)](https://academy.hackthebox.com/)
[![MITRE ATLAS](https://img.shields.io/badge/Framework-MITRE%20ATLAS%20Aligned-yellow.svg)](https://atlas.mitre.org/)
[![Tests](https://img.shields.io/badge/Tests-10%2F10%20Passing-brightgreen.svg)](tests/)

---

## 📌 Executive Summary

This project transforms an exploratory Hack The Box (HTB) research notebook into a **hardened, modular, production-ready Machine Learning Network Intrusion Detection System (ML-NIDS)**. 

Modern Security Operations Centers (SOCs) face overwhelming alert fatigue and sophisticated network adversaries. Traditional signature-based detection mechanisms (e.g., Snort, Suricata rules) frequently fail against zero-day exploits, novel polymorphic payloads, and distributed multi-stage attacks. This system leverages statistical telemetry analysis and ensemble machine learning (`RandomForestClassifier`) to classify network traffic into benign baseline communication and four core adversary tactical categories (**Denial of Service**, **Network Reconnaissance / Probing**, **Privilege Escalation**, and **Unauthorized Remote Access**).

Refactored according to rigorous software engineering and MLOps standards, this codebase provides deterministic dataset management, schema validation, feature space alignment, headless metric generation, and an automated deployment client for evaluation microservices.

---

## 🎯 AI Red Teaming & Threat Surface Analysis

In an operational AI Red Teaming engagement, evaluating a machine learning model is not merely about achieving high test accuracy—it requires rigorous threat modeling against the **MITRE ATLAS (Adversarial Threat Landscape for Artificial-Intelligence Systems)** framework.

```
       ADVERSARIAL ATTACK SURFACES ON THE ML PIPELINE
       ==============================================

  +-----------------------+     +-----------------------+     +-----------------------+
  |  Evasion / Jitter     |     | Model Deserialization |     | Model Inversion /     |
  |  Perturbation Attacks |     | Code Execution (ACE)  |     | Extraction Attacks    |
  +-----------------------+     +-----------------------+     +-----------------------+
              |                             |                             |
              v                             v                             v
   [Network Flow Packets] ------> [Serialized .joblib] -------> [API / SOC Endpoint]
   (Manipulating count,            (Malicious __reduce__         (Boundary discovery
    duration, src_bytes)            payload execution)            via repeated probes)
```

### 1. Evasion via Adversarial Perturbation ([AML.T0015](https://atlas.mitre.org/techniques/AML.T0015))
* **Vulnerability:** Random Forest decision boundaries rely heavily on specific statistical thresholds (e.g., `src_bytes`, `count`, `same_srv_rate`).
* **Red Team Vector:** An adversary conducting network reconnaissance or DoS attacks can introduce artificial delays, packet padding, or connection fragmentation to perturb telemetry features across decision thresholds while maintaining malicious payload efficacy.
* **Defensive Mitigation:** Multi-feature correlation analysis, feature importance auditing, and adversarial data augmentation during training.

### 2. Model Serialization & Deserialization Vulnerabilities ([AML.T0010](https://atlas.mitre.org/techniques/AML.T0010))
* **Vulnerability:** Standard Python model persistence via `joblib` or `pickle` relies on object serialization protocols that can execute arbitrary Python bytecode upon deserialization.
* **Red Team Vector:** If the model upload endpoint (`http://localhost:8001/api/upload`) deserializes incoming `.joblib` files directly on the server without cryptographic signature verification or sandboxing, an attacker can embed an arbitrary code execution (ACE) payload via `__reduce__()`.
* **Defensive Mitigation:** Implementation of SHA-256 cryptographic artifact signing, migration toward non-executable formats (e.g., ONNX, Safetensors), and strict network-isolated upload sandboxing.

### 3. Model Extraction and Boundary Inversion ([AML.T0024](https://atlas.mitre.org/techniques/AML.T0024))
* **Vulnerability:** Exposed inference endpoints returning detailed prediction confidence vectors (`class_probabilities`).
* **Red Team Vector:** Querying the model with crafted synthetic flows allows an attacker to map decision boundaries, reverse-engineer feature weights, and identify low-confidence blind spots.
* **Defensive Mitigation:** Rate-limiting inference queries, returning discrete categorical labels rather than raw probability distributions to external callers, and logging anomaly detection telemetry for behavioral drift.

### 4. Data Poisoning & Label Flipping ([AML.T0018](https://atlas.mitre.org/techniques/AML.T0018))
* **Vulnerability:** Uncurated telemetry ingestion pipelines that retrain models on live network captures.
* **Red Team Vector:** Injecting slow-rate anomalous traffic labeled as benign to poison model decision trees over time.
* **Defensive Mitigation:** Strict schema validation (`KDDDataLoader.validate_schema`), statistical baseline profiling, and holdout benchmark validation sets.

---

## 🔬 Hack The Box (HTB) Methodologies Applied

This repository implements the complete methodology defined in **HTB Module 292: Applied AI in Information Security**, bringing defensive AI research to production standards.

### 1. Tactical Attack Categorization
Raw intrusion logs contain over 30 distinct attack signatures. In tactical defense and threat intelligence mapping, these are grouped into five actionable classes:

| Class ID | Tactical Category | MITRE ATT&CK Mapping | Representative Signatures in NSL-KDD |
| :---: | :--- | :--- | :--- |
| **0** | **Normal** | Baseline Traffic | Standard benign traffic |
| **1** | **Denial of Service (DoS)** | Impact (TA0040) | `neptune`, `smurf`, `back`, `teardrop`, `pod`, `apache2`, `mailbomb`, `processtable`, `udpstorm`, `worm` |
| **2** | **Probe (Reconnaissance)** | Reconnaissance (TA0043) | `ipsweep`, `nmap`, `portsweep`, `satan`, `mscan`, `saint` |
| **3** | **Privilege Escalation (U2R)** | Privilege Escalation (TA0004) | `buffer_overflow`, `rootkit`, `perl`, `ps`, `loadmdoule`, `sqlattack`, `xterm` |
| **4** | **Unauthorized Access (R2L)** | Initial Access (TA0001) | `guess_passwd`, `ftp_write`, `imap`, `phf`, `multihop`, `warezclient`, `spy`, `sendmail`, `snmpguess` |

### 2. Rigorous Dataset Hygiene & Split Strategy
To eliminate data leakage and ensure realistic validation:
1. **Initial Split:** 80% Development / 20% Holdout Test Partition (`random_state=1337`).
2. **Secondary Split:** Development partition split 70% Training / 30% Validation (`random_state=1337`).
3. **Imbalance Auditing:** Privilege Escalation represents <0.1% of all records. To evaluate true operational effectiveness, the system computes **Macro F1-score** alongside standard Weighted metrics to prevent majority-class bias from masking undetected rare attacks.

### 3. Automated Lab Integration & Endpoint Deployment
In the Hack The Box module, the final objective is deploying the trained anomaly detector to an automated verification API (`http://localhost:8001/api/upload`). The `ModelUploader` class provides an enterprise-grade client with exponential backoff, retry mechanisms, and detailed HTTP response telemetry.

---

## 🛠️ Technical Skills Demonstrated

* **Production MLOps & Software Architecture:** Transformed ad-hoc Jupyter Notebook code into a modular Python package adhering to Clean Architecture principles, complete with configuration management, robust logging, and CLI orchestration.
* **Cybersecurity Telemetry Engineering:** One-hot encoding of high-cardinality network protocols and services with persistent schema alignment, ensuring that unseen production flows cannot trigger dimensionality mismatch runtime errors.
* **Ensemble Learning Optimization:** Configured multi-threaded `RandomForestClassifier` achieving **99.49% overall accuracy** and sub-second classification across 30,000+ test flows.
* **Explainable AI (XAI) for Threat Intelligence:** Feature importance analysis pinpointing the primary statistical telemetry indicators of compromise (`src_bytes`, `dst_bytes`, `same_srv_rate`, `count`, `serror_rate`).
* **Automated CI/CD Quality Assurance:** Complete unit test suite (`pytest`) validating schema integrity, edge-case categorical alignments, attack mappings, and model serialization.

---

## 📂 Repository Architecture

```
Network-Anomaly-Detector/
├── .gitignore                      # Git exclusion rules (prevents tracking models/data)
├── requirements.txt                # Pinned production and testing dependencies
├── main.py                         # Unified CLI pipeline entrypoint
├── config/
│   ├── __init__.py                 # Configuration package interface
│   └── settings.py                 # Schema, taxonomies, hyperparameters, and path definitions
├── src/
│   ├── __init__.py                 # Core package exports
│   ├── data_loader.py              # Download, extraction, caching, and schema validation
│   ├── preprocessor.py             # Feature encoding, attack mapping, and split logic
│   ├── model.py                    # RandomForest wrapper, training, and persistence
│   ├── evaluate.py                 # Headless metrics computation and heatmap generation
│   ├── predict.py                  # Standalone inference engine for real-time traffic
│   └── uploader.py                 # Resilient API upload client with exponential backoff
├── tests/
│   ├── __init__.py                 # Test suite initializer
│   ├── test_data_loader.py         # Schema validation and empty file tests
│   ├── test_preprocessor.py        # Mapping, encoding alignment, and split tests
│   ├── test_model.py               # Model training, prediction, and serialization tests
│   └── test_uploader.py            # Upload client error handling and resilience tests
├── scripts/
│   └── run_pipeline.sh             # Production bash execution runner
├── models/                         # Serialized model & preprocessor artifacts (gitignored)
└── reports/                        # Evaluation heatmaps and metric JSONs (gitignored)
```

---

## 🚀 Installation & Quickstart

### 1. Prerequisites
Ensure Python 3.10+ is installed on your Linux / macOS / Windows system:
```bash
python3 --version
```

### 2. Environment Setup
```bash
# Clone repository
git clone https://github.com/your-username/Network-Anomaly-Detector.git
cd Network-Anomaly-Detector

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Automated Unit Tests
Verify all core components pass unit and regression tests:
```bash
python3 -m pytest tests/ -v
```

---

## 💻 CLI Usage Guide

The system is managed through `main.py`, providing flexible options for development, testing, and production workflows.

### Run Full End-to-End Pipeline
Executes dataset ingestion, feature preprocessing, training, validation/test evaluation, plot generation, and artifact serialization:
```bash
python3 main.py --pipeline
```

### Run Rapid Subsample Pipeline (For Development & Benchmarking)
Uses 10% of the dataset for rapid verification:
```bash
python3 main.py --pipeline --sample-fraction 0.1
```

### Run Real-Time Flow Inference Demo
Classifies simulated normal and malicious packet flows using the trained model:
```bash
python3 main.py --predict-sample
```

### Upload Trained Model to Evaluation API
Transmits the saved `.joblib` model artifact to the Hack The Box scoring endpoint:
```bash
python3 main.py --upload --api-url http://localhost:8001/api/upload
```

### Download Dataset Only
Downloads and verifies the NSL-KDD dataset without training:
```bash
python3 main.py --download-only
```

---

## 📊 Empirical Evaluation & Results

Trained on **83,169 samples** and evaluated against **29,704 holdout test samples**, the detector demonstrates production-grade intrusion detection capabilities:

### Test Set Performance Summary
* **Overall Test Accuracy:** `99.49%`
* **Weighted Precision:** `0.9947`
* **Weighted Recall:** `0.9949`
* **Weighted F1-Score:** `0.9947`
* **Macro F1-Score:** `0.8547`

### Per-Class Performance Breakdown

| Class Label | Precision | Recall | F1-Score | Support | Operational Risk Context |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Normal** | 0.99 | 1.00 | 1.00 | 15,402 | Low False Positive rate protects SOC operations from alert fatigue. |
| **DoS** | 1.00 | 1.00 | 1.00 | 10,721 | Perfect detection of SYN floods, UDP storms, and resource exhaustion. |
| **Probe** | 0.99 | 1.00 | 1.00 | 2,796 | High fidelity in detecting Nmap and port sweep reconnaissance. |
| **Privilege** | 0.62 | 0.24 | 0.34 | 21 | Highly imbalanced rare events (U2R exploits) requiring specialized heuristics. |
| **Access** | 0.96 | 0.92 | 0.94 | 764 | Robust defense against credential brute-forcing and unauthorized logins. |

### Top 10 Intrusion Telemetry Features

Feature sensitivity analysis reveals the primary network attributes informing the classifier's decisions:

```
  1. src_bytes                  (Score: 0.1302)  --> Outbound data volume anomaly
  2. dst_bytes                  (Score: 0.0778)  --> Inbound data transfer anomaly
  3. same_srv_rate              (Score: 0.0686)  --> Repeated connection target uniformity
  4. count                      (Score: 0.0619)  --> Connection frequency spike
  5. dst_host_same_srv_rate     (Score: 0.0531)  --> Host-level service concentration
  6. diff_srv_rate              (Score: 0.0485)  --> Multi-port probing diversity
  7. dst_host_diff_srv_rate     (Score: 0.0482)  --> Target host port scanning
  8. serror_rate                (Score: 0.0424)  --> SYN error rate (DoS signature)
  9. dst_host_srv_serror_rate   (Score: 0.0401)  --> Server-level SYN error rate
 10. dst_host_serror_rate       (Score: 0.0376)  --> Destination host SYN rejection
```

### Visual Artifacts
Upon pipeline completion, high-resolution confusion matrix heatmaps and machine-readable JSON metrics are automatically generated in the `reports/` directory:
* `reports/confusion_matrix_validation.png`
* `reports/confusion_matrix_test.png`
* `reports/metrics_test_set.json`

---

## 🔒 Security Best Practices & Deployment Recommendations

1. **Model Hardening:** In production environments, replace native `joblib` deserialization with ONNX runtime models or isolated worker containers to mitigate ACE risks.
2. **Adversarial Resilience:** Retrain periodically with adversarial examples generated via boundary attack simulations.
3. **Threshold Tuning:** For high-security environments, lower the classification threshold for the **Privilege** class to prioritize recall over precision when hunting for stealthy zero-day root access attempts.

---

## 📜 License
This project is open-source under the MIT License. Developed for defensive AI security research, cybersecurity education, and red teaming assessment workflows.
