"""Global configuration, feature schema, attack taxonomies, and pipeline hyperparameters.

This module consolidates all operational parameters for the Network Anomaly Detection
pipeline, maintaining strict adherence to Hack The Box Applied AI InfoSec (Module 292)
standards and MITRE ATT&CK tactical classifications.
"""

from pathlib import Path
from typing import Dict, List

# Base filesystem directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# Dataset acquisition configuration
DATASET_URL = (
    "https://cdn.services-k8s.prod.aws.htb.systems/content/modules/292/KDD_dataset.zip"
)
DEFAULT_DATA_FILE = DATA_DIR / "KDD+.txt"
DEFAULT_MODEL_FILE = MODELS_DIR / "network_anomaly_detection_model.joblib"
DEFAULT_PREPROCESSOR_FILE = MODELS_DIR / "preprocessor_metadata.joblib"

# HTB Evaluation API endpoint
DEFAULT_API_URL = "http://localhost:8001/api/upload"

# Complete NSL-KDD 43-column schema
COLUMNS: List[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "attack",
    "level",
]

# Categorical features for one-hot encoding
FEATURES_TO_ENCODE: List[str] = ["protocol_type", "service"]

# 34 Continuous and statistical network flow attributes
NUMERIC_FEATURES: List[str] = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# Attack categories mapping based on MITRE ATT&CK & HTB Module 292
ATTACK_CATEGORIES: Dict[str, List[str]] = {
    "dos": [
        "apache2",
        "back",
        "land",
        "neptune",
        "mailbomb",
        "pod",
        "processtable",
        "smurf",
        "teardrop",
        "udpstorm",
        "worm",
    ],
    "probe": [
        "ipsweep",
        "mscan",
        "nmap",
        "portsweep",
        "saint",
        "satan",
    ],
    "privilege": [
        "buffer_overflow",
        "loadmdoule",  # Preserved as spelled in NSL-KDD benchmark dataset
        "perl",
        "ps",
        "rootkit",
        "sqlattack",
        "xterm",
    ],
    "access": [
        "ftp_write",
        "guess_passwd",
        "http_tunnel",
        "imap",
        "multihop",
        "named",
        "phf",
        "sendmail",
        "snmpgetattack",
        "snmpguess",
        "spy",
        "warezclient",
        "warezmaster",
        "xclock",
        "xsnoop",
    ],
}

# Numerical target mapping for multi-class classification
CLASS_MAPPING: Dict[str, int] = {
    "normal": 0,
    "dos": 1,
    "probe": 2,
    "privilege": 3,
    "access": 4,
}

CLASS_LABELS: List[str] = ["Normal", "DoS", "Probe", "Privilege", "Access"]

# Data split parameters
RANDOM_STATE: int = 1337
TEST_SIZE: float = 0.2
VAL_SIZE: float = 0.3

# RandomForestClassifier hyperparameter profile
RF_PARAMS: Dict = {
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": 0,
}

# Logging configuration format
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
