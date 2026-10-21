#!/usr/bin/env bash
# ==============================================================================
# Network Anomaly Detector - Pipeline Automation Script
# Hack The Box Applied AI in InfoSec / AI Red Teaming
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=========================================================="
echo " Starting Network Anomaly Detection Pipeline"
echo "=========================================================="

# Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "[-] Error: python3 is not installed or not in PATH."
    exit 1
fi

# Run the complete pipeline
python3 main.py --pipeline "$@"

echo "=========================================================="
echo " Pipeline execution completed successfully."
echo " Reports available in: $PROJECT_ROOT/reports"
echo " Models stored in:     $PROJECT_ROOT/models"
echo "=========================================================="
