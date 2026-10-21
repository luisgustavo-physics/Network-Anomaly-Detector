"""Production CLI entrypoint for the Network Anomaly Detection System.

Orchestrates data ingestion, feature preprocessing, model training, evaluation,
reporting, and deployment/upload to Hack The Box evaluation APIs.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Optional

from config.settings import (
    DEFAULT_API_URL,
    DEFAULT_DATA_FILE,
    DEFAULT_MODEL_FILE,
    DEFAULT_PREPROCESSOR_FILE,
    LOG_FORMAT,
    REPORTS_DIR,
)
from src.data_loader import KDDDataLoader
from src.evaluate import ModelEvaluator
from src.model import NetworkAnomalyModel
from src.predict import NetworkAnomalyPredictor
from src.preprocessor import KDDPreprocessor
from src.uploader import ModelUploader

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("NetworkAnomalyDetector")


def run_pipeline(
    data_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    preprocessor_path: Optional[Path] = None,
    reports_dir: Optional[Path] = None,
    api_url: Optional[str] = None,
    upload_model: bool = False,
    sample_fraction: Optional[float] = None,
) -> None:
    """Execute the end-to-end Machine Learning Intrusion Detection Pipeline."""
    logger.info("Initializing full Network Anomaly Detection pipeline...")

    # 1. Ingestion
    loader = KDDDataLoader()
    df = loader.load_data(file_path=data_path)

    if sample_fraction and 0.0 < sample_fraction < 1.0:
        logger.info("Subsampling dataset by factor %.2f for expedited run", sample_fraction)
        df = df.sample(frac=sample_fraction, random_state=1337).reset_index(drop=True)

    # 2. Preprocessing & Partitioning
    preprocessor = KDDPreprocessor()
    X, multi_y, binary_y = preprocessor.prepare_dataset(df)
    splits = preprocessor.split_data(X, multi_y)

    # 3. Model Training
    model = NetworkAnomalyModel()
    model.train(splits["train_X"], splits["train_y"])

    # 4. Validation Evaluation
    evaluator = ModelEvaluator(reports_dir=reports_dir)
    val_preds = model.predict(splits["val_X"])
    val_metrics = evaluator.evaluate(
        splits["val_y"],
        val_preds,
        dataset_name="Validation Set",
        plot_filename="confusion_matrix_validation.png",
    )

    # 5. Test Evaluation
    test_preds = model.predict(splits["test_X"])
    test_metrics = evaluator.evaluate(
        splits["test_y"],
        test_preds,
        dataset_name="Test Set",
        plot_filename="confusion_matrix_test.png",
    )

    # 6. Feature Sensitivity & Explainability
    importances = model.get_feature_importances(
        feature_names=preprocessor.feature_columns,
        top_n=10,
    )
    print("--------------------------------------------------------")
    print(" TOP 10 INTRUSION DETECTION TELEMETRY FEATURES:")
    print("--------------------------------------------------------")
    for rank, (feat, score) in enumerate(importances, start=1):
        print(f" {rank:2d}. {feat:<35} (Score: {score:.4f})")
    print("--------------------------------------------------------\n")

    # 7. Persistence
    saved_model_path = model.save(filepath=model_path)
    saved_prep_path = preprocessor.save(filepath=preprocessor_path)
    logger.info("Artifacts saved: Model -> %s | Preprocessor -> %s", saved_model_path, saved_prep_path)

    # 8. Upload to Evaluation API (Hack The Box Lab Endpoint)
    if upload_model:
        uploader = ModelUploader(api_url=api_url)
        upload_result = uploader.upload(saved_model_path)
        print("\n========================================================")
        print("           MODEL ARTIFACT UPLOAD STATUS")
        print("========================================================")
        print(f" Target Endpoint: {uploader.api_url}")
        print(f" Success:         {upload_result.get('success')}")
        print(f" HTTP Code:       {upload_result.get('status_code')}")
        if upload_result.get("response"):
            print(f" API Response:\n{json.dumps(upload_result['response'], indent=2)}")
        if upload_result.get("error"):
            print(f" Note/Error:      {upload_result.get('error')}")
        print("========================================================\n")


def run_predict_sample(model_path: Optional[Path] = None) -> None:
    """Run demonstration inference on synthetic network traffic samples."""
    logger.info("Executing sample network telemetry inference demonstration...")
    predictor = NetworkAnomalyPredictor(model_path=model_path)

    synthetic_samples = [
        # Normal web browsing packet flow
        {
            "duration": 0,
            "protocol_type": "tcp",
            "service": "http",
            "src_bytes": 215,
            "dst_bytes": 45076,
            "count": 1,
            "srv_count": 1,
            "same_srv_rate": 1.0,
            "diff_srv_rate": 0.0,
            "dst_host_count": 9,
            "dst_host_srv_count": 9,
            "dst_host_same_srv_rate": 1.0,
        },
        # High-frequency SYN flood pattern (DoS - Neptune profile)
        {
            "duration": 0,
            "protocol_type": "tcp",
            "service": "private",
            "src_bytes": 0,
            "dst_bytes": 0,
            "count": 250,
            "srv_count": 12,
            "serror_rate": 1.0,
            "srv_serror_rate": 1.0,
            "same_srv_rate": 0.05,
            "diff_srv_rate": 0.07,
            "dst_host_count": 255,
            "dst_host_srv_count": 12,
            "dst_host_same_srv_rate": 0.05,
            "dst_host_serror_rate": 1.0,
            "dst_host_srv_serror_rate": 1.0,
        },
        # Port scanning flow (Probe - Nmap profile)
        {
            "duration": 0,
            "protocol_type": "tcp",
            "service": "other",
            "src_bytes": 0,
            "dst_bytes": 0,
            "count": 150,
            "srv_count": 1,
            "rerror_rate": 1.0,
            "srv_rerror_rate": 1.0,
            "same_srv_rate": 0.01,
            "diff_srv_rate": 0.99,
            "dst_host_count": 255,
            "dst_host_srv_count": 1,
            "dst_host_diff_srv_rate": 0.98,
        },
    ]

    results = predictor.predict(synthetic_samples)
    print("\n========================================================")
    print("          REAL-TIME TELEMETRY INFERENCE DEMO")
    print("========================================================")
    for res in results:
        status_flag = "[MALICIOUS]" if res["is_anomaly"] else "[BENIGN]   "
        print(f" Flow #{res['record_index'] + 1}: {status_flag} Predicted: {res['predicted_label']:<10} "
              f"| Confidence: {res['confidence'] * 100:.1f}%")
        print(f"   Distribution: {res['class_probabilities']}")
    print("========================================================\n")


def main() -> None:
    """CLI Argument parser and execution dispatch."""
    parser = argparse.ArgumentParser(
        description="Network Anomaly Detector: Production-ready ML Intrusion Detection System.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run the complete end-to-end pipeline: load, preprocess, train, evaluate, and save.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download and extract the NSL-KDD dataset without training.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload trained model artifact to evaluation endpoint.",
    )
    parser.add_argument(
        "--predict-sample",
        action="store_true",
        help="Run inference on representative sample network telemetry flows.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_FILE,
        help="Path to KDD+.txt dataset file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_FILE,
        help="Path to save or load the model .joblib file.",
    )
    parser.add_argument(
        "--preprocessor-path",
        type=Path,
        default=DEFAULT_PREPROCESSOR_FILE,
        help="Path to save or load preprocessor metadata.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory to save evaluation plots and metrics JSON.",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=DEFAULT_API_URL,
        help="URL of the target model upload endpoint.",
    )
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="Fraction of data to sample (e.g. 0.1 for rapid test runs).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging.",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Default action if no specific action specified
    if not (args.pipeline or args.download_only or args.upload or args.predict_sample):
        # Default to running the full pipeline
        args.pipeline = True

    try:
        if args.download_only:
            loader = KDDDataLoader()
            path = loader.download_and_extract()
            logger.info("Dataset verified at %s", path)

        if args.pipeline:
            run_pipeline(
                data_path=args.data_path,
                model_path=args.model_path,
                preprocessor_path=args.preprocessor_path,
                reports_dir=args.reports_dir,
                api_url=args.api_url,
                upload_model=args.upload,
                sample_fraction=args.sample_fraction,
            )

        elif args.upload and not args.pipeline:
            uploader = ModelUploader(api_url=args.api_url)
            uploader.upload(args.model_path)

        if args.predict_sample:
            run_predict_sample(model_path=args.model_path)

    except Exception as e:
        logger.exception("Fatal error during pipeline execution: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
