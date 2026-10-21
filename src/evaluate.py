"""Model evaluation, classification telemetry, and confusion matrix visualization."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
# Use headless backend for non-interactive / production / server environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config.settings import CLASS_LABELS, REPORTS_DIR

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Computes production-grade evaluation metrics and generates visualization artifacts."""

    def __init__(self, reports_dir: Optional[Union[str, Path]] = None) -> None:
        self.reports_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        y_true: Union[np.ndarray, List[int]],
        y_pred: Union[np.ndarray, List[int]],
        dataset_name: str = "Test Set",
        save_plot: bool = True,
        plot_filename: Optional[str] = None,
    ) -> Dict[str, Union[float, Dict]]:
        """Compute comprehensive performance metrics for multi-class classification.

        Args:
            y_true: Ground truth target labels.
            y_pred: Predicted class labels.
            dataset_name: Identifier for dataset partition (e.g. 'Validation Set', 'Test Set').
            save_plot: Whether to render and save confusion matrix heatmap to disk.
            plot_filename: Optional custom file name for the heatmap plot.

        Returns:
            Dict containing metric summaries, confusion matrix, and classification report dict.
        """
        logger.info("Computing evaluation metrics for: %s", dataset_name)

        # Standard weighted metrics
        accuracy = float(accuracy_score(y_true, y_pred))
        precision_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        recall_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # Macro metrics (crucial in security contexts with rare attack categories like U2R)
        precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        labels = list(range(len(CLASS_LABELS)))

        # Text and dictionary classification reports
        clf_report_text = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=CLASS_LABELS,
            zero_division=0,
        )
        clf_report_dict = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        )

        # Confusion Matrix
        conf_mat = confusion_matrix(y_true, y_pred, labels=labels)

        results = {
            "dataset_name": dataset_name,
            "accuracy": accuracy,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "confusion_matrix": conf_mat.tolist(),
            "classification_report": clf_report_dict,
        }

        # Print structured console output
        self._print_evaluation_summary(results, clf_report_text)

        # Generate and save heatmap artifact
        if save_plot:
            filename = plot_filename or f"confusion_matrix_{dataset_name.lower().replace(' ', '_')}.png"
            plot_path = self.reports_dir / filename
            self.plot_confusion_matrix(conf_mat, dataset_name, plot_path)
            results["plot_path"] = str(plot_path)

        # Save metrics JSON artifact
        json_path = self.reports_dir / f"metrics_{dataset_name.lower().replace(' ', '_')}.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info("Evaluation metrics saved to: %s", json_path)

        return results

    def plot_confusion_matrix(
        self,
        conf_matrix: np.ndarray,
        dataset_name: str,
        output_path: Path,
    ) -> None:
        """Render and save a high-resolution confusion matrix heatmap."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS,
            cbar=True,
        )
        plt.title(f"Network Anomaly Detection - {dataset_name}", fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Tactical Class", fontsize=11)
        plt.ylabel("Actual Tactical Class", fontsize=11)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info("Confusion matrix plot saved to: %s", output_path)

    @staticmethod
    def _print_evaluation_summary(metrics: Dict, clf_report_text: str) -> None:
        """Format and print evaluation summary to stdout."""
        print(f"\n========================================================")
        print(f"       EVALUATION REPORT: {metrics['dataset_name'].upper()}")
        print(f"========================================================")
        print(f" Accuracy:           {metrics['accuracy']:.4f}")
        print(f" Precision (Weighted):{metrics['precision_weighted']:.4f}")
        print(f" Recall (Weighted):   {metrics['recall_weighted']:.4f}")
        print(f" F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f" F1-Score (Macro):    {metrics['f1_macro']:.4f}")
        print(f"--------------------------------------------------------")
        print(" Detailed Class Breakdown:\n")
        print(clf_report_text)
        print(f"========================================================\n")
