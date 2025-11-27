import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import (auc, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_curve)


class Evaluator:
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = (
            class_names if class_names else [str(i) for i in range(num_classes)]
        )

        # Ensure artifacts directories exist
        os.makedirs("artifacts/img", exist_ok=True)
        os.makedirs("artifacts/csv", exist_ok=True)

    def _get_filename(
        self, base_name: str, tags: Optional[Dict[str, str]], ext: str
    ) -> str:
        """Helper to construct filename with tags."""
        folder = "img" if ext == "png" else ext

        if not tags:
            return f"artifacts/{folder}/{base_name}.{ext}"

        # Construct suffix from tags (e.g. _bert-tiny_f66bce0c)
        # We prioritize regex_hash if present, then text_model_name
        parts = [base_name]

        if "text_model_name" in tags:
            # Clean model name (prajjwal1/bert-tiny -> prajjwal1_bert-tiny)
            clean_name = tags["text_model_name"].replace("/", "_")
            parts.append(clean_name)

        if "regex_hash" in tags:
            parts.append(tags["regex_hash"])

        return f"artifacts/{folder}/{'_'.join(parts)}.{ext}"

    def compute_metrics(
        self, targets: np.ndarray, preds: np.ndarray
    ) -> Dict[str, float]:
        """
        Computes F1, Precision, Recall (micro and macro).
        """
        metrics = {}
        for avg in ["micro", "macro"]:
            metrics[f"f1_{avg}"] = f1_score(
                targets, preds, average=avg, zero_division=0
            )
            metrics[f"precision_{avg}"] = precision_score(
                targets, preds, average=avg, zero_division=0
            )
            metrics[f"recall_{avg}"] = recall_score(
                targets, preds, average=avg, zero_division=0
            )

        return metrics

    def save_per_class_metrics(
        self,
        targets: np.ndarray,
        preds: np.ndarray,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Calculates precision/recall per class and saves to CSV.
        """
        logger.info("Calculating per-class metrics...")

        precision = precision_score(targets, preds, average=None, zero_division=0)
        recall = recall_score(targets, preds, average=None, zero_division=0)
        f1 = f1_score(targets, preds, average=None, zero_division=0)

        # Support (number of true instances)
        support = targets.sum(axis=0)

        data = []
        for i, name in enumerate(self.class_names):
            data.append(
                {
                    "class_name": name,
                    "precision": precision[i],
                    "recall": recall[i],
                    "f1": f1[i],
                    "support": support[i],
                }
            )

        df = pd.DataFrame(data)
        # Sort by support (descending) to see most frequent classes first
        df = df.sort_values("support", ascending=False)

        output_path = self._get_filename("per_class_metrics", tags, "csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Per-class metrics saved to {output_path}")

    def plot_roc_curve(
        self,
        targets: np.ndarray,
        probs: np.ndarray,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Plots Micro-Averaged ROC Curve.
        """
        logger.info("Generating ROC curve...")

        # Flatten for micro-average
        fpr, tpr, _ = roc_curve(targets.ravel(), probs.ravel())
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(
            fpr,
            tpr,
            label=f"Micro-average ROC curve (area = {roc_auc:0.2f})",
            color="darkorange",
            lw=2,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (Micro-Average)")
        plt.legend(loc="lower right")

        output_path = self._get_filename("roc_curve", tags, "png")
        plt.savefig(output_path)
        plt.close()
        logger.info(f"ROC curve saved to {output_path}")

    def plot_pr_curve(
        self,
        targets: np.ndarray,
        probs: np.ndarray,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Plots Micro-Averaged Precision-Recall Curve.
        """
        logger.info("Generating Precision-Recall curve...")

        precision, recall, _ = precision_recall_curve(targets.ravel(), probs.ravel())
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(10, 8))
        plt.plot(
            recall,
            precision,
            label=f"Micro-average PR curve (area = {pr_auc:0.2f})",
            color="blue",
            lw=2,
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (Micro-Average)")
        plt.legend(loc="lower left")

        output_path = self._get_filename("pr_curve", tags, "png")
        plt.savefig(output_path)
        plt.close()
        logger.info(f"PR curve saved to {output_path}")

    def plot_global_confusion_matrix(
        self,
        targets: np.ndarray,
        preds: np.ndarray,
        tags: Optional[Dict[str, str]] = None,
    ):
        """
        Plots a global 2x2 confusion matrix (TP/FP/FN/TN) across all classes.
        """
        logger.info("Generating global confusion matrix...")

        # Flatten to treat as binary classification problem
        cm = confusion_matrix(targets.ravel(), preds.ravel())

        # cm is [[TN, FP], [FN, TP]]

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Negative", "Predicted Positive"],
            yticklabels=["True Negative", "True Positive"],
        )
        plt.title("Global Confusion Matrix (All Classes Flattened)")

        output_path = self._get_filename("global_confusion_matrix", tags, "png")
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Global confusion matrix saved to {output_path}")
