from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.metrics import f1_score


class Evaluator:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def compute_metrics(
        self, targets: np.ndarray, preds: np.ndarray
    ) -> Dict[str, float]:
        """
        Computes F1 micro and macro scores.
        """
        f1_micro = f1_score(targets, preds, average="micro", zero_division=0)
        f1_macro = f1_score(targets, preds, average="macro", zero_division=0)

        return {"f1_micro": f1_micro, "f1_macro": f1_macro}

    def plot_confusion_matrix(
        self,
        targets: np.ndarray,
        preds: np.ndarray,
        save_path: str = "confusion_matrix.png",
    ):
        """
        Plots a confusion matrix (co-occurrence for multi-label).
        """
        logger.info("Generating confusion matrix...")

        # Initialize matrix (NumClasses x NumClasses)
        cm = np.zeros((self.num_classes, self.num_classes))

        # Iterate over samples
        for i in range(targets.shape[0]):
            true_indices = np.where(targets[i] == 1)[0]
            pred_indices = np.where(preds[i] == 1)[0]

            # For every true label, see what was predicted
            for t in true_indices:
                for p in pred_indices:
                    cm[t, p] += 1

        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=False, fmt="g", cmap="Blues")
        plt.xlabel("Predicted Label Index")
        plt.ylabel("True Label Index")
        plt.title("Multi-Label Confusion Matrix (Co-occurrence)")
        output_path = "artifacts/img/confusion_matrix.png"
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Confusion matrix saved to {output_path}")
