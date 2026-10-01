from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from collections import Counter
import csv
from absl import logging


class DatasetEvaluator:
    """
    Base evaluator.

    Accumulates parsed predictions and ground truths and computes generic
    classification metrics. Dataset-specific evaluators remain responsible
    for parsing raw model outputs.
    """

    def __init__(self):
        self.raw_records = []
        self._max_class_toklen = None
        self.reset()

    def reset(self):
        self.preds = []
        self.gts = []
        self.raw_records = []

    def add_raw(self, pred, gt):
        """Register raw LLM output and ground truth."""
        self.raw_records.append([pred, gt])

    def add(self, pred, gt):
        """Register one parsed prediction / ground-truth pair."""
        self.preds.append(pred)
        self.gts.append(gt)

    def max_class_toklen(self, llm_tokenizer=None) -> int:
        return 32

    def compute_metrics(self):
        """
        Generic scalar classification metrics.

        These remain scalar so they can safely be sent to Lightning/W&B.
        Detailed per-class diagnostics are handled separately.
        """
        if not self.gts:
            return {}

        precision_macro, recall_macro, f1_macro, _ = (
            precision_recall_fscore_support(
                self.gts,
                self.preds,
                average="macro",
                zero_division=0,
            )
        )

        precision_weighted, recall_weighted, f1_weighted, _ = (
            precision_recall_fscore_support(
                self.gts,
                self.preds,
                average="weighted",
                zero_division=0,
            )
        )

        return {
            "accuracy": accuracy_score(self.gts, self.preds),
            "macro_precision": precision_macro,
            "macro_recall": recall_macro,
            "macro_f1": f1_macro,
            "weighted_precision": precision_weighted,
            "weighted_recall": recall_weighted,
            "weighted_f1": f1_weighted,
        }

    def compute_classification_diagnostics(self):
        """
        Detailed classification diagnostics.

        Returns JSON-serializable data:
        - per-class precision / recall / F1 / support
        - prediction distribution
        - ground-truth distribution
        - confusion matrix

        If the evaluator defines class_names and class_to_label, human-readable
        class names are used.
        """
        if not self.gts:
            return {}

        labels = sorted(set(self.gts) | set(self.preds))

        label_to_name = {}
        if hasattr(self, "class_names") and hasattr(self, "class_to_label"):
            for name, label_idx in self.class_to_label.items():
                label_to_name[label_idx] = name

        def name_for(label):
            return label_to_name.get(label, str(label))

        precision, recall, f1, support = precision_recall_fscore_support(
            self.gts,
            self.preds,
            labels=labels,
            zero_division=0,
        )

        per_class = {}
        for label, p, r, f, s in zip(
            labels, precision, recall, f1, support
        ):
            per_class[name_for(label)] = {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }

        pred_counts = Counter(self.preds)
        gt_counts = Counter(self.gts)

        cm = confusion_matrix(
            self.gts,
            self.preds,
            labels=labels,
        )

        return {
            "labels": [name_for(label) for label in labels],
            "per_class": per_class,
            "prediction_distribution": {
                name_for(label): int(pred_counts.get(label, 0))
                for label in labels
            },
            "ground_truth_distribution": {
                name_for(label): int(gt_counts.get(label, 0))
                for label in labels
            },
            "confusion_matrix": cm.tolist(),
        }

    def log_raw(self, path="logs/output.csv"):
        """Log raw LLM output and ground truth to CSV."""
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Prediction", "Ground Truth"])

            for pred, gt in self.raw_records:
                writer.writerow([pred, gt])

        logging.info("Raw outputs logged to %s", path)