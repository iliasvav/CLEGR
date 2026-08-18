from sklearn.metrics import accuracy_score
import csv
from absl import logging


class DatasetEvaluator:
    """
    Base class: just accumulates preds & gts, and knows how to compute a default accuracy.
    Subclasses implement __call__ freely, and use self.add(pred, gt) to register each example.
    """

    def __init__(self):
        self.reset()
        self.raw_records = []
        self._max_class_toklen = None

    def reset(self):
        self.preds = []
        self.gts = []

    def add_raw(self, pred, gt):
        "register the raw llm outputs"
        self.raw_records.append([pred, gt])

    def add(self, pred, gt):
        """Register one example (pred, gt)."""
        self.preds.append(pred)
        self.gts.append(gt)
        
    def max_class_toklen(self, llm_tokenizer=None) -> int:
        return 32

    def compute_metrics(self):
        """By default: simple accuracy."""
        return {
            "accuracy": accuracy_score(self.gts, self.preds),
        }

    def log_raw(self, path="logs/output.csv"):
        """
        Log the raw LLM output and the ground truth to a CSV file.
        """
        # ensure the logs directory exists
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Prediction", "Ground Truth"])
            for pred, gt in self.raw_records:
                writer.writerow([pred, gt])
        logging.info("Raw outputs logged to %s", path)
