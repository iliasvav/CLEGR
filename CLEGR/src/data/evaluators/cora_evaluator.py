from transformers import AutoTokenizer
from src.data.evaluators.evaluator import DatasetEvaluator
from collections import Counter
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
)

class CoraEvaluator(DatasetEvaluator):
    def __init__(self):
        super().__init__()
        self.class_names = [
            "theory", "reinforcement learning", "genetic algorithms",
            "neural networks", "probabilistic methods", "case based",
            "rule learning"
        ]
        self.class_to_label = {c: i for i, c in enumerate(self.class_names)}
        self.class_to_label["None"] = len(self.class_names)

    def compute_generation_diagnostics(self, tokenizer=None):
        """
        Analyze raw free-generation behavior on Cora.

        Strict validity requires the complete stripped output to exactly equal
        one of the seven allowed class names.

        This is deliberately different from match_prediction(), which retains
        the legacy startswith() behavior.
        """
        total = len(self.raw_records)

        if total == 0:
            return {}

        valid_count = 0
        strict_correct = 0
        valid_correct = 0

        malformed_outputs = []
        token_lengths = []

        strict_pred_counts = Counter()
        valid_by_gt = Counter()
        total_by_gt = Counter()

        strict_gts = []
        strict_preds = []

        invalid_label = "INVALID"

        for raw_output, gt in self.raw_records:
            raw_output = str(raw_output)
            gt = str(gt).strip()
            stripped = raw_output.strip()

            total_by_gt[gt] += 1

            if tokenizer is not None:
                token_count = len(
                    tokenizer(
                        raw_output,
                        add_special_tokens=False,
                    ).input_ids
                )
                token_lengths.append(token_count)

            is_valid = stripped in self.class_names

            if is_valid:
                valid_count += 1
                valid_by_gt[gt] += 1
                strict_pred = stripped

                if strict_pred == gt:
                    strict_correct += 1
                    valid_correct += 1
            else:
                strict_pred = invalid_label
                malformed_outputs.append(stripped)

            strict_pred_counts[strict_pred] += 1
            strict_gts.append(gt)
            strict_preds.append(strict_pred)

        malformed_count = total - valid_count

        strict_labels = self.class_names + [invalid_label]

        precision, recall, f1, support = precision_recall_fscore_support(
            strict_gts,
            strict_preds,
            labels=strict_labels,
            zero_division=0,
        )

        per_class = {}
        for label, p, r, f, s in zip(
            strict_labels,
            precision,
            recall,
            f1,
            support,
        ):
            per_class[label] = {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }

        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            strict_gts,
            strict_preds,
            labels=self.class_names,
            average="macro",
            zero_division=0,
        )

        weighted_p, weighted_r, weighted_f1, _ = (
            precision_recall_fscore_support(
                strict_gts,
                strict_preds,
                labels=self.class_names,
                average="weighted",
                zero_division=0,
            )
        )

        cm = confusion_matrix(
            strict_gts,
            strict_preds,
            labels=strict_labels,
        )

        per_class_valid_rate = {}
        for class_name in self.class_names:
            class_total = total_by_gt[class_name]

            per_class_valid_rate[class_name] = (
                valid_by_gt[class_name] / class_total
                if class_total > 0
                else 0.0
            )

        malformed_counter = Counter(malformed_outputs)

        result = {
            "strict_accuracy": strict_correct / total,
            "valid_rate": valid_count / total,
            "malformed_rate": malformed_count / total,
            "conditional_valid_accuracy": (
                valid_correct / valid_count
                if valid_count > 0
                else 0.0
            ),

            "strict_macro_precision": float(macro_p),
            "strict_macro_recall": float(macro_r),
            "strict_macro_f1": float(macro_f1),

            "strict_weighted_precision": float(weighted_p),
            "strict_weighted_recall": float(weighted_r),
            "strict_weighted_f1": float(weighted_f1),

            "num_samples": total,
            "num_valid": valid_count,
            "num_malformed": malformed_count,

            "prediction_distribution": {
                label: int(strict_pred_counts.get(label, 0))
                for label in strict_labels
            },

            "per_class": per_class,
            "per_class_valid_rate": per_class_valid_rate,

            "confusion_matrix_labels": strict_labels,
            "confusion_matrix": cm.tolist(),

            "top_malformed_outputs": [
                {
                    "output": output,
                    "count": count,
                }
                for output, count in malformed_counter.most_common(20)
            ],
        }

        if token_lengths:
            result["generation_length"] = {
                "mean": float(np.mean(token_lengths)),
                "median": float(np.median(token_lengths)),
                "min": int(np.min(token_lengths)),
                "max": int(np.max(token_lengths)),
                "distribution": {
                    str(length): int(count)
                    for length, count in sorted(
                        Counter(token_lengths).items()
                    )
                },
            }

        return result

    def match_prediction(self, prediction: str, ground_truths: list[str]) -> str:
        p = prediction.strip()
        for truth in ground_truths:
            if p.startswith(truth):
                return truth
        return "None"
    
    def max_class_toklen(self, llm_tokenizer: AutoTokenizer) -> int:
        # Tokenize each class name separately
        tokenized_class_names = [llm_tokenizer(name, add_special_tokens=False).input_ids for name in self.class_names]

        if self._max_class_toklen is None:
            self._max_class_toklen = max(len(class_name_toks) for class_name_toks in tokenized_class_names)

            # Print all class names with their token length
            for class_name, class_name_toks in zip(self.class_names, tokenized_class_names):
                print(f"{class_name}: {len(class_name_toks)}")
                

        return self._max_class_toklen
    
    def __call__(self, data_obj: str, llm_output: str) -> int:
        """
        Process one example:
          pred_str: raw LLM output (string)
          data_obj:   the pyg data object
        Returns the predicted label index.
        """
        gt_str = data_obj.label
        if isinstance(gt_str, list):
            gt_str = gt_str[0]
            
        self.add_raw(llm_output, gt_str)
        
        matched = self.match_prediction(llm_output, self.class_names)
        p_lbl = self.class_to_label[matched]
        g_lbl = self.class_to_label[gt_str]

        # register for later metric computation
        self.add(p_lbl, g_lbl)
        return p_lbl
