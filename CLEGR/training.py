import os
import torch
import torch.multiprocessing
import lightning as pl
from tqdm import tqdm
from absl import app, flags, logging
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    Callback,
)
from lightning.pytorch.strategies import DDPStrategy
import json
import statistics

# Import project modules
from src.data.evaluators import get_evaluator, DatasetEvaluator
from src.models import get_glm_lightning_module
from src.models.glm.base import BaseGLM
from src.utils import (
    load_dataset,
    split_dataset,
    create_dataloaders,
    initialize_model,
)

# LoRA imports
from peft import PeftModel

from torch_geometric.utils import to_undirected
from torch_geometric.loader import DataLoader

# Define command-line flags
FLAGS = flags.FLAGS

flags.DEFINE_string("method", "g-retriever", "Method to use for training.")
flags.DEFINE_string("root", "datasets", "Root directory for raw and processed data.")
flags.DEFINE_string("dataset", "clegr-reasoning", "The dataset to use.")
flags.DEFINE_integer("num_gpus", 1, "Number of GPUs for training.")
flags.DEFINE_integer("seed", 0, "The random seed.")
flags.DEFINE_integer("batch_size", 32, "Batch size for the training DataLoader.")
flags.DEFINE_float("lr", 1e-3, "Learning rate for the optimizer.")
flags.DEFINE_integer("epochs", 50, "Number of epochs for training with Lightning.")
flags.DEFINE_integer(
    "manual_epochs", 5, "Number of epochs for manual training (if not using Lightning)."
)
flags.DEFINE_bool(
    "use_lightning", True, "Whether to use PyTorch Lightning for training."
)
flags.DEFINE_integer("num_workers", 10, "Number of worker processes for DataLoader.")
flags.DEFINE_float("train_ratio", 0.6, "Proportion of the dataset to use for training.")
flags.DEFINE_float("val_ratio", 0.2, "Proportion of the dataset to use for validation.")
flags.DEFINE_string(
    "accelerator",
    "gpu",
    "Accelerator type for PyTorch Lightning (e.g., 'gpu' or 'cpu').",
)
flags.DEFINE_string("strategy", "ddp", "Training strategy (e.g., 'ddp', 'fsdp').")
flags.DEFINE_integer("gnn_hidden_channels", 64, "Number of hidden channels in the GNN.")
flags.DEFINE_integer("gnn_num_layers", 2, "Number of layers in the GNN.")
flags.DEFINE_integer("n_gnn_toks", 10, "Number of GNN tokens to pass to the LLM.")
flags.DEFINE_string("llm_model_name", "phi-3.5", "Name of the LLM model to use.")
flags.DEFINE_string(
    "llm_params_file",
    "src/llm_params.json",
    "Path to JSON file containing LLM parameters mapping.",
)
flags.DEFINE_string(
    "logging_level", "INFO", "Logging level: DEBUG, INFO, WARNING, ERROR, or CRITICAL."
)
flags.DEFINE_string("save_dir", "models", "Directory to save the model checkpoints.")
flags.DEFINE_bool("evaluate", False, "Whether to only evaluate the model.")
flags.DEFINE_integer("max_samples", -1, "Maximum number of samples to test on.")

flags.DEFINE_bool(
    "use_lora",
    False,
    "Whether to use LoRA for training the LLM. Requires peft library.",
)

# scheduler and gradient clipping flags
flags.DEFINE_integer("scheduler_step_size", 3, "Step size for StepLR scheduler.")
flags.DEFINE_float("scheduler_gamma", 0.1, "Decay factor for StepLR scheduler.")
flags.DEFINE_float("gradient_clip_val", 1.0, "Value for gradient clipping.")

# wandb-related flags
flags.DEFINE_bool("use_wandb", False, "Enable Weights & Biases logging.")
flags.DEFINE_string("wandb_project", "training-glms", "wandb project name.")
flags.DEFINE_string("wandb_entity", None, "wandb team or user name.")
flags.DEFINE_enum(
    "wandb_mode", "online", ["online", "offline", "disabled"], "wandb mode."
)

#black box flags
flags.DEFINE_enum("edge_drop_mode", "none", ["none", "random", "degree"], "Edge dropping strategy: random or degree-aware.")
flags.DEFINE_float("edge_drop_prob", 0.0, "Fraction of undirected edges to drop.")
flags.DEFINE_float("edge_drop_degree_power", 1.0, "Strength of degree bias for degree-aware edge dropping.")
flags.DEFINE_integer("edge_drop_seed", 42, "Random seed for edge dropping.")

flags.DEFINE_enum("node_feature_noise", "none", ["none", "shuffle", "gaussian", "zero"], "Node feature perturbation mode.",)
flags.DEFINE_float("node_feature_noise_std", 0.1, "Std for Gaussian node feature noise.")
flags.DEFINE_integer("node_feature_seed", 42, "Random seed for node feature perturbation.")

flags.DEFINE_float("adversarial_edge_add_prob", 0.0, "Number of adversarial edges to add as a fraction of the original undirected edge count.",)
flags.DEFINE_integer("adversarial_edge_seed", 42, "Random seed for adversarial edge addition",)

flags.DEFINE_enum("context_perturbation", "none", ["none", "shuffle", "cross_class_shuffle", "constant_full", "constant_title"], "Context perturbation mode.")
flags.DEFINE_integer("context_shuffle_seed", 42, "Random seed for shuffling context across samples.")

flags.DEFINE_bool("no_graph_token", False, "If True, do not provide GNN graph tokens to the LLM during inference")

#report accuracy after each epoch
flags.DEFINE_bool("eval_each_epoch", False, "Run full generative validation evaluation after each epoch.")
flags.DEFINE_string("eval_split", "test", "Which split to evaluate: train, val, test.")
flags.DEFINE_string("context_mode", "full", "Context ablation mode: full, no_title, no_abstract, empty.")

flags.DEFINE_enum("eval_mode", "generation", ["generation", "label_scoring"], "Evaluation mode.")

flags.DEFINE_bool("shuffle_graph_tokens", False, "Shuffle node-level GNN representations before the projector for diagnostic evaluation.",)
flags.DEFINE_integer("shuffle_graph_seed", 42, "Random seed used for graph-token shuffling.",)

def get_candidate_labels(dataset):
    labels = sorted(list({str(dataset[i].label).strip() for i in range(len(dataset))}))
    logging.info("Found %d candidate labels: %s", len(labels), labels)
    return labels

def evaluate_label_scoring(
    model: BaseGLM,
    loader,
    evaluator: DatasetEvaluator,
    candidate_labels,
    graph_data=None,
    output_path=None,
):
    model.eval()

    graph_data = (
        graph_data.to(model.llm.device)
        if graph_data
        else None
    )

    num_samples = (
        min(len(loader.dataset), FLAGS.max_samples)
        if FLAGS.max_samples > 0
        else len(loader.dataset)
    )

    logging.info(
        "Evaluating with label scoring on %d samples",
        num_samples,
    )

    score_records = []

    with torch.no_grad():
        for i, data in tqdm(
            enumerate(loader),
            desc="Label scoring",
            total=num_samples,
        ):
            if i >= num_samples:
                break

            data = data.to(model.llm.device)

            true_label = data.label

            if isinstance(true_label, list):
                gt_string = str(true_label[0]).strip()
            else:
                gt_string = str(true_label).strip()

            scores = {}

            for candidate in candidate_labels:
                data.label = [candidate]

                loss = model(
                    data,
                    graph_data=graph_data,
                )

                scores[candidate] = float(loss.item())

            data.label = true_label

            ranked = sorted(
                scores.items(),
                key=lambda x: x[1],
            )

            best_label, best_score = ranked[0]

            if len(ranked) > 1:
                second_label, second_score = ranked[1]
                score_margin = second_score - best_score
            else:
                second_label = None
                second_score = None
                score_margin = None

            evaluator(data, best_label)

            # Recover a useful sample identifier if available.
            sample_id = None

            if hasattr(data, "node_id"):
                sample_id = data.node_id
            elif hasattr(data, "id"):
                sample_id = data.id

            if torch.is_tensor(sample_id):
                if sample_id.numel() == 1:
                    sample_id = sample_id.item()
                else:
                    sample_id = sample_id.detach().cpu().tolist()

            score_records.append({
                "sample_index": i,
                "sample_id": sample_id,
                "ground_truth": gt_string,
                "prediction": best_label,
                "correct": best_label == gt_string,

                "best_score": best_score,
                "runner_up": second_label,
                "runner_up_score": second_score,
                "score_margin": score_margin,

                "candidate_scores": scores,
            })

    metrics = evaluator.compute_metrics()

    classification_diagnostics = (
        evaluator.compute_classification_diagnostics()
    )

    margins = [
        record["score_margin"]
        for record in score_records
        if record["score_margin"] is not None
    ]

    best_scores = [
        record["best_score"]
        for record in score_records
    ]

    scoring_summary = {}

    if margins:
        scoring_summary.update({
            "mean_score_margin": float(
                sum(margins) / len(margins)
            ),
            "median_score_margin": float(
                statistics.median(margins)
            ),
            "min_score_margin": float(min(margins)),
            "max_score_margin": float(max(margins)),
        })

    if best_scores:
        scoring_summary.update({
            "mean_winning_score": float(
                sum(best_scores) / len(best_scores)
            ),
            "median_winning_score": float(
                statistics.median(best_scores)
            ),
        })

    diagnostics = {
        "evaluation_mode": "label_scoring",
        "metrics": metrics,
        "classification": classification_diagnostics,
        "scoring_summary": scoring_summary,
        "samples": score_records,
    }

    # Add useful scalar summaries to normal logs/W&B.
    metrics.update(scoring_summary)

    evaluator.log_raw(
        output_path
        if output_path is not None
        else "logs/output.csv"
    )

    save_diagnostics(
        diagnostics,
        output_path,
    )

    return metrics

def build_shuffled_context_map(dataset, mode="shuffle", seed=42):
    """
    Build a mapping:
        target node_id -> context from another node
    """
    assert mode in ["shuffle", "cross_class_shuffle"]
    node_ids = []
    contexts_by_node_id = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        node_id = sample.node_id
        if torch.is_tensor(node_id):
            node_id = int(node_id.item())
        else:
            node_id = int(node_id)
        node_ids.append(node_id)
        contexts_by_node_id[node_id] = {
            "Title": sample.context["Title"],
            "Abstract": sample.context["Abstract"],
        }
    generator = torch.Generator()
    generator.manual_seed(seed)
    shuffled_map = {}
    if mode == "shuffle":
        perm = torch.randperm(
            len(node_ids),
            generator=generator,
        ).tolist()
        for i, target_node_id in enumerate(node_ids):
            source_node_id = node_ids[perm[i]]
            shuffled_map[target_node_id] = (
                contexts_by_node_id[source_node_id]
            )
        unchanged = sum(
            node_ids[i] == node_ids[perm[i]]
            for i in range(len(node_ids))
        )
        logging.info(
            "Random context shuffle: "
            "%d samples, %d unchanged assignments",
            len(node_ids),
            unchanged,
        )
    elif mode == "cross_class_shuffle":
        if not hasattr(dataset, "graph_data"):
            raise ValueError(
                "cross_class_shuffle requires dataset.graph_data."
            )
        if not hasattr(dataset.graph_data, "y"):
            raise ValueError(
                "cross_class_shuffle requires dataset.graph_data.y."
            )
        labels = dataset.graph_data.y.cpu().view(-1)
        available_node_ids = torch.tensor(
            node_ids,
            dtype=torch.long,
        )
        for target_node_id in node_ids:
            target_class = labels[target_node_id].item()
            candidate_mask = (
                labels[available_node_ids] != target_class
            )
            candidates = available_node_ids[candidate_mask]
            if len(candidates) == 0:
                raise RuntimeError(
                    f"No cross-class context candidate "
                    f"for node {target_node_id}."
                )
            random_idx = torch.randint(
                0,
                len(candidates),
                (1,),
                generator=generator,
            ).item()
            source_node_id = int(
                candidates[random_idx].item()
            )
            assert (
                labels[target_node_id].item()
                != labels[source_node_id].item()
            )
            shuffled_map[target_node_id] = (
                contexts_by_node_id[source_node_id]
            )
        logging.info(
            "Cross-class context shuffle created for %d nodes "
            "with 0 same-class assignments.",
            len(shuffled_map),
        )
    return shuffled_map

def add_adversarial_cross_class_edges(edge_index, labels, add_prob, seed=42,):
    """
    Add undirected edges between nodes belonging to different classes.
    """
    if add_prob <= 0:
        return edge_index
    assert add_prob >= 0.0
    device = edge_index.device
    edge_index_cpu = edge_index.cpu()
    labels_cpu = labels.cpu().view(-1)
    row, col = edge_index_cpu
    src = torch.minimum(row, col)
    dst = torch.maximum(row, col)
    pairs = torch.stack([src, dst], dim=0)
    unique_pairs = torch.unique(pairs, dim=1)
    num_original_edges = unique_pairs.size(1)
    num_nodes = labels_cpu.numel()
    num_to_add = int(round(add_prob * num_original_edges))
    if num_to_add == 0:
        return edge_index
    existing_edges = {
        (int(u), int(v))
        for u, v in unique_pairs.t().tolist()
    }
    generator = torch.Generator()
    generator.manual_seed(seed)
    new_edges = set()
    max_attempts = max(num_to_add * 100, 10000)
    attempts = 0
    while len(new_edges) < num_to_add and attempts < max_attempts:
        attempts += 1
        u = int(
            torch.randint(
                0,
                num_nodes,
                (1,),
                generator=generator,
            ).item()
        )
        v = int(
            torch.randint(
                0,
                num_nodes,
                (1,),
                generator=generator,
            ).item()
        )
        if u == v:
            continue
        if labels_cpu[u].item() == labels_cpu[v].item():
            continue
        a = min(u, v)
        b = max(u, v)
        pair = (a, b)
        if pair in existing_edges:
            continue
        if pair in new_edges:
            continue
        new_edges.add(pair)
    if len(new_edges) < num_to_add:
        print(
            f"WARNING: Requested {num_to_add} adversarial edges "
            f"but could only generate {len(new_edges)}."
        )
    if len(new_edges) == 0:
        return edge_index
    added_pairs = torch.tensor(
        list(new_edges),
        dtype=edge_index_cpu.dtype,
    ).t()
    combined_pairs = torch.cat(
        [unique_pairs, added_pairs],
        dim=1,
    )
    new_edge_index = to_undirected(
        combined_pairs,
        num_nodes=num_nodes,
    )
    return new_edge_index.to(device)

def perturb_node_features(x, mode="none", seed=42, noise_std=0.1):
    if mode == "none":
        return x
    assert mode in ["shuffle", "gaussian", "zero"]
    device = x.device
    x_cpu = x.cpu()
    generator = torch.Generator()
    generator.manual_seed(seed)
    if mode == "shuffle":
        perm = torch.randperm(x_cpu.size(0), generator=generator)
        new_x = x_cpu[perm]
    elif mode == "gaussian":
        noise = torch.randn(x_cpu.shape, generator=generator) * noise_std
        new_x = x_cpu + noise
    elif mode == "zero":
        new_x = torch.zeros_like(x_cpu)
    return new_x.to(device)

def drop_edges_undirected(edge_index, drop_prob, seed=42):
    if drop_prob <= 0:
        return edge_index
    assert 0.0 <= drop_prob <= 1.0
    device = edge_index.device
    edge_index_cpu = edge_index.cpu()
    row,col = edge_index_cpu
    src = torch.minimum(row,col)
    dst = torch.maximum(row,col)
    pairs = torch.stack([src, dst], dim=0)
    unique_pairs = torch.unique(pairs, dim=1)
    generator = torch.Generator()
    generator.manual_seed(seed)
    keep_mask = torch.rand(unique_pairs.size(1), generator=generator) > drop_prob
    kept_pairs = unique_pairs[:, keep_mask]
    new_edge_index = to_undirected(kept_pairs)
    return new_edge_index.to(device)

def drop_edges_degree_aware(edge_index, drop_prob, seed=42, degree_power=1.0,):
    if drop_prob <= 0:
        return edge_index
    assert 0.0 <= drop_prob <= 1.0
    assert degree_power >= 0.0
    device = edge_index.device
    edge_index_cpu = edge_index.cpu()
    row, col = edge_index_cpu
    src = torch.minimum(row, col)
    dst = torch.maximum(row, col)
    pairs = torch.stack([src, dst], dim=0)
    unique_pairs = torch.unique(pairs, dim=1)
    num_edges = unique_pairs.size(1)
    if num_edges == 0:
        return edge_index
    num_drop = int(round(drop_prob * num_edges))
    if num_drop <= 0:
        return edge_index
    if num_drop >= num_edges:
        empty_edge_index = torch.empty(
            (2, 0),
            dtype=edge_index.dtype,
            device=device,
        )
        return empty_edge_index
    u = unique_pairs[0]
    v = unique_pairs[1]
    num_nodes = int(torch.max(unique_pairs).item()) + 1
    degree = torch.zeros(num_nodes, dtype=torch.float)
    degree.scatter_add_(0, u, torch.ones_like(u, dtype=torch.float),)
    degree.scatter_add_(0, v, torch.ones_like(v, dtype=torch.float),)
    edge_scores = degree[u] + degree[v]
    weights = edge_scores.pow(degree_power)
    weights = weights + 1e-12
    generator = torch.Generator()
    generator.manual_seed(seed)
    drop_indices = torch.multinomial(
        weights,
        num_samples=num_drop,
        replacement=False,
        generator=generator,
    )
    keep_mask = torch.ones(num_edges, dtype=torch.bool)
    keep_mask[drop_indices] = False
    kept_pairs = unique_pairs[:, keep_mask]
    new_edge_index = to_undirected(
        kept_pairs,
        num_nodes=num_nodes,
    )
    return new_edge_index.to(device) 

def build_output_csv_path():
    parts = [
        FLAGS.dataset,
        FLAGS.llm_model_name,
        f"eval-{FLAGS.eval_mode}",
        f"split-{FLAGS.eval_split}",
        f"context-{FLAGS.context_mode}",
    ]
    if FLAGS.context_perturbation != "none":
        parts.extend([
            f"contextpert-{FLAGS.context_perturbation}",
            f"contextseed-{FLAGS.context_shuffle_seed}",
        ])
    if FLAGS.no_graph_token:
        parts.append("no-graph-token")
    if FLAGS.edge_drop_prob > 0:
        parts.extend([
            f"edge-{FLAGS.edge_drop_mode}",
            f"p-{FLAGS.edge_drop_prob:g}",
            f"seed-{FLAGS.edge_drop_seed}",
        ])
    if FLAGS.node_feature_noise != "none":
        parts.extend([
            f"node-{FLAGS.node_feature_noise}",
            f"node-seed-{FLAGS.node_feature_seed}",
        ])
    if FLAGS.adversarial_edge_add_prob > 0:
        parts.extend([
            "advedge",
            f"p-{FLAGS.adversarial_edge_add_prob:g}",
            f"advseed-{FLAGS.adversarial_edge_seed}",
        ])
    filename = "_".join(parts)
    filename = filename.replace("/", "_")
    return os.path.join("logs", filename + ".csv")

def diagnostics_path_from_output(output_path):
    if output_path is None:
        return "logs/diagnostics.json"

    base, _ = os.path.splitext(output_path)
    return base + "_diagnostics.json"


def save_diagnostics(diagnostics, output_path):
    path = diagnostics_path_from_output(output_path)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            diagnostics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    logging.info("Detailed diagnostics saved to %s", path)
class EvalEachEpochCallback(Callback):
    def __init__(self, val_loader, dataset_name):
        super().__init__()
        self.val_loader = val_loader
        self.dataset_name = dataset_name

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        logging.info("Running validation accuracy evaluation after epoch %s", trainer.current_epoch)

        evaluator = get_evaluator(self.dataset_name)

        metrics = evaluate(
            pl_module.model,
            self.val_loader,
            evaluator,
            graph_data=pl_module.graph_data,
        )

        def flatten_metrics(metrics, prefix=""):
            flat = {}
            for key, value in metrics.items():
                name = f"{prefix}_{key}" if prefix else key
                if isinstance(value, dict):
                    flat.update(flatten_metrics(value, name))
                elif isinstance(value, (int, float)):
                    flat[name] = value
            return flat

        flat_metrics = flatten_metrics(metrics)
        for key, value in flat_metrics.items():
            metric_name = f"epoch_val_{key}"
            pl_module.log(
                metric_name,
                float(value),
                on_step=False,
                on_epoch=True,
                prog_bar=("overall_accuracy" in key or key == "accuracy"),
                sync_dist=True,
            )
            logging.info(
                "Epoch %s - %s: %s",
                trainer.current_epoch,
                metric_name,
                value,
            )

class SaveLoRAEveryEpochCallback(Callback):
    def __init__(self, save_dir, method, llm_model_name, dataset, seed):
        super().__init__()
        self.lora_dir = os.path.join(
            save_dir,
            method,
            f"lora_{llm_model_name}_{dataset}_seed_{seed}"
        )
    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if not hasattr(pl_module.model, "llm_generator"):
            return
        if not hasattr(pl_module.model.llm_generator, "save_pretrained"):
            return
        pl_module.model.llm_generator.save_pretrained(self.lora_dir)
        logging.info("Saved LoRA adapters to %s", self.lora_dir)

def train_with_lightning(
    model, train_loader, val_loader, epochs: int, wandb_logger: WandbLogger = None
):
    """Trains the model using PyTorch Lightning, with optional wandb logging."""
    strategy = DDPStrategy() if FLAGS.strategy.lower() == "ddp" else None

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{FLAGS.save_dir}/{FLAGS.method}/",
        filename=f"projector_{FLAGS.llm_model_name}_{FLAGS.dataset}_seed_{FLAGS.seed}",
        monitor="val_loss",
    )

    ckpt_path = os.path.join(
        FLAGS.save_dir,
        FLAGS.method,
        f"projector_{FLAGS.llm_model_name}_{FLAGS.dataset}_seed_{FLAGS.seed}.ckpt",
    )
    if os.path.exists(ckpt_path):
        logging.error(
            "Checkpoint already exists at %s; exiting to avoid overwrite.", ckpt_path
        )
        raise FileExistsError(f"Checkpoint exists: {ckpt_path}")

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_monitor]
    if FLAGS.eval_each_epoch:
        callbacks.append(EvalEachEpochCallback(val_loader, FLAGS.dataset))

    if FLAGS.use_lora:
        callbacks.append(
            SaveLoRAEveryEpochCallback(
                FLAGS.save_dir,
                FLAGS.method,
                FLAGS.llm_model_name,
                FLAGS.dataset,
                FLAGS.seed,
            )
        )

    # Trainer with gradient clipping
    trainer = pl.Trainer(
        max_epochs=epochs,
        log_every_n_steps=10,
        accelerator=FLAGS.accelerator,
        devices=FLAGS.num_gpus,
        strategy=strategy,
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=FLAGS.gradient_clip_val,
        gradient_clip_algorithm="norm",
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logging.info("Training with Lightning completed.")

    # Save LoRA adapters if used
    if getattr(model.model, 'use_lora', False):
        lora_dir = os.path.join(
            FLAGS.save_dir,
            FLAGS.method,
            f"lora_{FLAGS.llm_model_name}_{FLAGS.dataset}_seed_{FLAGS.seed}"
        )
        model.model.llm_generator.save_pretrained(lora_dir)
        logging.info("Saved LoRA adapters to %s", lora_dir)


def evaluate(
    model: BaseGLM,
    loader,
    evaluator: DatasetEvaluator,
    graph_data=None,
    output_path=None,
):
    model.eval()

    graph_data = (
        graph_data.to(model.llm.device)
        if graph_data
        else None
    )

    max_tok_len = evaluator.max_class_toklen(
        model.llm.tokenizer
    )

    num_samples = (
        min(len(loader.dataset), FLAGS.max_samples)
        if FLAGS.max_samples > 0
        else len(loader.dataset)
    )

    logging.info(
        "Evaluating on %d samples",
        num_samples,
    )

    with torch.no_grad():
        for i, data in tqdm(
            enumerate(loader),
            desc="Evaluating",
            total=num_samples,
        ):
            if i >= num_samples:
                break

            if i >= 5:
                break

            data = data.to(model.llm.device)

            out_seq = model.inference(
                data,
                max_out_tokens=max_tok_len,
                graph_data=graph_data,
                no_graph_token=FLAGS.no_graph_token,
            )[0]

            evaluator(data, out_seq)

    # Legacy / parsed metrics
    metrics = evaluator.compute_metrics()

    # Generic detailed classification diagnostics
    diagnostics = {
        "evaluation_mode": "generation",
        "parsed_metrics": metrics,
        "parsed_classification":
            evaluator.compute_classification_diagnostics(),
    }

    # Dataset-specific generation diagnostics
    if hasattr(evaluator, "compute_generation_diagnostics"):
        generation_diagnostics = (
            evaluator.compute_generation_diagnostics(
                tokenizer=model.llm.tokenizer,
            )
        )

        diagnostics["generation"] = generation_diagnostics

        # Add the most important scalar diagnostics to metrics so that
        # they are also printed/logged normally.
        scalar_keys = [
            "strict_accuracy",
            "valid_rate",
            "malformed_rate",
            "conditional_valid_accuracy",
            "strict_macro_precision",
            "strict_macro_recall",
            "strict_macro_f1",
            "strict_weighted_precision",
            "strict_weighted_recall",
            "strict_weighted_f1",
        ]

        for key in scalar_keys:
            if key in generation_diagnostics:
                metrics[key] = generation_diagnostics[key]

    evaluator.log_raw(
        output_path
        if output_path is not None
        else "logs/output.csv"
    )

    save_diagnostics(
        diagnostics,
        output_path,
    )

    return metrics


def main(argv):
    torch.multiprocessing.set_start_method("fork", force=True)
    if FLAGS.num_gpus > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(f"cuda:{local_rank}")

    del argv

    # Set logging level
    numeric_level = getattr(logging, FLAGS.logging_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % FLAGS.logging_level)
    logging.set_verbosity(numeric_level)

    torch.set_float32_matmul_precision("high")

    # Initialize WandbLogger if enabled
    wandb_logger = None
    if FLAGS.use_wandb and FLAGS.wandb_mode != "disabled":
        wandb_logger = WandbLogger(
            project=FLAGS.wandb_project,
            entity=FLAGS.wandb_entity,
            name=f"{FLAGS.method}_{FLAGS.llm_model_name}_{FLAGS.dataset}_seed_{FLAGS.seed}",
            config=FLAGS.flag_values_dict(),
            log_model=False,
            save_dir=FLAGS.save_dir,
            mode=FLAGS.wandb_mode,
        )

    # Load dataset
    try:
        dataset = load_dataset(FLAGS.root)
    except Exception:
        logging.exception("Failed to load dataset")
        raise

    logging.info("Dataset details: %s", dataset)
    try:
        first_sample = dataset[0]
        logging.info(
            "First sample - id: %s, question: %s, label: %s, context: %s",
            first_sample.id,
            first_sample.question,
            first_sample.label,
            first_sample.context,
        )
    except Exception as e:
        logging.error("Error accessing data list items: %s", e)

    shuffled_context_map = None
    if FLAGS.context_perturbation in [
        "shuffle",
        "cross_class_shuffle",
    ]:
        if FLAGS.dataset != "cora":
            raise ValueError(
                "Context perturbation is currently implemented for Cora only."
            )
        shuffled_context_map = build_shuffled_context_map(
            dataset,
            mode=FLAGS.context_perturbation,
            seed=FLAGS.context_shuffle_seed,
        )
        logging.info(
            "Built context mapping: mode=%s, seed=%s, samples=%d",
            FLAGS.context_perturbation,
            FLAGS.context_shuffle_seed,
            len(shuffled_context_map),
        )

    if FLAGS.edge_drop_prob > 0 and hasattr(dataset, "graph_data"):
        old_edges = dataset.graph_data.edge_index.size(1)
        if FLAGS.edge_drop_mode == "random":
            dataset.graph_data.edge_index = drop_edges_undirected(
                dataset.graph_data.edge_index,
                drop_prob=FLAGS.edge_drop_prob,
                seed=FLAGS.edge_drop_seed,
            )
        elif FLAGS.edge_drop_mode == "degree":
            dataset.graph_data.edge_index = drop_edges_degree_aware(
                dataset.graph_data.edge_index,
                drop_prob=FLAGS.edge_drop_prob,
                seed=FLAGS.edge_drop_seed,
                degree_power=FLAGS.edge_drop_degree_power,
            )
        else:
            raise ValueError(f"Unknown edge_drop_mode: {FLAGS.edge_drop_mode}")
        new_edges = dataset.graph_data.edge_index.size(1)
        logging.info(
            "Applied edge drop: mode=%s, p=%s, seed=%s, edges %s -> %s",
            FLAGS.edge_drop_mode,
            FLAGS.edge_drop_prob,
            FLAGS.edge_drop_seed,
            old_edges,
            new_edges,
        )

    if (FLAGS.adversarial_edge_add_prob > 0 and hasattr(dataset, "graph_data")):
        if not hasattr(dataset.graph_data, "y"):
            raise ValueError("Adversarial cross-class edge addition requires graph_data.y labels.")
        old_edges = dataset.graph_data.edge_index.size(1)
        dataset.graph_data.edge_index = add_adversarial_cross_class_edges(
            dataset.graph_data.edge_index,
            dataset.graph_data.y,
            add_prob=FLAGS.adversarial_edge_add_prob,
            seed=FLAGS.adversarial_edge_seed,
        )
        new_edges = dataset.graph_data.edge_index.size(1)
        logging.info(
            "Applied adversarial cross-class edge addition: "
            "add_prob=%s, seed=%s, directed edges %s -> %s",
            FLAGS.adversarial_edge_add_prob,
            FLAGS.adversarial_edge_seed,
            old_edges,
            new_edges,
        )

    if FLAGS.node_feature_noise != "none" and hasattr(dataset, "graph_data"):
        logging.info(
            "Applying node feature perturbation: mode=%s, seed=%s, std=%s",
            FLAGS.node_feature_noise,
            FLAGS.node_feature_seed,
            FLAGS.node_feature_noise_std,
        )
        dataset.graph_data.x = perturb_node_features(
            dataset.graph_data.x,
            mode=FLAGS.node_feature_noise,
            seed=FLAGS.node_feature_seed,
            noise_std=FLAGS.node_feature_noise_std,
        )

    # Split and create loaders
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, FLAGS.seed)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, FLAGS.batch_size, FLAGS.num_workers
    )

    # Initialize model
    # Pass use_lora flag into initializer if supported
    model = initialize_model(
        FLAGS.lr,
        graph_data=dataset.graph_data,
        use_lora=(FLAGS.use_lora and not FLAGS.evaluate),
    )

    model.model.context_mode = FLAGS.context_mode
    model.model.context_perturbation = FLAGS.context_perturbation
    model.model.shuffled_context_map = shuffled_context_map

    # Train or evaluate
    if not FLAGS.evaluate and FLAGS.method != "llm-only":
        if FLAGS.use_lightning:
            train_with_lightning(
                model, train_loader, val_loader, FLAGS.epochs, wandb_logger
            )
    else:
        # Load best checkpoint
        try:
            lm_module_cls = get_glm_lightning_module(FLAGS.method)
            lm_module = lm_module_cls.load_from_checkpoint(
                f"{FLAGS.save_dir}/{FLAGS.method}/projector_{FLAGS.llm_model_name}_{FLAGS.dataset}_seed_{FLAGS.seed}.ckpt",
                model=model.model,
                lr=FLAGS.lr,
                graph_data=dataset.graph_data,
                scheduler_step_size=FLAGS.scheduler_step_size,
                scheduler_gamma=FLAGS.scheduler_gamma,
            )
            # Load LoRA adapters if used
            if FLAGS.use_lora:
                lora_dir = os.path.join(
                    FLAGS.save_dir,
                    FLAGS.method,
                    f"lora_{FLAGS.llm_model_name}_{FLAGS.dataset}_seed_{FLAGS.seed}"
                )
                if not os.path.exists(lora_dir):
                    logging.error("LoRA directory does not exist: %s", lora_dir)
                    return
                lm_module.model.llm_generator = PeftModel.from_pretrained(
                    lm_module.model.llm_generator, lora_dir
                )
                logging.info("Loaded LoRA adapters from %s", lora_dir)
                logging.info("Loaded LoRA model class: %s", type(lm_module.model.llm_generator))
                logging.info("Active PEFT adapters: %s", getattr(lm_module.model.llm_generator, "active_adapters", None))
            model = lm_module
            model.model.context_mode = FLAGS.context_mode
            model.model.context_perturbation = FLAGS.context_perturbation
            model.model.shuffled_context_map = shuffled_context_map
        except Exception as e:
            if FLAGS.method == "llm-only":
                logging.info("Loading LLM only model. No checkpoint found.")
            else:
                logging.error("Error loading model checkpoint: %s", e)
                return

        # Evaluate
        evaluator = get_evaluator(FLAGS.dataset)
        eval_dataset = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }[FLAGS.eval_split]
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=FLAGS.num_workers,
        )
        output_csv_path = build_output_csv_path()
        logging.info("Predictions will be saved to %s", output_csv_path)
        if FLAGS.eval_mode == "generation":
            metrics = evaluate(
                model.model,
                eval_loader,
                evaluator,
                graph_data=dataset.graph_data,
                output_path=output_csv_path,
            )
        elif FLAGS.eval_mode == "label_scoring":
            candidate_labels = get_candidate_labels(dataset)
            metrics = evaluate_label_scoring(
                model.model,
                eval_loader,
                evaluator,
                candidate_labels,
                graph_data=dataset.graph_data,
                output_path=output_csv_path,
            )
        else:
            raise ValueError(f"Unknown eval_mode: {FLAGS.eval_mode}")

        # Log final metrics
        for key, value in metrics.items():
            logging.info(f"{key}: {value}")

    # Finish wandb run
    if wandb_logger is not None:
        if FLAGS.evaluate or FLAGS.method == "llm-only":
            wandb_logger.log_metrics(metrics)
        wandb.finish()


if __name__ == "__main__":
    app.run(main)
