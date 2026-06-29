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
flags.DEFINE_float("edge_drop_prob", 0.0, "Probability of randomly dropping edges at inference time.")
flags.DEFINE_integer("edge_drop_seed", 42, "Random seed for edge dropping.")

flags.DEFINE_string("node_feature_noise", "none", "Node feature perturbation: none, shuffle, gaussian, zero.")
flags.DEFINE_float("node_feature_noise_std", 0.1, "Std for Gaussian node feature noise.")
flags.DEFINE_integer("node_feature_seed", 42, "Random seed for node feature perturbation.")

#report accuracy after each epoch
flags.DEFINE_bool("eval_each_epoch", False, "Run full generative validation evaluation after each epoch.")
flags.DEFINE_string("eval_split", "test", "Which split to evaluate: train, val, test.")
flags.DEFINE_string("context_mode", "full", "Context ablation mode: full, no_title, no_abstract, empty.")

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

        for key, value in metrics.items():
            metric_name = f"epoch_val_{key}"

            pl_module.log(
                metric_name,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
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


def evaluate(model: BaseGLM, loader, evaluator: DatasetEvaluator, graph_data=None):
    model.eval()
    graph_data = graph_data.to(model.llm.device) if graph_data else None
    max_tok_len = evaluator.max_class_toklen(model.llm.tokenizer)

    num_samples = (
        min(len(loader.dataset), FLAGS.max_samples)
        if FLAGS.max_samples > 0
        else len(loader.dataset)
    )
    logging.info("Evaluating on %d samples", num_samples)

    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), desc="Evaluating", total=num_samples):
            if i >= num_samples:
                break

            data = data.to(model.llm.device)
            out_seq = model.inference(
                data, max_out_tokens=max_tok_len, graph_data=graph_data
            )[0]
            evaluator(data, out_seq)

    metrics = evaluator.compute_metrics()
    evaluator.log_raw()
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
    except Exception as e:
        logging.error("Failed to load dataset: %s", e)
        return

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

    if FLAGS.edge_drop_prob > 0 and hasattr(dataset, "graph_data"):
        old_edges = dataset.graph_data.edge_index.size(1)
        dataset.graph_data.edge_index = drop_edges_undirected(
            dataset.graph_data.edge_index,
            drop_prob=FLAGS.edge_drop_prob,
            seed=FLAGS.edge_drop_seed,
        )
        new_edges = dataset.graph_data.edge_index.size(1)
        logging.info(
            "Applied inference-time edge drop: p=%s, seed=%s, edges %s -> %s",
            FLAGS.edge_drop_prob,
            FLAGS.edge_drop_seed,
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
        metrics = evaluate(
            model.model,
            eval_loader,
            evaluator,
            graph_data=dataset.graph_data,
        )

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
