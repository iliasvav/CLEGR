import os
import random

import numpy as np
import pandas as pd
import torch
import argparse
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import training


FLAGS = training.FLAGS


# ============================================================
# Experiment configuration
# ============================================================

METHOD = "tea-glm"
DATASET = "cora"
SEED = 0

BATCH_SIZE = 1
NUM_WORKERS = 0

def parse_args():
    parser = argparse.ArgumentParser(
        description="Layer-wise representation probe for Cora TEA-GLM."
    )
    parser.add_argument(
        "--llm_model_name",
        type=str,
        required=True,
        choices=["phi-1.5", "qwen2.5-1.5b"],
    )
    return parser.parse_args()

# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset helpers
# ============================================================

def get_node_id(data):
    """
    Return the canonical PyG/Cora node ID stored in a dataset item.
    """

    node_id = data.node_id

    if torch.is_tensor(node_id):
        return int(node_id.view(-1)[0].item())

    return int(node_id)


def get_split_node_ids(split_dataset):
    """
    Recover canonical Cora node IDs from a project split.

    Important:
    split positions are NOT assumed to equal graph node IDs.
    """

    node_ids = []

    for item in split_dataset:
        node_ids.append(get_node_id(item))

    return np.asarray(node_ids, dtype=np.int64)


# ============================================================
# Linear probing
# ============================================================

def fit_linear_probe(
    X,
    y,
    train_ids,
    val_ids,
    test_ids,
    representation_name,
    layer_idx,
):
    """
    Fit a Ridge linear classifier on one hidden-state representation.

    Hidden states are stored as float16 to reduce memory use.
    The current layer is converted to float32 before sklearn fitting.
    """

    X = X.astype(np.float32, copy=False)

    X_train = X[train_ids]
    X_val = X[val_ids]
    X_test = X[test_ids]

    y_train = y[train_ids]
    y_val = y[val_ids]
    y_test = y[test_ids]

    clf = RidgeClassifier(
        alpha=1.0,
    )

    clf.fit(
        X_train,
        y_train,
    )

    train_pred = clf.predict(X_train)
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    return {
        "representation": representation_name,
        "layer": layer_idx,
        "train_accuracy": accuracy_score(
            y_train,
            train_pred,
        ),
        "val_accuracy": accuracy_score(
            y_val,
            val_pred,
        ),
        "test_accuracy": accuracy_score(
            y_test,
            test_pred,
        ),
    }


# ============================================================
# Hidden-state extraction
# ============================================================

@torch.no_grad()
def extract_layer_representations(
    glm,
    graph_data,
    dataset,
):
    """
    Extract two representations from every LLM hidden-state level.

    1. graph_mean
       Mean pooling over the 10 continuous graph-token positions.

    2. answer_position
       Hidden state at the final prompt token immediately before
       autoregressive class-label generation.

    No answer is passed to _get_embeds(), so no ground-truth class
    token is present during representation extraction.
    """

    glm.eval()

    device = glm.llm.device
    graph_data = graph_data.to(device)

    # --------------------------------------------------------
    # Frozen TEA-pretrained GNN representations
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("COMPUTING FROZEN GNN REPRESENTATIONS")
    print("=" * 72)

    node_representations = glm.encode_graph(
        graph_data
    )

    print(
        "GNN representations:",
        tuple(node_representations.shape),
    )

    assert node_representations.ndim == 2

    assert (
        node_representations.size(0)
        == graph_data.num_nodes
    )

    num_nodes = graph_data.num_nodes

    hidden_dim = (
        glm.llm_generator.config.hidden_size
    )

    # --------------------------------------------------------
    # Labels indexed directly by canonical graph node ID
    # --------------------------------------------------------

    labels = (
        graph_data.y
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    assert labels.shape[0] == num_nodes

    # --------------------------------------------------------
    # Track that every canonical node appears exactly once
    # --------------------------------------------------------

    seen = np.zeros(
        num_nodes,
        dtype=np.bool_,
    )

    graph_mean_all = None
    answer_position_all = None
    num_hidden_levels = None

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print()
    print("=" * 72)
    print(
        f"EXTRACTING {FLAGS.llm_model_name} HIDDEN STATES"
    )
    print("=" * 72)

    for sample_idx, data in tqdm(
        enumerate(loader),
        total=len(dataset),
        desc="Nodes",
    ):
        data = data.to(device)

        node_id = get_node_id(data)

        assert 0 <= node_id < num_nodes

        assert not seen[node_id], (
            f"Node {node_id} encountered more than once."
        )

        seen[node_id] = True

        # ----------------------------------------------------
        # Frozen GNN representation for this node
        # ----------------------------------------------------

        node_x = (
            node_representations[node_id]
            .unsqueeze(0)
        )

        assert node_x.shape == (
            1,
            node_representations.size(1),
        )

        # ----------------------------------------------------
        # Trained projector -> 10 continuous graph tokens
        # ----------------------------------------------------

        projected = glm.projector(
            node_x
        )

        assert projected.ndim == 3
        assert projected.size(0) == 1

        graph_tokens = projected[0]

        n_graph_tokens = (
            graph_tokens.size(0)
        )

        assert (
            n_graph_tokens
            == glm.n_gnn_toks
        )

        assert (
            graph_tokens.size(1)
            == hidden_dim
        )

        embedding = [
            graph_tokens
        ]

        # ----------------------------------------------------
        # Construct same context as normal TEA inference
        # ----------------------------------------------------

        context = glm.get_context(
            data
        )

        # ----------------------------------------------------
        # Build prompt WITHOUT ground-truth answer
        # ----------------------------------------------------

        (
            inputs_embeds,
            attention_mask,
            label_input_ids,
        ) = glm.llm._get_embeds(
            data.question,
            context,
            embedding,
        )

        assert label_input_ids is None

        # ----------------------------------------------------
        # Recover exact graph-token positions
        #
        # Input sequence:
        #
        # [prefix]
        # [continuous graph tokens]
        # [context/question/user suffix]
        # ----------------------------------------------------

        (
            batch_size,
            _question_tokens,
            _context_tokens,
            _user_suffix_tokens,
            prefix_embeds,
            _pad_embeds,
        ) = glm.llm._encode_inputs(
            data.question,
            context,
        )

        assert batch_size == 1

        prefix_len = (
            prefix_embeds.size(0)
        )

        graph_start = (
            prefix_len
        )

        graph_end = (
            graph_start
            + n_graph_tokens
        )

        seq_len = (
            inputs_embeds.size(1)
        )

        valid_len = int(
            attention_mask[0]
            .sum()
            .item()
        )

        # batch_size=1 means no left padding is expected.
        assert (
            valid_len
            == seq_len
        )

        answer_position = (
            valid_len - 1
        )

        assert graph_start >= 0

        assert (
            graph_end
            <= answer_position
        )

        # ----------------------------------------------------
        # First-node indexing verification
        # ----------------------------------------------------

        if sample_idx == 0:

            inserted_graph_tokens = (
                inputs_embeds[
                    0,
                    graph_start:graph_end,
                    :
                ]
            )

            max_difference = (
                inserted_graph_tokens.float()
                - graph_tokens.float()
            ).abs().max().item()

            print()
            print()
            print(
                "First-node indexing verification"
            )
            print(
                "--------------------------------"
            )
            print(
                "node_id:",
                node_id,
            )
            print(
                "prefix_len:",
                prefix_len,
            )
            print(
                "graph positions:",
                graph_start,
                "to",
                graph_end - 1,
            )
            print(
                "answer position:",
                answer_position,
            )
            print(
                "sequence length:",
                seq_len,
            )
            print(
                "max graph-token difference:",
                f"{max_difference:.8f}",
            )

            assert (
                max_difference
                < 1e-4
            )

        with glm.llm.autocast_context:

            outputs = glm.llm_generator(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        hidden_states = (
            outputs.hidden_states
        )

        # ----------------------------------------------------
        # Allocate CPU storage after first forward pass
        # ----------------------------------------------------

        if graph_mean_all is None:

            num_hidden_levels = len(
                hidden_states
            )

            print()
            print(
                "Hidden-state levels:",
                num_hidden_levels,
            )
            print(
                "Hidden size:",
                hidden_dim,
            )

            graph_mean_all = np.empty(
                (
                    num_hidden_levels,
                    num_nodes,
                    hidden_dim,
                ),
                dtype=np.float16,
            )

            answer_position_all = np.empty(
                (
                    num_hidden_levels,
                    num_nodes,
                    hidden_dim,
                ),
                dtype=np.float16,
            )

        assert (
            len(hidden_states)
            == num_hidden_levels
        )

        # ----------------------------------------------------
        # Extract graph-mean and answer-position vectors
        # ----------------------------------------------------

        for layer_idx, hidden in enumerate(
            hidden_states
        ):

            graph_hidden = hidden[
                0,
                graph_start:graph_end,
                :
            ]

            assert (
                graph_hidden.size(0)
                == n_graph_tokens
            )

            answer_hidden = hidden[
                0,
                answer_position,
                :
            ]

            graph_mean = (
                graph_hidden.mean(
                    dim=0
                )
            )

            graph_mean_all[
                layer_idx,
                node_id,
                :,
            ] = (
                graph_mean
                .float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )

            answer_position_all[
                layer_idx,
                node_id,
                :,
            ] = (
                answer_hidden
                .float()
                .cpu()
                .numpy()
                .astype(np.float16)
            )

        # ----------------------------------------------------
        # Release temporary GPU tensors
        # ----------------------------------------------------

        del outputs
        del hidden_states
        del inputs_embeds

    assert seen.all(), (
        f"Only saw "
        f"{int(seen.sum())} / "
        f"{num_nodes} nodes."
    )

    print()
    print(
        "All canonical nodes extracted exactly once: OK"
    )

    return (
        graph_mean_all,
        answer_position_all,
        labels,
    )


# ============================================================
# Layer-wise probing
# ============================================================

def run_layer_probes(
    graph_mean_all,
    answer_position_all,
    labels,
    train_ids,
    val_ids,
    test_ids,
    results_path,
):
    """
    Fit one Ridge linear probe for every hidden-state level
    and both representation types.
    """

    os.makedirs(
        os.path.dirname(
            results_path
        ),
        exist_ok=True,
    )

    results = []

    num_hidden_levels = (
        graph_mean_all.shape[0]
    )

    print()
    print("=" * 72)
    print("FITTING LAYER-WISE LINEAR PROBES")
    print("=" * 72)

    print(
        "Hidden-state levels:",
        num_hidden_levels,
    )

    print(
        "Probes to fit:",
        num_hidden_levels * 2,
    )

    representations = [
        (
            "graph_mean",
            graph_mean_all,
        ),
        (
            "answer_position",
            answer_position_all,
        ),
    ]

    for (
        representation_name,
        representation_array,
    ) in representations:

        print()
        print(
            "Representation:",
            representation_name,
        )
        print("-" * 72)

        for layer_idx in range(
            num_hidden_levels
        ):

            result = fit_linear_probe(
                X=representation_array[
                    layer_idx
                ],
                y=labels,
                train_ids=train_ids,
                val_ids=val_ids,
                test_ids=test_ids,
                representation_name=(
                    representation_name
                ),
                layer_idx=layer_idx,
            )

            results.append(
                result
            )

            print(
                f"Layer {layer_idx:02d} | "
                f"train="
                f"{result['train_accuracy']:.4f} | "
                f"val="
                f"{result['val_accuracy']:.4f} | "
                f"test="
                f"{result['test_accuracy']:.4f}"
            )

            # Save after every layer so partial results survive
            # interruption.
            pd.DataFrame(
                results
            ).to_csv(
                results_path,
                index=False,
            )

    return pd.DataFrame(
        results
    )


# ============================================================
# Final summary
# ============================================================

def print_summary(
    results_df,
    results_path
):
    print()
    print("=" * 72)
    print("LAYER-WISE PROBE SUMMARY")
    print("=" * 72)

    for representation_name in [
        "graph_mean",
        "answer_position",
    ]:

        subset = (
            results_df[
                results_df[
                    "representation"
                ]
                == representation_name
            ]
            .sort_values("layer")
            .reset_index(drop=True)
        )

        best_row = subset.loc[
            subset[
                "test_accuracy"
            ].idxmax()
        ]

        first_row = (
            subset.iloc[0]
        )

        final_row = (
            subset.iloc[-1]
        )

        print()
        print(
            representation_name
        )

        print(
            "  Layer 0 test accuracy: "
            f"{first_row['test_accuracy']:.6f}"
        )

        print(
            "  Best test accuracy:    "
            f"{best_row['test_accuracy']:.6f} "
            f"(layer "
            f"{int(best_row['layer'])})"
        )

        print(
            "  Final test accuracy:   "
            f"{final_row['test_accuracy']:.6f}"
        )

    print()
    print(
        "Saved results to:",
        results_path,
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    llm_model = args.llm_model_name

    results_path = (
        f"logs/cora_tea_{llm_model}_layer_probe.csv"
    )
    seed_everything(
        SEED
    )

    FLAGS([
        "cora_tea_phi_layer_probe.py",
        f"--method={METHOD}",
        f"--dataset={DATASET}",
        f"--llm_model_name={llm_model}",
        "--evaluate",
        "--use_lora",
        f"--seed={SEED}",
        f"--batch_size={BATCH_SIZE}",
        f"--num_workers={NUM_WORKERS}",
        "--num_gpus=1",
    ])

    print()
    print("=" * 72)
    print("CORA TEA-GLM LAYER-WISE PROBE")
    print("Backbone:", llm_model)
    print("=" * 72)

    print(
        "Method:",
        FLAGS.method,
    )

    print(
        "Dataset:",
        FLAGS.dataset,
    )

    print(
        "LLM:",
        FLAGS.llm_model_name,
    )

    print(
        "Seed:",
        FLAGS.seed,
    )

    # --------------------------------------------------------
    # Load clean canonical Cora
    # --------------------------------------------------------

    print()
    print(
        "Loading clean Cora..."
    )

    dataset = (
        training.load_dataset(
            FLAGS.root,
            FLAGS.dataset,
        )
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = training.split_dataset(
        dataset,
        FLAGS.seed,
    )

    print(
        "Split sizes:",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    assert (
        len(dataset)
        == 2708
    )

    assert (
        len(train_dataset)
        == 1624
    )

    assert (
        len(val_dataset)
        == 542
    )

    assert (
        len(test_dataset)
        == 542
    )

    assert (
        dataset.graph_data.num_nodes
        == 2708
    )

    # --------------------------------------------------------
    # Reconstruct project split using canonical node IDs
    # --------------------------------------------------------

    train_ids = (
        get_split_node_ids(
            train_dataset
        )
    )

    val_ids = (
        get_split_node_ids(
            val_dataset
        )
    )

    test_ids = (
        get_split_node_ids(
            test_dataset
        )
    )

    assert (
        len(np.unique(train_ids))
        == 1624
    )

    assert (
        len(np.unique(val_ids))
        == 542
    )

    assert (
        len(np.unique(test_ids))
        == 542
    )

    all_split_ids = (
        np.concatenate(
            [
                train_ids,
                val_ids,
                test_ids,
            ]
        )
    )

    assert (
        len(
            np.unique(
                all_split_ids
            )
        )
        == 2708
    )

    print(
        "Canonical node-id split reconstruction: OK"
    )

    # --------------------------------------------------------
    # Reconstruct TEA-GLM architecture
    #
    # During evaluation LoRA is NOT created here.
    # The trained adapter is loaded separately below.
    # --------------------------------------------------------

    print()
    print(
        "Initializing TEA-GLM architecture..."
    )

    model = training.initialize_model(
        FLAGS.lr,
        graph_data=dataset.graph_data,
        use_lora=False,
    )

    # --------------------------------------------------------
    # Load trained projector checkpoint
    # --------------------------------------------------------

    lm_module_cls = (
        training.get_glm_lightning_module(
            FLAGS.method
        )
    )

    ckpt_path = (
        f"{FLAGS.save_dir}/"
        f"{FLAGS.method}/"
        f"projector_"
        f"{FLAGS.llm_model_name}_"
        f"{FLAGS.dataset}_"
        f"seed_{FLAGS.seed}.ckpt"
    )

    print()
    print(
        "Loading projector checkpoint:"
    )

    print(
        ckpt_path
    )

    assert os.path.exists(
        ckpt_path
    )

    lm_module = (
        lm_module_cls.load_from_checkpoint(
            ckpt_path,
            model=model.model,
            lr=FLAGS.lr,
            graph_data=dataset.graph_data,
            scheduler_step_size=(
                FLAGS.scheduler_step_size
            ),
            scheduler_gamma=(
                FLAGS.scheduler_gamma
            ),
        )
    )

    print(
        "Projector checkpoint loaded: OK"
    )

    # --------------------------------------------------------
    # Verify projector weights really came from checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_weight = (
        checkpoint["state_dict"][
            "model.projector.0.weight"
        ]
    )

    checkpoint_bias = (
        checkpoint["state_dict"][
            "model.projector.0.bias"
        ]
    )

    loaded_weight = (
        lm_module
        .model
        .projector[0]
        .weight
        .detach()
        .cpu()
    )

    loaded_bias = (
        lm_module
        .model
        .projector[0]
        .bias
        .detach()
        .cpu()
    )

    max_weight_difference = (
        checkpoint_weight
        .sub(loaded_weight)
        .abs()
        .max()
        .item()
    )

    max_bias_difference = (
        checkpoint_bias
        .sub(loaded_bias)
        .abs()
        .max()
        .item()
    )

    print(
        "Projector weight load max diff:",
        f"{max_weight_difference:.8f}",
    )

    print(
        "Projector bias load max diff:",
        f"{max_bias_difference:.8f}",
    )

    assert torch.allclose(
        checkpoint_weight,
        loaded_weight,
    )

    assert torch.allclose(
        checkpoint_bias,
        loaded_bias,
    )

    print(
        "Projector checkpoint verification: OK"
    )

    del checkpoint

    lora_dir = os.path.join(
        FLAGS.save_dir,
        FLAGS.method,
        (
            f"lora_"
            f"{FLAGS.llm_model_name}_"
            f"{FLAGS.dataset}_"
            f"seed_{FLAGS.seed}"
        ),
    )

    print()
    print(
        "Loading LoRA adapter:"
    )

    print(
        lora_dir
    )

    assert os.path.exists(
        lora_dir
    )

    lm_module.model.llm_generator = (
        training.PeftModel.from_pretrained(
            lm_module.model.llm_generator,
            lora_dir,
        )
    )

    print(
        "LoRA adapter loaded: OK"
    )

    print(
        "LLM class:",
        type(
            lm_module
            .model
            .llm_generator
        ),
    )

    active_adapters = getattr(
        lm_module
        .model
        .llm_generator,
        "active_adapters",
        None,
    )

    print(
        "Active adapters:",
        active_adapters,
    )

    assert active_adapters is not None

    assert (
        "default"
        in active_adapters
    )

    print(
        "LoRA adapter verification: OK"
    )

    glm = (
        lm_module.model
    )

    glm.eval()

    (
        graph_mean_all,
        answer_position_all,
        labels,
    ) = extract_layer_representations(
        glm=glm,
        graph_data=dataset.graph_data,
        dataset=dataset,
    )

    del lm_module
    del model
    del glm

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Fit layer-wise Ridge linear probes
    # --------------------------------------------------------

    results_df = (
        run_layer_probes(
            graph_mean_all=(
                graph_mean_all
            ),
            answer_position_all=(
                answer_position_all
            ),
            labels=labels,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            results_path=results_path,
        )
    )

    # --------------------------------------------------------
    # Print concise summary
    # --------------------------------------------------------

    print_summary(
        results_df,
        results_path,
    )


if __name__ == "__main__":
    main()