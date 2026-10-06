import argparse
import json
import os

import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.data import get_dataset
from src.models import GraphSAGE


SEED = 0
N_GNN_TOKS = 10


MODEL_CONFIGS = {
    "phi-1.5": {
        "pretrain_name": "phi-1_5",
        "projector_name": "phi-1.5",
    },
    "qwen2.5-1.5b": {
        "pretrain_name": "qwen2.5-1.5b",
        "projector_name": "qwen2.5-1.5b",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Linear probe on trained TEA projector representations."
    )
    parser.add_argument(
        "--llm_model_name",
        type=str,
        required=True,
        choices=MODEL_CONFIGS.keys(),
        help="LLM backbone used by TEA-GLM.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    llm_name = args.llm_model_name
    config = MODEL_CONFIGS[llm_name]

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print("TEA projector linear probe")
    print("=" * 70)
    print("LLM backbone:", llm_name)

    # ------------------------------------------------------------
    # 1. Load clean Cora
    # ------------------------------------------------------------
    DatasetClass = get_dataset("cora")
    dataset = DatasetClass(root="datasets")
    graph = dataset.graph_data

    split = dataset.split(seed=SEED)

    train_ids = np.array(split["train"])
    val_ids = np.array(split["val"])
    test_ids = np.array(split["test"])

    print("\nDataset")
    print("Nodes:", graph.num_nodes)
    print("Features:", tuple(graph.x.shape))
    print("Labels:", tuple(graph.y.shape))
    print("Edges:", tuple(graph.edge_index.shape))

    print("\nSplit")
    print("Train:", len(train_ids))
    print("Val:", len(val_ids))
    print("Test:", len(test_ids))

    # ------------------------------------------------------------
    # 2. Load exact TEA-pretrained GraphSAGE
    # ------------------------------------------------------------
    pretrain_base = (
        f"models/tea-glm/"
        f"pretrain_cora_seed_{SEED}_{config['pretrain_name']}"
    )

    params_path = pretrain_base + "_model_params.json"
    weights_path = pretrain_base + "_model.pth"

    print("\nGNN checkpoint")
    print("Params:", params_path)
    print("Weights:", weights_path)

    if not os.path.exists(params_path):
        raise FileNotFoundError(params_path)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(weights_path)

    with open(params_path, "r") as f:
        param_dict = json.load(f)

    print("\nLoaded GNN parameters:")
    for k, v in param_dict.items():
        print(f"  {k}: {v}")

    gnn = GraphSAGE(**param_dict)

    gnn_state = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=False,
    )

    gnn.load_state_dict(gnn_state)
    gnn.eval()

    # ------------------------------------------------------------
    # 3. Extract frozen GNN embeddings
    # ------------------------------------------------------------
    with torch.no_grad():
        z = gnn(
            graph.x.cpu(),
            graph.edge_index.cpu(),
        )

    print("\nGNN representation")
    print("Shape:", tuple(z.shape))
    print("NaNs:", torch.isnan(z).sum().item())
    print("Infs:", torch.isinf(z).sum().item())

    # ------------------------------------------------------------
    # 4. Load trained downstream projector
    # ------------------------------------------------------------
    ckpt_path = (
        f"models/tea-glm/"
        f"projector_{config['projector_name']}_cora_seed_{SEED}.ckpt"
    )

    print("\nProjector checkpoint")
    print("Path:", ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = ckpt["state_dict"]

    weight_key = "model.projector.0.weight"
    bias_key = "model.projector.0.bias"

    if weight_key not in state_dict:
        raise KeyError(
            f"{weight_key} not found in checkpoint. "
            f"Available projector keys: "
            f"{[k for k in state_dict if 'projector' in k]}"
        )

    W = state_dict[weight_key]
    b = state_dict[bias_key]

    print("Projector weight shape:", tuple(W.shape))
    print("Projector bias shape:", tuple(b.shape))

    # Sanity check: projector input must match GNN output
    if W.shape[1] != z.shape[1]:
        raise ValueError(
            f"Dimension mismatch: GNN produces {z.shape[1]} dimensions "
            f"but projector expects {W.shape[1]}."
        )

    # ------------------------------------------------------------
    # 5. Apply exact learned projector
    # ------------------------------------------------------------
    with torch.no_grad():
        projected_flat = torch.nn.functional.linear(z, W, b)

    print("\nFlat projected shape:", tuple(projected_flat.shape))

    if projected_flat.shape[1] % N_GNN_TOKS != 0:
        raise ValueError(
            f"Projected dimension {projected_flat.shape[1]} "
            f"is not divisible by {N_GNN_TOKS} graph tokens."
        )

    token_dim = projected_flat.shape[1] // N_GNN_TOKS

    projected_tokens = projected_flat.view(
        projected_flat.shape[0],
        N_GNN_TOKS,
        token_dim,
    )

    print("Projected token shape:", tuple(projected_tokens.shape))
    print("Token dimension:", token_dim)

    # Flatten all 10 graph tokens exactly as in the original probe.
    X_proj = projected_tokens.reshape(
        projected_tokens.shape[0],
        -1,
    ).numpy()

    y = graph.y.cpu().numpy()

    # ------------------------------------------------------------
    # 6. Same split, same linear probe
    # ------------------------------------------------------------
    X_train = X_proj[train_ids]
    X_val = X_proj[val_ids]
    X_test = X_proj[test_ids]

    y_train = y[train_ids]
    y_val = y[val_ids]
    y_test = y[test_ids]

    clf = LogisticRegression(
        max_iter=5000,
        random_state=SEED,
    )

    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print("\n" + "=" * 70)
    print("PROJECTOR LINEAR PROBE RESULTS")
    print("=" * 70)
    print(f"LLM backbone:   {llm_name}")
    print(f"GNN dimension:  {z.shape[1]}")
    print(f"Token dimension:{token_dim}")
    print(f"Probe dimension:{X_proj.shape[1]}")
    print(f"Train accuracy: {train_acc:.6f}")
    print(f"Val accuracy:   {val_acc:.6f}")
    print(f"Test accuracy:  {test_acc:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()