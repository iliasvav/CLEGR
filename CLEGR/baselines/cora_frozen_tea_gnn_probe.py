import json
import os

import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.data import get_dataset
from src.models import GraphSAGE


SEED = 0
DATASET = "cora"
ROOT = "datasets"
METHOD = "tea-glm"

# Change this between phi-1.5 and phi-3.5
LLM_NAME = "phi-1_5"

PRETRAIN_PATH = f"models/{METHOD}/pretrain_{DATASET}_seed_{SEED}"


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 70)
    print("Frozen TEA-GNN linear probe")
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. Load the exact same Cora dataset used by training.py
    # ------------------------------------------------------------
    DatasetClass = get_dataset(DATASET)
    dataset = DatasetClass(root=ROOT)

    graph = dataset.graph_data

    print("\nDataset")
    print("Nodes:", graph.num_nodes)
    print("Features:", tuple(graph.x.shape))
    print("Labels:", tuple(graph.y.shape))
    print("Edges:", tuple(graph.edge_index.shape))

    # ------------------------------------------------------------
    # 2. Reproduce the exact project split
    # ------------------------------------------------------------
    split = dataset.split(seed=SEED)

    train_ids = torch.tensor(split["train"], dtype=torch.long)
    val_ids = torch.tensor(split["val"], dtype=torch.long)
    test_ids = torch.tensor(split["test"], dtype=torch.long)

    print("\nSplit")
    print("Train:", len(train_ids))
    print("Val:", len(val_ids))
    print("Test:", len(test_ids))

    # ------------------------------------------------------------
    # 3. Load the exact TEA-pretrained GraphSAGE
    # ------------------------------------------------------------
    llm_name_for_file = os.path.basename(LLM_NAME)

    params_path = (
        f"{PRETRAIN_PATH}_{llm_name_for_file}_model_params.json"
    )
    weights_path = (
        f"{PRETRAIN_PATH}_{llm_name_for_file}_model.pth"
    )

    print("\nCheckpoint files")
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

    state_dict = torch.load(
        weights_path,
        map_location="cpu",
        weights_only=False,
    )

    gnn.load_state_dict(state_dict)
    gnn.eval()

    # ------------------------------------------------------------
    # 4. Extract the exact representation used by TEA-GLM
    # ------------------------------------------------------------
    with torch.no_grad():
        z = gnn(
            graph.x.cpu(),
            graph.edge_index.cpu(),
        )

    print("\nRepresentation")
    print("z shape:", tuple(z.shape))
    print("z dtype:", z.dtype)

    # Basic sanity checks
    print("NaNs:", torch.isnan(z).sum().item())
    print("Infs:", torch.isinf(z).sum().item())
    print(
        "Mean embedding norm:",
        z.norm(dim=1).mean().item(),
    )

    # ------------------------------------------------------------
    # 5. Same node IDs, same labels
    # ------------------------------------------------------------
    X = z.cpu().numpy()
    y = graph.y.cpu().numpy()

    X_train = X[train_ids.numpy()]
    X_val = X[val_ids.numpy()]
    X_test = X[test_ids.numpy()]

    y_train = y[train_ids.numpy()]
    y_val = y[val_ids.numpy()]
    y_test = y[test_ids.numpy()]

    # ------------------------------------------------------------
    # 6. Linear probe
    # ------------------------------------------------------------
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
    print("RESULTS")
    print("=" * 70)
    print(f"Train accuracy: {train_acc:.6f}")
    print(f"Val accuracy:   {val_acc:.6f}")
    print(f"Test accuracy:  {test_acc:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()