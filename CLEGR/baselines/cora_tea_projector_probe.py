
import json
import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.data import get_dataset
from src.models import GraphSAGE


SEED = 0

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

print("Split sizes:")
print("Train:", len(train_ids))
print("Val:", len(val_ids))
print("Test:", len(test_ids))


# ------------------------------------------------------------
# 2. Load exact pretrained TEA GraphSAGE
# ------------------------------------------------------------
pretrain_base = "models/tea-glm/pretrain_cora_seed_0_phi-1_5"

with open(pretrain_base + "_model_params.json", "r") as f:
    param_dict = json.load(f)

gnn = GraphSAGE(**param_dict)

gnn_state = torch.load(
    pretrain_base + "_model.pth",
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

print("\nGNN embedding shape:", tuple(z.shape))


# ------------------------------------------------------------
# 4. Load trained downstream projector
# ------------------------------------------------------------
ckpt_path = "models/tea-glm/projector_phi-1.5_cora_seed_0.ckpt"

ckpt = torch.load(
    ckpt_path,
    map_location="cpu",
    weights_only=False,
)

state_dict = ckpt["state_dict"]

W = state_dict["model.projector.0.weight"]
b = state_dict["model.projector.0.bias"]

print("Projector weight shape:", tuple(W.shape))
print("Projector bias shape:", tuple(b.shape))


# ------------------------------------------------------------
# 5. Apply exact learned projector
# ------------------------------------------------------------
with torch.no_grad():
    projected_flat = torch.nn.functional.linear(z, W, b)

print("\nFlat projected shape:", tuple(projected_flat.shape))

# TEA-GML uses 10 graph tokens
n_gnn_toks = 10
token_dim = projected_flat.shape[1] // n_gnn_toks

projected_tokens = projected_flat.view(
    projected_flat.shape[0],
    n_gnn_toks,
    token_dim,
)

print("Projected token shape:", tuple(projected_tokens.shape))

# For linear probing, flatten all 10 graph tokens
X_proj = projected_tokens.reshape(
    projected_tokens.shape[0],
    -1,
).numpy()

y = graph.y.cpu().numpy()


# ------------------------------------------------------------
# 6. Linear probe on projector output
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

print("\n" + "=" * 60)
print("PROJECTOR LINEAR PROBE")
print("=" * 60)
print(f"Train accuracy: {train_acc:.6f}")
print(f"Val accuracy:   {val_acc:.6f}")
print(f"Test accuracy:  {test_acc:.6f}")
print("=" * 60)