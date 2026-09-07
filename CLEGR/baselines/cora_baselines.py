# Majority class baseline

# ============================================================
# CORA BASELINE 1: MAJORITY CLASS
# ============================================================

from collections import Counter
from src.utils import split_dataset

SEED = 0

from src.data.cora import CoraDataset
cora = CoraDataset(root="datasets/cora", force_reload=False)

train_dataset, val_dataset, test_dataset = split_dataset(
    cora,
    SEED,
)

print("=== SPLIT SIZES ===")
print("Train:", len(train_dataset))
print("Val:  ", len(val_dataset))
print("Test: ", len(test_dataset))


# ------------------------------------------------------------
# Get labels
# ------------------------------------------------------------

def get_labels(dataset):
    labels = []

    for i in range(len(dataset)):
        label = dataset[i].label

        if isinstance(label, (list, tuple)):
            label = label[0]

        labels.append(str(label).strip())

    return labels


train_labels = get_labels(train_dataset)
val_labels = get_labels(val_dataset)
test_labels = get_labels(test_dataset)


# ------------------------------------------------------------
# Find majority class using TRAIN ONLY
# ------------------------------------------------------------

train_counts = Counter(train_labels)

majority_class, majority_count = train_counts.most_common(1)[0]

print("\n=== TRAIN CLASS DISTRIBUTION ===")

for label, count in train_counts.most_common():
    print(
        f"{label:25s}: "
        f"{count:4d} "
        f"({count / len(train_labels):.4f})"
    )


print("\nMajority class:", majority_class)
print(
    "Train majority proportion:",
    f"{majority_count / len(train_labels):.4f}"
)

# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------

def majority_accuracy(labels, majority_label):
    correct = sum(
        label == majority_label
        for label in labels
    )

    return correct / len(labels)


train_acc = majority_accuracy(
    train_labels,
    majority_class,
)

val_acc = majority_accuracy(
    val_labels,
    majority_class,
)

test_acc = majority_accuracy(
    test_labels,
    majority_class,
)


print("\n=== MAJORITY BASELINE RESULTS ===")
print(f"Train accuracy: {train_acc:.4f}")
print(f"Val accuracy:   {val_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")

# Logistic Regression

# ============================================================
# CORA BASELINE 2: LOGISTIC REGRESSION
# Input: PCA-500 node features only
# ============================================================

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.utils import split_dataset

SEED = 0

# ------------------------------------------------------------
# Same split used by training.py
# ------------------------------------------------------------

train_dataset, val_dataset, test_dataset = split_dataset(
    cora,
    SEED,
)

print("=== SPLIT SIZES ===")
print("Train:", len(train_dataset))
print("Val:  ", len(val_dataset))
print("Test: ", len(test_dataset))

# ------------------------------------------------------------
# Get node indices for each split
# ------------------------------------------------------------

def get_node_ids(dataset):
    ids = []

    for sample in dataset:
        node_id = sample.node_id

        if hasattr(node_id, "item"):
            node_id = int(node_id.item())
        else:
            node_id = int(node_id)

        ids.append(node_id)

    return np.array(ids)


train_idx = get_node_ids(train_dataset)
val_idx = get_node_ids(val_dataset)
test_idx = get_node_ids(test_dataset)

# ------------------------------------------------------------
# Extract graph features + numeric labels
# ------------------------------------------------------------

X = cora.graph_data.x.cpu().numpy()
y = cora.graph_data.y.cpu().numpy()

X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

X_test = X[test_idx]
y_test = y[test_idx]

print("\n=== DATA SHAPES ===")
print("X_train:", X_train.shape)
print("X_val:  ", X_val.shape)
print("X_test: ", X_test.shape)

# ------------------------------------------------------------
# Train Logistic Regression
# ------------------------------------------------------------

clf = LogisticRegression(
    max_iter=5000,
    random_state=SEED,
)

clf.fit(X_train, y_train)

# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

train_pred = clf.predict(X_train)
val_pred = clf.predict(X_val)
test_pred = clf.predict(X_test)

# ------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------

train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)
test_acc = accuracy_score(y_test, test_pred)

print("\n=== LOGISTIC REGRESSION RESULTS ===")
print(f"Train accuracy: {train_acc:.4f}")
print(f"Val accuracy:   {val_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")

# MLP

# ============================================================
# CORA BASELINE 3: MLP
# Input: PCA-500 node features only
# ============================================================

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import split_dataset

SEED = 0
# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# ------------------------------------------------------------
# Same split used by training.py
# ------------------------------------------------------------

train_dataset, val_dataset, test_dataset = split_dataset(
    cora,
    SEED,
)

def get_node_ids(dataset):
    ids = []

    for sample in dataset:
        node_id = sample.node_id

        if hasattr(node_id, "item"):
            node_id = int(node_id.item())
        else:
            node_id = int(node_id)

        ids.append(node_id)

    return torch.tensor(ids, dtype=torch.long)

train_idx = get_node_ids(train_dataset)
val_idx = get_node_ids(val_dataset)
test_idx = get_node_ids(test_dataset)

# ------------------------------------------------------------
# Features and labels
# ------------------------------------------------------------

x = cora.graph_data.x.to(device)
y = cora.graph_data.y.to(device)

train_idx = train_idx.to(device)
val_idx = val_idx.to(device)
test_idx = test_idx.to(device)

print("\n=== DATA ===")
print("Features:", x.shape)
print("Classes:", int(y.max().item()) + 1)

print(
    "Split:",
    len(train_idx),
    len(val_idx),
    len(test_idx),
)

# ------------------------------------------------------------
# MLP
# ------------------------------------------------------------

class MLP(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.5,
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            in_channels,
            hidden_channels,
        )

        self.fc2 = nn.Linear(
            hidden_channels,
            out_channels,
        )

        self.dropout = dropout

    def forward(self, x):

        x = self.fc1(x)
        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.fc2(x)

        return x

model = MLP(
    in_channels=x.size(1),
    hidden_channels=64,
    out_channels=int(y.max().item()) + 1,
    dropout=0.5,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4,
)

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

@torch.no_grad()
def accuracy(index):

    model.eval()

    logits = model(x)

    predictions = logits[index].argmax(dim=1)

    return (
        predictions == y[index]
    ).float().mean().item()

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

best_val_acc = -1.0
best_state = None
best_epoch = None

MAX_EPOCHS = 500
PATIENCE = 50

epochs_without_improvement = 0

for epoch in range(1, MAX_EPOCHS + 1):

    model.train()

    optimizer.zero_grad()

    logits = model(x)

    loss = F.cross_entropy(
        logits[train_idx],
        y[train_idx],
    )

    loss.backward()
    optimizer.step()

    val_acc = accuracy(val_idx)

    if val_acc > best_val_acc:

        best_val_acc = val_acc
        best_epoch = epoch

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        epochs_without_improvement = 0

    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 25 == 0:

        train_acc = accuracy(train_idx)

        print(
            f"Epoch {epoch:3d} | "
            f"Loss {loss.item():.4f} | "
            f"Train {train_acc:.4f} | "
            f"Val {val_acc:.4f}"
        )

    if epochs_without_improvement >= PATIENCE:
        print(
            f"\nEarly stopping at epoch {epoch}"
        )
        break

# ------------------------------------------------------------
# Restore best validation model
# ------------------------------------------------------------

model.load_state_dict(best_state)

# ------------------------------------------------------------
# Final results
# ------------------------------------------------------------

train_acc = accuracy(train_idx)
val_acc = accuracy(val_idx)
test_acc = accuracy(test_idx)

print("\n=== MLP RESULTS ===")
print("Best epoch:", best_epoch)
print(f"Train accuracy: {train_acc:.4f}")
print(f"Val accuracy:   {val_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")

# GCN

# ============================================================
# CORA BASELINE 4: GCN
# Input: PCA-500 node features + graph topology
# ============================================================

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from src.utils import split_dataset

SEED = 0


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ------------------------------------------------------------
# Same split used by training.py
# ------------------------------------------------------------

train_dataset, val_dataset, test_dataset = split_dataset(
    cora,
    SEED,
)


def get_node_ids(dataset):

    ids = []

    for sample in dataset:

        node_id = sample.node_id

        if hasattr(node_id, "item"):
            node_id = int(node_id.item())
        else:
            node_id = int(node_id)

        ids.append(node_id)

    return torch.tensor(
        ids,
        dtype=torch.long,
    )


train_idx = get_node_ids(train_dataset).to(device)
val_idx = get_node_ids(val_dataset).to(device)
test_idx = get_node_ids(test_dataset).to(device)


# ------------------------------------------------------------
# Graph
# ------------------------------------------------------------

x = cora.graph_data.x.to(device)
y = cora.graph_data.y.to(device)
edge_index = cora.graph_data.edge_index.to(device)

num_classes = int(y.max().item()) + 1

print("\n=== DATA ===")
print("Features:", x.shape)
print("Edges:", edge_index.shape)
print("Classes:", num_classes)

print(
    "Split:",
    len(train_idx),
    len(val_idx),
    len(test_idx),
)


# ------------------------------------------------------------
# GCN
# ------------------------------------------------------------

class GCN(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.5,
    ):
        super().__init__()

        self.conv1 = GCNConv(
            in_channels,
            hidden_channels,
        )

        self.conv2 = GCNConv(
            hidden_channels,
            out_channels,
        )

        self.dropout = dropout

    def forward(
        self,
        x,
        edge_index,
    ):

        x = self.conv1(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(
            x,
            edge_index,
        )

        return x


model = GCN(
    in_channels=x.size(1),
    hidden_channels=64,
    out_channels=num_classes,
    dropout=0.5,
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4,
)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

@torch.no_grad()
def accuracy(index):

    model.eval()

    logits = model(
        x,
        edge_index,
    )

    predictions = logits[index].argmax(dim=1)

    return (
        predictions == y[index]
    ).float().mean().item()


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

best_val_acc = -1.0
best_state = None
best_epoch = None

MAX_EPOCHS = 500
PATIENCE = 50

epochs_without_improvement = 0


for epoch in range(1, MAX_EPOCHS + 1):

    model.train()

    optimizer.zero_grad()

    logits = model(
        x,
        edge_index,
    )

    loss = F.cross_entropy(
        logits[train_idx],
        y[train_idx],
    )

    loss.backward()
    optimizer.step()

    val_acc = accuracy(val_idx)

    if val_acc > best_val_acc:

        best_val_acc = val_acc
        best_epoch = epoch

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

    if epoch == 1 or epoch % 25 == 0:

        train_acc = accuracy(train_idx)

        print(
            f"Epoch {epoch:3d} | "
            f"Loss {loss.item():.4f} | "
            f"Train {train_acc:.4f} | "
            f"Val {val_acc:.4f}"
        )

    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping at epoch {epoch}"
        )

        break


# ------------------------------------------------------------
# Restore best validation model
# ------------------------------------------------------------

model.load_state_dict(best_state)


# ------------------------------------------------------------
# Final results
# ------------------------------------------------------------

train_acc = accuracy(train_idx)
val_acc = accuracy(val_idx)
test_acc = accuracy(test_idx)


print("\n=== GCN RESULTS ===")
print("Best epoch:", best_epoch)
print(f"Train accuracy: {train_acc:.4f}")
print(f"Val accuracy:   {val_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")

# GraphSAGE

# ============================================================
# CORA BASELINE 5: GRAPHSAGE
# Input: PCA-500 node features + graph topology
# Same split as TEA-GLM
# ============================================================

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv
from src.utils import split_dataset

SEED = 0


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ------------------------------------------------------------
# Exact same split
# ------------------------------------------------------------

train_dataset, val_dataset, test_dataset = split_dataset(
    cora,
    SEED,
)


def get_node_ids(dataset):

    ids = []

    for sample in dataset:

        node_id = sample.node_id

        if hasattr(node_id, "item"):
            node_id = int(node_id.item())
        else:
            node_id = int(node_id)

        ids.append(node_id)

    return torch.tensor(ids, dtype=torch.long)


train_idx = get_node_ids(train_dataset).to(device)
val_idx = get_node_ids(val_dataset).to(device)
test_idx = get_node_ids(test_dataset).to(device)


# ------------------------------------------------------------
# Graph
# ------------------------------------------------------------

x = cora.graph_data.x.to(device)
y = cora.graph_data.y.to(device)
edge_index = cora.graph_data.edge_index.to(device)

num_classes = int(y.max().item()) + 1

print("\n=== DATA ===")
print("Features:", x.shape)
print("Edges:", edge_index.shape)
print("Classes:", num_classes)
print(
    "Split:",
    len(train_idx),
    len(val_idx),
    len(test_idx),
)


# ------------------------------------------------------------
# GraphSAGE
# ------------------------------------------------------------

class GraphSAGE(nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        dropout=0.5,
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            in_channels,
            hidden_channels,
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            out_channels,
        )

        self.dropout = dropout

    def forward(
        self,
        x,
        edge_index,
    ):

        x = self.conv1(
            x,
            edge_index,
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )

        x = self.conv2(
            x,
            edge_index,
        )

        return x


model = GraphSAGE(
    in_channels=x.size(1),
    hidden_channels=64,
    out_channels=num_classes,
    dropout=0.5,
).to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4,
)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

@torch.no_grad()
def accuracy(index):

    model.eval()

    logits = model(
        x,
        edge_index,
    )

    predictions = logits[index].argmax(dim=1)

    return (
        predictions == y[index]
    ).float().mean().item()


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

best_val_acc = -1.0
best_state = None
best_epoch = None

MAX_EPOCHS = 500
PATIENCE = 50

epochs_without_improvement = 0


for epoch in range(1, MAX_EPOCHS + 1):

    model.train()

    optimizer.zero_grad()

    logits = model(
        x,
        edge_index,
    )

    loss = F.cross_entropy(
        logits[train_idx],
        y[train_idx],
    )

    loss.backward()
    optimizer.step()

    val_acc = accuracy(val_idx)

    if val_acc > best_val_acc:

        best_val_acc = val_acc
        best_epoch = epoch

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        epochs_without_improvement = 0

    else:

        epochs_without_improvement += 1

    if epoch == 1 or epoch % 25 == 0:

        train_acc = accuracy(train_idx)

        print(
            f"Epoch {epoch:3d} | "
            f"Loss {loss.item():.4f} | "
            f"Train {train_acc:.4f} | "
            f"Val {val_acc:.4f}"
        )

    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping at epoch {epoch}"
        )

        break


# ------------------------------------------------------------
# Restore best validation model
# ------------------------------------------------------------

model.load_state_dict(best_state)


# ------------------------------------------------------------
# Final results
# ------------------------------------------------------------

train_acc = accuracy(train_idx)
val_acc = accuracy(val_idx)
test_acc = accuracy(test_idx)


print("\n=== GRAPHSAGE RESULTS ===")
print("Best epoch:", best_epoch)
print(f"Train accuracy: {train_acc:.4f}")
print(f"Val accuracy:   {val_acc:.4f}")
print(f"Test accuracy:  {test_acc:.4f}")