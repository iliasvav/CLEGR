
import torch
from torch.utils.data import Dataset
import hashlib

def drop_edges_undirected(data, drop_prob, seed=42):
    """Drop undirected edge pairs while preserving edge attributes."""
    if not 0.0 <= drop_prob <= 1.0:
        raise ValueError("drop_prob must be between 0 and 1")

    data = data.clone()

    if drop_prob == 0.0 or data.edge_index.size(1) == 0:
        return data

    row, col = data.edge_index.cpu()

    # Map (u,v) and (v,u) to the same undirected pair.
    pairs = torch.stack([
        torch.minimum(row, col),
        torch.maximum(row, col)
    ], dim=1)

    unique_pairs, inverse = torch.unique(
        pairs, dim=0, return_inverse=True
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    # One random decision per undirected pair.
    keep_pairs = (
        torch.rand(len(unique_pairs), generator=generator)
        >= drop_prob
    )

    # Apply that decision to both directions.
    keep_mask = keep_pairs[inverse].to(data.edge_index.device)

    data.edge_index = data.edge_index[:, keep_mask]

    if getattr(data, "edge_attr", None) is not None:
        if data.edge_attr.size(0) != keep_mask.numel():
            raise ValueError(
                "edge_attr and edge_index have inconsistent lengths: "
                f"{data.edge_attr.size(0)} vs {keep_mask.numel()}"
            )

        data.edge_attr = data.edge_attr[keep_mask]

    return data


class EdgeDropDataset(Dataset):
    def __init__(self, dataset, drop_prob, seed=42):
        self.dataset = dataset
        self.drop_prob = drop_prob
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]

        # Generate a stable seed using the graph ID.
        graph_id = str(data.graph_id)

        digest = hashlib.sha256(
            f"{self.seed}:{graph_id}".encode("utf-8")
        ).digest()

        graph_seed = int.from_bytes(
            digest[:8], byteorder="big"
        ) % (2**63 - 1)

        return drop_edges_undirected(
            data,
            drop_prob=self.drop_prob,
            seed=graph_seed,
        )

