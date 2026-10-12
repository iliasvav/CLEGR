import torch
import numpy as np

from torch.utils.data import Dataset
from torch_geometric.data import Data


class CompactCLEGRDataset(Dataset):
    def __init__(
        self,
        root="datasets",
        size="small",
        seed=42,
        transform=None,
    ):
        if size != "small":
            raise ValueError(
                "Only the small CLEGR dataset has been compacted."
            )

        path = f"{root}/clegr-reasoning/processed/compact_fixed.pt"

        saved = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        self.graph_bank = saved["graph_bank"]
        self.qa_records = saved["qa_records"]
        self.seed = seed
        self.transform = transform
        self.type = "graph"

    def __len__(self):
        return len(self.qa_records)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()

        record = self.qa_records[idx]
        graph = self.graph_bank[record["graph_id"]]

        data = Data(**graph)

        for key, value in record.items():
            setattr(data, key, value)

        if self.transform is not None:
            data = self.transform(data)

        return data

    
    
    def split(self, seed, split_ratio=(0.6, 0.2, 0.2)):
        graph_ids = [
            record["graph_id"]
            for record in self.qa_records
        ]

        unique_graph_ids = np.unique(graph_ids)

        rng = np.random.RandomState(seed)
        shuffled_graph_ids = rng.permutation(unique_graph_ids)

        n = len(shuffled_graph_ids)
        train_end = int(n * split_ratio[0])
        val_end = int(n * (split_ratio[0] + split_ratio[1]))

        train_graphs = set(shuffled_graph_ids[:train_end])
        val_graphs = set(shuffled_graph_ids[train_end:val_end])
        test_graphs = set(shuffled_graph_ids[val_end:])

        train_indices = [
            i for i, gid in enumerate(graph_ids)
            if gid in train_graphs
        ]
        val_indices = [
            i for i, gid in enumerate(graph_ids)
            if gid in val_graphs
        ]
        test_indices = [
            i for i, gid in enumerate(graph_ids)
            if gid in test_graphs
        ]

        return {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }

