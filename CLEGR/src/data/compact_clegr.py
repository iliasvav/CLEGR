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

    
    def split(
        self,
        seed,
        split_ratio=(0.6, 0.2, 0.2),
        graph_wise=False,
    ):
        rng = np.random.RandomState(seed)

        if not graph_wise:
            # Original behavior: split individual questions.
            indices = rng.permutation(len(self))

            train_end = int(len(self) * split_ratio[0])
            val_end = int(
                len(self) * (split_ratio[0] + split_ratio[1])
            )

            return {
                "train": indices[:train_end],
                "val": indices[train_end:val_end],
                "test": indices[val_end:],
            }

        graph_ids = list(self.graph_bank.keys())
        shuffled_graphs = rng.permutation(graph_ids)

        train_end = int(len(graph_ids) * split_ratio[0])
        val_end = int(
            len(graph_ids) * (split_ratio[0] + split_ratio[1])
        )

        train_graphs = set(shuffled_graphs[:train_end])
        val_graphs = set(shuffled_graphs[train_end:val_end])
        test_graphs = set(shuffled_graphs[val_end:])

        splits = {"train": [], "val": [], "test": []}

        for idx, record in enumerate(self.qa_records):
            graph_id = record["graph_id"]

            if graph_id in train_graphs:
                splits["train"].append(idx)
            elif graph_id in val_graphs:
                splits["val"].append(idx)
            else:
                assert graph_id in test_graphs
                splits["test"].append(idx)

        return {
            name: np.asarray(indices, dtype=np.int64)
            for name, indices in splits.items()
        }
