"""import os
from typing import Any, List
import numpy as np
import torch
import pandas as pd
from torch_geometric.data.data import BaseData
from torch_geometric.data.dataset import Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from src.data.base import BaseDataset

class CoraDataset(BaseDataset):
    def __init__(self, root: str, transform=None, force_reload: bool = False, preprocessor_llm: Any = None):
        super().__init__(dataset_name="cora", root=root, transform=transform, force_reload=force_reload)
        self.type = "node"
        self.preprocessor_llm = preprocessor_llm
        self.load(self.processed_paths[1])
        self.graph_data = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self) -> List[str]:
        return ["graph_data.pt", "data_list.pt"]

    def process(self):
        raw_path = os.path.join(self.root, "cora_raw")
        assert os.path.exists(raw_path), f"Raw data not found at {raw_path}"

        dataset = Planetoid(root=os.path.join(self.root, "cora_pyg"), name="Cora", transform=self.transform)
        graph: Dataset | BaseData = dataset[0]

        path = os.path.join(self.root, "cora_raw", "cora.content")
        df_tmp = pd.read_csv(path, sep='\t', header=None, dtype=str)
        idx_features_labels = df_tmp.values
        original_ids = idx_features_labels[:, 0]

        duplicate_groups = [[137, 2144], [283, 1260], [366, 1127, 1995], [503, 574, 2348], [995, 1338, 2102, 2103], [709, 1897], [950, 1495], [1020, 2076], [1040, 1719, 1720], [1033, 1586], [1031, 2205]]
        nodes_to_remove = [node for group in duplicate_groups for node in group[1:]]
        mask = torch.ones(graph.num_nodes, dtype=torch.bool)
        mask[nodes_to_remove] = False

        edge_mask = mask[graph.edge_index[0]] & mask[graph.edge_index[1]]
        graph.edge_index = graph.edge_index[:, edge_mask]
        new_indices = -torch.ones(graph.num_nodes, dtype=torch.long)
        new_indices[mask] = torch.arange(mask.sum())
        graph.edge_index = new_indices[graph.edge_index]

        graph.x = graph.x[mask]
        graph.y = graph.y[mask]
        graph.train_mask = graph.train_mask[mask]
        graph.val_mask = graph.val_mask[mask]
        graph.test_mask = graph.test_mask[mask]
        graph.num_nodes = mask.sum()
        
        if graph.x.shape[1] > 500:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=500)
            graph.x = torch.tensor(pca.fit_transform(graph.x)).to(torch.float)
            
        torch.save(graph, self.processed_paths[0])
        common_y = graph.y

        with open(os.path.join(self.root, "cora_raw/mccallum/cora/papers")) as f:
            lines = f.readlines()

        pid_filename = {}
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                numeric_id = parts[0]
                url_str = parts[1].strip()
                pid_filename[numeric_id] = url_str
                pid_filename[url_str] = url_str

        class_names = ["theory", "reinforcement learning", "genetic algorithms", "neural networks", "probabilistic methods", "case based", "rule learning"]
        data_list = []
        QUESTION = "Answer the following question: Which subcategory does this paper belong to? Please only output the most likely answer from the following subcategories and nothing else: theory, reinforcement learning, genetic algorithms, neural networks, probabilistic methods, case based, rule learning.\\nAnswer: "

        current_idx = 0
        for i in range(min(len(original_ids), graph.num_nodes)):
            if not mask[i]:
                continue

            original_id = original_ids[i]
            try:
                fn = None
                if original_id in pid_filename:
                    fn = pid_filename[original_id]
                elif original_id.strip() in pid_filename:
                    fn = pid_filename[original_id.strip()]
                
                lines = []
                if fn:
                    file_path = os.path.join(self.root, "cora_raw/mccallum/cora/extractions", fn)
                    if os.path.exists(file_path):
                        with open(file_path, 'r', errors='ignore') as f:
                            lines = f.read().splitlines()
                    else:
                        base_fn = fn.split('#')[-1]
                        extractions_dir = os.path.join(self.root, "cora_raw/mccallum/cora/extractions")
                        for possible_file in os.listdir(extractions_dir):
                            if base_fn in possible_file:
                                with open(os.path.join(extractions_dir, possible_file), 'r', errors='ignore') as f:
                                    lines = f.read().splitlines()
                                break

                title, abstract = "", ""
                if fn and lines:
                    for line in lines:
                        if "Title:" in line: title = line[7:]
                        if "Abstract:" in line: abstract = line[10:]

                context = {
                    "Title": title.strip() if title else "No available title",
                    "Abstract": abstract.strip() if abstract else "No available abstract"
                }

                data_list.append(Data(
                    id=original_id, question=QUESTION, context=context,
                    label=class_names[common_y[current_idx].item()], node_id=current_idx
                ))
                current_idx += 1
            except Exception:
                data_list.append(Data(
                    id=original_id, question=QUESTION, 
                    context={"Title": "No available title", "Abstract": "No available abstract"},
                    label=class_names[common_y[current_idx].item()], node_id=current_idx
                ))
                current_idx += 1

        self.save(data_list, self.processed_paths[1])
        
    def split(self, seed, split_ratio: List[float] = [0.6, 0.2, 0.2]):
        num_samples = len(self)
        indices = np.random.RandomState(seed).permutation(num_samples)
        train_split = int(num_samples * split_ratio[0])
        val_split = int(num_samples * (split_ratio[0] + split_ratio[1]))
        return {
            "train": indices[:train_split],
            "val": indices[train_split:val_split],
            "test": indices[val_split:],
        }"""

import os
from typing import Any, List

import numpy as np
import torch

from torch_geometric.data.data import BaseData
from torch_geometric.data.dataset import Dataset
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

from src.data.base import BaseDataset


class CoraDataset(BaseDataset):
    def __init__(
        self,
        root: str,
        transform=None,
        force_reload: bool = False,
        preprocessor_llm: Any = None,
    ):
        super().__init__(
            dataset_name="cora",
            root=root,
            transform=transform,
            force_reload=force_reload,
        )

        self.type = "node"
        self.preprocessor_llm = preprocessor_llm

        self.load(self.processed_paths[1])
        self.graph_data = torch.load(
            self.processed_paths[0],
            weights_only=False,
        )

    @property
    def processed_file_names(self) -> List[str]:
        return [
            "graph_data.pt",
            "data_list.pt",
        ]

    def process(self):

        # -----------------------------------------------------
        # 1. Load canonical PyG / Planetoid Cora
        # -----------------------------------------------------

        dataset = Planetoid(
            root=os.path.join(self.root, "cora_pyg"),
            name="Cora",
            transform=self.transform,
        )

        graph: Dataset | BaseData = dataset[0]

        print("\nLoaded canonical Cora")
        print("Nodes:", graph.num_nodes)
        print("Features:", graph.x.shape)
        print("Labels:", graph.y.shape)
        print("Edges:", graph.edge_index.shape)

        # -----------------------------------------------------
        # 2. Keep existing PCA preprocessing
        # -----------------------------------------------------

        if graph.x.shape[1] > 500:
            from sklearn.decomposition import PCA

            print(
                f"Applying PCA: "
                f"{graph.x.shape[1]} -> 500 dimensions"
            )

            pca = PCA(n_components=500)

            graph.x = torch.tensor(
                pca.fit_transform(graph.x.cpu().numpy()),
                dtype=torch.float,
            )

        # -----------------------------------------------------
        # 3. Save graph
        # -----------------------------------------------------

        torch.save(
            graph,
            self.processed_paths[0],
        )

        # -----------------------------------------------------
        # 4. Class names
        # -----------------------------------------------------

        class_names = [
            "theory",
            "reinforcement learning",
            "genetic algorithms",
            "neural networks",
            "probabilistic methods",
            "case based",
            "rule learning",
        ]

        # -----------------------------------------------------
        # 5. GLM classification question
        # -----------------------------------------------------

        QUESTION = (
            "Answer the following question: "
            "Which subcategory does this paper belong to? "
            "Please only output the most likely answer from "
            "the following subcategories and nothing else: "
            "theory, reinforcement learning, genetic algorithms, "
            "neural networks, probabilistic methods, case based, "
            "rule learning.\n"
            "Answer: "
        )

        # -----------------------------------------------------
        # 6. Construct exactly one sample per PyG node
        #
        # node_id now refers directly to the PyG node.
        # -----------------------------------------------------

        data_list = []

        for node_id in range(graph.num_nodes):

            data_list.append(
                Data(
                    id=str(node_id),

                    question=QUESTION,

                    context={
                        "Title": "",
                        "Abstract": "",
                    },

                    label=class_names[
                        graph.y[node_id].item()
                    ],

                    node_id=node_id,
                )
            )

        # -----------------------------------------------------
        # 7. Sanity checks
        # -----------------------------------------------------

        assert len(data_list) == graph.num_nodes, (
            f"Mismatch: {len(data_list)} samples "
            f"but graph has {graph.num_nodes} nodes."
        )

        assert graph.num_nodes == 2708, (
            f"Expected canonical Cora to contain 2708 nodes, "
            f"got {graph.num_nodes}."
        )

        assert graph.y.shape[0] == graph.num_nodes

        print("\nCora preprocessing complete")
        print("Graph nodes:", graph.num_nodes)
        print("Data samples:", len(data_list))
        print("Feature shape:", graph.x.shape)
        print("No McCallum title/abstract data is being used.")

        self.save(
            data_list,
            self.processed_paths[1],
        )

    def split(
        self,
        seed,
        split_ratio: List[float] = [0.6, 0.2, 0.2],
    ):
        num_samples = len(self)

        indices = np.random.RandomState(
            seed
        ).permutation(num_samples)

        train_split = int(
            num_samples * split_ratio[0]
        )

        val_split = int(
            num_samples
            * (
                split_ratio[0]
                + split_ratio[1]
            )
        )

        return {
            "train": indices[:train_split],
            "val": indices[
                train_split:val_split
            ],
            "test": indices[val_split:],
        }