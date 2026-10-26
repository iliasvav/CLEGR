import json
import os
from typing import Optional
import torch
import lightning as pl

from torch_geometric.data import Batch, Data
import torch.nn.functional as F

from src.modifiers import Modifier
from src.models.glm.base import BaseGLM
from src.models import GraphSAGE
from src.models.llm.llm_wrapper import LLM


class TEA_GLM(BaseGLM):
    r"""
    Code adapted from TEA-GLM

    Args:
        llm (LLM): The language model to use.
        gnn (torch.nn.Module): The graph neural network to use.
        use_lora (bool): Whether to use LORA or not.
        projector_out_channels (int): The number of output channels of the projector.
        n_gnn_toks (int): The number of GNN tokens to use in the LLM prompt.
    """

    def __init__(
        self,
        llm: LLM,
        projector_out_channels: int,
        use_lora: bool = False,
        n_gnn_toks: int = 1,
        pretrain_path: str = None,
        shuffle_graph_tokens: bool = False,
        shuffle_seed: int = 42,
        pretrain_llm_name: str = None,
    ) -> None:
        self.shuffle_graph_tokens = shuffle_graph_tokens
        self.shuffle_seed = shuffle_seed
        llm_name = (
            pretrain_llm_name if pretrain_llm_name is not None else os.path.basename(llm.model_name)
        )
        with open(f"{pretrain_path}_model_params.json", "r") as f:
            param_dict = json.load(f)

        if pretrain_path is not None and "clegr" in pretrain_path.lower():
            param_dict["task_level"] = "graph_classification"
        else:
            param_dict.setdefault("task_level", "node_classification")
        gnn = GraphSAGE(**param_dict).to(llm.device)

        state_dict = torch.load(
            f"{pretrain_path}_model.pth", weights_only=False
        )
        gnn.load_state_dict(state_dict)

        super().__init__(llm, gnn, use_lora, projector_out_channels, n_gnn_toks)

        self.prepare_projector()

    def prepare_projector(self):
        projector_in_channels = self.gnn.n_classes
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(
                projector_in_channels, self.projector_out_channels * self.n_gnn_toks
            ),
            # torch.nn.ReLU(),
            torch.nn.Unflatten(-1, (self.n_gnn_toks, self.projector_out_channels)),
        ).to(self.llm.device)

    def encode_graph(self, data):
        x = self.gnn(data.x, data.edge_index, batch=data.batch)
        if self.gnn.task_level == "graph_classification":
            expected_graphs = int(data.batch.max().item()) + 1
            assert x.size(0) == expected_graphs, (
                f"Expected {expected_graphs} graph embeddings, "
                f"but got {x.size(0)}."
            )
        if self.shuffle_graph_tokens:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.shuffle_seed)
            perm = torch.randperm(
                x.size(0),
                generator=generator,
                device="cpu",
            ).to(x.device)
            x = x[perm]
            if not hasattr(self, "_shuffle_message_printed"):
                print(
                    f"[TEA diagnostic] Shuffling GNN node representations "
                    f"with seed={self.shuffle_seed}"
                )
                self._shuffle_message_printed = True
        return x

    def forward(
        self,
        data: Batch,
        graph_data: Optional[Data] = None,
    ):
        self.llm.eval()

        data = data.to(self.llm.device)

        node_id = data.node_id if hasattr(data, "node_id") else None
        context = data.context if hasattr(data, "context") else None

        if graph_data is not None:
            x = self.encode_graph(graph_data)
        else:
            x = self.encode_graph(data)
        if not torch.isfinite(x).all():
            raise RuntimeError("Non-finite GNN embeddings")

        # filter based on node_id
        if node_id is not None:
            x = x[node_id]
        x = self.projector(x)
        if not torch.isfinite(x).all():
            raise RuntimeError("Non-finite projector outputs")
        # print(x.shape)
        xs = [hi.squeeze(0) for hi in x.split(1, dim=0)]
        # print(xs)

        # convert context to text from dict
        con = self.get_context(data)

        # Handle questions without node features:
        # batch_unique = data.batch.unique()
        # batch_size = len(data.question)
        # if len(batch_unique) < batch_size:
        #     xs = [xs[i] if i in batch_unique else None for i in range(batch_size)]

        # print(xs.shape)
        (
            inputs_embeds,
            attention_mask,
            label_input_ids,
        ) = self.llm._get_embeds(data.question, con, xs, data.label)

        """with self.llm.autocast_context:
            outputs = self.llm_generator(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss"""

        answer_lengths = (label_input_ids != -100).sum(dim=1)
        max_answer_len = int(answer_lengths.max().item())
        assert max_answer_len > 0, "No supervised answer tokens found."
        base_model = self.llm_generator.get_base_model()

        with self.llm.autocast_context:
            hidden_states = base_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state

            answer_hidden = hidden_states[:, -(max_answer_len + 1):-1, :]
            answer_logits = base_model.lm_head(answer_hidden)
            answer_targets = label_input_ids[:, -max_answer_len:]

            def check_finite(name, tensor):
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(
                        f"Non-finite values detected in {name}: "
                        f"NaNs={torch.isnan(tensor).sum().item()}, "
                        f"Infs={torch.isinf(tensor).sum().item()}"
                    )
            check_finite("answer_hidden", answer_hidden)
            check_finite("answer_logits", answer_logits)
            loss = F.cross_entropy(
                answer_logits.float().reshape(-1, answer_logits.size(-1)),
                answer_targets.reshape(-1),
                ignore_index=-100,
            )
            check_finite("loss", loss)

        return loss

    @torch.no_grad()
    def inference(
        self,
        data,
        max_out_tokens: int,
        graph_data: Optional[Data] = None,
        modifier: Optional[Modifier] = None,
        no_graph_token: bool = False,
    ):
        self.llm.eval()

        data = data.to(self.llm.device)

        # Extract optional fields if they exist.
        node_id = data.node_id if hasattr(data, "node_id") else None
        context = data.context if hasattr(data, "context") else None
        question = data.question
        
        if no_graph_token:
            xs = None
        else:
            if graph_data is not None:
                x = self.encode_graph(graph_data)
            else:
                x = self.encode_graph(data)

            if node_id is not None:
                x = x[node_id]

            x = self.projector(x)




            if isinstance(node_id, list):
                x = x.view(1, -1, x.shape[-1])
            xs = [hi.squeeze(0) for hi in x.split(1, dim=0)]

            batch_unique = data.batch.unique()
            batch_size = len(question)

            if len(batch_unique) < batch_size:
                xs = [
                    xs[i] if i in batch_unique else None
                    for i in range(batch_size)
                ]
        con = self.get_context(data)
        inputs_embeds, attention_mask, _ = self.llm._get_embeds(
            question, con, xs, modifier=modifier
        )
        with self.llm.autocast_context:
            outputs = self.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_length=max_out_tokens,
                return_dict_in_generate=True,
            )
        return self.llm.tokenizer.batch_decode(outputs, skip_special_tokens=True)



class NumericalDebugger:
    """Check module outputs and incoming gradients for NaN/Inf."""

    def __init__(self, model, get_step, get_rank, log_every=25):
        self.handles = []
        self.get_step = get_step
        self.get_rank = get_rank
        self.log_every = log_every

        def inspect(name, tensor, direction):
            if not isinstance(tensor, torch.Tensor):
                return
            if not tensor.is_floating_point():
                return

            with torch.no_grad():
                t = tensor.detach()
                finite = torch.isfinite(t).all().item()
                step = self.get_step()
                rank = self.get_rank()

                if not finite:
                    nan_count = torch.isnan(t).sum().item()
                    inf_count = torch.isinf(t).sum().item()
                    message = (
                        f"[NUMERIC FAILURE] step={step} rank={rank} "
                        f"direction={direction} module={name} "
                        f"dtype={t.dtype} shape={tuple(t.shape)} "
                        f"NaNs={nan_count} Infs={inf_count}"
                    )
                    if direction == "backward":
                        print(message + " [AMP may skip this step]", flush=True)
                        return
                    raise RuntimeError(message)
                if step % self.log_every == 0:
                    if (
                        name.startswith("projector.")
                        or name.startswith("llm.llm.model.layers.0.")
                        or name.startswith("llm.llm.model.norm.")
                    ):

                        f = t.float()
                        print(
                            f"[NUMERIC] rank={rank} step={step} "
                            f"{direction} {name} "
                            f"dtype={t.dtype} "
                            f"absmax={f.abs().max().item():.5g} "
                            f"mean={f.mean().item():.5g} "
                            f"std={f.std(unbiased=False).item():.5g}",
                            flush=True,
                        )

        def tensors(obj):
            if isinstance(obj, torch.Tensor):
                yield obj
            elif isinstance(obj, (tuple, list)):
                for item in obj:
                    yield from tensors(item)
            elif isinstance(obj, dict):
                for item in obj.values():
                    yield from tensors(item)

        def make_forward_hook(name):
            def hook(module, inputs, output):
                for i, t in enumerate(tensors(output)):
                    inspect(f"{name}.output[{i}]", t, "forward")
            return hook

        def make_backward_hook(name):
            def hook(module, grad_input, grad_output):
                for i, t in enumerate(tensors(grad_input)):
                    inspect(f"{name}.grad_input[{i}]", t, "backward")
                for i, t in enumerate(tensors(grad_output)):
                    inspect(f"{name}.grad_output[{i}]", t, "backward")
            return hook

        # Instrument important boundaries without registering hooks
        # on every 4-bit quantized linear layer.
        
        registered = []

        for name, module in model.named_modules():
            if (
                name == "projector"
                or name.endswith(".norm")
                or (
                    ".layers." in name
                    and name.rsplit(".", 1)[-1].isdigit()
                )
            ):
                self.handles.append(
                    module.register_forward_hook(make_forward_hook(name))
                )
                self.handles.append(
                    module.register_full_backward_hook(make_backward_hook(name))
                )
                registered.append(name)

        print(
            f"[NUMERIC SETUP] rank={self.get_rank()} "
            f"registered {len(registered)} modules",
            flush=True,
        )

        for name in registered:
            print(f"[NUMERIC SETUP] {name}", flush=True)

        if not registered:
            raise RuntimeError(
                "NumericalDebugger registered ZERO modules!"
            )


    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class TEA_GLM_LightningModule(pl.LightningModule):
    def __init__(
        self,
        model,
        lr=1e-3,
        scheduler_step_size=10,
        scheduler_gamma=0.1,
        graph_data=None,
        debug_numerics=False,
    ):
        super().__init__()
        # Save hyperparameters for logging and checkpointing
        self.save_hyperparameters(ignore=["model", "graph_data"])
        self.model = model
        self.strict_loading = (
            False  # Set to False because we only care about the projector
        )
        # Graph data moved to device if provided
        self.graph_data = (
            graph_data.to(self.model.llm.device) if graph_data is not None else None
        )
        self.debug_numerics = debug_numerics
        self.numerical_debugger = None

    def forward(self, batch):
        return self.model(batch, self.graph_data)

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log("train_loss", loss)
        return loss

    def on_fit_start(self):
        if self.debug_numerics:
            self.numerical_debugger = NumericalDebugger(
                model=self.model,
                get_step=lambda: self.global_step,
                get_rank=lambda: self.global_rank,
                log_every=25,
            )
            print(
                f"[DEBUG STATUS] rank={self.global_rank} "
                f"debug_numerics={self.debug_numerics}",
                flush=True,
            )

    def on_fit_end(self):
        if self.numerical_debugger is not None:
            self.numerical_debugger.remove()
            self.numerical_debugger = None


    """def on_after_backward(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    raise RuntimeError(
                        f"Non-finite gradient in {name}"
                    )"""

    def on_train_batch_start(self, batch, batch_idx):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if not torch.isfinite(param).all():
                    raise RuntimeError(
                        f"Non-finite parameter at batch {batch_idx}: {name}"
                    )

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log("val_loss", loss, batch_size=1, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Optimizer on trainable parameters
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.hparams.lr,
        )
        # StepLR scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.scheduler_step_size,
            gamma=self.hparams.scheduler_gamma,
        )
        # Return both optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # call every epoch
                "monitor": "val_loss",  # for ReduceLROnPlateau (not used here)
            },
        }

    """def state_dict(self):
        return {k: v for k, v in super().state_dict().items() if "projector" in k}"""

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        return {
            k: v
            for k, v in sd.items()
            if "projector" in k or "lora_" in k
        }

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Recovery checkpoints intentionally exclude the frozen LLM and GNN.
        # Load only the stored trainable weights.
        return super().load_state_dict(
            state_dict,
            strict=False,
            assign=assign,
        )
