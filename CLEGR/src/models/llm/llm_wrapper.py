import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from src.modifiers import Modifier

try:
    from transformers.tokenization_utils_base import BatchEncoding
except ImportError:
    BatchEncoding = Dict

IGNORE_INDEX = -100
MAX_TXT_LEN = 16384
MAX_NEW_TOKENS = 32
PADDING_SIDE = 'left'
class LLM(torch.nn.Module):
    r"""A wrapper around a Large Language Model (LLM) from HuggingFace.

    model_name (str): The HuggingFace model name, *e.g.*, :obj:`"llama2"` or
        :obj:`"gemma"`.
    num_params (int): An integer representing how many parameters the
        HuggingFace model has, in billions. This is used to automatically
        allocate the correct number of GPUs needed, given the available GPU
        memory of your GPUs.
    dtype (torch.dtype, optional): The data type to use for the LLM.
        (default :obj: `torch.bloat16`)
    """
    def __init__(
        self,
        model_name: str,
        num_params: int,
        dtype=torch.float16,
    ) -> None:
        super().__init__()

        self.model_name = model_name

        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
        kwargs = dict(revision='main',dtype=torch.float32,)
        config = AutoConfig.from_pretrained(model_name)
        if not hasattr(config, 'pad_token_id') or config.pad_token_id is None:
            config.pad_token_id = config.eos_token_id
        q_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            #bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_compute_dtype=torch.float32,
        )
        kwargs["quantization_config"] = q_config
        kwargs.pop("config", None)
        kwargs["config"] = config

        print(f"4-bit loading enabled for {model_name}")

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            **kwargs,
        )
        print("LM head dtype:", self.llm.lm_head.weight.dtype)
        print("Embedding dtype:", self.llm.get_input_embeddings().weight.dtype)

        """kwargs = dict(revision='main')
        kwargs['low_cpu_mem_usage'] = True
        kwargs['device_map'] = f"cuda:{torch.cuda.current_device()}"
        kwargs['torch_dtype'] = dtype"""
        
        self.dtype = dtype

        print(f"Setting up '{model_name}' with configuration: {kwargs}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
        )
        self.tokenizer.padding_side = PADDING_SIDE
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError(
                    f"Tokenizer for {model_name} has neither a pad token nor an EOS token."
                )
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        if getattr(self.llm, "generation_config", None) is not None:
            self.llm.generation_config.pad_token_id = self.tokenizer.pad_token_id
        (
            self.user_prefix,
            self.user_suffix,
            self.answer_suffix,
        ) = self._build_prompt_format()
        print(f"\nPrompt format for {self.model_name}")
        print("  chat_template:", self.tokenizer.chat_template is not None)
        print("  user_prefix:", repr(self.user_prefix))
        print("  user_suffix:", repr(self.user_suffix))
        print("  answer_suffix:", repr(self.answer_suffix))
        print(
            "  pad:",
            repr(self.tokenizer.pad_token),
            self.tokenizer.pad_token_id,
        )
        print(
            "  bos:",
            repr(self.tokenizer.bos_token),
            self.tokenizer.bos_token_id,
        )
        print(
            "  eos:",
            repr(self.tokenizer.eos_token),
            self.tokenizer.eos_token_id,
        )
        #self.llm = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.word_embedding = self.llm.model.get_input_embeddings()

        self.device = self.llm.device
        #self.autocast_context = torch.amp.autocast(device_type="cuda", dtype=dtype)
        self.autocast_context = nullcontext()

    def _build_prompt_format(self) -> tuple[str, str, str]:
        """
        Build a model-appropriate prompt format while preserving the location
        where continuous graph embeddings are inserted.
        Returns:
            user_prefix:
                Text occurring before the graph embeddings.
            user_suffix:
                Text occurring after context/question and immediately before the
                assistant's answer.
            answer_suffix:
                Text appended after the ground-truth answer during training.
        """

        # ------------------------------------------------------------------
        # Instruction/chat models
        # ------------------------------------------------------------------
        if self.tokenizer.chat_template is not None:
            content_marker = "CLEGR_CONTENT_MARKER_48291"
            answer_marker = "CLEGR_ANSWER_MARKER_73916"
            # Render the native inference prompt.
            inference_prompt = self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": content_marker,
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

            if inference_prompt.count(content_marker) != 1:
                raise ValueError(
                    f"Could not safely locate user content inside the chat "
                    f"template for {self.model_name}."
                )

            user_prefix, user_suffix = inference_prompt.split(
                content_marker,
                1,
            )
            # Render a complete training conversation so we can obtain the
            # model-native ending after an assistant answer.
            training_prompt = self.tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": content_marker,
                    },
                    {
                        "role": "assistant",
                        "content": answer_marker,
                    },
                ],
                tokenize=False,
                add_generation_prompt=False,
            )

            if training_prompt.count(answer_marker) != 1:
                raise ValueError(
                    f"Could not safely locate assistant content inside the chat "
                    f"template for {self.model_name}."
                )

            _, answer_suffix = training_prompt.split(
                answer_marker,
                1,
            )

            return user_prefix, user_suffix, answer_suffix
        # ------------------------------------------------------------------
        # Base causal LMs with no chat template, e.g. Phi-1.5
        # ------------------------------------------------------------------
        # Use the model's real BOS token if one exists.
        user_prefix = self.tokenizer.bos_token or ""
        # Plain completion-style interface.
        #
        # The graph embeddings will appear immediately after the prefix:
        #
        #   BOS
        #   [GRAPH TOKENS]
        #   question
        #   \nAnswer:
        #
        # This is much more appropriate for a base causal LM than fake [INST]
        # tokens belonging to another model family.
        user_suffix = "\nAnswer:"

        # End the supervised target with the model's actual EOS token.
        answer_suffix = self.tokenizer.eos_token or ""

        return user_prefix, user_suffix, answer_suffix

    def _encode_inputs(
        self,
        question: List[str],
        context: Optional[List[str]] = None,
    ) -> tuple:

        batch_size = len(question)

        questions = self.tokenizer(
            question,
            add_special_tokens=False,
        )

        if context is not None:
            context = self.tokenizer(
                context,
                add_special_tokens=False,
            )

        user_suffix_tokens = self.tokenizer(
            self.user_suffix,
            add_special_tokens=False,
        )

        prefix_tokens = self.tokenizer(
            self.user_prefix,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0].to(self.device)

        if len(prefix_tokens) > 0:
            prefix_embeds = self.word_embedding(prefix_tokens)
        else:
            prefix_embeds = torch.empty(
                (0, self.word_embedding.embedding_dim),
                device=self.device,
                dtype=self.word_embedding.weight.dtype,
            )

        pad_token = torch.tensor(
            self.tokenizer.pad_token_id,
            device=self.device,
        )

        pad_embeds = self.word_embedding(
            pad_token
        ).unsqueeze(0)

        return (
            batch_size,
            questions,
            context,
            user_suffix_tokens,
            prefix_embeds,
            pad_embeds,
        )

    def _label_input_ids(
        self,
        i: int,
        label: BatchEncoding,
        eos_tokens: BatchEncoding,
    ) -> List[int]:
        label_input_ids = label.input_ids[i][:MAX_NEW_TOKENS]
        label_input_ids = label_input_ids + eos_tokens.input_ids
        return label_input_ids

    def _input_ids(
        self,
        i: int,
        context: BatchEncoding,
        question: BatchEncoding,
        user_suffix_tokens: BatchEncoding,
    ) -> List[int]:

        input_ids: List[int] = []

        if context is not None:
            input_ids += context.input_ids[i][:MAX_TXT_LEN]

        input_ids += question.input_ids[i]
        input_ids += user_suffix_tokens.input_ids

        return input_ids

    def _inputs_embeds(
        self,
        i: int,
        input_ids: List[int],
        prefix_embeds: Tensor,
        embedding: Optional[List[Tensor]] = None,
    ) -> Tensor:

        inputs_embeds = self.word_embedding(
            torch.tensor(
                input_ids,
                device=self.device,
            )
        )

        to_cat = [prefix_embeds]

        if embedding is not None and embedding[i] is not None:
            to_cat.append(embedding[i])

        to_cat.append(inputs_embeds)

        return torch.cat(
            to_cat,
            dim=0,
        ).to(self.device)

    def _append_embeds(
        self,
        inputs_embeds: Tensor,
        batch_inputs_embeds: List[Tensor],
        batch_attention_mask: List[List[int]],
        label_input_ids: List[int] = None,
        batch_label_input_ids: Optional[List[List[int]]] = None,
    ) -> tuple:
        batch_inputs_embeds.append(inputs_embeds)
        batch_attention_mask.append([1] * inputs_embeds.size(0))
        if label_input_ids is not None:
            pad = inputs_embeds.size(0) - len(label_input_ids)
            label_input_ids = [IGNORE_INDEX] * pad + label_input_ids
            batch_label_input_ids.append(label_input_ids)
        return batch_inputs_embeds, batch_attention_mask, batch_label_input_ids

    def _pad_embeds(
        self,
        pad_embeds: Tensor,
        batch_inputs_embeds: List[Tensor],
        batch_attention_mask: List[List[int]],
        batch_label_input_ids: Optional[List[List[int]]] = None,
    ) -> tuple:
        max_length = max([x.size(0) for x in batch_inputs_embeds])
        batch_size = len(batch_inputs_embeds)
        for i in range(batch_size):
            pad = max_length - batch_inputs_embeds[i].size(0)
            batch_inputs_embeds[i] = torch.cat([
                pad_embeds.repeat(pad, 1),
                batch_inputs_embeds[i],
            ])
            batch_attention_mask[i] = [0] * pad + batch_attention_mask[i]
            if batch_label_input_ids is not None:
                tmp = [IGNORE_INDEX] * pad + batch_label_input_ids[i]
                batch_label_input_ids[i] = tmp
        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0)
        attention_mask = torch.tensor(batch_attention_mask, device=self.device)
        label_input_ids = None
        if batch_label_input_ids is not None:
            label_input_ids = torch.tensor(batch_label_input_ids,
                                           device=self.device)
        return inputs_embeds, attention_mask, label_input_ids

    def _get_embeds(
        self,
        question: List[str],
        context: Optional[List[str]] = None,
        embedding: Optional[List[Tensor]] = None,
        answer: Optional[List[str]] = None,
        modifier: Optional[Modifier] = None,
    ) -> tuple:
        (
            batch_size,
            question,
            context,
            user_suffix_tokens,
            prefix_embeds,
            pad_embeds,
        ) = self._encode_inputs(
            question,
            context,
        )
        
        # Apply the modifier
        if modifier is not None:
            question, context, embedding = modifier(question, context, embedding, self.tokenizer)
        
        batch_label_input_ids = None
        if answer is not None:
            label = self.tokenizer(answer, add_special_tokens=False)
            answer_suffix_tokens = self.tokenizer(self.answer_suffix,add_special_tokens=False,)
            batch_label_input_ids = []

        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            input_ids = self._input_ids(i, context, question, user_suffix_tokens)
            if answer is not None:
                label_input_ids = self._label_input_ids(i,label,answer_suffix_tokens,)                
                input_ids += label_input_ids
            else:
                label_input_ids = None

            inputs_embeds = self._inputs_embeds(i, input_ids, prefix_embeds,embedding)

            (
                batch_inputs_embeds,
                batch_attention_mask,
                batch_label_input_ids,
            ) = self._append_embeds(
                inputs_embeds,
                batch_inputs_embeds,
                batch_attention_mask,
                label_input_ids,
                batch_label_input_ids,
            )

        inputs_embeds, attention_mask, label_input_ids = self._pad_embeds(
            pad_embeds, batch_inputs_embeds, batch_attention_mask,
            batch_label_input_ids)

        return inputs_embeds, attention_mask, label_input_ids

    def forward(
        self,
        question: List[str],
        answer: List[str],
        context: Optional[List[str]] = None,
        embedding: Optional[List[Tensor]] = None,
    ) -> Tensor:
        r"""The forward pass.

        Args:
            question (list[str]): The questions/prompts.
            answer (list[str]): The answers/labels.
            context (list[str], optional): Additional context to give to the
                LLM, such as textified knowledge graphs. (default: :obj:`None`)
            embedding (list[torch.Tensor], optional): RAG embedding
                tensors, *i.e.* the embedded form of :obj:`context`. Either
                :obj:`context` or :obj:`embedding` should be used, not
                both. (default: :obj:`None`)
        """
        inputs_embeds, attention_mask, label_input_ids = self._get_embeds(
            question, context, embedding, answer)

        with self.autocast_context:
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )
        return outputs.loss

    @torch.no_grad()
    def inference(
        self,
        question: List[str],
        context: Optional[List[str]] = None,
        embedding: Optional[List[Tensor]] = None,
        max_tokens: Optional[int] = MAX_NEW_TOKENS,
    ) -> List[str]:
        r"""The inference pass.

        Args:
            question (list[str]): The questions/prompts.
            answer (list[str]): The answers/labels.
            context (list[str], optional): Additional context to give to the
                LLM, such as textified knowledge graphs. (default: :obj:`None`)
            embedding (list[torch.Tensor], optional): RAG embedding
                tensors, *i.e.* the embedded form of :obj:`context`. Either
                :obj:`context` or :obj:`embedding` should be used, not
                both. (default: :obj:`None`)
            max_tokens (int, optional): How many tokens for the LLM to
                generate. (default: :obj:`32`)
        """
        inputs_embeds, attention_mask, _ = self._get_embeds(
            question, context, embedding)

        generation_eos_ids = self._generation_eos_token_ids()
        generation_kwargs = {
            "inputs_embeds": inputs_embeds,
            "max_new_tokens": max_tokens,
            "attention_mask": attention_mask,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if generation_eos_ids is not None:
            generation_kwargs["eos_token_id"] = generation_eos_ids

        with self.autocast_context:
            outputs = self.llm.generate(
                **generation_kwargs
            )

        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.model_name})'

    def _generation_eos_token_ids(self) -> Optional[List[int]]:
        eos_ids = []

        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(self.tokenizer.eos_token_id)

        # For chat models, answer_suffix usually begins with the native
        # end-of-turn control token.
        if self.tokenizer.chat_template is not None and self.answer_suffix:
            suffix_ids = self.tokenizer(
                self.answer_suffix,
                add_special_tokens=False,
            ).input_ids
            if suffix_ids:
                eos_ids.append(suffix_ids[0])

        eos_ids = list(dict.fromkeys(eos_ids))

        if not eos_ids:
            return None

        return eos_ids