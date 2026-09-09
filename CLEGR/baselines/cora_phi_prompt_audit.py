import os
import random

import numpy as np
import torch

from peft import PeftModel
from torch_geometric.loader import DataLoader

import training


FLAGS = training.FLAGS


# ============================================================
# Configuration
# ============================================================

METHOD = "tea-glm"
DATASET = "cora"
SEED = 0

# Run one model at a time:
#
# AUDIT_LLM=phi-1.5 PYTHONPATH=. python baselines/cora_phi_prompt_audit.py
#
# AUDIT_LLM=phi-3.5 PYTHONPATH=. python baselines/cora_phi_prompt_audit.py
#
LLM_MODEL = os.environ.get(
    "AUDIT_LLM",
    "phi-1.5",
)

BATCH_SIZE = 1
NUM_WORKERS = 0


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Helpers
# ============================================================

def get_node_id(data):
    node_id = data.node_id

    if torch.is_tensor(node_id):
        return int(
            node_id.view(-1)[0].item()
        )

    if isinstance(node_id, (list, tuple)):
        return int(node_id[0])

    return int(node_id)


def as_single_string(value):
    """
    Convert a batch-size-1 value to a single string.
    """

    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple)):
        assert len(value) == 1
        return str(value[0])

    return str(value)


def ids_to_tokens(tokenizer, ids):
    if torch.is_tensor(ids):
        ids = ids.detach().cpu().tolist()

    return tokenizer.convert_ids_to_tokens(
        list(ids)
    )


def print_tokenization(
    tokenizer,
    title,
    text,
):
    """
    Print exact tokenizer output for one string.
    """

    encoded = tokenizer(
        text,
        add_special_tokens=False,
    )

    ids = encoded.input_ids
    tokens = ids_to_tokens(
        tokenizer,
        ids,
    )

    print()
    print(title)
    print("-" * 72)

    print(
        "TEXT:",
        repr(text),
    )

    print(
        "TOKEN COUNT:",
        len(ids),
    )

    print(
        "IDS:",
        ids,
    )

    print(
        "TOKENS:",
        tokens,
    )

    print(
        "DECODED:",
        repr(
            tokenizer.decode(ids)
        ),
    )

    return ids


def print_side_by_side_boundary_test(
    tokenizer,
    left,
    right,
    description,
):
    """
    Compare:

        tokenize(left) + tokenize(right)

    versus:

        tokenize(left + right)

    This matters because the wrapper tokenizes several prompt
    components independently and concatenates token IDs.
    """

    left_ids = tokenizer(
        left,
        add_special_tokens=False,
    ).input_ids

    right_ids = tokenizer(
        right,
        add_special_tokens=False,
    ).input_ids

    separate_ids = (
        left_ids
        + right_ids
    )

    joint_ids = tokenizer(
        left + right,
        add_special_tokens=False,
    ).input_ids

    separate_tokens = (
        ids_to_tokens(
            tokenizer,
            separate_ids,
        )
    )

    joint_tokens = (
        ids_to_tokens(
            tokenizer,
            joint_ids,
        )
    )

    equal = (
        separate_ids
        == joint_ids
    )

    print()
    print(
        "BOUNDARY TEST:",
        description,
    )

    print("-" * 72)

    print(
        "LEFT:",
        repr(left),
    )

    print(
        "RIGHT:",
        repr(right),
    )

    print()
    print(
        "SEPARATELY TOKENIZED"
    )

    print(
        "ids:",
        separate_ids,
    )

    print(
        "tokens:",
        separate_tokens,
    )

    print(
        "decoded:",
        repr(
            tokenizer.decode(
                separate_ids
            )
        ),
    )

    print()
    print(
        "JOINTLY TOKENIZED"
    )

    print(
        "ids:",
        joint_ids,
    )

    print(
        "tokens:",
        joint_tokens,
    )

    print(
        "decoded:",
        repr(
            tokenizer.decode(
                joint_ids
            )
        ),
    )

    print()
    print(
        "IDENTICAL TOKENIZATION:",
        equal,
    )

    return equal


# ============================================================
# Load exact trained TEA model
# ============================================================

def load_model(dataset):
    print()
    print("=" * 72)
    print("INITIALIZING TEA-GLM")
    print("=" * 72)

    model = training.initialize_model(
        FLAGS.lr,
        graph_data=dataset.graph_data,
        use_lora=False,
    )

    lm_module_cls = (
        training.get_glm_lightning_module(
            FLAGS.method
        )
    )

    checkpoint_path = (
        f"{FLAGS.save_dir}/"
        f"{FLAGS.method}/"
        f"projector_"
        f"{FLAGS.llm_model_name}_"
        f"{FLAGS.dataset}_"
        f"seed_{FLAGS.seed}.ckpt"
    )

    print()
    print(
        "Projector checkpoint:"
    )

    print(
        checkpoint_path
    )

    assert os.path.exists(
        checkpoint_path
    )

    lm_module = (
        lm_module_cls.load_from_checkpoint(
            checkpoint_path,
            model=model.model,
            lr=FLAGS.lr,
            graph_data=dataset.graph_data,
            scheduler_step_size=(
                FLAGS.scheduler_step_size
            ),
            scheduler_gamma=(
                FLAGS.scheduler_gamma
            ),
        )
    )

    print(
        "Projector loaded: OK"
    )

    lora_dir = os.path.join(
        FLAGS.save_dir,
        FLAGS.method,
        (
            f"lora_"
            f"{FLAGS.llm_model_name}_"
            f"{FLAGS.dataset}_"
            f"seed_{FLAGS.seed}"
        ),
    )

    print()
    print(
        "LoRA directory:"
    )

    print(
        lora_dir
    )

    if os.path.exists(
        lora_dir
    ):
        lm_module.model.llm_generator = (
            PeftModel.from_pretrained(
                lm_module
                .model
                .llm_generator,
                lora_dir,
            )
        )

        print(
            "LoRA loaded: OK"
        )

        print(
            "Active adapters:",
            getattr(
                lm_module
                .model
                .llm_generator,
                "active_adapters",
                None,
            ),
        )

    else:
        print(
            "WARNING: LoRA directory not found."
        )

    lm_module.model.eval()

    return (
        model,
        lm_module,
    )


# ============================================================
# Label audit
# ============================================================

def get_candidate_labels(dataset):
    labels = sorted(
        {
            str(
                dataset[i].label
            ).strip()
            for i in range(
                len(dataset)
            )
        }
    )

    return labels


def audit_all_labels(
    tokenizer,
    user_suffix,
    answer_suffix,
    candidate_labels,
):
    print()
    print("=" * 72)
    print("ALL CORA LABEL TOKENIZATIONS")
    print("=" * 72)

    for label in candidate_labels:

        print_tokenization(
            tokenizer,
            f"LABEL: {label}",
            label,
        )

        # ----------------------------------------------------
        # Important alternatives
        # ----------------------------------------------------

        print_tokenization(
            tokenizer,
            f"LABEL WITH LEADING SPACE: {label}",
            " " + label,
        )

        print_side_by_side_boundary_test(
            tokenizer=tokenizer,
            left=user_suffix,
            right=label,
            description=(
                f"user_suffix + {label}"
            ),
        )

        print_side_by_side_boundary_test(
            tokenizer=tokenizer,
            left=user_suffix,
            right=" " + label,
            description=(
                f"user_suffix + SPACE + {label}"
            ),
        )

        print_side_by_side_boundary_test(
            tokenizer=tokenizer,
            left=label,
            right=answer_suffix,
            description=(
                f"{label} + answer_suffix"
            ),
        )

        print()
        print("=" * 72)


# ============================================================
# Exact sample prompt
# ============================================================

@torch.no_grad()
def audit_one_sample(
    glm,
    graph_data,
    data,
):
    tokenizer = (
        glm.llm.tokenizer
    )

    llm_wrapper = (
        glm.llm
    )

    device = (
        glm.llm.device
    )

    graph_data = (
        graph_data.to(device)
    )

    data = (
        data.to(device)
    )

    node_id = get_node_id(
        data
    )

    question = (
        as_single_string(
            data.question
        )
    )

    label = (
        as_single_string(
            data.label
        )
    )

    context = glm.get_context(
        data
    )

    context_string = None

    if context is not None:
        context_string = (
            as_single_string(
                context
            )
        )

    print()
    print("=" * 72)
    print("SAMPLE")
    print("=" * 72)

    print(
        "NODE ID:",
        node_id,
    )

    print()
    print(
        "QUESTION:"
    )

    print(
        repr(question)
    )

    print()
    print(
        "CONTEXT:"
    )

    print(
        repr(
            context_string
        )
    )

    print()
    print(
        "GROUND-TRUTH LABEL:"
    )

    print(
        repr(label)
    )

    # --------------------------------------------------------
    # Exact graph representation
    # --------------------------------------------------------

    node_representations = (
        glm.encode_graph(
            graph_data
        )
    )

    node_x = (
        node_representations[
            node_id
        ]
        .unsqueeze(0)
    )

    projected = (
        glm.projector(
            node_x
        )
    )

    graph_tokens = (
        projected[0]
    )

    print()
    print(
        "GRAPH TOKEN SHAPE:",
        tuple(
            graph_tokens.shape
        ),
    )

    print(
        "GRAPH TOKEN MEAN NORM:",
        float(
            graph_tokens
            .float()
            .norm(
                dim=-1
            )
            .mean()
            .item()
        ),
    )

    embedding = [
        graph_tokens
    ]

    # --------------------------------------------------------
    # Prompt components
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("PROMPT COMPONENTS")
    print("=" * 72)

    print(
        "user_prefix:",
        repr(
            llm_wrapper.user_prefix
        ),
    )

    print(
        "user_suffix:",
        repr(
            llm_wrapper.user_suffix
        ),
    )

    print(
        "answer_suffix:",
        repr(
            llm_wrapper.answer_suffix
        ),
    )

    print_tokenization(
        tokenizer,
        "USER PREFIX",
        llm_wrapper.user_prefix,
    )

    if context_string is not None:
        print_tokenization(
            tokenizer,
            "CONTEXT",
            context_string,
        )

    print_tokenization(
        tokenizer,
        "QUESTION",
        question,
    )

    print_tokenization(
        tokenizer,
        "USER SUFFIX",
        llm_wrapper.user_suffix,
    )

    print_tokenization(
        tokenizer,
        "GROUND-TRUTH LABEL",
        label,
    )

    print_tokenization(
        tokenizer,
        "GROUND-TRUTH LABEL WITH LEADING SPACE",
        " " + label,
    )

    print_tokenization(
        tokenizer,
        "ANSWER SUFFIX",
        llm_wrapper.answer_suffix,
    )

    # --------------------------------------------------------
    # Boundary tests
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("TOKENIZATION BOUNDARY TESTS")
    print("=" * 72)

    print_side_by_side_boundary_test(
        tokenizer=tokenizer,
        left=(
            llm_wrapper.user_suffix
        ),
        right=label,
        description=(
            "user suffix + ground-truth label"
        ),
    )

    print_side_by_side_boundary_test(
        tokenizer=tokenizer,
        left=(
            llm_wrapper.user_suffix
        ),
        right=" " + label,
        description=(
            "user suffix + SPACE + ground-truth label"
        ),
    )

    print_side_by_side_boundary_test(
        tokenizer=tokenizer,
        left=label,
        right=(
            llm_wrapper.answer_suffix
        ),
        description=(
            "ground-truth label + answer suffix"
        ),
    )

    # --------------------------------------------------------
    # Build inference input
    # --------------------------------------------------------

    (
        inference_embeds,
        inference_attention,
        inference_labels,
    ) = llm_wrapper._get_embeds(
        data.question,
        context,
        embedding,
    )

    assert (
        inference_labels
        is None
    )

    # --------------------------------------------------------
    # Build training input
    # --------------------------------------------------------

    (
        training_embeds,
        training_attention,
        training_labels,
    ) = llm_wrapper._get_embeds(
        data.question,
        context,
        embedding,
        data.label,
    )

    # --------------------------------------------------------
    # Recover exact component token IDs
    # --------------------------------------------------------

    prefix_ids = tokenizer(
        llm_wrapper.user_prefix,
        add_special_tokens=False,
    ).input_ids

    question_ids = tokenizer(
        question,
        add_special_tokens=False,
    ).input_ids

    context_ids = []

    if context_string is not None:
        context_ids = tokenizer(
            context_string,
            add_special_tokens=False,
        ).input_ids[:16384]

    user_suffix_ids = tokenizer(
        llm_wrapper.user_suffix,
        add_special_tokens=False,
    ).input_ids

    label_ids = tokenizer(
        label,
        add_special_tokens=False,
    ).input_ids[:32]

    answer_suffix_ids = tokenizer(
        llm_wrapper.answer_suffix,
        add_special_tokens=False,
    ).input_ids

    n_graph_tokens = (
        graph_tokens.size(0)
    )

    # --------------------------------------------------------
    # Sequence layout
    # --------------------------------------------------------

    prefix_start = 0
    prefix_end = len(
        prefix_ids
    )

    graph_start = (
        prefix_end
    )

    graph_end = (
        graph_start
        + n_graph_tokens
    )

    context_start = (
        graph_end
    )

    context_end = (
        context_start
        + len(context_ids)
    )

    question_start = (
        context_end
    )

    question_end = (
        question_start
        + len(question_ids)
    )

    suffix_start = (
        question_end
    )

    suffix_end = (
        suffix_start
        + len(user_suffix_ids)
    )

    answer_start = (
        suffix_end
    )

    answer_end = (
        answer_start
        + len(label_ids)
    )

    answer_suffix_start = (
        answer_end
    )

    answer_suffix_end = (
        answer_suffix_start
        + len(answer_suffix_ids)
    )

    print()
    print("=" * 72)
    print("SEQUENCE LAYOUT")
    print("=" * 72)

    print(
        "prefix:",
        prefix_start,
        "to",
        prefix_end - 1,
    )

    print(
        "graph tokens:",
        graph_start,
        "to",
        graph_end - 1,
    )

    if context_ids:
        print(
            "context:",
            context_start,
            "to",
            context_end - 1,
        )
    else:
        print(
            "context: none"
        )

    print(
        "question:",
        question_start,
        "to",
        question_end - 1,
    )

    print(
        "user suffix:",
        suffix_start,
        "to",
        suffix_end - 1,
    )

    print(
        "answer:",
        answer_start,
        "to",
        answer_end - 1,
    )

    if answer_suffix_ids:
        print(
            "answer suffix:",
            answer_suffix_start,
            "to",
            answer_suffix_end - 1,
        )
    else:
        print(
            "answer suffix: none"
        )

    print()
    print(
        "Inference sequence length:",
        inference_embeds.size(1),
    )

    print(
        "Training sequence length:",
        training_embeds.size(1),
    )

    print(
        "Expected inference length:",
        suffix_end,
    )

    print(
        "Expected training length:",
        answer_suffix_end,
    )

    assert (
        inference_embeds.size(1)
        == suffix_end
    )

    assert (
        training_embeds.size(1)
        == answer_suffix_end
    )

    # --------------------------------------------------------
    # Construct human-readable training sequence
    # --------------------------------------------------------

    logical_tokens = []

    for token_id in prefix_ids:
        logical_tokens.append(
            (
                "PREFIX",
                token_id,
            )
        )

    for graph_idx in range(
        n_graph_tokens
    ):
        logical_tokens.append(
            (
                f"GRAPH_{graph_idx}",
                None,
            )
        )

    for token_id in context_ids:
        logical_tokens.append(
            (
                "CONTEXT",
                token_id,
            )
        )

    for token_id in question_ids:
        logical_tokens.append(
            (
                "QUESTION",
                token_id,
            )
        )

    for token_id in user_suffix_ids:
        logical_tokens.append(
            (
                "USER_SUFFIX",
                token_id,
            )
        )

    for token_id in label_ids:
        logical_tokens.append(
            (
                "ANSWER",
                token_id,
            )
        )

    for token_id in answer_suffix_ids:
        logical_tokens.append(
            (
                "ANSWER_SUFFIX",
                token_id,
            )
        )

    assert (
        len(logical_tokens)
        == training_embeds.size(1)
    )

    assert (
        training_labels.size(1)
        == len(logical_tokens)
    )

    # --------------------------------------------------------
    # Print loss mask
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("EXACT TRAINING SEQUENCE + LOSS MASK")
    print("=" * 72)

    supervised_positions = []

    labels_cpu = (
        training_labels[0]
        .detach()
        .cpu()
        .tolist()
    )

    for position, (
        component,
        token_id,
    ) in enumerate(
        logical_tokens
    ):

        target_id = (
            labels_cpu[
                position
            ]
        )

        supervised = (
            target_id
            != -100
        )

        if supervised:
            supervised_positions.append(
                position
            )

        if token_id is None:
            token_text = (
                f"<{component}>"
            )

            token_id_display = "-"
        else:
            token_text = repr(
                tokenizer.decode(
                    [token_id]
                )
            )

            token_id_display = str(
                token_id
            )

        if supervised:
            target_text = repr(
                tokenizer.decode(
                    [target_id]
                )
            )

            target_display = (
                f"{target_id} "
                f"{target_text}"
            )
        else:
            target_display = "-"

        print(
            f"{position:03d} | "
            f"{component:14s} | "
            f"id={token_id_display:>6s} | "
            f"token={token_text:24s} | "
            f"LOSS={'YES' if supervised else 'no ':3s} | "
            f"target={target_display}"
        )

    # --------------------------------------------------------
    # Verify loss mask
    # --------------------------------------------------------

    expected_supervised_start = (
        answer_start
    )

    expected_supervised_end = (
        answer_suffix_end
    )

    expected_positions = list(
        range(
            expected_supervised_start,
            expected_supervised_end,
        )
    )

    print()
    print("=" * 72)
    print("LOSS MASK VERIFICATION")
    print("=" * 72)

    print(
        "Supervised positions:",
        supervised_positions,
    )

    print(
        "Expected positions:",
        expected_positions,
    )

    print(
        "Loss mask correct:",
        supervised_positions
        == expected_positions,
    )

    assert (
        supervised_positions
        == expected_positions
    )

    # --------------------------------------------------------
    # Verify actual target token IDs
    # --------------------------------------------------------

    actual_supervised_ids = [
        labels_cpu[position]
        for position
        in supervised_positions
    ]

    expected_target_ids = (
        label_ids
        + answer_suffix_ids
    )

    print()
    print(
        "Actual supervised IDs:"
    )

    print(
        actual_supervised_ids
    )

    print()
    print(
        "Expected label+suffix IDs:"
    )

    print(
        expected_target_ids
    )

    print()
    print(
        "Target IDs correct:",
        actual_supervised_ids
        == expected_target_ids,
    )

    assert (
        actual_supervised_ids
        == expected_target_ids
    )

    # --------------------------------------------------------
    # Generation EOS tokens
    # --------------------------------------------------------

    generation_eos_ids = (
        llm_wrapper
        ._generation_eos_token_ids()
    )

    print()
    print("=" * 72)
    print("GENERATION STOP TOKENS")
    print("=" * 72)

    print(
        "EOS IDs:",
        generation_eos_ids,
    )

    if generation_eos_ids is not None:

        for eos_id in (
            generation_eos_ids
        ):

            print(
                eos_id,
                repr(
                    tokenizer.decode(
                        [eos_id]
                    )
                ),
                tokenizer.convert_ids_to_tokens(
                    eos_id
                ),
            )

    # --------------------------------------------------------
    # Compare inference prefix and training prefix
    # --------------------------------------------------------

    prompt_length = (
        inference_embeds.size(1)
    )

    prefix_difference = (
        training_embeds[
            0,
            :prompt_length,
            :
        ].float()
        - inference_embeds[
            0
        ].float()
    ).abs().max().item()

    print()
    print("=" * 72)
    print("TRAINING / INFERENCE PREFIX CONSISTENCY")
    print("=" * 72)

    print(
        "Prompt length:",
        prompt_length,
    )

    print(
        "Max embedding difference:",
        f"{prefix_difference:.10f}",
    )

    print(
        "Prefix identical:",
        prefix_difference
        < 1e-6,
    )

    assert (
        prefix_difference
        < 1e-6
    )


# ============================================================
# Main
# ============================================================

def main():
    seed_everything(
        SEED
    )

    FLAGS(
        [
            "cora_phi_prompt_audit.py",
            f"--method={METHOD}",
            f"--dataset={DATASET}",
            (
                f"--llm_model_name="
                f"{LLM_MODEL}"
            ),
            "--evaluate",
            "--use_lora",
            f"--seed={SEED}",
            f"--batch_size={BATCH_SIZE}",
            f"--num_workers={NUM_WORKERS}",
            "--num_gpus=1",
        ]
    )

    print()
    print("=" * 72)
    print("CORA PHI PROMPT / TOKENIZATION AUDIT")
    print("=" * 72)

    print(
        "Model:",
        FLAGS.llm_model_name,
    )

    print(
        "Dataset:",
        FLAGS.dataset,
    )

    print(
        "Seed:",
        FLAGS.seed,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print()
    print(
        "Loading clean Cora..."
    )

    dataset = (
        training.load_dataset(
            FLAGS.root,
            FLAGS.dataset,
        )
    )

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = training.split_dataset(
        dataset,
        FLAGS.seed,
    )

    print(
        "Split sizes:",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    assert len(dataset) == 2708

    # --------------------------------------------------------
    # Exact trained model
    # --------------------------------------------------------

    (
        base_model,
        lm_module,
    ) = load_model(
        dataset
    )

    glm = (
        lm_module.model
    )

    tokenizer = (
        glm.llm.tokenizer
    )

    print()
    print("=" * 72)
    print("TOKENIZER / MODEL FORMAT")
    print("=" * 72)

    print(
        "Model:",
        glm.llm.model_name,
    )

    print(
        "Has chat template:",
        tokenizer.chat_template
        is not None,
    )

    print(
        "BOS:",
        repr(
            tokenizer.bos_token
        ),
        tokenizer.bos_token_id,
    )

    print(
        "EOS:",
        repr(
            tokenizer.eos_token
        ),
        tokenizer.eos_token_id,
    )

    print(
        "PAD:",
        repr(
            tokenizer.pad_token
        ),
        tokenizer.pad_token_id,
    )

    print(
        "user_prefix:",
        repr(
            glm.llm.user_prefix
        ),
    )

    print(
        "user_suffix:",
        repr(
            glm.llm.user_suffix
        ),
    )

    print(
        "answer_suffix:",
        repr(
            glm.llm.answer_suffix
        ),
    )

    # --------------------------------------------------------
    # Candidate-label audit
    # --------------------------------------------------------

    candidate_labels = (
        get_candidate_labels(
            dataset
        )
    )

    print()
    print(
        "Candidate labels:"
    )

    for label in (
        candidate_labels
    ):
        print(
            " ",
            repr(label),
        )

    audit_all_labels(
        tokenizer=tokenizer,
        user_suffix=(
            glm.llm.user_suffix
        ),
        answer_suffix=(
            glm.llm.answer_suffix
        ),
        candidate_labels=(
            candidate_labels
        ),
    )

    # --------------------------------------------------------
    # One exact test sample
    # --------------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    sample = next(
        iter(test_loader)
    )

    audit_one_sample(
        glm=glm,
        graph_data=(
            dataset.graph_data
        ),
        data=sample,
    )

    print()
    print("=" * 72)
    print("AUDIT COMPLETE")
    print("=" * 72)

    print(
        "Model:",
        FLAGS.llm_model_name,
    )

    print()
    print(
        "Next: compare this output "
        "with the other Phi model."
    )


if __name__ == "__main__":
    main()