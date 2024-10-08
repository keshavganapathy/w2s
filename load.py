import torch
import datasets
import re


INSTRUCTION_TEMPLATE = "### Human:\n"
RESPONSE_TEMPLATE = "### Response:\n"

# TODO: remove this line after everything debugged
datasets.disable_caching()
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

DATASET_CONFIGS = {
    # Short response
    "sciq": ("allenai/sciq", None, ("train", "test"), 1.0, 1.0),
    "arc": ("allenai/ai2_arc", "ARC-Challenge", ("train", "test"), 1.0, 1.0),
}

DIFFICULTY_DATASET_ROOT = "Aakriti05/"


def load_sft_dataset(sargs):
    if not sargs.is_easy_to_hard:
        dataset = load_dataset(
            DATASET_CONFIGS[sargs.dataset_name][0],
            name=DATASET_CONFIGS[sargs.dataset_name][1],
            cache_dir="./cache",
        )
        train_dataset, eval_dataset = (
            dataset[DATASET_CONFIGS[sargs.dataset_name][2][0]],
            dataset[DATASET_CONFIGS[sargs.dataset_name][2][1]],
        )
        dataset_splits = train_dataset.train_test_split(
            test_size=0.5 * DATASET_CONFIGS[sargs.dataset_name][3], seed=sargs.seed
        )
        train_dataset, transfer_dataset = (
            dataset_splits["test"],
            dataset_splits["train"],
        )
        transfer_dataset = transfer_dataset.select(
            range(min(len(transfer_dataset), len(train_dataset)))
        )
    else:
        original_dataset = load_dataset(
            DATASET_CONFIGS[sargs.dataset_name][0],
            name=DATASET_CONFIGS[sargs.dataset_name][1],
            cache_dir="./cache",
        )
        eval_size = len(original_dataset[DATASET_CONFIGS[sargs.dataset_name][2][1]])
        dataset = load_dataset(
            DIFFICULTY_DATASET_ROOT + sargs.dataset_name,
            cache_dir="./cache",
        )["train"].sort("Diff_rating")
        train_dataset = dataset.select(range((len(dataset) - eval_size) // 2))
        transfer_dataset = dataset.select(
            range((len(dataset) - eval_size) // 2, len(dataset) - eval_size)
        )
        transfer_dataset = transfer_dataset.select(
            range(min(len(transfer_dataset), len(train_dataset)))
        )
        eval_dataset = dataset.select(range(len(dataset) - eval_size, len(dataset)))

    if sargs.test_limit >= 0:
        train_dataset = train_dataset.select(
            range(min(len(train_dataset), sargs.test_limit))
        )
        eval_dataset = eval_dataset.select(
            range(min(len(eval_dataset), sargs.test_limit))
        )
        transfer_dataset = transfer_dataset.select(
            range(min(len(transfer_dataset), sargs.test_limit))
        )

    return train_dataset, transfer_dataset, eval_dataset


def load_pretrain_model_tokenizer(sargs, accelerator, mode="weak"):
    model = AutoModelForCausalLM.from_pretrained(
        sargs.model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": accelerator.local_process_index},
        trust_remote_code=False,
        use_cache=True,
        cache_dir="./cache",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        sargs.model_name,
        model_max_length=sargs.model_max_length,
        padding=True,
        truncation=True,
        # Important, do not change
        padding_side="left",
        use_fast=True,
        trust_remote_code=False,
        cache_dir="./cache",
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer



def sciq_formatter_func(example):
    support = example["support"].lstrip()
    return {
        "question": f"{support}\n{example['question']}",
        "choices": [
            example["distractor1"],
            example["distractor2"],
            example["distractor3"],
            example["correct_answer"],
        ],
        "target": 3,
    }


def arc_formatter_func(example):
    return {
        "question": f"{example['question']}",
        "choices": example["choices"]["text"],
        "target": example["choices"]["label"].index(example["answerKey"]),
    }



def format_dataset(dataset, dataset_name, num_proc):
    formatter_func = {
        "sciq": sciq_formatter_func,
        "arc": arc_formatter_func,
    }[dataset_name]
    dataset = dataset.map(
        formatter_func,
        num_proc=num_proc,
        # Otherwise will be missused by Trainers
        remove_columns=None,
        desc="Formatting dataset",
    )
    dataset = dataset.map(
        lambda example: {
            "prompt": INSTRUCTION_TEMPLATE + example["question"],
            "response": RESPONSE_TEMPLATE + example["choices"][example["target"]],
            "text": INSTRUCTION_TEMPLATE
            + example["question"]
            + "\n"
            + RESPONSE_TEMPLATE
            + example["choices"][example["target"]],
            "response_list": [
                RESPONSE_TEMPLATE + choice for choice in example["choices"]
            ],
            "text_list": [
                INSTRUCTION_TEMPLATE
                + example["question"]
                + "\n"
                + RESPONSE_TEMPLATE
                + choice
                for choice in example["choices"]
            ],
        },
        num_proc=num_proc,
        desc="Formatting dataset",
    )
    return dataset
