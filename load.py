import torch
import datasets
datasets.disable_caching()
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

INSTRUCTION_TEMPLATE = "### Human:\n"
RESPONSE_TEMPLATE = "### Response:\n"

DATASET_CONFIGS = {
    "gsm8k": ("gsm8k", "main", ("train", "test"), 1.0, 1.0),
    "math": ("math_dataset", "all", ("train", "test"), 1.0, 1.0)
}

def load_sft_dataset(sargs):
    dataset_config = DATASET_CONFIGS[sargs.dataset_name]
    if not sargs.is_easy_to_hard:
        dataset = load_dataset(
            dataset_config[0],
            name=dataset_config[1],
            cache_dir="./cache",
        )
        train_dataset, eval_dataset = (
            dataset[dataset_config[2][0]],
            dataset[dataset_config[2][1]],
        )
        dataset_splits = train_dataset.train_test_split(
            test_size=0.5 * dataset_config[3], seed=sargs.seed
        )
        train_dataset, transfer_dataset = (
            dataset_splits["test"],
            dataset_splits["train"],
        )
        transfer_dataset = transfer_dataset.select(
            range(min(len(transfer_dataset), len(train_dataset)))
        )
    else:
        pass
        # original_dataset = load_dataset(
        #     dataset_config[0],
        #     name=dataset_config[1],
        #     cache_dir="./cache",
        # )
        # eval_size = len(original_dataset[dataset_config[2][1]])
        # dataset = load_dataset(
        #     DIFFICULTY_DATASET_ROOT + sargs.dataset_name,
        #     cache_dir="./cache",
        # )["train"].sort("Diff_rating")
        # train_dataset = dataset.select(range((len(dataset) - eval_size) // 2))
        # transfer_dataset = dataset.select(
        #     range((len(dataset) - eval_size) // 2, len(dataset) - eval_size)
        # )
        # transfer_dataset = transfer_dataset.select(
        #     range(min(len(transfer_dataset), len(train_dataset)))
        # )
        # eval_dataset = dataset.select(range(len(dataset) - eval_size, len(dataset)))

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

def get_model_and_tokenizer(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        low_cpu_mem_usage=False,
        trust_remote_code=False,
        use_cache=True,
        cache_dir="/fs/class-projects/fall2024/cmsc473/c473g001/cache",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=args.model_max_length,
        padding=True,
        truncation=True,
        padding_side="left",
        use_fast=True,
        trust_remote_code=False,
        cache_dir="/fs/class-projects/fall2024/cmsc473/c473g001/cache",
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def gsm8k_formatter_func(example):
    question = example['question']
    answer = example['answer']
    return {
        'text': INSTRUCTION_TEMPLATE + question + '\n' + RESPONSE_TEMPLATE + answer,
    }

def math_formatter_func(example):
    question = example['problem']
    answer = example['solution']
    return {
        'text': INSTRUCTION_TEMPLATE + question + '\n' + RESPONSE_TEMPLATE + answer,
    }

def format_dataset(dataset, dataset_name, num_proc):
    formatter_func = {
        "gsm8k": gsm8k_formatter_func,
        "math": math_formatter_func,
    }[dataset_name]
    dataset = dataset.map(
        formatter_func,
        num_proc=num_proc,
        # Otherwise will be missused by Trainers
        remove_columns=None,
        desc="Formatting dataset",
    )
    return dataset

def tokenize_function(examples, tokenizer, max_seq_length):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
    )
    # Create labels by copying input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()

    # Replace padding token IDs in labels with -100
    padding_token_id = tokenizer.pad_token_id
    tokenized["labels"] = [
        [(label if label != padding_token_id else -100) for label in labels]
        for labels in tokenized["labels"]
    ]

    return tokenized

def get_dataloaders(args, tokenizer, train_dataset, eval_dataset):
    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.model_max_length),
        batched=True,
        num_proc=args.num_proc,
        remove_columns=train_dataset.column_names,
    )

    eval_dataset = eval_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.model_max_length),
        batched=True,
        num_proc=args.num_proc,
        remove_columns=eval_dataset.column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        num_workers=0,
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
        num_workers=0,
    )

    return train_dataloader, eval_dataloader


def data_preparation(difficulty=-1):
    assert difficulty >= -1 and difficulty <=6
    dataset = load_dataset(
        "furonghuang-lab/Easy2Hard-Bench",
        "E2H-GSM8K",
        split="eval",
    ).select_columns(
        ["question", "answer", "rating_quantile"]
    ).sort(
        "rating_quantile"
    )
    if difficulty != -1:
        return dataset.select(range(2*difficulty*100, (2*difficulty+1)*100))
    else:
        return dataset