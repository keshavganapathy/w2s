import argparse
import os
import psutil
import torch
import datasets
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    HfArgumentParser,
    get_scheduler,
)
from accelerate import Accelerator
from accelerate.utils import set_seed, DeepSpeedPlugin
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import Optional
from peft import get_peft_model, LoraConfig, TaskType
from load import get_model_and_tokenizer, load_sft_dataset, format_dataset, get_dataloaders

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return f"RSS: {mem_info.rss / 1e9:.2f} GB, VMS: {mem_info.vms / 1e9:.2f} GB"


@dataclass
class TrainingArguments:
    model_name: str = field(
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    dataset_name: str = field(metadata={"help": "The name of the dataset to use."})
    is_easy_to_hard: bool = field(
        default=False, metadata={"help": "Not sure what this is"}
    )
    test_limit: int = field(
        default=-1, metadata={"help": "Limit for the number of test examples"}
    )
    num_proc: int = field(
        default=4, metadata={"help": "Number of processes for data preprocessing"}
    )
    seed: int = field(default=42, metadata={"help": "Seed for random number generators"})
    model_max_length: int = field(
        default=256, metadata={"help": "Maximum sequence length"}
    )
    output_dir: str = field(default="./output", 
        metadata={"help": "Output directory for model checkpoints"}
    )
    overwrite_output_dir: bool = field(default=False)
    num_train_epochs: int = field(default=3)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    learning_rate: float = field(default=5e-5)
    weight_decay: float = field(default=0.0)
    logging_steps: int = field(default=500)
    evaluation_strategy: str = field(default="no")
    save_strategy: str = field(default="no")
    deepspeed: Optional[str] = field(
        default=None, metadata={"help": "Path to deepspeed config file"}
    )
    fp16: bool = field(default=False)


def main():
    parser = HfArgumentParser(TrainingArguments)
    training_args = parser.parse_args_into_dataclasses()[0]

    set_seed(training_args.seed)
    # Initialize Accelerator
    if training_args.deepspeed:
        accelerator = Accelerator(deepspeed_config_file=training_args.deepspeed)
    else:
        accelerator = Accelerator()

    accelerator.print("Loading model and tokenizer...")
    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(training_args)
    accelerator.print("Model and tokenizer loaded.")
    # Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Adjust according to the model
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Load dataset
    accelerator.print("Loading dataset...")
    train_dataset, transfer_dataset, eval_dataset = load_sft_dataset(training_args)

    # Format dataset
    train_dataset = format_dataset(
        train_dataset, training_args.dataset_name, training_args.num_proc
    )
    eval_dataset = format_dataset(
        eval_dataset, training_args.dataset_name, training_args.num_proc
    )

    # Get dataloaders
    train_dataloader, eval_dataloader = get_dataloaders(
        training_args, training_args, tokenizer, train_dataset, eval_dataset
    )
    accelerator.print("Dataset loaded.")

    

    # Prepare model and optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
    )

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    model.gradient_checkpointing_enable()
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = (
        len(train_dataloader) // training_args.gradient_accumulation_steps
    )
    if len(train_dataloader) % training_args.gradient_accumulation_steps != 0:
        num_update_steps_per_epoch += 1
    max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps,
    )

    # Train!
    total_batch_size = (
        training_args.per_device_train_batch_size
        * accelerator.num_processes
        * training_args.gradient_accumulation_steps
    )

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num Epochs = {training_args.num_train_epochs}")
    accelerator.print(
        f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}"
    )
    accelerator.print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    accelerator.print(
        f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}"
    )
    accelerator.print(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(training_args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / training_args.gradient_accumulation_steps
            accelerator.backward(loss)

            if (
                step % training_args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if step % training_args.logging_steps == 0:
                accelerator.print(
                    f"Epoch {epoch}, step {step}, loss {loss.item()}, {log_memory_usage()}"
                )

        model.eval()
        eval_loss = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
        eval_loss = eval_loss / len(eval_dataloader)
        accelerator.print(f"Epoch {epoch}, eval_loss {eval_loss.item()}")

    # Save the model
    if accelerator.is_main_process:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            training_args.output_dir, save_function=accelerator.save
        )
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()