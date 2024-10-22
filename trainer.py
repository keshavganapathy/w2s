import os
import json
import grp
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from transformers import set_seed
from peft import LoraConfig
from transformers import AutoModelForCausalLM

from load import load_pretrain_model_tokenizer


def train(model_name, train_dataset, eval_dataset, sargs, accelerator, mode="weak"):
    set_seed(sargs.seed)

    output_dir = os.path.join(
        sargs.w2s_folder,
        "models",
        f"{mode}" + ("_" if sargs.is_easy_to_hard else ""),
    )

    model, tokenizer = load_pretrain_model_tokenizer(sargs, accelerator, mode)

    if "pythia" in model_name:
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "query_key_value",
                "dense_h_to_4h",
                "dense_4h_to_h",
                "dense",
            ],
        )
    elif "Qwen" in model_name:
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj", "w1", "w2"],
        )
    elif "Llama" in model_name:
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        )

    if sargs.is_completion_only:
        response_template = "### Response:\n"
        collator = DataCollatorForCompletionOnlyLM(
            response_template, tokenizer=tokenizer
        )
    else:
        collator = None
        
    if sargs.config_file:
        with open(sargs.config_file, 'r') as f:
            config_dict = json.load(f)
        training_args = TrainingArguments(**config_dict)
        training_args.output_dir = output_dir

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        packing=False,
        dataset_text_field="text",
        max_seq_length=sargs.model_max_length,
        peft_config=lora_config,
        data_collator=collator,
    )
    trainer.train()

    trainer.save_model(training_args.output_dir)

    return trainer.model
