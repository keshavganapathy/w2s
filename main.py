import os
import warnings
from dataclasses import dataclass, field
from typing import Optional
import torch
from datetime import timedelta
from transformers import HfArgumentParser
from transformers import set_seed
from accelerate import InitProcessGroupKwargs
from accelerate import Accelerator
from accelerate.utils import gather_object


from load import load_sft_dataset, format_dataset
from trainer import train


process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])

warnings.filterwarnings(
    "ignore", message="You passed a tokenizer with `padding_side` not equal to `right`"
)


@dataclass
class ScriptArguments:
    model_name: str = field(metadata={"help": "Name of the weak model to use"})
    dataset_name: str = field(metadata={"help": "Name of the dataset to use"})
    is_easy_to_hard: bool = field(
        default=False, metadata={"help": "Whether to use easy-to-hard sampling"}
    )
    # Never used for now
    # pred_batch_size: int = field(
    #     default=8, metadata={"help": "Batch size for prediction"}
    # )
    num_proc: int = field(
        default=4, metadata={"help": "Number of processes for data loading"}
    )
    model_max_length: int = field(
        default=512, metadata={"help": "Maximum length of the model"}
    )
    is_completion_only: bool = field(
        default=True, metadata={"help": "Whether to use completion only"}
    )
    test_limit: int = field(
        default=-1, metadata={"help": "whether to run on a limited number of examples"}
    )
    # Seed
    seed: int = field(default=42, metadata={"help": "Random seed"})
    w2s_folder: str = field(
        default="./", metadata={"help": "Path to the folder"}
    )
    temp_folder: str = field(
        default="/tmp/w2s", metadata={"help": "Temporary folder path"}
    )
    config_file: Optional[str] = field(
        default=None, metadata={"help": "Path to the training configuration JSON file"}
    )


def main(sargs):
    # if accelerator.is_main_process:
        # Load dataset
    train_dataset, transfer_dataset, eval_dataset = load_sft_dataset(sargs)

    # Preprocess dataset
    train_dataset = format_dataset(train_dataset, sargs.dataset_name, sargs.num_proc)
    eval_dataset = format_dataset(eval_dataset, sargs.dataset_name, sargs.num_proc)
    transfer_dataset = format_dataset(transfer_dataset, sargs.dataset_name, sargs.num_proc)

    accelerator.wait_for_everyone()

    # Train strong model
    strong_model = train(
        sargs.model_name,
        transfer_dataset,
        eval_dataset,
        sargs,
        accelerator,
        mode="strong",
    )


    # Train weak model
    weak_model = train(
        sargs.model_name, 
        train_dataset, 
        eval_dataset, 
        sargs, 
        accelerator, 
        mode="weak",
    )




if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    sargs = parser.parse_args_into_dataclasses()[0]
    set_seed(sargs.seed)

    # if os.path.exists(
    #     os.path.join(sargs.w2s_folder, "results", format_run_name(sargs) + ".csv")
    # ):
    #     print("Results already exist, skipping...")
    # else:
    main(sargs)
