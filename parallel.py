import warnings
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from transformers import set_seed
from train_parallel import train

warnings.filterwarnings(
    "ignore", message="You passed a tokenizer with `padding_side` not equal to `right`"
)

@dataclass
class ScriptArguments:
    model_name: str = field(metadata={"help": "Name of the model to use"})
    dataset_name: str = field(metadata={"help": "Name of the dataset to use"})
    is_easy_to_hard: bool = field(
        default=False, metadata={"help": "Whether to use easy-to-hard sampling"}
    )
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
        default=-1, metadata={"help": "Limit on the number of test examples"}
    )
    seed: int = field(default=42, metadata={"help": "Random seed"})
    output_dir: str = field(
        default="./outputs", metadata={"help": "Directory to save the model"}
    )
    config_file: str = field(
        default=None, metadata={"help": "Path to the training configuration JSON file"}
    )
    lora: bool = field(
        default=False, metadata={"help": "Whether to use LoRA fine-tuning"}
    )
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    num_epochs: int = field(default=3, metadata={"help": "Number of training epochs"})
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Training batch size per device"}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Evaluation batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Gradient accumulation steps"}
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "Learning rate"})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay"})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm"})
    lr_scheduler_type: str = field(
        default="linear", metadata={"help": "Type of LR scheduler"}
    )
    num_warmup_steps: int = field(
        default=0, metadata={"help": "Number of warmup steps"}
    )
    logging_steps: int = field(
        default=50, metadata={"help": "Logging steps"}
    )

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    set_seed(args.seed)    
    train(args)

if __name__ == "__main__":
    main()
