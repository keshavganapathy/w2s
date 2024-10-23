# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0

accelerate launch --num_processes=4 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no main.py\
    --model_name="EleutherAI/pythia-70m-deduped"\
    --is_easy_to_hard="False"\
    --dataset_name="gsm8k"\
    --num_proc=4\
    --model_max_length=2048\
    --is_completion_only="True"\
    --test_limit=-1\
    --w2s_folder="./"\
    --config_file="./configs/training_config.json"
