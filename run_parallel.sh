export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0

accelerate launch --multi_gpu --num_processes=4 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no parallel.py\
    --model_name="EleutherAI/pythia-410m-deduped"\
    --is_easy_to_hard="False"\
    --dataset_name="arc"\
    --num_proc=4\
    --model_max_length=512\
    --is_completion_only="True"\
    --test_limit=-1\
    --w2s_folder="./"\
    --config_file="./configs/training_config.json"
