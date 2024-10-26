export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_ASYNC_ERROR_HANDLING=1

# accelerate launch --config_file fsdp_config.yaml parallel.py \
#     --model_name meta-llama/Llama-3.2-1B-Instruct \
#     --dataset_name gsm8k \
#     --num_epochs 3 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 5e-5 \
#     --weight_decay 0.01 \
#     --num_warmup_steps 0 \
#     --logging_steps 50 \
#     --output_dir ./outputs \
#     --seed 42
#  EleutherAI/pythia-410m-deduped 
#  Qwen/Qwen2.5-1.5B-Instruct 
#  microsoft/Phi-3.5-mini-instruct
echo "launch training"
accelerate launch --config_file ./configs/deepspeed_accelerate_config.yaml --deepspeed_config_file ./configs/deepspeed_config.json train_parallel.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --dataset_name gsm8k \
    --output_dir ./output \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --logging_steps 100