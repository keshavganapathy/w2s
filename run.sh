# export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch --num_processes=4 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no main.py\
    --model_name="EleutherAI/pythia-70m-deduped"\
    --is_easy_to_hard="False"\
    --dataset_name="arc"\
    --num_epochs=5\
    --learning_rate=5e-5\
    --train_batch_size=6\
    --pred_batch_size=4\
    --num_proc=4\
    --model_max_length=512\
    --is_completion_only="True"\
    --test_limit=-1\
    --w2s_folder="./"\