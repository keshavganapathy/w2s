# w2s-debate


[Draft](https://www.overleaf.com/project/66e754e11c5ae7457ada36bb)

## Environment requirements
Use python 3.11/3.12

Install the packages in the `requirements.txt` file and make sure the python version is correct.

Install the pytorch cuda version with the following command:
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

Load the cuda module before submit your work or you can add this in your sbatch submit script.
`module load cuda/11.8.0`

Yihan have tested the cuda 11.8.0 works well.

Add `#SBATCH --gres=gpu:4` in your sbatch job script to request 4 GPUs for distributed training.

Create your own `submit.sh` starting with:

```

#!/bin/bash
#SBATCH --job-name=train_test
#SBATCH --ntasks=4
#SBATCH --mem=64gb
#SBATCH --partition=class          
#SBATCH --account=class            
#SBATCH --qos=high                 
#SBATCH --gres=gpu:rtxa5000:4               
#SBATCH --time=01:00:00            
#SBATCH --output=result_%j.out   # Standard output and error log

```

Add necessary bash command to run the `run_parallel.sh` (huggingface login if you are using model needing authorization)

And you can use `sbatch submit.sh` to launch the job.

`upload.py` file is used for uploading your trained model to the huggingface to save your local storage.


## File description
So far, a trivial way to test accuracy is to randomly select 100 questions from the dataset, and measure accuracy based on responses. Below is a description of relevant files:
```

evaluate_model - script to test the accuracy of local model, currently implemented for pythia_debate_160m
evaluate_qwen2.5 - script to test the accuracy of Qwen2.5-Math-1.5B-Instruct. Can be found here: [https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct)
evaluate_tinyllama.py - script to test the accuracy of TinyLlama_v1.1_math_code. Can be found here: [https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct](https://huggingface.co/TinyLlama/TinyLlama_v1.1_math_code)
best_of_n - script to simulate best of n debate, generalized, all models are currently the  Qwen2.5-Math-1.5B-Instruct.

```

Results so far for best_of_n:
Running accuracy of Model 1 after 100 questions: 41.00%
Running accuracy of Model 2 after 100 questions: 41.00%
Running accuracy of Model 3 after 100 questions: 41.00%
Running accuracy of Model 4 after 100 questions: 41.00%
Running accuracy of Model 5 after 100 questions: 35.00%
Running accuracy of Majority Vote after 100 questions: 48.00%
