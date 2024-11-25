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

Script to download huggingface models:
```
huggingface-cli download LanguageBind/LanguageBind_Image --local-dir ./LanguageBind_Image
```


## File Description
A simple way to test accuracy is to randomly select 100 questions from the dataset and measure accuracy based on the responses. Below is a description of relevant files:

- **evaluate_model.py**: Script to test the accuracy of the local model, currently implemented for `pythia_debate_160m`.
- **evaluate_qwen2.5.py**: Script to test the accuracy of `Qwen2.5-Math-1.5B-Instruct`. Can be found here: [Qwen2.5-Math-1.5B-Instruct on Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct)
- **evaluate_tinyllama.py**: Script to test the accuracy of `TinyLlama_v1.1_math_code`. Can be found here: [TinyLlama_v1.1_math_code on Hugging Face](https://huggingface.co/TinyLlama/TinyLlama_v1.1_math_code)
- **best_of_n.py**: Script to simulate best-of-n debate, generalized. All models currently tested are `Qwen2.5-Math-1.5B-Instruct`.

## Results for Best-of-n

| Model                  | Running Accuracy (100 Questions) |
|------------------------|----------------------------------|
| Model 1                | 41.00%                           |
| Model 2                | 41.00%                           |
| Model 3                | 41.00%                           |
| Model 4                | 41.00%                           |
| Model 5                | 35.00%                           |
| Majority Vote          | 48.00%                           |

## Contributions Log
- Week of 11/11 and 11/18 - Initial implementaton of deliberation on the chain of thought, but running into prompting issues. For example, looking at outputs models seem to be performing better, but final answers and such arent in the format we want.
- Week of 11/4. (Keshav and Yihan) Use Tom's suggestions to use SOTA LLMs to rerun baseline experiments with gemini-1.0-pro. Found, best of n: 83.29% multiple round: 85.90. Challenges, free API limits, and costs for other LLMs.
- Week of 10/28. (Keshav) Wrote the code for best of n, and debate with deliberation used in midterm presentation.

## Results

Self consistency using 3 agents

CoT: Chain of Thought. Just add "Step-by-step" in the promtps.

Result in () is on the full test set (1319). The default is 650.

| Model                         | one model     | best of n         | self-consistency (2 round) | self-consistency (3 round) |
|-------------------------------|-------------- | ----------------- | -------------------------- |----------------------------|
| Gemini Pro 1.0 (600)          | 77.82%        | 84.24%            | 84.63%                     | 85.40%                     |
| Qwen2.5-7B-Instruct           | 90.77%(91.28%)| 90.77%(91.28%)    | 91.08%(91.43%)             | 91.23%(91.58%)             |
| Llama-3.1-8B-Instruct         | 85.38%(85.97%)| 85.38%(86.13%)    | 85.69%(86.50%)             | 85.69%(86.50%)             |
| Phi-3-small-8k-instruct       | 89.61%        | 89.61%            | 89.31%                     | 89.61%                     |
| Qwen2.5-7B-Instruct (CoT)     | 90.31%        | 90.31%            | 91.08%                     | 90.92%                     |
| Llama-3.1-8B-Instruct (CoT)   | 87.38%        | 87.38%            | 88.15%                     | 87.69%                     |

The bare Qwen2.5-7B-Instruct's performence is questionable. This model will generate answer using CoT automatically (without prompt engineering)

#### Difficulty level break down
Qwen2.5-7B-Instruct:

| Round | Difficulty 0 | Difficulty 1 | Difficulty 2 | Difficulty 3 | Difficulty 4 |
|-------|--------------|--------------|--------------|--------------|--------------|
| 0     | 99.26        | 95.2         | 91.18        | 85.37        | 82.31        |
| 1     | 99.26        | 95.2         | 91.91        | 85.37        | 83.08        |
| 2     | 99.26        | 95.2         | 91.91        | 86.18        | 83.08        |

Llama-3.1-8B-Instruct

| Round | Difficulty 0 | Difficulty 1 | Difficulty 2 | Difficulty 3 | Difficulty 4 |
|-------|--------------|--------------|--------------|--------------|--------------|
| 0     | 95.59        | 93.6         | 87.5         | 83.74        | 66.15        |
| 1     | 94.85        | 93.6         | 88.97        | 83.74        | 66.92        |
| 2     | 94.85        | 93.6         | 88.24        | 83.74        | 67.69        |

Llama-3.1-8B-Instruct (CoT)

| Round | Difficulty 0 | Difficulty 1 | Difficulty 2 | Difficulty 3 | Difficulty 4 |
|-------|--------------|--------------|--------------|--------------|--------------|
| 0     | 97.06        | 92.8         | 88.97        | 86.99        | 70.77        |
| 1     | 97.79        | 92.0         | 88.97        | 89.43        | 72.31        |
| 2     | 97.06        | 91.2         | 89.71        | 88.62        | 71.54        |

### Multi model communication

2 model consistency 

| Model agent/reference         | Qwen2.5-7B-Instruct       | Llama-3.1-8B-Instruct         |
|-------------------------------|---------------------------| ------------------------------|
| Qwen2.5-7B-Instruct           | 91.08%(3 agent)           | 89.08%                        |
| Llama-3.1-8B-Instruct         | 88.33%                    | 86.50%(3 agent)               |


2 model consistency (CoT)

| Model agent/reference         | Qwen2.5-7B-Instruct       | Llama-3.1-8B-Instruct         |
|-------------------------------|---------------------------| ------------------------------|
| Qwen2.5-7B-Instruct           | 91.08%(3 agent)           | 85.00%                        |
| Llama-3.1-8B-Instruct         | 90.62%                    | 88.15%(3 agent)               |

With good reference thought process, model gets better performence.

3 model consistency 

Overall: 91.38%, 91.69%, 91.23%

Better than all models self-consistency results.

| Model                              | base (noref) | reference 2other (2 round) | reference 2other (3 round) |
|------------------------------------| -------------| -------------------------- |----------------------------|
| Qwen2.5-7B-Instruct (CoT)          | 90.31%       | 90.77%                     | 91.38%                     |
| Llama-3.1-8B-Instruct (CoT)        | 87.38%       | 90.92%                     | 91.38%                     |
| Ministral-8B-Instruct-2410 (CoT)   | 87.38%       | 90.61%                     | 90.15%                     |

Stronger model (better baseline performence) get better results by consulting the references answers from two weaker model

#### Difficulty level break down
Qwen2.5-7B-Instruct:

| Round | Difficulty 0 | Difficulty 1 | Difficulty 2 | Difficulty 3 | Difficulty 4 |
|-------|--------------|--------------|--------------|--------------|--------------|
| 0     | 99.26        | 96.0         | 89.71        | 86.18        | 80.0         |
| 1     | 99.26        | 94.4         | 94.12        | 88.62        | 76.92        |
| 2     | 98.53        | 96.0         | 93.38        | 88.62        | 80.0         |

Llama-3.1-8B-Instruct

| Round | Difficulty 0 | Difficulty 1 | Difficulty 2 | Difficulty 3 | Difficulty 4 |
|-------|--------------|--------------|--------------|--------------|--------------|
| 0     | 97.06        | 92.0         | 88.24        | 88.62        | 70.77        |
| 1     | 99.26        | 94.4         | 92.65        | 88.62        | 79.23        |
| 2     | 99.26        | 95.2         | 94.85        | 88.62        | 78.46        |

Ministral-8B-Instruct-2410

| Round | Difficulty 0 | Difficulty 1 | Difficulty 2 | Difficulty 3 | Difficulty 4 |
|-------|--------------|--------------|--------------|--------------|--------------|
| 0     | 97.06        | 92.8         | 88.97        | 82.93        | 74.62        |
| 1     | 97.79        | 97.6         | 93.38        | 85.37        | 78.46        |
| 2     | 98.53        | 95.2         | 91.91        | 85.37        | 79.23        |

#### Upper bound for 3 models

Union of the answers from 3 models 

accuracy: 95.08, 93.54, 92.46

| Round | Difficulty 0 | Difficulty 1 | Difficulty 2 | Difficulty 3 | Difficulty 4 |
|-------|--------------|--------------|--------------|--------------|--------------|
| 0     | 100.0        | 99.2         | 95.59        | 94.31        | 86.15        |
| 1     | 99.26        | 98.4         | 96.32        | 90.24        | 83.08        |
| 2     | 99.26        | 96.0         | 94.85        | 89.43        | 82.31        |

#### 3 model communication include self answer with history

Overall: 92.15%, 92.46%, 92.15%
| Round | Difficulty 0 | Difficulty 1 | Difficulty 2 | Difficulty 3 | Difficulty 4 |
|-------|--------------|--------------|--------------|--------------|--------------|
| 0     | 99.26        | 96.8         | 94.12        | 87.8         | 82.31        |
| 1     | 98.53        | 96.8         | 93.38        | 90.24        | 83.08        |
| 2     | 98.53        | 95.2         | 92.65        | 90.24        | 83.85        |

| Model                              | base (noref) | reference self+other (2 r) | reference self+other (3 r) |
|------------------------------------| -------------| -------------------------- |----------------------------|
| Qwen2.5-7B-Instruct (CoT)          | 90.77%       | 91.85%                     | 92.46%                     |
| Llama-3.1-8B-Instruct (CoT)        | 87.85%       | 90.31%                     | 90.46%                     |
| Ministral-8B-Instruct-2410 (CoT)   | 88.15%       | 90.31%                     | 90.46%                     |

Qwen2.5-7B-Instruct (CoT)

| Round | Difficulty 0 | Difficulty 1 | Difficulty 2 | Difficulty 3 | Difficulty 4 |
|-------|--------------|--------------|--------------|--------------|--------------|
| 0     | 99.26        | 96.0         | 90.44        | 86.99        | 80.77        |
| 1     | 98.53        | 96.0         | 92.65        | 89.43        | 82.31        |
| 2     | 98.53        | 96.0         | 92.65        | 91.06        | 83.85        |

Llama-3.1-8B-Instruct (CoT)

| Round | Difficulty 0 | Difficulty 1 | Difficulty 2 | Difficulty 3 | Difficulty 4 |
|-------|--------------|--------------|--------------|--------------|--------------|
| 0     | 97.79        | 92.8         | 88.24        | 88.62        | 71.54        |
| 1     | 99.26        | 93.6         | 91.91        | 89.43        | 76.92        |
| 2     | 99.26        | 92.8         | 91.91        | 87.8         | 80.0         |

Ministral-8B-Instruct-2410 (CoT)

| Round | Difficulty 0 | Difficulty 1 | Difficulty 2 | Difficulty 3 | Difficulty 4 |
|-------|--------------|--------------|--------------|--------------|--------------|
| 0     | 98.53        | 91.2         | 91.91        | 85.37        | 73.08        |
| 1     | 99.26        | 95.2         | 91.91        | 89.43        | 75.38        |
| 2     | 99.26        | 95.2         | 92.65        | 90.24        | 74.62        |





