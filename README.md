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
