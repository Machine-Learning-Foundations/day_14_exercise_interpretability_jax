#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=A40short
#SBATCH --job-name=tr_net
#SBATCH --output=./out/train_net-%j.out
#SBATCH --error=./out/train_net-%j.err
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task=8
echo "training network..."

module load cuDNN
module load Python
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt

python ./src/mnist_integrated.py