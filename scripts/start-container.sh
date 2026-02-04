#! /bin/bash
#SBATCH -J ollama-alekele
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=48G
#SBATCH -p short
#SBATCH --nodelist=neptune

apptainer exec --nv /data/users/alekele/ollama.sif ollama serve
