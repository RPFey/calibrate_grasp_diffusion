#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a40:1
#SBATCH --nodes=1
##SBATCH --partition=kostas-compute
##SBATCH --qos=kd-high
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00
#SBATCH --signal=SIGUSR1@180
#SBATCH --output=./logs/cluster/%x-%j.out

cd ~/calibrate_grasp_diffusion
mkdir -p logs/cluster

SPEC_FILE=$1

hostname
echo $SLURM_ARRAY_TASK_ID '/' $SLURM_ARRAY_TASK_COUNT
echo $SPEC_FILE

source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate se3diff
export PYOPENGL_PLATFORM=egl

whereis python3.11
srun python3.11 \
        scripts/train/train_scene_pointcloud_6d_grasp_diffusion.py \
        --spec_file ${SPEC_FILE} --saving_root ./logs/ \
        --data_root ./data/scene_2048/
