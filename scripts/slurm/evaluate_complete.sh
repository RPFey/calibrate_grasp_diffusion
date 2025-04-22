#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --mem=64G
#SBATCH --cpus-per-task=24
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --exclude=mp-2080ti-0.grasp.maas,dj-2080ti-0.grasp.maas,kd-2080ti-1.grasp.maas,kd-2080ti-2.grasp.maas,kd-2080ti-3.grasp.maas,kd-2080ti-4.grasp.maas,enough-oryx.grasp.maas,ee-3090-1.grasp.maas,ee-3090-0.grasp.maas
##SBATCH --partition=kostas-compute
##SBATCH --qos=kd-high
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=1-00:00:00
#SBATCH --signal=SIGUSR1@180
#SBATCH --output=./logs/%x-%j.out

cd ~/calibrate_grasp_diffusion

SPEC_FILE=$1

hostname
echo $SPEC_FILE

source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate se3diff
export PYOPENGL_PLATFORM=egl

whereis python3.11
srun python3.11 \
        scripts/train/train_scene_pointcloud_6d_grasp_diffusion.py \
        --saving_root logs/complete_prev_375b35/ \
        --data_root /mnt/kostas-graid/datasets/boshu/grasp/data/scene_2048/ \
        --eval --spec_file ${SPEC_FILE}
