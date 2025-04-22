#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --gpus=a40:1
#SBATCH --nodes=1
#SBATCH --exclude=mp-2080ti-0.grasp.maas,dj-2080ti-0.grasp.maas,kd-2080ti-1.grasp.maas,kd-2080ti-2.grasp.maas,kd-2080ti-3.grasp.maas,kd-2080ti-4.grasp.maas,enough-oryx.grasp.maas,ee-3090-1.grasp.maas,ee-3090-0.grasp.maas
#SBATCH --partition=kostas-compute
#SBATCH --qos=kd-high
##SBATCH --partition=batch
##SBATCH --qos=normal
#SBATCH --time=1-00:00:00
#SBATCH --signal=SIGUSR1@180
#SBATCH --output=./logs/bullet_evaluate/%x-%j.out

cd ~/calibrate_grasp_diffusion
mkdir -p logs/bullet_evaluate

SPEC_FILE=$1
WEIGHT=$2
NUM_OBJECTS=${3:-2} # Default to 2 if not provided
NUM_SEEDS=${4:-4} # Default to 4 if not provided, this is for sampling multiple seeds
NUM_GRASPS=${5}

hostname
echo $SPEC_FILE
echo $WEIGHT
echo $NUM_OBJECTS
echo $NUM_SEEDS
echo $NUM_GRASPS

source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate se3diff
export PYOPENGL_PLATFORM=egl

whereis python3.11
srun python3.11 \
        se3dif/samplers/bullet_sampler.py \
        --num_grasps ${NUM_GRASPS} \
        --spec_file ${SPEC_FILE} --weight ${WEIGHT} \
        --num_objects ${NUM_OBJECTS} --num_seeds ${NUM_SEEDS} --num_processes 16
