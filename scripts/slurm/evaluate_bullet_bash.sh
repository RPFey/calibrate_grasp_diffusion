SPEC_FILE=$1
WEIGHT=$2
NUM_OBJECTS=${3:-2} # Default to 2 if not provided
NUM_SEEDS=${4:-4} # Default to 4 if not provided, this is for sampling multiple seeds
NUM_GRASPS=${5}

echo $SPEC_FILE
echo $WEIGHT
echo $NUM_OBJECTS
echo $NUM_SEEDS
echo $NUM_GRASPS

source /mnt/kostas-graid/sw/envs/boshu/miniconda3/bin/activate se3diff
export PYOPENGL_PLATFORM=egl

whereis python3.11
python3.11 \
        se3dif/samplers/bullet_sampler.py \
        --num_grasps ${NUM_GRASPS} \
        --spec_file ${SPEC_FILE} --weight ${WEIGHT} \
        --num_objects ${NUM_OBJECTS} --num_seeds ${NUM_SEEDS} --num_processes 16
