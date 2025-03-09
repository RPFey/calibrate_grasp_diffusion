# Pytorch implementation of Diffusion models in SE(3) for grasp and motion generation

This library provides the tools for training and sampling diffusion models in SE(3),
implemented in PyTorch. 
We apply them to learn 6D grasp distributions. We use the learned distribution as cost function
for grasp and motion optimization problems.
See reference [1] for additional details.

[[Website]](https://sites.google.com/view/se3dif/home)      [[Preprint]](https://arxiv.org/pdf/2209.03855.pdf)

<img src="assets/grasp_dif.gif" alt="diffusion" style="width:800px;"/>

## Installation

Refer to `build.sh` file

## Prepare data

You can also use the data under `\mnt` folder.

```bash
ln -s /mnt/kostas-graid/datasets/boshu/data data
```

## Run

Train using scene point cloud

```bash
cd scripts
python train/train_scene_pointcloud_6d_grasp_diffusion.py --spec_file multiobject_scene_graspdif --saving_root ../logs/ --data_root ../data/scene_2048/

# Headless mode
PYOPENGL_PLATFORM=egl python train/train_scene_pointcloud_6d_grasp_diffusion.py --spec_file multiobject_scene_graspdif --saving_root ../logs/ --data_root /mnt/kostas-graid/datasets/boshu/grasp/data/scene_2048
```

## Experiment

```bash
##############
# Ablation 1 #
##############

# Complete Scene 
PYOPENGL_PLATFORM=egl python train/train_scene_pointcloud_6d_grasp_diffusion.py --spec_file multiobject_scene_graspdif --saving_root ../logs/ --data_root /mnt/kostas-graid/datasets/boshu/grasp/data/scene_2048

# Only Targets Point Cloud.
PYOPENGL_PLATFORM=egl python train/train_scene_pointcloud_6d_grasp_diffusion.py --spec_file multiobject_partialp_graspdif --saving_root ../logs/ --data_root /mnt/kostas-graid/datasets/boshu/grasp/data/partial_scene_2048
```


## Exp Logs

March 09 2025

Some experiments I have tried: 

1. baseline `sbatch scripts/slurm/pointcloud.sh` -- this is the baseline in the paper

2. train with scene information `sbatch scripts/slurm/run.sh multiobject_scene_graspdif`

3. train with cross entropy loss (Bernoulli Parameterization) `sbatch scripts/slurm/run.sh multiobject_scene_graspdif_bernoulli_ce`

4. train with AP Loss (Bernoulli Parameterization) `sbatch scripts/slurm/run.sh multiobject_scene_graspdif_bernoulli_ap`

5. train with Dirichlet distribution `sbatch scripts/slurm/run.sh multiobject_scene_graspdif_dirichlet`

5. train with updated Score Matching Loss `sbatch scripts/slurm/run.sh multiobject_scene_graspdif_fix`
