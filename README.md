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

```bash
ln -s /mnt/kostas-graid/datasets/boshu/data data
```

## Run

Train using scene point cloud

```bash
cd scripts
python train/train_scene_pointcloud_6d_grasp_diffusion.py --spec_file multiobject_scene_graspdif --saving_root ../logs/ --data_root ../data/scene_2048/

# Headless mode
PYOPENGL_PLATFORM=egl python train/train_scene_pointcloud_6d_grasp_diffusion.py --spec_file multiobject_scene_graspdif --saving_root ../logs/ --data_root ../data/scene_2048/
```
