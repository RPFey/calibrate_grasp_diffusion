import numpy as np
import torchvision
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import pyglet

from se3dif.samplers import Grasp_AnnealedLD
from se3dif.utils import to_numpy
from se3dif.visualization import grasp_visualization


def denoising_summary(model, model_input, ground_truth, info, writer, iter, prefix=""):
    observation = model_input['visual_context']
    batch = 8

    ## 1. visualize generated grasps ##
    model.eval()
    target_index = model_input.get('target_index', None)
    if target_index is not None:
        target_index = target_index[:1, ...]
        
    model.set_latent(observation[:1, ...], target_index=target_index)
    generator = Grasp_AnnealedLD(model, batch=batch, T=30, T_fit=50, device=observation.device)
    H, energy = generator.sample() #[0]

    H = to_numpy(H)
    H[:, :3, -1]*=1/8
    energy = to_numpy(energy).reshape(-1)
    if observation.dim()==3:
        point_cloud = to_numpy(model_input['visual_context'])[0,...]/8.
    else:
        point_cloud = to_numpy(model_input['point_cloud'])[0,...]/8.
    
    if target_index is not None:
        target_index = target_index[0, :, 0].cpu().numpy()

    try:
        image = grasp_visualization.get_scene_grasps_image(H, energies=energy, p_cloud=point_cloud, target_index=target_index)
        figure = plt.figure()
        plt.imshow(image)
        if writer is not None:
            writer.add_figure("diffusion/generated_grasps", figure, global_step=iter)
        else:
            plt.savefig(f"generated_grasps_{iter}.png")
    except pyglet.window.NoSuchConfigException as e:
        print("pyglet window not found, skipping grasp visualization")
