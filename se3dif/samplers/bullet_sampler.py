import sksparse.cholmod as skch

import numpy as np
import torch
import os, os.path as osp
import open3d as o3d
import glob
from torch.utils.data import Dataset, DataLoader

import theseus as th
from theseus import SO3
from se3dif.utils import SO3_R3
# sample poses 
from se3dif.samplers import ApproximatedGrasp_AnnealedLD, Grasp_AnnealedLD
from se3dif.datasets.acronym_dataset import PointcloudSceneAcronymAndSDFDataset

from franka_env.grasp_generator import ClutterRemovalSim, render_images, evaluate_grasp_pose, Label
from franka_env.btsim import Rotation, Transform, CameraIntrinsic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BulletEvaluator:
    def __init__(self, save_dir, num_grasps = 512):
        self.num_grasps = num_grasps
        self.save_dir = save_dir
    
    def evaluate_model(self, model, total_timestep, num_objects, seed=42, viz=False):
        if isinstance(num_objects, int):
            num_objects = [num_objects]
        
        sampler = Grasp_AnnealedLD(model, batch=self.num_grasps,
                                        T=70, T_fit=50, k_steps=1, 
                                        device=device)
        save_dir = os.path.join(self.save_dir, "{:06d}".format(total_timestep))
        os.makedirs(save_dir, exist_ok=True)
        
        total_trials = []
        success_trials = []
        
        for num_object in num_objects:
            total_trial = 0
            success_trial = 0
            
            sim = ClutterRemovalSim("pile", gui=viz, seed=seed)
            sim.reset(num_object)
            sim.save_state()
            
            # render synthetic depth images
            # MAX_VIEWPOINT_COUNT = 4
            n = 8 # np.random.randint(MAX_VIEWPOINT_COUNT) + 1
            depth_imgs, extrinsics, segs = render_images(sim, n)
            
            # create point cloud using open3d
            intrinsics = o3d.camera.PinholeCameraIntrinsic(
                width=depth_imgs[0].shape[1],
                height=depth_imgs[0].shape[0],
                fx=sim.camera.intrinsic.fx,
                fy=sim.camera.intrinsic.fy,
                cx=sim.camera.intrinsic.cx,
                cy=sim.camera.intrinsic.cy
            )
            
            # find all visible indices
            all_segs = np.stack(segs, axis=0)
            unique_ids = np.unique(all_segs)
            
            for target in unique_ids:
                # skip desk and box
                if target in [0, 1]:
                    continue
                
                # scene point cloud
                target_pcd = o3d.geometry.PointCloud()
                scene_pcd = o3d.geometry.PointCloud()
                
                # back project 
                for depth, ext, seg in zip(depth_imgs, extrinsics, segs):
                    depth_image = o3d.geometry.Image(depth * (seg == target))
                    target_pcd += o3d.geometry.PointCloud.create_from_depth_image(
                        depth_image, intrinsics, extrinsic = Transform.from_list(ext).as_matrix(), depth_scale=1.0, depth_trunc=2.0
                    )
                    
                    depth_image = o3d.geometry.Image(depth * (seg != target))
                    scene_pcd += o3d.geometry.PointCloud.create_from_depth_image(
                        depth_image, intrinsics, extrinsic = Transform.from_list(ext).as_matrix(), depth_scale=1.0, depth_trunc=2.0
                    )
                
                # normalize & scale
                if np.asarray(target_pcd.points).shape[0] > 512:
                    target_pcd = target_pcd.farthest_point_down_sample(512)
                if np.asarray(scene_pcd.points).shape[0] > 512:
                    scene_pcd = scene_pcd.farthest_point_down_sample(512)
                target_mean = np.mean(np.asarray(target_pcd.points), axis=0)
                
                complete_pc = np.concatenate([np.asarray(target_pcd.points), np.asarray(scene_pcd.points)], axis=0)
                target_index = np.concatenate([np.ones(len(target_pcd.points)), np.zeros(len(scene_pcd.points))], axis=0)
                
                complete_pc_norm = (complete_pc - target_mean) * 8
                # sample poses
                if model.num_scene_points > 0:
                    complete_pc_norm_t = torch.from_numpy(complete_pc_norm).to(device).view(1, -1, 3).float()   
                    target_index_t = torch.from_numpy(target_index).to(device).view(1, -1, 1).float()
                    model.set_latent(complete_pc_norm_t, target_index_t)
                else:
                    target_pc = complete_pc_norm[target_index == 1]
                    target_pc = torch.from_numpy(target_pc).to(device).view(1, -1, 3).float()
                    model.set_latent(target_pc)    
                
                grasp_poses, scores = sampler.sample()
                grasp_poses = grasp_poses.cpu().numpy()
                grasp_poses[:, :3, 3] = (grasp_poses[:, :3, 3] / 8) + target_mean
                
                # evaluate 
                if len(grasp_poses) > 0:
                    results = []
                    for grasp in grasp_poses:
                        execute_grasp = grasp.copy()
                        # inertial offset 
                        execute_grasp[:3, 3] += 0.04 * execute_grasp[:3, 2]
                        execute_grasp = execute_grasp @ np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
                        sim.restore_state()
                        outcome, width = evaluate_grasp_pose(sim, execute_grasp)
                        if outcome == Label.SUCCESS:
                            print("Grasp success")
                            results.append(1)
                        else:
                            results.append(0)
                    
                    results = np.array(results)
                    total_trial += len(results)
                    success_trial += np.sum(results).item()
                    
                    # one success is enough
                    if np.any(results > 0):
                        # only save successful grasps
                        success_grasps = grasp_poses[results > 0]
                        failure_grasps = grasp_poses[results == 0]
                        
                        np.savez(
                            os.path.join(save_dir, f"seed{seed}-obj{num_object}-t{target}.npz"),
                            scene_pts = complete_pc,
                            target_index = target_index,
                            grasps = success_grasps,
                            F_grasps = failure_grasps,
                        )
                
            success_trials.append(success_trial)
            total_trials.append(total_trial)
        
        return total_trials, success_trials 
                
    def get_dataset(self, timestep, num_objects, dataset_args={}):
        if len(timestep) == 0:
            print("Accquire all time step results")
            timestep = os.listdir(self.save_dir)
        elif isinstance(timestep[0], int):
            timestep = ["{:06d}".format(t) for t in timestep]
        elif isinstance(timestep, int):
            timestep = ["{:06d}".format(timestep)]
        elif isinstance(timestep, str):
            timestep = [timestep]
        
        total_files = []
        for n, t in zip(num_objects, timestep):
            files = glob.glob(os.path.join(self.save_dir, t, f'seed*-obj{n}-t*.npz'))
            total_files += files
        
        dataset = PointcloudSceneAcronymAndSDFDataset(self.save_dir, **dataset_args)
        dataset.files = total_files
        return dataset
    
if __name__ == '__main__':
    from se3dif.datasets import AcronymGraspsDirectory
    from se3dif.models.loader import load_model
    from se3dif.samplers import ApproximatedGrasp_AnnealedLD, Grasp_AnnealedLD
    from se3dif.utils import to_numpy, to_torch, load_experiment_specifications
    from se3dif.visualization import grasp_visualization
    import argparse
    
    args = argparse.ArgumentParser()
    args.add_argument("--spec_file", type=str, default="/root/spec")
    args.add_argument("--weight", type=str, default="/root/data/weights/ckpt.pth")
    args.add_argument("--viz", action='store_true', default=False, help='visualize the grasps')
    opt = args.parse_args()

    save_dir = "/mnt/kostas-graid/datasets/boshu/grasp_buffer"
    evaluator = BulletEvaluator(save_dir)
    
    model_args = load_experiment_specifications(opt.spec_file)
    model_args['device'] = device
    model = load_model(model_args)
    model.to(device)
    
    model_states = torch.load(opt.weight, map_location=device)
    if 'model_state' in model_states:
        model.load_state_dict(model_states["model_state"])
    else:
        model.load_state_dict(model_states)
    
    for s in range(1):
        evaluator.evaluate_model(model, 1000, 2, s, opt.viz)
        
    dataset = evaluator.get_dataset([], [2])
    dataset[0]