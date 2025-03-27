import os
import numpy as np
import open3d as o3d
import torch

base_dir = os.path.abspath(os.path.dirname(__file__))
pts_dir = os.path.join(base_dir)

def get_3d_pts(
        file = os.path.join(pts_dir,'UniformPts.npy'), 
        scale = np.ones(3), loc = np.zeros(3), n_points=100):
    """
        file: str 
            path to the 3D points file
        scale: scale of the 3D points
            default is 1.
        loc: location of the 3D points 
            offset for Franka gripper is [0, 0, 0.06]
        n_points: number of points to
    """
    pts = np.load(file)
    # if len(pts) >= n_points:
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(pts)
    #     pcd = pcd.farthest_point_down_sample(n_points)
    #     pts = np.asarray(pcd.points)

    pts = (pts[:n_points, :] + loc) * scale # + loc
    return torch.Tensor(pts)

if __name__ == '__main__':
    
    pts = get_3d_pts(scale=0.2, n_points=30)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.numpy())
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([pcd, coord])