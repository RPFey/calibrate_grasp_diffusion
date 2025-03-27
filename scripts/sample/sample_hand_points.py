import open3d as o3d
import numpy as np

if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--hand_obj", type=str, default="/root/data")
    args.add_argument("--finger_obj", type=str, default="/root/data")
    opt = args.parse_args()
    
    hand_mesh = o3d.io.read_triangle_mesh(opt.hand_obj)
    left_finger_mesh = o3d.io.read_triangle_mesh(opt.finger_obj)
    right_finger_mesh = o3d.io.read_triangle_mesh(opt.finger_obj)
    
    left_transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0.04], [0, 0, 1, 0.0584], [0, 0, 0, 1]])
    left_finger_mesh.transform(left_transform)
    right_transform = np.array([[-1, 0, 0, 0], [0, -1, 0, -0.04], [0, 0, 1, 0.0584], [0, 0, 0, 1]])
    right_finger_mesh.transform(right_transform)
    
    
    hand_pts = hand_mesh.sample_points_uniformly(10000)
    left_finger_pts = left_finger_mesh.sample_points_uniformly(1000)
    right_finger_pts = right_finger_mesh.sample_points_uniformly(1000)
    
    gripper_pts = hand_pts + left_finger_pts + right_finger_pts
    gripper_pts = gripper_pts.farthest_point_down_sample(512)
    
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([gripper_pts, coord])
    np.save("gripper_pts.npy", np.asarray(gripper_pts.points))