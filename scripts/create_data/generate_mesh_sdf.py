import os
import pickle
import h5py
import shutil
import random
import threading
import open3d as o3d
from tqdm import tqdm
from pathlib import Path
import matplotlib as mpl
import time
import signal
from mesh_to_sdf import sample_sdf_near_surface, get_surface_point_cloud

import trimesh
import logging
logging.getLogger("trimesh").setLevel(9000)
import numpy as np
from sklearn.neighbors import KDTree
import math
import pyrender
import argparse

from se3dif.utils import makedirs
import acronym_tools
from acronym_tools import Scene, load_mesh, load_grasps, create_gripper_marker

DATA_FOLDER = 'data'
OBJ_CLASSES = ['Cup', 'Mug', 'Fork', 'Hat', 'Bottle', 'Bowl', 'Car', 'Donut', 'Laptop', 'MousePad', 'Pencil',
                'Plate', 'ScrewDriver', 'WineBottle','Backpack', 'Bag', 'Banana', 'Battery', 'BeanBag', 'Bear',
                'Book', 'Books', 'Camera','CerealBox', 'Cookie','Hammer', 'Hanger', 'Knife', 'MilkCarton', 'Painting',
                'PillBottle', 'Plant','PowerSocket', 'PowerStrip', 'PS3', 'PSP', 'Ring', 'Scissors', 'Shampoo', 'Shoes',
                'Sheep', 'Shower', 'Sink', 'SoapBottle', 'SodaCan','Spoon', 'Statue', 'Teacup', 'Teapot', 'ToiletPaper',
                'ToyFigure', 'Wallet','WineGlass', 'Cow', 'Sheep', 'Cat', 'Dog', 'Pizza', 'Elephant', 'Donkey', 'RubiksCube', 'Tank', 'Truck', 'USBStick']

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException("Function timed out!")

def run_with_timeout(func, timeout, *args, **kwargs):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)  # Set an alarm for `timeout` seconds
    try:
        result = func(*args, **kwargs)
    except TimeoutException:
        result = -1  # Handle timeout case
    except np.linalg.LinAlgError as e:
        result = -1
    finally:
        signal.alarm(0)  # Disable alarm
    return result

#OBJ_CLASSES = ['Bottle']
## Set data folder
base_folder = os.path.abspath(os.path.dirname(__file__)+'/..')
root_folder = os.path.abspath(os.path.join(base_folder, '..'))
data_folder = os.path.join(root_folder, DATA_FOLDER)
grasps_folder = os.path.join(data_folder, 'grasps')
meshes_folder = os.path.join(data_folder, 'meshes')
sdf_folder = os.path.join(data_folder, 'scene_sdf')
makedirs(sdf_folder)

## Copied from mesh_to_sdf
def get_unit_spherize_scale(mesh):
    """
    Get the scale factor to spherize the mesh

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    return np.max(distances)

def generate_mesh_sdf(mesh, absolute=True, normalize=False, n_points=200000):
    """
    Generate a signed distance field for a mesh
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh
    absolute : bool
        Whether to take the absolute value of the SDF
    normalize : bool
        Whether to normalize the SDF to [0, 1]
    n_points : int
        The number of points to sample
    """
    q_sdf, pcl = sample_sdf_near_surface(mesh, number_of_points=n_points, return_gradients=False)
    query_points, sdf = q_sdf[0], q_sdf[1]

    if absolute:
        neg_sdf_idxs = np.argwhere(sdf<0)[:,0]
        sdf[neg_sdf_idxs] = -sdf[neg_sdf_idxs]

    if normalize:
        sdf_max = sdf.max()
        sdf_min = sdf.min()
        sdf = (sdf - sdf_min) / (sdf_max - sdf_min)

    return query_points, sdf

class PyrenderScene(Scene):
    def as_pyrender_scene(self):
        """Return pyrender scene representation.

        Returns:
            pyrender.Scene: Representation of the scene
        """
        pyrender_scene = pyrender.Scene()
        for obj_id, obj_mesh in self._objects.items():
            mesh = pyrender.Mesh.from_trimesh(obj_mesh, smooth=False)
            pyrender_scene.add(mesh, name=obj_id, pose=self._poses[obj_id])
        return pyrender_scene


class SceneRenderer:
    def __init__(
        self,
        pyrender_scene,
        fov=np.pi / 6.0,
        width=400,
        height=400,
        aspect_ratio=1.0,
        z_near=0.001,
    ):
        """Create an image renderer for a scene.

        Args:
            pyrender_scene (pyrender.Scene): Scene description including object meshes and their poses.
            fov (float, optional): Field of view of camera. Defaults to np.pi/6.
            width (int, optional): Width of camera sensor (in pixels). Defaults to 400.
            height (int, optional): Height of camera sensor (in pixels). Defaults to 400.
            aspect_ratio (float, optional): Aspect ratio of camera sensor. Defaults to 1.0.
            z_near (float, optional): Near plane closer to which nothing is rendered. Defaults to 0.001.
        """
        self._fov = fov
        self._width = width
        self._height = height
        self._z_near = z_near
        self._scene = pyrender_scene

        self._camera = pyrender.PerspectiveCamera(
            yfov=fov, aspectRatio=aspect_ratio, znear=z_near
        )

    def get_trimesh_camera(self):
        """Get a trimesh object representing the camera intrinsics.

        Returns:
            trimesh.scene.cameras.Camera: Intrinsic parameters of the camera model
        """
        return trimesh.scene.cameras.Camera(
            fov=(np.rad2deg(self._fov), np.rad2deg(self._fov)),
            resolution=(self._height, self._width),
            z_near=self._z_near,
        )

    def _to_pointcloud(self, depth, color_mask=None):
        """Convert depth image to pointcloud given camera intrinsics.

        Args:
            depth (np.ndarray): Depth image.

        Returns:
            np.ndarray: Point cloud.
        """
        fy = fx = 0.5 / np.tan(self._fov * 0.5)  # aspectRatio is one.
        height = depth.shape[0]
        width = depth.shape[1]

        mask = np.where(depth > 0)

        x = mask[1]
        y = mask[0]
        
        if color_mask is not None:
            pts_mask = color_mask[y, x]
        else:
            pts_mask = None

        normalized_x = (x.astype(np.float32) - width * 0.5) / width
        normalized_y = (y.astype(np.float32) - height * 0.5) / height

        world_x = normalized_x * depth[y, x] / fx
        world_y = normalized_y * depth[y, x] / fy
        world_z = depth[y, x]
        ones = np.ones(world_z.shape[0], dtype=np.float32)

        return np.vstack((world_x, world_y, world_z, ones)).T, pts_mask

    def render(self, camera_pose, target_id="", render_pc=True):
        """Render RGB/depth image, point cloud, and segmentation mask of the scene.

        Args:
            camera_pose (np.ndarray): Homogenous 4x4 matrix describing the pose of the camera in scene coordinates.
            target_id (str, optional): Object ID which is used to create the segmentation mask. Defaults to ''.
            render_pc (bool, optional): If true, point cloud is also returned. Defaults to True.

        Returns:
            np.ndarray: Color image.
            np.ndarray: Depth image.
            np.ndarray: Point cloud.
            np.ndarray: Segmentation mask.
        """
        # Keep local to free OpenGl resources after use
        renderer = pyrender.OffscreenRenderer(
            viewport_width=self._width, viewport_height=self._height
        )

        # add camera and light to scene
        scene = self._scene.as_pyrender_scene()
        scene.add(self._camera, pose=camera_pose, name="camera")
        light = pyrender.SpotLight(
            color=np.ones(4),
            intensity=3.0,
            innerConeAngle=np.pi / 16,
            outerConeAngle=np.pi / 6.0,
        )
        scene.add(light, pose=camera_pose, name="light")

        # render the full scene
        color, depth = renderer.render(scene)

        segmentation = np.zeros(depth.shape, dtype=np.uint8)

        # hide all objects
        for node in scene.mesh_nodes:
            node.mesh.is_visible = False

        # Render only target object and add to segmentation mask
        for node in scene.mesh_nodes:
            if node.name == target_id:
                node.mesh.is_visible = True
                _, object_depth = renderer.render(scene)
                mask = np.logical_and(
                    (np.abs(object_depth - depth) < 1e-6), np.abs(depth) > 0
                )
                segmentation[mask] = 1

        for node in scene.mesh_nodes:
            node.mesh.is_visible = True

        if render_pc:
            pc, pc_mask = self._to_pointcloud(depth, segmentation)
        else:
            pc, pc_mask = None, None

        return color, depth, segmentation, pc, pc_mask

if __name__ == '__main__':
    
    args = argparse.ArgumentParser()
    args.add_argument("--mesh_root", type=str, default="/root/calibrate_grasp_diffusion/data")
    args.add_argument("--support", type=str, default="/root/calibrate_grasp_diffusion/data/grasps/Table/Table_19e80d699bcbd3168821642e9a54505_0.004756380229523233.h5")
    args.add_argument(
        "--support_scale", default=0.025, help="Scale factor of support mesh."
    )
    args.add_argument("--viz", action="store_true", default=False)
    
    # support = "/root/calibrate_grasp_diffusion/data/grasps/Table/Table_19e80d699bcbd3168821642e9a54505_0.004756380229523233.h5"
    args = args.parse_args()
    support = args.support
    
    # load the support mesh
    support_mesh = load_mesh(
        support, mesh_root_dir=args.mesh_root, scale=(0.01, 0.0075, 0.01)
    )

    scene_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    for obj_cls in OBJ_CLASSES:
        grasp_cls_folder = os.path.join(grasps_folder, obj_cls)
        for filename in tqdm(os.listdir(grasp_cls_folder), desc="Processing {}".format(obj_cls)):
            ## Load mesh
            target_mesh = load_mesh(os.path.join(grasp_cls_folder, filename), mesh_root_dir=args.mesh_root)
            if not isinstance(target_mesh, trimesh.Trimesh):
                continue
            extents = target_mesh.bounding_box.extents
            ball_range = np.max(extents) * 1.5
            
            # check collisions
            T, success = load_grasps(os.path.join(grasp_cls_folder, filename))
            
            # TODO Add Some other Random objects.
            target_name = "obj0"
            scene = PyrenderScene()
            scene.add_object("support_object", support_mesh, pose=np.eye(4), support=True)
            result = run_with_timeout(scene.place_object, 10, target_name, target_mesh)
            if result == -1:
                print("Failed to place object")
                continue
            
            trials = 0
            while len(scene._objects) < 5 and trials < 20:
                name = np.random.choice(OBJ_CLASSES)
                collision_cls_folder = os.path.join(grasps_folder, name)
                random_mesh_filename = np.random.choice(os.listdir(collision_cls_folder))
                random_mesh = load_mesh(
                    os.path.join(collision_cls_folder, random_mesh_filename),
                    mesh_root_dir=args.mesh_root,
                )
                if isinstance(random_mesh, trimesh.Trimesh):
                    scale = random.random() + 0.5
                    random_mesh.apply_scale(scale)
                    result = run_with_timeout(scene.place_object, 10, f"obj{len(scene._objects)}", random_mesh)
                    if result == -1:
                        print("Failed to place object")
                        continue
                    print(trials)
                    trials += 1

            # Crop the target mesh
            target_T = scene._poses[target_name]
            moc_T = scene.get_transform(target_name, "com")
            mos = moc_T[:3, 3]
            obj_pose = scene._poses[target_name]
            gripper_mesh = trimesh.load(
                Path(acronym_tools.__file__).parent.parent / "data/franka_gripper_collision_mesh.stl"
            )
            collision_free = np.array(
                [
                    i
                    for i, t in enumerate(T[success == 1])
                    if not scene.in_collision_with(
                        gripper_mesh, transform=np.dot(obj_pose, t)
                    )
                ]
            )
            print("Number of collision free grasps: ", len(collision_free))
            if len(collision_free) < 2:
                continue
            
            query_pts, _ = generate_mesh_sdf(target_mesh)
            query_pts = query_pts @ target_T[:3, :3].T + target_T[:3, 3]
            distance = np.linalg.norm(query_pts - mos[None, :], axis=1)
            query_pts = query_pts[distance < ball_range]
            
            # get o3d mesh and compute SDF
            o3d_mesh = scene.as_open3d_scene()     
            ray_scene = o3d.t.geometry.RaycastingScene()
            for m in o3d_mesh:
                m_t = o3d.t.geometry.TriangleMesh.from_legacy(m)
                ray_scene.add_triangles(m_t)    
            query_point = o3d.core.Tensor(query_pts, dtype=o3d.core.Dtype.Float32)
            unsigned_distance = ray_scene.compute_distance(query_point)
            signed_distance = ray_scene.compute_signed_distance(query_point)
            signed_distance = signed_distance.numpy()
            
            # ball query
            # pts = scene.sample_points(pts_density = 2e5)
            # scene_pts = []
            # target_index = []
            # target_center = np.zeros((3, ))
            # for name, pt in pts.items():
            #     if name != target_name:
            #         distance = np.linalg.norm(pt - mos[None, :], axis=1)
            #         crop_pt = pt[distance < 2 * ball_range]
            #         target_index.append(np.zeros((len(crop_pt), ), dtype=np.uint8))
            #     else:
            #         crop_pt = pt
            #         target_center = pt.mean(axis=0)
            #         target_index.append(np.ones((len(crop_pt), ), dtype=np.uint8))
                
            #     scene_pts.append(crop_pt)
            # scene_pts = np.concatenate(scene_pts, axis=0)
            # target_index = np.concatenate(target_index, axis=0)

            # choose camera intrinsics and extrinsics
            renderer = SceneRenderer(scene, fov=np.pi / 3, width=640, height=480)
            trimesh_camera = renderer.get_trimesh_camera()
            feasible_poses = T[success == 1][collision_free]
            rand_idx = np.random.choice(len(feasible_poses))
            camera_pose = feasible_poses[rand_idx] @ np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]) # change x right, y up, z back
            distance = np.random.uniform(low=0.7, high=0.9)
            camera_pose[:3, 3] = camera_pose[:3, 3] + camera_pose[:3, 2] * distance
            color, depth, segmentation, scene_pts, target_index = renderer.render(
                camera_pose=camera_pose, target_id=target_name
            )
            
            if segmentation.max() == 0:
                print("No target object in the scene")
                continue

            c2w = camera_pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            scene_pts = scene_pts @ c2w.T
            target_center = np.sum(scene_pts * target_index[:, None], axis=0) / np.sum(target_index)
            scene_pts = scene_pts[:, :3]
            target_center = target_center[:3]
        
            # visualize the scene
            # object_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            # object_coord.transform(target_T)
            # cam_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            # cam_coord.transform(c2w)
            # o3d_mesh.append(scene_origin)
            # o3d_mesh.append(object_coord)
            # o3d.visualization.draw_geometries(o3d_mesh)
                        
            # Centralize & Transform
            centralize = np.eye(4)
            centralize[:3, 3] = -target_center
            scene_pts = scene_pts - target_center
            query_pts = query_pts - target_center

            grasps = []
            for t in T[success == 1][collision_free]:
                transform = centralize @ obj_pose @ t # @ 
                grasps.append(transform)
            grasps = np.stack(grasps, axis=0)
            
            # collect Failure grasps
            F_grasps = T[success == 0]
            collision_label = np.ones((T[success == 1].shape[0], ), np.uint8)
            collision_label[collision_free] = 0
            collistion_grasp = feasible_poses[collision_label]
            neg_grasps = np.concatenate([F_grasps, collistion_grasp], axis=0)
            F_grasps = []
            for t in neg_grasps:
                transform = centralize @ obj_pose @ t
                F_grasps.append(transform)
            F_grasps = np.stack(F_grasps, axis=0)

            # visualize the crop scene
            if args.viz:
                crop_pcd = o3d.geometry.PointCloud()
                crop_pcd.points = o3d.utility.Vector3dVector(scene_pts)
                grasps_o3d = []            
                for idx, t in enumerate(T[success == 1][collision_free]):
                    transform = centralize @ obj_pose @ t # @ 
                    g = create_gripper_marker(color=[0, 255, 0])
                    g.apply_transform(transform)

                    m = o3d.geometry.TriangleMesh()
                    m.vertices = o3d.utility.Vector3dVector(g.vertices)
                    m.triangles = o3d.utility.Vector3iVector(g.faces)
                    m.paint_uniform_color([0, 1, 0])
                    grasps_o3d.append(m)

                    grasp_origin = grasps[idx]
                    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                    origin.transform(grasp_origin)
                    grasps_o3d.append(origin)

                cmap = mpl.cm.get_cmap("plasma")
                colors = cmap((signed_distance - signed_distance.min()) / (signed_distance.max() - signed_distance.min()))[:, :3]
                pcd_sdf = o3d.geometry.PointCloud()
                pcd_sdf.points = o3d.utility.Vector3dVector(query_pts)
                pcd_sdf.colors = o3d.utility.Vector3dVector(colors)
                o3d.visualization.draw_geometries([crop_pcd, scene_origin, origin, *grasps_o3d])

            ## save info
            save_sdf_folder = os.path.join(sdf_folder, obj_cls)
            makedirs(save_sdf_folder)

            sdf_mesh = filename.split('.obj')[0] + '.npz'
            save_file = os.path.join(save_sdf_folder, sdf_mesh)
            sdf_dict = {
                'scene_pts': scene_pts,
                "target_index": target_index,
                'xyz': query_pts,
                'sdf': signed_distance,
                'grasps': grasps,
                "F_grasps": F_grasps
            }

            np.savez(save_file, **sdf_dict)

