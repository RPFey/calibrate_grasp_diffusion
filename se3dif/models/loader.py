import os
import torch
import torch.nn as nn
import numpy as np

from se3dif import models


from se3dif.utils import get_pretrained_models_src, load_experiment_specifications
pretrained_models_dir = get_pretrained_models_src()

# TODO Change to MoE structure.
class DirichletEnergy(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        """ Energy model for Dirichlet distribution 
        
        Args:
            in_dim (int): input dimension
            hidden_dim (int): hidden dimension, exponential of 2
        """
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        """ 
        Args:
            x (torch.Tensor): input tensor of shape (..., in_dim)
        Returns:
            torch.Tensor: output tensor of shape (..., 2)
        """
        vote = self.gate(x)
        num_gates = vote.shape[-1]
        
        positive = torch.sum(vote[..., :num_gates//2], dim=-1, keepdim=True)
        negative = torch.sum(vote[..., num_gates//2:], dim=-1, keepdim=True)
        return torch.cat([positive, negative], dim=-1)

class Exp(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        return torch.exp(x)
    
def load_model(args, saving_root = None) -> models.GraspDiffusionFields:
    pretrained_models_dir_ = saving_root if saving_root is not None else pretrained_models_dir 
    if 'pretrained_model' in args:
        model_args = load_experiment_specifications(os.path.join(pretrained_models_dir_, args['pretrained_model']))
        args["NetworkArch"] = model_args["NetworkArch"]
        args["NetworkSpecs"] = model_args["NetworkSpecs"]
        args["num_scene_points"] = model_args.get("num_scene_points", -1)
        args["num_target_points"] = model_args.get("num_target_points", -1)

    if args['NetworkArch'] == 'GraspDiffusion':
        model = load_grasp_diffusion(args)
    elif args['NetworkArch'] == 'PointcloudGraspDiffusion':
        model = load_pointcloud_grasp_diffusion(args)

    if 'pretrained_model' in args:
        model_path = os.path.join(pretrained_models_dir, args['pretrained_model'], 'model.pth')
        states = torch.load(model_path)
        if 'model_state' in states:
            model.load_state_dict(states['model_state'])
        else:
            model.load_state_dict(states)

        if args['device'] != 'cpu':
            model = model.to(args['device'], dtype=torch.float32)

    elif 'saving_folder' in args:
        load_model_dir = os.path.join(args['saving_folder'], 'checkpoints', 'model_current.pth')
        try:
            if args['device'] == 'cpu':
                states = torch.load(load_model_dir, map_location=torch.device('cpu'))
                model.load_state_dict(states['model_state'])
            else:
                states = torch.load(load_model_dir)
                model.load_state_dict(states['model_state'])
        except:
            pass

    return model


def load_grasp_diffusion(args) -> models.GraspDiffusionFields:
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    lat_params = params['latent_codes']
    points_params = params['points']
    # vision encoder
    vision_encoder = models.vision_encoder.LatentCodes(num_scenes=lat_params['num_scenes'], latent_size=lat_params['latent_size'])
    # Geometry encoder
    geometry_encoder = models.geometry_encoder.map_projected_points
    # Feature Encoder
    feature_encoder = models.nets.TimeLatentFeatureEncoder(
            enc_dim=feat_enc_params['enc_dim'],
            latent_size= lat_params['latent_size'],
            dims = feat_enc_params['dims'],
            out_dim=feat_enc_params['out_dim'],
            dropout=feat_enc_params['dropout'],
            dropout_prob=feat_enc_params['dropout_prob'],
            norm_layers = feat_enc_params['norm_layers'],
            latent_in = feat_enc_params["latent_in"],
            xyz_in_all = feat_enc_params["xyz_in_all"],
            use_tanh = feat_enc_params["use_tanh"],
            latent_dropout = feat_enc_params["latent_dropout"],
            weight_norm= feat_enc_params["weight_norm"]
        )
    # 3D Points
    import pdb; pdb.set_trace()
    if 'loc' in points_params:
        points = models.points.get_3d_pts(filename = "gripper_pts.npy",
                                        n_points = points_params['n_points'], loc = np.array(points_params['loc']),
                                        scale = np.array(points_params['scale']))
    else:
        points = models.points.get_3d_pts(n_points=points_params['n_points'])
    
    # Energy Based Model
    in_dim = points_params['n_points'] * feat_enc_params['out_dim']
    hidden_dim = 512
    energy_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
    )

    model = models.GraspDiffusionFields(vision_encoder=vision_encoder, feature_encoder=feature_encoder, geometry_encoder=geometry_encoder,
                                       decoder=energy_net, points=points).to(device)
    return model


def load_pointcloud_grasp_diffusion(args) -> models.GraspDiffusionFields:
    device = args['device']
    params = args['NetworkSpecs']
    feat_enc_params = params['feature_encoder']
    v_enc_params = params['encoder']
    points_params = params['points']
    # vision encoder
    # Energy Based Model
    num_scene_points = args.get('num_scene_points', -1)
    if v_enc_params.get('type', 'none') == 'vnn2':
        in_features = 3
        vision_encoder = models.vision_encoder.VNN2Pointnet2(out_features=v_enc_params['latent_size'], device=device, in_features=in_features)
    else:
        in_features = 4 if num_scene_points > 0 else 3
        vision_encoder = models.vision_encoder.VNNPointnet2(out_features=v_enc_params['latent_size'], device=device, in_features=in_features)
    # Geometry encoder
    geometry_encoder = models.geometry_encoder.map_projected_points
    # Feature Encoder
    feature_encoder = models.nets.TimeLatentFeatureEncoder(
            enc_dim=feat_enc_params['enc_dim'],
            latent_size= v_enc_params['latent_size'],
            dims = feat_enc_params['dims'],
            out_dim=feat_enc_params['out_dim'],
            dropout=feat_enc_params['dropout'],
            dropout_prob=feat_enc_params['dropout_prob'],
            norm_layers = feat_enc_params['norm_layers'],
            latent_in = feat_enc_params["latent_in"],
            xyz_in_all = feat_enc_params["xyz_in_all"],
            use_tanh = feat_enc_params["use_tanh"],
            latent_dropout = feat_enc_params["latent_dropout"],
            weight_norm= feat_enc_params["weight_norm"]
        )
    # 3D Points
    if 'loc' in points_params:
        pty_file = 'UniformPts.npy' if 'type' not in points_params else points_params['type']
        points = models.points.get_3d_pts(filename = pty_file,
                                        n_points = points_params['n_points'], loc = np.array(points_params['loc']),
                                        scale = np.array(points_params['scale']))
    
    else:
        points = models.points.get_3d_pts(n_points=points_params['n_points'])
    
    # Energy Based Model
    in_dim = points_params['n_points'] * feat_enc_params['out_dim']

    # get nenergy net final distribution
    hidden_dim = 512
    
    distribution = args.get('distribution', 'direct')
    if distribution in ['bernoulli', 'direct']:
        energy_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    elif distribution in ['dirichlet', 'dirichlet_neg']:
        energy_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    dirichlet_scale = args.get('dirichlet_scale', 1.0)
    alpha_activation = args.get('alpha_activation', 'exp')
    learnable_temp = args.get('learnable_temp', False)
    model = models.GraspDiffusionFields(vision_encoder=vision_encoder, feature_encoder=feature_encoder, 
                                        geometry_encoder=geometry_encoder, decoder=energy_net, points=points, 
                                        num_scene_points=num_scene_points, distribution=distribution, 
                                        dirichlet_scale=dirichlet_scale, alpha_activation=alpha_activation, learnable_temp = learnable_temp)
    model.to(device)
    return model