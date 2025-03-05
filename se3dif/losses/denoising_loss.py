import torch
import torch.nn as nn
import numpy as np

from se3dif.utils import SO3_R3
import theseus as th
from theseus import SO3

class ProjectedSE3DenoisingLoss():
    def __init__(self, field='denoise', delta = 1., grad=False):
        self.field = field
        self.delta = delta
        self.grad = grad

    # TODO check sigma value
    def marginal_prob_std(self, t, sigma=0.5):
        return torch.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    def __call__(self, model, model_input, ground_truth, val=False, eps=1e-5):

        ## Set input ##
        H = model_input['x_ene_pos']
        # context = model_input['visual_context']
        # model.set_latent(context, batch=H.shape[1])
        H = H.reshape(-1, 4, 4)

        ## 1. H to vector ##
        H_th = SO3_R3(R=H[..., :3, :3], t=H[..., :3, -1])
        xw = H_th.log_map()

        ## 2. Sample perturbed datapoint ##
        random_t = torch.rand_like(xw[...,0], device=xw.device) * (1. - eps) + eps
        z = torch.randn_like(xw)
        std = self.marginal_prob_std(random_t)
        perturbed_x = xw + z * std[..., None]
        perturbed_x = perturbed_x.detach()
        perturbed_x.requires_grad_(True)

        ## Get gradient ##
        with torch.set_grad_enabled(True):
            perturbed_H = SO3_R3().exp_map(perturbed_x).to_matrix()
            energy = model(perturbed_H, random_t)
            grad_energy = torch.autograd.grad(energy.sum(), perturbed_x,
                                              only_inputs=True, retain_graph=True, create_graph=True)[0]

        # Compute L1 loss
        z_target = z / std[...,None]
        loss_fn = nn.L1Loss()
        loss = loss_fn(grad_energy, z_target) / 10.
        
        # TEST PART
        with torch.no_grad():
            pos = model_input['x_ene_pos'].detach() # (B, N, 4, 4)
            num_batches = pos.shape[0]
            num_poses_perbatch = pos.shape[1]
            perturbed_H = perturbed_H.detach()
            perturbed_H = perturbed_H.reshape(num_batches, -1, 4, 4) # (B, N, 4, 4)
            poses = torch.cat([pos, perturbed_H], dim=1).view(-1, 4, 4) # (B * 2N, 4, 4)
            final_t = torch.ones_like(poses[:, 0, 0]) * 1e-3
            energy = model(poses, final_t)
            energy = energy.view(num_batches, -1) # (B, 2N)
            
            labels = torch.ones_like(energy)
            labels[:, num_poses_perbatch:] = 0

            # compute ap for each batch
            # real data points have lower energy
            energy_sort = torch.argsort(energy, dim=1, descending=False)
            labels_sort = labels.gather(1, energy_sort)
            precision = torch.cumsum(labels_sort, dim=1) / torch.arange(1, num_poses_perbatch * 2 + 1).to(labels_sort.device)
            recall = torch.cumsum(labels_sort, dim=1) / num_poses_perbatch
            delta_recall = recall[:, 1:] - recall[:, :-1]
            ap = (precision[:, 1:] * delta_recall).sum(dim=1) + precision[:, 0] * recall[:, 0]
            ap = ap.mean()
            
        info = {self.field: grad_energy, 'noise_ap': ap}
        loss_dict = {"Score loss": loss}
        return loss_dict, info
    
class DirichletSE3DenoisingLoss():
    def __init__(self, field='denoise', delta = 1., grad=False):
        self.field = field
        self.delta = delta
        self.grad = grad

    # TODO check sigma value
    def marginal_prob_std(self, t, sigma=0.5):
        return torch.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    def __call__(self, model, model_input, ground_truth, val=False, eps=1e-5):

        ## Set input ##
        H = model_input['x_ene_pos']
        # context = model_input['visual_context']
        # model.set_latent(context, batch=H.shape[1])
        H = H.reshape(-1, 4, 4)

        ## 1. H to vector ##
        H_th = SO3_R3(R=H[...,:3, :3], t=H[...,:3, -1])
        xw = H_th.log_map()

        ## 2. Sample perturbed datapoint ##
        random_t = torch.rand_like(xw[...,0], device=xw.device) * (1. - eps) + eps
        z = torch.randn_like(xw)
        std = self.marginal_prob_std(random_t)
        perturbed_x = xw + z * std[..., None]
        perturbed_x = perturbed_x.detach()
        perturbed_x.requires_grad_(True)

        ## Get gradient ##
        with torch.set_grad_enabled(True):
            perturbed_H = SO3_R3().exp_map(perturbed_x).to_matrix()
            energy = model(perturbed_H, random_t)
            grad_energy = torch.autograd.grad(energy.sum(), perturbed_x,
                                              only_inputs=True, retain_graph=True, create_graph=True)[0]

        # Compute L1 loss
        z_target = z / std[...,None]
        # loss_fn = nn.L1Loss()
        # loss = loss_fn(grad_energy, z_target) / 10.
        cos_similarity = nn.functional.cosine_similarity(grad_energy, z_target, dim=-1)
        loss = 1 - cos_similarity.mean()

        info = {self.field: grad_energy}
        loss_dict = {"Score loss": loss}
        return loss_dict, info
    
class ProjectedSE3DenoisingCOSLoss():
    def __init__(self, field='denoise', delta = 1., grad=False):
        self.field = field
        self.delta = delta
        self.grad = grad

    # TODO check sigma value
    def marginal_prob_std(self, t, sigma=0.5):
        return torch.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    def __call__(self, model, model_input, ground_truth, val=False, eps=1e-5):

        ## Set input ##
        H = model_input['x_ene_pos']
        # context = model_input['visual_context']
        # model.set_latent(context, batch=H.shape[1])
        H = H.reshape(-1, 4, 4)

        ## 1. H to vector ##
        H_th = SO3_R3(R=H[...,:3, :3], t=H[...,:3, -1])
        xw = H_th.log_map()

        ## 2. Sample perturbed datapoint ##
        random_t = torch.rand_like(xw[...,0], device=xw.device) * (1. - eps) + eps
        z = torch.randn_like(xw)
        std = self.marginal_prob_std(random_t)
        perturbed_x = xw + z * std[..., None]
        perturbed_x = perturbed_x.detach()
        perturbed_x.requires_grad_(True)

        ## Get gradient ##
        with torch.set_grad_enabled(True):
            perturbed_H = SO3_R3().exp_map(perturbed_x).to_matrix()
            energy = model(perturbed_H, random_t)
            grad_energy = torch.autograd.grad(energy.sum(), perturbed_x,
                                              only_inputs=True, retain_graph=True, create_graph=True)[0]

        # Compute L1 loss
        z_target = z / std[...,None]
        cos_similarity = nn.functional.cosine_similarity(grad_energy, z_target, dim=-1)
        loss = 1 - cos_similarity.mean()

        info = {self.field: grad_energy}
        loss_dict = {"Score loss": loss}
        return loss_dict, info

class SE3DenoisingLoss():

    def __init__(self, field='denoise', delta = 1., grad=False):
        self.field = field
        self.delta = delta
        self.grad = grad

    # TODO check sigma value
    def marginal_prob_std(self, t, sigma=0.5):
        return torch.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    def log_gaussian_on_lie_groups(self, x, context):
        R_p = SO3.exp_map(x[...,3:])
        delta_H = th.compose(th.inverse(context[0]), R_p)
        log = delta_H.log_map()

        dt = x[...,:3] - context[1]

        tlog = torch.cat((dt, log), -1)
        return -0.5 * tlog.pow(2).sum(-1)/(context[2]**2)

    def __call__(self, model, model_input, ground_truth, val=False, eps=1e-5):

        ## From Homogeneous transformation to axis-angle ##
        H = model_input['x_ene_pos']
        # n_grasps = H.shape[1]
        # c = model_input['visual_context']
        # model.set_latent(c, batch=n_grasps)

        H_in = H.reshape(-1, 4, 4)
        H_in = SO3_R3(R=H_in[:, :3, :3], t=H_in[:, :3, -1])
        tw = H_in.log_map()
        #######################

        ## 1. Compute noisy sample SO(3) + R^3##
        random_t = torch.rand_like(tw[...,0], device=tw.device) * (1. - eps) + eps
        z = torch.randn_like(tw)
        std = self.marginal_prob_std(random_t)
        noise = z * std[..., None]
        noise_t = noise[..., :3]
        noise_rot = SO3.exp_map(noise[...,3:])
        R_p = th.compose(H_in.R, noise_rot)
        t_p = H_in.t + noise_t
        #############################

        ## 2. Compute target score ##
        w_p = R_p.log_map()
        tw_p = torch.cat((t_p, w_p), -1).requires_grad_()
        log_p = self.log_gaussian_on_lie_groups(tw_p, context=[H_in.R, H_in.t, std])
        target_grad = torch.autograd.grad(log_p.sum(), tw_p, only_inputs=True)[0]
        target_score = target_grad.detach()
        #############################

        ## 3. Get diffusion grad ##
        x_in = tw_p.detach().requires_grad_(True)
        H_in = SO3_R3().exp_map(x_in).to_matrix()
        t_in = random_t
        energy = model(H_in, t_in)
        grad_energy = torch.autograd.grad(energy.sum(), x_in, only_inputs=True,
                                          retain_graph=True, create_graph=True)[0]

        ## 4. Compute loss ##
        loss_fn = nn.L1Loss()
        loss = loss_fn(grad_energy, -target_score) / 20.

        info = {self.field: energy}
        loss_dict = {"Score loss": loss}
        return loss_dict, info
    
    # def compute_loss(self, )

