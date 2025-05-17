import torch
import torch.nn as nn
import numpy as np

from se3dif.utils import SO3_R3
import theseus as th
from theseus import SO3

class ProjectedSE3DenoisingLoss():
    def __init__(self, field='denoise', delta = 1., 
                    grad=False, sigma=0.5, band=1e-5, 
                    lambda_sigma='constant'):
        self.field = field
        self.delta = delta
        self.grad = grad
        self.sigma = sigma
        self.band = band
        self.lambda_sigma = lambda_sigma

    # TODO check sigma value
    def marginal_prob_std(self, t):
        return torch.sqrt((self.sigma ** (2 * t) - 1.) / (2. * np.log(self.sigma)))

    def __call__(self, model, model_input, ground_truth, val=False):

        ## Set input ##
        H = model_input['x_ene_pos']
        # context = model_input['visual_context']
        # model.set_latent(context, batch=H.shape[1])
        H = H.reshape(-1, 4, 4)

        ## 1. H to vector ##
        H_th = SO3_R3(R=H[..., :3, :3], t=H[..., :3, -1])
        xw = H_th.log_map()

        ## 2. Sample perturbed datapoint ##
        random_t = torch.rand_like(xw[...,0], device=xw.device) * (1. - self.band) + self.band
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
        z_target = z / std[..., None]
        if self.lambda_sigma == 'constant':
            loss_fn = nn.L1Loss()
            loss = loss_fn(grad_energy, z_target) / 10.
        elif self.lambda_sigma == 'adaptive':
            # In Song Yang's paper, https://arxiv.org/pdf/1907.05600
            # They try to keep the magnitude of the loss for different noise similar.
            l1_distance = torch.sum(torch.abs(grad_energy - z_target), dim = -1)
            loss = torch.mean(l1_distance * std) / 10.
        
        info = {self.field: grad_energy}
        
        # TODO TEST PART, Uncomment if needed
        # with torch.no_grad():
        #     pos = model_input['x_ene_pos'].detach() # (B, N, 4, 4)
        #     num_batches = pos.shape[0]
        #     num_poses_perbatch = pos.shape[1]
        #     perturbed_H = perturbed_H.detach()
        #     perturbed_H = perturbed_H.reshape(num_batches, -1, 4, 4) # (B, N, 4, 4)
        #     poses = torch.cat([pos, perturbed_H], dim=1).view(-1, 4, 4) # (B * 2N, 4, 4)
        #     final_t = torch.ones_like(poses[:, 0, 0]) * 1e-3
        #     energy = model(poses, final_t)
        #     energy = energy.view(num_batches, -1) # (B, 2N)
            
        #     labels = torch.ones_like(energy)
        #     labels[:, num_poses_perbatch:] = 0

        #     # compute ap for each batch
        #     # real data points have lower energy
        #     energy_sort = torch.argsort(energy, dim=1, descending=False)
        #     labels_sort = labels.gather(1, energy_sort)
        #     precision = torch.cumsum(labels_sort, dim=1) / torch.arange(1, num_poses_perbatch * 2 + 1).to(labels_sort.device)
        #     recall = torch.cumsum(labels_sort, dim=1) / num_poses_perbatch
        #     delta_recall = recall[:, 1:] - recall[:, :-1]
        #     ap = (precision[:, 1:] * delta_recall).sum(dim=1) + precision[:, 0] * recall[:, 0]
        #     ap = ap.mean()      
        #     info["noise_ap"] = ap
        
        loss_dict = {"Score loss": loss}
        return loss_dict, info
    
class ProjectedNegSE3DenoisingLoss():
    """
        Compute the SM Loss on negative samples
            We imagine those negative samples are sampled and supervise gradient on those negative samples
    
    """
    def __init__(self, field='neg_denoise', delta = 1., 
                 grad=False, sigma=0.5):
        self.field = field
        self.delta = delta
        self.grad = grad
        self.sigma = sigma

    # TODO check sigma value
    def marginal_prob_std(self, t):
        return torch.sqrt((self.sigma ** (2 * t) - 1.) / (2. * np.log(self.sigma)))

    def __call__(self, model, model_input, ground_truth, val=False, eps=1e-5):

        ## Set input ##
        H = model_input['x_ene_pos']
        batch_size = H.shape[0]
        H = H.reshape(-1, 4, 4)

        ## 1. H to vector ##
        H_th = SO3_R3(R=H[..., :3, :3], t=H[..., :3, -1])
        xw = H_th.log_map()

        ## 2. Sample perturbed datapoint ##
        # z = torch.randn_like(xw)
        # perturbed_x = xw + z * std[..., None] # (Nk, 6)
        neg_H = model_input['x_neg_ene'].reshape(-1, 4, 4)
        neg_H_th = SO3_R3(R=neg_H[..., :3, :3], t=neg_H[..., :3, -1])
        perturbed_x = neg_H_th.log_map()
        random_t = torch.rand_like(perturbed_x[...,0], device=xw.device) * (1. - eps) + eps
        std = self.marginal_prob_std(random_t) # (BM, )
        
        valid_index = model_input['valid_ene_pos'].reshape(batch_size, 1, -1) # (B, N)
        distance = torch.sum(
            (perturbed_x.reshape(batch_size, -1, 1, 6) - xw.reshape(batch_size, 1, -1, 6)) ** 2,
            dim = -1
        ) # (B, M, N)
        
        # NN assignment, compute the probability of occurrence
        # this will be the weight for loss
        distance_min = torch.min(distance, dim=-1)[0] # (B, M)
        loss_weight = torch.exp(- distance_min / (2 * std.reshape(batch_size, -1) ** 2)) / \
                    std.reshape(batch_size, -1)  # (B, M)
        loss_weight.clamp_(max=1.)
        
        # compute z target 
        # TODO you can also use NN assignment to compute target
        weight = valid_index * torch.exp( - distance / (2 * std.reshape(batch_size, -1, 1) ** 2)) # (B, M, N)
        weight = torch.nn.functional.normalize(weight, p=1, dim=-1) # (B, M, N)
        average_target = torch.einsum('bmn,bnk->bmk', weight, xw.reshape(batch_size, -1, 6)) # (B, M, 6)
        z_target = (perturbed_x - average_target.view(-1, 6)) / std[..., None] ** 2
        
        perturbed_x = perturbed_x.detach()
        perturbed_x.requires_grad_(True)
        
        ## Get gradient ##
        with torch.set_grad_enabled(True):
            perturbed_H = SO3_R3().exp_map(perturbed_x).to_matrix()
            energy = model(perturbed_H, random_t)
            grad_energy = torch.autograd.grad(energy.sum(), perturbed_x,
                                              only_inputs=True, retain_graph=True, create_graph=True)[0]

        # Compute L1 loss
        loss_fn = nn.L1Loss()
        loss = torch.mean(
            loss_weight.reshape(-1) * \
                torch.sum( torch.abs(grad_energy - z_target), dim = -1)
        ) / 10.
        
        info = {self.field: grad_energy}
        loss_dict = {"Neg Score loss": loss}
        return loss_dict, info
    
class ProjectedNegDirichletSE3DenoisingLoss():
    """
        Dual Dirichlet SM Loss
            Here, we compute both the positive & negative SM Loss
    
    """
    def __init__(self, field='neg_dirichlet_denoise', 
                 delta = 1., grad=False, sigma=0.5, 
                 weight=0.1, band=1e-5, lambda_sigma='constant'):
        self.field = field
        self.delta = delta
        self.grad = grad
        self.sigma = sigma
        self.weight = weight
        self.band = band
        self.lambda_sigma = lambda_sigma

    # TODO check sigma value
    def marginal_prob_std(self, t):
        return torch.sqrt((self.sigma ** (2 * t) - 1.) / (2. * np.log(self.sigma)))
    
    def compute_sm_loss(self, model, model_input, name='pos'):
        ## Set input ##
        H = model_input['x_ene_pos'] if name == 'pos' else model_input['x_neg_ene']
        # context = model_input['visual_context']
        # model.set_latent(context, batch=H.shape[1])
        H = H.reshape(-1, 4, 4)

        ## 1. H to vector ##
        H_th = SO3_R3(R=H[..., :3, :3], t=H[..., :3, -1])
        xw = H_th.log_map()

        ## 2. Sample perturbed datapoint ##
        # eps = 1e-5
        random_t = torch.rand_like(xw[...,0], device=xw.device) * (1. - self.band) + self.band
        z = torch.randn_like(xw)
        std = self.marginal_prob_std(random_t)
        perturbed_x = xw + z * std[..., None]
        perturbed_x = perturbed_x.detach()
        perturbed_x.requires_grad_(True)

        ## Get gradient ##
        with torch.set_grad_enabled(True):
            perturbed_H = SO3_R3().exp_map(perturbed_x).to_matrix()
            
            # Take the false energy
            logits = model.get_logits(perturbed_H, random_t)
            alphas = model.alpha_fn(logits)
            S = (alphas + 1).sum(dim=-1)
            energy = -1 * torch.log(alphas[:, 0] / S + 1e-6) * torch.exp(model.temperature) if name == 'pos' else \
                            -1 * torch.log(alphas[:, 1] / S + 1e-6) * torch.exp(model.temperature)

            grad_energy = torch.autograd.grad(energy.sum(), perturbed_x,
                                              only_inputs=True, retain_graph=True, create_graph=True)[0]

        # Compute L1 loss
        z_target = z / std[..., None]
        if self.lambda_sigma == 'constant':
            loss_fn = nn.L1Loss()
            loss = loss_fn(grad_energy, z_target) * self.weight # / 10.
        elif self.lambda_sigma == 'adaptive':
            # In Song Yang's paper, https://arxiv.org/pdf/1907.05600
            # They try to keep the magnitude of the loss for different noise similar.
            l1_distance = torch.sum(torch.abs(grad_energy - z_target), dim = -1)
            loss = torch.mean(l1_distance * std) * self.weight # / 10.
        
        return loss

    def __call__(self, model, model_input, ground_truth, val=False):
        assert model.distribution == 'dirichlet_neg', "Model should be dirichlet distribution"
        
        loss =( self.compute_sm_loss(model, model_input, name='pos') + \
                self.compute_sm_loss(model, model_input, name='neg') ) / 2
        loss_dict = {"Dual Score loss": loss}
        info = {}
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
        # compute log of delta R
        R_p = SO3.exp_map(x[...,3:])
        delta_H = th.compose(th.inverse(context[0]), R_p)
        log = delta_H.log_map()
        # delta t - translation
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
        with torch.enable_grad():
            tw_p = torch.cat((t_p, w_p), -1).requires_grad_()
            log_p = self.log_gaussian_on_lie_groups(tw_p, context=[H_in.R, H_in.t, std])
            target_grad = torch.autograd.grad(log_p.sum(), tw_p, only_inputs=True)[0]
        target_score = target_grad.detach()
        #############################

        ## 3. Get diffusion grad ##
        with torch.enable_grad():
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

