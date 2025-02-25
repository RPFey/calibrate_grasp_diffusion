import torch
import torch.nn as nn
import se3dif.models as models

class SDFLoss():
    def __init__(self, field='sdf', delta = 0.6, grad=True):
        self.field = field
        self.delta = delta
        self.grad = grad

    def __call__(self, model:models.GraspDiffusionFields, 
                        model_input, ground_truth, val=False):
        loss_dict = dict()
        label = ground_truth[self.field].squeeze().reshape(-1)

        ## Set input ##
        x_sdf = model_input['x_sdf'].detach().requires_grad_()
        
        ## Compute model output ##
        sdf = model.compute_sdf(x_sdf.view(-1, 3))

        ## Reconstruction Loss ##
        loss = nn.L1Loss(reduction='mean')
        pred_clip_sdf = torch.clip(sdf, -10., self.delta)
        target_clip_sdf = torch.clip(label, -10., self.delta)
        l_rec = loss(pred_clip_sdf, target_clip_sdf)

        ## Total Loss
        loss_dict[self.field] = l_rec

        info = {'sdf': sdf}
        return loss_dict, info

class NLLLoss():
    def __init__(self, field='nll', eps = 1e-3, grad=True):
        self.field = field
        self.eps = eps

    def __call__(self, model:models.GraspDiffusionFields, 
                        model_input, ground_truth, val=False):
        loss_dict = dict()
        with torch.no_grad():
            label = model_input["x_ene_pos"].reshape(-1, 4, 4)
            n_label = label.clone()
            n_label[:, :3, 3] += torch.rand((label.shape[0], 3), device=label.device)
            
        ## Compute model output ##
        random_t = torch.rand((label.shape[0], ), device=label.device) * (1. - self.eps) + self.eps
        pos_concentration = model.get_logits(label, random_t)
        pos_dist = torch.distributions.Dirichlet(pos_concentration)
        positive_label = torch.ones_like(pos_concentration)
        positive_label[:, 0].fill_(self.eps)
        positive_label[:, 1].fill_(1 - self.eps)
        l_pos = -pos_dist.log_prob(positive_label).mean()
        
        neg_concentration = model.get_logits(n_label, random_t)
        neg_dist = torch.distributions.Dirichlet(neg_concentration)
        negative_label = torch.ones_like(neg_concentration)
        negative_label[:, 0].fill_(1 - self.eps)
        negative_label[:, 1].fill_(self.eps)
        l_neg = -neg_dist.log_prob(negative_label).mean()

        ## Total Loss
        loss_dict[self.field] = l_pos + l_neg

        info = {'l_pos': l_pos, 'l_neg': l_neg}
        return loss_dict, info
