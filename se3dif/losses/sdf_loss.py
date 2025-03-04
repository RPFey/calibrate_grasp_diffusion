import torch
import torch.nn as nn
import se3dif.models as models

class SDFLoss():
    """ 
        Compute the L1 SDF Loss, Only compute at the points where the SDF <= self.delta
    """
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


class CELoss():
    def __init__(self, field='ce', eps=1e-3, T = 30):
        self.field = field
        self.eps = eps
        self.T = T

    def __call__(self, model:models.GraspDiffusionFields, model_input,
                    ground_truth, val=False):
        loss_dict = dict()
        label = model_input["x_ene_pos"].reshape(-1, 4, 4)
        n_label = model_input["x_neg_ene"].reshape(-1, 4, 4) # 
        pos_num = label.shape[0]
        
        labels = torch.cat((label, n_label), dim=0)
        final_t = torch.ones((labels.shape[0], )).to(labels.device) * (1 / self.T) + self.eps
        logits = model.get_logits(labels, final_t).view(-1)
        prob = torch.sigmoid(logits)
        
        l_pos = -1 * torch.log(prob[:pos_num]).clamp(min=-100.).mean()
        l_neg = -1 * torch.log(1 - prob[pos_num:]).clamp(min=-100.).mean()

        ## Total Loss
        loss_dict[self.field + '_pos'] = l_pos
        loss_dict[self.field + '_neg'] = l_neg

        info = {}
        return loss_dict, info

class DirichletLoss():
    def __init__(self, field='dirichlet', eps=1e-3):
        self.field = field
        self.eps = eps

    def __call__(self, model:models.GraspDiffusionFields, model_input,
                    ground_truth, val=False):
        loss_dict = dict()
        label = model_input["x_ene_pos"].reshape(-1, 4, 4)
        n_label = model_input["x_neg_ene"].reshape(-1, 4, 4) # 
        pos_num = label.shape[0]
        
        labels = torch.cat((label, n_label), dim=0)
        final_t = torch.ones((labels.shape[0], )).to(labels.device) * (1 / self.T) + self.eps
        logits = model.get_logits(labels, final_t) # (N, 2)
        alphas = logits + 1
        S = alphas.sum(dim=-1, keepdim=True) # (N, 1)
        ps = alphas / S
        
        targets = torch.zeros_like(logits)
        targets[:pos_num, 0] = 1
        targets[pos_num:, 1] = 1
        
        loss = (targets - ps) ** 2 + ps * (1 - ps) / (S + 1)
        loss = loss.sum(dim=-1).mean()
        
        loss_dict[self.field] = loss

        info = {'dirichlet_loss': loss}
        return loss_dict, info