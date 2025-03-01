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


class CELoss():
    def __init__(self, field='ce', eps=1e-6):
        self.field = field
        self.eps = eps

    def __call__(self, model:models.GraspDiffusionFields, model_input):
        loss_dict = dict()
        label = model_input["x_ene_pos"].reshape(-1, 4, 4)
        n_label = model_input["x_neg_ene"].reshape(-1, 4, 4) # 
            
        ## Compute model output ##
        random_t = torch.rand((label.shape[0], ), device=label.device) * (1. - self.eps) + self.eps
        pos_logit = model.get_logits(label, random_t)
        pos_prob = torch.sigmoid(pos_logit)
        l_pos = torch.log(pos_prob + self.eps).mean()
        
        neg_logit = model.get_logits(n_label, random_t)
        neg_prob = torch.sigmoid(neg_logit)
        l_neg = torch.log(1 - neg_prob).mean()

        ## Total Loss
        loss_dict[self.field] = l_pos + l_neg

        info = {'l_pos': l_pos, 'l_neg': l_neg}
        return loss_dict, info
