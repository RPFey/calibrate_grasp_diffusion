import torch
import torch.nn as nn
import se3dif.models as models
import numpy as np

from torchmetrics.classification import BinaryAveragePrecision

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
        # single sdf is place holder
        if ground_truth[self.field].shape[1] <= 1:
            print("SDF Loss: No SDF values to compute loss")
            return None, None
        
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
    def __init__(self, field='ce', eps=1e-6, T = 30):
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
        final_t = torch.rand((labels.shape[0], )).to(labels.device) * (1 - self.eps) + self.eps
        logits = model.get_logits(labels, final_t).view(-1)
        prob = torch.sigmoid(logits)
        
        l_pos = -1 * torch.log(prob[:pos_num]).clamp(min=-100.).mean()
        l_neg = -1 * torch.log(1 - prob[pos_num:]).clamp(min=-100.).mean()

        ## Total Loss
        loss_dict[self.field + '_pos'] = l_pos
        loss_dict[self.field + '_neg'] = l_neg

        info = {} 
        
        # Uncomment to compute AP
        # with torch.no_grad():
        #     bap = BinaryAveragePrecision(thresholds=None)
        #     pred = prob.detach()
        #     label = torch.ones_like(prob).long()
        #     label[pos_num:] = 0
        #     ap = bap(pred, label).item()
        #     info['ap'] = ap
            
        return loss_dict, info

class DirichletLoss():
    def __init__(self, field='dirichlet', eps=1e-3):
        self.field = field
        self.eps = eps
        self.T = 30

    def __call__(self, model:models.GraspDiffusionFields, model_input,
                    ground_truth, val=False):
        loss_dict = dict()
        label = model_input["x_ene_pos"].reshape(-1, 4, 4)
        n_label = model_input["x_neg_ene"].reshape(-1, 4, 4) # 
        pos_num = label.shape[0]
        
        labels = torch.cat((label, n_label), dim=0)
        random_t = torch.rand((labels.shape[0], )).to(labels.device) * (1 - self.eps) + self.eps
        logits = model.get_logits(labels, random_t) # (N, 2)
        alphas = model.alpha_fn(logits) + 1
        S = alphas.sum(dim=-1, keepdim=True) # (N, 1)
        ps = alphas / S
        
        targets = torch.zeros_like(logits)
        targets[:pos_num, 0] = 1
        targets[pos_num:, 1] = 1
        
        loss = (targets - ps) ** 2 + ps * (1 - ps) / (S + 1)
        loss = loss.sum(dim=-1).mean()
        
        loss_dict[self.field] = loss
        
        info = {}
        
        # Uncomment to compute AP
        # with torch.no_grad():
        #     bap = BinaryAveragePrecision(thresholds=None)
        #     pred = ps[:, 0].detach()
        #     label = targets[:, 0].detach().long()
        #     ap = bap(pred, label).item()
        #     info['ap'] = ap
          
        return loss_dict, info
    
class APLossImpl (nn.Module):
    """ Differentiable AP loss, through quantization. From the paper:

        Learning with Average Precision: Training Image Retrieval with a Listwise Loss
        Jerome Revaud, Jon Almazan, Rafael Sampaio de Rezende, Cesar de Souza
        https://arxiv.org/abs/1906.07589

        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}

        Returns: 1 - mAP (mean AP for each n in {1..N})
                 Note: typically, this is what you wanna minimize
    """
    def __init__(self, nq=25, min=0, max=1):
        nn.Module.__init__(self)
        assert isinstance(nq, int) and 2 <= nq <= 100
        self.nq = nq
        self.min = min
        self.max = max
        gap = max - min
        assert gap > 0
        # Initialize quantizer as non-trainable convolution
        self.quantizer = q = nn.Conv1d(1, 2*nq, kernel_size=1, bias=True)
        q.weight = nn.Parameter(q.weight.detach(), requires_grad=False)
        q.bias = nn.Parameter(q.bias.detach(), requires_grad=False)
        a = (nq-1) / gap
        # First half equal to lines passing to (min+x,1) and (min+x+1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[:nq] = -a
        q.bias[:nq] = torch.from_numpy(a*min + np.arange(nq, 0, -1))  # b = 1 + a*(min+x)
        # First half equal to lines passing to (min+x,1) and (min+x-1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[nq:] = a
        q.bias[nq:] = torch.from_numpy(np.arange(2-nq, 2, 1) - a*min)  # b = 1 - a*(min+x)
        # First and last one as a horizontal straight line
        q.weight[0] = q.weight[-1] = 0
        q.bias[0] = q.bias[-1] = 1

    def forward(self, x, label, qw=None, ret='1-mAP'):
        assert x.shape == label.shape  # N x M
        N, M = x.shape
        # Quantize all predictions
        q = self.quantizer(x.unsqueeze(1))
        q = torch.min(q[:, :self.nq], q[:, self.nq:]).clamp(min=0)  # N x Q x M

        nbs = q.sum(dim=-1)  # number of samples  N x Q = c
        rec = (q * label.view(N, 1, M).float()).sum(dim=-1)  # number of correct samples = c+ N x Q
        prec = rec.cumsum(dim=-1) / (1e-16 + nbs.cumsum(dim=-1))  # precision
        rec /= rec.sum(dim=-1).unsqueeze(1)  # norm in [0,1]

        ap = (prec * rec).sum(dim=-1)  # per-image AP

        if ret == '1-mAP':
            if qw is not None:
                ap *= qw  # query weights
            return 1 - ap.mean()
        elif ret == 'AP':
            assert qw is None
            return ap
        else:
            raise ValueError("Bad return type for APLoss(): %s" % str(ret))

    def measures(self, x, gt, loss=None):
        if loss is None:
            loss = self.forward(x, gt)
        return {'loss_ap': float(loss)}
    
    
class APLoss():
    def __init__(self, field='ap', time_mode="random",
                    eps=1e-6, weight=1.0):
        self.field = field
        self.eps = eps
        self.ap_impl = APLossImpl()
        self.ap_impl.to(torch.device("cuda:0"))
        self.time_mode = time_mode
        self.weight = weight

    def __call__(self, model:models.GraspDiffusionFields, model_input,
                    ground_truth, val=False):
        loss_dict = dict()
        grasps = torch.cat([model_input["x_ene_pos"], model_input["x_neg_ene"]], dim = 1) # (B, N, 4, 4)
        pos_num = model_input["x_ene_pos"].shape[1]
        batch_size = grasps.shape[0]
        grasps = grasps.view(-1, 4, 4)

        if self.time_mode == 'random':
            final_t = torch.rand((grasps.shape[0], )).to(grasps.device) * (1. - self.eps) + self.eps
        elif self.time_mode == 'half':
            final_t = torch.rand((grasps.shape[0], )).to(grasps.device) * (1. - self.eps) * 0.5 + self.eps
        else:
            final_t = torch.ones((grasps.shape[0], )).to(grasps.device) * model.final_t
        
        logits = model.get_logits(grasps, final_t).view(-1)
        prob = torch.sigmoid(logits)
        prob = prob.reshape(batch_size, -1) # (B, N)
        
        labels = torch.ones_like(prob)
        labels[:, pos_num:] = 0
        ap_loss = self.ap_impl(prob, labels)

        loss_dict[self.field] = self.weight * ap_loss
        info = {'ap': 1 - ap_loss.item()}
        return loss_dict, info
    
class DirichletAPLoss():
    def __init__(self, field='dirichlet_ap', 
                        mode="direct", eps=1e-6, 
                        mll='normalize', reg=-1, band=1):
        self.field = field
        self.eps = eps
        self.ap_impl = APLossImpl()
        self.ap_impl.to(torch.device("cuda:0"))
        self.mode = mode
        self.mll = mll
        self.band = band
        assert self.eps < self.band <= 1, "Band must be in (eps, 1]"
        self.reg = reg

    def __call__(self, model:models.GraspDiffusionFields, model_input,
                    ground_truth, val=False):
        loss_dict = dict()
        if "generated_grasps" not in model_input:
            grasps = torch.cat([model_input["x_ene_pos"], model_input["x_neg_ene"]], dim = 1) # (B, N, 4, 4)
        else:
            grasps = torch.cat([model_input["x_ene_pos"], model_input["generated_grasps"], model_input["x_neg_ene"]], dim = 1) # (B, N, 4, 4)
        
        pos_num = model_input["x_ene_pos"].shape[1]
        neg_num = model_input["x_neg_ene"].shape[1]
        batch_size = grasps.shape[0]
        grasps = grasps.view(-1, 4, 4)

        random_t = torch.rand((grasps.shape[0], )).to(grasps.device) * (self.band - self.eps)  + self.eps
        logits = model.get_logits(grasps, random_t) # (BN, 2)
        if self.mll == 'normalize':
            # normalize prob for S and F
            alphas = model.alpha_fn(logits) + 1
            S = alphas.sum(dim=-1)
        else:
            # only use the positive
            alphas = model.alpha_fn(logits)
            S = (alphas + 1).sum(dim=-1)
        
        probPos = (alphas[:, 0] / S).reshape(batch_size, -1) # (B, N)
        probNeg = (alphas[:, 1] / S).reshape(batch_size, -1) # (B, N)
            
        if self.mode == "direct":
            targets = torch.zeros_like(probPos)
            targets[:, :pos_num] = 1
            ap_loss = self.ap_impl(probPos, targets)
            loss_dict["pos_ap"] = ap_loss
            ap = 1 - ap_loss.item()
        elif self.mode == "inverse":
            targets = torch.zeros_like(probPos)
            targets[:, -neg_num:] = 1
            ap_loss = self.ap_impl(probNeg, targets)
            loss_dict["neg_ap"] = ap_loss
            ap = 1 - ap_loss.item()
        else:
            targets_pos = torch.zeros_like(probPos)
            targets_pos[:, :pos_num] = 1
            ap_loss_pos = self.ap_impl(probPos, targets_pos)
            
            targets_neg = torch.zeros_like(probPos)
            targets_neg[:, -neg_num:] = 1
            ap_loss_neg = self.ap_impl(probNeg, targets_neg)
            loss_dict["pos_ap"] = ap_loss_pos / 2
            loss_dict["neg_ap"] = ap_loss_neg / 2
            ap = 1 - ap_loss_pos.item()

        if self.reg > 0:
            loss_dict["reg"] = self.reg * torch.mean( 
                torch.abs( torch.sum(alphas, dim=-1) )
            )
            
        info = {'ap': ap} 
        return loss_dict, info
    
class EBMLoss():
    def __init__(self, field='ebm', regularization=0.01):
        self.regularization = regularization
        self.field = field
        
    def __call__(self, model:models.GraspDiffusionFields, 
                    model_input, ground_truth, val=False):
        loss_dict = dict()
        label = model_input["x_ene_pos"].reshape(-1, 4, 4)
        if 'generated_grasps' in model_input:
            n_label = model_input["generated_grasps"].reshape(-1, 4, 4)
        else:
            return None, {}
        
        pos_energy = model(label)
        neg_energy = model(n_label)

        info = {} 
        loss = torch.mean(pos_energy) - torch.mean(neg_energy) + \
                self.regularization * torch.mean(neg_energy ** 2)
        loss_dict[self.field] = loss
        info['pos_energy'] = pos_energy.mean().item()
        info['neg_energy'] = neg_energy.mean().item()   
        
        return loss_dict, info