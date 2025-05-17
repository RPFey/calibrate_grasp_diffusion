from .denoising_loss import *
from .sdf_loss import *

def get_losses(args):
    losses = args['Losses']

    loss_fns = {}
    if 'sdf_loss' in losses:
        loss_fns['sdf'] = SDFLoss(
            **losses['sdf_loss'])
    if 'projected_denoising_loss' in losses:
        loss_fns['denoise'] = ProjectedSE3DenoisingLoss(
            **losses['projected_denoising_loss'])
    if 'projected_neg_denoising_l1loss' in losses:
        loss_fns['neg_denoise'] = ProjectedNegSE3DenoisingLoss(
            **losses['projected_neg_denoising_l1loss'])
    if 'projected_neg_dirichlet_denoising_l1loss' in losses:
        loss_fns['neg_denoise'] = ProjectedNegDirichletSE3DenoisingLoss(
            **losses['projected_neg_dirichlet_denoising_l1loss'])
    if 'denoising_loss' in losses:
        loss_fns['denoise'] = SE3DenoisingLoss()
    if 'ranking_loss' in losses:
        loss_fns['ranking_loss'] = PairwiseRankingLoss(
            **losses['ranking_loss'] )
    if 'celoss' in losses:
        loss_fns['ce'] = CELoss()
    if 'dirichlet' in losses:
        loss_fns['dirichlet'] = DirichletLoss()
    if 'aploss' in losses:
        loss_fns['aploss'] = APLoss(
            **losses['aploss']
        )
    if 'dirichlet_aploss' in losses:
        loss_fns['dirichlet_aploss'] = DirichletAPLoss(**losses['dirichlet_aploss'])
    if 'ebm_loss' in losses:
        loss_fns['ebm'] = EBMLoss()

    loss_dict = LossDictionary(loss_dict=loss_fns)
    return loss_dict


class LossDictionary():

    def __init__(self, loss_dict):
        self.fields = loss_dict.keys()
        self.loss_dict = loss_dict

    def loss_fn(self, model, model_input, ground_truth, val=False):
        losses = {}
        infos = {}
        
        # set the visual context for the model
        c = model_input['visual_context']
        target_index = model_input.get('target_index', None)    
        model.set_latent(c, target_index=target_index)
        
        for field in self.fields:
            loss_fn_k = self.loss_dict[field]
            loss, info = loss_fn_k(model, model_input, ground_truth, val)
            if loss is None:
                print(f"{field} Loss is None, Skip")
                continue
            losses = {**losses, **loss}
            infos = {**infos, **info}

        return losses, infos