from .denoising_loss import *
from .sdf_loss import SDFLoss, CELoss, DirichletLoss, APLoss

def get_losses(args):
    losses = args['Losses']

    loss_fns = {}
    if 'sdf_loss' in losses:
        loss_fns['sdf'] = SDFLoss()
    if 'projected_denoising_loss' in losses:
        loss_fns['denoise'] = ProjectedSE3DenoisingLoss()
    if 'projected_fix_denoising_l1loss' in losses:
        loss_fns['denoise'] = ProjectedFixedSE3DenoisingLoss()
    if 'projected_denoising_cosloss' in losses:
        loss_fns['denoise'] = ProjectedSE3DenoisingCOSLoss()
    if 'denoising_loss' in losses:
        loss_fns['denoise'] = SE3DenoisingLoss()
    if 'dirichlet_denoising_loss' in losses:
        loss_fns['denoise'] = DirichletSE3DenoisingLoss()
    if 'celoss' in losses:
        loss_fns['ce'] = CELoss()
    if 'dirichlet' in losses:
        loss_fns['dirichlet'] = DirichletLoss()
    if 'aploss' in losses:
        loss_fns['aploss'] = APLoss()

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
            losses = {**losses, **loss}
            infos = {**infos, **info}

        return losses, infos