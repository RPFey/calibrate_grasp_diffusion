import os
import glob

# set pyglet to headless mode
if os.environ.get('PYOPENGL_PLATFORM', '') == 'egl':
    import pyglet
    pyglet.options['headless'] = True

import copy
import configargparse
from se3dif.utils import get_root_src

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from se3dif import datasets, losses, summaries, trainer
from se3dif.models import loader
from se3dif.utils import load_experiment_specifications
from se3dif.trainer.learning_rate_scheduler import get_learning_rate_schedules

base_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.abspath(os.path.dirname(__file__ + '/../../../../../'))


def parse_args():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

    p.add_argument('--specs_file_dir', type=str, default=os.path.join(base_dir, 'params')
                   , help='root for saving logging')

    p.add_argument('--spec_file', type=str, default='multiobject_scene_graspdif'
                   , help='root for saving logging')

    p.add_argument('--num_workers', type=int, default=16
                   , help='root for saving logging')

    p.add_argument('--summary', type=bool, default=True
                   , help='activate or deactivate summary')

    p.add_argument('--saving_root', type=str, default=os.path.join(get_root_src(), 'logs')
                   , help='root for saving logging')

    p.add_argument('--models_root', type=str, default=root_dir
                   , help='root for saving logging')
    
    p.add_argument("--data_root", type=str, default=os.path.join(root_dir, 'data', 'scene_sdf')
                   , help='root for saving logging')
    
    p.add_argument('--eval', action='store_true', default=False, help='evaluate model')
    
    p.add_argument('--eval_ckpt', type=str, default=None, help='checkpoint to evaluate, \
                    if None, evaluate all the checkpoints in the model directory')

    p.add_argument('--device',  type=str, default='cuda')
    
    p.add_argument('--class_type', type=str, default='Mug')

    opt = p.parse_args()
    return opt


def main(opt):

    ## Load training args ##
    spec_file = os.path.join(opt.specs_file_dir, opt.spec_file)
    args = load_experiment_specifications(spec_file)

    # saving directories
    root_dir = opt.saving_root
    exp_dir  = os.path.join(root_dir, args['exp_log_dir'])
    args['saving_folder'] = exp_dir

    if opt.device =='cuda':
        if 'cuda_device' in args:
            cuda_device = args['cuda_device']
        else:
            cuda_device = 0
        device = torch.device('cuda:' + str(cuda_device) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    ## Dataset
    
    train_dataset = datasets.PointcloudSceneAcronymAndSDFDataset(opt.data_root,
                                                                    num_scene_pts=args["num_scene_points"], num_target_pts=args["num_target_points"])
    train_dataloader = DataLoader(train_dataset, num_workers = opt.num_workers, batch_size=args['TrainSpecs']['batch_size'], 
                                  shuffle=True, drop_last=True, pin_memory=True)
    test_dataset = datasets.PointcloudSceneAcronymAndSDFDataset(opt.data_root,  split='test',
                                                                num_scene_pts=args["num_scene_points"], num_target_pts=args["num_target_points"])
    test_dataloader = DataLoader(test_dataset, num_workers = 1, batch_size=1, shuffle=True, drop_last=True,
                                 pin_memory=True)

    ## Model
    args['device'] = device
    model = loader.load_model(args)

    # Losses
    loss = losses.get_losses(args)
    loss_fn = val_loss_fn = loss.loss_fn

    ## Summaries
    summary = summaries.get_summary(args, opt.summary)

    ## Optimizer
    lr_schedules = get_learning_rate_schedules(args)
    optimizer = torch.optim.Adam([
            {
                "params": model.vision_encoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": model.feature_encoder.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
            {
                "params": model.decoder.parameters(),
                "lr": lr_schedules[2].get_learning_rate(0),
            },
        ])

    # Train
    if not opt.eval:
        trainer.train(model=model.float(), train_dataloader=train_dataloader, epochs=args['TrainSpecs']['num_epochs'], model_dir= exp_dir,
                    summary_fn=summary, device=device, lr=1e-4, optimizers=[optimizer],
                    steps_til_summary=args['TrainSpecs']['steps_til_summary'],
                    epochs_til_checkpoint=args['TrainSpecs']['epochs_til_checkpoint'],
                    loss_fn=loss_fn, iters_til_checkpoint=args['TrainSpecs']['iters_til_checkpoint'],
                    clip_grad=False, val_loss_fn=val_loss_fn, overwrite=True,
                    val_dataloader=test_dataloader)
    else:
        if opt.eval_ckpt is not None:            
            states = torch.load(opt.eval_ckpt, map_location=device)
            model.load_state_dict(states['model_state'], strict=True)
            total_steps = states['steps']   
            start_epochs = total_steps // len(train_dataloader)
            trainer.eval(model=model, val_dataloader=test_dataloader, logdir=exp_dir, summary_fn=summary,
                            loss_fn=loss_fn, device=device, epoch = start_epochs, total_steps = total_steps)

        else:
            checkpoints_dir = os.path.join(exp_dir, 'checkpoints')
            writer = SummaryWriter(os.path.join(exp_dir, 'eval_summaries'))
            ckpts = glob.glob(os.path.join(checkpoints_dir, 'model_epoch_????_iter_??????.pth'))
            ckpts.sort()
            for ckpt in ckpts:
                print(" Evaluate checkpoint: ", ckpt)
                states = torch.load(ckpt, map_location=device)
                model.load_state_dict(states['model_state'], strict=True)
                total_steps = states['steps']   
                start_epochs = total_steps // len(train_dataloader)

                trainer.eval(model=model, val_dataloader=test_dataloader, loss_fn=loss_fn, logdir=exp_dir, summary_fn=summary,
                                 device=device, writer=writer, epoch = start_epochs, total_steps = total_steps)

if __name__ == '__main__':
    args = parse_args()
    main(args)