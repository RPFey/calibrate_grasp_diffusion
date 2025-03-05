import os
import time
import datetime
import numpy as np
import torch

from collections import defaultdict

from se3dif.utils import makedirs, dict_to_device, ClusterStateManager
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAveragePrecision
from tqdm.autonotebook import tqdm
import logging

# log to .log file
cm = ClusterStateManager()

def compute_difference(grasp_prediction: torch.Tensor, grasp_gt: torch.Tensor) -> torch.Tensor:
    """
    Compute the difference between the predicted and ground truth grasps.
    
    Args:
        grasp_prediction (torch.Tensor): Predicted grasp poses.
            shape (B, N, 4, 4)
        grasp_gt (torch.Tensor): Ground truth grasp poses.
            shape (B, N, 4, 4)
    Returns:
        torch.Tensor: Difference between the predicted and ground truth grasps.
            shape (B, N, N, 2); (angular, translation)
    """
    grasp_prediction_rot = grasp_prediction[:, :, None, :3, :3]
    grasp_gt_rot = grasp_gt[:, None, :, :3, :3]
    
    # compute the angular distance
    R_diff = torch.matmul(grasp_prediction_rot, grasp_gt_rot.transpose(3, 4)) # (B, N, N, 3, 3)
    R_diff = torch.clamp(R_diff, -1, 1)
    delta_R = ( 
                torch.sum(R_diff[..., list(range(3)), list(range(3))], dim = -1) - 1
        ) / 2
    delta_R = torch.clamp(delta_R, -1, 1)
    angular_distance = torch.acos(delta_R) # (B, N, N)
    
    # compute translation distance
    t_diff = grasp_prediction[:, :, None, :3, 3] - grasp_gt[:, None, :, :3, 3] # (B, N, N, 3)
    translation_distance = torch.norm(t_diff, dim=-1) # (B, N, N)
    difference = torch.stack([angular_distance, translation_distance], dim=-1) # (B, N, N, 2)
    
    return difference

def train(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn=None, iters_til_checkpoint=None, val_dataloader=None, clip_grad=False, val_loss_fn=None,
          overwrite=True, optimizers=None, batches_per_validation=10,  rank=0, max_steps=None, device='cpu'):

    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    ## Build saving directories
    makedirs(model_dir)
    
    summaries_dir = os.path.join(model_dir, 'summaries')
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    
    if os.path.exists(os.path.join(checkpoints_dir, 'model_current.pth')):
        # load from the previous checkpoint
        states = torch.load(os.path.join(checkpoints_dir, 'model_current.pth'), map_location=device)
        model.load_state_dict(states['model_state'], strict=True)
        for optim, state in zip(optimizers, states['optimizers']):
            optim.load_state_dict(state)
        if rank == 0:
            logging.info("Loaded model from the previous checkpoint")
        total_steps = states['steps']             
    else:
        total_steps = 0

    if rank == 0:
        makedirs(summaries_dir)
        makedirs(checkpoints_dir)

        exp_name = datetime.datetime.now().strftime("%m.%d.%Y %H:%M:%S")
        writer = SummaryWriter(summaries_dir+ '/' + exp_name)
        logging.basicConfig(filename=os.path.join(summaries_dir, exp_name, 'training.log'), level=logging.INFO)

    with tqdm(range(total_steps, len(train_dataloader) * epochs)) as pbar:
        train_losses = []
        for epoch in range(epochs):
            # if not epoch % epochs_til_checkpoint and epoch and rank == 0:
            #     torch.save(model.state_dict(),
            #                os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
            #     np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
            #                np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = dict_to_device(model_input, device)
                gt = dict_to_device(gt, device)
                
                if cm.should_exit():
                    cm.requeue()

                forward_start_time = time.time()
                losses, iter_info = loss_fn(model, model_input, gt)
                
                if rank == 0:
                    logging.info("Forward time: %0.6f" % (time.time() - forward_start_time))
                    if 'ap' in iter_info:
                        writer.add_scalar("train_ap", iter_info["ap"], total_steps)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if rank == 0:
                        writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())
                if rank == 0:
                    writer.add_scalar("total_train_loss", train_loss, total_steps)

                if not total_steps % steps_til_summary and rank == 0:
                    if os.path.exists(os.path.join(checkpoints_dir, 'model_current.pth')):
                        os.remove(os.path.join(checkpoints_dir, 'model_current.pth'))
                    
                    state_dict = {
                        "model_state": model.state_dict(),
                        "optimizers": [optim.state_dict() for optim in optimizers],
                        "steps": total_steps
                    }
                    torch.save(state_dict, 
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    if summary_fn is not None:
                        summary_fn(model, model_input, gt, iter_info, writer, total_steps)

                backward_start_time = time.time()
                for optim in optimizers:
                    optim.zero_grad()
                train_loss.backward()
                if rank == 0:
                    logging.info("Backward time: %0.6f" % (time.time() - backward_start_time))

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                for optim in optimizers:
                    optim.step()

                if rank == 0:
                    pbar.update(1)

                if not total_steps % steps_til_summary and rank == 0:
                    print("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - forward_start_time))

                    if val_dataloader is not None:
                        # sample poses 
                        from se3dif.samplers import ApproximatedGrasp_AnnealedLD, Grasp_AnnealedLD

                        with torch.no_grad():
                            model.eval()
                            aps = []
                            
                            # compute val loss
                            val_losses = defaultdict(list)
                            for val_i, (model_input, gt) in tqdm(enumerate(val_dataloader), desc='Validation'):
                                model_input = dict_to_device(model_input, device)
                                gt = dict_to_device(gt, device)
                                bap = BinaryAveragePrecision(thresholds=None)
                                
                                if cm.should_exit():
                                    cm.requeue()
                                
                                # The visual context is already set in the loss function here ! 
                                val_loss, val_iter_info = loss_fn(model, model_input, gt, val=True)
                                
                                # sample grasps
                                H_grasps = model_input["x_ene_pos"] # (B, N, 4, 4)
                                num_grasps = H_grasps.shape[1]
                                generator = Grasp_AnnealedLD(model, batch=num_grasps, T=70, T_fit=50, k_steps=1, device=model_input['visual_context'].device)
                                H_sampled = generator.sample()[0] # (B, N, 4, 4)
                                H_sampled = H_sampled.unsqueeze(0)
                            
                                # compute the angular and translation distance between H and H_grasps
                                difference = compute_difference(H_sampled, H_grasps) # (B, N, N, 2)
                                acc_matches = torch.min(difference, dim=2)[0]
                                recall_matches = torch.min(difference, dim=1)[0]
                                
                                # Binary Classfication AP
                                if model.distribution != 'direct':
                                    p_pose, f_pose = model_input["x_ene_pos"].view(-1, 4, 4), model_input["x_neg_ene"].view(-1, 4, 4)
                                    pos_num = p_pose.shape[0]
                                    
                                    poses = torch.cat((p_pose, f_pose), dim=0)
                                    final_t = torch.ones((poses.shape[0], )).to(poses.device) * (1 / generator.T)
                                    logprob = model(poses, final_t).view(-1)
                                    pred = torch.exp(logprob)
                                    
                                    label = torch.ones(pred.shape[0]).to(pred.device).long()
                                    label[pos_num:] = 0
                                    bap(pred, label)      
                                    aps.append(bap.compute().item())                              

                                for name, value in val_loss.items():
                                    val_losses[name].append(value.cpu().numpy())
                                    
                                val_losses["acc_avg_angular"].append(acc_matches[:, :, 0].mean().item())
                                val_losses["acc_avg_translation"].append(acc_matches[:, :, 1].mean().item())
                                val_losses["acc"].append(
                                    torch.sum(
                                        torch.bitwise_and(acc_matches[:, :, 0] < 5 * np.pi / 180, acc_matches[:, :, 1] < 0.05)
                                    ).item()
                                )
                                
                                val_losses['rec_avg_angular'].append(recall_matches[:, :, 0].mean().item())
                                val_losses['rec_avg_translation'].append(recall_matches[:, :, 1].mean().item())
                                val_losses["recall"].append(
                                    torch.sum(
                                        torch.bitwise_and(recall_matches[:, :, 0] < 5 * np.pi / 180, recall_matches[:, :, 1] < 0.05)
                                    ).item()
                                )
                                val_losses['total'].append(num_grasps)
                                  
                        for loss_name, loss in val_losses.items():
                            if loss_name in ['total', 'acc']:
                                if loss_name == 'acc':
                                    acc_rate = sum(val_losses['acc']) / sum(val_losses['total'])
                                    writer.add_scalar('val_acc', acc_rate, total_steps)
                            else:
                                single_loss = np.mean(loss)
                                if summary_fn is not None:
                                    summary_fn(model, model_input, gt, val_iter_info, writer, total_steps, 'val_')
                                    writer.add_scalar('val_' + loss_name, single_loss, total_steps)
                        
                        mAP = np.array(aps).mean()
                        writer.add_scalar('val_mAP', mAP, total_steps)
                        model.train()

                if (iters_til_checkpoint is not None) and (not total_steps % iters_til_checkpoint) and rank == 0:
                    state_dict = {
                        "model_state": model.state_dict(),
                        "optimizers": [optim.state_dict() for optim in optimizers],
                        "steps": total_steps
                    }
                    
                    torch.save(state_dict,
                               os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                    # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%04d_iter_%06d.pth' % (epoch, total_steps)),
                    #            np.array(train_losses))

                total_steps += 1
                if max_steps is not None and total_steps==max_steps:
                    break

            if max_steps is not None and total_steps==max_steps:
                break
        
        state_dict = {
            "model_state": model.state_dict(),
            "optimizers": [optim.state_dict() for optim in optimizers],
            "steps": total_steps
        }   
        torch.save(state_dict, os.path.join(checkpoints_dir, 'model_final.pth'))
        # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'), np.array(train_losses))

        return model, optimizers