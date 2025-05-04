import os
import time
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

import matplotlib.pyplot as plt
from se3dif import datasets
from se3dif.utils import makedirs, dict_to_device, ClusterStateManager, SO3_R3
from se3dif.samplers.bullet_sampler import BulletEvaluator
from se3dif.samplers import Grasp_AnnealedLD
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAveragePrecision, BinaryPrecisionRecallCurve
from tqdm.autonotebook import tqdm
import logging

class MultipleDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
    
    def __len__(self):
        return sum(self.lengths)
    
    def __getitem__(self, idx):
        for i, length in enumerate(self.lengths):
            if idx < length:
                return self.datasets[i][idx]
            idx -= length
        raise IndexError("Index out of range")

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
          run_bullet=False, optimizers=None, batches_per_validation=10,  rank=0, max_steps=None, device='cpu',
          generation_step=-1):

    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    ## Build saving directories
    makedirs(model_dir)
    summaries_dir = os.path.join(model_dir, 'summaries')
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    acronym_dataset = train_dataloader.dataset
    
    if os.path.exists(os.path.join(checkpoints_dir, 'model_current.pth')):
        # load from the previous checkpoint
        states = torch.load(os.path.join(checkpoints_dir, 'model_current.pth'), map_location=device)
        model.load_state_dict(states['model_state'], strict=True)
        for optim, state in zip(optimizers, states['optimizers']):
            optim.load_state_dict(state)
        if rank == 0:
            logging.info("Loaded model from the previous checkpoint")
        total_steps = states['steps']   
        start_epochs = total_steps // len(train_dataloader)
    else:
        total_steps = 0
        start_epochs = 0

    if rank == 0:
        makedirs(summaries_dir)
        makedirs(checkpoints_dir)

        exp_name = datetime.datetime.now().strftime("%m.%d.%Y %H:%M:%S")
        writer = SummaryWriter(summaries_dir + '/' + exp_name)
        logging.basicConfig(filename=os.path.join(summaries_dir, exp_name, 'training.log'), level=logging.INFO)

    full_loader = [train_dataloader]
    val_dataloaders = {"acronym": val_dataloader}
    
    # create & add new bullet dataset
    if run_bullet:
        bullet_dataset = datasets.PointcloudSceneAcronymAndSDFDataset(acronym_dataset.root_dir, class_type=["Bullet"], split='train',
                                                                    num_scene_pts=acronym_dataset.num_scene_pts, num_target_pts=acronym_dataset.num_target_pts)
        bullet_dataloader = DataLoader(bullet_dataset, batch_size=train_dataloader.batch_size, shuffle=True, num_workers=train_dataloader.num_workers)
        full_loader = [train_dataloader, bullet_dataloader]
        # update the start epoch
        start_epochs = total_steps // ( len(train_dataloader) + len(bullet_dataloader) )
        
        bullet_val_dataset = datasets.PointcloudSceneAcronymAndSDFDataset(acronym_dataset.root_dir, class_type=["Bullet"], split='test',
                                                                    num_scene_pts=acronym_dataset.num_scene_pts, num_target_pts=acronym_dataset.num_target_pts)
        bullet_val_dataloader = DataLoader(bullet_val_dataset, batch_size=1, shuffle=True, num_workers=1)
        val_dataloaders["bullet"] = bullet_val_dataloader
    
    # The batch argument does not take effect here
    # generation_step = -1
    sampler = Grasp_AnnealedLD(model, batch=1, T=generation_step, T_fit=2, k_steps=1, device=device)
    with tqdm(range(total_steps, len(train_dataloader) * epochs)) as pbar:
        train_losses = []
        for epoch in range(start_epochs, epochs):
            # aps = []
            model.train()
            for loader in full_loader:
                for step, (model_input, gt) in enumerate(loader):
                    model_input = dict_to_device(model_input, device)
                    gt = dict_to_device(gt, device)
                    
                    if cm.should_exit():
                        writer.close()
                        cm.requeue()
                    
                    # generate poses in current batch
                    if generation_step > 0:
                        # # set the current generation step
                        # curr_iter_gen_step = np.random.randint(10, generation_step + 1)
                        # sampler.T = curr_iter_gen_step

                        # sampling
                        forward_start_time = time.time()
                        # disable grad
                        model.eval()
                        for params in model.parameters():  
                            params.requires_grad = False
                        
                        # set the visual context for the model
                        c = model_input['visual_context']
                        target_index = model_input.get('target_index', None)    
                        model.set_latent(c, target_index=target_index)
                        pose_batch = model_input["x_ene_pos"].shape[0] * model_input["x_ene_pos"].shape[1]
                        updated_samples, _ = sampler.sample(batch=pose_batch)
                        model_input["generated_grasps"] = updated_samples.detach().reshape(model_input["x_ene_pos"].shape[0], -1, 4, 4)
                        
                        # enable grad
                        for params in model.parameters():
                            params.requires_grad = True
                        model.train()
                        logging.info("Sampling time: %0.6f" % (time.time() - forward_start_time))

                    forward_start_time = time.time()
                    losses, iter_info = loss_fn(model, model_input, gt)
                    
                    if rank == 0:
                        forward_time  = time.time() - forward_start_time
                        logging.info("Forward time: %0.6f" % (forward_time))
                        if 'ap' in iter_info:
                            writer.add_scalar("train_ap", iter_info["ap"], total_steps)
                        if 'noise_ap' in iter_info:
                            writer.add_scalar("train_noise_ap", iter_info["noise_ap"], total_steps)

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()
                        if rank == 0:
                            writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += single_loss
                    
                    # if 'ap' in losses: 
                    #     aps.append(1 - losses["ap"]) # only for ap loss

                    train_losses.append(train_loss.item())
                    if rank == 0:
                        writer.add_scalar("total_train_loss", train_loss, total_steps)

                    backward_start_time = time.time()
                    for optim in optimizers:
                        optim.zero_grad()
                    train_loss.backward()
                    if rank == 0:
                        backward_time = time.time() - backward_start_time
                        logging.info("Backward time: %0.6f" % (backward_time))

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                    for optim in optimizers:
                        optim.step()

                    if rank == 0:
                        pbar.update(1)
                        pbar.set_postfix(suffix=f"Ep {epoch}, f-t {forward_time:.4f}, b-t {backward_time:.4f}")

                    total_steps += 1
                    if max_steps is not None and total_steps==max_steps:
                        break

            # dynamic adjust number of positive samples
            # if len(aps) > 0:
            #     average_ap = np.array(aps).mean()
            #     last_pos_num = train_dataloader.dataset.num_pos
            #     if average_ap > 0.75:
            #         logging.info("Decrease the number of positive samples")
            #         train_dataloader.dataset.num_pos = train_dataloader.dataset.num_pos // 2
            #         train_dataloader.dataset.num_pos = max(4, train_dataloader.dataset.num_pos)
            #     elif average_ap < 0.25:
            #         logging.info("Increase the number of positive samples")
            #         train_dataloader.dataset.num_pos = train_dataloader.dataset.num_pos * 2
            #         train_dataloader.dataset.num_pos = min(128, train_dataloader.dataset.num_pos)
            #     train_dataloader.dataset.num_neg = train_dataloader.dataset.num_neg + (last_pos_num - train_dataloader.dataset.num_pos)
            
            if (not epoch % (epochs_til_checkpoint // 2) ) and rank == 0:
                print("Step Summary ... ")
                if os.path.exists(os.path.join(checkpoints_dir, 'model_current.pth')):
                    os.remove(os.path.join(checkpoints_dir, 'model_current.pth'))
                
                state_dict = {
                    "model_state": model.state_dict(),
                    "optimizers": [optim.state_dict() for optim in optimizers],
                    "steps": total_steps
                }
                torch.save(state_dict, os.path.join(checkpoints_dir, 'model_current.pth'))
                
                # this function is weird ... 
                # TODO Uncomment these lines for visualization. 
                # It does not work on a40 card and will cause memory issue.
                # if summary_fn is not None:
                #     summary_fn(model, model_input, gt, iter_info, writer, total_steps)
            
            if (not epoch % epochs_til_checkpoint) and rank == 0 and epoch > 0:

                print("Save Checkpoint ... ")
                state_dict = {
                    "model_state": model.state_dict(),
                    "optimizers": [optim.state_dict() for optim in optimizers],
                    "steps": total_steps
                }
                torch.save(state_dict, os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                
                # evaltion 
                if val_dataloader is not None:
                    for name, val_loader in val_dataloaders.items():
                        eval(model, val_loader, val_loss_fn, logdir=model_dir, summary_fn=summary_fn, prefix=name,
                             device=device, writer=writer, epoch=epoch, total_steps=total_steps)
                    
                    # run bullet_evaluator
                    tmp_dir = os.path.join(model_dir, 'tmp')
                    evaluator = BulletEvaluator(tmp_dir, 64, save_data = False)   
                    try:
                        for n in [2, 4, 8]:
                            random_seeds = np.random.choice(int(1e4), (4, ), replace=False)
                            # random sample 20 seeds from 
                            totals = 0
                            successes = 0
                            
                            for s in random_seeds:
                                total, success = evaluator.evaluate_model(model, total_steps, n, s)
                                totals += len(total)
                                total_success = sum(np.array(success) > 0)
                                if not isinstance(total_success, int):
                                    total_success = total_success.item()
                                successes += total_success
                                
                            success_ratio = successes / totals
                            writer.add_scalar(f'bullet_eval_{n}', success_ratio, total_steps)

                            if successes <= 0:
                                raise StopIteration(f"Bullet evaluation failed at {n}")
                    
                    except StopIteration as e:
                        print(e)
                    
                    # create & add new bullet dataset
                    # if run_bullet:
                    #     complementary_dataset = evaluator.get_dataset([total_steps], list(range(2, 8)),
                    #                                                 dataset_args = {"num_scene_pts": acronym_dataset.num_scene_pts, 
                    #                                                                     "num_target_pts": acronym_dataset.num_target_pts} )
                    #     complementary_dataloader = DataLoader(complementary_dataset, 
                    #                                             batch_size=train_dataloader.batch_size, shuffle=True, num_workers=train_dataloader.num_workers)
                    #     full_loader = [complementary_dataloader, train_dataloader]

            if max_steps is not None and total_steps==max_steps:
                break
        
        state_dict = {
            "model_state": model.state_dict(),
            "optimizers": [optim.state_dict() for optim in optimizers],
            "steps": total_steps
        }   
        torch.save(state_dict, os.path.join(checkpoints_dir, 'model_final.pth'))
        # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'), np.array(train_losses))
        
        writer.close()
        return model, optimizers
    
    
def train_ebm(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir, loss_fn,
          summary_fn=None, iters_til_checkpoint=None, val_dataloader=None, clip_grad=False, val_loss_fn=None,
          run_bullet=False, optimizers=None, batches_per_validation=10,  rank=0, max_steps=None, device='cpu'):

    if optimizers is None:
        optimizers = [torch.optim.Adam(lr=lr, params=model.parameters())]

    if val_dataloader is not None:
        assert val_loss_fn is not None, "If validation set is passed, have to pass a validation loss_fn!"

    ## Build saving directories
    makedirs(model_dir)
    summaries_dir = os.path.join(model_dir, 'summaries')
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    acronym_dataset = train_dataloader.dataset
    
    if os.path.exists(os.path.join(checkpoints_dir, 'model_current.pth')):
        # load from the previous checkpoint
        states = torch.load(os.path.join(checkpoints_dir, 'model_current.pth'), map_location=device)
        model.load_state_dict(states['model_state'], strict=True)
        for optim, state in zip(optimizers, states['optimizers']):
            optim.load_state_dict(state)
        if rank == 0:
            logging.info("Loaded model from the previous checkpoint")
        total_steps = states['steps']   
        start_epochs = total_steps // len(train_dataloader)
    else:
        total_steps = 0
        start_epochs = 0

    if rank == 0:
        makedirs(summaries_dir)
        makedirs(checkpoints_dir)

        exp_name = datetime.datetime.now().strftime("%m.%d.%Y %H:%M:%S")
        writer = SummaryWriter(summaries_dir + '/' + exp_name)
        logging.basicConfig(filename=os.path.join(summaries_dir, exp_name, 'training.log'), level=logging.INFO)

    sample_steps = 10
    sampler = Grasp_AnnealedLD(model, batch=1, T=0, T_fit=sample_steps, k_steps=1, enable_time = False, device=device)
    with tqdm(range(total_steps, len(train_dataloader) * epochs)) as pbar:
        train_losses = []
        for epoch in range(start_epochs, epochs):

            # aps = []
            model.train()
            for step, (model_input, gt) in enumerate(train_dataloader):
                model_input = dict_to_device(model_input, device)
                gt = dict_to_device(gt, device)
                
                if cm.should_exit():
                    writer.close()
                    cm.requeue()
                
                sampling_iterations = 5
                pose_batch = model_input["x_ene_pos"].shape[0] * model_input["x_ene_pos"].shape[1]
                init_samples = None
                for k in range(sampling_iterations):
                
                    # update samples
                    forward_start_time = time.time()
                    # disable grad
                    model.eval()
                    for params in model.parameters():  
                        params.requires_grad = False
                    
                    # set the visual context for the model
                    c = model_input['visual_context']
                    target_index = model_input.get('target_index', None)    
                    model.set_latent(c, target_index=target_index)
                    
                    if init_samples is not None:
                        num_poes = init_samples.shape[0]  
                        resample = int(num_poes * 0.1)
                        reinit = SO3_R3().sample(resample).to(device)
                        reinit_index = np.random.choice(num_poes, resample, replace=False)
                        init_samples[reinit_index] = reinit
                    
                    updated_samples, scores = sampler.sample(H0=init_samples, batch=pose_batch)
                    model_input["generated_grasps"] = updated_samples.detach()
                    init_samples = model_input["generated_grasps"]
                    
                    # enable grad
                    for params in model.parameters():
                        params.requires_grad = True
                    model.train()
                    logging.info("Sampling time: %0.6f" % (time.time() - forward_start_time))
                    
                    forward_start_time = time.time()
                    losses, iter_info = loss_fn(model, model_input, gt)
                    
                    if rank == 0:
                        forward_time  = time.time() - forward_start_time
                        logging.info("Forward time: %0.6f" % (forward_time))
                        if 'ap' in iter_info:
                            writer.add_scalar("train_ap", iter_info["ap"], total_steps)
                        if 'noise_ap' in iter_info:
                            writer.add_scalar("train_noise_ap", iter_info["noise_ap"], total_steps)

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()
                        if rank == 0:
                            writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += single_loss
                        
                    train_losses.append(train_loss.item())
                    if rank == 0:
                        writer.add_scalar("total_train_loss", train_loss, total_steps)

                    backward_start_time = time.time()
                    for optim in optimizers:
                        optim.zero_grad()
                    train_loss.backward()
                    if rank == 0:
                        backward_time = time.time() - backward_start_time
                        logging.info("Backward time: %0.6f" % (backward_time))

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                    for optim in optimizers:
                        optim.step()

                    if rank == 0:
                        pbar.update(1)
                        pbar.set_postfix(suffix=f"Ep {epoch}, f-t {forward_time:.4f}, b-t {backward_time:.4f}")

                    total_steps += 1
                    if max_steps is not None and total_steps==max_steps:
                        break

            # dynamic adjust number of positive samples
            # if len(aps) > 0:
            #     average_ap = np.array(aps).mean()
            #     last_pos_num = train_dataloader.dataset.num_pos
            #     if average_ap > 0.75:
            #         logging.info("Decrease the number of positive samples")
            #         train_dataloader.dataset.num_pos = train_dataloader.dataset.num_pos // 2
            #         train_dataloader.dataset.num_pos = max(4, train_dataloader.dataset.num_pos)
            #     elif average_ap < 0.25:
            #         logging.info("Increase the number of positive samples")
            #         train_dataloader.dataset.num_pos = train_dataloader.dataset.num_pos * 2
            #         train_dataloader.dataset.num_pos = min(128, train_dataloader.dataset.num_pos)
            #     train_dataloader.dataset.num_neg = train_dataloader.dataset.num_neg + (last_pos_num - train_dataloader.dataset.num_pos)
            
            if (not epoch % (epochs_til_checkpoint // 2) ) and rank == 0:
                print("Step Summary ... ")
                if os.path.exists(os.path.join(checkpoints_dir, 'model_current.pth')):
                    os.remove(os.path.join(checkpoints_dir, 'model_current.pth'))
                
                state_dict = {
                    "model_state": model.state_dict(),
                    "optimizers": [optim.state_dict() for optim in optimizers],
                    "steps": total_steps
                }
                torch.save(state_dict, os.path.join(checkpoints_dir, 'model_current.pth'))
                
                # this function is weird ... 
                # TODO Uncomment these lines for visualization. 
                # It does not work on a40 card and will cause memory issue.
                # if summary_fn is not None:
                #     summary_fn(model, model_input, gt, iter_info, writer, total_steps)
            
            if (not epoch % epochs_til_checkpoint) and rank == 0:

                print("Save Checkpoint ... ")
                state_dict = {
                    "model_state": model.state_dict(),
                    "optimizers": [optim.state_dict() for optim in optimizers],
                    "steps": total_steps
                }
                torch.save(state_dict, os.path.join(checkpoints_dir, 'model_epoch_%04d_iter_%06d.pth' % (epoch, total_steps)))
                
                # evaltion 
                if val_dataloader is not None:
                    eval(model, val_dataloader, val_loss_fn, logdir=model_dir, summary_fn=summary_fn,
                         device=device, writer=writer, epoch=epoch, total_steps=total_steps)

            if max_steps is not None and total_steps==max_steps:
                break
        
        state_dict = {
            "model_state": model.state_dict(),
            "optimizers": [optim.state_dict() for optim in optimizers],
            "steps": total_steps
        }   
        torch.save(state_dict, os.path.join(checkpoints_dir, 'model_final.pth'))
        # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'), np.array(train_losses))
        
        writer.close()
        return model, optimizers
    

@torch.no_grad()
def eval(model, val_dataloader, loss_fn, logdir, summary_fn, prefix='',
            device=torch.device("cuda:0"),  writer=None, epoch = 0, total_steps = 0):
    # sample poses 
    from se3dif.samplers import ApproximatedGrasp_AnnealedLD, Grasp_AnnealedLD
    model.eval()
    # aps = []
    bap = BinaryAveragePrecision(thresholds=None)
    bprc = BinaryPrecisionRecallCurve(thresholds=None) 
    
    # log conformal score
    preds = []
    labels = []
    
    # compute val loss
    val_losses = defaultdict(list)
    for val_i, (model_input, gt) in tqdm(enumerate(val_dataloader), desc='Validation'):
        model_input = dict_to_device(model_input, device)
        gt = dict_to_device(gt, device)
        
        if cm.should_exit():
            if writer is not None:
                writer.close()
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
        if "x_neg_ene" in model_input:
            p_pose, f_pose = model_input["x_ene_pos"].view(-1, 4, 4), model_input["x_neg_ene"].view(-1, 4, 4)
            pos_num = p_pose.shape[0]
            
            poses = torch.cat((p_pose, f_pose), dim=0)
            final_t = torch.ones((poses.shape[0], )).to(poses.device) * model.final_t
            logprob = -1 * model(poses, final_t).view(-1)                
            label = torch.ones(logprob.shape[0]).to(logprob.device).long()
            label[pos_num:] = 0

            pred =  torch.exp(logprob - logprob.max()) \
                        if model.distribution == 'direct' \
                            else torch.exp(logprob)    
                            
            preds.append(pred)
            labels.append(label)
            
        # TODO Uncomment these lines for visualization. 
        # It does not work on a40 card and will cause memory issue.
        # if summary_fn is not None and val_i % 10 == 0:
        #     summary_fn(model, model_input, gt, val_iter_info, writer, int(total_steps * 1e3) + val_i, 'val_')
        
        for name, value in val_loss.items():
            if value is not None:
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
        
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    
    # for direct ("Bolzman"), simply normalize to [0, 1]
    # it will not change the order and affect AP computation
    bap.update(preds, labels) 
    bprc.update(preds, labels)
    
    # plot SR rate over quantile
    total_pred, total_gt = preds.cpu().numpy(), labels.cpu().numpy()
    n_bins = 15
    pred_q = np.linspace(0, 1, n_bins + 1)
    cnts = []
    conf = []
    sr = []
    for i in range(n_bins):
        quant_index = (total_pred >= pred_q[i]) & (total_pred < pred_q[i + 1])
        cnt = np.sum(quant_index)
        cnts.append(cnt)
        if cnt == 0:
            # if no samples in this bin, set sr to 0
            sr.append(0)
            conf.append(0)
        else:
            sr.append(np.mean(total_gt[quant_index]))
            conf.append(np.mean(total_pred[quant_index]))
    
    sr, cnts, conf = np.array(sr), np.array(cnts), np.array(conf)
    weight = cnts / np.sum(cnts)
    error = np.abs( conf - sr )
    ece = np.sum(weight * error)
    
    quantile_fig = plt.figure()
    ax = quantile_fig.add_subplot(111)
    ax.plot(np.arange(n_bins) / n_bins, sr)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Quantile")
    ax.set_ylabel("SR")
 
    mAP = bap.compute().item() # np.array(aps).mean()
    fig_, ax_ = bprc.plot(score=True)
    # save the pr curve stats from bprc
    
    print(f"Ep {epoch}, Validation mAP: {mAP}")
    stats = {}
    if writer is not None:
        writer.add_scalar(f'{prefix}val_mAP', mAP, total_steps)
        writer.add_scalar(f'{prefix}val_ece', ece, total_steps)
        writer.add_figure(f'{prefix}val_precision_recall_curve', fig_, total_steps)
        writer.add_figure(f'{prefix}val_quantile_sr', quantile_fig, total_steps)
    else:
        stats[f'{prefix}val_mAP'] = mAP
        stats[f'{prefix}val_ece'] = ece
        precision, recall, thresholds = bprc.compute()
        # convert to numpy
        precision, recall, thresholds = precision.cpu().numpy(), recall.cpu().numpy(), thresholds.cpu().numpy()
        fig_.savefig(os.path.join(logdir, f'{prefix}-precision_recall_curve-{total_steps}.png'))
        quantile_fig.savefig(os.path.join(logdir, f'{prefix}-quantile_sr-{total_steps}.png'))
        np.savez(os.path.join(logdir, f'{prefix}-precision_recall_curve-{total_steps}.npz'), 
                 precision=precision, recall=recall, thresholds=thresholds, quantile_sr = sr)
      
    for loss_name, loss in val_losses.items():
        if loss_name in ['total', 'acc']:
            if loss_name == 'acc':
                acc_rate = sum(val_losses['acc']) / sum(val_losses['total'])
                if writer is not None:
                    writer.add_scalar(f'{prefix}val_acc', acc_rate, total_steps)
                else:
                    stats[f'{prefix}val_acc'] = acc_rate
        else:
            single_loss = np.mean(loss)
            if writer is not None:
                writer.add_scalar(f'{prefix}val_' + loss_name, single_loss, total_steps)
            else:
                stats[f'{prefix}val_' + loss_name] = single_loss
            
    # compute cp score
    # S <=> 1 - pred ; F <=> pred
    cp_score = labels * (1 - preds) + (1 - labels) * preds
    
    # find the 90% quantile of cp_score
    cp_score = cp_score.cpu().numpy()
    cp_score_90 = np.quantile(cp_score, 0.9)
    cp_score_95 = np.quantile(cp_score, 0.95)
    
    if writer is not None:
        writer.add_scalar(f'{prefix}val_cp_90', cp_score_90, total_steps)
        writer.add_scalar(f'{prefix}val_cp_95', cp_score_95, total_steps)
    else:
        stats[f'{prefix}val_cp_90'] = cp_score_90
        stats[f'{prefix}val_cp_95'] = cp_score_95
    
    import pprint
    pprint.pprint(stats)
        