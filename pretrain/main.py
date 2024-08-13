'''
Description: 
Date: 2023-06-04 12:00:53
LastEditTime: 2024-08-13 16:08:29
FilePath: /chengdongzhou/action/MaskCAE/pretrain/main.py
'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


import datetime
import math
import sys
import time
from functools import partial
from typing import List

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

import dist
import encoder
from decoder import LightDecoder
from models import build_sparse_encoder
from sampler import DistInfiniteBatchSampler, worker_init_fn
from MaskCAE import MaskCAE
from utils import arg_util, misc, lamb
from utils.lr_control import lr_wd_annealing, get_param_groups
import numpy as np
import random


def get_data_size(data_name, is_sup = False ):
    Model_Seen_SSL_F_Size = {
        'ucihar': (     32, 9     ) ,
        'motion': (     32, 12    ) ,
        'uschad': (     32, 6     ) ,
    }
    Model_Seen_Sup_F_Size = {
        'ucihar': (     128, 9     ) ,
        'motion': (     128, 12    ) ,
        'uschad': (     32,  6     ) ,
    }
    if is_sup:
        size_dict = Model_Seen_Sup_F_Size
    else:
        size_dict = Model_Seen_SSL_F_Size
    if data_name in size_dict:
        pass
    else:
        raise Exception( 'please input correct data name')
    return size_dict[data_name]


def main_pt(args:arg_util.Args,dataset_train):

    
    # build data
    print(f'[build data for pre-training] ...\n')
    data_loader_train   = DataLoader(
                            dataset         =   dataset_train, 
                            num_workers     =   args.dataloader_workers, 
                            pin_memory      =   True,
                            batch_sampler   =   DistInfiniteBatchSampler(
                                    dataset_len     =   len(dataset_train), 
                                    glb_batch_size  =   args.glb_batch_size,
                                    shuffle         =   True, 
                                    filling         =   True, 
                                    rank            =   dist.get_rank(), 
                                    world_size      =   dist.get_world_size(),
                            ), 
                            worker_init_fn  =   worker_init_fn
    )
    itrt_train, iters_train = iter(data_loader_train), len(data_loader_train)
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size_per_gpu}, iters_train={iters_train}')
    
    # build encoder and decoder
    print(f'==> is sup? {args.is_sup}')
    enc: encoder.SparseEncoder  = build_sparse_encoder(args.model,dataset_name =args.dataset_name ,input_size=get_data_size(args.dataset_name,args.is_sup), sbn=args.sbn, drop_path_rate=args.dp, verbose=False,mode=args.ablation_mode)
    dec                         = LightDecoder(enc.downsample_raito, sbn=args.sbn,mode=args.ablation_mode)
    model                       = MaskCAE(
                                        sparse_encoder          =   enc, 
                                        dense_decoder           =   dec, 
                                        mask_ratio              =   args.mask,
                                        densify_norm            =   args.densify_norm, 
                                        sbn                     =   args.sbn,
                                        mode                    =   args.ablation_mode,
                                        sup_window_length       =   args.is_sup
                                ).cuda()
    print(f'[PT model] model = {model}\n')

    
    # build optimizer and lr_scheduler
    param_groups: List[dict] = get_param_groups(model, nowd_keys={'cls_token', 'pos_embed', 'mask_token', 'gamma'})
    opt_clz = {
        'sgd':      partial(torch.optim.SGD,        momentum=0.9,           nesterov=True       ),
        'adamw':    partial(torch.optim.AdamW,      betas=(0.9,             args.ada)           ),
        'lamb':     partial(lamb.TheSameAsTimmLAMB, betas=(0.9, args.ada),  max_grad_norm=5.0   ),
    }[args.opt]
    optimizer = opt_clz(params=param_groups, lr=args.lr, weight_decay=0.0)
    print(f'[optimizer] optimizer({opt_clz}) ={optimizer}\n')
    
    # try to resume
    ep_start, performance_desc = misc.load_checkpoint(args.resume_from, model, optimizer)
    if ep_start >= args.ep: # load from a complete checkpoint file
        print(f'  [*] [PT already done]    Min/Last Recon Loss: {performance_desc}')
    else:   # perform pre-training
        tb_lg = misc.TensorboardLogger(args.tb_lg_dir, is_master=dist.is_master(), prefix='pt')
        min_loss = 1e9
        print(f'[PT start] from ep{ep_start}')
        
        pt_start_time = time.time()
        for ep in range(ep_start, args.ep):
            ep_start_time = time.time()
            tb_lg.set_step(ep * iters_train)
            if hasattr(itrt_train, 'set_epoch'):
                itrt_train.set_epoch(ep)
            
            stats = pre_train_one_ep(ep, args, tb_lg, itrt_train, iters_train, model, optimizer)
            last_loss   = stats['last_loss']
            min_loss    = min(min_loss, last_loss)
            performance_desc = f'{min_loss:.4f} {last_loss:.4f}'
            if args.ablation_mode != '':
                misc.save_checkpoint(f'{args.ablation_mode}Mode_{args.ep}Epochs_{args.dataset_name}_{args.Part}Part_{args.num_of_cv}NCV_{args.model}_{args.mask}_still_pretraining.pth', args, ep, performance_desc, model.state_dict(with_config=True), optimizer.state_dict())
                misc.save_checkpoint_for_finetune(f'{args.ablation_mode}Mode_{args.ep}Epochs_{args.dataset_name}_{args.Part}Part_{args.num_of_cv}NCV_{args.model}_{args.mask}_pretrained.pth', args, model.sparse_encoder.sp_cnn.state_dict())
            else:
                misc.save_checkpoint(f'{args.dataset_name}_{args.Part}Part_{args.num_of_cv}NCV_{args.model}_{args.mask}_still_pretraining.pth', args, ep, performance_desc, model.state_dict(with_config=True), optimizer.state_dict())
                misc.save_checkpoint_for_finetune(f'{args.dataset_name}_{args.Part}Part_{args.num_of_cv}NCV_{args.model}_{args.mask}_pretrained.pth', args, model.sparse_encoder.sp_cnn.state_dict())
            
            ep_cost     = round(time.time() - ep_start_time, 2) + 1    # +1s: approximate the following logging cost
            remain_secs = (args.ep-1 - ep) * ep_cost
            remain_time = datetime.timedelta(seconds=round(remain_secs))
            finish_time = time.strftime("%m-%d %H:%M", time.localtime(time.time() + remain_secs))
            print(f'  [*] [ep{ep}/{args.ep}]    Min/Last Recon Loss: {performance_desc},    Cost: {ep_cost}s,    Remain: {remain_time},    Finish @ {finish_time}')
            
            args.cur_ep = f'{ep + 1}/{args.ep}'
            args.remain_time, args.finish_time = str(remain_time), str(finish_time)
            args.last_loss = last_loss
            args.log_epoch()
            
            tb_lg.update(min_loss=min_loss, head='train', step=ep)
            tb_lg.update(rest_hours=round(remain_secs/60/60, 2), head='z_burnout', step=ep)
            tb_lg.flush()
        
        # finish pre-training
        tb_lg.update(min_loss=min_loss, head='result', step=ep_start)
        tb_lg.update(min_loss=min_loss, head='result', step=args.ep)
        tb_lg.flush()
        print(f'final args:\n{str(args)}')
        print('\n\n')
        print(f'  [*] [PT finished]    Min/Last Recon Loss: {performance_desc},    Total Cost: {(time.time() - pt_start_time) / 60 / 60:.1f}h\n')
        print('\n\n')
        tb_lg.close()
        time.sleep(10)
    
    args.remain_time, args.finish_time = '-', time.strftime("%m-%d %H:%M", time.localtime(time.time()))
    args.log_epoch()


def pre_train_one_ep(ep, args: arg_util.Args, tb_lg: misc.TensorboardLogger, itrt_train, iters_train, model: DistributedDataParallel, optimizer):
    model.train()
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('max_lr', misc.SmoothedValue(window_size=1, fmt='{value:.5f}'))
    header = f'[PT] Epoch {ep}:'
    
    optimizer.zero_grad()
    early_clipping = args.clip > 0 and not hasattr(optimizer, 'global_grad_norm')
    late_clipping = hasattr(optimizer, 'global_grad_norm')
    if early_clipping:
        params_req_grad = [p for p in model.parameters() if p.requires_grad]
    
    for it, (inp, _) in enumerate(me.log_every(iters_train, itrt_train, 3, header)):
        # adjust lr and wd
        min_lr, max_lr, min_wd, max_wd = lr_wd_annealing(optimizer, args.lr, args.wd, args.wde, it + ep * iters_train, args.wp_ep * iters_train, args.ep * iters_train)
        
        # forward and backward
        inp = inp.cuda().float()
        # MaskCAE.forward
        loss = model(inp, active_b1fs=None, vis=False)
        optimizer.zero_grad()
        loss.backward()
        loss = loss.item()
        if not math.isfinite(loss):
            print(f'[rk{dist.get_rank():02d}] Loss is {loss}, stopping training!', force=True, flush=True)
            sys.exit(-1)
        
        # optimize
        grad_norm = None
        if early_clipping: grad_norm = torch.nn.utils.clip_grad_norm_(params_req_grad, args.clip).item()
        optimizer.step()
        if late_clipping: grad_norm = optimizer.global_grad_norm
        torch.cuda.synchronize()
        
        # log
        me.update(last_loss=loss)
        me.update(max_lr=max_lr)
        tb_lg.update(loss=me.meters['last_loss'].global_avg, head='train_loss')
        tb_lg.update(sche_lr=max_lr, head='train_hp/lr_max')
        tb_lg.update(sche_lr=min_lr, head='train_hp/lr_min')
        tb_lg.update(sche_wd=max_wd, head='train_hp/wd_max')
        tb_lg.update(sche_wd=min_wd, head='train_hp/wd_min')
        
        if grad_norm is not None:
            me.update(orig_norm=grad_norm)
            tb_lg.update(orig_norm=grad_norm, head='train_hp')
        tb_lg.set_step()
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}


def set_seed(seed):
    # fix seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    from create_dataset import GetHarDataset, dotdict
    from dataloaders import data_set
    import os
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    print(f'initial args:\n{str(args)}')
    args.log_epoch()
    set_seed(0)


    if args.DATASET_PRE == 'self' or args.DATASET_PRE == 'Sup':
        data_args   = dotdict()
        if args.DATASET_PRE == 'Sup':
            data_args.yaml_path = 'configs/yaml/data_sup.yaml'
        else:
            data_args.yaml_path = 'configs/yaml/data.yaml'
        data_args, dataset\
                    = GetHarDataset(dataset_name=args.dataset_name,data_args=data_args)
        for currenct_cv in range(dataset.num_of_cv):
            print("================ {} Mode ====================".format(dataset.exp_mode))
            print("================ {} CV   ====================".format(dataset.num_of_cv))
            dataset.update_train_val_test_keys()
            args.Part       = dataset.index_of_cv
            args.num_of_cv  = dataset.num_of_cv    
            train_data      = data_set(data_args,dataset,"train")
            main_pt(args,train_data)

