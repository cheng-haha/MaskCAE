from torch.optim.lr_scheduler import MultiStepLR,ExponentialLR,CosineAnnealingLR,StepLR,LambdaLR
import torch
from configs.base_configs import args
import numpy as np
import torch.optim as optim
import torch.nn as nn
from timm.models.layers import trunc_normal_
import models
import random
import math
from bisect import bisect_right
import numpy as np
import os
from pprint import pformat
from typing import List, Tuple, Callable
from timm.optim import Lamb
from torch.optim import SGD,AdamW,Adam
from functools import partial
from timm.loss import SoftTargetCrossEntropy, BinaryCrossEntropy
import sys

def LrSchedulerSet(optimizer,args):
    '''
    default: StepLR , gamma is 0.1 , step_size is 40
    '''
    if args.lr_scheduler == 'C':
        print('==>LrScheduler Set CosineAnnealingLR')
        return  CosineAnnealingLR( optimizer, T_max=args.n_epochs )
    elif args.lr_scheduler == 'S':
        print(f'==>LrScheduler Set StepLR , decay epoch is {args.decay_epochs} , gamma is {args.gamma}')
        return  StepLR( optimizer, step_size=args.decay_epochs, gamma=args.gamma )
    elif args.lr_scheduler == 'E':
        print('==>LrScheduler Set ExponentialLR')
        return  ExponentialLR( optimizer, gamma=args.gamma )
    elif args.lr_scheduler == 'M':
        print(f'==>LrScheduler Set MultiStepLR , scale is {args.milestones} , gamma is {args.gamma} ')
        return  MultiStepLR( optimizer , args.milestones , gamma=args.gamma )
    elif args.lr_scheduler == 'W':
        print(f'==>LrScheduler Set MultiStepLR with Warm up {args.warm_up_epochs}, scale is {args.milestones} , gamma is {args.gamma} ')
        # args.linear_gap = args.learning_rate / args.warm_up_epochs
        warm_up_with_multistep_lr = lambda epoch:  epoch / args.warm_up_epochs \
            if epoch <= args.warm_up_epochs else args.gamma**len([m for m in args.milestones if m <= epoch])
        return  LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)
    elif args.lr_scheduler == None and args.handcrafted_lr:
        print(f'==>LrScheduler is set to manually adjust')
    else:
        raise NotImplementedError('Not Implement this Lr_Scheduler!')

def adjust_lr(optimizer, epoch, args, step=0, all_iters_per_epoch=0):
    '''this version is MultiStepLR with warm up'''
    cur_lr = 0.
    if epoch < args.warm_up_epochs:
        cur_lr = args.learning_rate * float(1 + step + epoch*all_iters_per_epoch)/(args.warm_up_epochs *all_iters_per_epoch)
    else:
        epoch = epoch - args.warm_up_epochs
        # multistep:
        cur_lr = args.learning_rate * 0.1 ** bisect_right(args.milestones, epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr

    return cur_lr



def set_seed(seed):
    # fix seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def GetModel():
    try:
        model       = getattr(models, args.model)(args.dataset).cuda()
    except Exception as e:
        raise NotImplementedError("{} is not implemented".format(args.model))
    return model


def GetMOS(opt_type='sgd'):
    '''
    M: Model
    O: Optimier
    S: Scheduler
    '''
    # print(args.model)
    try:
        model = getattr(models, args.model)(args.dataset).cuda()
    except Exception as e:
        raise NotImplementedError("{} is not implemented".format(args.model))

    if opt_type == 'sgd':
        optimizer = optim.SGD(      params          =   model.parameters(),  
                                    lr              =   args.learning_rate,
                                    momentum        =   args.momentum ,
                                    weight_decay    =   args.weight_decay,
                                    # nesterov        =   True 
                                )
    elif opt_type == 'adam':
        optimizer = optim.Adam(     params          =   model.parameters()  , 
                                    lr              =   args.learning_rate  , 
                                    # betas         =   [0.9,0.999]         ,
                                    weight_decay    =   args.weight_decay
                               )
    elif opt_type == 'adamw':
        optimizer = optim.AdamW(    params          =   model.parameters()  , 
                                    lr              =   args.learning_rate  , 
                                    # betas         =   [0.9,0.999]         ,
                                    weight_decay    =   args.weight_decay
                               )
    else:
        raise NotImplementedError
    scheduler = LrSchedulerSet( optimizer , args )
    return model, optimizer, scheduler


def create_model_opt() :
    model: torch.nn.Module = getattr(models, args.model)(args.dataset).cuda()
    model_para = f'{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M'
    print(f'[model={args.model}] [#para={model_para}] {model}\n')
    model.train()
    opt_cls = {
        'sgd':  SGD,
        'adam': Adam, 'adamw': AdamW,
        'lamb': partial(Lamb, max_grad_norm=1e7, always_adapt=True, bias_correction=False),
    }
    param_groups: List[dict] = get_param_groups(model, nowd_keys={'cls_token', 'pos_embed', 'mask_token', 'gamma'}, lr_scale=args.lr_scale)
    optimizer = opt_cls[args.opt_type](param_groups, lr=args.learning_rate, weight_decay=args.weight_decay) if args.opt_type !='sgd' else opt_cls[args.opt_type](param_groups, lr=args.learning_rate, weight_decay=args.weight_decay,momentum=args.momentum)
    print(f'[optimizer={type(optimizer)}]')
   
    if 'lamb' in args.opt_type:
        # label smoothing is solved in AdaptiveMixup with `label_smoothing`, so here smoothing=0
        criterion = BinaryCrossEntropy(smoothing=0, target_threshold=None)
    else:
        criterion = nn.CrossEntropyLoss()
    print(f'[loss_fn] {criterion}')
    return  model, optimizer, criterion

def lr_wd_annealing(optimizer, peak_lr, wd, cur_it, wp_it, max_it):
    wp_it = round(wp_it)
    if cur_it < wp_it:
        cur_lr = 0.005 * peak_lr + 0.995 * peak_lr * cur_it / wp_it
    else:
        ratio = (cur_it - wp_it) / (max_it - 1 - wp_it)
        cur_lr = 0.001 * peak_lr + 0.999 * peak_lr * (0.5 + 0.5 * math.cos(math.pi * ratio))
    
    min_lr, max_lr = cur_lr, cur_lr
    min_wd, max_wd = wd, wd
    for param_group in optimizer.param_groups:
        scaled_lr = param_group['lr'] = cur_lr * param_group.get('lr_scale', 1)  # 'lr_scale' could be assigned
        min_lr, max_lr = min(min_lr, scaled_lr), max(max_lr, scaled_lr)
        scaled_wd = param_group['weight_decay'] = wd * param_group.get('weight_decay_scale', 1)  # 'weight_decay_scale' could be assigned
        min_wd, max_wd = min(min_wd, scaled_wd), max(max_wd, scaled_wd)
    return min_lr, max_lr, min_wd, max_wd

def get_param_groups(model, nowd_keys=(), lr_scale=0.0):
    using_lr_scale = hasattr(model, 'get_layer_id_and_scale_exp') and 0.0 < lr_scale < 1.0
    print(f'[get_ft_param_groups][lr decay] using_lr_scale={using_lr_scale}, ft_lr_scale={lr_scale}')
    para_groups, para_groups_dbg = {}, {}
    
    for name, para in model.named_parameters():
        if not para.requires_grad:
            continue  # frozen weights
        if len(para.shape) == 1 or name.endswith('.bias') or any(k in name for k in nowd_keys):
            wd_scale, group_name = 0., 'no_decay'
        else:
            wd_scale, group_name = 1., 'decay'
        
        if using_lr_scale:
            layer_id, scale_exp = model.get_layer_id_and_scale_exp(name)
            group_name = f'layer{layer_id}_' + group_name
            this_lr_scale = lr_scale ** scale_exp
            dbg = f'[layer {layer_id}][sc = {lr_scale} ** {scale_exp}]'
        else:
            this_lr_scale = 1
            dbg = f'[no scale]'
        
        if group_name not in para_groups:
            para_groups[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': this_lr_scale}
            para_groups_dbg[group_name] = {'params': [], 'weight_decay_scale': wd_scale, 'lr_scale': dbg}
        para_groups[group_name]['params'].append(para)
        para_groups_dbg[group_name]['params'].append(name)
    
    for g in para_groups_dbg.values():
        g['params'] = pformat(', '.join(g['params']), width=200)
    
    print(f'[get_ft_param_groups] param groups = \n{pformat(para_groups_dbg, indent=2, width=250)}\n')
    return list(para_groups.values())


def load_checkpoint(args,resume_from, model, optimizer):
    try:
        if args.ablationMode != '':
            resume_from = resume_from+ f'{args.ablationMode}Mode_{args.pre_epochs}Epochs_{args.dataset}_{args.Part}Part_{args.num_of_cv}NCV_{args.model}_{args.mask}_pretrained.pth' 
        else:
            resume_from = resume_from+ f'{args.dataset}_{args.Part}Part_{args.num_of_cv}NCV_{args.model}_{args.mask}_pretrained.pth'
    except AttributeError:
        print('some variable is not exist')

    if os.path.exists(resume_from):
        print(f'==> ckpt `is {resume_from}`!')
        # return 0, '[no performance_desc]'
        print(f'[try to resume from file `{resume_from}`]')
        checkpoint = torch.load(resume_from, map_location='cpu')
        assert checkpoint.get('is_pretrain', False) == False, 'Please do not use `*_still_pretraining.pth`, which is ONLY for resuming the pretraining. Use `*_pretrained.pth` or `*_finetuned*.pth` instead.'
        
        ep_start, performance_desc = checkpoint.get('epoch', -1) + 1, checkpoint.get('performance_desc', '[no performance_desc]')
        missing, unexpected = model.load_state_dict(checkpoint.get('module', checkpoint), strict=False)
        print(f'[load_checkpoint] missing_keys={missing}')
        print(f'[load_checkpoint] unexpected_keys={unexpected}')
        print(f'[load_checkpoint] ep_start={ep_start}, performance_desc={performance_desc}')
        
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print(f'==> ckpt `{resume_from}` not exist')
        if args.model in ['BaseCNN','BaseCNNSup','DeepConvLstm','Transformer_HAR','DeepConvLstmAttn','SA_HAR']:
            pass
        else:
            sys.exit(1)
