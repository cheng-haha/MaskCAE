'''
Description: 
Date: 2023-03-18 16:59:36
LastEditTime: 2024-08-13 17:54:05
FilePath: /chengdongzhou/action/MaskCAE/trainers/maskcae_trainer.py
'''
from torch.utils.data import DataLoader
from sklearn import metrics
from configs.base_configs import args
from utils.setup import create_model_opt , lr_wd_annealing,load_checkpoint ,LrSchedulerSet
import torch.nn as nn
from utils.metric import Timer
from utils.logger import Statistics
from tqdm import tqdm, trange
import torch
from torch.nn import functional as F
import wandb 
from trainers.evalution import evaluate
import os 

def freeze_modules_by_name(model, module_names):
    for name, module in model.named_modules():
        if name in module_names:
            for param in module.parameters():
                param.requires_grad = False
    return model

def LinearEavlution(model):
    return freeze_modules_by_name(model,['layer1','layer2','layer3','layer4'])

def Partial_ﬁne_tuning(model,layer_numbers):
    freeze_layer_numbers = 4 - layer_numbers
    freeze_layers = [f'layer{ln+1}' for ln in range(freeze_layer_numbers)]
    return freeze_modules_by_name(model,freeze_layers)

def GetEvaMode(model,logger):
    if args.linear_evaluation:
        logger.info( '==> Linear Probing....')
        model = LinearEavlution(model)
        eva_flag = 'lp'
    else:
        if args.partial_ft:
            logger.info( '==> Partial Fine Tuning....')
            model = Partial_fine_tuning(model,args.ft_layernumbers)
            eva_flag = 'pft'
        else:
            logger.info( '==> Fine Tuning....')
            eva_flag = 'ft'
    return model,eva_flag

def maskcae_train(train_set,valid_set,logger):
    train_loader = DataLoader( train_set, batch_size=args.batch_size, shuffle=True, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True )
    val_loader   = DataLoader( valid_set, batch_size=args.batch_size, shuffle=False, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True )

    model, optimizer, criterion_cls = create_model_opt()

    stat = {}
    stat['best_mf1'] = 0.0
    # wandb.watch(model,log_freq=50,log="all")
    # logging
    logger.info( args )
    print( f"==> load pre-trained model..." )
    load_checkpoint(args,resume_from=args.pre_trained_path,model=model,optimizer=optimizer)
    model,eva_flag = GetEvaMode(model=model,logger=logger)
    timer = Timer()
    iters_train     = len(train_loader)
    Test_losses ,Acc_tests, mF1_tests,wF1_tests, Recall_tests, Precision_tests = [],[],[],[],[],[]
    for epoch in trange( args.epochs, desc='Training_epoch' ):
        model.train()
        total_num, total_loss = 0, 0
        label_list, predicted_list = [], []
        for idx, (data, label) in enumerate(train_loader):
            # adjust lr and wd
            cur_it  = idx + epoch * iters_train
            min_lr, max_lr, min_wd, max_wd = lr_wd_annealing(optimizer, args.learning_rate, args.weight_decay, cur_it, args.warm_up_epochs * iters_train, args.epochs * iters_train)
            timer.start()
            inputs, label  = data.cuda().float(), label.cuda().long()
            output         = model(inputs)['output']
            loss    = criterion_cls(output, label)
            optimizer.zero_grad()
            loss.backward()
            # orig_norm = nn.utils.clip_grad_norm_( model.parameters(), 100 )
            optimizer.step()

            timer.stop()
            total_loss += loss.detach().item() 
            with torch.no_grad():
                _, predicted = torch.max( output, 1 )
                label_list.append(label)
                predicted_list.append( predicted )
            total_num = total_num + len( label )
            
        label       = torch.cat(label_list).cpu().detach().numpy()
        predicted   = torch.cat(predicted_list).cpu().detach().numpy()
        acc_train   = ( predicted == label ).sum() / total_num
        f1_train    = metrics.f1_score( label, predicted, average='macro' )
        loss_train  = total_loss/len(train_loader)

        logger.info( f'Epoch:[{epoch}/{args.epochs}] - {eva_flag}_loss:{loss_train:.7f}, {eva_flag}_train@Acc: {acc_train:.5f}, {eva_flag}_train@F1: {f1_train:.5f}')
        
        wandb.log({f'{eva_flag} train loss'  :loss_train ,
                   f'{eva_flag} train Acc'   :acc_train ,
                   f'{eva_flag} train F1'    :f1_train ,
                   'min_lr':min_lr,
                   'max_lr':max_lr,
                   'min_wd':min_wd,
                   'max_wd':max_wd
                   } ,
                   commit=False) if args.use_wandb else None
        
        Test_loss , Acc_test, mF1_test,wF1_test, Recall_test, Precision_test, stat = evaluate(model, logger=logger, eval_loader = val_loader , epoch =  epoch , is_test=False, stat=stat)

        for elem , save_res in zip((Test_loss,Acc_test, mF1_test, wF1_test, Recall_test, Precision_test),(Test_losses,Acc_tests, mF1_tests,wF1_tests, Recall_tests, Precision_tests)):
            save_res.append(elem)

    stat = Statistics(stat,Test_losses, Acc_tests, mF1_tests, wF1_tests , Recall_tests, Precision_tests,sum_time= timer.sum() )
    # wandb.save('model.h5')

    return stat