'''
Description: 
Date: 2023-05-07 18:23:46
LastEditTime: 2024-08-13 18:55:56
FilePath: /chengdongzhou/action/MaskCAE/main.py
'''

import os
import torch
from configs.base_configs import args, dict_to_markdown
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

import numpy as np

from create_dataset import GetHarDataset, dotdict
from dataloaders import data_set
from utils.logger import initialize_logger,Recorder
from utils.setup import set_seed,GetModel
from train import MetaTrain
from trainers.evalution import evaluate
import warnings
import wandb
warnings.filterwarnings("ignore")
import pyfiglet
print('------------------------------------------------------------------')
result = pyfiglet.figlet_format(text="MaskCAE", font="slant")
print(result)
print('------------------------------------------------------------------')
use_cuda = torch.cuda.is_available()

DATASET_PRE = args.dataset_pre



if __name__ == "__main__":
    print(dict_to_markdown(vars(args)))
    args.save_folder = os.path.join(args.model_path, args.dataset, args.model, args.mode , args.trial)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    # logging    
    log_dir = os.path.join(args.save_folder, 'train.log')
    logger  = initialize_logger(log_dir)
    # writer  = SummaryWriter(args.save_folder + '/run')
    total_losses, acc_tests, mf1_tests, wf1_tests, recall_tests, precision_tests =  [ [ [] for i in range(args.times)] for i in range(6) ]
    for i in range(args.times):
        logger.info(f'/n-------run time {i}--------\n')
        args.time   = i
        set_seed(i+1)

        data_args   = dotdict()

        data_args.yaml_path = 'configs/yaml/data.yaml'
        args.is_sup         = False
                
        data_args, dataset\
                    = GetHarDataset(dataset_name=args.dataset,data_args=data_args)
        for currenct_cv in range(dataset.num_of_cv):
            print("================ {} Mode ====================".format(dataset.exp_mode))
            print("================ {} CV   ====================".format(dataset.num_of_cv))
            dataset.update_train_val_test_keys()
            args.Part       = dataset.index_of_cv
            args.num_of_cv  = dataset.num_of_cv
            recorder        = Recorder(args)
            
            train_data      = data_set(data_args,dataset,"train")
            test_data       = data_set(data_args,dataset,"test")
            vali_data       = data_set(data_args,dataset,"vali")
        
            wandb.init(
                project=f'{args.mode}',
                reinit=True,
                config=vars(args),
                # entity='NanjingNormalUniversity',
                # settings=wandb.Settings(start_method="fork")
                # mode='offline'
                # name=recorder.save_alg+f'_{i}seed',
                # id=recorder.save_alg+f'_{i}seed',
                # resume="allow"
                ) if args.use_wandb else None
            #train
            
            train = MetaTrain(args)
            print(f'The training method is {train.__name__}')
            stat = train(train_data,vali_data,logger)
            recorder.rs_valid_loss       = stat['Test_losses']
            recorder.rs_valid_acc        = stat['Acc_tests']
            recorder.rs_valid_mf1        = stat['mF1_tests']
            recorder.rs_valid_wf1        = stat['wF1_tests']
            recorder.rs_valid_recall     = stat['Recall_tests']
            recorder.rs_valid_precision  = stat['Precision_test']
            recorder.save_results(i)
            wandb.join() if args.use_wandb else None
            model = GetModel()
            total_loss, acc_test, mf1_test, wf1_test, recall_test, precision_test , stat = evaluate(model,logger=logger,epoch = stat['best_epoch'],eval_loader=test_data,stat=stat)
            for res ,save_lsit in zip( [total_loss, acc_test, mf1_test, wf1_test, recall_test, precision_test], [total_losses, acc_tests, mf1_tests, wf1_tests, recall_tests, precision_tests]):
                save_lsit[i].append(res)

    logger.info(f'==>LOCV  Acc List is {acc_tests}\n')
    logger.info(f'==>LOCV  mF1 List is {mf1_tests}\n')
    logger.info(f'==>LOCV  wF1 List is {wf1_tests}\n')
    print('====================== Averaging Results ======================')
    average_times   = lambda lst : list(np.mean(lst,axis=0)) if args.dataset != 'uschad' else list(np.squeeze(lst))
    total_losses    = average_times(total_losses)
    acc_tests       = average_times(acc_tests)
    mf1_tests       = average_times(mf1_tests)
    wf1_tests       = average_times(wf1_tests)
    recall_tests    = average_times(recall_tests)
    precision_tests = average_times(precision_tests)
    for metric in ['acc', 'mf1', 'wf1', 'recall', 'precision']  :
        setattr( recorder, f'rs_test_{metric}', eval( f'{metric}_tests' ) )
        
    logger.info(f'==>LOCV Averaged Acc is {np.around( np.mean( recorder.rs_test_acc       ) * 100 , 3 )}\n')
    logger.info(f'==>LOCV Averaged mF1 is {np.around( np.mean( recorder.rs_test_mf1       ) * 100 , 3 )}\n')
    logger.info(f'==>LOCV Averaged wF1 is {np.around( np.mean( recorder.rs_test_wf1       ) * 100 , 3 )}\n')

