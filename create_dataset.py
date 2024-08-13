
import warnings
warnings.filterwarnings("ignore")

import os 
import sys 
sys.path.append(os.path.dirname(__file__))

from torch.utils.data import DataLoader

from dataloaders import data_set,data_dict
import torch
import yaml
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

WINDOWS_SAVE_PATH   = r"./MaskCAE/datasets/Sliding_window"
ROOT_PATH           = r"./data/raw"

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

  

def GetHarDataset(dataset_name , data_args):
    
    data_args.freq_save_path   = None
    data_args.window_save_path = WINDOWS_SAVE_PATH
    data_args.root_path        = ROOT_PATH

    data_args.drop_transition  = False

    data_args.batch_size       = 128
    data_args.shuffle          = True
    data_args.drop_last        = False
    data_args.train_vali_quote = 0.8
    data_args.down_sample      = True

    data_args.data_name           =  dataset_name  # wisdm


    # 是否作augmentation difference
    data_args.difference          =  False 

    # 是否作augmentation  filtering
    data_args.filtering           =  False

    # 是否作augmentation  magnitude
    data_args.magnitude           =  False
    # 是否按照权重抽取批次
    data_args.weighted_sampler    =  False

    data_args.load_all            = None
    # 是在load数据的时候  wavelet_filtering
    data_args.wavelet_filtering          = False
    data_args.number_wavelet_filtering   = 10

    data_args.datanorm_type       =  "standardization" # None ,"standardization", "minmax"

    data_args.pos_select          = None
    data_args.sensor_select       = None
    data_args.sample_wise         = False
    data_args.multicrop           = False

    data_args.representation_type = "time"
    

    config_file = open(data_args.yaml_path, mode='r')
    data_config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = data_config[data_args.data_name]
    
    data_args.root_path         = os.path.join(data_args.root_path,config["filename"])
    data_args.sampling_freq     = config["sampling_freq"]
    data_args.valid_rate        = config["valid_rate"]
    data_args.overlap_rate      = config["overlap_rate"]
    data_args.down_sample       = config["down_sample"]
    window_seconds              = config["window_seconds"]
    data_args.windowsize        = int(window_seconds * data_args.sampling_freq) 
    data_args.c_in              = config["num_channels"]
    data_args.exp_mode          = config["exp_mode"]

    data_args.input_length      =  data_args.windowsize
    # input information
    if data_args.wavelet_filtering :
        
        if data_args.windowsize%2==1:
            N_ds = int(torch.log2(torch.tensor(data_args.windowsize-1)).floor()) - 2
        else:
            N_ds = int(torch.log2(torch.tensor(data_args.windowsize)).floor()) - 2

        data_args.f_in            =  data_args.number_wavelet_filtering*N_ds+1
    else:
        data_args.f_in            =  1

    return data_args , data_dict[data_args.data_name](data_args)

def CheckDataloader(data_args , dataset):
    print("================ {} Mode ====================".format(dataset.exp_mode))
    print("================ {} CV ======================".format(dataset.num_of_cv))
    for i in range(dataset.num_of_cv):
        dataset.update_train_val_test_keys()
        train_data  = data_set(data_args,dataset,"train")
        test_data   = data_set(data_args,dataset,"test")
        vali_data   = data_set(data_args,dataset,"vali")
        
        
        # form the dataloader
        train_data_loader = DataLoader(train_data,  
                                        batch_size   =  data_args.batch_size,
                                        shuffle      =  data_args.shuffle,
                                        num_workers  =  4,
                                        drop_last    =  data_args.drop_last)

        vali_data_loader = DataLoader(vali_data,  
                                        batch_size   =  data_args.batch_size,
                                        shuffle      =  data_args.shuffle,
                                        num_workers  =  4,
                                        drop_last    =  data_args.drop_last)

        test_data_loader = DataLoader(test_data,  
                                        batch_size   =  data_args.batch_size,
                                        shuffle      =  data_args.shuffle,
                                        num_workers  =  4,
                                        drop_last    =  data_args.drop_last)

        for datas,label in train_data_loader:
            if data_args.multicrop:
                [print(data.shape) for data in datas ]
            else:
                print(datas.shape)
            break






def test(dataset):
    data_args       = dotdict()
    data_args.all   = False
    data_args.name  = dataset
    if data_args.all:
        assert data_args.name == None, f'can not choice the dataset {data_args.name}'
        dataset_list = ['hapt','oppo','wisdm','dg','uschad','pamap2','rw','skoda','dsads','motion','ucihar','mhealth']
        for dataset_name in dataset_list:
            print(f'==>generate {dataset_name} datset-----------------------')
            data_args , dataset = GetHarDataset(dataset_name,data_args)
            CheckDataloader(data_args , dataset)
    else:
        assert data_args.name , 'please give a dataset name'
        print(f'==>generate {data_args.name} datset-----------------------')
        data_args , dataset = GetHarDataset(data_args.name,data_args)
        CheckDataloader(data_args , dataset)


if __name__ == '__main__':
    np.random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,default='motion')
    args = parser.parse_args()
    test(args.dataset)