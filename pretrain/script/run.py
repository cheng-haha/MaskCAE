'''
Description: 
Date: 2023-03-22 11:34:07
LastEditTime: 2024-08-13 19:33:58
FilePath: /chengdongzhou/action/MaskCAE/pretrain/script/run.py
'''
import os
import argparse
parser      = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default=None)
parser.add_argument('--device',type=int,default=1)
args        = parser.parse_args()
mode        = 'Base'
mask_radio  = [0.3]
ep          = 1000    
for radio in mask_radio:
    os.system(
        f'python pretrain/main.py --dataset_name {args.dataset} --mask {radio} --ep {ep} --device {args.device}'
            )
