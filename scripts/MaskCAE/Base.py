'''
Description: 
Date: 2023-03-22 11:34:07
LastEditTime: 2024-08-13 17:49:14
FilePath: /chengdongzhou/action/MaskCAE/scripts/MaskCAE/Base.py
'''
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default=None)
parser.add_argument('--config',type=str,default=None)
parser.add_argument('--device',type=int,default=1)
parser.add_argument('--times',type=int,default=5)
parser.add_argument('--dataset_pre',type=str,default='self')

args = parser.parse_args()
mode =  'maskcae'
datasets = ['motion','ucihar','uschad']
if args.dataset :
    assert args.dataset in datasets , 'not support this dataset'
    datasets   = [ dataset.strip() for dataset in args.dataset.split(',')]
    
for dataset in datasets:
    os.system(
        f'python main.py --dataset {dataset} --mode {mode} --config {args.config}  --dataset_pre {args.dataset_pre} --device {args.device} --times {args.times}'
            )
