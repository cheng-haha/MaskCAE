'''
Description: 
Date: 2023-03-22 11:34:07
LastEditTime: 2024-08-13 19:11:15
FilePath: /chengdongzhou/action/MaskCAE/scripts/MaskCAE/supevisedlearning.py
'''
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,default=None)
parser.add_argument('--config',type=str,default=None)
parser.add_argument('--device',type=int,default=2)
parser.add_argument('--times',type=int,default=5)
parser.add_argument('--dataset_pre',type=str,default='Sup')

args = parser.parse_args()
mode =  'maskcae_diffmodel'
datasets = ['motion','ucihar','uschad']
models      = ['Transformer_HAR','DeepConvLstmAttn','SA_HAR']
# models      = ['BaseCNNSup','Conv4NetSup']
if args.dataset :
    assert args.dataset in datasets , 'not support this dataset'
    datasets   = [ dataset.strip() for dataset in args.dataset.split(',')]
    
for dataset in datasets:
    for model in models:
        if model == 'Conv4NetSup':
            os.system(
                f'python main.py --dataset {dataset} --mode {mode} --config {args.config}  --dataset_pre {args.dataset_pre} --model {model} --times {args.times} --mask {0.3}'
                    )

        else:
            os.system(
                f'python main.py --dataset {dataset} --mode {mode} --config {args.config}  --dataset_pre {args.dataset_pre} --model {model} --times {args.times}'
                    )
