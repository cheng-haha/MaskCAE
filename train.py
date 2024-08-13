'''
Description: 
Date: 2023-03-18 14:25:30
LastEditTime: 2024-08-13 19:28:11
FilePath: /chengdongzhou/action/MaskCAE/train.py
'''
from trainers import *

def MetaTrain(args):
    '''
    This is also an index.
    '''
    if args.mode in [ 'maskcae','maskcae_diffmodel'] :
        return maskcae_train
    else:
        raise NotImplementedError




