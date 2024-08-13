'''
Description: 
Date: 2024-08-13 13:08:14
LastEditTime: 2024-08-13 16:09:47
FilePath: /chengdongzhou/action/MaskCAE/pretrain/simpletest.py
'''

import turtle
import torch
import encoder
from decoder import LightDecoder
from models import build_sparse_encoder
from MaskCAE import MaskCAE
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
args = dotdict()
args.model = 'Conv4Net'
args.input_size: turtle = (33,12)
args.sbn = False
args.dp: float = 0.0
args.mask = 0.6
args.densify_norm: str = ''
enc: encoder.SparseEncoder  = build_sparse_encoder(args.model, input_size=args.input_size, sbn=args.sbn, drop_path_rate=args.dp, verbose=False)
dec                         = LightDecoder(enc.downsample_raito, sbn=args.sbn)
model                       = MaskCAE(
                                    sparse_encoder          =   enc, 
                                    dense_decoder           =   dec, 
                                    mask_ratio              =   args.mask,
                                    densify_norm            =   args.densify_norm, 
                                    sbn                     =   args.sbn,
                            ).to(args.device)
inp = torch.randn(2,1,128,12)
loss = model(inp, active_b1fs=None, vis=False)
print(loss)