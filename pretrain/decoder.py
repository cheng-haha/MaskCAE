'''
Description: 
Date: 2023-06-04 12:00:53
LastEditTime: 2024-08-13 13:17:13
FilePath: /chengdongzhou/action/MaskCAE/pretrain/decoder.py
'''
import math
from typing import List

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from utils.misc import is_pow2n


class UNetBlock(nn.Module):
    def __init__(self, cin, cout, bn2d,mode):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(cin, cin, kernel_size=(4,1), stride=(2,1), padding=(1,0), bias=True)
        if mode == 'PlainDecoder':
            self.conv = nn.Sequential(
                nn.Conv2d(cin, cin, kernel_size=(5,1), stride=(1,1), padding=(2,0), bias=False), bn2d(cin), 
                nn.ReLU6(inplace=True),
                nn.Conv2d(cin, cout, kernel_size=(5,1), stride=(1,1), padding=(2,0), bias=False), bn2d(cout),
            )
        elif mode == '9x9IFD':
            self.conv = nn.Sequential(
                nn.Conv2d(cin, cin, kernel_size=9, stride=1, padding=4, bias=False), bn2d(cin), 
                nn.ReLU6(inplace=True),
                nn.Conv2d(cin, cout, kernel_size=9, stride=1, padding=4, bias=False), bn2d(cout),
            )
        elif mode == '7x7IFD':
            self.conv = nn.Sequential(
                nn.Conv2d(cin, cin, kernel_size=7, stride=1, padding=3, bias=False), bn2d(cin), 
                nn.ReLU6(inplace=True),
                nn.Conv2d(cin, cout, kernel_size=7, stride=1, padding=3, bias=False), bn2d(cout),
            )
        elif mode == '3x3IFD':
            self.conv = nn.Sequential(
                nn.Conv2d(cin, cin, kernel_size=3,  stride=1, padding=1, bias=False), bn2d(cin), 
                nn.ReLU6(inplace=True),
                nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False), bn2d(cout),
            )
        else:         
            self.conv = nn.Sequential(
                nn.Conv2d(cin, cin, kernel_size=5, stride=1, padding=2, bias=False), bn2d(cin), 
                nn.ReLU6(inplace=True),
                nn.Conv2d(cin, cout, kernel_size=5, stride=1, padding=2, bias=False), bn2d(cout),
            )    
    def forward(self, x):
        x = self.up_sample(x)
        return self.conv(x)


class LightDecoder(nn.Module):
    def __init__(self, up_sample_ratio, width=256, sbn=False,mode=''): 
        super().__init__()
        self.width  = width
        self.mode   = mode
        assert is_pow2n(up_sample_ratio)
        n           = round(math.log2(up_sample_ratio))
        channels    = [self.width // 2 ** i for i in range(n + 1)] 
        bn2d        = nn.SyncBatchNorm if sbn else nn.BatchNorm2d
        self.dec    = nn.ModuleList([UNetBlock(cin, cout, bn2d, mode) for (cin, cout) in zip(channels[:-1], channels[1:])])
        self.proj   = nn.Conv2d(channels[-1], 1, kernel_size=1, stride=1, bias=True)
        
        self.initialize()
    
    def forward(self, to_dec: List[torch.Tensor]):
        x = 0
        for i, d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.dec[i](x)
        return self.proj(x)
    
    def extra_repr(self) -> str:
        return f'width={self.width}'
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
