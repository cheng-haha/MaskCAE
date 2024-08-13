'''
Description: 
Date: 2024-08-13 13:08:13
LastEditTime: 2024-08-13 13:14:10
FilePath: /chengdongzhou/action/MaskCAE/models/MaskedConvNet.py
'''
import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from typing import List
from common import channel_list,conv_list,maxp_list,first_maxp_list
from utils.metric import GetFeatureMapSize
from configs.base_configs import args

class ConvNet(nn.Module):
    def __init__(self, data_name='ucihar' ,use_adativeavg=False, sub_number = 4  ):
        super(ConvNet, self).__init__()
        self.channel= channel = channel_list[data_name]
        conv_params = conv_list[data_name]
        max_params  = maxp_list[data_name]
        first_max   = first_maxp_list[data_name]

        self.layer1  = self.maker_layers( 1         , channel[0], conv_params=conv_params, pooling_params=first_max )
        self.layer2  = self.maker_layers( channel[0], channel[1], conv_params=conv_params, pooling_params=max_params )
        self.layer3  = self.maker_layers( channel[1], channel[2], conv_params=conv_params, pooling_params=max_params )
        self.layer4  = self.maker_layers( channel[2], channel[3], conv_params=conv_params, pooling_params=max_params )
        h,w = GetFeatureMapSize(data_name,sub_number)
        self.adaptiveAvg = nn.AdaptiveAvgPool2d( ( 1, w ) ) if use_adativeavg else nn.Identity()
        self.classifier  = nn.Linear(  channel[2]*w if use_adativeavg else channel[2]* h * w , channel[-1]   )

    def maker_layers(self,inp,oup,conv_params=None,pooling_params=None):
        assert isinstance(conv_params,list),print('the format of kernel params is error')
        assert isinstance(pooling_params,list) or pooling_params == None ,print('the format of pooling params is error')
        return nn.Sequential(
            nn.Conv2d( inp, oup, *(conv_params) ),
            nn.BatchNorm2d( oup ),
            nn.ReLU( True ),
            nn.MaxPool2d( *(pooling_params) ) if pooling_params else nn.Identity()
        )
    
    def get_downsample_ratio(self) -> int:
        return 32
    
    def get_feature_map_channels(self) -> List[int]:
        return self.channel[:-1]
    
    def forward(self, x: torch.Tensor, hierarchical=False):
        B,_,_,_ = x.size()
        if hierarchical:
            ls  = []
            x   = self.layer1(x); ls.append(x)
            x   = self.layer2(x); ls.append(x)        
            x   = self.layer3(x); ls.append(x)
            x   = self.layer4(x); ls.append(x)
            return ls
        else:
            x   = self.adaptiveAvg(x)
            x   = x.view(B,-1)
            x   = self.classifier( x )
            return x


@torch.no_grad()
def convnet_test():
    cnn = ConvNet()
    print('get_downsample_ratio:', cnn.get_downsample_ratio())
    print('get_feature_map_channels:', cnn.get_feature_map_channels())
    
    downsample_ratio = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()
    
    # check the forward function
    B, C, H, W = 4, 1, 128, 9
    inp = torch.rand(B, C, H, W)
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print([tuple(t.shape) for t in feats])
    
    # check the downsample ratio
    feats = cnn(inp, hierarchical=True)
    assert feats[-1].shape[-2] == H // downsample_ratio
    assert feats[-1].shape[-1] == W // 1
    
    # check the channel number
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 4
        assert feat.shape[1] == ch


if __name__ == '__main__':
    convnet_test()
