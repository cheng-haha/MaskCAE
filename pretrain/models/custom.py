# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from typing import List
from timm.models.registry import register_model
from common import channel_list,conv_list,maxp_list,first_maxp_list
from utils.metric import get_data_size
import torch.nn.functional as F

def GetFeatureMapSize(data_name,idex_layer,padding=False, larger_max=False , is_sup = False ):
    size = get_data_size(data_name, is_sup)[2:]
    maxpooling_size = maxp_list[data_name]
    first_maxp_size = first_maxp_list[data_name]
    h,w = size
    if idex_layer > 0:
        for i in range(idex_layer):
            if padding:
                if larger_max and i == 0:
                    h = (( h  - first_maxp_size[0][0] + first_maxp_size[2][0] * 2 + first_maxp_size[1][0] ) // first_maxp_size[1][0]) 
                    w = (( w  - first_maxp_size[0][1] + first_maxp_size[2][1] * 2 + first_maxp_size[1][1] ) // first_maxp_size[1][1]) 
                else:
                    h = (( h  - maxpooling_size[0][0] + maxpooling_size[2][0] * 2 + maxpooling_size[1][0] ) // maxpooling_size[1][0]) 
                    w = (( w  - maxpooling_size[0][1] + maxpooling_size[2][1] * 2 + maxpooling_size[1][1] ) // maxpooling_size[1][1]) 
            else:
                h //= maxpooling_size[0][0]
                w //= maxpooling_size[0][1]
        return ( h , w )
    elif idex_layer == 0:
        return ( h , w )
    else:
        raise  ValueError(f'check your idex_layer')

class ConvNet3(nn.Module):
    def __init__(self, data_name='ucihar' ,use_adativeavg=False, sub_number = 4  ):
        super(ConvNet3, self).__init__()
        self.channel= channel = channel_list[data_name]
        conv_params = conv_list[data_name]
        max_params  = maxp_list[data_name]
        first_max   = first_maxp_list[data_name]
        print(f'Dataset is {data_name}')
        self.dataset_name = data_name
        self.layer1  = self.maker_layers( 1         , channel[0], conv_params=conv_params, pooling_params=first_max )
        self.layer2  = self.maker_layers( channel[0], channel[1], conv_params=conv_params, pooling_params=max_params )
        self.layer3  = self.maker_layers( channel[1], channel[2], conv_params=conv_params, pooling_params=max_params )
        h,w = GetFeatureMapSize(data_name,sub_number,padding = True)
        self.adaptiveAvg = nn.AdaptiveAvgPool2d( ( 1, w ) ) if use_adativeavg else nn.Identity()
        self.classifier  = nn.Linear(  channel[3]*w if use_adativeavg else channel[3]* h * w , channel[-1]   )

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
        return 16
    
    def get_feature_map_channels(self) -> List[int]:
        return self.channel[:-2]
    
    def forward(self, x: torch.Tensor, hierarchical=False):
        B,_,_,_ = x.size()
        if hierarchical:
            ls  = []
            x   = self.layer1(x); ls.append(x)
            x   = self.layer2(x); ls.append(x)        
            x   = self.layer3(x); ls.append(x)
            return ls
        else:
            x   = self.adaptiveAvg(x)
            x   = x.view(B,-1)
            x   = self.classifier( x )
            return x

class ConvNet4(nn.Module):
    def __init__(self, data_name='ucihar' ,use_adativeavg=False, sub_number = 4  ):
        super(ConvNet4, self).__init__()
        self.channel= channel = channel_list[data_name]
        conv_params = conv_list[data_name]
        max_params  = maxp_list[data_name]
        first_max   = first_maxp_list[data_name]
        print(f'Dataset is {data_name}')
        self.dataset_name = data_name
        self.layer1  = self.maker_layers( 1         , channel[0], conv_params=conv_params, pooling_params=max_params )
        self.layer2  = self.maker_layers( channel[0], channel[1], conv_params=conv_params, pooling_params=max_params )
        self.layer3  = self.maker_layers( channel[1], channel[2], conv_params=conv_params, pooling_params=max_params )
        self.layer4  = self.maker_layers( channel[2], channel[3], conv_params=conv_params, pooling_params=max_params )
        h,w = GetFeatureMapSize(data_name,sub_number,padding = True)
        self.adaptiveAvg = nn.AdaptiveAvgPool2d( ( 1, w ) ) if use_adativeavg else nn.Identity()
        self.classifier  = nn.Linear(  channel[3]*w if use_adativeavg else channel[3]* h * w , channel[-1]   )

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
        return 16
    
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


class ConvNetSup(nn.Module):
    def __init__(self, data_name='ucihar' ,use_adativeavg=False, sub_number = 4  ):
        super(ConvNetSup, self).__init__()
        self.channel= channel = channel_list[data_name]
        conv_params = conv_list[data_name]
        max_params  = maxp_list[data_name]
        first_max   = first_maxp_list[data_name]
        print(f'Dataset is {data_name}')
        self.dataset_name = data_name
        self.layer1  = self.maker_layers( 1         , channel[0], conv_params=conv_params, pooling_params=first_max )
        self.layer2  = self.maker_layers( channel[0], channel[1], conv_params=conv_params, pooling_params=max_params )
        self.layer3  = self.maker_layers( channel[1], channel[2], conv_params=conv_params, pooling_params=max_params )
        self.layer4  = self.maker_layers( channel[2], channel[3], conv_params=conv_params, pooling_params=max_params )
        h,w = GetFeatureMapSize(data_name,sub_number,padding = True, larger_max= True , is_sup= True )
        self.adaptiveAvg = nn.AdaptiveAvgPool2d( ( 1, w ) ) if use_adativeavg else nn.Identity()
        self.classifier  = nn.Linear(  channel[3]*w if use_adativeavg else channel[3]* h * w , channel[-1]   )

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


class ConvNetwith3x3(nn.Module):

    def __init__(self, data_name='ucihar' ,use_adativeavg=False, sub_number = 4  ):
        super(ConvNetwith3x3, self).__init__()
        self.channel= channel = channel_list[data_name]
        conv_params = conv_list[data_name]
        max_params  = maxp_list[data_name]
        first_max   = first_maxp_list[data_name]
        print(f'Dataset is {data_name}')
        self.dataset_name = data_name
        self.layer1  = self.maker_layers( 1         , channel[0], conv_params=conv_params, pooling_params=first_max )
        self.layer2  = self.maker_layers( channel[0], channel[1], conv_params=conv_params, pooling_params=max_params )
        self.layer3  = self.maker_layers( channel[1], channel[2], conv_params=conv_params, pooling_params=max_params )
        self.layer4  = self.maker_layers( channel[2], channel[3], conv_params=conv_params, pooling_params=max_params )
        h,w = GetFeatureMapSize(data_name,sub_number,padding = True)
        self.adaptiveAvg = nn.AdaptiveAvgPool2d( ( 1, w ) ) if use_adativeavg else nn.Identity()
        self.classifier  = nn.Linear(  channel[3]*w if use_adativeavg else channel[3]* h * w , channel[-1]   )

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
        """
        The forward with `hierarchical=True` would ONLY be used in `SparseEncoder.forward` (see `pretrain/encoder.py`).
        
        :param inp_bchw: input sensor tensor, shape: (batch_size, channel, time length, sensor feat).
        :param hierarchical: return the logits (not hierarchical), or the feature maps (hierarchical).
        :return:
            - hierarchical == False: return the logits of the classification task, shape: (batch_size, num_classes).
            - hierarchical == True: return a list of all feature maps, which should have the same length as the return value of `get_feature_map_channels`.
            E.g., for a Conv4Net, it should return a list [1st_feat_map, 2nd_feat_map, 3rd_feat_map, 4th_feat_map].
                    for an input size of 224, the shapes are [(B, 32, 32, 9), (B, 64, 16, 9), (B, 128, 8, 9), (B, 256, 4, 9)]
        """
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

@register_model
def Conv3Net(   pretrained=False,
                in_chans=1,
                num_classes = None,
                **kwargs):
    return ConvNet3(**kwargs)

@register_model
def Conv4Net(   pretrained=False,
                in_chans=1,
                num_classes = None,
                **kwargs):
    return ConvNet4(**kwargs)

@register_model
def Conv4NetSup(   pretrained=False,
                in_chans=1,
                num_classes = None,
                **kwargs):
    return ConvNetSup(**kwargs)

@torch.no_grad()
def convnet_test():
    from timm.models import create_model
    dataset_name = 'uschad'

    cnn = create_model('Conv4Net',dataset_name)

    print('get_downsample_ratio:', cnn.get_downsample_ratio())
    print('get_feature_map_channels:', cnn.get_feature_map_channels())

    
    downsample_ratio = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()
    
    # check the forward function
    B, C, H, W = get_data_size(dataset_name)
    inp = torch.rand(B, C, 34, 6)
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print([tuple(t.shape) for t in feats])
    
    # check the downsample ratio
    feats = cnn(inp, hierarchical=True)
    
    # check the channel number
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 4
        assert feat.shape[1] == ch


if __name__ == '__main__':
    convnet_test()
