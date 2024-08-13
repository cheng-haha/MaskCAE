'''
Description: 
Date: 2023-05-22 16:58:22
LastEditTime: 2024-08-13 19:31:13
FilePath: /chengdongzhou/action/MaskCAE/models/ConvNet.py
'''
import torch.nn as nn
import torch.nn.functional as F
import os 
import sys 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from common import  channel_list,conv_list,maxp_list,first_maxp_list
from utils.metric import get_data_size,GetFlopsAndParams,GetFeatureMapSize


    
class Conv4Net(nn.Module):
    def __init__(self, data_name='ucihar' ,use_adativeavg=False, sub_number = 4  ):
        super(Conv4Net, self).__init__()
        self.dataset = data_name
        self.channel= channel = channel_list[data_name]
        conv_params = conv_list[data_name]
        max_params  = maxp_list[data_name]
        first_max   = first_maxp_list[data_name]

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
    def forward(self, x: torch.Tensor):
        B,_,_,_ = x.size()
        x   = F.pad(x, pad=(0,0,1,1), mode='replicate')
        x   = self.layer1(x)
        x   = self.layer2(x)     
        x   = self.layer3(x)
        x   = self.layer4(x)
        x   = self.adaptiveAvg(x)
        x   = x.view(B,-1)
        x   = self.classifier( x )
        res = {}
        res['output'] = x
        return res


class Conv4NetSup(nn.Module):
    def __init__(self, data_name='ucihar' ,use_adativeavg=False, sub_number = 4  ):
        super(Conv4NetSup, self).__init__()
        self.dataset = data_name
        self.channel= channel = channel_list[data_name]
        conv_params = conv_list[data_name]
        max_params  = maxp_list[data_name]
        first_max_params = first_maxp_list[data_name]

        self.layer1  = self.maker_layers( 1         , channel[0], conv_params=conv_params, pooling_params=first_max_params )
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
            nn.MaxPool2d( *(pooling_params) ) if pooling_params else nn.Identity(),
        )
    
    def forward(self, x: torch.Tensor):
        B,_,_,_ = x.size()
        if self.dataset in ['uschad']:
            x   = F.pad(x, pad=(0,0,1,1), mode='replicate')
        x   = self.layer1(x)
        x   = self.layer2(x)     
        x   = self.layer3(x)
        x   = self.layer4(x)
        x   = self.adaptiveAvg(x)
        x   = x.view(B,-1)
        x   = self.classifier( x )
        res = {}
        res['output'] = x
        return res

        
if __name__ == "__main__":
    import torch
    f_size =  {
                'ucihar':   (30    ,    9   ),
                'uschad':   (30     ,   6   ),
                'motion':   (30     ,   12  )
                        }
    dataset = 'motion'
    x = torch.randn(64,1,*f_size[dataset])
    model = Conv4Net(dataset)
    # print(model,model.layer2[1])
    GetFlopsAndParams(dataset,model)
    print(model(x)['output'].shape)
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))