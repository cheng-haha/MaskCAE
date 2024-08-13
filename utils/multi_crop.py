'''
Description: 
Date: 2023-05-23 16:02:49
LastEditTime: 2023-05-30 13:34:05
FilePath: /chengdongzhou/action/tRexHAR/utils/multi_crop.py
'''
##################################################
# Multi-crop related code re-used from DINO
# https://github.com/facebookresearch/dino
##################################################

import random

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from configs.base_configs import args 
from utils.augmentations import gen_aug


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, encoder, head):
        super().__init__()
        # disable layers dedicated to ImageNet labels classification
        self.encoder = encoder
        self.head = head

    def forward(self, x ):
        # convert to list
        if self.training:
            if not isinstance(x, list):
                x = [x]
            idx_crops = torch.cumsum(
                torch.unique_consecutive(
                    torch.tensor([inp.shape[2] for inp in x]),
                    return_counts=True,
                )[1],
                0,
            )
            
            start_idx, output_list,_cls_out = 0, [torch.empty(0).to(x[0].device) for i in range(3) ] , torch.empty(0).to(x[0].device)
            for end_idx in idx_crops:
                if start_idx < args.mc_global_number:
                    _out = self.encoder(torch.cat(x[start_idx:end_idx]))
                else:
                    _out = self.encoder( gen_aug( torch.cat(x[start_idx:end_idx]).cpu().squeeze(1), ssh_type=args.data_aug ).unsqueeze(1).cuda().float() )
                # accumulate outputs
                _rep_list, _cls  = _out['rep_list'] , _out['cls']
                _cls_out         = torch.cat((_cls_out,_cls))
                output_list      = [ torch.cat((output_list[x], rep)) for x , rep in enumerate( _rep_list ) ]
                start_idx        = end_idx
            # Run the head forward on the concatenated features
            return [self.head(output) for output in output_list ] , _cls_out
        else:
            return None , self.encoder(x)['cls']

        


if __name__ == '__main__':
    x = torch.randn(1,3,224,224)