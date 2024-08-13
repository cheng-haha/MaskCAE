from pprint import pformat
from typing import List

import sys
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

import encoder
from decoder import LightDecoder
import torch.nn.functional as F

class MaskCAE(nn.Module):
    def __init__(
            self, sparse_encoder: encoder.SparseEncoder, dense_decoder: LightDecoder,
            mask_ratio=0.6, densify_norm='bn', sbn=False, mode ='', sup_window_length = False,
    ):
        super().__init__()
        self.dataset_name               = sparse_encoder.dataset_name
        input_size, downsample_raito    = sparse_encoder.input_size, sparse_encoder.downsample_raito
        self.downsample_raito           = downsample_raito
        self.Tmap_size                  = input_size[0] // downsample_raito
        self.Smap_size                  = input_size[1]
        self.mask_ratio                 = mask_ratio
        # len_keep为非mask的区域
        self.len_keep                   = round(self.Tmap_size * self.Smap_size * (1 - mask_ratio))
        
        self.sparse_encoder             = sparse_encoder
        self.dense_decoder              = dense_decoder
        
        self.sbn                        = sbn
        self.mode                       = mode
        self.sup_window_length          = sup_window_length
        self.hierarchy                  = len(sparse_encoder.enc_feat_map_chs)
        self.densify_norm_str           = densify_norm.lower()
        self.densify_norms              = nn.ModuleList()
        self.densify_projs              = nn.ModuleList()
        self.mask_tokens                = nn.ParameterList()
        
        # build the `densify` layers
        e_widths, d_width = self.sparse_encoder.enc_feat_map_chs, self.dense_decoder.width
        e_widths: List[int]
        for i in range(self.hierarchy): # from the smallest feat map to the largest; i=0: the last feat map; i=1: the second last feat map ...
            # dims.pop()表示移除列表中的最后一个元素，并且返回该元素的值
            e_width = e_widths.pop()
            # create mask token
            p = nn.Parameter(torch.zeros(1, e_width, 1, 1))
            trunc_normal_(p, mean=0, std=.02, a=-.02, b=.02)
            self.mask_tokens.append(p)
            
            # create densify norm
            if self.densify_norm_str == 'bn':
                densify_norm = (encoder.SparseSyncBatchNorm2d if self.sbn else encoder.SparseBatchNorm2d)(e_width)
            else:
                densify_norm = nn.Identity()
            self.densify_norms.append(densify_norm)
            
            # create densify proj
            if (i == 0 and e_width == d_width) :
                densify_proj = nn.Identity()    # todo: NOTE THAT CONVNEXT-S WOULD USE THIS, because it has a width of 768 that equals to the decoder's width 768
                print(f'[MaskCAE.__init__, densify {i+1}/{self.hierarchy}]: use nn.Identity() as densify_proj')
            else:
                kernel_size = 1 if i <= 0 else 5
                if self.mode == 'DwConv_InfoProj':
                    densify_proj = nn.Sequential(
                        nn.Conv2d(e_width, e_width, kernel_size=(kernel_size,1), stride=1, padding=(kernel_size // 2,0) , groups=e_width , bias=True),
                        encoder.SparseBatchNorm2d(e_width),
                        nn.ReLU6(True),
                        nn.Conv2d(e_width, d_width, kernel_size=1, stride=1 , bias=True),
                        encoder.SparseBatchNorm2d(d_width),
                        nn.ReLU(True)
                    )
                elif self.mode == 'BN_InfoProj':
                    densify_proj = nn.Sequential(
                        nn.Conv2d(e_width, d_width, kernel_size=(kernel_size,1), stride=1, padding=(kernel_size // 2,0) ,bias=True),
                        encoder.SparseBatchNorm2d(d_width)
                    )
                elif self.mode == 'BN_Relu_InfoProj':
                    densify_proj = nn.Sequential(
                        nn.Conv2d(e_width, d_width, kernel_size=(kernel_size,1), stride=1, padding=(kernel_size // 2,0) ,bias=True),
                        encoder.SparseBatchNorm2d(d_width),
                        nn.ReLU(True)
                    )
                else:
                    densify_proj = nn.Conv2d(e_width, d_width, kernel_size=(kernel_size,1), stride=1, padding=(kernel_size // 2,0) ,bias=True)
                    
                print(f'[MaskCAE.__init__, densify {i+1}/{self.hierarchy}]: densify_proj(ksz={kernel_size}, #para={sum(x.numel() for x in densify_proj.parameters()) / 1e6:.2f}M)')
            self.densify_projs.append(densify_proj)
            
            # todo: the decoder's width follows a simple halfing rule; you can change it to any other rule
            d_width //= 2
        
        print(f'[MaskCAE.__init__] dims of mask_tokens={tuple(p.numel() for p in self.mask_tokens)}')
        
        self.register_buffer('norm_black', torch.zeros(1, 1, *input_size))
        self.vis_active = self.vis_active_ex = self.vis_inp = self.vis_inp_mask = ...
    
    def mask(self, B: int, device, generator=None):
        # 这里注意self.fmap_size已经通过encoder计算得到，这代表的是每个patch的长度，也是最后输出特征图的尺寸
        f: int  = self.Tmap_size
        s: int  = self.Smap_size
        idx     = torch.rand(B, f * s, generator=generator).argsort(dim=1)
        # len_keep为非mask的区域
        idx     = idx[:, :self.len_keep].to(device)  # (B, len_keep)
        return torch.zeros(B, f * s, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, f, s)
    
    def forward(self, inp_bchw: torch.Tensor, active_b1fs=None, vis=False):
        if not self.sup_window_length:
            inp_bchw   = F.pad(inp_bchw, pad=(0,0,1,1), mode='replicate')
        # step1. Mask
        if active_b1fs is None:     # rand mask 
            # 这段函数返回的就是mask的网格，大小与patch一致
            active_b1fs: torch.BoolTensor = self.mask(inp_bchw.shape[0], inp_bchw.device)  # (B, 1, f, s)
        encoder._cur_active = active_b1fs    # (B, 1, f, s)
        # 对mask进行扩张到original input shape，比率还是相同
        # downsample_raito为累乘的下采样次数，例如resnet下采样四次，downsample_raito就为32，通过downsample_raito,我们可能将最后的特征图大小恢复到原始尺寸
        active_b1hw = active_b1fs.repeat_interleave(self.downsample_raito, 2) # (B, 1, H, W)
        masked_bchw = inp_bchw * active_b1hw
        
        # step2. Encode: get hierarchical encoded sparse features (a list containing 4 feature maps at 4 scales)
        # [S1,S2,S3,S4]
        fea_bcffs: List[torch.Tensor] = self.sparse_encoder(masked_bchw)
        # [S4,S3,S2,S1]
        fea_bcffs.reverse()  # after reversion: from the smallest feature map to the largest
        if self.mode == 'WithoutInfoProj':
            fea_bcffs = [fea_bcffs[0]]
        # step3. Densify: get hierarchical dense features for decoding
        cur_active  = active_b1fs     # (B, 1, f, s)
        to_dec      = []
        for i, bcff in enumerate(fea_bcffs):  # from the smallest feature map to the largest
            if bcff is not None:
                # norm layer
                bcff                = self.densify_norms[i](bcff)
                # if self.mode != 'plain_conv':
                # 1, c, 1, 1 --> b, c, f * 2 ** (4-i), f * 2 ** (4-i)
                mask_tokens         = self.mask_tokens[i].expand_as(bcff)
                # b, 1, f * 2 ** (4-i), f * 2 ** (4-i) --> b, c, f * 2 ** (4-i), f * 2 ** (4-i), 非mask的取bcff里的原值，mask的取mask_tokens的填充值
                bcff                = torch.where(cur_active.expand_as(bcff), bcff, mask_tokens)   # fill in empty (non-active) positions with [mask] tokens
                # learning mask tokens
                bcff: torch.Tensor  = self.densify_projs[i](bcff)
            # bcff为encoder里的层级特征图，值得注意的是只有底层为b1ff大小，之后每一层级f乘以2，最后的大小为原图(B, 1, H, s)
            # 如果stag下标为1,2,3,4;那么to_dec里就排序就是4,3,2,1
            to_dec.append(bcff)
            # dilate the mask map, from (B, 1, f, s) to (B, 1, H, s)
            cur_active              = cur_active.repeat_interleave(2, dim=2)
        # step4. Decode and reconstruct
        rec_bchw        =   self.dense_decoder(to_dec)
        if self.mode    != 'WithoutNormLoss' : # masked only and you can set mask radio to 0.0 for the ablation experiment without masking. 
            inp, rec    =   self.patchify( inp_bchw ), self.patchify( rec_bchw )   #  inp and rec: (B, L = f*f, N = C*downsample_raito**2)
            mean        =   inp.mean(  dim=-1, keepdim=True )
            var         = ( inp.var( dim=-1, keepdim=True ) + 1e-6) ** .5
            inp         = ( inp - mean ) / var
            l2_loss     = ( (rec - inp) ** 2 ).mean( dim=2, keepdim=False )  #  (B, L, C) == mean   ==> (B, L)
            # 计算掩码 non_active 为 mask 的点，将非mask点全部从计算图中剥离，也就是说只计算mask的损失
            non_active  = active_b1fs.logical_not().int().view(active_b1fs.shape[0], -1)  # (B, 1, f, s) => (B, L)
            recon_loss  = l2_loss.mul_( non_active ).sum() / ( non_active.sum() + 1e-8 )  # loss only on masked (non-active) patches
        else: # all loss
            inp, rec    =   self.patchify( inp_bchw ), self.patchify( rec_bchw )   #  inp and rec: (B, L = f*f, N = C*downsample_raito**2)
            l2_loss     = ( (rec - inp) ** 2 ).mean( dim=2, keepdim=False )  #  (B, L, C) == mean   ==> (B, L)
            # 计算掩码 non_active 为 mask 的点，将非mask点全部从计算图中剥离，也就是说只计算mask的损失
            non_active  = active_b1fs.logical_not().int().view(active_b1fs.shape[0], -1)  # (B, 1, f, s) => (B, L)
            recon_loss  = l2_loss.mul_( non_active ).sum() / ( non_active.sum() + 1e-8 )  # loss only on masked (non-active) patches

        if vis:
            masked_bchw = inp_bchw * active_b1hw
            rec_bchw    = self.unpatchify( rec * var + mean )
            rec_or_inp  = torch.where( active_b1hw, inp_bchw, rec_bchw )
            return [ inp_bchw, masked_bchw, rec_or_inp , active_b1hw ]
        else:
            return recon_loss
    
    def patchify( self, bchw ):
        p       = self.downsample_raito
        h       = self.Tmap_size
        w       = self.Smap_size 
        B, C    = bchw.shape[:2]
        bchw    = bchw.reshape( shape=(B, C, h, p, w, 1)    )
        bchw    = torch.einsum('bchpwq->bhwpqc', bchw)
        bln     = bchw.reshape( shape=(B, h * w, C * p )    )  # (B, f*f, 3*downsample_raito**2)
        return bln
    
    def unpatchify( self, bln ):
        p       = self.downsample_raito
        h       = self.Tmap_size
        w       = self.Smap_size 
        B, C    = bln.shape[0], bln.shape[-1] // p 
        bln     = bln.reshape(  shape=(B, h, w, p, 1, C)    )
        bln     = torch.einsum('bhwpqc->bchpwq', bln)
        bchw    = bln.reshape(  shape=(B, C, h * p, w )     )
        return bchw
    
    def __repr__(self):
        return (
            f'\n'
            f'[MaskCAE.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[MaskCAE.structure]: {super(MaskCAE, self).__repr__().replace(MaskCAE.__name__, "")}'
        )
    
    def get_config(self):
        return {
            # self
            'mask_ratio': self.mask_ratio,
            'densify_norm_str': self.densify_norm_str,
            'sbn': self.sbn, 'hierarchy': self.hierarchy,
            
            # enc
            'sparse_encoder.input_size': self.sparse_encoder.input_size,
            # dec
            'dense_decoder.width': self.dense_decoder.width,
        }
    
    def state_dict( self, destination = None, prefix = '', keep_vars = False, with_config = False ):
        state               = super(MaskCAE, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if with_config:
            state['config'] = self.get_config()
        return state
    
    def load_state_dict( self, state_dict, strict=True ):
        config: dict        = state_dict.pop('config', None)
        incompatible_keys   = super(MaskCAE, self).load_state_dict(state_dict, strict=strict)
        if config is not None:
            for k, v in self.get_config().items():
                ckpt_v      = config.get(k, None)
                if ckpt_v   != v:
                    err     = f'[SparseMIM.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={ckpt_v})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err, file=sys.stderr)
        return incompatible_keys
    
    def denorm_for_vis( self, normalized_im ):
        normalized_im       = (normalized_im * self.imn_s).add_(self.imn_m)
        return torch.clamp(normalized_im, 0, 1)
