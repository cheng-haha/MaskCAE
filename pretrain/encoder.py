import torch
import torch.nn as nn

_cur_active: torch.Tensor = None            # B1fs
def _get_active_ex_or_ii(H, returning_active_ex=True):
    downsample_raito = H // _cur_active.shape[2]
    # B1ff -> B,1,f*downsample_raito,f*downsample_raito，对mask进行重复扩张，也就是说每个层级的特征图所乘的mask网格是不一致的，_cur_active为布尔张量，
    # 但仍然需要注意的是每个层级的特征图的mask radio永远都是一致的
    active_ex = _cur_active.repeat_interleave(downsample_raito, 2)
    return active_ex if returning_active_ex else active_ex.squeeze(1).nonzero(as_tuple=True)  # ii: bi, hi, wi


def sp_conv_forward(self, x: torch.Tensor):
    x = super(type(self), self).forward(x)
    x *= _get_active_ex_or_ii(H=x.shape[2], returning_active_ex=True)    # (BCHW) *= (B1HW), mask the output of conv
    return x


def sp_bn_forward(self, x: torch.Tensor):
    ii = _get_active_ex_or_ii(H=x.shape[2], returning_active_ex=False)
    
    bhwc        = x.permute(0, 2, 3, 1)
    nc          = bhwc[ii]                                  # select the features on non-masked positions to form a flatten feature `nc`
    nc          = super(type(self), self).forward(nc)       # use BN1d to normalize this flatten feature `nc`
    
    bchw        = torch.zeros_like(bhwc)
    bchw[ii]    = nc
    bchw        = bchw.permute(0, 3, 1, 2)
    return bchw


class SparseConv2d(nn.Conv2d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseMaxPooling(nn.MaxPool2d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseAvgPooling(nn.AvgPool2d):
    forward = sp_conv_forward   # hack: override the forward function; see `sp_conv_forward` above for more details


class SparseBatchNorm2d(nn.BatchNorm1d):
    forward = sp_bn_forward     # hack: override the forward function; see `sp_bn_forward` above for more details


class SparseSyncBatchNorm2d(nn.SyncBatchNorm):
    forward = sp_bn_forward     # hack: override the forward function; see `sp_bn_forward` above for more details



class SparseEncoder(nn.Module):
    def __init__(self, cnn, input_size, sbn=False, verbose=False, mode='' ):
        super(SparseEncoder, self).__init__()
        if mode != 'plain_conv':
            print('==>sparse convolution')
            self.sp_cnn = SparseEncoder.dense_model_to_sparse(m=cnn, verbose=verbose, sbn=sbn)
        else:# plain convolution, do not
            self.sp_cnn = cnn
        self.dataset_name= cnn.dataset_name
        self.input_size, self.downsample_raito, self.enc_feat_map_chs = input_size, cnn.get_downsample_ratio(), cnn.get_feature_map_channels()
    
    @staticmethod
    def dense_model_to_sparse(m: nn.Module, verbose=False, sbn=False):
        oup = m
        if isinstance(m, nn.Conv2d):
            m: nn.Conv2d
            bias    = m.bias is not None
            oup     = SparseConv2d(
                m.in_channels, m.out_channels,
                kernel_size=m.kernel_size, stride=m.stride, padding=m.padding,
                dilation=m.dilation, groups=m.groups, bias=bias, padding_mode=m.padding_mode,
            )
            oup.weight.data.copy_(m.weight.data)
            if bias:
                oup.bias.data.copy_(m.bias.data)
        elif isinstance(m, nn.MaxPool2d):
            m: nn.MaxPool2d
            oup = SparseMaxPooling(m.kernel_size, stride=m.stride, padding=m.padding, dilation=m.dilation, return_indices=m.return_indices, ceil_mode=m.ceil_mode)
        elif isinstance(m, nn.AvgPool2d):
            m: nn.AvgPool2d
            oup = SparseAvgPooling(m.kernel_size, m.stride, m.padding, ceil_mode=m.ceil_mode, count_include_pad=m.count_include_pad, divisor_override=m.divisor_override)
        elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m: nn.BatchNorm2d
            oup = (SparseSyncBatchNorm2d if sbn else SparseBatchNorm2d)(m.weight.shape[0], eps=m.eps, momentum=m.momentum, affine=m.affine, track_running_stats=m.track_running_stats)
            oup.weight.data.copy_(m.weight.data)
            oup.bias.data.copy_(m.bias.data)
            oup.running_mean.data.copy_(m.running_mean.data)
            oup.running_var.data.copy_(m.running_var.data)
            oup.num_batches_tracked.data.copy_(m.num_batches_tracked.data)
            if hasattr(m, "qconfig"):
                oup.qconfig = m.qconfig
        elif isinstance(m, (nn.Conv1d,)):
            raise NotImplementedError
        
        for name, child in m.named_children():
            oup.add_module(name, SparseEncoder.dense_model_to_sparse(child, verbose=verbose, sbn=sbn))
        del m
        return oup
    
    def forward(self, x):
        return self.sp_cnn(x, hierarchical=True)
