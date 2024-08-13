from thop.profile import profile
from thop import clever_format 
import torch
import numpy as np
import time
from common import maxp_list,channel_list,conv_list,first_maxp_list

def Reture_F_SIZE(dataset , is_sup=False):
    F_SIZE =  {
                'ucihar':   (30    ,   9    )   ,
                'uschad':   (30    ,   6    )   ,
                'motion':   (30    ,   12   )  
                        }
    SUP_F_SIZE = {
                'ucihar':   (128    ,   9    ) ,
                'uschad':   (30     ,   6    ) ,
                'motion':   (128    ,   12    )
                        }
    if not is_sup:
        return F_SIZE[dataset]
    else:
        return SUP_F_SIZE[dataset]

class Timer:
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

def get_data_size(data_name, is_sup = False ):
    Model_Seen_SSL_F_Size = {
        'ucihar': (     1, 1, 32, 9     ) ,
        'motion': (     1, 1, 32, 12    ) ,
        'uschad': (     1, 1, 32, 6     ) ,
    }
    Model_Seen_Sup_F_Size = {
        'ucihar': (     1, 1, 128, 9     ) ,
        'motion': (     1, 1, 128, 12    ) ,
        'uschad': (     1, 1, 32,  6     ) ,
    }
    if is_sup:
        size_dict = Model_Seen_Sup_F_Size
    else:
        size_dict = Model_Seen_SSL_F_Size
    if data_name in size_dict:
        pass
    else:
        raise Exception( 'please input correct data name')
    return size_dict[data_name]

def get_classes(data_name):
    if data_name == 'ucihar':
        classes = 6
    elif data_name == 'motion':
        classes = 6
    elif data_name == 'uschad':
        classes = 12
    else:
        raise Exception( 'please input correct data name')
    return classes


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

def GetFeatureMapSizeByConv(data_name,idex_layer):
    size = get_data_size(data_name)[2:]
    conv_size = conv_list[data_name]
    h,w = size
    if idex_layer > 0:
        for i in range(idex_layer):
            h   =  h - conv_size[0][0] + 1
            w   =  w - conv_size[0][1] + 1
        return ( h , w )
    else:
        raise  ValueError(f'check your idex_layer')

def GetFeatureMapSizeBySelfMax(data_name , max_size, idex_layer):
    size = get_data_size(data_name)[2:]
    h,w  = size
    if idex_layer > 0:
        for i in range(idex_layer):
            h //= max_size[0]
            w //= max_size[1]
        return ( h , w )
    else:
        raise  ValueError(f'check your idex_layer')

def GetFlopsAndParams(dataset,net,logger = None , is_sup = False ):
    X       = torch.randn(1,1,*Reture_F_SIZE(dataset,is_sup))   
    total_ops, total_params= profile(net, (X,), verbose=False)
    flops, params = clever_format([total_ops*2, total_params], "%.3f")
    if logger:
        logger.info(f'==>FLOPs of model that runs the {dataset} is {flops}')
        logger.info(f'==>Params of model that runs the {dataset} is {params}'  )
    else:
        print(f'==>FLOPs of model that runs the {dataset} is {flops}')
        print(f'==>Params of model that runs the {dataset} is {params}'  )
    return flops,params

def DeConvOrthDist(kernel, stride = 2, padding = 1):
    [o_c, i_c, w, h] = kernel.shape
    output = torch.conv2d(kernel, kernel, stride=stride, padding=padding)
    target = torch.zeros((o_c, o_c, output.shape[-2], output.shape[-1])).cuda()
    ct = int(np.floor(output.shape[-1]/2))
    target[:,:,ct,ct] = torch.eye(o_c).cuda()
    # print(target.shape)
    return torch.norm( output - target )

def GetInferredSpeed(dataset,model,times, is_sup = False ):
    x       = torch.randn(1,1,*Reture_F_SIZE(dataset,is_sup))   
    each_times = []
    model.eval()
    for i in range(times):
        start_time = time.time()
        model(x)
        stop_time = time.time()
        each_times.append((stop_time-start_time)*1000)
    return np.mean(each_times)