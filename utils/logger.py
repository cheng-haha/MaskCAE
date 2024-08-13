
import logging
import numpy as np
import os
import pathlib
import pandas as pd
import time
from configs.base_configs import args
import h5py
from utils.setup import GetModel
from utils.metric import GetInferredSpeed,GetFlopsAndParams

import re 

def Statistics(stat, Test_losses, Acc_tests, mF1_tests, wF1_tests , Recall_tests, Precision_tests, sum_time = 0):
    stat['time']            = sum_time
    stat['Test_losses']     = Test_losses
    stat['Acc_tests']       = Acc_tests
    stat['mF1_tests']       = mF1_tests
    stat['wF1_tests']       = wF1_tests
    stat['Recall_tests']    = Recall_tests
    stat['Precision_test']  = Precision_tests
    return stat

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name   = name
        self.fmt    = fmt
        self.reset()

    def reset(self):
        self.val    = 0
        self.avg    = 0
        self.sum    = 0
        self.count  = 0

    def update(self, val, n=1):
        self.val    =   val
        self.sum    +=  val * n
        self.count  +=  n
        self.avg    =   self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


                
def initialize_logger(file_dir):
    """Print the results in the log file."""
    logger      = logging.getLogger()
    fhandler    = logging.FileHandler(filename=file_dir, mode='a')
    chhander    = logging.StreamHandler() if args.chhander else None
    formatter   = logging.Formatter(fmt='[%(asctime)s]  %(message)s',
        datefmt ='%m-%d %H:%M')
    fhandler.setFormatter(formatter)
    chhander.setFormatter(formatter) if args.chhander else None
    logger.addHandler(fhandler)
    logger.addHandler(chhander) if args.chhander else None
    logger.setLevel(logging.INFO)
    return logger


def record_result(result, epoch, acc, mf1,wf1 ,recall , precision, c_mat , record_flag = 0):
    """ Record evaluation results."""
    if record_flag == 0:
        result.write('Best validation epoch | accuracy: {:.4f}, mF1: {:.4f}, wF1: {:.4f}, Rec: {:.4f}, Pre: {:.4f} (at epoch {})\n'.format(acc, mf1, wf1 ,recall,precision, epoch))
    elif record_flag == -1:
        result.write('\n\nTest (Best) | accuracy: {:.4f}, mF1: {:.4f}, wF1: {:.4f}, Rec: {:.4f}, Pre: {:.4f}\n'.format(acc, mf1,wf1,recall , precision, epoch))
        result.write(np.array2string(c_mat))
        result.flush()
        result.close
    elif record_flag == -2:
        result.write('\n\nTest (Final) | accuracy: {:.4f}, mF1: {:.4f}, wF1: {:.4f}, Rec: {:.4f}, Pre: {:.4f}\n'.format(acc, mf1,wf1,recall , precision, epoch))
    elif record_flag == -3:
        result.write('\n\nFinal validation epoch | accuracy: {:.4f}, mF1: {:.4f}, wF1: {:.4f}, Rec: {:.4f}, Pre: {:.4f}\n'.format(acc, mf1,wf1,recall , precision, epoch))
    

