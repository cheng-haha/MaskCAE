
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


def write_result_to_csv(**kwargs):
    if not os.path.exists("runs/"):
        print('==>create runs file')
        os.mkdir("runs/")
    core_logging(**kwargs)

class Recorder(object):
    def __init__(self,args):
        self.args      = args
        self.save_alg  = self.args.dataset \
                +  "_" + self.args.model                        +  "_"  +  str(self.args.mode)            + 'm'\
                +  "_" + str(self.args.epochs)          + "e"   +  "_"  +  str(self.args.batch_size)      + 'b' \
                +  "_" + str(self.args.learning_rate)   + "l"     
        self.choiced_mode()
        print(f'#####save file name:{self.save_alg}')
        self.save_path = f"./results/{self.args.mode}/"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.rs_train_acc   , self.rs_train_loss, \
        self.rs_valid_acc   , self.rs_valid_loss,  \
        self.rs_valid_wf1   , self.rs_valid_mf1 , \
        self.rs_valid_recall, self.rs_valid_precision, \
        self.rs_test_acc   ,\
        self.rs_test_wf1   , self.rs_test_mf1 ,\
        self.rs_test_recall, self.rs_test_precision \
        =   [], [], \
            [], [], \
            [], [], \
            [], [], \
            [],     \
            [], [], \
            [], []

    def choiced_mode(self):
        self.eva_contain_method()


    # Ablation Exp
    def eva_contain_method(self):
        '''you must need a different symbol, like 'ab' '''
        TrackingVariableList = re.findall(r'\w+', getTrackingVariables(self.args.mode) )
        if TrackingVariableList != []:
            print(TrackingVariableList)
            Tracking = '_' + '_'.join([ '{' + f'{elem}' + '}'+ f'{elem}' for elem in TrackingVariableList]) 
            self.save_alg += Tracking.format(**vars(self.args))
        else:
            print('==>Do not track any variables.')

    def save_results(self,time):
        if not self.args.not_save_res:
            # setting
            save_alg = self.save_alg + "_" + str(time)
            if len(self.rs_valid_acc) != 0:
                print(f"{save_alg}")
                with h5py.File( self.save_path + f'{save_alg}.h5', 'w') as hf:
                    hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                    hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                    hf.create_dataset('rs_valid_loss', data=self.rs_valid_loss)
                    hf.create_dataset('rs_valid_acc', data=self.rs_valid_acc)
                    hf.create_dataset('rs_valid_wf1', data=self.rs_valid_wf1)
                    hf.create_dataset('rs_valid_mf1', data=self.rs_valid_mf1)
                    hf.create_dataset('rs_valid_recall', data=self.rs_valid_recall)
                    hf.create_dataset('rs_valid_precision', data=self.rs_valid_precision)

                    hf.close()
            else:
                raise ValueError('please check your config or code')

    def get_all_training_data_value( self ):
        avg_train_loss  = np.zeros((self.args.times, self.args.epochs))
        avg_train_acc   = np.zeros((self.args.times, self.args.epochs))
        avg_test_loss   = np.zeros((self.args.times, self.args.epochs))
        avg_acc         = np.zeros((self.args.times, self.args.epochs))
        avg_mf1         = np.zeros((self.args.times, self.args.epochs))
        avg_wf1         = np.zeros((self.args.times, self.args.epochs))
        avg_rec         = np.zeros((self.args.times, self.args.epochs))
        avg_pre         = np.zeros((self.args.times, self.args.epochs))
        for i in range(self.args.times):
            alg  = self.save_alg
            alg += "_" + str(i)
            avg_train_loss[i,:], avg_train_acc[i,:] , avg_test_loss[i,:],\
                avg_acc[i, :],avg_mf1[i, :],avg_wf1[i, :],avg_rec[i, :], avg_pre[i, :]  = np.array(
                        self.simple_read_data(alg) )[: , :self.args.epochs]


        return avg_train_loss,avg_train_acc,avg_test_loss, avg_acc, avg_mf1, avg_wf1 ,avg_rec,avg_pre

    def check_data(self,data):
        try:
            data[0]
        except IndexError as e:
            data = np.zeros(self.args.epochs)
        return data 

    def simple_read_data(self,alg):
        hf = h5py.File(self.save_path + f'{alg}.h5', 'r')
        rs_train_acc        = np.array(hf.get('rs_train_acc')[:])
        rs_train_loss       = np.array(hf.get('rs_train_loss')[:])
        rs_valid_loss       = np.array(hf.get('rs_valid_loss')[:])
        rs_valid_acc        = np.array(hf.get('rs_valid_acc')[:])
        rs_valid_mf1        = np.array(hf.get('rs_valid_mf1')[:])
        rs_valid_wf1        = np.array(hf.get('rs_valid_wf1')[:])
        rs_valid_recall     = np.array(hf.get('rs_valid_recall')[:])
        rs_valid_precision  = np.array(hf.get('rs_valid_precision')[:])
        rs_train_acc        = self.check_data(rs_train_acc)
        rs_train_loss       = self.check_data(rs_train_loss)
        
        return rs_train_loss , rs_train_acc, rs_valid_loss, rs_valid_acc, rs_valid_mf1, rs_valid_wf1 , rs_valid_recall , rs_valid_precision

    def read_alg(self,alg = None):
        if alg:
            if self.args.times == 1 :
                alg = alg + '_' + '1t'+ '_' + "avg"
            else:
                alg = alg + '_' + "avg"
            return alg
        else:
            if self.args.times == 1 :
                alg = self.save_alg + '_' + '1t'+ '_' + "avg"
            else:
                alg = self.save_alg + '_' + "avg"
            return alg


    def average_data(self, logger , not_writer = False , model_state = None ):
        rs_train_loss , rs_train_acc, rs_valid_loss, rs_valid_acc, rs_valid_mf1, rs_valid_wf1 , rs_valid_recall , rs_valid_precision  = self.get_all_training_data_value()
        train_loss          = np.average(rs_train_loss, axis=0)
        train_acc           = np.average(rs_train_acc, axis=0)
        valid_loss          = np.average(rs_valid_loss, axis=0)

        glob_valid_acc      = np.average(rs_valid_acc, axis=0)
        glob_valid_mf1      = np.average(rs_valid_mf1,axis=0)
        glob_valid_wf1      = np.average(rs_valid_wf1,axis=0)
        glob_valid_rec      = np.average(rs_valid_recall,axis=0)
        glob_valid_pre      = np.average(rs_valid_precision, axis=0)
        
        model          =  GetModel() if not model_state else model_state
        inferredspeed  =  GetInferredSpeed(self.args.dataset,model.to('cpu'),500, self.args.is_sup)
        Macs , Params  =  GetFlopsAndParams(self.args.dataset,model.to('cpu'),logger=logger,is_sup=self.args.is_sup)
        if not not_writer:
            assert  isinstance(self.rs_test_acc,list) and self.rs_test_acc != [] 
            write_result_to_csv(
                            **vars(self.args),
                            path = 'result.csv',
                            macs = Macs,params      =   Params,
                            MeanTestAcc             =   str( np.around( np.mean( self.rs_test_acc       ) * 100 , 3 ) ),
                            MeanTestmF1             =   str( np.around( np.mean( self.rs_test_mf1       ) * 100 , 3 ) ),
                            MeanTestwF1             =   str( np.around( np.mean( self.rs_test_wf1       ) * 100 , 3 ) ),
                            MeanTestRecall          =   str( np.around( np.mean( self.rs_test_recall    ) * 100 , 3 ) ),
                            MeanTestPrecision       =   str( np.around( np.mean( self.rs_test_precision ) * 100 , 3 ) ),
                                                            
                            MeanValidAcc            =   str( np.around( np.mean(glob_valid_acc[-50:]) * 100, 3  ) ),
                            MeanValidmF1            =   str( np.around( np.mean(glob_valid_mf1[-50:]) * 100, 3  ) ),
                            MeanValidwF1            =   str( np.around( np.mean(glob_valid_wf1[-50:]) * 100, 3  ) ),
                            MeanValidRecall         =   str( np.around( np.mean(glob_valid_rec[-50:]) * 100, 3  ) ),
                            MeanValidPrecision      =   str( np.around( np.mean(glob_valid_pre[-50:]) * 100, 3  ) ),
                            inferredspeed           =   str( np.around( inferredspeed , 5 ) ) 
                            )

        alg = self.read_alg()
        if (len(glob_valid_acc) != 0) & (len(valid_loss) != 0 ):
            with h5py.File( self.save_path + f'{alg}.h5', 'w') as hf:
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_valid_loss', data=self.rs_valid_loss)
                hf.create_dataset('rs_valid_acc', data=self.rs_valid_acc)
                hf.create_dataset('rs_valid_mf1', data=self.rs_valid_mf1)
                hf.create_dataset('rs_valid_wf1', data=self.rs_valid_wf1)
                hf.create_dataset('rs_valid_recall', data=self.rs_valid_recall)
                hf.create_dataset('rs_valid_precision', data=self.rs_valid_precision)
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_wf1', data=self.rs_test_wf1)
                hf.create_dataset('rs_test_mf1', data=self.rs_test_mf1)
                hf.create_dataset('rs_test_recall', data=self.rs_test_recall)
                hf.create_dataset('rs_test_precision', data=self.rs_test_precision)
                hf.close()
                
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
    


def add_Variables(results,BaseLoggingText,BaseLoggingVariables,TrackingVariables,kwargs):
    if not results.exists():
            results.write_text(
                BaseLoggingText + TrackingVariables
            )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        # print(dict(**kwargs))
            f.write(
                (
                    BaseLoggingVariables + re.sub(r"\w+", r"{\g<0>}", TrackingVariables ) 

                ).format(now=now, **kwargs)
            )

def getTrackingVariables(mode):
    print(mode)
    if mode == 'Base':
        TrackingVariables =  "num_of_cv, Part,\n"
    elif mode in ['maskcae_ft','maskcae']:
        TrackingVariables =  "mask, num_of_cv, Part, linear_evaluation, \n"
    else:
        TrackingVariables =  '\n'
    return TrackingVariables

def core_logging(**kwargs):
    results = pathlib.Path("runs") / "{mode}_{path}".format( args.mode ,**kwargs)
    BaseLoggingText = 'Time, DataSet, Model, MODE, Epochs, Batch_Size, Learning_Rate, Weight_Decay, Macs, Params, times, MeanTestAcc, MeanTestmF1, MeanTestwF1, MeanTestRecall, MeanTestPrecision, MeanValidAcc, MeanValidmF1, MeanValidwF1, MeanValidRecall, MeanValidPrecision, Inferredspeed'
    BaseLoggingVariables = '{now}, {dataset}, {model}, {mode}, {epochs}, {batch_size}, {learning_rate}, {weight_decay}, {macs}, {params}, {times}, {MeanTestAcc}, {MeanTestmF1}, {MeanTestwF1}, {MeanTestRecall}, {MeanTestPrecision}, {MeanValidAcc}, {MeanValidmF1}, {MeanValidwF1}, {MeanValidRecall}, {MeanValidPrecision}, {inferredspeed}'
    TrackingVariables = getTrackingVariables(args.mode)
    add_Variables(results,BaseLoggingText,BaseLoggingVariables,TrackingVariables,kwargs)