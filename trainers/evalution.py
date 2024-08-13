from sklearn.metrics import confusion_matrix
from sklearn import metrics
from configs.base_configs import args
import torch.nn as nn
import os
import torch
from utils.logger import record_result 
import numpy as np
import wandb 
from torch.utils.data import DataLoader
import copy

def save_rep(model,eval_loader):
    # assert  isinstance(self.model.classifier, nn.Identity) , print('Wrong Model!')
    X = {'layer1':[],'layer2':[],'layer3':[]}
    Y = []
    model.cuda()
    model.eval()
    # print(idx)
    for layer, layer_x  in enumerate( list( X.values() ) ):
        for x, y in eval_loader:
            x, y = x.cuda().float(), y.cuda().long()
            model.zero_grad()
            rep  = model(x)['feature_map_list']
            if layer_x == []:
                layer_x = rep[layer].cpu().detach().numpy()
            else:
                layer_x = np.concatenate((layer_x , rep[layer].cpu().detach().numpy() ), axis=0 )
            X[f'layer{str(layer+1)}'] = layer_x
            if layer == 0:
                for i in range(len(y)):
                    Y.append(y[i].cpu().detach().numpy())
    Y = np.array(Y)
    save_path = os.path.join( args.save_folder , 'Attention' )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    assert len(X['layer1']) ==len(X['layer2'])==len(X['layer3'])== len(Y) , 'Layer length is not equal!'
    print(f'all test samples is {len(Y)}')
    np.save(save_path + '/feature_maps.npy', X, allow_pickle=True)
    np.save(save_path + '/y_data.npy', Y)
    

def evaluate(model,logger, eval_loader, epoch, is_test=True, mode='best', no_dict_ouput = False, stat = {}):
    # Is Not Validation
    if is_test:
        eval_loader = DataLoader( eval_loader, batch_size=args.batch_size, shuffle=False, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True )
        model.load_state_dict(torch.load(os.path.join(args.save_folder, mode + '.pth')), strict=False)
    model.eval()

    criterion_cls = nn.CrossEntropyLoss()

    total_loss , corect_num , total_num, mf1_test, wf1_test , Recall_test , Precision_test = 0.0 , 0 , 0 , 0.0 , 0.0 , 0.0 , 0.0
    label_list, predicted_list = [], []
    with torch.no_grad():
        for idx, (data, label) in enumerate(eval_loader):
            model.eval()
            data, label = data.cuda().float(), label.cuda().long()
            if no_dict_ouput:
                output = model(data)
            else:
                output = model(data)['output']
            total_loss += criterion_cls(output, label).detach().item()
            _, predicted = torch.max(output, 1)
            label_list.append(label)
            predicted_list.append(predicted)
            label          = label.cpu().detach().numpy()
            predicted      = predicted.cpu().detach().numpy()
            corect_num    += (predicted == label).sum()
            total_num     += len(label)
    batch_loss     = total_loss / len(eval_loader)
    ALL_label      = torch.cat(label_list).cpu().detach().numpy()
    ALL_predicted  = torch.cat(predicted_list).cpu().detach().numpy()
    acc_test       = corect_num/ total_num
    test_error     = 1.0 - acc_test
    mf1_test       = metrics.f1_score(ALL_label, ALL_predicted, average='macro')
    wf1_test       = metrics.f1_score(ALL_label, ALL_predicted, average='weighted')
    Recall_test    = metrics.recall_score(ALL_label, ALL_predicted, average='macro') 
    Precision_test = metrics.precision_score(ALL_label, ALL_predicted, average='macro') 
    
    if is_test:
        logger.info('=> test@Acc: {:.5f}%, test@mF1: {:.5f}, test@wF1: {:.5f}, test@Rec: {:.5f}, test@Pre: {:.5f}'.format(acc_test, mf1_test, wf1_test ,Recall_test,Precision_test))
        print('\033[34m####################### test@Acc: {:.5f}%, test@mF1: {:.5f}, test@wF1: {:.5f} #######################\033[0m'.format(acc_test, mf1_test, wf1_test))
        c_mat  = confusion_matrix(ALL_label, ALL_predicted)
        result = open(os.path.join(args.save_folder, 'result'), 'a+')
        record_result(result, epoch, acc_test, mf1_test,wf1_test ,Recall_test,Precision_test, c_mat)
        np.save(os.path.join(args.save_folder,'TestConMat.npy') , c_mat )
    else:
        wandb.log({'loss':batch_loss ,"Test Error":test_error, "ACC":acc_test,'mF1' :mf1_test,'wF1' :wf1_test, 'Rec': Recall_test , 'Pre': Precision_test } ) if args.use_wandb else None

        logger.info('=> valid@Acc: {:.5f}, valid@mF1: {:.5f}, valid@wF1: {:.5f}, valid@Rec: {:.5f}, valid@Pre: {:.5f}'.format(acc_test, mf1_test, wf1_test ,Recall_test,Precision_test))
        logger.info('=> cls_loss: {:.7f}'.format(batch_loss))
        
       
        if mf1_test > stat['best_mf1']:
            logger.info('=> new best valid mf1, save model weights...')
            # metrics for mesuring performance of model
            stat['best_acc'] = acc_test
            stat['best_mf1'] = mf1_test
            stat['best_wf1'] = wf1_test
            stat['best_recall'] = Recall_test
            stat['best_precision'] = Precision_test
            # calculate best confusion matrix
            stat['cmt'] = confusion_matrix(ALL_label, ALL_predicted)
            
            stat['best_epoch'] = epoch
            torch.save( model.state_dict(), os.path.join(args.save_folder, 'best.pth') )

                

    # save final model
    if epoch == args.epochs - 1 and (not is_test):
        torch.save( model.state_dict(), os.path.join( args.save_folder, 'final.pth' ) )
        #NOTE save final confusion matrix 
        CM = confusion_matrix(ALL_label, ALL_predicted)
        np.save(os.path.join(args.save_folder,'FinalConMat.npy') , CM )
        result = open(os.path.join(args.save_folder, 'result'), 'w')
        logger.info( 'Done' )
        logger.info( 'Best performance achieved at epoch {}, Best@Acc: {:.5f}, Best@mF1: {:.5f}, Best@wF1: {:.5f}, Best@Rec: {:.5f}, Best@Pre: {:.5f}'\
            .format( stat['best_epoch'], stat['best_acc'], stat['best_mf1'], stat['best_wf1'],stat['best_recall'],stat['best_precision'] ) )
        logger.info( stat['cmt'])
        #NOTE logging confusion matrix of best epoch
        record_result( result, epoch=stat['best_epoch'], 
                      acc = stat['best_acc'], 
                      mf1 = stat['best_mf1'], 
                      wf1 = stat['best_wf1'],
                      recall = stat['best_recall'],
                      precision = stat['best_precision'],
                      c_mat = stat['cmt'] ,
                      record_flag= 0 )
        if args.vis_feature:
            save_rep( model , eval_loader=eval_loader)
    return total_loss, acc_test, mf1_test, wf1_test, Recall_test, Precision_test , stat


def evaluate_mobilehar(model,logger, eval_loader, epoch, is_test=True, mode='best', no_dict_ouput = False, stat = {}):
    # Is Not Validation
    if is_test:
        eval_loader = DataLoader( eval_loader, batch_size=args.batch_size, shuffle=False, drop_last=False,
                              num_workers=args.num_workers, pin_memory=True )
        model.load_state_dict(torch.load(os.path.join(args.save_folder, mode + '.pth')), strict=False)
    model.eval()
    if is_test:
        print('================ reparameterize model =====================')
        if hasattr(model,'reparameterize_model'):
            model.reparameterize_model()
    criterion_cls = nn.CrossEntropyLoss()

    total_loss , corect_num , total_num, mf1_test, wf1_test , Recall_test , Precision_test = 0.0 , 0 , 0 , 0.0 , 0.0 , 0.0 , 0.0
    label_list, predicted_list = [], []
    with torch.no_grad():
        for idx, (data, label) in enumerate(eval_loader):
            model.eval()
            data, label = data.cuda().float(), label.cuda().long()
            if no_dict_ouput:
                output = model(data)
            else:
                output = model(data)['output']
            total_loss += criterion_cls(output, label).detach().item()
            _, predicted = torch.max(output, 1)
            label_list.append(label)
            predicted_list.append(predicted)
            label          = label.cpu().detach().numpy()
            predicted      = predicted.cpu().detach().numpy()
            corect_num    += (predicted == label).sum()
            total_num     += len(label)
    batch_loss     = total_loss / len(eval_loader)
    ALL_label      = torch.cat(label_list).cpu().detach().numpy()
    ALL_predicted  = torch.cat(predicted_list).cpu().detach().numpy()
    acc_test       = corect_num/ total_num
    test_error     = 1.0 - acc_test
    mf1_test       = metrics.f1_score(ALL_label, ALL_predicted, average='macro')
    wf1_test       = metrics.f1_score(ALL_label, ALL_predicted, average='weighted')
    Recall_test    = metrics.recall_score(ALL_label, ALL_predicted, average='macro') 
    Precision_test = metrics.precision_score(ALL_label, ALL_predicted, average='macro') 
    
    if is_test:
        logger.info('=> test@Acc: {:.5f}%, test@mF1: {:.5f}, test@wF1: {:.5f}, test@Rec: {:.5f}, test@Pre: {:.5f}'.format(acc_test, mf1_test, wf1_test ,Recall_test,Precision_test))
        print('\033[34m####################### test@Acc: {:.5f}%, test@mF1: {:.5f}, test@wF1: {:.5f} #######################\033[0m'.format(acc_test, mf1_test, wf1_test))
        stat['reparam_model'] = model
        c_mat  = confusion_matrix(label, predicted)
        result = open(os.path.join(args.save_folder, 'result'), 'a+')
        record_result(result, epoch, acc_test, mf1_test,wf1_test ,Recall_test,Precision_test, c_mat)

    else:
        wandb.log({'loss':batch_loss ,"Test Error":test_error, "ACC":acc_test,'mF1' :mf1_test,'wF1' :wf1_test, 'Rec': Recall_test , 'Pre': Precision_test } ) if args.use_wandb else None

        logger.info('=> valid@Acc: {:.5f}, valid@mF1: {:.5f}, valid@wF1: {:.5f}, valid@Rec: {:.5f}, valid@Pre: {:.5f}'.format(acc_test, mf1_test, wf1_test ,Recall_test,Precision_test))
        logger.info('=> cls_loss: {:.7f}'.format(batch_loss))

        if mf1_test > stat['best_mf1']:
                # metrics for mesuring performance of model
                stat['best_acc'] = acc_test
                stat['best_mf1'] = mf1_test
                stat['best_wf1'] = wf1_test
                stat['best_recall'] = Recall_test
                stat['best_precision'] = Precision_test
                # calculate best confusion matrix
                stat['cmt'] = confusion_matrix(ALL_label, ALL_predicted)
                
                stat['best_epoch'] = epoch
                torch.save( model.state_dict(), os.path.join(args.save_folder, 'best.pth') )
            

    # save final model
    if epoch == args.epochs - 1 and (not is_test):
        torch.save( model.state_dict(), os.path.join( args.save_folder, 'final.pth' ) )
        #NOTE save final confusion matrix 
        CM = confusion_matrix(ALL_label, ALL_predicted)
        np.save(os.path.join(args.save_folder,'FinalConMat.npy') , CM )
        result = open(os.path.join(args.save_folder, 'result'), 'w')
        logger.info( 'Done' )
        logger.info( 'Best performance achieved at epoch {}, Best@Acc: {:.5f}, Best@mF1: {:.5f}, Best@wF1: {:.5f}, Best@Rec: {:.5f}, Best@Pre: {:.5f}'\
            .format( stat['best_epoch'], stat['best_acc'], stat['best_mf1'], stat['best_wf1'],stat['best_recall'],stat['best_precision'] ) )
        logger.info( stat['cmt'])
        #NOTE logging confusion matrix of best epoch
        record_result( result, epoch=stat['best_epoch'], 
                      acc = stat['best_acc'], 
                      mf1 = stat['best_mf1'], 
                      wf1 = stat['best_wf1'],
                      recall = stat['best_recall'],
                      precision = stat['best_precision'],
                      c_mat = stat['cmt'] ,
                      record_flag= 0 )
        if args.vis_feature:
            save_rep( model , eval_loader=eval_loader)
    return total_loss, acc_test, mf1_test, wf1_test, Recall_test, Precision_test , stat


def evaluate_cae(models,logger, eval_loader, epoch, is_test=True, mode='best', state = {}):
    # Is Not Validation
    if is_test:
        models[0].load_state_dict(torch.load(os.path.join(args.save_folder, f'{args.time}s_model_' + mode + '.pth')), strict=False)
        models[1].load_state_dict(torch.load(os.path.join(args.save_folder, f'{args.time}s_classifier_' + mode + '.pth')), strict=False)
    models[0].eval()
    models[1].eval()

    criterion_cls = nn.CrossEntropyLoss()

    total_loss , corect_num , total_num, mf1_test , wf1_test , Recall_test , Precision_test = 0.0 , 0 , 0 , 0.0 , 0.0 , 0.0 , 0.0 
    label_list, predicted_list = [], []
    with torch.no_grad():
        for idx, (data, label) in enumerate(eval_loader):
            models[0].eval()
            models[1].eval()
            data, label = data.cuda().float(), label.cuda().long()
            res         = models[0](data)
            hidden      = res['encoder']
            output      = models[1](hidden)
            total_loss += criterion_cls(output, label).detach().item()
            _, predicted = torch.max(output, 1)
            label_list.append(label)
            predicted_list.append(predicted)
            label          = label.cpu().detach().numpy()
            predicted      = predicted.cpu().detach().numpy()
            corect_num    += (predicted == label).sum()
            total_num     += len(label)
    batch_loss     = total_loss / len(eval_loader)
    ALL_label      = torch.cat(label_list).cpu().detach().numpy()
    ALL_predicted  = torch.cat(predicted_list).cpu().detach().numpy()
    acc_test       = corect_num/ total_num
    test_error     = 1.0 - acc_test
    mf1_test       = metrics.f1_score(ALL_label, ALL_predicted, average='macro')
    wf1_test       = metrics.f1_score(ALL_label, ALL_predicted, average='weighted')
    Recall_test    = metrics.recall_score(ALL_label, ALL_predicted, average='macro') 
    Precision_test = metrics.precision_score(ALL_label, ALL_predicted, average='macro') 
    wandb.log({'loss':batch_loss ,"Test Error":test_error, "ACC":acc_test,'mF1' :mf1_test,'wF1' :wf1_test, 'Rec': Recall_test , 'Pre': Precision_test } ) if args.use_wandb else None
    
    if is_test:
        logger.info('=> test@Acc: {:.5f}%, test@mF1: {:.5f}, test@wF1: {:.5f}, test@Rec: {:.5f}, test@Pre: {:.5f}'.format(acc_test, mf1_test, wf1_test ,Recall_test,Precision_test))
        print('=> test@Acc: {:.5f}%, test@mF1: {:.5f}'.format(acc_test, mf1_test))
        c_mat  = confusion_matrix(label, predicted)
        result = open(os.path.join(args.save_folder, 'result'), 'a+')
        record_result(result, epoch, acc_test, mf1_test,wf1_test ,Recall_test,Precision_test, c_mat)
    else:
        logger.info('=> valid@Acc: {:.5f}, valid@mF1: {:.5f}, valid@wF1: {:.5f}, valid@Rec: {:.5f}, valid@Pre: {:.5f}'.format(acc_test, mf1_test, wf1_test ,Recall_test,Precision_test))
        logger.info('=> cls_loss: {:.7f}'.format(batch_loss))

        if acc_test > state['best_mf1']:
                # metrics for mesuring performance of model
                state['best_acc'] = acc_test
                state['best_mf1'] = mf1_test
                state['best_wf1'] = wf1_test
                state['best_recall'] = Recall_test
                state['best_precision'] = Precision_test
                # calculate best confusion matrix
                state['cmt'] = confusion_matrix(ALL_label, ALL_predicted)
                
                state['best_epoch'] = epoch
                
                state['best_epoch'] = epoch
                torch.save(models[0].state_dict(), os.path.join(args.save_folder, f'{args.time}s_model_best.pth'))
                torch.save(models[1].state_dict(), os.path.join(args.save_folder, f'{args.time}s_classifier_best.pth'))            

    # save final model
    if epoch == args.epochs - 1:
        # save final classifier
        torch.save(models[1].state_dict(), os.path.join(args.save_folder, f'{args.time}s_classifier_final.pth'))
        #NOTE save final confusion matrix 
        CM = confusion_matrix(ALL_label, ALL_predicted)
        np.save(os.path.join(args.save_folder,'FinalConMat.npy') , CM )
        result = open(os.path.join(args.save_folder, 'result'), 'w')
        logger.info( 'Done' )
        logger.info( 'Best performance achieved at epoch {}, Best@Acc: {:.5f}, Best@mF1: {:.5f}, Best@wF1: {:.5f}, Best@Rec: {:.5f}, Best@Pre: {:.5f}'\
            .format( state['best_epoch'], state['best_acc'], state['best_mf1'], state['best_wf1'],state['best_recall'],state['best_precision'] ) )
        logger.info( state['cmt'])
        #NOTE logging confusion matrix of best epoch
        record_result( result, epoch=state['best_epoch'], 
                      acc = state['best_acc'], 
                      mf1 = state['best_mf1'], 
                      wf1 = state['best_wf1'],
                      recall = state['best_recall'],
                      precision = state['best_precision'],
                      c_mat = state['cmt'] ,
                      record_flag= 0 )
        if args.vis_feature:
            save_rep( models[0] , eval_loader=eval_loader)
    return total_loss, acc_test, mf1_test, wf1_test, Recall_test, Precision_test , state