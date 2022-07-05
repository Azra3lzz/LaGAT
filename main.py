# -*- coding: utf-8 -*-

import os
import gc
import time
import tensorflow as tf
import numpy as np
from collections import defaultdict
from keras import backend as K
from keras import optimizers

from utils import pickle_load, format_filename, write_log
from models import KGCN,KGCN_Multi
from config import ModelConfig, PROCESSED_DATA_DIR,  ENTITY_VOCAB_TEMPLATE, \
    RELATION_VOCAB_TEMPLATE, ADJ_ENTITY_TEMPLATE, ADJ_RELATION_TEMPLATE, LOG_DIR, PERFORMANCE_LOG, \
    DRUG_VOCAB_TEMPLATE

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def get_optimizer(op_type, learning_rate):
    if op_type == 'sgd':
        return optimizers.SGD(learning_rate)
    elif op_type == 'rmsprop':
        return optimizers.RMSprop(learning_rate)
    elif op_type == 'adagrad':
        return optimizers.Adagrad(learning_rate)
    elif op_type == 'adadelta':
        return optimizers.Adadelta(learning_rate)
    elif op_type == 'adam':
        return optimizers.Adam(learning_rate,clipnorm=5)# clipvalue = 5)#, 
    else:
        raise ValueError('Optimizer Not Understood: {}'.format(op_type))
def takeThird(elem):
    return elem[2]
def train(train_d,dev_d,test_d,kfold,dataset, neighbor_sample_size, ent_embed_dim, n_depth, l2_weight, lr, optimizer_type,
          batch_size, aggregator_type, n_epoch,lc,c,r,head,callbacks_to_add=None, overwrite = True):
    config = ModelConfig()
    #
    config.neighbor_sample_size = neighbor_sample_size
    config.ent_embed_dim = ent_embed_dim
    config.n_depth = n_depth
    config.l2_weight = l2_weight
    config.dataset=dataset
    config.K_Fold=kfold
    config.lr = lr
    config.optimizer = get_optimizer(optimizer_type, lr)
    config.batch_size = batch_size
    config.aggregator_type = aggregator_type
    config.n_epoch = n_epoch
    config.callbacks_to_add = callbacks_to_add
    config.lc = lc
    config.c = c
    config.r = r
    config.head = head
    '''
    config.drug_vocab = pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                             DRUG_VOCAB_TEMPLATE,
                                                             dataset=dataset))
    '''
    config.relation_vocab_size = len(pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                                 RELATION_VOCAB_TEMPLATE,
                                                                 dataset=dataset,HOP = n_depth,N = neighbor_sample_size)))
    
    config.entity_vocab_size = len(pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                               ENTITY_VOCAB_TEMPLATE,
                                                               dataset=dataset,HOP = n_depth,N = neighbor_sample_size)))
    config.adj_entity = np.load(format_filename(PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE,
                                                dataset=dataset,HOP = n_depth,N = neighbor_sample_size))
    config.adj_relation = np.load(format_filename(PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE,
                                                  dataset=dataset,HOP = n_depth,N = neighbor_sample_size))

    config.exp_name = f'kgcn_{dataset}_neigh_{neighbor_sample_size}_embed_{ent_embed_dim}_depth_' \
                      f'{n_depth}_agg_{aggregator_type}_optimizer_{optimizer_type}_lr_{lr}_' \
                      f'batch_size_{batch_size}_epoch_{n_epoch}'
    callback_str = '_' + '_'.join(config.callbacks_to_add)
    callback_str = callback_str.replace('_modelcheckpoint', '').replace('_earlystopping', '')
    config.exp_name += callback_str
    

    train_log = {'exp_name': config.exp_name, 'batch_size': batch_size, 'optimizer': optimizer_type,
                 'epoch': n_epoch, 'learning_rate': lr}
    print('Logging Info - Experiment: %s' % config.exp_name) 
    model_save_path = os.path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    model = KGCN(config)

    train_data=np.array(train_d)
    valid_data=np.array(dev_d)
    test_data=np.array(test_d)
    if not os.path.exists(model_save_path) or overwrite:
        start_time = time.time()
        model.fit(x_train=[train_data[:, :1], train_data[:, 1:2]], y_train=train_data[:, 2:3],
                  x_valid=[valid_data[:, :1], valid_data[:, 1:2]], y_valid=valid_data[:, 2:3])
        elapsed_time = time.time() - start_time
        print('Logging Info - Training time: %s' % time.strftime("%H:%M:%S",
                                                                 time.gmtime(elapsed_time)))
        train_log['train_time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    print('Logging Info - Evaluate over valid data:')
    model.load_best_model()
    #
    auc, acc, f1,aupr = model.score(x=[valid_data[:, :1], valid_data[:, 1:2]], y=valid_data[:, 2:3])
    print(f'Logging Info - dev_auc: {auc}, dev_acc: {acc}, dev_f1: {f1}, dev_aupr: {aupr}'
          )
    train_log['dev_auc'] = auc
    train_log['dev_acc'] = acc
    train_log['dev_f1'] = f1
    train_log['dev_aupr']=aupr
    train_log['k_fold']=kfold
    train_log['dataset']=dataset
    train_log['aggregate_type']=config.aggregator_type
    
    #
    print('Logging Info - Evaluate over test data:')
    model.load_best_model()
    auc, acc, f1, aupr = model.score(x=[test_data[:, :1], test_data[:, 1:2]], y=test_data[:, 2:3])
    
    #
    if r == 1 and kfold == 1:
        y_pred = model.predict([test_data[:, :1], test_data[:, 1:2]])
        y_label = model.predict([test_data[:, :1], test_data[:, 1:2]]).flatten()
        for i in range(len(y_label)):
            if y_label[i] >= 0.5:
                y_label[i] = 1
            else:
                y_label[i] = 0
        prediction = []
        for i in range(len(test_data)):
            prediction.append([str(test_data[i,0]),str(test_data[i,1]),str(round(y_pred[i],5))])
        
        '''
        case = [prediction[i] for i in range(len(label)) if label[i]==test_data[i, 2:3]]
        case.sort(key = takeThird,reverse=True)
        np.savetxt(LOG_DIR+f'/kgcn_model_{dataset}_{aggregator_type}_case{c}.txt',np.array(case),fmt="%s")
        '''
        np.savetxt(LOG_DIR+f'/kgcn_model_{dataset}_{aggregator_type}_case{c}.txt',np.array(prediction),fmt="%s")
        #
        #The current notation can only be used when hop=1 and c=3
        if c == 3:
            neibor,attention = model.predict_attention([test_data[:, :1], test_data[:, 1:2]])
            neibor_entity_one = neibor[0][1].numpy()
            neibor_entity_two = neibor[1][1].numpy()
            np.savetxt(LOG_DIR+f'/kgcn_model_{dataset}_{aggregator_type}_{c}_neiborone.txt',neibor_entity_one,fmt="%d")
            np.savetxt(LOG_DIR+f'/kgcn_model_{dataset}_{aggregator_type}_{c}_neibortwo.txt',neibor_entity_two,fmt="%d")
            np.savetxt(LOG_DIR+f'/kgcn_model_{dataset}_{aggregator_type}_{c}_at_one.txt',attention[0],fmt="%.3f")
            np.savetxt(LOG_DIR+f'/kgcn_model_{dataset}_{aggregator_type}_{c}_at_two.txt',attention[1],fmt="%.3f")
        '''
        nerbor_attention_one = []
        for i in range(len(neibor_entity_one)):
            temp = []
            num_neibor = len(neibor_entity_one[i])
            for j in range(num_neibor):
                temp.append((str(neibor_entity_one[i][j]),round(attention_one[0][i][j]/num_neibor,5)))
            nerbor_attention_one.append(temp)
        nerbor_attention_two = []
        for i in range(len(neibor_entity_two)):
            temp = []
            num_neibor = len(neibor_entity_two[i])
            for j in range(num_neibor):
                temp.append((str(neibor_entity_two[i][j]),round(attention_two[1][i][j]/num_neibor,5)))
            nerbor_attention_two.append(temp)
        
        prediction = []
        for i in range(len(test_data)):
            prediction.append([str(test_data[i,0]),str(test_data[i,1]),str(round(y_pred[i],5)),nerbor_attention_one[i],nerbor_attention_two[i]])
        prediction.sort(key = takeThird,reverse = True)
        total_case = {str(prediction[i][0])+'-'+str(prediction[i][1])+','+str(prediction[i][2]):[prediction[i][3],prediction[i][4]] for i in range(len(prediction))}
        write_log(LOG_DIR+f'/kgcn_model_{dataset}_{aggregator_type}_{c}_TotalAt.txt',log = total_case)
        '''
    train_log['test_auc'] = auc
    train_log['test_acc'] = acc
    train_log['test_f1'] = f1
    train_log['test_aupr'] =aupr
    print(f'Logging Info - test_auc: {auc}, test_acc: {acc}, test_f1: {f1}, test_aupr: {aupr}')
    
    train_log['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG), log=train_log, mode='a')
    del model
    gc.collect()
    K.clear_session()
    return train_log

def train_M(train_d,dev_d,test_d,kfold,dataset, neighbor_sample_size, ent_embed_dim, n_depth, l2_weight, lr, optimizer_type,
          batch_size, aggregator_type, n_epoch,lc,c,r,head,callbacks_to_add=None, overwrite=True):
    config = ModelConfig()
    #
    config.neighbor_sample_size = neighbor_sample_size
    config.ent_embed_dim = ent_embed_dim
    config.n_depth = n_depth
    config.l2_weight = l2_weight
    config.dataset=dataset
    config.K_Fold=kfold
    config.lr = lr
    config.optimizer = get_optimizer(optimizer_type, lr)
    config.batch_size = batch_size
    config.aggregator_type = aggregator_type
    config.n_epoch = n_epoch
    config.callbacks_to_add = callbacks_to_add
    config.lc = lc
    config.c = c
    config.r = r
    config.head = head
    config.checkpoint_monitor = 'val_acc' 
    config.early_stopping_monitor = 'val_acc'

    '''
    config.drug_vocab = pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                             DRUG_VOCAB_TEMPLATE,
                                                             dataset=dataset))
    '''
    config.relation_vocab_size = len(pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                                 RELATION_VOCAB_TEMPLATE,
                                                                 dataset=dataset,HOP = n_depth,N = neighbor_sample_size)))
    
    config.entity_vocab_size = len(pickle_load(format_filename(PROCESSED_DATA_DIR,
                                                               ENTITY_VOCAB_TEMPLATE,
                                                               dataset=dataset,HOP = n_depth,N = neighbor_sample_size)))
                                                               
    config.adj_entity = np.load(format_filename(PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE,
                                                dataset=dataset,HOP = n_depth,N = neighbor_sample_size))
    config.adj_relation = np.load(format_filename(PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE,
                                                  dataset=dataset,HOP = n_depth,N = neighbor_sample_size))

    config.exp_name = f'kgcn_M_{dataset}_neigh_{neighbor_sample_size}_embed_{ent_embed_dim}_depth_' \
                      f'{n_depth}_agg_{aggregator_type}_optimizer_{optimizer_type}_lr_{lr}_' \
                      f'batch_size_{batch_size}_epoch_{n_epoch}'
    callback_str = '_' + '_'.join(config.callbacks_to_add)
    callback_str = callback_str.replace('_modelcheckpoint', '').replace('_earlystopping', '')
    config.exp_name += callback_str
    

    train_log = {'exp_name': config.exp_name, 'batch_size': batch_size, 'optimizer': optimizer_type,
                 'epoch': n_epoch, 'learning_rate': lr}
    print('Logging Info - Experiment: %s' % config.exp_name) 
    model_save_path = os.path.join(config.checkpoint_dir, '{}.hdf5'.format(config.exp_name))
    model = KGCN_Multi(config)

    train_data=np.array(train_d)
    valid_data=np.array(dev_d)
    test_data=np.array(test_d)
    if not os.path.exists(model_save_path) or overwrite:
        start_time = time.time()
        model.fit(x_train=[train_data[:, :1], train_data[:, 1:2]], y_train=train_data[:, 2:3],
                  x_valid=[valid_data[:, :1], valid_data[:, 1:2]], y_valid=valid_data[:, 2:3])
        elapsed_time = time.time() - start_time
        print('Logging Info - Training time: %s' % time.strftime("%H:%M:%S",
                                                                 time.gmtime(elapsed_time)))
        train_log['train_time'] = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

    print('Logging Info - Evaluate over valid data:')
    model.load_best_model()#
    #
    acc, f1 = model.score(x=[valid_data[:, :1], valid_data[:, 1:2]], y=valid_data[:, 2:3])

    print(f'Logging Info - dev_acc: {acc}, dev_f1: {f1}')
    train_log['dev_acc'] = acc
    train_log['dev_f1'] = f1
    
    train_log['k_fold']=kfold
    train_log['dataset']=dataset
    train_log['aggregate_type']=config.aggregator_type
    #
    print('Logging Info - Evaluate over test data:')
    model.load_best_model()
    acc, f1 = model.score(x=[test_data[:, :1], test_data[:, 1:2]], y=test_data[:, 2:3])
   
    train_log['test_acc'] = acc
    train_log['test_f1'] = f1
    
    print(f'Logging Info - test_acc: {acc}, test_f1: {f1}')
    #
    if r == 1 and kfold == 1:
        y_pred = model.predict([test_data[:, :1], test_data[:, 1:2]])
        label = np.argmax(y_pred,1)
        prediction = []
        for i in range(len(test_data)):
            prediction.append([str(test_data[i,0]),str(test_data[i,1]),str(round(y_pred[i][label[i]],5)),label[i],test_data[i,2]])
        
        '''
        case = [prediction[i] for i in range(len(label)) if label[i]==test_data[i, 2:3]]
        case.sort(key = takeThird,reverse=True)
        np.savetxt(LOG_DIR+f'/kgcn_model_M_{dataset}_{aggregator_type}_case{c}.txt',np.array(case),fmt="%s")
        '''
        np.savetxt(LOG_DIR+f'/kgcn_model_M_{dataset}_{aggregator_type}_case{c}.txt',np.array(prediction),fmt="%s")
        #
        #The current notation can only be used when hop=1 and c=3
        if c == 3:
            neibor,attention = model.predict_attention([test_data[:, :1], test_data[:, 1:2]])
            neibor_entity_one = neibor[0][1].numpy()
            neibor_entity_two = neibor[1][1].numpy()
            np.savetxt(LOG_DIR+f'/kgcn_model_M_{dataset}_{aggregator_type}_{c}_neiborone.txt',neibor_entity_one,fmt="%d")
            np.savetxt(LOG_DIR+f'/kgcn_model_M_{dataset}_{aggregator_type}_{c}_neibortwo.txt',neibor_entity_two,fmt="%d")
            np.savetxt(LOG_DIR+f'/kgcn_model_M_{dataset}_{aggregator_type}_{c}_at_one.txt',attention[0],fmt="%.3f")
            np.savetxt(LOG_DIR+f'/kgcn_model_M_{dataset}_{aggregator_type}_{c}_at_two.txt',attention[1],fmt="%.3f")
        '''
        nerbor_attention_one = []
        for i in range(len(neibor_entity_one)):
            temp = []
            num_neibor = len(neibor_entity_one[i])
            for j in range(num_neibor):
                temp.append((str(neibor_entity_one[i][j]),round(attention_one[0][i][j]/num_neibor,5)))
            nerbor_attention_one.append(temp)
        nerbor_attention_two = []
        for i in range(len(neibor_entity_two)):
            temp = []
            num_neibor = len(neibor_entity_two[i])
            for j in range(num_neibor):
                temp.append((str(neibor_entity_two[i][j]),round(attention_two[1][i][j]/num_neibor,5)))
            nerbor_attention_two.append(temp)
        
        label = np.argmax(y_pred,1)
        prediction = []
        for i in range(len(test_data)):
            prediction.append([str(test_data[i,0]),str(test_data[i,1]),str(round(y_pred[i][label[i]],5)),label[i],test_data[i,2],nerbor_attention_one[i],nerbor_attention_two[i]])
        prediction.sort(key = takeThird,reverse = True)
        total_case = {str(prediction[i][0])+'-'+str(prediction[i][1])+'-'+str(prediction[i][4])+','+str(prediction[i][2])+','+str(prediction[i][3]):[prediction[i][5],prediction[i][6]] for i in range(len(prediction))}
        write_log(LOG_DIR+f'/kgcn_model_M_{dataset}_{aggregator_type}_{c}_TotalAt.txt',log = total_case)
        '''
    train_log['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    write_log(format_filename(LOG_DIR, PERFORMANCE_LOG), log=train_log, mode='a')
    del model
    gc.collect()
    K.clear_session()
    return train_log

