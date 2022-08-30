'''
@Author: your name
@Date: 2019-12-20 19:02:25
@LastEditTime: 2020-05-26 20:58:12
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /matengfei/KGCN_Keras-master/config.py
'''
# -*- coding: utf-8 -*-

import os

RAW_DATA_DIR = os.getcwd()+'/raw_data'
PROCESSED_DATA_DIR = os.getcwd()+'/data'
LOG_DIR = os.getcwd()+'/log'
MODEL_SAVED_DIR = os.getcwd()+'/ckpt'
TENSORBOARD_DIR= os.getcwd()+'/TB_log'

KG_FILE = {
           'drugbank':os.path.join(RAW_DATA_DIR,'drugbank','train2id.txt'),
           'kegg':os.path.join(RAW_DATA_DIR,'kegg','train2id.txt')}
ENTITY2ID_FILE = {
                    'drugbank':os.path.join(RAW_DATA_DIR,'drugbank','entity2id.txt'),
                    'kegg':os.path.join(RAW_DATA_DIR,'kegg','entity2id.txt')}
EXAMPLE_FILE = {
               'drugbank':os.path.join(RAW_DATA_DIR,'drugbank','approved_example.txt'),
               'kegg':os.path.join(RAW_DATA_DIR,'kegg','approved_example.txt')}
SEPARATOR = {'drugbank':'\t','kegg':'\t'}

THRESHOLD = {'drugbank':4,'kegg':4} 



#
DRUG_VOCAB_TEMPLATE = '{dataset}_drug_vocab.pkl'
ENTITY_VOCAB_TEMPLATE = '{dataset}_entity_vocab_{HOP}_{N}.pkl'
RELATION_VOCAB_TEMPLATE = '{dataset}_relation_vocab_{HOP}_{N}.pkl'
ADJ_ENTITY_TEMPLATE = '{dataset}_adj_entity_{HOP}_{N}.npy'
ADJ_RELATION_TEMPLATE = '{dataset}_adj_relation_{HOP}_{N}.npy'
TRAIN_DATA_TEMPLATE = '{dataset}_train.npy'
DEV_DATA_TEMPLATE = '{dataset}_dev.npy'
TEST_DATA_TEMPLATE = '{dataset}_test.npy'
RESULT_LOG={'drugbank':'drugbank_result.txt','kegg':'kegg_result.txt'}
PERFORMANCE_LOG = 'kgcn_performance.log'
DRUG_EXAMPLE='{dataset}_examples.npy'
KFOLD_DATASET = '{dataset}_subsets.pkl'

class ModelConfig(object):
    def __init__(self):
        self.neighbor_sample_size = 16 # neighbor sampling size
        self.ent_embed_dim = 64  # dimension of entity embedding
        self.rel_embed_dim = 32   # dimension of relation embedding
        self.n_depth = 2    # depth of receptive field
        self.l2_weight = 1e-7  # l2 regularizer weight
        self.lr = 1e-2  # learning rate
        self.batch_size = 1024
        self.aggregator_type = 'neigh'
        self.n_epoch = 50
        self.optimizer = 'adam'
        self.num_rel = 86
        self.head = 1

        self.drug_vocab = None
        self.subgraph = None
        self.subgraphid = None
        self.entity_vocab_size = None
        self.relation_vocab_size = None
        self.adj_entity = None
        self.adj_relation = None

        self.exp_name = None
        self.model_name = None
        
        self.tensorboard_dir = TENSORBOARD_DIR
        # checkpoint configuration 
        self.checkpoint_dir = MODEL_SAVED_DIR
        self.checkpoint_monitor = 'val_auc'
        self.checkpoint_save_best_only = True
        self.checkpoint_save_weights_only = True
        self.checkpoint_save_weights_mode = 'max'
        self.checkpoint_verbose = 1

        # early_stoping configuration
        self.early_stopping_monitor = 'val_auc'
        self.early_stopping_mode = 'max'
        self.early_stopping_patience = 5
        self.early_stopping_verbose = 1

        self.dataset='kegg'
        self.K_Fold=1
        self.callbacks_to_add = None

        self.lc = 1
        self.c = 3
        self.r = 0
        
