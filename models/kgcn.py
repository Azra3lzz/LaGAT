# -*- coding: utf-8 -*-
from sklearn import neighbors
from keras.engine.topology import Layer
from keras.layers import *
from keras.regularizers import l2
from keras.models import Model
from keras import backend as K  
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve
import sklearn.metrics as m
import numpy as np
from keras.utils.np_utils import to_categorical
from layers import Aggregator
from callbacks import KGCNMetric,MultiMetric
from models.base_model import BaseModel
from config import ModelConfig,LOG_DIR
from models.AttentionMode import GAT_const,TBA,GAT,KGNN_base

class GetReceptiveField(Layer):
    
    def __init__(self,config:ModelConfig,name='receptive_filed',**kwargs):
        super(GetReceptiveField,self).__init__(name = name,**kwargs) 
        self.config = config
        
    def call(self,x):
        
        neigh_ent_list = [x]
        neigh_rel_list = []
        n_neighbor = K.shape(self.config.adj_entity)[1]

        for i in range(self.config.n_depth):
            new_neigh_ent = K.gather(self.config.adj_entity, K.cast(
                neigh_ent_list[-1], dtype='int64')) 
            new_neigh_rel = K.gather(self.config.adj_relation, K.cast(
                neigh_ent_list[-1], dtype='int64'))
            

            neigh_ent_list.append(
                K.reshape(new_neigh_ent, (-1, n_neighbor ** (i + 1))))
            neigh_rel_list.append(
                K.reshape(new_neigh_rel, (-1, n_neighbor ** (i + 1))))
            

        return neigh_ent_list + neigh_rel_list


class SqueezeLayer(Layer):
    def __init__(self,name='squeeze',**kwargs):
        super(SqueezeLayer,self).__init__(name = name,**kwargs)
    def call(self,x):
        return K.squeeze(x, axis=1)

class SoftmaxLayer(Layer):
    def __init__(self,config:ModelConfig,initializer='glorot_normal', regularizer=None,name='SoftmaxLayer',**kwargs):
        super(SoftmaxLayer,self).__init__(name = name,**kwargs)
        self.config = config
        self.initializer = initializer
        self.regularizer = regularizer
    def build(self, input_shape):
        self.w = self.add_weight(name='softmax_w', shape=(2*input_shape[0][1], 86),
                                 initializer=self.initializer, regularizer=self.regularizer)
        
        super(SoftmaxLayer, self).build(input_shape)
    def call(self,inputs):
        inputs = K.concatenate(inputs)
        return K.softmax(K.dot(inputs,self.w))
################### 
#
class KGCN_Multi(BaseModel):
    def __init__(self, config):
        super(KGCN_Multi, self).__init__(config)

    def build(self):
        input_drug_one = Input(
            shape=(1, ), name='input_drug_one', dtype='int64')
        input_drug_two = Input(
            shape=(1, ), name='input_drug_two', dtype='int64') 
           
        entity_embedding = Embedding(input_dim=self.config.entity_vocab_size,
                                     output_dim=self.config.ent_embed_dim,
                                     embeddings_initializer='glorot_normal',
                                     embeddings_regularizer=l2(
                                         self.config.l2_weight),
                                     name='entity_embedding')
        relation_embedding = Embedding(input_dim=self.config.relation_vocab_size,
                                       output_dim=self.config.ent_embed_dim,
                                       embeddings_initializer='glorot_normal',
                                       embeddings_regularizer=l2(
                                           self.config.l2_weight),
                                       name='relation_embedding')
        
        # Drug one
        get_receptive_field_one = GetReceptiveField(self.config,name='receptive_filed_drug_one')
        receptive_list_drug_one = get_receptive_field_one(input_drug_one)
        neineigh_ent_list_drug_one = receptive_list_drug_one[:self.config.n_depth+1]
        neigh_rel_list_drug_one = receptive_list_drug_one[self.config.n_depth+1:]
        
        neigh_ent_embed_list_drug_one = [entity_embedding(
            neigh_ent) for neigh_ent in neineigh_ent_list_drug_one]
        
        neigh_rel_embed_list_drug_one = [relation_embedding(
            neigh_rel) for neigh_rel in neigh_rel_list_drug_one]
        # Drug two
        get_receptive_field = GetReceptiveField(self.config,name='receptive_filed_drug')
        receptive_list = get_receptive_field(input_drug_two)
        neigh_ent_list = receptive_list[:self.config.n_depth+1]
        neigh_rel_list = receptive_list[self.config.n_depth+1:]
        
        neigh_ent_embed_list = [entity_embedding(
            neigh_ent) for neigh_ent in neigh_ent_list]
        
        neigh_rel_embed_list = [relation_embedding(
            neigh_rel) for neigh_rel in neigh_rel_list]

        
        #########
        #
        if self.config.c != 1 and self.config.c != 2:
            drug_embedding = Embedding(input_dim=self.config.entity_vocab_size,
                                    output_dim=self.config.ent_embed_dim,
                                    embeddings_initializer='glorot_normal',
                                    embeddings_regularizer=l2(
                                        self.config.l2_weight),
                                    name='drug_embedding')
            drug_embed_one = drug_embedding(input_drug_one)
            drug_embed_two = drug_embedding(input_drug_two)
        e_drug_one = neigh_ent_embed_list_drug_one[0]
        lc_list_one = [] 
        e_drug_two = neigh_ent_embed_list[0]
        lc_list_two = []
        
        #
        for depth in range(self.config.n_depth):
            #Attention
            #KGNN
            if self.config.c == 0:                             
                attention_scale_one = KGNN_base(
                    self.config,
                    name = f'cross_attention_{depth}_one')
                attention_scale = KGNN_base(
                    self.config,
                    name = f'cross_attention_{depth}') 
            #GAT-const
            if self.config.c == 1:                             
                attention_scale_one = GAT_const(
                    self.config,
                    name = f'cross_attention_{depth}_one')
                attention_scale = GAT_const(
                    self.config,
                    name = f'cross_attention_{depth}')
            #GAT
            if self.config.c == 2:                             
                attention_scale_one = GAT(
                    self.config, 
                    self.config.aggregator_type,
                    regularizer=l2(self.config.l2_weight),
                    name = f'cross_attention_{depth}_one')
                attention_scale = GAT(
                    self.config,
                    self.config.aggregator_type,      
                    regularizer=l2(self.config.l2_weight),
                    name = f'cross_attention_{depth}')
            #TBA
            if self.config.c == 3:                             
                attention_scale_one = TBA(
                    self.config,  
                    name = f'cross_attention_{depth}_one')
                attention_scale = TBA(
                    self.config,
                    name = f'cross_attention_{depth}')
            
            
                
            #
            if self.config.c != 2 :
                aggregator_one = Aggregator[self.config.aggregator_type](
                    activation='tanh' if depth == self.config.n_depth-1 else 'relu',
                    regularizer=l2(self.config.l2_weight),
                    name=f'aggregator_{depth+1}_drug_one'
                )
                aggregator = Aggregator[self.config.aggregator_type](
                    activation='tanh' if depth == self.config.n_depth-1 else 'relu',
                    regularizer=l2(self.config.l2_weight),
                    name=f'aggregator_{depth+1}_drug'
                )
            
            
            for hop in range(self.config.n_depth-depth):
                temp_one = neigh_ent_embed_list_drug_one[hop + 1]
                temp = neigh_ent_embed_list[hop + 1]
                
                if self.config.c == 0:
                    neighbor_embed_one = attention_scale_one([drug_embed_one,neigh_rel_embed_list_drug_one[hop],temp_one])
                    neighbor_embed = attention_scale([drug_embed_one,neigh_rel_embed_list[hop],temp])

                if self.config.c == 1:
                    neighbor_embed_one = attention_scale_one(temp_one)
                    neighbor_embed = attention_scale(temp)
                    
                if self.config.c == 2:
                    neigh_ent_embed_list_drug_one[hop] = attention_scale_one([neigh_ent_embed_list_drug_one[hop],temp_one])
                    neigh_ent_embed_list[hop] = attention_scale([neigh_ent_embed_list[hop],temp])
                
                if self.config.c == 3:
                    neighbor_embed_one,attention_value_one = attention_scale_one([drug_embed_two,temp_one])
                    neighbor_embed,attention_value = attention_scale([drug_embed_one,temp])  
                

                if self.config.c != 2 :
                    neigh_ent_embed_list_drug_one[hop] = aggregator_one([neigh_ent_embed_list_drug_one[hop], neighbor_embed_one])
                    neigh_ent_embed_list[hop] = aggregator([neigh_ent_embed_list[hop], neighbor_embed])
                
                
            if self.config.lc == 1:    
                #layer-wise 
                lc_list_one.append(neigh_ent_embed_list_drug_one[0])
                lc_list_two.append(neigh_ent_embed_list[0])
            
        
        #layer-wise 
        if self.config.lc == 1:
            e_drug_one = K.concatenate([e_drug_one,K.concatenate(lc_list_one)])
            e_drug_two = K.concatenate([e_drug_two,K.concatenate(lc_list_two)])
        else:
            e_drug_one = neigh_ent_embed_list_drug_one[0]
            e_drug_two = neigh_ent_embed_list[0]
        
            
        
        squeeze_layer = SqueezeLayer()
        drug1_squeeze_embed = squeeze_layer(e_drug_one)
        drug2_squeeze_embed = squeeze_layer(e_drug_two)
        softmax_layer = SoftmaxLayer(self.config,regularizer=l2(self.config.l2_weight))
        
        drug_drug_score = softmax_layer([drug1_squeeze_embed, drug2_squeeze_embed])

        model = Model([input_drug_one, input_drug_two], drug_drug_score)
        model.compile(optimizer=self.config.optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model
    

    def add_metrics(self, x_train, y_train, x_valid, y_valid):
        self.callbacks.append(MultiMetric(x_train, y_train, x_valid, y_valid,
                                         self.config.aggregator_type, self.config.dataset, self.config.K_Fold))

    def fit(self, x_train, y_train, x_valid, y_valid):
        self.callbacks = []
        self.add_metrics(x_train, y_train, x_valid, y_valid)
        self.init_callbacks()
        print('Logging Info - Start training...')
        Y_train = to_categorical(y_train)
        Y_valid = to_categorical(y_valid)
        self.model.fit(x=x_train, y=Y_train, batch_size=self.config.batch_size,
                       epochs=self.config.n_epoch, validation_data=(
                           x_valid, Y_valid),
                       callbacks=self.callbacks)
        print('Logging Info - training end...')

    def predict(self, x):
        return self.model.predict(x)

    def predict_attention(self,x):
        neighbor_model = Model(inputs = self.model.input,outputs = [self.model.get_layer('receptive_filed_drug_one').output,self.model.get_layer('receptive_filed_drug').output])
        at_model = Model(inputs = self.model.input,outputs = [self.model.get_layer('cross_attention_0_one').output[1],self.model.get_layer('cross_attention_0').output[1]])
        
        return neighbor_model(x),at_model.predict(x)

    def score(self, x, y):
        y_true = y.flatten()
        y_pred = np.argmax(self.model.predict(x),1).flatten()
        acc = m.accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = m.f1_score(y_true=y_true, y_pred=y_pred,average='macro')
        return acc, f1
################### 
class SigmoidLayer(Layer):
    def __init__(self,name='sigmoid',**kwargs):
        super(SigmoidLayer,self).__init__(name = name,**kwargs)
    def call(self,x):
        return K.sigmoid(K.sum(x[0] * x[1], axis=-1, keepdims=True))

class KGCN(BaseModel):
    def __init__(self, config):
        super(KGCN, self).__init__(config)

    def build(self):
        input_drug_one = Input(
            shape=(1, ), name='input_drug_one', dtype='int64')
        input_drug_two = Input(
            shape=(1, ), name='input_drug_two', dtype='int64') 
           
        entity_embedding = Embedding(input_dim=self.config.entity_vocab_size,
                                     output_dim=self.config.ent_embed_dim,
                                     embeddings_initializer='glorot_normal',
                                     embeddings_regularizer=l2(
                                         self.config.l2_weight),
                                     name='entity_embedding')
        relation_embedding = Embedding(input_dim=self.config.relation_vocab_size,
                                       output_dim=self.config.ent_embed_dim,
                                       embeddings_initializer='glorot_normal',
                                       embeddings_regularizer=l2(
                                           self.config.l2_weight),
                                       name='relation_embedding')
        
        # Drug one
        get_receptive_field_one = GetReceptiveField(self.config,name='receptive_filed_drug_one')
        receptive_list_drug_one = get_receptive_field_one(input_drug_one)
        neineigh_ent_list_drug_one = receptive_list_drug_one[:self.config.n_depth+1]
        neigh_rel_list_drug_one = receptive_list_drug_one[self.config.n_depth+1:]
        
        neigh_ent_embed_list_drug_one = [entity_embedding(
            neigh_ent) for neigh_ent in neineigh_ent_list_drug_one]
        
        neigh_rel_embed_list_drug_one = [relation_embedding(
            neigh_rel) for neigh_rel in neigh_rel_list_drug_one]
        # Drug two
        get_receptive_field = GetReceptiveField(self.config,name='receptive_filed_drug')
        receptive_list = get_receptive_field(input_drug_two)
        neigh_ent_list = receptive_list[:self.config.n_depth+1]
        neigh_rel_list = receptive_list[self.config.n_depth+1:]
        
        neigh_ent_embed_list = [entity_embedding(
            neigh_ent) for neigh_ent in neigh_ent_list]
        
        neigh_rel_embed_list = [relation_embedding(
            neigh_rel) for neigh_rel in neigh_rel_list]

        
        #########
        #
        if self.config.c != 1 and self.config.c != 2:
            drug_embedding = Embedding(input_dim=self.config.entity_vocab_size,
                                    output_dim=self.config.ent_embed_dim,
                                    embeddings_initializer='glorot_normal',
                                    embeddings_regularizer=l2(
                                        self.config.l2_weight),
                                    name='drug_embedding')
            drug_embed_one = drug_embedding(input_drug_one)
            drug_embed_two = drug_embedding(input_drug_two)
        e_drug_one = neigh_ent_embed_list_drug_one[0]
        lc_list_one = [] 
        e_drug_two = neigh_ent_embed_list[0]
        lc_list_two = []
        
        #
        for depth in range(self.config.n_depth):
            #Attention
            #KGNN
            if self.config.c == 0:                             
                attention_scale_one = KGNN_base(
                    self.config,
                    name = f'cross_attention_{depth}_one')
                attention_scale = KGNN_base(
                    self.config,
                    name = f'cross_attention_{depth}') 
            #GAT-const
            if self.config.c == 1:                             
                attention_scale_one = GAT_const(
                    self.config,
                    name = f'cross_attention_{depth}_one')
                attention_scale = GAT_const(
                    self.config,
                    name = f'cross_attention_{depth}')
            #GAT
            if self.config.c == 2:                             
                attention_scale_one = GAT(
                    self.config, 
                    self.config.aggregator_type,
                    regularizer=l2(self.config.l2_weight),
                    name = f'cross_attention_{depth}_one')
                attention_scale = GAT(
                    self.config,
                    self.config.aggregator_type,      
                    regularizer=l2(self.config.l2_weight),
                    name = f'cross_attention_{depth}')
            #TBA
            if self.config.c == 3:                             
                attention_scale_one = TBA(
                    self.config,#head = 4,  
                    name = f'cross_attention_{depth}_one')
                attention_scale = TBA(
                    self.config,#head = 4,
                    name = f'cross_attention_{depth}')
            
            
                
            #
            if self.config.c != 2 :
                aggregator_one = Aggregator[self.config.aggregator_type](
                    activation='tanh' if depth == self.config.n_depth-1 else 'relu',
                    regularizer=l2(self.config.l2_weight),
                    name=f'aggregator_{depth+1}_drug_one'
                )
                aggregator = Aggregator[self.config.aggregator_type](
                    activation='tanh' if depth == self.config.n_depth-1 else 'relu',
                    regularizer=l2(self.config.l2_weight),
                    name=f'aggregator_{depth+1}_drug'
                )
            
            
            for hop in range(self.config.n_depth-depth):
                temp_one = neigh_ent_embed_list_drug_one[hop + 1]
                temp = neigh_ent_embed_list[hop + 1]
                
                if self.config.c == 0:
                    neighbor_embed_one = attention_scale_one([drug_embed_one,neigh_rel_embed_list_drug_one[hop],temp_one])
                    neighbor_embed = attention_scale([drug_embed_one,neigh_rel_embed_list[hop],temp])

                if self.config.c == 1:
                    neighbor_embed_one = attention_scale_one(temp_one)
                    neighbor_embed = attention_scale(temp)
                    
                if self.config.c == 2:
                    neigh_ent_embed_list_drug_one[hop] = attention_scale_one([neigh_ent_embed_list_drug_one[hop],temp_one])
                    neigh_ent_embed_list[hop] = attention_scale([neigh_ent_embed_list[hop],temp])
                
                if self.config.c == 3:
                    neighbor_embed_one,attention_value_one = attention_scale_one([drug_embed_two,temp_one])
                    neighbor_embed,attention_value = attention_scale([drug_embed_one,temp])  
                

                if self.config.c != 2 :
                    neigh_ent_embed_list_drug_one[hop] = aggregator_one([neigh_ent_embed_list_drug_one[hop], neighbor_embed_one])
                    neigh_ent_embed_list[hop] = aggregator([neigh_ent_embed_list[hop], neighbor_embed])
                
                
            if self.config.lc == 1:    
                #layer-wise 
                lc_list_one.append(neigh_ent_embed_list_drug_one[0])
                lc_list_two.append(neigh_ent_embed_list[0])
            
        
        #layer-wise 
        if self.config.lc == 1:
            e_drug_one = K.concatenate([e_drug_one,K.concatenate(lc_list_one)])
            e_drug_two = K.concatenate([e_drug_two,K.concatenate(lc_list_two)])
        else:
            e_drug_one = neigh_ent_embed_list_drug_one[0]
            e_drug_two = neigh_ent_embed_list[0]
        
            
        
        squeeze_layer = SqueezeLayer()
        drug1_squeeze_embed = squeeze_layer(e_drug_one)
        drug2_squeeze_embed = squeeze_layer(e_drug_two)
        sigmoid_layer = SigmoidLayer()
        
        drug_drug_score = sigmoid_layer([drug1_squeeze_embed, drug2_squeeze_embed])

        model = Model([input_drug_one, input_drug_two], drug_drug_score)
        model.compile(optimizer=self.config.optimizer,
                      loss='binary_crossentropy', metrics=['acc'])
        model.summary()
        return model
    

    def add_metrics(self, x_train, y_train, x_valid, y_valid):
        self.callbacks.append(KGCNMetric(x_train, y_train, x_valid, y_valid,
                                         self.config.aggregator_type, self.config.dataset, self.config.K_Fold))

    def fit(self, x_train, y_train, x_valid, y_valid):
        self.callbacks = []
        self.add_metrics(x_train, y_train, x_valid, y_valid)
        self.init_callbacks()
        print('Logging Info - Start training...')
        self.model.fit(x=x_train, y=y_train, batch_size=self.config.batch_size,
                       epochs=self.config.n_epoch, validation_data=(
                           x_valid, y_valid),
                       callbacks=self.callbacks)
        print('Logging Info - training end...')

    def predict(self, x):
        return self.model.predict(x).flatten()

    def predict_attention(self,x):
        neighbor_model = Model(inputs = self.model.input,outputs = [self.model.get_layer('receptive_filed_drug_one').output,self.model.get_layer('receptive_filed_drug').output])
        at_model = Model(inputs = self.model.input,outputs = [self.model.get_layer('cross_attention_0_one').output[1],self.model.get_layer('cross_attention_0').output[1]])
        
        return neighbor_model(x),at_model.predict(x)

    def score(self, x, y, threshold=0.5):
        y_true = y.flatten()
        y_pred = self.model.predict(x).flatten()
        auc = roc_auc_score(y_true=y_true, y_score=y_pred)
        p, r, t = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
        aupr = m.auc(r, p)
        y_pred = [1 if prob >= threshold else 0 for prob in y_pred]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)
        return auc, acc, f1, aupr
