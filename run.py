# -*- coding: utf-8 -*-
import sys
import random
import os
import numpy as np
from collections import defaultdict
sys.path.append(os.getcwd()) #add the env path
from sklearn.model_selection import train_test_split
from main import train,train_M
import argparse
from config import DRUG_EXAMPLE, RESULT_LOG, PROCESSED_DATA_DIR, LOG_DIR, MODEL_SAVED_DIR, KG_FILE, \
    EXAMPLE_FILE,  DRUG_VOCAB_TEMPLATE,TRAIN_DATA_TEMPLATE,DEV_DATA_TEMPLATE,TEST_DATA_TEMPLATE, ENTITY_VOCAB_TEMPLATE,\
    RELATION_VOCAB_TEMPLATE, SEPARATOR, RAW_DATA_DIR,ADJ_ENTITY_TEMPLATE, ADJ_RELATION_TEMPLATE, KFOLD_DATASET,ModelConfig
from utils import pickle_dump, pickle_load,format_filename,write_log

def read_example_file(file_path:str,separator:str,drug_vocab:dict):
    print(f'Logging Info - Reading example file: {file_path}')
    examples=[]
    
    with open(file_path,encoding='utf8') as reader:
        for idx,line in enumerate(reader):
            d1,d2,flag=line.strip().split(separator)[:3]
            if int(d1) not in drug_vocab:
                drug_vocab[int(d1)] = len(drug_vocab)
                if int(d2) not in drug_vocab:
                    drug_vocab[int(d2)] = len(drug_vocab)
            else: 
                if int(d2) not in drug_vocab:
                    drug_vocab[int(d2)] = len(drug_vocab)
            if int(d1) in drug_vocab and int(d2) in drug_vocab:
                examples.append([drug_vocab[int(d1)],drug_vocab[int(d2)],int(flag)])
    
    examples_matrix=np.array(examples)
    print(f'num of drugs: {len(drug_vocab)}')
    print(f'size of example: {examples_matrix.shape}')
    
    return examples_matrix

def read_kg(file_path: str, drug_vocab:dict, relation_vocab: dict, hop:int,neighbor_sample_size: int):
    print(f'Logging Info - Reading kg file: {file_path}')

    kg = defaultdict(list)
    with open(file_path, encoding='utf8') as reader:
            count=0
            for line in reader:
                if count==0:
                    count+=1
                    continue
                head, tail, relation = line.strip().split(' ') 
                kg[int(head)].append((int(tail), int(relation)))
                kg[int(tail)].append((int(head), int(relation)))
                if int(relation) not in relation_vocab:
                    relation_vocab[int(relation)] = len(relation_vocab)
    subgraphkg = defaultdict(list)
    condition_dict = {key:value for key,value in drug_vocab.items()}
    neighbor_dict = {}
    subgraphid_dict = {key:value for key,value in drug_vocab.items()}
    for i in range(hop):
        for key,value in kg.items():
            if key in condition_dict:
                n_neighbor = len(value)
                sample_indices = np.random.choice(
                n_neighbor,
                neighbor_sample_size,
                replace=False if n_neighbor >= neighbor_sample_size else True)
                for j in sample_indices:
                    if value[j][0] not in subgraphid_dict:
                        neighbor_dict[value[j][0]] = len(subgraphid_dict)
                        subgraphid_dict[value[j][0]] = len(subgraphid_dict)
                    subgraphkg[subgraphid_dict[key]].append((subgraphid_dict[value[j][0]],value[j][1]))
        condition_dict = {key:value for key,value in neighbor_dict.items()}
        neighbor_dict = {}
    print(f'Logging Info - num of subgraph entities: {len(subgraphid_dict)}, '
          f'num of relations: {len(relation_vocab)}'
          f'shape of adj_entity:{len(subgraphkg)}')

    print('Logging Info - Constructing adjacency matrix...')
    n_entity = len(subgraphkg)
    adj_entity = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)
    adj_relation = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)
    for entity_value in range(n_entity):
        all_neighbors = subgraphkg[entity_value]
        n_neighbor = len(all_neighbors)
        adj_entity[entity_value] = np.array([all_neighbors[i][0] for i in range(n_neighbor)])
        adj_relation[entity_value] = np.array([all_neighbors[i][1] for i in range(n_neighbor)])

    return adj_entity, adj_relation,subgraphid_dict


def process_data(K:int,dataset:str,N:int,HOP:int,B:int,LR:float,ND:int,LC:int,C:int,R:int,S:int,MulHead:int):
    drug_vocab = {}
    relation_vocab = {}
    total_examples = []
   
    if dataset == 'kegg':
        examples_file=format_filename(PROCESSED_DATA_DIR, DRUG_EXAMPLE, dataset=dataset)
        examples = read_example_file(EXAMPLE_FILE[dataset], SEPARATOR[dataset],drug_vocab)
        np.save(examples_file,examples)
        total_examples.append(examples)
    elif dataset == 'drugbank':
        train_examples_file=format_filename(PROCESSED_DATA_DIR, TRAIN_DATA_TEMPLATE, dataset=dataset)
        train_examples = read_example_file(os.path.join(RAW_DATA_DIR,dataset,'train.txt'), SEPARATOR[dataset],drug_vocab)
        np.save(train_examples_file,train_examples)
        total_examples.append(train_examples)
        
        test_examples_file=format_filename(PROCESSED_DATA_DIR, TEST_DATA_TEMPLATE, dataset=dataset)
        test_examples = read_example_file(os.path.join(RAW_DATA_DIR,dataset,'test.txt'), SEPARATOR[dataset],drug_vocab)
        np.save(test_examples_file,test_examples)
        total_examples.append(test_examples)

        dev_examples_file=format_filename(PROCESSED_DATA_DIR, DEV_DATA_TEMPLATE, dataset=dataset)
        dev_examples = read_example_file(os.path.join(RAW_DATA_DIR,dataset,'dev.txt'), SEPARATOR[dataset],drug_vocab)
        np.save(dev_examples_file,dev_examples)
        total_examples.append(dev_examples)
    pickle_dump(format_filename(PROCESSED_DATA_DIR, DRUG_VOCAB_TEMPLATE, dataset=dataset),
                drug_vocab)
    
    adj_entity_file = format_filename(PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE, dataset=dataset,HOP = HOP,N = N)
    adj_relation_file = format_filename(PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE, dataset=dataset,HOP = HOP,N = N)
    subgraph_dict_file = format_filename(PROCESSED_DATA_DIR, ENTITY_VOCAB_TEMPLATE, dataset=dataset,HOP = HOP,N = N)
    relation_dict_file = format_filename(PROCESSED_DATA_DIR, RELATION_VOCAB_TEMPLATE, dataset=dataset,HOP = HOP,N = N)
    if not os.path.exists(adj_entity_file) or not os.path.exists(adj_relation_file) or not os.path.exists(subgraph_dict_file) or not os.path.exists(relation_dict_file):
        adj_entity, adj_relation,subgraphid_dict = read_kg(KG_FILE[dataset], drug_vocab, relation_vocab,HOP,N)
        pickle_dump(subgraph_dict_file,subgraphid_dict)
        pickle_dump(relation_dict_file,relation_vocab)
        
        np.save(adj_entity_file, adj_entity)
        print('Logging Info - Saved:', adj_entity_file)
        
        np.save(adj_relation_file, adj_relation)
        print('Logging Info - Saved:', adj_relation_file)
    
    
    cross_validation(K,total_examples,dataset,N,HOP,B,LR,ND,LC,C,R,S,MulHead)

def cross_validation(K_fold,total_examples,dataset,neighbor_sample_size,HOP,B,LR,ND,LC,C,R,S,MulHead):
    if dataset == 'kegg':
        examples = total_examples[0]
        dataset_file = format_filename(PROCESSED_DATA_DIR, KFOLD_DATASET, dataset=dataset)
        if S == 1: #test the generalization ability of the model in the cold start 
            if not os.path.exists(dataset_file):
                L = len(examples)#examples为numpy转换过的多重列表
                drug_dict = defaultdict(list)
                for i in range(L):
                    drug_dict[examples[i][0]].append(i)
                    drug_dict[examples[i][1]].append(i)
                D = len(drug_dict)
                cold_number = int(D*0.2)

                cold_drug = random.sample(set(drug_dict.keys()),cold_number)
                
                warm_drug = set(drug_dict.keys()).difference(cold_drug)
                
                warm_example_index = []
                for drug in warm_drug:
                    for j in drug_dict[drug]:
                        if examples[j][0] not in warm_drug or examples[j][1] not in warm_drug:
                            drug_dict[drug].remove(j)
                    
                    warm_example_index += drug_dict[drug]
                warm_example_index = list(set(warm_example_index))
                print(len(warm_example_index))

                total_example_index = [warm for warm in warm_example_index]
                cold_drug_subsets = dict() 
                cold_example_index = defaultdict(list)
                for k in range(5):
                    cold_drug_subsets[k] = random.sample(cold_drug,int(len(cold_drug)/5))
                    temp = list(warm_drug)+list(cold_drug_subsets[k])
                    for drug in cold_drug_subsets[k]:
                        for index_c in drug_dict[drug]:
                            if examples[index_c][0] not in temp or examples[index_c][1] not in temp:
                                drug_dict[drug].remove(index_c)
                        cold_example_index[k] += drug_dict[drug]

                    cold_example_index[k] = list(set(cold_example_index[k]))
                    total_example_index += cold_example_index[k]
                    cold_drug = list(set(cold_drug).difference(cold_drug_subsets[k]))
                n_total = len(total_example_index)
                print(n_total)
                
                subsets = dict()
                n_subsets = int(n_total/5)
                print(n_subsets)
                for l in range(5):
                    warm_number = n_subsets - len(cold_example_index[l])
                    print(warm_number)
                    warm_index_list = random.sample(warm_example_index,warm_number)
                    subsets[l] = warm_index_list + cold_example_index[l]
                    warm_example_index = list(set(warm_example_index).difference(warm_index_list))
                pickle_dump(dataset_file,subsets)
            else:
                subsets = pickle_load(dataset_file)
        else:
            #Randomly divide the dataset example into 5 parts
            subsets=dict()
            n_subsets=int(len(examples)/5)
            remain=set(range(0,len(examples)-1))
            for i in reversed(range(0,4)):
                subsets[i]=random.sample(remain,n_subsets)
                remain=remain.difference(subsets[i])
            subsets[4]=remain

    #aggregator_types=['neigh','concat']
    aggregator_types=['neigh']
    for t in aggregator_types:
        count=1
        auc_std = []
        aupr_std = []
        acc_std = []
        f1_std = []
        if dataset == 'kegg':
            temp={'dataset':(HOP,neighbor_sample_size,ND,C,LC),'aggregator_type':t,'avg_auc':0.0,'avg_acc':0.0,'avg_f1':0.0,'avg_aupr':0.0,'auc_std':0.0,'acc_std':0.0,'f1_std':0.0,'aupr_std':0.0}
        else:
            temp={'dataset':(HOP,neighbor_sample_size,ND,C,LC),'aggregator_type':t,'avg_acc':0.0,'avg_f1':0.0,'acc_std':0.0,'f1_std':0.0}
        for i in reversed(range(0,K_fold)):
            if dataset == 'kegg':
                test_d=examples[list(subsets[i])]
                train_d=[]
                val_d,test_data=train_test_split(test_d,test_size=0.5)#Randomly select validation set data and test set data
                for j in range(0,5):
                    if i!=j :
                        train_d.extend(examples[list(subsets[j])])
                train_data=np.array(train_d)  
            if dataset == 'kegg':           
                train_log=train(
                kfold=count,
                dataset=dataset,
                train_d= train_data,
                dev_d= val_d,
                test_d= test_data,
                neighbor_sample_size=neighbor_sample_size,
                ent_embed_dim = ND,
                n_depth=HOP,
                l2_weight=1e-7,
                lr=LR,#1e-2/5e-3
                optimizer_type='adam',
                batch_size=B,#1024
                aggregator_type=t,
                n_epoch=50,
                callbacks_to_add=['modelcheckpoint', 'earlystopping'],
                lc = LC,
                c = C,
                r = R,
                head = MulHead
                )  
            else:
                train_log=train_M(
                kfold=count,
                dataset=dataset,
                train_d=total_examples[0] ,
                dev_d=total_examples[2] ,
                test_d=total_examples[1] ,
                neighbor_sample_size=neighbor_sample_size,
                ent_embed_dim = ND,
                n_depth=HOP,
                l2_weight=1e-7,
                lr=LR,#1e-2/5e-3
                optimizer_type='adam',
                batch_size=B,#1024
                aggregator_type=t,
                n_epoch=50,
                callbacks_to_add=['modelcheckpoint', 'earlystopping'],
                lc = LC,
                c = C,
                r = R,
                head = MulHead
                )   
            count+=1
            temp['avg_acc']=temp['avg_acc']+train_log['test_acc']
            temp['avg_f1']=temp['avg_f1']+train_log['test_f1']
            acc_std.append(train_log['test_acc'])
            f1_std.append(train_log['test_f1'])
            if dataset == 'kegg':
                temp['avg_auc']=temp['avg_auc']+train_log['test_auc']
                temp['avg_aupr']=temp['avg_aupr']+train_log['test_aupr']
                auc_std.append(train_log['test_auc'])
                aupr_std.append(train_log['test_aupr'])
        
        for key in temp:
            if key=='aggregator_type' or key=='dataset':
                continue
            temp[key]=temp[key]/K_fold
        temp['acc_std'] = np.std(np.array(acc_std))
        temp['f1_std'] = np.std(np.array(f1_std))
        if dataset == 'kegg':
            temp['auc_std'] = np.std(np.array(auc_std))
            temp['aupr_std'] = np.std(np.array(aupr_std))
        write_log(format_filename(LOG_DIR, RESULT_LOG[dataset]),temp,'a')
        if dataset == 'kegg':
            print(f'Logging Info - {K_fold} fold result: avg_auc: {temp["avg_auc"]}, avg_acc: {temp["avg_acc"]}, avg_f1: {temp["avg_f1"]}, avg_aupr: {temp["avg_aupr"]}')
        else:
            print(f'Logging Info - {K_fold} fold result:  avg_acc: {temp["avg_acc"]}, avg_f1: {temp["avg_f1"]}')
   
if __name__ == '__main__':
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(MODEL_SAVED_DIR):
        os.makedirs(MODEL_SAVED_DIR)
    parser = argparse.ArgumentParser()
    parser.description = 'please enter parameters or use defalut values'
    parser.add_argument("-d","--D",help="Choose which dataset to use, the default is kegg",default="kegg")
    parser.add_argument("-n","--N",help="Select the number of neighbor samples for each node, the default is 16",type=int,default="16")
    parser.add_argument("-hop","--HOP",help="Select the depth of neighbor sampling, default is 1",type=int,default="1")
    parser.add_argument("-b","--B",help="The value of batchsize for each epoch of training, the default is 1024",type=int,default="1024")
    parser.add_argument("-lr","--LR",help="The value of lr for each epoch of training, the default is 1e-2",type=float,default="1e-2")
    parser.add_argument("-nd","--ND",help="The value of node dimension , the default is 64",type=int,default="64")

    parser.add_argument("-lc","--LC",help="this parameter decide whether to use Layer-wise concat, the default is 1",type=int,default="1")
    parser.add_argument("-c","--C",help="this parameter decide which cross-attention layer to be used, the default is 3",type=int,default="3")
    parser.add_argument("-r","--R",help="this parameter determines whether to export the test results for visualization, the default is 0",type=int,default="0")
    parser.add_argument("-s","--S",help="this parameter determines whether to test the generalization ability of the model in the cold start scenario, the default is 0",type=int,default="0")
    parser.add_argument("-head","--MulHead",help="this parameter decide num of Multi-head attention,default is 1",type=int,default="1")
    args = parser.parse_args()
    model_config = ModelConfig()
    process_data(5,args.D,args.N,args.HOP,args.B,args.LR,args.ND,args.LC,args.C,args.R,args.S,args.MulHead)



