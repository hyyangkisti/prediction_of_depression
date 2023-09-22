# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 08:57:37 2023
main page for codes
the core codes
@author: Minyoung Yun Korea Institute of Science and Technology Information
"""

import pandas as pd
import numpy as np
import pyreadr
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import stellargraph as sg
from stellargraph.mapper import PaddedGraphGenerator
from tensorflow.keras.optimizers import Adam, RMSprop
from stellargraph.layer import DeepGraphCNN
from stellargraph import StellarGraph
from sklearn import model_selection
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

def excel_import():
    #address to export file from
    ads = r'C:\Users\user\Desktop\Legacy\Project\OECD_EDU\depression\pone.0060188.s004.xlsx'
    w_excel0 = pd.read_excel(ads, sheet_name=1) 
    return w_excel0

def find_6th(subjno_u,feel):
    #find the 6th day!
    days = np.zeros([len(subjno_u),2])
    for idx1,ii1 in enumerate(subj_list):
        cnt0 = 0
        for ii2 in range(10): # 10days
            chk_np = feel[ii1[ii2*11: (ii2+1)*11],:] #10 beeping points
            
            if np.any(chk_np[:-1,:] > -4): # lowest is -4 #str(int(subj_mark[73][1])).isnumeric()
                cnt0 += 1
            
            if cnt0 > 5:
                break
                    
        days[idx1,0] = subjno_u[idx1]
        days[idx1,1] = ii2        
    return days    

def binary_f(days, subj_list, fct0):
    subj_mark = np.zeros([len(subj_list),5])
    nan_list=[]
    for ii6 in range(len(subj_list)):
        subj = subj_list[ii6]
        no_6 = days[ii6,1]
        Fived = feel[subj[0:((int(no_6)-1)+1)*11],2:5] #5day data
        lastd = feel[subj[int(no_6)*11: (int(no_6)+1)*11],2:5]#last day data
        
        fived_avg = [np.nanmean(Fived[:,j]) for j in range(3)]; f_avg_t = sum(fived_avg)
        lastd_avg = [np.nanmean(lastd[:,j]) for j in range(3)]; l_avg_t = sum(lastd_avg)
        
        factor = fct0
        if f_avg_t > factor*l_avg_t:
            mark = 1;  #higher negative feelings   
        elif f_avg_t <= factor*l_avg_t:
            mark = 0;
        else:
            mark = 0; nan_list.append(ii6)#same or less neg feelings
        
        subj_mark[ii6,0] = days[ii6,0]
        subj_mark[ii6,1] = f_avg_t
        subj_mark[ii6,2] = l_avg_t
        subj_mark[ii6,3] = mark    
        
        # count the depressing cases (6th > 1~5th avg) 
        len(np.where(subj_mark[:,3]>0)[0])
        
    return subj_mark, nan_list


def ppl_feel_avg5(subj_list,feel):
    subj_mark = np.zeros([len(subj_list),5])
    ppl_feel_avg5 = np.zeros((len(subj_list), 6))
    for ii6 in range(len(subj_list)):
        subj = subj_list[ii6]
        no_6 = days[ii6,1]
        Fived = feel[subj[0:((int(no_6)-1)+1)*11],:] #5day data
        
        fived_avg = [np.nanmean(Fived[:,j]) for j in range(6)]
        ppl_feel_avg5[ii6,:] = fived_avg
        
    return ppl_feel_avg5


class AdditionalValidationSets(tf.keras.callbacks.Callback):
    def __init__(self, validation_sets, verbose=0, batch_size=None):
        """
        :param validation_sets:
        a list of 3-tuples (validation_data, validation_targets, validation_set_name)
        or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        #for validation_set in self.validation_sets:
        #    if len(validation_set) not in [3, 4]:
        #        raise ValueError()
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)

        # record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # evaluate on the additional validation sets
        validation_data, validation_targets = self.validation_sets
        #results = self.model.evaluate(validation_data,validation_targets, verbose=self.verbose, batch_size=self.batch_size)
        results = self.model.evaluate(validation_targets, verbose=self.verbose, batch_size=self.batch_size)
        #print(results)
        

def graph_CNN_norun(graphdb, graphlabels, no_raw = 126):
    graphs, graph_labels = graphdb, graphlabels #including the raw data
    
    summary = pd.DataFrame(
        [(g.number_of_nodes(), g.number_of_edges()) for g in graphs],
        columns=["nodes", "edges"],
    )
    summary.describe().round(1)
    
    graph_labels = pd.get_dummies(graph_labels, drop_first=True)
    generator = PaddedGraphGenerator(graphs=graphs)
    
    #First we create the base DGCNN model that includes the graph convolutional and SortPooling layers.
    k = 35  # the number of rows for the output tensor
    layer_sizes = [32, 32, 32, 1]
    
    dgcnn_model = DeepGraphCNN(
        layer_sizes=layer_sizes,
        activations=["tanh", "tanh", "tanh", "tanh"],
        k=k,
        bias=False,
        generator=generator,
    )
    x_inp, x_out = dgcnn_model.in_out_tensors()
    
    
    #Next, we add the convolutional, max pooling, and dense layers.
    x_out = Conv1D(filters=16, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
    x_out = MaxPool1D(pool_size=2)(x_out)
    x_out = Conv1D(filters=32, kernel_size=5, strides=1)(x_out)
    x_out = Flatten()(x_out)
    x_out = Dense(units=128, activation="relu")(x_out)
    x_out = Dropout(rate=0.5)(x_out)
    predictions = Dense(units=1, activation="sigmoid")(x_out)
    
    
    #Finally, we create the Keras model and prepare it for training by specifying the loss and optimisation algorithm.
    model = Model(inputs=x_inp, outputs=predictions)
    
    # loss function : defined, built-in binary_crossentropy
    model.compile(
        optimizer=RMSprop(learning_rate=0.001), loss='mean_squared_error', metrics=["acc"],
    )
    

    train_graphs, test_graphs = model_selection.train_test_split(
        graph_labels, train_size=0.8, test_size=None, stratify=graph_labels,
    )
      
    gen = PaddedGraphGenerator(graphs=graphs)
    
    train_gen = gen.flow(
        list(train_graphs.index - 1),
        targets=train_graphs.values,
        batch_size=50, #batch_size=50,
        symmetric_normalization=False,
    )
    
    test_gen = gen.flow(
        list(test_graphs.index - 1),
        targets=test_graphs.values,
        batch_size=1,
        symmetric_normalization=False,
    )
    
    #We can now train the model by calling it’s fit method.
    history = model.fit(
        train_gen, epochs=500, verbose=1, validation_data=test_gen, shuffle=True #, callbacks = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta = 0.0001, patience = 10)
    ) 
    sg.utils.plot_history(history)
    
    return train_gen, test_gen, test_graphs, model

def evaluate_preds(test_gen,test_graphs,model):
    pred = model.predict(test_gen)
    #bin_pred = [1 if p > np.mean(pred) else 0 for p in pred]
    bin_pred = [1 if p > 0.5 else 0 for p in pred]
    true = test_graphs.values
    true[true==-1] = 0

    f_score = f1_score(true, bin_pred)    
    auc = roc_auc_score(true, bin_pred)
    pr = average_precision_score(true, bin_pred)    
    
    return auc, pr, f_score

def evaluate_raw(test_gen,test_graphs,model, no_raw):
    pred = model.predict(test_gen)
    #bin_pred = [1 if p > np.mean(pred) else 0 for p in pred]
    bin_pred = [1 if p > 0.5 else 0 for p in pred]
    bin_pred = np.array(bin_pred)
    true = test_graphs.values #true[true==-1] = 0
    this_raw = np.where(np.array(test_graphs.index)<no_raw)
    
    true1 = true[this_raw]; bin_pred1 = bin_pred[this_raw]
    f_score = f1_score(true1, bin_pred1)    
    try:
        auc = roc_auc_score(true1, bin_pred1)
    except:
        auc = 0
    pr = average_precision_score(true1, bin_pred1)    
      
    return auc, pr, f_score


"main code"
w_excel0 = excel_import()

#find index of the values to be replaced
subjno_u = w_excel0["subjno"].unique() #subject no.
subjno =  w_excel0.iloc[:,1].to_numpy()
#select the columns with the depression data
feel = w_excel0.iloc[:,6:12].to_numpy() #from 6 to 11 cheerful to relaxed

#find index per each subjno
subj_list = []
for ii0 in subjno_u:
    temp0 = w_excel0[(w_excel0['st_period']==0) & (w_excel0['subjno']==ii0)].index
    subj_list.append(temp0)

#find the 6th day!
days = find_6th(subjno_u,feel)
#average feeling for each person
ppl_feel_avg5 = ppl_feel_avg5(subj_list,feel)

#binary classification mark
subj_mark, nan_list = binary_f(days, subj_list, fct0=1)

#import the matrix info (to be used as edge weight)
results = pyreadr.read_r(r'C:\Users\user\Desktop\Legacy\Project\OECD_EDU\depression\edited0.rds')

# create graphlabels and input for keras graph cnn
repeat = 150 #repeat 150 times
howgood = np.zeros([repeat,6]) # 6 because 3 for raw, 3 for test set
for ii in range(repeat):
    print(ii)
    s_size = 50
    #data augmentation
    s_chr = ['a','b','c','d','e','f']
    
    d = []
    for p in s_chr:
        for q in s_chr:
            d.append(
                {
                    'source': p,
                    'target': q,
                }
            )
    
    edge_templ = pd.DataFrame(d)
    
    subj_mark1 =  np.delete(subj_mark, nan_list,0) #remove if any (1th~5th, 6th) is nan
    ppl_feel_avg5_1 = np.delete(ppl_feel_avg5, nan_list,0) #ppl_feel_avg5 from 'edit6thdata.py'
    graphlabels0 = subj_mark1[:,3]; graphlabels = np.empty([0,0])
    
    # stds for each component of 6x6 matrix
    edge_weight = results[None].to_numpy()          
            
    graphdb = []
    for itr in range(s_size): # no of aumentation
        edge = edge_templ
        graphlabels = np.append(graphlabels, graphlabels0)
        
        for idx, i in enumerate(subj_mark1): #each person
            #print(idx)
            node_0 = {'f1' : ppl_feel_avg5_1[idx,:] ,'keyword' : ['a','b','c','d','e','f']}
            node = pd.DataFrame(data = node_0)
            node = node.set_index('keyword') 
            
            edge_weight1 = edge_weight[:,:,idx]            
            if itr > 0:          
                aug_gauss_std = np.zeros([6,6])
                for i1 in range(6):
                    for j1 in range(6):                      
                        this0 = abs(0.01*edge_weight1[i1,j1]) #1% variation
                        this_edge_w1 = edge_weight1[i1,j1]
                        aug_gauss_std[i1,j1] = random.uniform(this_edge_w1-this0,this_edge_w1+this0)
                              
                #gaussian nosie augmentation
                edge_weight1 = aug_gauss_std           
                           
            #edge insert
            if 'weight' in edge: edge = edge.drop(["weight"], axis='columns') #clean slate
            edge.insert(2, "weight", np.reshape(edge_weight1,[6*6,1]))
            
            graphdb.append(StellarGraph(node, edges=edge))  
            
    # turn samples into a runnabble form
    train_gen, test_gen, test_graphs, model = graph_CNN_norun(graphdb, graphlabels, no_raw = 126)
    
    #prediction efficiency indicators
    #auc = roc_auc_score, pr = average_precision_score, f_score = f1_score
    auc, pr, f_score = evaluate_preds(test_gen,test_graphs,model) #with augmented data
    auc_r, pr_r, f_score_r = evaluate_raw(test_gen,test_graphs,model, no_raw=126) #raw data only
    howgood[ii,:] = [auc,pr, f_score, auc_r, pr_r, f_score_r]

#remove nan in the array
howgood[np.isnan(howgood)] = 0 
fnl_mean = [np.mean(howgood[:,j]) for j in range(6)] #average the values