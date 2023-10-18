import os
import random
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import sfit as s_fit

import logging
tf.get_logger().setLevel(logging.ERROR)

def get_names(x,variable_names):
    pass

def plot_histogram(dic, variable_names, label=''):
    variable_names = variable_names[1:]
    
    x = []
    values = []
    
    for i,key in enumerate(dic):
        x.append(i+1)
        values.append(dic[key])
    
    names = [str(key) for key in dic]#get_names(x,variable_names)
    
    plt.figure()
    plt.bar(x,values, label=label)
    plt.xlim([0.5,len(x)+0.5])
    plt.xticks(x, names, rotation=45)
    plt.legend()

def sorted_keys(dic):
    return sorted(sorted(dic.keys()))

def Merge(dict_1, dict_2):
    result = dict_1 | dict_2
    return result

def unique_key(key):
    a, b = sorted(key)
    return (a,b)    

def reducer(dic,norm):
    reduced_second_order = {}
    for betas in dic:
        aux={}
        for keys in dic[betas]:
            new_key = unique_key(keys)
            try:
                aux[new_key] += dic[betas][keys]
            except:
                aux[new_key] = 0
                aux[new_key] += dic[betas][keys]
        
        reduced_second_order[betas] = {key: (aux[key]/norm if aux[key]/norm<1 else 1) for key in sorted_keys(aux)} ## solucion cutre!
        
    return reduced_second_order

def histogram_data(data, beta_vec, norm):    
    first_order = {}
    second_order = {}
    for beta in beta_vec:
        first_order[beta] = []
        second_order[beta] = {}
    
    for d in data:
        for keys in beta_vec:
            if keys>0.0:
                first_order[keys].append(d[2][keys])
                if keys in d[3]:
                    for pair in d[3][keys]:
                        try:
                            second_order[keys][pair]+=1
                        except:
                            second_order[keys][pair]=1

    for keys in first_order:
        if keys>0.0:
            first_order[keys] = {i+1: val for i, val in enumerate(np.divide(np.sum(first_order[keys],axis=0),norm))}
    
    second_order = reducer(second_order,norm)
    
    return first_order, second_order

def get_r2(vec):
    ''' Compute the r^2 index from the result data structure
    '''
    res = []
    for i in range(len(vec)):
        res.append([])
    
    for r in vec:
        for v,aa in zip(res,r):
            v.append(round(aa[2],3))
    return res

def beta_tester(X, y, k_folds = 1, n_runs = 1, layers = [[5]], alpha = 0.05):
    ''' Runs a linear (OLS) sm model and as many NN models as we indicate at layers
    
    Parameters
    ----------
    X: numpy array of shape (N, p)
        Input data used to perform the tests
    y: numpy array of shape (N, )
        True outcomes
    k_folds: int (default 5)
        number of k-folds for the cross-validation
    n_runs: int (default 10)
        number of independent runs at different cores
    layers: list of lists
        the number of interior list represent the number of different NN models to be trained. The inner lists represent the number of
        layers and neuron at each layer. if layers=[], only a OLS will be trainned.
    beta_vec: list of float numbers (between 0 and 1)
        list of regularizations. The test will be repeated for each regularization.
    variable_names: list of names for the variables.        
        
    Returns
    -------
    final_return: list of lists. 
        The first element corresponds to the OLS results [model, r^2, loss]. Then, for each NN model, the function 
        returns [[x,y], r^2, loss], where [x,y] are the data arranged to be plotted as a heatmap.
    
    '''
    
    index = list(range(len(y)))
    random.shuffle(index)
      
    index_ = kfolds(index,k_folds)
      
    for layer in layers:
        results = Parallel(n_jobs=-1, backend='loky')(delayed(get_optimal_beta)(X, 
                                                                          y, 
                                                                          index=index_,
                                                                          i_test=i%k_folds,
                                                                          layers=layer,
                                                                          alpha=alpha)  for i in range(0, n_runs))
        
    return beta_dics(results)


def get_optimal_beta(X, y, index, i_test, layers, batch_size = 16, nr_epochs = 100, alpha=0.05):
    ''' Fits a NN model and perform the sfit test for each value in the list of beta_vec (regularization)
    
    Parameters
    ----------
    X: numpy array of shape (N, p)
        Input data used to perform the tests
    y: numpy array of shape (N, )
        True outcomes
    index: list of lists
        list that contain k-list with the indexes of the datapoints pertaining at each fold
    i_test: int
        integer that indicate the position of the fold used for testing in the index parameter
    layers: list of integers
        each element of the list, N, represent a layer with N neurons
    batch_size: integer > 0
        batch size used for training
    nr_epochs: integer > 0
        maximum number of epochs during training
    
    Returns
    -------
    optimal_beta: float
        optimal beta value for the trained model
    '''
    
    dim = np.shape(X)[1]
    X = np.asarray(X)
    
    # define the test group from the folds
    X_test = X[index[i_test]]
    y_test = y[index[i_test]]
    
    X_train, y_train = np.empty((0, dim)), np.empty((0))
    
    # the rest of the datapoints are used to train the model
    for j,ind in enumerate(index):
        if j!=i_test:
            X_train, y_train = np.vstack((X_train, X[ind])), np.append(y_train, y[ind])
    
    # using the information in 'layers', the arquitecture of the NN model is defined
    inputs = Input(shape=(dim,))
    prev_layer = inputs
    for layer in layers:
        hidden = Dense(layer, activation='relu')(prev_layer)
        prev_layer = hidden
        
    output = Dense(1, activation='linear')(prev_layer)
    
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=int(nr_epochs*0.15))
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error',run_eagerly=True)
    model.fit(x=X_train,
              y=y_train,
              batch_size=batch_size,
              epochs=nr_epochs,
              validation_split=0.1,
              callbacks=[early_stop],
              verbose=0)
    
    sfit_NN = s_fit.sfit_first_order(model=model,
                                   loss=s_fit.absolute_loss,
                                   alpha=alpha,
                                   beta=-1,
                                   x=X_test,
                                   y=y_test,
                                   verbose = False)
    return sfit_NN[5] #optimal_beta

def beta_dics(opt_beta):
    betas={}
    for beta in sorted(opt_beta):
        try:
            betas[beta] += 1
        except:
            betas[beta] = 1
    return betas

def different_models(X, y, k_folds = 5, n_runs = 10, layers = [[10]], 
                     beta_vec=[2e-2], alpha = 0.05, variable_names = [], second_order_bool=True):
    ''' Runs a linear (OLS) sm model and as many NN models as we indicate at layers
    
    Parameters
    ----------
    X: numpy array of shape (N, p)
        Input data used to perform the tests
    y: numpy array of shape (N, )
        True outcomes
    k_folds: int (default 5)
        number of k-folds for the cross-validation
    n_runs: int (default 10)
        number of independent runs at different cores
    layers: list of lists
        the number of interior list represent the number of different NN models to be trained. The inner lists represent the number of
        layers and neuron at each layer. if layers=[], only a OLS will be trainned.
    beta_vec: list of float numbers (between 0 and 1)
        list of regularizations. The test will be repeated for each regularization.
    variable_names: list of names for the variables.        
        
    Returns
    -------
    final_return: list of lists. 
        The first element corresponds to the OLS results [model, r^2, loss]. Then, for each NN model, the function 
        returns [[x,y], r^2, loss], where [x,y] are the data arranged to be plotted as a heatmap.
    
    '''
    
    if variable_names==[]: # we assign a name by its possition in case names were not provided
        for i in range(np.shape(X)[1]-1):
            variable_names.append(str(i))
   
    # Fit linear regression:
    # all data is used in the OLS fitting
    model = sm.OLS(y,X)
    model_lin = model.fit()
    linear_r2 = round(model_lin.rsquared,3)
    linear_sign = get_index_pvalues(model_lin,alpha)
    linear_loss = mean_squared_error(y, model_lin.predict(X))

    final_return = [[model_lin,linear_sign,linear_r2, linear_loss]]
    
    index = list(range(len(y)))
    random.shuffle(index)
      
    index_ = kfolds(index,k_folds)
      
    for layer in layers:
        results = Parallel(n_jobs=-1, backend='loky')(delayed(single_run)(X, 
                                                                          y, 
                                                                          index=index_,
                                                                          i_test=i%k_folds,
                                                                          layers=layer, 
                                                                          beta_vec=beta_vec,
                                                                          second_order_bool=second_order_bool)  for i in range(0, n_runs))
        
        #res = heat_map(results, beta_vec = beta_vec, norm = n_runs, variable_names = variable_names)
        res = []
        res.append(np.mean(np.transpose(results)[1])) # add r2_mean
        res.append(np.mean(np.transpose(results)[0])) # add base_lose
        final_return.append(res)
    
    return final_return, results


def kfolds(index,k):
    ''' support funtion to re-arrange the indexs of the folds for the cross-validation'''
    aux = int(len(index)/k)
    folds = []
    for j in range(k-1):
        folds.append(index[j*aux:(j+1)*aux])
    folds.append(index[(k-1)*aux:])
    
    return folds        

def single_run(X, y, index, i_test, layers, batch_size = 16, nr_epochs = 100, beta_vec = [1e-2], second_order_bool=False):
    ''' Fits a NN model and perform the sfit test for each value in the list of beta_vec (regularization)
    
    Parameters
    ----------
    X: numpy array of shape (N, p)
        Input data used to perform the tests
    y: numpy array of shape (N, )
        True outcomes
    index: list of lists
        list that contain k-list with the indexes of the datapoints pertaining at each fold
    i_test: int
        integer that indicate the position of the fold used for testing in the index parameter
    layers: list of integers
        each element of the list, N, represent a layer with N neurons
    batch_size: integer > 0
        batch size used for training
    nr_epochs: integer > 0
        maximum number of epochs during training
    beta_vec: list of float numbers (between 0 and 1)
        list of regularizations. The test will be repeated for each regularization.
    
    Returns
    -------
    base_loss: float
        loss computed over the test sample after training
    r2: float between 0 and 1
        r^2 index
    dic: dictionary
        dictionary contianing, for each regularization value, a vector of 1s for significant variables and 0s for non significant once.    
    '''
    
    dim = np.shape(X)[1]
    X = np.asarray(X)
    
    # define the test group from the folds
    X_test = X[index[i_test]]
    y_test = y[index[i_test]]
    
    X_train, y_train = np.empty((0, dim)), np.empty((0))
    
    # the rest of the datapoints are used to train the model
    for j,ind in enumerate(index):
        if j!=i_test:
            X_train, y_train = np.vstack((X_train, X[ind])), np.append(y_train, y[ind])
    
    # using the information in 'layers', the arquitecture of the NN model is defined
    inputs = Input(shape=(dim,))
    prev_layer = inputs
    for layer in layers:
        hidden = Dense(layer, activation='relu')(prev_layer)
        prev_layer = hidden
        
    output = Dense(1, activation='linear')(prev_layer)
    
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=int(nr_epochs*0.15))
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error',run_eagerly=True)
    model.fit(x=X_train,
              y=y_train,
              batch_size=batch_size,
              epochs=nr_epochs,
              validation_split=0.1,
              callbacks=[early_stop],
              verbose=0)
    
    # for each regularization value, a sfit test is performed and the significant variables saved to perform statistics 
    dic = {}
    dic_sec = {}    
    for beta in beta_vec:
        sfit_NN = s_fit.sfit_first_order(model=model,
                                       loss=s_fit.absolute_loss,
                                       alpha=0.05,
                                       beta=beta,
                                       x=X_test,#.to_numpy(), 
                                       y=y_test,
                                       verbose = False)
        optimal_beta = sfit_NN[5]
        if sfit_NN[4] and second_order_bool:
            sfit_NN_sec = s_fit.sfit_second_order(model=model,
                                       loss=s_fit.absolute_loss,
                                       alpha=0.05,
                                       beta=optimal_beta,
                                       x=X_test,#.to_numpy(), 
                                       y=y_test,
                                       s_1=sfit_NN[0],
                                       u_1=sfit_NN[2],
                                       verbose = False)
            #dic_sec[beta] = one_hot_significant(sfit_NN_sec[0], dim)
            dic_sec[optimal_beta] = [key for key in sfit_NN_sec[1]]
        
        dic[optimal_beta] = one_hot_significant(sfit_NN[0], dim)
    base_loss = mean_squared_error(y_test, model.predict(X_test, verbose=0))
    r2 = r_index(X_train, y_train, model)
    return base_loss, r2, dic, dic_sec

def heat_map(results, beta_vec, norm, variable_names):
    '''generates the data for the heat_map
    '''
   
    variable_names = variable_names[1:]
    dic = {}
    
    for beta in beta_vec:
        dic[beta] = []
    
    for res in results:
        for keys in res[2]:
            dic[keys].append(res[2][keys])
    
    for keys in dic:
        dic[keys] = np.divide(np.sum(dic[keys],axis=0),norm)
        
    max_ = []
    for key in dic:
        max_.append(np.max(dic[key]))
    max__ = np.max(max_)
    if max__<1.:
        for key in dic:
            #dic[key] = np.divide(dic[key],max__)
            dic[key] = dic[key] #normalizar
    
    aux = np.array(list(dic.values()))
    index = range(np.shape(aux)[1])
    
    mean = np.mean(aux,axis=0)
    index = [x for _,x in sorted(zip(mean,index))] #sort by mean
        
    data = np.transpose(np.array(  aux ))[index]
    variable_names = np.asarray(variable_names)[index]
    
    data = np.flipud(data)
    variable_names = np.flipud(variable_names)
    
    return [data,variable_names]
    
def plot_heat_map(data, variable_names, beta_vec):
    ''' plot the heatmap from the data using the names of the variable'''
    
    fig = plt.figure(figsize=(6,4)) 
    ax = fig.add_subplot(111)
    
    ax = sns.heatmap(data,cbar_kws={'label': 'Normalized Significance'})
    ax.set_yticks(range(np.shape(data)[0]), variable_names,rotation = 0, fontsize = 8)
    
    print(np.shape(data)[1])
    
        
    try:
        ax.set_xticks(range(np.shape(data)[1]),list(np.round(beta_vec,3)),rotation = 45, fontsize = 8)
    except:
        try:
            ax.set_xticks(range(np.shape(data)[1]),list(np.round(beta_vec,4)),rotation = 45, fontsize = 8)
        except:
            ax.set_xticks(range(np.shape(data)[1]),list(beta_vec),rotation = 45, fontsize = 8)
        
    ax.figure.axes[-1].yaxis.label.set_size(8)
    ax.set_xlabel(r'$\beta$')

def one_hot_significant(l, dim):
    ''' returns a list with 1 for the significant variables and 0 otherwise'''
    dic = get_null_dic(dim)
    for num in l:
        dic[num] +=1
    return list(dic.values())

def get_null_dic(n):
    ''' generate a empty dictionary'''
    dic = {}
    for i in range(1,n):
        dic[i] = 0
    return dic

def read_data(path, file_name):
    df = pd.read_excel(os.path.join(path, file_name))

    df.drop(columns = ["oc1","oc2","oc3","oc4","oc5","oc6","oc7","oc8",
                   "balanced_scenario_2", "business_scenario_2", "tech_scenario_2",
                   "Company Name", "Company2", "Company Name Mayusculas", "Respondent Name",
                   'b2b'], inplace=True)
    X = df[["role","business", "tech", "mix","edu_high_school","edu_bachelor", 
        "edu_master","edu_phd","experience", "gender", "age"]]

    missing = ["tenure", "efficiency_experience", "growth_experience", "ai_experience"]

    for var in missing:
        y = df[var]
        X_test = X[y.isna()]
        y_train = y[y.notnull()]
        X_train = X[y.notnull()]
        reg = LinearRegression().fit(X_train, y_train)
        y_test = reg.predict(X_test)
        df.loc[y.isna(),var] = [max(0,i) for i in y_test]

    df.sort_values("balanced_scenario", inplace=True)
    return df

def r_index(X, y, model):
    ''' compute the r^2 index for a model that has the method model.predict'''
    try:
        y_pred = pd.DataFrame(model.predict(X,verbose=0))
    except:
        y_pred = pd.DataFrame(model.predict(X))
    y = pd.DataFrame(y)
    
    RSS = np.sum(np.square(y-y_pred))
    TSS = np.sum(np.square(y-y.mean()))
    return round((1-RSS/TSS)[0],3)


def get_index_pvalues(model,alpha=0.05):
    ''' given an OLS trained model, the function returns the index of the significat variables'''
    aux = model.pvalues<=alpha
    aux = np.where(aux==True)[0]
    try:
        if aux[0]==0:
            return aux[1:]
        else:
            return aux
    except:
        return []