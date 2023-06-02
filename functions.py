import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
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

def get_r2(vec):
    # function obtain the r^2 index from the result data structure
    res = []
    for i in range(len(vec)):
        res.append([])
    
    for r in vec:
        for v,aa in zip(res,r):
            v.append(round(aa[2],3))
    return res

def different_models(X, y, n_runs = 10, layers = [[10]], beta_vec=[2e-2], alpha = 0.05, threshold = 0.9, variable_names = []):
    # function that runs a linear sm model and as many NN models as we indicate at layers. If layer=[] it will just performs a OLS
    
    if variable_names==[]: # we assign a name by its possition in case names were not provided
        for i in range(np.shape(X)[1]-1):
            variable_names.append(str(i))
    
    # Fit linear regression:
    model = sm.OLS(y, X.to_numpy())
    model_lin = model.fit()
    linear_r2 = r_index(X, y, model_lin)
    linear_sign = get_index_pvalues(model_lin,alpha)

    final_return = [[model_lin,linear_sign,linear_r2]]
    
    
    for layer in layers: 
        results = Parallel(n_jobs=-1, backend='loky')(delayed(single_run)(X, 
                                                                          y, 
                                                                          layers=layer, 
                                                                          beta_vec=beta_vec)  for i in range(0, n_runs))
        
        res = heat_map(results, beta_vec = beta_vec, norm = n_runs, variable_names = variable_names)
        res.append(np.mean(np.transpose(results)[1]))
        final_return.append(res)
    
    return final_return

def heat_map(results, beta_vec, norm, variable_names):
    # generates the data for the heat_map
    
    dic = {}
    for beta in beta_vec:
        dic[round(beta,3)] = []
    
    for res in results:
        for keys in res[2]:
            dic[keys].append(res[2][keys])
    
    for keys in dic:
        dic[keys] = np.divide(np.sum(dic[keys],axis=0),norm)
    
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
    
    fig = plt.figure(figsize=(6,4)) 
    ax = fig.add_subplot(111)
    
    ax = sns.heatmap(data,cbar_kws={'label': 'Normalized Significance'})
    ax.set_yticks(range(np.shape(data)[0]), variable_names,rotation = 0, fontsize = 8)
        
    ax.set_xticks(range(np.shape(data)[1]),list(np.round(beta_vec,3)),rotation = 45, fontsize = 8)
    ax.figure.axes[-1].yaxis.label.set_size(8)
    ax.set_xlabel(r'$\beta$')

def single_run(X, y, layers, batch_size = 32, nr_epochs = 100, beta_vec = [1e-2]):
    # single trainning of a NN model
    dim = np.shape(X)[1]
    
    inputs = Input(shape=(dim,))
    prev_layer = inputs
    for layer in layers:
        hidden = Dense(layer, activation='tanh')(prev_layer)
        prev_layer = hidden
        
    output = Dense(1, activation='linear')(prev_layer)
    
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=int(nr_epochs*0.2))
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error',run_eagerly=True)
    model.fit(x=X,
              y=y,
              batch_size=batch_size,
              epochs=nr_epochs,
              validation_split=0.2,
              callbacks=[early_stop],
              verbose=0)
    dic = {}
    for beta in beta_vec:
        sfit_NN = s_fit.sfit_first_order(model=model,
                                       loss=s_fit.absolute_loss,
                                       alpha=0.05,
                                       beta=beta,
                                       x=X.to_numpy(),
                                       y=y,
                                       verbose = 0)
        dic[round(beta,3)] = one_hot_significant(sfit_NN[0])
    base_loss = mean_squared_error(y, model.predict(X, verbose=0))
    r2 = r_index(X, y, model)
    return base_loss, r2, dic

def one_hot_significant(l):
    dic = get_null_dic()
    for num in l:
        dic[num] +=1
    return list(dic.values())

def get_null_dic(n = 24):
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
    try:
        y_pred = pd.DataFrame(model.predict(X,verbose=0))
    except:
        y_pred = pd.DataFrame(model.predict(X))
    y = pd.DataFrame(y)
    
    RSS = np.sum(np.square(y-y_pred))
    TSS = np.sum(np.square(y-y.mean()))
    return round((1-RSS/TSS)[0],3)


def get_index_pvalues(model,alpha=0.05):
    aux = model.pvalues<=alpha
    aux = np.where(aux==True)[0]
    if aux[0]==0:
        return aux[1:]
    else:
        return aux