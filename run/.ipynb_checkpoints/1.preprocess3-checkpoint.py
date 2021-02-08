#!/usr/bin/env python
# coding: utf-8

# In[1]:


def seed_everything(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    #tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# In[2]:


import os 
import random
import re
import warnings
import sys
sys.path.append('../')
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import category_encoders as ce
from category_encoders.cat_boost import CatBoostEncoder
import xfeat
import torch
from mypipe.config import Config
#seed_everything(seed=2021)
RUN_NAME = "exp00"
config = Config(RUN_NAME, folds=5)


# In[3]:


train = pd.read_csv(os.path.join(config.INPUT, "train.csv"))
test = pd.read_csv(os.path.join(config.INPUT, "test.csv"))


# In[6]:


important_BERT= pd.read_csv(os.path.join(config.INPUT, "rough_fineBERT.csv")).iloc[:,[420,705,157,186,299]]


# In[4]:


## raw features
def goal2feature(input_df):
    tmp = input_df["goal"]
    tmp = tmp.replace("100000+", "100000-100000")
    tmp = np.array([g.split("-") for g in tmp], dtype="int")
    output_df = pd.DataFrame(tmp, columns=["goal_min", "goal_max"])
    output_df["goal_upper_flag"] = output_df["goal_min"] == 100000
    output_df["goal_lower_flag"] = output_df["goal_min"] == 1
    output_df["goal_mean"] = output_df[["goal_min", "goal_max"]].mean(axis=1)
    #output_df["goal_q25"] = output_df[["goal_min", "goal_max"]].quantile(q=0.25, axis=1)
    #output_df["goal_q75"] = output_df[["goal_min", "goal_max"]].quantile(q=0.75, axis=1)
    return output_df

def get_numerical_feature(input_df):
    cols = ["duration"]
    return input_df[cols].copy()

## binning
def get_bins(input_df):
    _input_df = pd.concat([
        input_df[["duration"]],
        goal2feature(input_df),
    ], axis=1)
    output_df = pd.DataFrame()
    output_df["bins_duration"] = pd.cut(_input_df["duration"],
                                        bins=[-1, 30, 45, 60, 100],
                                        labels=['bins_d1', 'bins_d2', 'bins_d3', 'bins_d4'])
    output_df["bins_goal"] = pd.cut(_input_df["goal_max"],
                                    bins=[-1, 19999, 49999, 79999, 99999, np.inf],
                                    labels=['bins_g1', 'bins_g2', 'bins_g3', 'bins_g4', 'bins_g5'])
    return output_df.astype(str)


# In[9]:


## cross features
# カテゴリ変数×カテゴリ変数
def get_cross_cat_features(input_df):
    _input_df = pd.concat([
        input_df,
        get_bins(input_df)
    ], axis=1).astype(str)
    output_df = pd.DataFrame()
    output_df["category3"] = _input_df["category1"] + _input_df["category2"] 
    output_df["country+category1"] = _input_df["country"] + _input_df["category1"]
    output_df["country+category2"] = _input_df["country"] + _input_df["category2"]
    output_df["country+category3"] = _input_df["country"] + output_df["category3"]
    output_df["bins_DurationGoal"] = _input_df["bins_duration"] + _input_df["bins_goal"]
    
    return output_df

# 数値変数×数値変数
def get_cross_num_features(input_df):
    _input_df = pd.concat([
        input_df,
        goal2feature(input_df), 
    ], axis=1)
    #seed_everything(seed=2021)
    output_df = pd.DataFrame()
    output_df["ratio_goalMean_duration"] = _input_df["goal_mean"] / (_input_df["duration"] + 1)
    output_df["prod_goalMean_duration"] = _input_df["goal_mean"] * (_input_df["duration"])
   ### 

    output_df["ratio_299_duration"] = _input_df["rough_fineBERT299"] / (_input_df["duration"] + 1)
    output_df["ratio_299_goalMean"] = _input_df["rough_fineBERT299"] / (_input_df["goal_mean"] + 1)
    output_df["prod_299_duration"] = _input_df["rough_fineBERT299"] * (_input_df["duration"])
    output_df["prod_299_goalMean"] = _input_df["rough_fineBERT299"] * (_input_df["goal_mean"])
    output_df["ratio_157_duration"] = _input_df["rough_fineBERT157"] / (_input_df["duration"] + 1)
    output_df["ratio_157_goalMean"] = _input_df["rough_fineBERT157"] / (_input_df["goal_mean"] + 1)
    output_df["prod_157_duration"] = _input_df["rough_fineBERT157"] * (_input_df["duration"])
    output_df["prod_157_goalMean"] = _input_df["rough_fineBERT157"] * (_input_df["goal_mean"])    
    output_df["ratio_420_duration"] = _input_df["rough_fineBERT420"] / (_input_df["duration"] + 1)
    output_df["ratio_420_goalMean"] = _input_df["rough_fineBERT420"] / (_input_df["goal_mean"] + 1)
    output_df["prod_420_duration"] = _input_df["rough_fineBERT420"] * (_input_df["duration"])
    output_df["prod_420_goalMean"] = _input_df["rough_fineBERT420"] * (_input_df["goal_mean"]) 
    output_df["ratio_186_duration"] = _input_df["rough_fineBERT186"] / (_input_df["duration"] + 1)
    output_df["ratio_186_goalMean"] = _input_df["rough_fineBERT186"] / (_input_df["goal_mean"] + 1)
    output_df["prod_186_duration"] = _input_df["rough_fineBERT186"] * (_input_df["duration"])
    output_df["prod_186_goalMean"] = _input_df["rough_fineBERT186"] * (_input_df["goal_mean"]) 
    output_df["ratio_705_duration"] = _input_df["rough_fineBERT705"] / (_input_df["duration"] + 1)
    output_df["ratio_705_goalMean"] = _input_df["rough_fineBERT705"] / (_input_df["goal_mean"] + 1)
    output_df["prod_705_duration"] = _input_df["rough_fineBERT705"] * (_input_df["duration"])
    output_df["prod_705_goalMean"] = _input_df["rough_fineBERT705"] * (_input_df["goal_mean"]) 

## 157,186,299,420,705,
    output_df["ratio_157_186"] = _input_df["rough_fineBERT157"] / (_input_df["rough_fineBERT186"] + 1)
    output_df["ratio_157_299"] = _input_df["rough_fineBERT157"] / (_input_df["rough_fineBERT299"] + 1)
    output_df["ratio_157_420"] = _input_df["rough_fineBERT157"] / (_input_df["rough_fineBERT420"] + 1)
    output_df["ratio_157_705"] = _input_df["rough_fineBERT157"] / (_input_df["rough_fineBERT705"] + 1)
    output_df["ratio_186_299"] = _input_df["rough_fineBERT186"] / (_input_df["rough_fineBERT299"] + 1)
    output_df["ratio_186_420"] = _input_df["rough_fineBERT186"] / (_input_df["rough_fineBERT420"] + 1)
    output_df["ratio_186_705"] = _input_df["rough_fineBERT186"] / (_input_df["rough_fineBERT705"] + 1)
    output_df["ratio_299_420"] = _input_df["rough_fineBERT299"] / (_input_df["rough_fineBERT420"] + 1)
    output_df["ratio_299_705"] = _input_df["rough_fineBERT299"] / (_input_df["rough_fineBERT705"] + 1)
    output_df["ratio_420_705"] = _input_df["rough_fineBERT420"] / (_input_df["rough_fineBERT705"] + 1)
    output_df["prod_157_186"]  = _input_df["rough_fineBERT157"] * (_input_df["rough_fineBERT186"])
    output_df["prod_157_299"]  = _input_df["rough_fineBERT157"] * (_input_df["rough_fineBERT299"])
    output_df["prod_157_420"]  = _input_df["rough_fineBERT157"] * (_input_df["rough_fineBERT420"])
    output_df["prod_157_705"]  = _input_df["rough_fineBERT157"] * (_input_df["rough_fineBERT705"])
    output_df["prod_186_299"]  = _input_df["rough_fineBERT186"] * (_input_df["rough_fineBERT299"])
    output_df["prod_186_420"]  = _input_df["rough_fineBERT186"] * (_input_df["rough_fineBERT420"])
    output_df["prod_186_705"]  = _input_df["rough_fineBERT186"] * (_input_df["rough_fineBERT705"])
    output_df["prod_299_420"]  = _input_df["rough_fineBERT299"] * (_input_df["rough_fineBERT420"])
    output_df["prod_299_705"]  = _input_df["rough_fineBERT299"] * (_input_df["rough_fineBERT705"])
    output_df["prod_420_705"]  = _input_df["rough_fineBERT420"] * (_input_df["rough_fineBERT705"])
    return output_df


# In[10]:


## count encoding and CBtarget encoding
def get_ce_features(input_df):
    seed_everything(seed=2021)
    _input_df = pd.concat([
        input_df, 
        get_cross_cat_features(input_df),
        get_bins(input_df)
    ], axis=1).astype(str)
    cols = [
        "category1",
        "category2",
        "category3",
        "country",
        "country+category1",
        "country+category2",
        "country+category3",
        "bins_duration",
        "bins_goal",
        "bins_DurationGoal",
    ]
    #count encoding
    encoder = ce.CountEncoder()
    output_df1 = encoder.fit_transform(_input_df[cols]).add_prefix("CE_")
    #CBtarget encoding
    CB_encoder = CatBoostEncoder(sigma=15, a=1, random_state=2021)
    output_df2 = CB_encoder.fit_transform(_input_df[cols], _input_df['state']).add_suffix("_CBtarget")
    
    output_df=pd.concat([output_df1,output_df2],axis=1)
    return output_df.copy()


# In[5]:


def aggregation(input_df):
    _input_df = pd.concat([
        input_df,
        goal2feature(input_df),
        get_cross_num_features(input_df),
        get_cross_cat_features(input_df),
    ], axis=1)
    #seed_everything(seed=2021)
    group_key = ["country",  # カテゴリ変数
                  "category1", # カテゴリ変数
                  "category2",  # カテゴリ変数
                  "category3",  # カテゴリ変数
                  "bins_DurationGoal",

                 
                ] # カテゴリ変数
                
    group_values = [  # 集約される数値特徴量
        "goal_mean",
        "duration",
        "ratio_goalMean_duration",
        "prod_goalMean_duration",
        "rough_fineBERT420", 	"rough_fineBERT705" ,"rough_fineBERT157", 	"rough_fineBERT186", 	"rough_fineBERT299",
        
        "prod_705_goalMean", "prod_420_goalMean", "prod_157_goalMean",
        "ratio_299_goalMean",  "prod_186_goalMean", 

    ]
    agg_methods = ["min", "max", "mean", "std", "count"]  # 集約方法, 
    output_df1, cols1 = xfeat.aggregation(_input_df, group_key[0], group_values, agg_methods)
    output_df2, cols2 = xfeat.aggregation(_input_df, group_key[1], group_values, agg_methods)
    output_df3, cols3 = xfeat.aggregation(_input_df, group_key[2], group_values, agg_methods)
    output_df4, cols4 = xfeat.aggregation(_input_df, group_key[3], group_values, agg_methods)
    output_df5, cols5 = xfeat.aggregation(_input_df, group_key[4], group_values, agg_methods)
   
    output_df=pd.concat([output_df1[cols1], output_df2[cols2], output_df3[cols3],
                         output_df4[cols4], output_df5[cols5]], axis=1)
    return output_df.copy()


# In[6]:


# html_content 含めた特徴量作成
def get_process_funcs():
    funcs = [
        goal2feature,
        get_numerical_feature,
        get_cross_num_features,
        get_ce_features,
        aggregation
    ]
    return funcs

def to_feature(input_df, funcs):
    output_df = pd.DataFrame()
    for func in funcs: #tqdm(funcs, total=len(funcs)):
        _df = func(input_df)
        assert len(_df) == len(input_df), func.__name__
        output_df = pd.concat([output_df, _df], axis=1)

    return output_df


# In[14]:


## preprocessing
input_df = pd.concat([train, test]).reset_index(drop=True)  # このコンペではルール的に可能なので、楽するためにconcatして前処理
input_df=pd.concat([input_df, important_BERT], axis=1)
# all featrues
process_funcs = get_process_funcs()
output_df = to_feature(input_df, process_funcs)

print(output_df.shape)
output_df.to_csv(os.path.join(config.OUTPUT, "rprocessed_table.csv"), index=False, header=True)
print('done')


# In[ ]:




