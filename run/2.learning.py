import os 
import re
import random
from time import time
import pickle
import warnings
import sys
sys.path.append('../')
warnings.filterwarnings("ignore")

import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

from sklearn.metrics import f1_score, fbeta_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMModel
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool, FeaturesData
import xgboost as xgb

#from hyperopt import hp, tpe, Trials, fmin

import re
from bs4 import BeautifulSoup
from fasttext import load_model
def seed_everything(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(seed=2021)
from mypipe.config import Config
#seed_everything(seed=2021)
RUN_NAME = "exp00"
config = Config(RUN_NAME, folds=5)

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

# fold ごとの index を返す関数を定義
def make_skf(train_x, train_y, random_state=2021):
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=random_state)
    folds_idx = [(t, v) for (t, v) in skf.split(train_x, train_y)]
    return folds_idx

# visualize result
def visualize_confusion_matrix(y_true,
                               pred_label,
                               height=.6,
                               labels=None):
    conf = confusion_matrix(y_true=y_true,
                            y_pred=pred_label,
                            normalize='true')
    n_labels = len(conf)
    size = n_labels * height
    fig, ax = plt.subplots(figsize=(size * 4, size * 3))
    sns.heatmap(conf, cmap='Blues', ax=ax, annot=True, fmt='.2f')
    ax.set_ylabel('Label')
    ax.set_xlabel('Predict')

    if labels is not None:
        ax.set_yticklabels(labels)
        ax.set_xticklabels(labels)
        ax.tick_params('y', labelrotation=0)
        ax.tick_params('x', labelrotation=90)

    plt.show()
    return fig

# 最適な閾値を求める関数
def threshold_optimization(y_true, y_pred, metrics=None):
    seed_everything(seed=2021)
    def f1_opt(x):
        if metrics is not None:
            score = -metrics(y_true, y_pred >= x)
        else:
            raise NotImplementedError
        return score
    result = minimize(f1_opt, x0=np.array([0.5]), method='Nelder-Mead')
    best_threshold = result['x'].item()
    return best_threshold

# 後で定義するモデルは確率で結果を出力するので、そこから最適なf1をgetする関数を定義
def optimized_f1(y_true, y_pred):
    seed_everything(seed=2021)
    bt = threshold_optimization(y_true, y_pred, metrics=f1_score)
    score = f1_score(y_true, y_pred >= bt)
    return score

def get_best(y_true, y_pred):
    def wight_opt(x):
        y_preds = np.average(list(y_pred.values()), axis=0, weights=x)
        score = -optimized_f1(y_true, y_preds)
        return score
    
    cons = [{'type': 'eq',
         'fun': lambda w: w.sum() - 1}]
    x0 = np.empty(len(oof)) ;x0.fill(1 / len(oof))
    result = minimize(wight_opt, x0,
                      constraints=cons, method='Nelder-Mead')
    best_weight = result.x/sum(result.x)
    
    bt=threshold_optimization(y_true, np.average(list(y_pred.values()), axis=0, weights=best_weight), metrics=f1_score)
    f1 = optimized_f1(y_true, np.average(list(y_pred.values()), axis=0, weights=best_weight))
    return best_weight, bt, f1

# LGBMModel の wrapper
class MyLGBMModel:
    def __init__(self, name=None, params=None, fold=None, train_x=None, train_y=None, test_x=None, metrics=None, seeds=None):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.name = name
        self.params = params
        self.metrics = metrics  # metrics で定義した関数
        self.kfold = fold  # make fold で定義した関数
        self.oof = None
        self.preds = None
        self.seeds = seeds if seeds is not None else [2020]  # seed average用のseed値
        self.models = {}  # 学習済みモデルを保持

    def build_model(self):
        model = LGBMModel(**self.params)
        return model

    def predict_cv(self):
        seed_everything(seed=2021)
        oof_seeds = []
        scores_seeds = []
        for seed in self.seeds:
            oof = []
            va_idxes = []
            scores = []
            train_x = self.train_x.values
            train_y = self.train_y.values
            fold_idx = self.kfold(self.train_x, self.train_y, random_state=seed) 

            # train and predict by cv folds
            for cv_num, (tr_idx, va_idx) in enumerate(fold_idx):
                tr_x, va_x = train_x[tr_idx], train_x[va_idx]
                tr_y, va_y = train_y[tr_idx], train_y[va_idx]
                va_idxes.append(va_idx)
                model = self.build_model()
    
                # fitting - train
                model.fit(tr_x, tr_y,
                          eval_set=[[va_x, va_y]],
                          early_stopping_rounds=100,
                          verbose=False)  
                model_name = f"{self.name}_SEED{seed}_FOLD{cv_num}_model.pkl"
                #self.models[model_name] = model  # save model
                pickle.dump(model, open(os.path.join(config.models, f'{model_name}'), 'wb'))#---------------------------!!!
                # predict - validation
                pred = model.predict(va_x)
                oof.append(pred)

                # validation score
                score = self.get_score(va_y, pred)
                scores.append(score)
                #print(f"SEED:{seed}, FOLD:{cv_num} =====> val_score:{score}", f"iterate:{model.best_iteration_}")

            # sort as default
            va_idxes = np.concatenate(va_idxes)
            oof = np.concatenate(oof)
            order = np.argsort(va_idxes)
            oof = oof[order]
            oof_seeds.append(oof)
            scores_seeds.append(np.mean(scores))
    
        oof = np.mean(oof_seeds, axis=0)
        oof_std=np.std(scores)
        self.oof = oof
        #print(f"model:{self.name} score:{self.get_score(self.train_y, oof)}(std:{oof_std})\n")
        return oof


    def tree_importance(self):
        seed_everything(seed=2021)
        # visualize feature importance
        feature_importance_df = pd.DataFrame()
        for i, (tr_idx, va_idx) in enumerate(self.kfold(self.train_x, self.train_y)):
            #print('\rfold %d/%d    ' %(i+1,FOLDS) ,end='')
            tr_x, va_x = self.train_x.values[tr_idx], self.train_x.values[va_idx]
            tr_y, va_y = self.train_y.values[tr_idx], self.train_y.values[va_idx]
            model = self.build_model()
            model.fit(tr_x, tr_y,
                      eval_set=[[va_x, va_y]],
                      early_stopping_rounds=100,
                      verbose=False)
            _df = pd.DataFrame()
            _df['feature_importance'] = model.feature_importances_
            _df['column'] = self.train_x.columns
            _df['fold'] = i + 1
            feature_importance_df = pd.concat([feature_importance_df, _df], axis=0, ignore_index=True)
        order = feature_importance_df.groupby('column')                     .sum()[['feature_importance']]                     .sort_values('feature_importance', ascending=False).index[:50]

        return  feature_importance_df
    
    def get_score(self, y_true, y_pred):
        seed_everything(seed=2021)
        score = self.metrics(y_true, y_pred)
        return score

table = pd.read_csv(os.path.join(config.INPUT, "processed_table.csv"))
basic = pd.read_csv(os.path.join(config.INPUT, "html_basic.csv"))
SVD = pd.read_csv(os.path.join(config.INPUT, "svd128.csv"))
fineBERT = pd.read_csv(os.path.join(config.INPUT, "rough_fineBERT.csv"))

#"""
df=reduce_mem_usage(pd.concat([
                              table, 
                              basic,
                              SVD,
                              fineBERT
                             ], axis=1))

###
train_y = pd.read_csv(os.path.join(config.INPUT, "train.csv"))['state']
train_x, test_x = df[:len(train_y)], df[len(train_y):].reset_index(drop=True)

NAME = "LightGBM001"
FOLDS, seeds= 12, [0,1,2]
omit = .8
# define model
LBGM_params = {
    "n_estimators": int(1e+5),
    "learning_rate": 1e-2,
    "num_leaves": 31,
    
    'colsample_bytree': .5, #=feature_fraction
    "reg_lambda": 3, #=lambda_l2
    
    "random_state": 2021,
    "n_jobs": 8,
    "importance_type": "gain",
    "objective": 'binary',
}
model = MyLGBMModel(name=NAME, params=LBGM_params, fold=make_skf, 
                    train_x=train_x, train_y=train_y, test_x=test_x, metrics=optimized_f1, 
                    seeds=seeds
                   )

# feature importance
importance_df = model.tree_importance()

# feature selections
cols = importance_df.groupby("column").mean().reset_index().sort_values("feature_importance", ascending=False)["column"].tolist()
selected_num =  len(cols) - int(len(cols)*omit)
selected_cols = cols[:selected_num]
pd.DataFrame(selected_cols).to_csv(os.path.join(config.OUTPUT, f"{NAME}_selected_cols.csv"), index=False, header=True)

model.train_x, model.test_x = model.train_x[selected_cols], model.test_x[selected_cols]

# train & inference
oof={}
oof[NAME]=model.predict_cv()
# --------------------------------------------------------------------------------------------------------------------------------------- #

NAME = "LightGBM002"
FOLDS, seeds= 12, [0,1,2]
omit = .8
# define model
LBGM_params = {
    "n_estimators": int(1e+5),
    "learning_rate": 1e-2,
    "num_leaves": 16,
    
    'colsample_bytree': .5, #=feature_fraction
    "reg_lambda": 3, #=lambda_l2
    
    "random_state": 2021,
    "n_jobs": 8,
    "importance_type": "gain",
    "objective": 'binary',
}
model = MyLGBMModel(name=NAME, params=LBGM_params, fold=make_skf, 
                    train_x=train_x, train_y=train_y, test_x=test_x, metrics=optimized_f1, 
                    seeds=seeds
                   )

# feature importance
importance_df = model.tree_importance()

# feature selections
cols = importance_df.groupby("column").mean().reset_index().sort_values("feature_importance", ascending=False)["column"].tolist()
selected_num =  len(cols) - int(len(cols)*omit)
selected_cols = cols[:selected_num]
pd.DataFrame(selected_cols).to_csv(os.path.join(config.OUTPUT, f"{NAME}_selected_cols.csv"), index=False, header=True)

model.train_x, model.test_x = model.train_x[selected_cols], model.test_x[selected_cols]
oof[NAME]=model.predict_cv()

# --------------------------------------------------------------------------------------------------------------------------------------- #
NAME = "LightGBM003"
FOLDS, seeds= 12, [0,1,2]
omit = .75
# define model
LBGM_params = {
    "n_estimators": int(1e+5),
    "learning_rate": 1e-2,
    "num_leaves": 48,
    
    'colsample_bytree': .5, #=feature_fraction
    "reg_lambda": 3, #=lambda_l2
    
    "random_state": 2021,
    "n_jobs": 8,
    "importance_type": "gain",
    "objective": 'binary',
}
model = MyLGBMModel(name=NAME, params=LBGM_params, fold=make_skf, 
                    train_x=train_x, train_y=train_y, test_x=test_x, metrics=optimized_f1, 
                    seeds=seeds
                   )

# feature importance
importance_df = model.tree_importance()

# feature selections
cols = importance_df.groupby("column").mean().reset_index().sort_values("feature_importance", ascending=False)["column"].tolist()
selected_num =  len(cols) - int(len(cols)*omit)
selected_cols = cols[:selected_num]
pd.DataFrame(selected_cols).to_csv(os.path.join(config.OUTPUT, f"{NAME}_selected_cols.csv"), index=False, header=True)

model.train_x, model.test_x = model.train_x[selected_cols], model.test_x[selected_cols]

oof[NAME]=model.predict_cv()
# --------------------------------------------------------------------------------------------------------------------------------------- #

best_weight, best_threshold, f1 = get_best(train_y, oof)
best=list(best_weight)
best.append(best_threshold)
pd.DataFrame(best).to_csv(os.path.join(config.OUTPUT, f"{RUN_NAME}_best.csv"), index=False, header=True)
