import os 
import re
import random
import warnings
import sys
sys.path.append('../')
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix

import texthero as hero
from texthero import stopwords as texthero_stopwords
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
#import nltk
#nltk.download('stopwords')

from collections import Counter
#import smart_open
import gensim
from gensim import corpora, models
from gensim.models import TfidfModel
from fasttext import load_model

import torch
from transformers import BertTokenizer, BertForSequenceClassification

from mypipe.config import Config

def seed_everything(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed=2021)
RUN_NAME = "exp00"
config = Config(RUN_NAME, folds=5)

stop_words = stopwords.words(['spanish', 'french', 'german'] )
default_stopwords = texthero_stopwords.DEFAULT#english
custom_stopwords = default_stopwords.union(stop_words)
language=['english','spanish', 'french', 'german'] 

# text cleansing 用の関数
def hero_rough(input_df, text_col):
    ## only remove html tags, do not remove punctuation
    custom_pipeline = [
        hero.preprocessing.fillna,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_punctuation,#
        hero.preprocessing.remove_diacritics,#
        hero.preprocessing.remove_whitespace
    ]
    texts = hero.clean(input_df[text_col], custom_pipeline)
    return texts

def hero_tags(input_df, text_col):
    ## only remove html tags, do not remove punctuation
    custom_pipeline = [
        hero.preprocessing.fillna,
        hero.preprocessing.remove_html_tags,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits
    ]
    texts = hero.clean(input_df[text_col], custom_pipeline)
    texts = hero.preprocessing.remove_stopwords(texts,custom_stopwords)
    texts = hero.preprocessing.remove_whitespace(texts)
    for l in language:
        texts=hero.preprocessing.stem(texts,language=l)
    return texts

def hero_text(input_df, text_col):
    ## get only text (remove html tags, punctuation & digits)
    custom_pipeline = [
        hero.preprocessing.fillna,
        hero.preprocessing.remove_html_tags,
        hero.preprocessing.lowercase,
        hero.preprocessing.remove_digits,
        hero.preprocessing.remove_punctuation,#
        hero.preprocessing.remove_diacritics,#
    ]
    texts = hero.clean(input_df[text_col], custom_pipeline)
    texts = hero.preprocessing.remove_stopwords(texts,custom_stopwords)
    texts = hero.preprocessing.remove_whitespace(texts)
    for l in language:
        texts=hero.preprocessing.stem(texts,language=l)
    return texts
# ------------------------------------------------------------ #

# text の基本的な情報をgetする関数
def get_basic(input_df, text_columns, hero=None, name=""):
    def _get_features(dataframe, column):
        _df = pd.DataFrame()
        _df[name + '_length'] = dataframe[column].apply(len)
        _df[name + '_!'] = dataframe[column].apply(lambda x: x.count('!'))
        _df[name + '_?'] = dataframe[column].apply(lambda x: x.count('?'))
        _df[name + '_tag'] = dataframe[column].apply(lambda x: x.count('<'))
        _df[name + '_http'] = dataframe[column].apply(lambda x: x.count('http'))#英語,スペイン、フランス,ドイツ,,イタリア,
        _df[name + '_I'] = dataframe[column].apply(lambda x: x.count('I')+x.count('Yo')+x.count('Je'))#\+x.count('Ich'))
        _df[name + '_you'] = dataframe[column].apply(lambda x: x.count('you')+x.count('tú')+x.count('toi'))#+x.count('sie'))
        _df[name + '_we'] = dataframe[column].apply(lambda x: x.count('we')+x.count('nosot')+x.count('nous'))#+x.count('wir'))
        _df[name + '_punctuation'] = dataframe[column].apply(lambda x: sum(x.count(w) for w in '.,;:'))
        _df[name + '_symbols'] = dataframe[column].apply(lambda x: sum(x.count(w) for w in '*&$%'))####
        _df[name + '_words'] = dataframe[column].apply(lambda x: len(x.split()))
        _df[name + '_unique_words'] = dataframe[column].apply(lambda x: len(set(w for w in x.split())))
        _df[name + '_unique_/_words'] = _df[name + '_unique_words'] / (_df[name + '_words'] +1 )
        _df[name + '_words_/_length'] = _df[name + '_words'] / (_df[name + '_length'] +1 )
        _df[name + '_http_/_length'] = _df[name + '_http'] / (_df[name + '_length'] +1 )
        _df[name + '_I_/_length'] = _df[name + '_I'] / (_df[name + '_length'] +1 )
        _df[name + '_you_/_length'] = _df[name + '_you'] / (_df[name + '_length'] +1 )
        _df[name + '_we_/_length'] = _df[name + '_we'] / (_df[name + '_length'] +1 )
        return _df
    
    # main の処理
    output_df = pd.DataFrame()
    output_df[text_columns] = input_df[text_columns].astype(str).fillna('missing')
    for c in text_columns:
        if hero is not None:
            output_df[c] = hero(output_df, c)
        output_df = _get_features(output_df, c)
    return output_df

# カウントベースの text vector をgetする関数 
def get_svd(input_df, 
            text_columns,
            hero=None,
            vectorizer=CountVectorizer(),
            transformer=TruncatedSVD(n_components=128, random_state=2021),
            name='html_count_svd'):
    seed_everything(seed=2021)
    output_df = pd.DataFrame()
    output_df[text_columns] = input_df[text_columns].astype(str).fillna('missing')
    features = []
    for c in text_columns:
        if hero is not None:
            output_df[c] = hero(output_df, c)

        sentence = vectorizer.fit_transform(output_df[c])
        feature = transformer.fit_transform(sentence)
        num_p = feature.shape[1]
        feature = pd.DataFrame(feature, columns=[name+str(num_p) + f'_{i:03}' for i in range(num_p)])
        features.append(feature)
    output_df = pd.concat(features, axis=1)
    return output_df
# ------------------------------------------------------------ #
# main の前処理関数たち

# text 前処理なしの basic features
def get_basicALL(input_df):
    output_df1 = get_basic(input_df=input_df,text_columns=["html_content"],
                           hero=None, name="raw")
    output_df2 = get_basic(input_df=input_df,text_columns=["html_content"],
                           hero=hero_tags, name="tags")
    output_df=pd.concat([output_df1, output_df2], axis=1)
    return output_df

# text 前処理なし html_content [tfidf -> sdv で次元削減(128)]
def get_svdALL(input_df):
    seed_everything(seed=2021)
    vectorizer=TfidfVectorizer()
    transformer=TruncatedSVD(n_components=128,  random_state=2021)
    output_df1 = get_svd(input_df,["html_content"], vectorizer=vectorizer,transformer=transformer,
                                hero=hero_rough, name="rough_svd")
    output_df2 = get_svd(input_df,["html_content"],vectorizer=vectorizer,transformer=transformer,
                           hero=hero_text, name="text_svd")
    
    output_df=pd.concat([output_df1, output_df2],axis=1)
    return output_df

train = pd.read_csv(os.path.join(config.INPUT, "train.csv"))
test = pd.read_csv(os.path.join(config.INPUT, "test.csv"))

input_df = pd.concat([train, test]).reset_index(drop=True)

output_df = get_svdALL(input_df)
output_df.to_csv(os.path.join(config.OUTPUT, "svd128.csv"), index=False, header=True)

output_df1 = get_basicALL(input_df)
output_df1.to_csv(os.path.join(config.OUTPUT, "html_basic.csv"), index=False, header=True)
