import os 
import re
import random
import warnings
import sys
sys.path.append('../')
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import torch
from torch import optim
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification

import texthero as hero
from bs4 import BeautifulSoup
from fasttext import load_model

from mypipe.config import Config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    #tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

seed_everything(seed=2021)
RUN_NAME = "exp00"
config = Config(RUN_NAME, folds=5)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

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
    bt = threshold_optimization(y_true, y_pred, metrics=f1_score)
    score = f1_score(y_true, y_pred >= bt)
    return score

def get_langID(input_df):
    #文章の抽出
    documents =[]
    #input_df=input_df[:len(train)]
    temp=input_df.html_content
    for i in range(len(input_df)): 
    #for i in range(100): 
        soup = BeautifulSoup(temp[i], "html.parser") 
        documents += [soup.find('div').get_text(strip=True)]
    new_documents=[re.sub(r"[0123456789_]", "", document) for document in documents]
    new_documents=[re.sub(r"\W", " ", document) for document in new_documents]

    #言語判定モデルの読み込み
    lang_model = load_model(os.path.join(config.INPUT, "lid.176.bin"))

    #言語判定結果の格納
    lang=[];lang+=[lang_model.predict(new_documents[i])[0] for i in range(len(input_df))]
    lang=pd.DataFrame(lang, columns=['language'])
    #English以外のレコードのindex
    multiID=input_df.index[lang.language != '__label__en']
    EngID=input_df.index[lang.language == '__label__en']
    return multiID, EngID

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

def get_Engdataloader(texts, y_labels, tokenizer):
    seed_everything(seed=2021)
    input_ids, attention_masks = [], []
    # 1文づつ処理
    for sent in texts:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      
                            add_special_tokens = True, # Special Tokenの追加
                            max_length = 108,           # 文章の長さを固定（Padding/Trancatinating）
                            pad_to_max_length = True,# PADDINGで埋める
                            return_attention_mask = True,   # Attention maksの作成
                            return_tensors = 'pt',     #  Pytorch tensorsで返す
                            truncation=True
                       )

        # 単語IDを取得    
        input_ids.append(encoded_dict['input_ids'])

        # Attention　maskの取得
        attention_masks.append(encoded_dict['attention_mask'])

    # リストに入ったtensorを縦方向（dim=0）へ結合
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # tenosor型に変換
    y_labels = torch.tensor(y_labels)
    
    # データセットクラスの作成
    dataset = TensorDataset(input_ids, attention_masks, y_labels)

    # データローダーの作成
    batch_size = 16

    # 訓練データローダー
    dataloader = DataLoader(
                dataset,  
                sampler = RandomSampler(dataset), # ランダムにデータを取得してバッチ化
                batch_size = batch_size,
                num_workers=0,
                worker_init_fn=worker_init_fn
                )

    return dataloader

def get_multidataloader(texts, y_labels, tokenizer):
    seed_everything(seed=2021)
    input_ids, attention_masks = [], []
    # 1文づつ処理
    for sent in texts:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      
                            add_special_tokens = True, # Special Tokenの追加
                            max_length = 48,           # 文章の長さを固定（Padding/Trancatinating）
                            pad_to_max_length = True,# PADDINGで埋める
                            return_attention_mask = True,   # Attention maksの作成
                            return_tensors = 'pt',     #  Pytorch tensorsで返す
                            truncation=True
                       )

        # 単語IDを取得    
        input_ids.append(encoded_dict['input_ids'])

        # Attention　maskの取得
        attention_masks.append(encoded_dict['attention_mask'])

    # リストに入ったtensorを縦方向（dim=0）へ結合
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # tenosor型に変換
    y_labels = torch.tensor(y_labels)
    
    # データセットクラスの作成
    dataset = TensorDataset(input_ids, attention_masks, y_labels)

    # データローダーの作成
    batch_size = 16

    # 訓練データローダー
    dataloader = DataLoader(
                dataset,  
                sampler = RandomSampler(dataset), # ランダムにデータを取得してバッチ化
                batch_size = batch_size,
                num_workers=0,
                worker_init_fn=worker_init_fn
                )
    return dataloader

def finetuning(dataloader, model, max_epoch = 1):
    seed_everything(seed=2021)  
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8) # Default epsilon value)
    
    # 訓練パートの定義
    def train(model):
        seed_everything(seed=2021) 
        model.train()
        for batch in dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            optimizer.zero_grad()
            loss = model(b_input_ids, 
                         attention_mask=b_input_mask,
                         token_type_ids=None, 
                         labels=b_labels
                        )['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model=model.to(device)
        return model
    
    # 学習の実行
    for epoch in range(max_epoch):
        model = train(model)

    return model

def get_engBERT(input_df, 
            text_columns, EngID,
            name="",
            hero=None,
            max_epoch = 1):
    seed_everything(seed=2021)
    output_df = pd.DataFrame()
    for c in text_columns:
        if hero is not None:
            input_df[c] = hero(input_df, c)
        output_df[c] = [[word for word in document.lower().split()] for document in input_df[c]]

        texts=[["missing"] if word==[] else word for word in output_df[c]]
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels = 2,
                                                              output_attentions = False,
                                                              output_hidden_states = True) ;model = model.to(device)#1431MiB
        if max_epoch>=1:
            train_texts=texts[:len(train)]
            Engy_labels=y_labels[EngID[EngID<len(train)]]
            Engtexts  = [text  for text in train_texts  if train_texts.index(text) in EngID[EngID<len(train)]]

            dataloader = get_Engdataloader(Engtexts, Engy_labels.reset_index(drop=True), tokenizer)
            
            model = finetuning(dataloader, model, max_epoch = max_epoch )
        
        
        Engtexts_BERT  = [text  for text in texts  if texts.index(text) in EngID]
        indexed_tokens = list(map(tokenizer.convert_tokens_to_ids, Engtexts_BERT))
        indexed_tokens =[indexed_token[:512] if len(indexed_token)>512 else indexed_token                                                                  for indexed_token in indexed_tokens ]
        
        model.eval()
        with torch.no_grad(): 
            all_encoder_layers = list(map(lambda x: np.mean(model(torch.tensor([x]).to(device))[1][-2].detach().cpu().numpy()[0], axis=0), indexed_tokens))
        output_df=pd.DataFrame(all_encoder_layers, columns=[name+f'_fineBERT{i}'                                                             for i in range(len(all_encoder_layers[0]))])
            
    return output_df

def get_multiBERT(input_df, 
            text_columns, multiID,
            name="",
            hero=None,
            max_epoch = 1):
    seed_everything(seed=2021)
    output_df = pd.DataFrame()
    for c in text_columns:
        if hero is not None:
            input_df[c] = hero(input_df, c)
        output_df[c] = [[word for word in document.lower().split()] for document in input_df[c]]

        texts=[["missing"] if word==[] else word for word in output_df[c]]
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        
        model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-uncased", num_labels = 2,
                                                              output_attentions = False,
                                                              output_hidden_states = True);model = model.to(device)#1431MiB
        if max_epoch>=1:
            train_texts=texts[:len(train)]
            multiy_labels=y_labels[multiID[multiID<len(train)]]
            multitexts= [text  for text in train_texts  if train_texts.index(text) in multiID[multiID<len(train)]]

            dataloader = get_multidataloader(multitexts, multiy_labels.reset_index(drop=True), tokenizer)
            
            model = finetuning(dataloader, model, max_epoch = max_epoch )
        
        multitexts_BERT  = [text  for text in texts  if texts.index(text) in multiID]
        indexed_tokens = list(map(tokenizer.convert_tokens_to_ids, multitexts_BERT))
        indexed_tokens =[indexed_token[:512] if len(indexed_token)>512 else indexed_token                                                                  for indexed_token in indexed_tokens ]
        model.eval()
        with torch.no_grad(): 
            all_encoder_layers = list(map(lambda x: np.mean(model(torch.tensor([x]).to(device))[1][-2].detach().cpu().numpy()[0], axis=0), indexed_tokens))

        output_df=pd.DataFrame(all_encoder_layers, columns=[name+f'_fineBERT{i}'                                                             for i in range(len(all_encoder_layers[0]))])
            
    return output_df

def get_fineBERTALL(input_df):
    multiID, EngID = get_langID(input_df)
    multi = get_multiBERT(input_df, ["html_content"] ,multiID, name='rough', hero=hero_rough,
                                                                          max_epoch = 1) ;multi.index=multiID
    eng  =  get_engBERT(input_df,   ["html_content"] ,EngID,   name='rough', hero=hero_rough,
                                                                          max_epoch = 1) ;eng.index = EngID
    
    output_df1=pd.concat([eng, multi], axis=0); output_df1=output_df1.sort_index()
    return output_df1

## preprocessin
train = pd.read_csv(os.path.join(config.INPUT, "train.csv"))
test = pd.read_csv(os.path.join(config.INPUT, "test.csv"))
input_df = pd.concat([train, test]).reset_index(drop=True)
y_labels = pd.read_csv(os.path.join(config.INPUT, "train.csv"))['state']

output_df = get_fineBERTALL(input_df)
output_df.to_csv(os.path.join(config.OUTPUT, "rough_fineBERT.csv"), index=False, header=True)

