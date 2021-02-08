## SIGNATE 22卒インターン選考コンペティション 15th place solution
### 1. 特徴量
  - html content
    - 前処理なし,textのみなどの前処理をして以下の特徴量を作成
    - html tag数, 文字数などbasicな特徴量
    - tf-idf変換 => Truncated SVD
    - BERT
    
  - html content 以外
    - raw: duration, goal
    - binning: duration & goal
    - 組み合わせ特徴量: duration, goal, category, country, binning など
    - 集約特徴量: 組み合わせ特徴量, カテゴリ変数などに対し、duration、goal、 BERTの最終層(一部)を「max, min, mean, std, count」で集約
    
  - 計1478の特徴量から296個を選択

### 2. 予測モデル
  - LightGBM
    - 深さを変えた３種類のモデルを使用
    - 12folds, 3seeds average

### 3. プログラムの実行
```
#言語判定モデルのダウンロード
wget -P ./data/ https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin 

#実行
. ./run.sh
```
