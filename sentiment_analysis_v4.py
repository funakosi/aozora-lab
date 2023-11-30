# 感情分析v4

# ライブラリ
# %%
import os
import datetime
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
from pathlib import Path
import japanize_matplotlib
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import torch.nn.functional as F

from library import tool
from library import preprocess
from library import score

# %%
# GPUチェック
tool.is_cuda_available()

# %%
# 設定ファイルを読み込み
model_settings = tool.ReadModelTokenizerTome('./settings/model_tokenizer.toml')

# モデルやトークナイザーの選択
PATTERN = 'pattern2'

model_settings.read(PATTERN)
print(model_settings.get_str())

#  モデル取得
# %%
print(model_settings.tokenizer, model_settings.model)
tokenizer = AutoTokenizer.from_pretrained(model_settings.tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(model_settings.model)

# 小説のスコア取得
# %%
TARGET_PATH = 'data/target2.csv'

target = pd.read_csv(TARGET_PATH, index_col=0)
print('全部:',target.shape)
print('対象:',target[target['対象']].shape)
print('len>=500:',target[target['length']>=500].shape)

# %%
# すべての小説のスコアを取得
def get_all_scores(target, force=False, test=True):
    for i, (_, row_data) in enumerate(target.iterrows()):
        try:
            print(f'{i}: {row_data["氏名"]} {row_data["作品名"]} {row_data["テキストファイルパス"]}')
            if row_data['対象'] and os.path.isfile(row_data['テキストファイルパス']):
                # スコアファイルなし or 強制フラグ:Trueの場合，スコア取得
                if not os.path.isfile(row_data['スコアファイルパス']) or force:
                    df = score.get_novel_score(tokenizer, model, row_data['テキストファイルパス'])
                    df.to_csv(row_data['スコアファイルパス'])
            else:
                print('skip data')
        except Exception as e:
            print(e)
        # テストフラグ:Trueなら1件のみ取得してループを抜ける
        if test:
            break

# %%
# すべての小説のスコアを取得
get_all_scores(target, force=True, test=False)

# %%
# 全スコアを取得
def get_all_score(list_path, line_num=500, log=False):
    target = pd.read_csv(list_path, index_col=0)
    target = target[~np.isnan(target['length'])] # 欠損データは対象外
    file_exists = [os.path.isfile(f) for f in target['スコアファイルパス']] # ファイル有無
    target = target[file_exists] # スコアファイルが存在するものだけが対象
    target = target[target['length'] >= line_num] # 文の総数がline_num以上
    
    target_score = {}
    for i, (_, data) in enumerate(target.iterrows()):
        score_path = data['スコアファイルパス']
        exist = os.path.isfile(score_path);
        if log:
            print(exist, data.name, data['氏名'], data['作品名'], score_path, data['length'])
        if not exist:
            continue
        mt = datetime.datetime.fromtimestamp(os.path.getmtime(score_path)) # 更新日付取得
        if mt.month != 11:  # 仮
            continue
        print(score_path, mt)
        df = pd.read_csv(score_path)
        window_size = int(df.shape[0] / 7)
        logit_score_mean = score.get_score_mean(df['logit_score'], window_size=window_size)
        logit_score_norm = score.score_normalize(logit_score_mean)
        target_score[data.name] = logit_score_norm.tolist() # index:スコア値
    
    return target_score

# %%
# 全スコア
all_score = get_all_score(TARGET_PATH)
print('len(all_score):', len(all_score))

# 取得したスコア値をPandasのDataFrame形式に変換する
columns = ['S{:02}'.format(i) for i in range(100)]
df_score = pd.DataFrame.from_dict(all_score, orient='index', columns=columns)
df_score

# %%
# スコア値を保存
save_path = 'data/all_score_1124.csv'
df_score.to_csv(save_path) # 必要に応じて実行

