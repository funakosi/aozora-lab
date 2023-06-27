# 感情分析v4
# スコア取得を関数化

# ライブラリ

# %%
import os
import sys
import glob
import shutil
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import torch.nn.functional as F

# %%
# Goole Colab環境か判断
# ローカル環境とColabo環境の両方で動作させたい(そのうち使う予定)
moduleList = sys.modules
ENV_COLAB = False
if 'google.colab' in moduleList:
    print("google_colab")
    ENV_COLAB = True
else:
    print("Not google_colab")
if ENV_COLAB:
    print("Execute in google_colab")

# %% [markdown]
# ## 関数定義

# %%
# スコア取得関数
"""
Arg:
    tokenizer
    model
    text: text(one line)
Returns:
    max logit
    max prediction
"""
def get_score(tokenizer, model, text):
    # 0: NEUTRAL  -> 0
    # 1: NEGATIVE -> -1
    # 2: POSITIVE -> 1
    coef_array = [0, -1, 1]
    # text: 文字列型を想定
    batch = tokenizer(text, padding=True, return_tensors='pt')

    with torch.no_grad():
        output = model(**batch)
        prediction = F.softmax(output.logits, dim=1)
        label = torch.max(output.logits, dim=1)
        value = label.values.item()
        index = label.indices.item()
    logit_value = value * coef_array[index]
    pred_value = torch.max(prediction).item() * coef_array[index]
    return logit_value, pred_value

# %%
# 小説のスコアを取得する関数
"""
Arg:
    file_path
Returns:
    data frame
"""
def get_novel_score(tokenizer, model, file_path):
    df = pd.read_csv(file_path)
    logit_score, pred_score = [], []
    for i, text in enumerate(tqdm(df['text'])):
        logit, pred = get_score(tokenizer, model, text)
        logit_score.append(logit)
        pred_score.append(pred)

    df['logit_score'] = logit_score
    df['pred_score'] = pred_score
    return df

# %%
# 指定されたwindowサイズで感情スコアの平均値を取得
def get_score_mean(scores, window_size=10, score_mean_num=100):
    score_mean = []
    slide = int((len(scores) - window_size - 1) / 99)
    for n in range(score_mean_num):
        start = slide * n
        if n == score_mean_num - 1:
            end = len(scores) - 1
        else:
            end = start + window_size
        score_mean.append(np.mean(scores[start:end]))
        # print(f'{n}:len(scores):{len(scores)},st:{start},ed:{end},{scores[end]}{np.array(scores)[-1]}')
    return score_mean

# %%
# スコアを0-1に規格化
def score_normalize(scores):
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

# %% [markdown]
# ## モデル取得

# %%
tokenizer = AutoTokenizer.from_pretrained("koheiduck/bert-japanese-finetuned-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("koheiduck/bert-japanese-finetuned-sentiment")

# %% [markdown]
# ### 動作確認

# %%
# 動作確認
nlp = pipeline('sentiment-analysis',model=model,tokenizer=tokenizer)
print(nlp("私はとっても幸せ"))
print(nlp("私はとっても不幸"))

# %% [markdown]
# ## 小説のスコア取得

# %%
target = pd.read_csv('data/target2.csv', index_col=0)
target.head(2)

# %%
print(target.shape)
print(target[target['対象']].shape)

# %%
for i, (_, row_data) in enumerate(target.iterrows()):
    try:
        print(f'{i}: {row_data["氏名"]} {row_data["作品名"]} {row_data["テキストファイルパス"]}')
        if row_data['対象'] and os.path.isfile(row_data['テキストファイルパス']):
            if not os.path.isfile(row_data['スコアファイルパス']):
                df = get_novel_score(tokenizer, model, row_data['テキストファイルパス'])
                df.to_csv(row_data['スコアファイルパス'])
        else:
            print('skip data')
    except Exception as e:
        print(e)

# %%



