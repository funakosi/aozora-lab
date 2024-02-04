import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torch
import torch.nn.functional as F

# スコアに掛ける係数を取得
def get_coef(model):
    array = []
    for key in model.config.id2label.keys():
        label = model.config.id2label[key]
        weight = 0
        if label.upper() == 'POSITIVE':
            weight = 1
        elif label.upper() == 'NEGATIVE':
            weight = -1
        array.append(weight)
    return array

# スコア取得関数
"""
Arg:
    tokenizer
    model
    text: text(one line)
    truncation: Bool(Default:True)
Returns:
    max logit
    max prediction
"""
def get_score(tokenizer, model, text, truncation=True):
    # 0: NEUTRAL  -> 0
    # 1: NEGATIVE -> -1
    # 2: POSITIVE -> 1
    # coef_array = [0, -1, 1]
    coef_array = get_coef(model)
    # text: 文字列型を想定
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = tokenizer(text, padding=True, truncation=truncation, return_tensors='pt')#.to(device)
    
    with torch.no_grad():
        output = model(batch['input_ids'], attention_mask=batch['attention_mask'])
        prediction = F.softmax(output.logits, dim=1)
        label = torch.max(output.logits, dim=1)
        value = label.values.item()
        index = label.indices.item()
    if (value < 0):
        print(f'<<< logit is minus. [{text}:{model.config.id2label[index]}:{value}] >>>')
    logit_value = value
    pred_value = torch.max(prediction).item()
    logit_score = value * coef_array[index]
    pred_score = pred_value * coef_array[index]
    label_name = model.config.id2label[index].upper()
    return logit_value, pred_value, logit_score, pred_score, label_name

# 小説データのテキストを取得
"""
Args:
    file_path
    skip_row: default=1
Returns:
    data frame
"""
def get_novel_text(file_path, skip_row=1):
    novel = pd.read_csv(file_path)
    length = len(novel)
    sentences = []
    for idx in np.arange(0, length, skip_row):
        sentence = ''.join(novel['text'].values[idx:idx+skip_row])
        sentences.append(sentence)
    # DataFrameを返す
    df = pd.DataFrame(sentences, columns=['text'])
    df.insert(0, 'type', ['本文' for i in range(len(sentences))]) # typeはすべて"本文"とする
    return df

# 小説のスコアを取得する関数
"""
Arg:
    file_path
Returns:
    data frame
"""
def get_novel_score(tokenizer, model, file_path, skip_row=1, truncation=True, alignment=True):
    # df = pd.read_csv(file_path)
    df = get_novel_text(file_path, skip_row)
    logit_value_list, pred_value_list = [], []
    logit_score_list, pred_score_list, label_name_list = [], [], []
    for i, text in enumerate(tqdm(df['text'])):
        logit_value, pred_value, logit_score, pred_score, label_name = get_score(tokenizer, model, text, truncation)
        logit_value_list.append(logit_value)
        pred_value_list.append(pred_value)
        logit_score_list.append(logit_score)
        pred_score_list.append(pred_score)
        label_name_list.append(label_name)

    df['logit_value'] = logit_value_list
    df['pred_value'] = pred_value_list
    df['logit_score'] = logit_score_list
    df['pred_score'] = pred_score_list
    df['label_name'] = label_name_list
    # logitが負の場合に対応
    diff = abs(df[df['logit_value']<0]['logit_value'].min())
    if diff > 0 and alignment:
        print('<< execute alignment. >>')
        df['logit_score'] = df['logit_score'] +  df['label_name'].apply(lambda d: diff if d != 'NEUTRAL' else 0)

    return df

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

# スコアを0-1に規格化
def score_normalize(scores):
    return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))