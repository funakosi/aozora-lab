import pandas as pd
import os
from zipfile import ZipFile
import urllib.error
import urllib.request


# 必要な特徴量のみ選択
def select_features(df_all):
    # 氏名，かな
    df_kanji = df_all['姓'] + ' ' + df_all['名']
    df_kana = df_all['姓読み'] + ' ' + df_all['名読み']
    df_name = pd.concat([df_all['人物ID'], df_kanji, df_kana], axis=1).rename(columns={0: '氏名', 1: '読み'})
    # 作品名
    df_work = df_all.loc[:, ['作品ID', '作品名', '副題', '作品名読み']]
    # URL
    df_url = df_all.loc[:, ['図書カードURL', 'テキストファイルURL']]
    # 結合
    df = pd.concat([df_name, df_work, df_url], axis=1)
    df = df.reset_index().drop(['index'], axis=1) # index振り直し
    df.insert(0, '対象', True) # 対象列の追加
    df['テキストファイルパス'] = '' # 保存場所
    df['備考'] = ''
    return df

# クレンジングされたデータを保存する
def save_text(target_file, org_dir, file_name, sep=',', force=False):
    try:
        save_file = f'{org_dir}/{file_name}'
        if os.path.isfile(save_file) and force==False:
            # Txtファイルの読み込み
            print(f'read {save_file}')
            df_org = pd.read_csv(save_file)
            pass
        else:
            # Zipファイルの読み込み
            print(f'read {target_file}')
            # Zipファイル内に複数ファイルがある場合の対応
            for file in ZipFile(target_file).namelist():
                if file.endswith('.txt'):
                    df_org = pd.read_csv(ZipFile(target_file).open(file), 
                                         encoding='cp932', names=['text'], sep=sep)
                    break
            # df_org = pd.read_csv(target_file, encoding='cp932', names=['text'], sep=sep)
            df_org.to_csv(save_file, index=False)
        return df_org
    except Exception as e:
        print(f'ERROR: {target_file}, {str(e)}')
        return None

# ファイルをダウンロード
def download_file(url, dst_path):
    try:
        with urllib.request.urlopen(url) as web_file:
            data = web_file.read()
            with open(dst_path, mode='wb') as local_file:
                local_file.write(data)
    except urllib.error.URLError as e:
        print(e)

# テキストから本文を取得
def get_text_body(target_df, author_name):
    # 本文の先頭を探す（'---…'区切りの直後から本文が始まる前提）
    head_txt = list(target_df[target_df['text'].str.contains(
        '-------------------------------------------------------')].index)
    # 本文の末尾を探す（'底本：'の直前に本文が終わる前提）
    atx = list(target_df[target_df['text'].str.contains('底本：')].index)
    if head_txt == []:
        # もし'---…'区切りが無い場合は、作家名の直後に本文が始まる前提
        head_txt = list(target_df[target_df['text'].str.contains(author_name)].index)
        head_txt_num = head_txt[0]+1
    else:
        # 2個目の'---…'区切り直後から本文が始まる
        head_txt_num = head_txt[1]+1
    return target_df[head_txt_num:atx[0]]

# 句点で区切る
def split_kuten(target_df, split_kuten=True):
    df = target_df.copy()
    # 句点で分割
    if split_kuten:
        df = target_df.assign(text=target_df['text'].str.split(r'(?<=。)(?=..)')).explode('text')
    return df

# type列追加
def add_type_column(target_df):
    df = target_df.copy()
    df.insert(loc=0, column='type', value='本文')
    df.loc[df['text'].str.contains('字下げ.*見出し*'), 'type'] = '見出し'
    return df

# 青空文庫の書式を削除
def remove_aozora_format(target_df):
    df = target_df.copy()
    # 青空文庫の書式削除
    df = df.replace({'text': {'《.*?》': ''}}, regex=True)
    df = df.replace({'text': {'［.*?］': ''}}, regex=True)
    df = df.replace({'text': {'｜': ''}}, regex=True)
    # 字下げ（行頭の全角スペース）を削除
    df = df.replace({'text': {'　': ''}}, regex=True)
    # 節区切りを削除
    # df = df.replace({'text': {'^.$': ''}}, regex=True) # 1文字
    # df = df.replace({'text': {'^―――.*$': ''}}, regex=True) # 先頭が"―――"
    df = df.replace({'text': {'^＊＊＊.*$': ''}}, regex=True)
    df = df.replace({'text': {'^×××.*$': ''}}, regex=True)
    # # 記号、および記号削除によって残ったカッコを削除
    # df = df.replace({'text': {'―': ''}}, regex=True)
    # df = df.replace({'text': {'…': ''}}, regex=True)
    # df = df.replace({'text': {'※': ''}}, regex=True)
    df = df.replace({'text': {'「」': ''}}, regex=True)
    # 一文字以下で構成されている行を削除 -> しない
    # df['length'] = df['text'].map(lambda x: len(x))
    # df = df[df['length'] > 1]
    return df

# インデックスをリセット
def reset_index(target_df):
    df = target_df.copy()
    # インデックスがずれるので振りなおす
    df = df.reset_index().drop(['index'], axis=1)

    # 空白行を削除する（念のため）
    df = df[~(df['text'] == '')]

    # インデックスがずれるので振り直し、文字の長さの列を削除する
    df = df.reset_index().drop(['index'], axis=1)
    return df