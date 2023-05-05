"""
参考: https://qiita.com/dzbt_dzbt/items/593dbd698a07c12a771c
事前準備: 以下の手順でデータを取得しておく

$ wget https://github.com/aozorahack/aozorabunko_text/archive/master.zip
$ unzip master.zip
$ mv aozorabunko_text-master bunko
$ cd bunko/cards/000148
$ find . -type f  -name '*.txt' -exec mv {} files \;
"""

import pandas as pd
from pathlib import Path

author_id = '000148'  # 青空文庫の作家番号
author_name = '夏目漱石'  # 青空文庫の表記での作家名

write_title = False # 2カラム目に作品名を入れるか
write_header = True # 1行目をカラム名にするか（カラム名「text」「title」）
save_utf8_org = True  # 元データをUTF-8にしたテキストファイルを保存するか
split_kuten = True # 句点「。」で分割するか

out_dir = Path(f'./bunko/cards/{author_id}/files/')  # ファイル出力先
tx_org_dir = Path(out_dir / './org/')  # 元テキストのUTF-8変換ファイルの保存先
tx_edit_dir = Path(out_dir / './edit/')  # テキスト整形後のファイル保存先


def text_cleanse_df(df):
    # 本文の先頭を探す（'---…'区切りの直後から本文が始まる前提）
    head_tx = list(df[df['text'].str.contains(
        '-------------------------------------------------------')].index)
    # 本文の末尾を探す（'底本：'の直前に本文が終わる前提）
    atx = list(df[df['text'].str.contains('底本：')].index)
    if head_tx == []:
        # もし'---…'区切りが無い場合は、作家名の直後に本文が始まる前提
        head_tx = list(df[df['text'].str.contains(author_name)].index)
        head_tx_num = head_tx[0]+1
    else:
        # 2個目の'---…'区切り直後から本文が始まる
        head_tx_num = head_tx[1]+1
    df_e = df[head_tx_num:atx[0]]

    # 青空文庫の書式削除
    df_e = df_e.replace({'text': {'《.*?》': ''}}, regex=True)
    df_e = df_e.replace({'text': {'［.*?］': ''}}, regex=True)
    df_e = df_e.replace({'text': {'｜': ''}}, regex=True)

    # 字下げ（行頭の全角スペース）を削除
    df_e = df_e.replace({'text': {'　': ''}}, regex=True)

    # 節区切りを削除
    df_e = df_e.replace({'text': {'^.$': ''}}, regex=True)
    df_e = df_e.replace({'text': {'^―――.*$': ''}}, regex=True)
    df_e = df_e.replace({'text': {'^＊＊＊.*$': ''}}, regex=True)
    df_e = df_e.replace({'text': {'^×××.*$': ''}}, regex=True)

    # 記号、および記号削除によって残ったカッコを削除
    # df_e = df_e.replace({'text': {'―': ''}}, regex=True)
    # df_e = df_e.replace({'text': {'…': ''}}, regex=True)
    # df_e = df_e.replace({'text': {'※': ''}}, regex=True)
    df_e = df_e.replace({'text': {'「」': ''}}, regex=True)

    # 句点で分割
    if split_kuten:
        # df.assign(コメント=df['コメント'].str.split(r'(?<=。)(?=..)')).explode('コメント')
        df_e.assign(text=df['text'].str.split(r'(?<=。)(?=.)')).explode('text')

    # 一文字以下で構成されている行を削除
    df_e['length'] = df_e['text'].map(lambda x: len(x))
    df_e = df_e[df_e['length'] > 1]

    # インデックスがずれるので振りなおす
    df_e = df_e.reset_index().drop(['index'], axis=1)

    # 空白行を削除する（念のため）
    df_e = df_e[~(df_e['text'] == '')]

    # インデックスがずれるので振り直し、文字の長さの列を削除する
    df_e = df_e.reset_index().drop(['index', 'length'], axis=1)
    return df_e


def save_cleanse_text(target_file):
    try:
        # ファイルの読み込み
        print(target_file)
        # Pandas DataFrameとして読み込む（cp932で読み込まないと異体字が読めない）
        df_tmp = pd.read_csv(target_file, encoding='cp932', names=['text'])
        # 元データをUTF-8に変換してテキストファイルを保存
        if save_utf8_org:
            out_org_file_nm = Path(target_file.stem + '_org_utf-8.tsv')
            df_tmp.to_csv(Path(tx_org_dir / out_org_file_nm), sep='\t',
                          encoding='utf-8', index=None)
        # テキスト整形
        df_tmp_e = text_cleanse_df(df_tmp)
        if write_title:
            # タイトル列を作る
            df_tmp_e['title'] = df_tmp['text'][0]
        out_edit_file_nm = Path(target_file.stem + '_clns_utf-8.txt')
        df_tmp_e.to_csv(Path(tx_edit_dir / out_edit_file_nm), sep='\t',
                        encoding='utf-8', index=None, header=write_header)
    except:
        print(f'ERROR: {target_file}')


def main():
    # zipファイルのリストを作成
    zip_list = list(out_dir.glob('*.txt'))
    # 保存ディレクトリを作成しておく
    tx_edit_dir.mkdir(exist_ok=True, parents=True)
    if save_utf8_org:
        tx_org_dir.mkdir(exist_ok=True, parents=True)

    for target_file in zip_list:
        save_cleanse_text(target_file)


if __name__ == '__main__':
    main()
