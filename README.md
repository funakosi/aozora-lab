# aozora-lab

## 青空文庫を使用した感情分析

- 先行研究
  - [1. The emotional arcs of stories are dominated by six basic shapes](https://arxiv.org/pdf/1606.07772.pdf)
  - [2. 物語展開を考慮した小説データからの表紙の自動生成](https://db-event.jpn.org/deim2019/post/papers/350.pdf)
- [青空文庫アクセスランキング](青空文庫データクレンジング.ipynb)
- [青空文庫対象 → data/target.csv(3,422)](青空文庫対象.ipynb)
  - 青空文庫全作品：18,798
  - 日本文学(分類番号:NDC 913)；5,181
  - 翻訳者は除外(役割フラグ:著者)：5,111
  - 新カナのみ対象(文字遣い種別:新字新仮名)：3,576
  - 作品著作権切れ(作品著作権フラグ:なし)：3,507
  - 作者名が空白は除外：3,422
- [ターゲットファイル更新  → data/target2.csv(3,422)](update_target.ipynb)
  - スコアファイルパス列追加
  - NaN→空白へ変換
  - スコアが取得できていない作品を抽出(153件→0件)
    - zipファイルに複数ファイルが存在する([対処法](https://stackoverflow.com/questions/44575251/reading-multiple-files-contained-in-a-zip-file-with-pandas))
    - 本文内にカンマが含まれている(pd.read_csvでエラー)
      - 対処法: エラーとなる作品は「@」を区切り文字として回避
    - 以下の2作品は青空文庫の書式となっていなかったので，最低限の編集を施しエラーを回避
      - 横光利一 時間
      - 横光利一 鳥
- 行数が500行以上に絞り込み → data/target2.csv(1,050→1,149)
  - target.csv or target2.csv length >= 500
- [感情分析v4](感情分析v4.ipynb)
  - 感情スコア取得
  - [setitment_analysis_v4.py](setitment_analysis_v4.py)
  - ipynb形式をpy形式にしたもの(nohup実行用)
  - [モデルの比較.ipynb](モデルの比較.ipynb)
  - 成果物ファイル
    - [data/all_score_0630.csv (1,036)](data/all_score_0630.csv)
    - [data/all_score_0731.csv (1,050)](data/all_score_0731.csv)
    - [data/all_score_0805.csv (1,149) *latest](data/all_score_0805.csv)
- [特異値分解](特異値分解v1.ipynb)
  - PCA,SVD,NMFで行列分解したもの
  - [NMFで分解](NMF.ipynb)
  - [FastICA](次元削除v1.ipynb)
- クラスタリング
  - [KernelKMeans](KernelKMeans.ipynb)
  - [KShape](KShape.ipynb)
  - [階層クラスタリング](PairwiseClustering.ipynb)
  - [KMeans,KMedoids](クラスタリングv1.ipynb)
- 時系列専用のクラスタリング
  - [TimeSeriesKMeans](TimeSeriesKMeans.ipynb)
- [self-organizing map](SOM.ipynb)
- その他
  - [感情曲線取得.ipynb](感情曲線取得.ipynb)
    - 小説の取得，クレンジング，スコア取得までを試せるコード

## モデルの比較

先行研究2.の筆者から共有していただいた３小説（フランダースの犬，銀河鉄道の夜，押絵と旅する男）の感情スコア値と，各モデルで取得した感情スコア値を[PyTS DTW](https://pyts.readthedocs.io/en/stable/generated/pyts.metrics.dtw.html#pyts.metrics.dtw)で比較したもの．

|                                             | 作品名           |              |                |        |        |
|---------------------------------------------|------------------|--------------|----------------|--------|--------|
| モデル                                      | フランダースの犬 | 銀河鉄道の夜 | 押絵と旅する男 | 合計   | 平均   |
| koheiduck/bert-japanese-finetuned-sentiment | 8.5              | 8.273        | 18.081         | 34.854 | 11.618 |
| A-Funakoshi/sample-text-classification-bert | 12.394           | 10.259       | 9.001          | 31.654 | 10.551 |