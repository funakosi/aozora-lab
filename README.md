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
- FineTuning
  - wrimeデータを使用したfine tuning
    - [FineTuning_v1.ipynb](FineTuning_v1.ipynb) .. 試行錯誤用
    - [FineTuning_v1.py](FineTuning_v1.py) .. バックグラウンド実行用
    - [finetune.sh](finetune.sh) .. バックグラウンド実行用シェル
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
  - settingsフォルダ以下に各種設定ファイルを配置
  - libraryフォルダ以下で関数をモジュール化
    - dtw.py .. オライリー「実践　時系列解析」に記載されていたdtw関数を定義
    - preprocess.py .. データの前処理用関数
    - score.py .. スコア取得用関数
    - tool .. 設定ファイル読み込み等の関数
  - [感情曲線取得.ipynb](感情曲線取得.ipynb)
    - 小説の取得，クレンジング，スコア取得までを試せるコード
  - [感情曲線取得v2.ipynb](感情曲線取得v2.ipynb)
    - モデルの比較で使用している３小説の感情曲線を表示

## モデルの比較

先行研究2.の筆者から共有していただいた３小説（フランダースの犬，銀河鉄道の夜，押絵と旅する男）の感情スコア値と，各モデルで取得した感情スコア値を[PyTS DTW](https://pyts.readthedocs.io/en/stable/generated/pyts.metrics.dtw.html#pyts.metrics.dtw)で比較したもの．モデル名はHugging Faceで公開されているモデルの名前．

|                                                    | 作品名           | 　           | 　             | 　   | 　   |
|----------------------------------------------------|------------------|--------------|----------------|------|------|
| モデル                                             | フランダースの犬 | 銀河鉄道の夜 | 押絵と旅する男 | 合計 | 平均 |
| koheiduck/bert-japanese-finetuned-sentiment        | 8.5              | 8.3          | 18.1           | 34.9 | 11.6 |
| A-Funakoshi/bert-finetuned-multilingual-sentiments | 9.6              | 9.3          | 12.2           | 31.1 | 10.4 |
| A-Funakoshi/bert-base-japanese-v3-wrime-sentiment  | 7.95             | 7.19         | 19.2           | 34.3 | 11.4 |