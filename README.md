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
    - [data/all_score_0805.csv (1,149)](data/all_score_0805.csv)
    - [data/all_score_1027.csv (1,149) *latest](data/all_score_1027.csv)
- FineTuning(最終版)
  - [pattern1: 公開されている事前学習済みモデル](https://huggingface.co/koheiduck/bert-japanese-finetuned-sentiment)
  - wrimeデータを使用したfinetuning
    - [pattern2: ベースモデル(ハイパーパラメータ固定)](finetuning_wrime_01_base.py)
    - [pattern3: Optunaを使用してハイパーパラメータを探索したモデル](finetuning_wrime_02_optuna.py)
  - multilingual-sentimentsデータを使用したfinetuning
    - [pattern4:ベースモデル(ハイパーパラメータ固定)](finetuning_multilingual_01_base.py)
    - [pattern5:Optunaを使用してハイパーパラメータを探索したモデル](finetuning_multilingual_02_adamw.py)
- FineTuning
  - wrimeデータを使用したfinetuning
    - [FineTuning_v1.ipynb](FineTuning_v1.ipynb) .. 試行錯誤用
    - [FineTuning_v1.py](FineTuning_v1.py) .. バックグラウンド実行用
    - [finetune.sh](finetune.sh) .. バックグラウンド実行用シェル
    - [FineTuning_wrime_v1.ipynb](FineTuning_wrime_v1.ipynb)
    - [FineTuning_wrime_v1.py](FineTuning_wrime_v1.ipynb)
    - [finetune_make_paramlist.ipynb](finetune_make_paramlist.ipynb) .. パラメータリスト作成用
      - data/hyper_parametersxxx.csv .. 作成されたリストやテスト用に編集したもの
  - wrimeデータを使用したfinetuning(optuna使用)
    - [finetuning_wrime_01_base.ipynb](finetuning_wrime_01_base.ipynb) .. ベースライン
    - [finetuning_wrime_02_optuna.ipynb](finetuning_wrime_02_optuna.ipynb) .. パラメータ最適化(試行錯誤用)
    - [finetuning_wrime_02_optuna.py](finetuning_wrime_02_optuna.py) .. パラメータ最適化(実行用)
    - [finetuning_wrime_03_optuna_bestrun.ipynb](finetuning_wrime_03_optuna_bestrun.ipynb) .. パラメータの最適な組み合わせでfinetuning
  - multilingual-sentimentsデータを使用したfinetuning(optuna使用)
    - [finetuning_multilingual_01_base.ipynb](finetuning_multilingual_01_base.ipynb) .. ベースライン
    - [finetuning_multilingual_01_base.py](finetuning_multilingual_01_base.py) .. ベースライン(実行用)
    - [finetuning_multilingual_02_adamw.ipynb](finetuning_multilingual_02_adamw.ipynb) .. adamwを使用したfientuning
    - [finetuning_multilingual_02_adamw.py](finetuning_multilingual_02_adamw.py) .. 同上(実行用)
    - [finetuning_multilingual_02_adafactor.py](finetuning_multilingual_02_adafactor.py) .. adafactorを使用したfinetuning

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

|                                                          |                          | 作品名           | 　           | 　             | 　       | 　      |
|----------------------------------------------------------|--------------------------|------------------|--------------|----------------|----------|---------|
| モデル                                                   | ハイパーパラメータ最適化 | フランダースの犬 | 銀河鉄道の夜 | 押絵と旅する男 | 合計     | 平均    |
| koheiduck/bert-japanese-finetuned-sentiment              | ー                       | 8.2              | 9.5          | 19.3           | 37.0     | 12.3    |
| **A-Funakoshi/bert-multilingual-sentiments-base**        | **なし**                 | **9.8**          | **10.5**     | **9.3**        | **29.7** | **9.9** |
| A-Funakoshi/bert-finetuned-multilingual-sentiments-adamw | あり                     | 9.1              | 10.8         | 21.1           | 41.0     | 13.7    |
| A-Funakoshi/bert-wrime-base                              | なし                     | 9.0              | 8.6          | 19.2           | 36.8     | 12.3    |
| A-Funakoshi/bert-base-japanese-v3-wrime-v2               | あり                     | 8.2              | 8.9          | 19.5           | 36.5     | 12.2    |