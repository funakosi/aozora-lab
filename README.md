# aozora-lab

## 青空文庫を使用した感情分析

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
  - スコアが取得できていない作品を抽出(153件)
- 行数が500行以上に絞り込み → data/target2.csv(1,050)
  - target.csv or target2.csv length >= 500
- [感情分析v4 → data/all_score_0630.csv (1,036)](感情分析v4.ipynb)
  - 感情スコア取得
  - [setitment_analysis_v4.py](setitment_analysis_v4.py)
  - ipynb形式をpy形式にしたもの(nohupとか使いたかったので)
- [特異値分解](特異値分解v1.ipynb)
  - PCA,SVD,NMFで行列分解したもの
  - [NMFで分解](NMF.ipynb)
- 時系列専用のクラスタリング
  - [TimeSeriesKMeans](TimeSeriesKMeans.ipynb)