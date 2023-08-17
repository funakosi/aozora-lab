import numpy as np

# オライリー「実践　時系列解析」に記載されていた関数
def distDTW(ts1, ts2):
    # 設定部分
    DTW = {}
    for i in range(len(ts1)):
        DTW[i, -1] = np.inf
    for i in range(len(ts2)):
        DTW[-1, i] = np.inf
    DTW[(-1, -1)] = 0

    # 1ステップずつ，最適な対応付を計算する部分
    for i in range(len(ts1)):
        for j in range(len(ts2)):
            dist = (ts1[i] - ts2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[i-1, j], DTW[(i, j-1)], DTW[(i-1, j-1)])
    
    # 完全な経路が見つかったら関連する距離を返す
    return np.sqrt(DTW[len(ts1)-1, len(ts2)-1])
