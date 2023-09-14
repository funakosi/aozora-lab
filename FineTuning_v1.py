import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timedelta, timezone
from datetime import datetime, timedelta, timezone
import torch

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score

# GPUが使用可能か判断
if torch.cuda.is_available():
    print('gpu is available')
else:
    raise Exception('gpu is NOT available')

# Goole Colab環境か判断
def is_colabo():
    moduleList = sys.modules
    if 'google.colab' in moduleList:
        return True
    else:
        return False

# メトリクスの定義
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def show_graph(df, parameters, regression, output='output.png'):
    suptitle_size = 23
    graph_title_size = 20
    legend_size = 18
    ticks_size = 13
    # 学習曲線
    fig = plt.figure(figsize=(20, 5))
    plt.suptitle(','.join([f'{e}: {parameters[e]}' for e in parameters.keys()]), fontsize=suptitle_size)
    # Train Loss
    plt.subplot(131)
    plt.title('Train Loss', fontsize=graph_title_size)
    plt.plot(df['loss'].dropna(), label='train')
    plt.legend(fontsize=legend_size)
    plt.yticks(fontsize=ticks_size)
    # Validation Loss
    plt.subplot(132)
    reg_str = f'$y={round(regression.coef_[0],5)}*x+{round(regression.intercept_,3)}$'
    plt.title(f'Val Loss', fontsize=graph_title_size)
    y = df['eval_loss'].dropna().values
    x = np.arange(len(y)).reshape(-1, 1)
    pred = regression.coef_ * x.ravel() + regression.intercept_  # 線形回帰直線
    plt.plot(y, color='tab:orange', label='val')
    plt.plot(pred, color='green', label='pred')
    plt.legend(fontsize=legend_size)
    plt.xlabel(reg_str, fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    # Accuracy/F1
    plt.subplot(133)
    plt.title('Accuracy/F1', fontsize=graph_title_size)
    plt.plot(df['eval_accuracy'].dropna(), label='accuracy')
    plt.plot(df['eval_f1'].dropna(), label='F1')
    plt.legend(fontsize=legend_size)
    plt.yticks(fontsize=ticks_size)
    plt.tight_layout()
    plt.show()
    plt.savefig(output)

def get_train_arg(output_dir, hyper_parameter, epoch_num = 2):
    if epoch_num <= 20:
        args = TrainingArguments(
            output_dir=output_dir,    # 結果の保存フォルダ
            # label_names=['label'],  # ラベルの名前
            num_train_epochs=epoch_num,      # エポック数
            learning_rate=hyper_parameter['lr'],       # 学習率
            per_device_train_batch_size=hyper_parameter['batch'], # 訓練時のバッチサイズ
            per_device_eval_batch_size=hyper_parameter['batch'],  # 評価時のバッチサイズ
            weight_decay=0.01,            # 正則化
            disable_tqdm=False,           # プログレスバー非表示
            logging_strategy="steps",     # ロギングのタイミング
            logging_steps=0.05, # ロギングのタイミング [0,1)
            evaluation_strategy="steps",  # 検証セットによる評価のタイミング
            save_strategy="steps",        # チェックポイントの保存タイミング
            push_to_hub=False,            # Huggin Haceにpushしない
            # load_best_model_at_end=True,  # 訓練後に開発セットで最良のモデルをロード
            metric_for_best_model="accuracy",  # 最良のモデルを決定する評価指標
            log_level="error",            # ログレベル
            fp16=True,                    # 自動混合精度演算の有効化
            lr_scheduler_type=hyper_parameter['type'], # "constant", "linear", "cosine"
        )
    else:
        args = TrainingArguments(
            output_dir=output_dir,    # 結果の保存フォルダ
            # label_names=['label'],  # ラベルの名前
            num_train_epochs=epoch_num,      # エポック数
            learning_rate=hyper_parameter['lr'],       # 学習率
            per_device_train_batch_size=hyper_parameter['batch'], # 訓練時のバッチサイズ
            per_device_eval_batch_size=hyper_parameter['batch'],  # 評価時のバッチサイズ
            weight_decay=0.01,            # 正則化
            disable_tqdm=False,           # プログレスバー非表示
            logging_strategy="epoch",     # ロギングのタイミング
            logging_steps=0.05, # ロギングのタイミング [0,1)
            evaluation_strategy="epoch",  # 検証セットによる評価のタイミング
            save_strategy="epoch",        # チェックポイントの保存タイミング
            push_to_hub=False,            # Huggin Haceにpushしない
            load_best_model_at_end=True,  # 訓練後に開発セットで最良のモデルをロード
            metric_for_best_model="accuracy",  # 最良のモデルを決定する評価指標
            log_level="error",            # ログレベル
            fp16=True,                    # 自動混合精度演算の有効化
            lr_scheduler_type=hyper_parameter['type'], # "constant", "linear", "cosine"
        )
    return args

def linear_regression(history_df):
    y = history_df['eval_loss'].dropna().values
    x = np.arange(len(y)).reshape(-1, 1)
    linear = LinearRegression().fit(x, y)
    return linear

def result_filename():
    JST = timezone(timedelta(hours=+9), 'JST')
    filename = f"result/result_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}.csv"
    return filename

def result_dirname():
    JST = timezone(timedelta(hours=+9), 'JST')
    dirname = f"result/result_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}"
    return dirname

# パターン選択
# Multilingual Sentiments Dataset
pattern1 = {
    'dataset_path': 'tyqiangz/multilingual-sentiments',
    'dataset_name': 'japanese',
    'text_column': 'text',  # トークナイズの対象列
    'save_dir': 'result/finetune_result1', # 結果を保存するフォルダ
}
# 主観と客観の感情分析データセット
pattern2 = {
    'dataset_path': 'llm-book/wrime-sentiment',
    'dataset_name': '',
    'text_column': 'sentence',
    'save_dir': 'result/finetune_result2' # 結果を保存するフォルダ
}

# どのパターンを使用するか決める
target_pattern = pattern2
print(target_pattern)

# データセットのロード
print(f"dataset :{target_pattern['dataset_path']}")
dataset = load_dataset(target_pattern['dataset_path'],
                       target_pattern['dataset_name'])

# 実験のためデータセットを縮小したい場合はコチラを有効化
# random.seed(42)
# dataset = DatasetDict({
#    "train": dataset['train'].select(
#        random.sample(range(dataset['train'].num_rows), k=500)),
#    "validation": dataset['validation'].select(
#        random.sample(range(dataset['validation'].num_rows), k=500)),
#    "test": dataset['test'].select(
#        random.sample(range(dataset['test'].num_rows), k=500)),
# })

# トークナイザのロード
model_ckpt = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# トークナイズ処理
def tokenize(batch):
    return tokenizer(batch[target_pattern['text_column']], padding=True, truncation=True, max_length=100)
dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)

# 事前学習モデルのロード
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = len(set(dataset['train']['label']))
model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels=num_labels)
)

# チューニング対象
# type: constant, (linear, cosine)
# lr: 5e-6, 7e-6, 1e-5, 1.5e-5, 2e-5
# batch size: 16, 32, 64, 128, 256
# weith_decay: 0, 1e-4, 5e-4, 1e-5
hyper_parameters = [
    {'epoch': 2, 'type': 'constant', 'lr': 5.0e-6, 'batch': 16},
    # {'epoch': 2, 'type': 'constant', 'lr': 5.0e-6, 'batch': 32},
    # {'epoch': 2, 'type': 'constant', 'lr': 5.0e-6, 'batch': 64},
    # {'epoch': 2, 'type': 'constant', 'lr': 5.0e-6, 'batch': 128},
    # {'epoch': 2, 'type': 'constant', 'lr': 5.0e-6, 'batch': 256},
    # {'epoch': 2, 'type': 'constant', 'lr': 7.5e-6, 'batch': 16},
    # {'epoch': 2, 'type': 'constant', 'lr': 7.5e-6, 'batch': 32},
    # {'epoch': 2, 'type': 'constant', 'lr': 7.5e-6, 'batch': 64},
    # {'epoch': 2, 'type': 'constant', 'lr': 7.5e-6, 'batch': 128},
    # {'epoch': 2, 'type': 'constant', 'lr': 7.5e-6, 'batch': 256},
    # {'epoch': 2, 'type': 'constant', 'lr': 1.0e-5, 'batch': 16}, 
    # {'epoch': 2, 'type': 'constant', 'lr': 1.0e-5, 'batch': 32},
    # {'epoch': 2, 'type': 'constant', 'lr': 1.0e-5, 'batch': 64}, 
    # {'epoch': 2, 'type': 'constant', 'lr': 1.0e-5, 'batch': 128},
    # {'epoch': 2, 'type': 'constant', 'lr': 1.0e-5, 'batch': 256},
    # {'epoch': 2, 'type': 'constant', 'lr': 1.25e-5, 'batch': 16},
    # {'epoch': 2, 'type': 'constant', 'lr': 1.25e-5, 'batch': 32},
    # {'epoch': 2, 'type': 'constant', 'lr': 1.25e-5, 'batch': 64},
    # {'epoch': 2, 'type': 'constant', 'lr': 1.25e-5, 'batch': 128}, 
    # {'epoch': 2, 'type': 'constant', 'lr': 1.25e-5, 'batch': 256}, 
    # {'epoch': 2, 'type': 'constant', 'lr': 1.5e-5, 'batch': 16},   
    # {'epoch': 2, 'type': 'constant', 'lr': 1.5e-5, 'batch': 32}, 
    # {'epoch': 2, 'type': 'constant', 'lr': 1.5e-5, 'batch': 64}, 
    # {'epoch': 2, 'type': 'constant', 'lr': 1.5e-5, 'batch': 128},
    # {'epoch': 2, 'type': 'constant', 'lr': 1.5e-5, 'batch': 256},
    # {'epoch': 2, 'type': 'constant', 'lr': 1.75e-5, 'batch': 16}, 
    # {'epoch': 2, 'type': 'constant', 'lr': 1.75e-5, 'batch': 32}, 
    # {'epoch': 2, 'type': 'constant', 'lr': 1.75e-5, 'batch': 64}, 
    # {'epoch': 2, 'type': 'constant', 'lr': 1.75e-5, 'batch': 128},
    # {'epoch': 2, 'type': 'constant', 'lr': 1.75e-5, 'batch': 256},
    # {'epoch': 2, 'type': 'constant', 'lr': 2.0e-5, 'batch': 16},  
    # {'epoch': 2, 'type': 'constant', 'lr': 2.0e-5, 'batch': 32}, 
    # {'epoch': 2, 'type': 'constant', 'lr': 2.0e-5, 'batch': 64}, 
    # {'epoch': 2, 'type': 'constant', 'lr': 2.0e-5, 'batch': 128},
    # {'epoch': 2, 'type': 'constant', 'lr': 2.0e-5, 'batch': 256},
]

dir_name = result_dirname()
os.makedirs(dir_name)

# 学習パラメータの設定
model_name = target_pattern['save_dir']

for idx, target in enumerate(hyper_parameters):
    hyper_parameter = hyper_parameters[idx]
    print(idx, hyper_parameter)

    # 訓練用の設定
    training_args = get_train_arg(model_name, hyper_parameter, epoch_num=hyper_parameter['epoch'])
    # Trainerの定義
    trainer = Trainer(
        model=model,                                # モデル
        args=training_args,                         # 学習パラメータ
        compute_metrics=compute_metrics,            # メトリクス
        train_dataset=dataset_encoded["train"],     # 訓練データ
        eval_dataset=dataset_encoded["validation"], # 検証データ
        tokenizer=tokenizer                         # トークナイザ
    )

    # トレーニング実行
    trainer.train()

    # ログ
    history_df = pd.DataFrame(trainer.state.log_history)

    # 線形回帰
    reg = linear_regression(history_df)

    # 結果を表示
    show_graph(history_df, hyper_parameter, reg, f'{dir_name}/output{idx+1}.png')

    # ログ追加
    hyper_parameter['train_loss'] = history_df['loss'].dropna().values
    hyper_parameter['val_loss'] = history_df['eval_loss'].dropna().values
    hyper_parameter['coef'] = 1 if reg.coef_[0] > 0 else -1

    # break

# 結果を保存
result_df = pd.DataFrame(hyper_parameters)
filename = f"{dir_name}/result.csv"
result_df.to_csv(filename)
print(f"save file: {filename}")

