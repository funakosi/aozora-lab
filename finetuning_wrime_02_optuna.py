# %%
import torch
# GPUが使用可能か判断
if torch.cuda.is_available():
    print('gpu is available')
else:
    raise Exception('gpu is NOT available')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
import torch
import random

# %%
from transformers.trainer_utils import set_seed

# 乱数シードを42に固定
set_seed(42)

# %%
from pprint import pprint
from datasets import load_dataset

# Hugging Face Hub上のllm-book/wrime-sentimentのリポジトリから
# データを読み込む
train_dataset = load_dataset("llm-book/wrime-sentiment", split="train", remove_neutral=False)
valid_dataset = load_dataset("llm-book/wrime-sentiment", split="validation", remove_neutral=False)
# pprintで見やすく表示する
pprint(train_dataset)
pprint(valid_dataset)

# %%
# トークナイザのロード
model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%
# トークナイズ処理
def preprocess_text(batch):
    encoded_batch = tokenizer(batch['sentence'], max_length=100)
    encoded_batch['labels'] = batch['label']
    return encoded_batch

# %%
encoded_train_dataset = train_dataset.map(
    preprocess_text,
    remove_columns=train_dataset.column_names,
)
encoded_valid_dataset = valid_dataset.map(
    preprocess_text,
    remove_columns=valid_dataset.column_names,
)

# %%
# ミニバッチ構築
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# %%
# オプティマイザ
OPTIMIZER_NAME = "adamw_torch"

# 最適化するハイパーパラメータ
def optuna_hp_space(trial):
    return {
        "lr_scheduler_type": trial.suggest_categorical("lr_scheduler_type", ["constant", "linear", "cosine"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128, 256]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
    }

# %%
# モデルの準備
from transformers import AutoModelForSequenceClassification

def model_init(trial):
    class_label = train_dataset.features["label"]
    label2id = {label: id for id, label in enumerate(class_label.names)}
    id2label = {id: label for id, label in enumerate(class_label.names)}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=class_label.num_classes,
        label2id=label2id,  # ラベル名からIDへの対応を指定
        id2label=id2label,  # IDからラベル名への対応を指定
    )
    return model

# %%
# 訓練の実行
from transformers import TrainingArguments

training_args = TrainingArguments(
    optim=OPTIMIZER_NAME,  # オプティマイザの種類
    output_dir="output_wrime",  # 結果の保存フォルダ
    # per_device_train_batch_size=32,  # 訓練時のバッチサイズ
    # per_device_eval_batch_size=32,  # 評価時のバッチサイズ
    # learning_rate=2e-5,  # 学習率
    # lr_scheduler_type="constant",  # 学習率スケジューラの種類
    warmup_ratio=0.1,  # 学習率のウォームアップの長さを指定
    num_train_epochs=3,  # エポック数
    save_strategy="epoch",  # チェックポイントの保存タイミング
    logging_strategy="epoch",  # ロギングのタイミング
    evaluation_strategy="epoch",  # 検証セットによる評価のタイミング
    load_best_model_at_end=True,  # 訓練後に開発セットで最良のモデルをロード
    metric_for_best_model="accuracy",  # 最良のモデルを決定する評価指標
    gradient_checkpointing=True,  # 勾配チェックポイント
    fp16=True,  # 自動混合精度演算の有効化
)

# %%
# メトリクスの定義
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# %%
from transformers import Trainer

trainer = Trainer(
    model=None,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_valid_dataset,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    model_init=model_init,
)

# %%
def compute_objective(metrics):
    return metrics["eval_f1"]

# %%
best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="optuna",
    hp_space=optuna_hp_space,
    n_trials=50,
    # n_trials=3,  # TEST
    compute_objective=compute_objective,
)

# %%
print('-'*80)
# ベスト-ハイパーパラメータ
print('optimizer:',OPTIMIZER_NAME)
print(best_trial)

## 最適化されたハイパーパラメータでFineTuning

# %%
# モデルの準備
from transformers import AutoModelForSequenceClassification

class_label = train_dataset.features["label"]
label2id = {label: id for id, label in enumerate(class_label.names)}
id2label = {id: label for id, label in enumerate(class_label.names)}
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=class_label.num_classes,
    label2id=label2id,  # ラベル名からIDへの対応を指定
    id2label=id2label,  # IDからラベル名への対応を指定
)
print(type(model).__name__)

# %%
# 訓練用の設定
from transformers import TrainingArguments

# ベストパラメータ
best_lr_type = best_trial.hyperparameters['lr_scheduler_type']
best_lr = best_trial.hyperparameters['learning_rate']
best_batch_size = best_trial.hyperparameters['per_device_train_batch_size']
best_weight_decay = best_trial.hyperparameters['weight_decay']
# 保存ディレクトリ
save_dir = f'bert-finetuned-wrime-{OPTIMIZER_NAME}'

training_args = TrainingArguments(
    output_dir=save_dir,  # 結果の保存フォルダ
    optim=OPTIMIZER_NAME,  # オプティマイザの種類
    per_device_train_batch_size=best_batch_size,  # 訓練時のバッチサイズ
    per_device_eval_batch_size=best_batch_size,  # 評価時のバッチサイズ
    learning_rate=best_lr,  # 学習率
    lr_scheduler_type=best_lr_type,  # 学習率スケジューラの種類
    weight_decay=best_weight_decay,  # 正則化
    warmup_ratio=0.1,  # 学習率のウォームアップの長さを指定
    num_train_epochs=100,  # エポック数
    # num_train_epochs=3,  # エポック数 TEST
    save_strategy="epoch",  # チェックポイントの保存タイミング
    logging_strategy="epoch",  # ロギングのタイミング
    evaluation_strategy="epoch",  # 検証セットによる評価のタイミング
    load_best_model_at_end=True,  # 訓練後に開発セットで最良のモデルをロード
    metric_for_best_model="accuracy",  # 最良のモデルを決定する評価指標
    fp16=True,  # 自動混合精度演算の有効化
)

# %%
# 訓練の実施
from transformers import Trainer
from transformers import EarlyStoppingCallback

trainer = Trainer(
    model=model,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_valid_dataset,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
trainer.train()

# %%
# モデルの保存
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

# %%
# 結果描画用関数
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def show_graph(df, suptitle, output='output.png'):
    suptitle_size = 23
    graph_title_size = 20
    legend_size = 18
    ticks_size = 13
    # 学習曲線
    fig = plt.figure(figsize=(20, 5))
    plt.suptitle(suptitle, fontsize=suptitle_size)
    # Train Loss
    plt.subplot(131)
    plt.title('Train Loss', fontsize=graph_title_size)
    plt.plot(df['loss'].dropna(), label='train')
    plt.legend(fontsize=legend_size)
    plt.yticks(fontsize=ticks_size)
    # Validation Loss
    plt.subplot(132)
    plt.title(f'Val Loss', fontsize=graph_title_size)
    y = df['eval_loss'].dropna().values
    x = np.arange(len(y)).reshape(-1, 1)
    plt.plot(y, color='tab:orange', label='val')
    plt.legend(fontsize=legend_size)
    plt.yticks(fontsize=ticks_size)
    # Accuracy/F1
    plt.subplot(133)
    plt.title('eval Accuracy/F1', fontsize=graph_title_size)
    plt.plot(df['eval_accuracy'].dropna(), label='accuracy')
    plt.plot(df['eval_f1'].dropna(), label='F1')
    plt.legend(fontsize=legend_size)
    plt.yticks(fontsize=ticks_size)
    plt.tight_layout()
    plt.savefig(output)

# %%
history_df = pd.DataFrame(trainer.state.log_history)
history_df.to_csv(f'{save_dir}/history.csv')
# 結果を表示
suptitle = f'batch:16, lr:{best_lr}, batch_size: {best_batch_size}, type:{best_lr_type}, weight_decay:{best_weight_decay}'
show_graph(history_df, suptitle, f'{save_dir}/output.png')
