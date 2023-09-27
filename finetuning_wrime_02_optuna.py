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
    encoded_batch = tokenizer(batch['sentence'], max_length=512)
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
    optim='adafactor',  # オプティマイザの種類
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
    n_trials=100,
    compute_objective=compute_objective,
)

# %%
print(best_trial)

