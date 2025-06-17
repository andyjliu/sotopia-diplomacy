import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1) 配置特征列和标签列
feature_cols = [
    "1. Game-Move", "2. Reasoning", "3. Rapport", "4. Apologies",
    "5. Compliment", "6. Personal-Thoughts", "7. Reassurance", "8. Share-Information"
]
label_col = "supply_gain"

# 2) 读取 CSV
df = pd.read_csv("data/df_corr_binary.csv")
df_hf = df[feature_cols + [label_col]]

# 3) 构造 Hugging Face Dataset 并随机拆分（回归无法分层）
hf_ds = Dataset.from_pandas(df_hf)
hf_split = hf_ds.train_test_split(test_size=0.2, seed=42)
train_ds, eval_ds = hf_split["train"], hf_split["test"]

# 4) 在 Dataset 上加入 “features” 和 float 型 “labels”
def preprocess(batch):
    batch["features"] = np.stack([batch[c] for c in feature_cols], axis=1).astype(np.float32)
    batch["labels"]   = np.array(batch[label_col], dtype=np.float32)
    return batch

train_ds = train_ds.map(preprocess, batched=True)
eval_ds  = eval_ds.map(preprocess,  batched=True)

# 5) 定义回归 MLP（输出单个连续值）
class MLPRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
    def forward(self, features, labels=None):
        # out: (batch_size,)
        out = self.net(features).view(-1)
        loss = None
        if labels is not None:
            # MSE 对应高斯 MLE
            loss = nn.MSELoss()(out, labels)
        return {"loss": loss, "logits": out}

# 6) 定义回归评估指标
def compute_metrics(p):
    preds  = p.predictions.flatten()
    labels = p.label_ids
    return {
        "mse": mean_squared_error(labels, preds),
        "mae": mean_absolute_error(labels, preds)
    }

# 7) 设置 TrainingArguments
training_args = TrainingArguments(
    output_dir="./checkpoints/mlp_supply_center_regression",
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=1e-4,
    num_train_epochs=80,
    load_best_model_at_end=True,
    metric_for_best_model="mse",
    greater_is_better=False,   # MSE 越小越好
    save_total_limit=3         # 只保留最好的3个checkpoint
)

# 8) 初始化 Trainer 并开始训练
model = MLPRegressor(in_dim=len(feature_cols))
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
    tokenizer=None,
    data_collator=None
)

trainer.train()
metrics = trainer.evaluate()
print(metrics)



# python mlp_supply_center_correlation_descrete_re.py > mlp_training_descrete_re.log