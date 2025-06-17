import torch
import torch.nn as nn
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import pandas as pd
import numpy as np

# 1) 定义与之前相同的 MLP 模型
class MLP(nn.Module):
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
    def forward(self, x):
        # Trainer 会传一个 dict 或 tensor，这里统一处理
        if isinstance(x, dict):
            x = x["features"]
        out = self.net(x)
        return {"logits": out}

# 2) 从 pandas 构造 Hugging Face Dataset
feature_cols = [
    "1. Game-Move", "2. Reasoning", "3. Rapport", "4. Apologies",
    "5. Compliment", "6. Personal-Thoughts", "7. Reassurance", "8. Share-Information"
]
df = pd.read_csv('data/df_corr_binary.csv')

df_hf = df[feature_cols + ["supply_center_binary"]].rename(columns={"supply_center_binary": "labels"})
hf_ds = Dataset.from_pandas(df_hf)

# 3) 划分 train/validation
hf_ds = hf_ds.class_encode_column("labels")
hf_ds = hf_ds.train_test_split(test_size=0.2, stratify_by_column="labels", seed=42)
train_ds = hf_ds["train"]
eval_ds  = hf_ds["test"]

# 4) 在 Dataset 上添加一个“features”字段，转换为 tensor
def preprocess(batch):
    # 将 features 合并成一个 tensor
    batch["features"] = np.stack([batch[c] for c in feature_cols], axis=1).astype(np.float32)
    return batch

train_ds = train_ds.map(preprocess, batched=True)
eval_ds  = eval_ds.map(preprocess,  batched=True)

# 5) 定义计算指标的函数
def compute_metrics(p):
    logits = p.predictions["logits"] if isinstance(p.predictions, dict) else p.predictions
    probs  = torch.sigmoid(torch.from_numpy(logits)).numpy().flatten()
    preds  = (probs >= 0.5).astype(int)
    labels = p.label_ids.flatten()
    return {
        "auc": roc_auc_score(labels, probs),
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# 6) 包装成一个 nn.Module，方便 Trainer 调用
class HFMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.mlp = MLP(in_dim)
    def forward(self, features, labels=None):
        out = self.mlp({ "features": features })["logits"]
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(out.view(-1), labels.float())
        return {
            "loss": loss,
            "logits": out.view(-1)
        }

# 7) 实例化 model 与 TrainingArguments
model = HFMLP(in_dim=len(feature_cols))

training_args = TrainingArguments(
    output_dir="./checkpoints/mlp_supply_center_binary_correlation_predictions",
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=1e-4,
    num_train_epochs=80,
    load_best_model_at_end=True,
    metric_for_best_model="auc",
    greater_is_better=True,
    save_total_limit=3,  # 只保留最好的3个checkpoint
)

# 8) 初始化 Trainer 并训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    compute_metrics=compute_metrics,
    tokenizer=None,            # 不需要 tokenizer
    data_collator=None         # 默认 collator 能处理 dict of tensors
)

trainer.train()

# 9) 在 eval 集上打印最终指标
metrics = trainer.evaluate()
print(metrics)


# python mlp_supply_center_correlation.py > mlp_training_binary.log