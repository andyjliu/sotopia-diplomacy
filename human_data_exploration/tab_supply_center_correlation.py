import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# # 1. 读数据
# feature_cols = ["1. Game-Move","2. Reasoning","3. Rapport","4. Apologies",
#                 "5. Compliment","6. Personal-Thoughts","7. Reassurance","8. Share-Information"]
# label_col = "supply_gain"
# df = pd.read_csv("data/df_corr_binary.csv")
#
# X = df[feature_cols].values.astype("float32")
# y_raw = df[label_col].values
# le = LabelEncoder(); y = le.fit_transform(y_raw)         # 映射到 0…C-1
#
# X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=None)  # 类别极端不平衡时不要 stratify

# 1. 读数据（二分类）
feature_cols = ["1. Game-Move","2. Reasoning","3. Rapport","4. Apologies",
                "5. Compliment","6. Personal-Thoughts","7. Reassurance","8. Share-Information"]
label_col = "supply_gain"
df = pd.read_csv("data/df_corr_binary.csv")

X = df[feature_cols].values.astype("float32")
y = df[label_col].values.astype("int")  # 直接用0/1二分类label

X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None)  # 类别极端不平衡时不要 stratify

# # 2. 建模
# tabnet = TabNetClassifier(
#         n_d=16, n_a=16, n_steps=5,
#         cat_idxs=[], cat_dims=[], cat_emb_dim=1,
#         optimizer_fn=torch.optim.Adam,
#         optimizer_params=dict(lr=2e-3),
#         scheduler_params={"step_size":10, "gamma":0.9},
#         scheduler_fn=torch.optim.lr_scheduler.StepLR,
#         seed=42
# )

# 2. 建模（二分类）
tabnet = TabNetClassifier(
        n_d=16, n_a=16, n_steps=5,
        cat_idxs=[], cat_dims=[], cat_emb_dim=1,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-3),
        scheduler_params={"step_size":10, "gamma":0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        seed=42
)

# # 3. 训练
# tabnet.fit(
#     X_train, y_train,
#     eval_set=[(X_val, y_val)],
#     eval_metric=["accuracy"],
#     max_epochs=200,
#     patience=20,
#     batch_size=1024,
#     virtual_batch_size=128
# )

# 3. 训练（二分类）
tabnet.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric=["accuracy"],
    max_epochs=200,
    patience=20,
    batch_size=1024,
    virtual_batch_size=128
)

# # 4. 评估
# preds = tabnet.predict(X_val)
# print("ACC:", accuracy_score(y_val, preds),
#       "F1-macro:", f1_score(y_val, preds, average="macro"))

# 4. 评估（二分类）
preds = tabnet.predict(X_val)
print("ACC:", accuracy_score(y_val, preds),
      "F1-macro:", f1_score(y_val, preds, average="macro"))
