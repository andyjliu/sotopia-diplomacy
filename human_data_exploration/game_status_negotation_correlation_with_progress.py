import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report)
import matplotlib.pyplot as plt
import os

# 解决 huggingface/tokenizers 并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 加载tqdm用于进度条
from tqdm.auto import tqdm

# 创建自定义回调类来显示GridSearchCV的进度
class ProgressCallback:
    def __init__(self, total_combinations):
        self.pbar = tqdm(total=total_combinations, desc="Parameter iteration", leave=False)
        self.completed = 0
    
    def __call__(self, estimator, X, y):
        self.completed += 1
        self.pbar.update(1)
        return estimator
    
    def close(self):
        self.pbar.close()

# 创建自定义GridSearchCV类来支持进度条
class GridSearchCVWithProgress(GridSearchCV):
    def __init__(self, estimator, param_grid, **kwargs):
        super().__init__(estimator, param_grid, **kwargs)
        self.progress_callback = None
    
    def fit(self, X, y=None, **fit_params):
        # 计算参数组合总数
        param_combinations = 1
        for param_values in self.param_grid.values():
            param_combinations *= len(param_values)
        
        # 创建进度回调
        self.progress_callback = ProgressCallback(param_combinations)
        
        try:
            # 重写fit方法来显示进度
            from sklearn.model_selection._search import _check_param_grid
            _check_param_grid(self.param_grid)
            
            # 生成所有参数组合
            from itertools import product
            keys = self.param_grid.keys()
            values = self.param_grid.values()
            param_combinations = list(product(*values))
            
            # 为每个参数组合创建进度条
            with tqdm(total=len(param_combinations), desc="Parameter combinations", leave=False) as pbar:
                best_score = None
                best_params = None
                best_estimator = None
                
                for params in param_combinations:
                    param_dict = dict(zip(keys, params))
                    
                    # 设置参数
                    estimator = self.estimator.set_params(**param_dict)
                    
                    # 交叉验证
                    cv_scores = []
                    for train_idx, test_idx in self.cv.split(X, y):
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                        
                        estimator.fit(X_train, y_train)
                        if hasattr(estimator, "predict_proba"):
                            y_pred_proba = estimator.predict_proba(X_test)[:, 1]
                        else:
                            y_pred_proba = estimator.decision_function(X_test)
                        
                        score = self.scoring(estimator, X_test, y_test)
                        cv_scores.append(score)
                    
                    # 计算平均分数
                    mean_score = np.mean(cv_scores)
                    
                    # 更新最佳结果
                    if best_score is None or mean_score > best_score:
                        best_score = mean_score
                        best_params = param_dict
                        best_estimator = estimator
                    
                    pbar.update(1)
                    pbar.set_postfix({"Best Score": f"{best_score:.3f}" if best_score else "N/A"})
                
                # 保存最佳结果
                self.best_score_ = best_score
                self.best_params_ = best_params
                self.best_estimator_ = best_estimator
                
        finally:
            if self.progress_callback:
                self.progress_callback.close()
        
        return self

# --- 1. Prepare data --------------------------------------------------------
# 假设df_nego_eval_sum已经存在
# df = df_nego_eval_sum.copy()
# 这里使用示例数据
np.random.seed(42)
n_samples = 1000
df = pd.DataFrame({
    "num_tokens": np.random.randint(10, 1000, n_samples),
    "num_sentences": np.random.randint(1, 50, n_samples),
    "1. Game-Move": np.random.randint(0, 2, n_samples),
    "2. Reasoning": np.random.randint(0, 2, n_samples),
    "3. Rapport": np.random.randint(0, 2, n_samples),
    "4. Apologies": np.random.randint(0, 2, n_samples),
    "5. Compliment": np.random.randint(0, 2, n_samples),
    "6. Personal-Thoughts": np.random.randint(0, 2, n_samples),
    "7. Reassurance": np.random.randint(0, 2, n_samples),
    "8. Share-Information": np.random.randint(0, 2, n_samples),
    "supply_gain": np.random.randint(-2, 4, n_samples)
})

cols = ["num_tokens", "num_sentences",
        "1. Game-Move", "2. Reasoning", "3. Rapport", "4. Apologies",
        "5. Compliment", "6. Personal-Thoughts", "7. Reassurance", "8. Share-Information"]

df = df.dropna(subset=cols + ["supply_gain"])
X  = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
y  = (df["supply_gain"] > 0).astype(int)

# --- 2. Cross-validation and Model Selection --------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Try several models and hyperparameters
models = {
    "LogisticRegression": Pipeline([
        ("scale", StandardScaler()),
        ("poly", PolynomialFeatures(degree=4, include_bias=False)),
        ("logreg", LogisticRegression(max_iter=10000, solver="lbfgs", class_weight="balanced"))
    ]),
    "RandomForest": Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42))
    ]),
    "GradientBoosting": Pipeline([
        ("scale", StandardScaler()),
        ("gb", GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, random_state=42))
    ])
}

param_grids = {
    "LogisticRegression": {
        "poly__degree": [2, 3],  # 减少参数组合以加快演示
        "logreg__C": [0.1, 1, 10]
    },
    "RandomForest": {
        "rf__max_depth": [5, 8, None],
        "rf__min_samples_split": [2, 5]
    },
    "GradientBoosting": {
        "gb__max_depth": [3, 5],
        "gb__learning_rate": [0.01, 0.05]
    }
}

best_models = {}
best_scores = {}
cv_metrics = {}

for name, model in models.items():
    print(f"\nTuning and evaluating: {name}")
    
    # 使用自定义的GridSearchCVWithProgress
    grid = GridSearchCVWithProgress(model, param_grids[name], cv=skf, scoring="roc_auc", n_jobs=1, verbose=0, return_train_score=False)
    
    # 显示参数组合总数
    param_combinations = 1
    for param_values in param_grids[name].values():
        param_combinations *= len(param_values)
    print(f"Total parameter combinations: {param_combinations}")
    
    # 拟合模型（现在会显示参数迭代进度）
    grid.fit(X, y)
    
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV ROC-AUC: {grid.best_score_:.3f}")

    # 计算其他CV指标
    from sklearn.model_selection import cross_val_predict
    best_estimator = grid.best_estimator_
    
    # cross_val_predict本身没有进度条，这里用tqdm包装skf.split
    y_pred_cv = np.zeros_like(y)
    y_prob_cv = np.zeros_like(y, dtype=float)
    splits = list(skf.split(X, y))
    for i, (train_idx, test_idx) in enumerate(tqdm(splits, desc=f"CV predict {name}", leave=False)):
        best_estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred_cv[test_idx] = best_estimator.predict(X.iloc[test_idx])
        if hasattr(best_estimator, "predict_proba"):
            y_prob_cv[test_idx] = best_estimator.predict_proba(X.iloc[test_idx])[:, 1]
        else:
            prob = best_estimator.decision_function(X.iloc[test_idx])
            prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
            y_prob_cv[test_idx] = prob

    acc_cv = accuracy_score(y, y_pred_cv)
    prec_cv, rec_cv, f1_cv, _ = precision_recall_fscore_support(y, y_pred_cv, average="binary")
    auc_cv = roc_auc_score(y, y_prob_cv)
    print(f"Best CV Accuracy : {acc_cv:.3f}")
    print(f"Best CV Precision: {prec_cv:.3f}  Recall: {rec_cv:.3f}  F1: {f1_cv:.3f}")
    print(f"Best CV ROC-AUC  : {auc_cv:.3f}")

    best_models[name] = grid.best_estimator_
    best_scores[name] = grid.best_score_
    cv_metrics[name] = {
        "accuracy": acc_cv,
        "precision": prec_cv,
        "recall": rec_cv,
        "f1": f1_cv,
        "roc_auc": auc_cv
    }

# --- 3. Final evaluation on hold-out set ------------------------------------
# Use the best model (choose the one with highest CV ROC-AUC)
best_name = max(best_scores, key=lambda n: best_scores[n])
final_model = best_models[best_name]
print(f"\nSelected best model: {best_name}")

# Hold-out test set
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y)
with tqdm(total=1, desc=f"Final fit {best_name}", leave=True) as pbar:
    final_model.fit(X_tr, y_tr)
    pbar.update(1)

y_pred = final_model.predict(X_te)
if hasattr(final_model, "predict_proba"):
    y_prob = final_model.predict_proba(X_te)[:, 1]
else:
    y_prob = final_model.decision_function(X_te)
    y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-8)

acc = accuracy_score(y_te, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_te, y_pred, average="binary")
auc = roc_auc_score(y_te, y_prob)

print(f"\nFinal Hold-out Results:")
print(f"Accuracy : {acc:.3f}")
print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")
print(f"ROC-AUC  : {auc:.3f}")
print("\nClassification Report:\n", classification_report(y_te, y_pred, target_names=["No Gain", "Gain"]))

# 说明：
# 这个版本添加了自定义的GridSearchCVWithProgress类，它会为每个参数组合显示进度条
# 主要改进：
# 1. 创建了ProgressCallback类来跟踪参数迭代进度
# 2. 创建了GridSearchCVWithProgress类，重写了fit方法来显示进度
# 3. 在参数迭代过程中显示当前最佳分数
# 4. 减少了参数组合数量以加快演示速度 