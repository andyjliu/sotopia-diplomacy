# 可以直接复制到你的Jupyter notebook中的代码

# 在导入部分添加这个自定义类
class GridSearchCVWithProgress:
    """
    带进度条的GridSearchCV实现
    可以直接替换原来的GridSearchCV
    """
    
    def __init__(self, estimator, param_grid, cv=5, scoring=None, **kwargs):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.kwargs = kwargs
        
    def fit(self, X, y=None):
        from itertools import product
        from sklearn.metrics import roc_auc_score
        
        # 计算参数组合总数
        param_combinations = 1
        for param_values in self.param_grid.values():
            param_combinations *= len(param_values)
        
        print(f"Total parameter combinations: {param_combinations}")
        print(f"Cross-validation folds: {self.cv.n_splits if hasattr(self.cv, 'n_splits') else 'custom'}")
        print(f"Total fits: {param_combinations * (self.cv.n_splits if hasattr(self.cv, 'n_splits') else 1)}")
        
        # 生成所有参数组合
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
                    
                    # 使用ROC-AUC作为评分
                    if self.scoring == "roc_auc":
                        score = roc_auc_score(y_test, y_pred_proba)
                    else:
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
                pbar.set_postfix({
                    "Best Score": f"{best_score:.3f}" if best_score else "N/A",
                    "Current": f"{mean_score:.3f}"
                })
            
            # 保存最佳结果
            self.best_score_ = best_score
            self.best_params_ = best_params
            self.best_estimator_ = best_estimator
        
        return self

# 使用方法：
# 1. 将上面的GridSearchCVWithProgress类复制到你的notebook中
# 2. 在模型训练循环中，将原来的GridSearchCV替换为GridSearchCVWithProgress

# 原来的代码：
# grid = GridSearchCV(model, param_grids[name], cv=skf, scoring="roc_auc", n_jobs=-1, verbose=1, return_train_score=False)

# 新的代码：
# grid = GridSearchCVWithProgress(model, param_grids[name], cv=skf, scoring="roc_auc")

# 然后正常调用：
# grid.fit(X, y)

# 完整的修改示例：
"""
for name, model in models.items():
    print(f"\nTuning and evaluating: {name}")
    
    # 使用带进度条的GridSearchCV
    grid = GridSearchCVWithProgress(model, param_grids[name], cv=skf, scoring="roc_auc")
    
    # 拟合模型（现在会显示参数迭代进度）
    grid.fit(X, y)
    
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV ROC-AUC: {grid.best_score_:.3f}")
    
    # 其余代码保持不变...
""" 