"""
Helper module for adding progress bars to GridSearchCV parameter iteration.
This can be imported into your Jupyter notebook to add progress tracking.
"""

import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.model_selection._search import _check_param_grid
from itertools import product

class GridSearchCVWithProgress(GridSearchCV):
    """
    GridSearchCV with progress bars for parameter iteration.
    
    This class extends sklearn's GridSearchCV to show progress bars
    during parameter grid search.
    """
    
    def fit(self, X, y=None, **fit_params):
        """
        Fit the model with progress tracking.
        
        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like, optional
            Target values
        **fit_params : dict
            Additional fit parameters
            
        Returns:
        --------
        self : object
            Returns self
        """
        # 检查参数网格
        _check_param_grid(self.param_grid)
        
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

def create_gridsearch_with_progress(estimator, param_grid, cv=5, scoring=None, **kwargs):
    """
    Create a GridSearchCV instance with progress bars.
    
    Parameters:
    -----------
    estimator : estimator object
        The estimator to be optimized
    param_grid : dict
        Dictionary with parameter names as keys and lists of parameter values
    cv : int, cross-validation generator or an iterable, default=5
        Cross-validation splitting strategy
    scoring : str, callable, list/tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model
    **kwargs : dict
        Additional arguments passed to GridSearchCV
        
    Returns:
    --------
    GridSearchCVWithProgress : object
        GridSearchCV instance with progress tracking
    """
    return GridSearchCVWithProgress(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        **kwargs
    )

# 使用示例：
"""
# 在你的notebook中，你可以这样使用：

# 1. 导入这个模块
from gridsearch_progress_helper import create_gridsearch_with_progress

# 2. 替换原来的GridSearchCV
# 原来的代码：
# grid = GridSearchCV(model, param_grids[name], cv=skf, scoring="roc_auc", n_jobs=-1, verbose=1, return_train_score=False)

# 新的代码：
grid = create_gridsearch_with_progress(
    model, 
    param_grids[name], 
    cv=skf, 
    scoring="roc_auc", 
    n_jobs=1,  # 注意：使用n_jobs=1来确保进度条正常工作
    verbose=0, 
    return_train_score=False
)

# 3. 拟合模型（现在会显示参数迭代进度）
grid.fit(X, y)
""" 