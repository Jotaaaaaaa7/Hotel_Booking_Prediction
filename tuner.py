from __future__ import annotations
from typing import Dict
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def build_searchers(pipelines: Dict[str, object],
                    cv: int = 5,
                    random_state: int = 42) -> Dict[str, object]:
    searchers: Dict[str, object] = {}

    # --- GridSearchCV ---
    grid_params = {
        "logistic_regression": {
            'clf__C': [0.01, 0.1, 1, 10, 100],
            'clf__solver': ['liblinear'],
        },
        "decision_tree": {
            'clf__max_depth': [5, 10, 15],
            'clf__min_samples_split': [2, 5, 10],
            'clf__criterion': ['gini', 'entropy']
        },
    }

    # --- RandomizedSearchCV ---
    rand_params = {
        "random_forest": {
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [5, 10, None],
            'clf__min_samples_split': [2, 5],
            'clf__criterion': ['gini', 'entropy']
        },
        "xgboost": {
            'clf__n_estimators':     [200, 300, 400],
            'clf__max_depth':        [4, 6, 8],
            'clf__learning_rate':    [0.01, 0.05, 0.1],
            'clf__subsample':        [0.7, 0.8, 1.0],
            'clf__colsample_bytree': [0.7, 0.8, 1.0],
            'clf__gamma':            [0, 0.1, 0.2]
        },
    }

    for name, pipe in pipelines.items():
        if name in grid_params:
            searchers[name] = GridSearchCV(
                estimator=pipe,
                param_grid=grid_params[name],
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
            )
        elif name in rand_params:
            searchers[name] = RandomizedSearchCV(
                estimator=pipe,
                param_distributions=rand_params[name],
                n_iter=5,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
                random_state=random_state,
                refit=True,
            )
        else:  # neural_net
            searchers[name] = pipe

    return searchers


