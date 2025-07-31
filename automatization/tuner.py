from __future__ import annotations
from typing import Dict
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import logging

logger = logging.getLogger(__name__)


def build_searchers(
    pipelines: Dict[str, object],
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Construye los objetos de búsqueda de hiperparámetros para los modelos.
    :param pipelines: Diccionario con pipelines de modelos.
    :param cv: Número de folds para validación cruzada.
    :param random_state: Semilla para reproducibilidad en RandomizedSearchCV.
    :return: Diccionario con objetos de búsqueda de hiperparámetros.
    """
    logger.info("Construyendo objetos de búsqueda de hiperparámetros")

    # GridSearch para Regresión Logística y Árbol de Decisión
    grid_params = {
        "logistic_regression": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__solver": ["liblinear"],
        },
        "decision_tree": {
            "clf__max_depth": [5, 10, None],
            "clf__min_samples_split": [2, 5],
        },
    }

    # RandomizedSearch para Random Forest y XGBoost
    rand_params = {
        "random_forest": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 5],
        },
        "xgboost": {
            "clf__n_estimators": [300, 400],
            "clf__max_depth": [4, 6],
            "clf__learning_rate": [0.05, 0.1],
        },
    }

    searchers: Dict[str, object] = {}

    # GridSearchCV
    for name, params in grid_params.items():
        searchers[name] = GridSearchCV(
            pipelines[name],
            param_grid=params,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )

    # RandomizedSearchCV
    for name, params in rand_params.items():
        searchers[name] = RandomizedSearchCV(
            pipelines[name],
            param_distributions=params,
            n_iter=8,
            cv=cv,
            scoring="roc_auc",
            random_state=random_state,
            n_jobs=-1,
            verbose=0,
        )

    # Añadimos la red neuronal sin tuning
    searchers["neural_network"] = pipelines["neural_network"]

    logger.info("Buscadores listos: %s", list(searchers))
    return searchers
