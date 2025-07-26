from __future__ import annotations
from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scikeras.wrappers import KerasClassifier
import tensorflow as tf

def _build_keras_model(input_dim: int = 238) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ]
    )

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    return model

def get_base_pipelines(preprocessor) -> Dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [("prep", preprocessor),
             ("clf", LogisticRegression(max_iter=1000, random_state=42))]
        ),
        "decision_tree": Pipeline(
            [("prep", preprocessor),
             ("clf", DecisionTreeClassifier(random_state=42))]
        ),
        "random_forest": Pipeline(
            [("prep", preprocessor),
             ("clf", RandomForestClassifier(random_state=42,
                                            n_jobs=-1,
                                            class_weight="balanced"))]
        ),
        "xgboost": Pipeline(
            [("prep", preprocessor),
             ("clf", XGBClassifier(random_state=42,
                                   objective="binary:logistic",
                                   eval_metric="logloss",
                                   tree_method="hist",))]
        ),
        "neural_network": Pipeline(
            [("prep", preprocessor),
             ("clf", KerasClassifier(model=_build_keras_model,
                                     epochs=200,
                                     batch_size=256,
                                     verbose=0))]
        ),
    }