from __future__ import annotations
from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.base import ClassifierMixin
import tensorflow as tf
import logging
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import tensorflow as tf, numpy as np

logger = logging.getLogger(__name__)


# Permite usar modelos de Tensorflow/Keras como clasificadores en scikit-learn
class FixedKerasClassifier(KerasClassifier, ClassifierMixin):
    _estimator_type = "classifier"


def _build_keras_model(meta: dict) -> tf.keras.Model:
    """
    Construye un modelo Keras basado en la metadata del modelo.
    :param meta: Metadata del modelo, debe contener "n_features_in_" para definir la entrada.
    """

    # nº de características de entrada
    input_dim = meta.get("n_features_in_")

    # Arquitectura de la Red Neuronal
    # Capa de entrada que espera datos con 'input_dim' características
    # 3 capas densas con activación ReLU y regularización L2
    # 3 capas de Dropout (20%) para evitar sobreajuste
    # Capa de salida con activación sigmoide para clasificación binaria
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu",
                              kernel_regularizer=regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu",
                              kernel_regularizer=regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu",
                              kernel_regularizer=regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # Compilación del modelo
    # Usamos Adam como optimizador, binary_crossentropy como pérdida
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.AUC(name="auc")])

    return model


def get_base_pipelines(preprocessor) -> Dict[str, Pipeline]:
    """
    Crea pipelines base para los modelos de clasificación.
    :param preprocessor: Preprocesador de datos (ColumnTransformer).
    :return: Diccionario con pipelines de modelos.
    """

    logger.info("Creando pipelines base")

    return {
        "logistic_regression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, random_state=42,
                   class_weight="balanced"))
        ]),
        "decision_tree": Pipeline([
            ("prep", preprocessor),
            ("clf", DecisionTreeClassifier(random_state=42,
                       class_weight="balanced"))
        ]),
        "random_forest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(random_state=42,
                                           n_jobs=-1,
                                           class_weight="balanced"))
        ]),
        "xgboost": Pipeline([
            ("prep", preprocessor),
            ("clf", XGBClassifier(
                random_state=42,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                scale_pos_weight=0.37))
        ]),
        "neural_network": Pipeline([
            ("prep", preprocessor),
            ("clf", FixedKerasClassifier(
                model=_build_keras_model,
                epochs=200,
                batch_size=256,
                verbose=0,
                validation_split=0.2,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                class_weight="balanced",
                random_state=42,
            ))
        ]),
    }
