import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_data(path: str | Path) -> pd.DataFrame:
    """
    Carga un dataset desde un archivo CSV y valida su contenido.
    :param path: Ruta al archivo CSV.
    :return: pd.DataFrame con los datos cargados.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        logger.error("CSV no encontrado: %s", csv_path.resolve())
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if "is_canceled" not in df.columns:
        logger.error("Falta la columna objetivo 'is_canceled'")
        raise ValueError("La columna 'is_canceled' no está en el dataset")

    logger.info("Datos cargados: %s, %s filas × %s columnas",
                csv_path.name, df.shape[0], df.shape[1])
    return df




