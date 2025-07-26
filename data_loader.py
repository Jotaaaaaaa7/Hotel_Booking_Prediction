import pandas as pd
from pathlib import Path

def load_data(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "is_canceled" not in df.columns:
        raise ValueError("La columna objetivo 'is_canceled' no está en el dataset.")
    return df




if __name__ == "__main__":
    a = Path('data/dataset_practica_final.csv')
    print(a, a.exists())

    b = Path('datdsdsd/aaa.csv')
    print(b, b.exists())