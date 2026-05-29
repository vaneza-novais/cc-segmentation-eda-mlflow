from pathlib import Path


from loguru import logger
from tqdm import tqdm
import typer

import mlflow # biblioteca do MLflow
import sklearn

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

# Definir e configurar tracking (MLflow)
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment('credit_card_segmentation')

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):
#%%


#%%

if __name__ == "__main__":
    app()
