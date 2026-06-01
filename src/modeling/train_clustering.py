from pathlib import Path


from loguru import logger
from tqdm import tqdm
import typer


from src.config import MODELS_DIR, PROCESSED_DATA_DIR

# Definir e configurar tracking (MLflow)
import mlflow # biblioteca do MLflow
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment('credit_card_segmentation')

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "features_train_cluster.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    # -----------------------------------------
):

    #---------------------------------------------------------------------
    # 1. Carregar os dados
    #---------------------------------------------------------------------
    import pandas as pd

    try:
        df_processed= pd.read_csv(features_path)
        logger.info(f'Dataset Treino carregado com sucesso. Formato inicial: {df_processed.shape}')
    except Exception as e:
        logger.error(f'Erro a ler arquivo {e}')
        # forçar o encerramento e dizer ao sistema que tem algo errado
        raise typer.Exit(code=1)

    #---------------------------------------------------------------------
    # 2. Definir o numero ideal de clusters
    #---------------------------------------------------------------------
    import sklearn

    
    #---------------------------------------------------------------------
    # 3. Pipeline do K-means
    #---------------------------------------------------------------------
    import sklearn



    #---------------------------------------------------------------------
    # 4. Treino do modelo
    #---------------------------------------------------------------------

if __name__ == "__main__":
    app()
