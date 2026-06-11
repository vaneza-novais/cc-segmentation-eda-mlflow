from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

# Definir e configurar tracking (MLflow)
import mlflow # biblioteca do MLflow
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment('credit_card_classifier')

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / 'features_test_cluster.csv',
    model_path: Path = MODELS_DIR / 'model_lr.pkl',
    predictions_path: Path = PROCESSED_DATA_DIR / 'test_predictions_cluster.csv',
    # -----------------------------------------
):
    #---------------------------------------------------------------------
    # 1. Carregar os dados
    #---------------------------------------------------------------------
    import pandas as pd

    try:
        df_test = pd.read_csv(features_path)
        logger.info(f'Dataset Teste carregado com sucesso. Formato inicial: {df_test.shape}')
    except Exception as e:
        logger.error(f'Erro a ler arquivo {e}')
        # forçar o encerramento e dizer ao sistema que tem algo errado
        raise typer.Exit(code=1)
    
    #---------------------------------------------------------------------
    # 2. Predição dos dados de teste
    #---------------------------------------------------------------------
    import joblib

    k = 6 # Definido no trreino

    logger.info('Aplicando o pipeline em teste.')
    pipeline_test = joblib.load(model_path)
    test_labels = pipeline_test.predict(df_test)

    # Inseriondo os labels como nova coluna
    df_test['cluster'] = test_labels

    #---------------------------------------------------------------------
    # 3. Salvar artefatos locais e MLflow
    #---------------------------------------------------------------------
    # Iniciando MLflow
    with mlflow.start_run(run_name = 'kmeans_test'):

        # Salvar o csv da predição
        df_test.to_csv(predictions_path, index=False)

        logger.success(f'Base com prediçoes salva localmente.')

        # Salvar os parametros
        mlflow.log_param('numb_user_test', df_test.shape[0])

        # Enviar o relatorio de predição para o MLflow
        mlflow.log_artifact(
            local_path= predictions_path, #oirgem local
            artifact_path= 'prediction_artifacts' # servidor
            )
        
        logger.success('Pipeline de predição concluído!')

if __name__ == '__main__':
    app()