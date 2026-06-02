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
    model_path: Path = MODELS_DIR / "model_kmeans.pkl",
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
    # 2. Pipeline do K-means
    #---------------------------------------------------------------------
    from sklearn.pipeline import make_pipeline
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    k = 5 # obtido através do gráfico do cotovelo

    # Iniciando MLflow

    with mlflow.start_run(run_name = 'kmeans'):

        logger.info("Criando o pipeline do modelo.")

        pipeline_kmeans = make_pipeline(
            KMeans(n_clusters= k, random_state= 0)
        )

        #---------------------------------------------------------------------
        # 3. Treino do modelo
        #---------------------------------------------------------------------
        import joblib

        pipeline_kmeans.fit(df_processed)

        logger.success("Modelo ajustado.")

        #---------------------------------------------------------------------
        # 4. Registro das métricas
        #---------------------------------------------------------------------
        labels = pipeline_kmeans.named_steps['kmeans'].labels_
        inercia = pipeline_kmeans.named_steps['kmeans'].inertia_

        logger.info('Calcular a silhueta')

        silhueta = silhouette_score(
            df_processed, labels, random_state=0
        )

        # registro dos parametros
        mlflow.log_param('k_clusters', k)
        mlflow.log_param('random_state', 0)

        mlflow.log_metric('inertia', inercia)
        mlflow.log_metric('silhouette', silhueta)

        logger.success('Métricas e Parametros registrados.')

        #---------------------------------------------------------------------
        # 5. Salvar artefatos locais e MLflow
        #---------------------------------------------------------------------
        # salvar o modelo
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline_kmeans,  model_path)
        
        logger.success("Modelo salvo localmente.")
        # Salvar artefato do modelo
        mlflow.log_artifact(
            local_path= model_path, #oirgem local
            artifact_path= 'model_artifacts' # servidor
            )
        
        logger.success("Artefatos salvos no servidor")

if __name__ == "__main__":
    app()
