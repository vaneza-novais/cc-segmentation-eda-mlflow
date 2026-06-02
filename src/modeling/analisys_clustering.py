from pathlib import Path


from loguru import logger
from tqdm import tqdm
import typer


from src.config import REPORTS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions_cluster.csv",
    report_path: Path = REPORTS_DIR / "perfil_clusters.csv",
     # -----------------------------------------
):
    #---------------------------------------------------------------------
    # 1. Carregar os dados
    #---------------------------------------------------------------------
    import pandas as pd

    try:
        df_prediction= pd.read_csv(predictions_path)
        logger.info(f'Dataset Treino carregado com sucesso. Formato inicial: {df_prediction.shape}')
    except Exception as e:
        logger.error(f'Erro a ler arquivo {e}')
        # forçar o encerramento e dizer ao sistema que tem algo errado
        raise typer.Exit(code=1)
    
     #---------------------------------------------------------------------
    # 2. Criar o perfil dos clusters (groupby)
    #---------------------------------------------------------------------
    logger.info('Calulando a média de cada feature por cluster')

    media_df = df_prediction.groupby('cluster').mean()
    media_df['qtde_por_cluster'] = df_prediction['cluster'].value_counts()

    #---------------------------------------------------------------------
    # 3. Gerar o csv
    #---------------------------------------------------------------------
    media_df.to_csv(report_path)
    logger.success(f"Relatório de cluster salvo.")

if __name__ == "__main__":
    app()
