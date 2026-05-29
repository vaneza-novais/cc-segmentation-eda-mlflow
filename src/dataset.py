#%%
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd
import numpy as np

from config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "CC_GENERAL.csv",
    output_path_train: Path = INTERIM_DATA_DIR / "dataset_train.csv",
    output_path_test: Path = INTERIM_DATA_DIR / "dataset_test.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")

    #---------------------------------------------------------------------
    # 1. Carregar os dados
    #---------------------------------------------------------------------
    try:
        df= pd.read_csv(input_path)
        logger.info(f'Dataset carregado com sucesso. Formato inicial: {df.shape}')
    except Exception as e:
        logger.error(f'Erro a ler arquivo {e}')
        # forçar o encerramento e dizer ao sistema que tem algo errado
        raise typer.Exit(code=1)
    
    #---------------------------------------------------------------------
    # 2. Limpeza dos dados
    #---------------------------------------------------------------------
    shape1 = df.shape[0] # so pra controle
    df = df.drop_duplicates() # remover duplicadas
    shape2 = df.shape[0] # so pra controle

    df = df.drop(columns= 'CUST_ID') # deletar coluna de ID

    logger.info(f'{shape1 - shape2} Registros deletados')
#%%
    #---------------------------------------------------------------------
    # 3. Separar em train e test
    #---------------------------------------------------------------------
    from sklearn.model_selection import train_test_split

    data_train, data_test = train_test_split(
        df, test_size= 0.2, random_state= 0
    )

    # Salvar na pasta interim
    data_train.to_csv(output_path_train, index = False)
    data_test.to_csv(output_path_test, index = False)

    logger.success('Etapa de criação de separação Treino/Teste Interim concluída.')

if __name__ == "__main__":
    app()
