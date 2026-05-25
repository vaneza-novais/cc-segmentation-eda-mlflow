from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer

from config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = INTERIM_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    # -----------------------------------------
):
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
    # 2. Tratar os nulos
    #---------------------------------------------------------------------
    logger.info(f'Quantidade inicial de nulos: {df.isnull().sum().sum()}')

    # A. Calculo da mediana
    mediana_credit = df['CREDIT_LIMIT'].median()
    mediana_min_pay = df['MINIMUM_PAYMENTS'].median()

    # B. Criar coluna booleana a respeito de minimum_payments
    df['null_minimum_payments'] = df['MINIMUM_PAYMENTS'].isnull()

    # C. Substituir nulos pela mediana
    df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(mediana_credit)
    df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(mediana_min_pay)

    logger.info(f'Quantidade final de nulos: {df.isnull().sum().sum()}')

    #---------------------------------------------------------------------
    # 3. Escalonamento das variáveis
    #---------------------------------------------------------------------
    # A. Lista das tranformacoes  e respectivas variaveis q precisam ser transformadas
    t_log = [
        'BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 
        'CASH_ADVANCE', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 
        'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS'
    ]

    t_quantil = [
        'BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 
        'PURCHASES_INSTALLMENTS_FREQUENCY', 'PRC_FULL_PAYMENT'
    ]

    # B. Transformação Log
    transf_log = FunctionTransformer(np.log1p)
    df[t_log] = transf_log.fit_transform(df[t_log])
        
    # C. Transformação Quantil
    transf_quantil = QuantileTransformer(output_distribution= 'normal', random_state = 0)
    df[t_quantil] = transf_quantil.fit_transform(df[t_quantil])

    #---------------------------------------------------------------------
    # 4. Salvar Artefatos/Metadados e gerar Processed
    #---------------------------------------------------------------------
    import joblib
    import json
    from config import MODELS_DIR

    logger.info('Iniciando exportação de artefatos e metadados.')

    # Criando uma pasta
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # A. Salvar os arquivos binarios .pkl em models
    # Salvar o transformador quantil utilizado
    joblib.dump(transf_quantil, MODELS_DIR / 'quantile_transformer.pkl')

    # Salvar os dados de medianas
    mapa_medianas = {
        'credit_limit': float(mediana_credit),
        'minimum_payments': float(mediana_min_pay)
    }

    joblib.dump(mapa_medianas, MODELS_DIR / 'imputation_medians.pkl')

    # B. Criar o metatados
    metadata = {
        'pipeline_stage': 'feature_engineering',
        'null_totals' : int(df.isnull().sum().sum()),
        
        # Valores de imputação
        'imputation_median_credit_limit' : float(mediana_credit),
        'imputation_median_minimum_payments' : float(mediana_min_pay),

        # Parametros de transformação
        'quantil_output_distribuition' : 'normal',
        'quantil_random_state' : 0,

        # Transformações de cada coluna
        'columns_transf_log' : t_log,
        'columns_transf_quantil' : t_quantil,

    }

    # C. Salvar json
    metadata_path = output_path.parent / 'features_metadata.json'

    with open(metadata_path, 'w', encoding= 'utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii= False)

    logger.success('Salvando dataset em PROCESSED.')
    df.to_csv(output_path, index = False)

    logger.success("Etapa de engenharia de features e documentação concluída!")
    
    # -----------------------------------------

if __name__ == "__main__":
    app()
