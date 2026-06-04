from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import pandas as pd
import numpy as np

from config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path_train: Path = INTERIM_DATA_DIR / "dataset_train.csv",
    input_path_test: Path = INTERIM_DATA_DIR / "dataset_test.csv",

    output_path_train: Path = PROCESSED_DATA_DIR / "features_train_cluster.csv",
    output_path_test: Path = PROCESSED_DATA_DIR / "features_test_cluster.csv",
    output_path_VIF: Path = PROCESSED_DATA_DIR / "features_VIF.csv",
    # -----------------------------------------
):
    #---------------------------------------------------------------------
    # 1. Carregar os dados
    #---------------------------------------------------------------------
    try:
        df_train= pd.read_csv(input_path_train)
        logger.info(f'Dataset Treino carregado com sucesso. Formato inicial: {df_train.shape}')
    except Exception as e:
        logger.error(f'Erro a ler arquivo {e}')
        # forçar o encerramento e dizer ao sistema que tem algo errado
        raise typer.Exit(code=1)

    try:
        df_test= pd.read_csv(input_path_test)
        logger.info(f'Dataset Teste carregado com sucesso. Formato inicial: {df_test.shape}')
    except Exception as e:
        logger.error(f'Erro a ler arquivo {e}')
        # forçar o encerramento e dizer ao sistema que tem algo errado
        raise typer.Exit(code=1)
    
    #---------------------------------------------------------------------
    # 2. Imputação e escalonamento das variáveis
    #---------------------------------------------------------------------
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, QuantileTransformer, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer

    logger.info(f'Quantidade inicial de nulos: {df_train.isnull().sum().sum()}')

    # A. Criar coluna booleana a respeito de minimum_payments
    df_train['null_minimum_payments'] = df_train['MINIMUM_PAYMENTS'].isnull()
    # criar no teste também, pra aplicar .transform
    df_test['null_minimum_payments'] = df_test['MINIMUM_PAYMENTS'].isnull()

    # B. Colunas a serem transformadas
    # Transformação log
    t_log = [
        'BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 
        'CASH_ADVANCE', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 
        'PURCHASES_TRX','CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS'
    ]
    # Transformação quantil
    t_quantil = [
        'BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 
        'PURCHASES_INSTALLMENTS_FREQUENCY', 'PRC_FULL_PAYMENT'
    ]

    # C. Calculo da mediana pra salvamento no json
    mediana_credit = df_train['CREDIT_LIMIT'].median()
    mediana_min_pay = df_train['MINIMUM_PAYMENTS'].median()

    #---------------------------------------------------------------------
    # 3. Criar pipeline
    #---------------------------------------------------------------------
    # A. Pipelines de apoio para imputar a mediana
    pipe_log = make_pipeline(
        SimpleImputer(strategy='median'), 
        FunctionTransformer(
            np.log1p,
            feature_names_out='one-to-one' #garante que os nomes de variaveis são os mesmo de entrada e saida
        ) 
    )
    
    pipe_quantil = make_pipeline(
        SimpleImputer(strategy='median'), 
        QuantileTransformer(output_distribution='normal', random_state=0)
    )
    # B. Pipeline final
    pipeline_clustering = make_pipeline(
        ColumnTransformer( # Limpar assimetria dos dados
            transformers=[
                ('log', pipe_log, t_log),
                ('quantil', pipe_quantil, t_quantil)
            ],
        remainder='passthrough' # manter as demais colunas
        )
    )
    # ColumnTRansform gera um array e não um df

    #---------------------------------------------------------------------
    # 4. Executar o pipeline
    #---------------------------------------------------------------------
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    def excluir_vif(df, limite = 5):
        df_temp = df.select_dtypes(include=[np.number]).copy()
          
        excluir = []
        features_vif = {}

        while True:
            # Criando df com valores de VIF por features
            vif_data = pd.DataFrame()
            vif_data['features'] = df_temp.columns
            vif_data['VIF'] = [
                variance_inflation_factor(df_temp.values, i) 
                for i in range(len(df_temp.columns))
            ]

            # Encontrar o que tem maior VIF
            maior_vif = vif_data.sort_values(by='VIF', ascending=False).iloc[0]
            feature = maior_vif['features']
            valor_vif = maior_vif['VIF']

            # Se o maior valor VIF > 5, delete
            if valor_vif > limite:
                excluir.append(feature) # lista de variaveis a serem excluidas
                features_vif[feature] = float(valor_vif) # dic com o nome e VIF

                df_temp = df_temp.drop(columns=[feature]) # apagar coluna e recomeçar

            else:
                break # interromper o loop

        return excluir, features_vif

    # A. Aplicando o pipeline
    logger.info("Executando o pipeline de transformações")
    X_train = pipeline_clustering.fit_transform(df_train)
    X_test = pipeline_clustering.transform(df_test)  

    # B. Recuperar a ordem e nome exatos da colunas que se perde no ColumnTRansform
    colunas_finais = pipeline_clustering.get_feature_names_out().tolist()    

    # C. Aplicar seleção por VIF
    df_vif = pd.DataFrame(X_train, columns = colunas_finais).astype(float) # df temporario
    logger.info("Calculando o VIF")

    lista_drop, dicionario_vif = excluir_vif(df_vif, limite=5)

    # D. Filtra as colunas restantes no treino 
    df_train = df_vif.drop(columns=lista_drop)
    colunas_restantes = df_train.columns.to_list()

    # E. Filtra as colunas restantes no teste 
    df_test = pd.DataFrame(X_test, columns = colunas_finais).astype(float) 
    df_test = df_test[colunas_restantes]

    # F. Aplicar um escalanador universal para k-means
    scaler = StandardScaler()

    logger.info("Executando a padronização de transformações")

    X_train = scaler.fit_transform(df_train)
    X_test = scaler.transform(df_test)  

    #---------------------------------------------------------------------
    # 5. Salvar Artefatos/Metadados
    #---------------------------------------------------------------------
    import joblib
    import json
    from config import MODELS_DIR

    logger.info('Iniciando exportação de artefatos e metadados.')

    # Criando uma pasta
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # A. Salvar o pipeline completo
    pipeline_clustering = make_pipeline(pipeline_clustering, scaler)
    joblib.dump(pipeline_clustering, MODELS_DIR / 'pipeline_clustering.pkl')

    # Salvar os dados de medianas
    mapa_medianas = {
        'credit_limit': float(mediana_credit),
        'minimum_payments': float(mediana_min_pay)
    }

    joblib.dump(mapa_medianas, MODELS_DIR / 'imputation_medians_clust.pkl')

    # B. Criar o metatados
    metadata = {
        'pipeline_stage': 'feature_engineering',
        'pipeline_type': 'clustering',
        
        # Valores de imputação
        'imputation_median_credit_limit' : float(mediana_credit),
        'imputation_median_minimum_payments' : float(mediana_min_pay),

        # Parametros de transformação
        'columns_transf_log': t_log,
        'columns_transf_quantil': t_quantil,
        'manual_boolean_features': ['null_minimum_payments'],
        'quantil_output_distribuition' : 'normal',
        'quantil_random_state' : 0,

        # Transformações de cada coluna
        'columns_transf_log' : t_log,
        'columns_transf_quantil' : t_quantil,

        # VIF:
        'removed_by_vif_list': lista_drop,
        'removed_by_vif_details': dicionario_vif,

    }

    # C. Salvar json
    metadata_path = output_path_train.parent / 'features_metadata_clust.json'

    with open(metadata_path, 'w', encoding= 'utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii= False)

    #---------------------------------------------------------------------
    # 6. Gerar Processed
    #---------------------------------------------------------------------
    df_train_processed = pd.DataFrame(X_train, columns = colunas_restantes)
    df_test_processed = pd.DataFrame(X_test, columns = colunas_restantes)

    logger.success('Salvando dataset Train em PROCESSED.')
    df_train_processed.to_csv(output_path_train, index = False)

    logger.success('Salvando dataset Test em PROCESSED.') 
    df_test_processed.to_csv(output_path_test, index = False)

    logger.success('Salvando infos de VIF PROCESSED.')
    df_vif_removidas = pd.DataFrame(
        dicionario_vif.items(),
        columns=['feature', 'VIF']
    )
    df_vif_removidas.to_csv(output_path_VIF, index = False)

    logger.success("Etapa de engenharia de features e documentação concluída!")
    logger.success("Uso exclusivo para clusterizacao")
    
    # -----------------------------------------

if __name__ == "__main__":
    app()