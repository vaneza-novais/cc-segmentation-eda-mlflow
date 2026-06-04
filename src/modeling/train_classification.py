from pathlib import Path


from loguru import logger
from tqdm import tqdm
import typer

import mlflow # biblioteca do MLflow
import sklearn

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

# Definir e configurar tracking (MLflow)
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment('credit_card_classifier')

app = typer.Typer()

# A finalidade desse modelo é fazer a predição do cluster de novos clientes
@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    prediction_path: Path = PROCESSED_DATA_DIR / 'test_predictions_cluster.csv',
    # labels_path: Path = PROCESSED_DATA_DIR / 'labels.csv',
    model_path: Path = MODELS_DIR / 'model.pkl',
    # -----------------------------------------
):
    
     #---------------------------------------------------------------------
    # 1. Carregar os dados
    #---------------------------------------------------------------------
    import pandas as pd

    try:
        df_prediction = pd.read_csv(prediction_path)
        logger.info(f'Dataset carregado com sucesso. Formato inicial: {df_prediction.shape}')
    except Exception as e:
        logger.error(f'Erro a ler arquivo {e}')
        # forçar o encerramento e dizer ao sistema que tem algo errado
        raise typer.Exit(code=1)
     #---------------------------------------------------------------------
    # 2. Separar feature(X) de target(Y)
    #---------------------------------------------------------------------
    logger.info(f'Separando features do target.')
    X = df_prediction.drop(columns= ['cluster'])
    Y = df_prediction['cluster']

    #---------------------------------------------------------------------
    # 3. Separar em train e test
    #---------------------------------------------------------------------
    from sklearn.model_selection import train_test_split

    logger.info(f'Separando os dados em 80% teste e 20% treino.')
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size= 0.2, random_state= 0
    )

    logger.info(f'Dados divididos - Treino Classificador: {X_train.shape}, Teste: {X_test.shape}')

    #---------------------------------------------------------------------
    # 4. Aplicação dos modelos no treino
    #---------------------------------------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    import joblib

    # Configurando Cross Validation
    kf = StratifiedKFold(
            n_splits = 10, shuffle = True, random_state = 0
        )
    
    logger.info(f'Configurado 10 folds para Cross Validation')

    metrics = ['f1_weighted', 'precision_weighted'] # F1score e precisão
    score_metrics = []

    #*********************************************************************
    # 4.1 Regressão logistica
    #*********************************************************************
    
    # Inicializar o MLflow
    with mlflow.start_run(run_name= 'logistic_regression'):
        logger.info(f'Aplicando Regressão Logistica')

        # A) Definindo o pipeline
        max_iter_lg = 100
        pipeline_lg = make_pipeline(
            LogisticRegression(max_iter = max_iter_lg, random_state = 0)
        )

        # B) Executando métricas
        pipeline_score = pipeline_lg

        for i in metrics:
            score = cross_val_score(
                pipeline_score, 
                X_train,
                Y_train,
                cv = kf,
                scoring= i 
                )   
            score_metrics.append((i, score))
  

        # C) Aplicação no treino
        logger.info(f'Ajuste de treino')

        pipeline_score.fit(X_train, Y_train) # AJuste do treino 

        # D) Registro no MLflow
        logger.info(f'Salvando registro no MLflow')
        for metric_name, metric_value in score_metrics:
            mlflow.log_metric(
                metric_name,
                metric_value.mean()
            )

        # registro dos parametros
        mlflow.log_param('max_iter', max_iter_lg)
        mlflow.log_param('random_state', 0)
        
        # Salvar localmente para uso na predição
        # joblib.dump(pipeline_score,  model_path)

        # Salvar artefato do modelo
        mlflow.log_artifact(
            local_path= model_path, #oirgem local
            artifact_path= 'model_artifacts' # servidor
            )
        
        logger.success('Artefatos salvos no servidor')

    #*********************************************************************
    # 4.2 Random Forest
    #*********************************************************************
    # Inicializar o MLflow
    with mlflow.start_run(run_name= 'random_forest'):

        logger.info(f'Aplicando Random Forest')

        # A) Definindo o pipeline
        n_est = 100
        depth = 5

        pipeline_rf = make_pipeline(
            RandomForestClassifier(
                n_estimators= n_est,
                max_depth= depth,
                random_state = 0
                )
        )

        # B) Executando métricas
        pipeline_score = pipeline_rf

        for i in metrics:
            score = cross_val_score(
                pipeline_score, 
                X_train,
                Y_train,
                cv = kf,
                scoring= i 
                )   
            score_metrics.append((i, score))
  
        # C) Aplicação no treino
        logger.info(f'Ajuste de treino')

        pipeline_score.fit(X_train, Y_train) # AJuste do treino    

        # D) Registro no MLflow
        logger.info(f'Salvando registro no MLflow')
        for metric_name, metric_value in score_metrics:
            mlflow.log_metric(
                metric_name,
                metric_value.mean()
            )

        # registro dos parametros
        mlflow.log_param('n_estimators', n_est)
        mlflow.log_param('max_depth', depth)
        mlflow.log_param('random_state', 0)
        
        # Salvar localmente para uso na predição
        # joblib.dump(pipeline_score,  model_path)

        # Salvar artefato do modelo
        mlflow.log_artifact(
            local_path= model_path, #oirgem local
            artifact_path= 'model_artifacts' # servidor
            )
        
        logger.success('Artefatos salvos no servidor')



if __name__ == '__main__':
    app()
