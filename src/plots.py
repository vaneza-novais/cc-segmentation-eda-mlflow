from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import FIGURES_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, MODELS_DIR

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / 'features_train_cluster.csv',
    output_path: Path = FIGURES_DIR / 'plot_cotovelo.png',
    input_path_cluster: Path = REPORTS_DIR / 'perfil_clusters.csv',
    output_path_cluster: Path = FIGURES_DIR / 'distribuition_clusters.png',
    input_path_model: Path = MODELS_DIR / 'model_rf.pkl',
    input_path_features_importance: Path = PROCESSED_DATA_DIR / 'test_predictions_cluster.csv',
    output_path_features_importance: Path = FIGURES_DIR / 'features_importance.png',
    # -----------------------------------------
):
    #---------------------------------------------------------------------
    # 1. Metodo do Cotovelo - Definir k
    #---------------------------------------------------------------------

    # A) Carregar os dados
    try:
        df= pd.read_csv(input_path)
        logger.info(f'Dataset Processed carregado com sucesso. Formato inicial: {df.shape}')
    except Exception as e:
        logger.error(f'Erro a ler arquivo {e}')
        # forçar o encerramento e dizer ao sistema que tem algo errado
        raise typer.Exit(code=1)

    # B) Testar faixa de valores de cluster (k)
    from sklearn.cluster import KMeans


    inercias = []

    faixa_k = range(1,11)

    logger.info('Testando faixa de valores.')

    for k in faixa_k:
        # treinar para cada valor k
        kmeans_teste = KMeans(
            n_clusters=k, random_state=42, n_init=10
        )
        kmeans_teste.fit(df)

        inercias.append(
            kmeans_teste.inertia_
        )

    # C) Plotar gráfico
    plt.figure(figsize=(8, 5))
    plt.plot(faixa_k, inercias, 'bx-') # 'bx-' cria uma linha azul com marcações em X
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inércia (Compactação dos Grupos)')
    plt.title('Método do Cotovelo para Escolha do K Ideal')
    plt.xticks(faixa_k) # Garante que todos os números de 1 a 10 apareçam no eixo X
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    logger.success('Gráfico de K gerado.')

    #---------------------------------------------------------------------
    # 2. Mostrar o tamanho dos clusters
    #---------------------------------------------------------------------
    # A) Carregar os dados
    try:
        df= pd.read_csv(input_path_cluster)
        logger.info(f'Dataset report carregado com sucesso. Formato inicial: {df.shape}')
    except Exception as e:
        logger.error(f'Erro a ler arquivo {e}')
        # forçar o encerramento e dizer ao sistema que tem algo errado
        raise typer.Exit(code=1)

    # B) Testar faixa de valores de cluster (k)
    from sklearn.cluster import KMeans

    logger.info('Gerando gráfico de distribuição dos grupos')

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df, x='cluster', y = 'qtde_por_cluster', hue='cluster', palette='viridis'
    )

    plt.title('Quantidade de Clientes por Cluster (Base de Teste)', fontsize=13, pad=15)
    plt.xlabel('Número do Cluster', fontsize=11)
    plt.ylabel('Total de Clientes', fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)


    # Salva a imagem com alta resolução para o seu Readme do GitHub
    plt.savefig(output_path_cluster, dpi=300, bbox_inches='tight')
    plt.close()
    logger.success(f'Gráfico de distribuição salvo.')

    #---------------------------------------------------------------------
    # 3. Entender a relevancia das variáveis
    #---------------------------------------------------------------------
    import joblib
    from sklearn.pipeline import make_pipeline

     # A) Carregar os dados
    try:
        df = pd.read_csv(input_path_features_importance)
        # Remover a coluna de cluster
        if 'cluster' in df.columns:
            df = df.drop(columns=['cluster'])
        features_names = df.columns
        logger.info(f'Dataset predição carregado com sucesso. Formato inicial: {df.shape}')
    except Exception as e:
        logger.error(f'Erro a ler arquivo {e}')
        # forçar o encerramento e dizer ao sistema que tem algo errado
        raise typer.Exit(code=1)

    # B) Carregar os modelos
    try:
        pipeline = joblib.load(input_path_model)
        logger.info(f'Modelo Random Forest carregado com sucesso')
    except Exception as e:
        logger.error(f'Erro a ler arquivo {e}')
        # forçar o encerramento e dizer ao sistema que tem algo errado
        raise typer.Exit(code=1)
    
    # Extrair o modelo do pipeline
    try:
        rf_model = pipeline.named_steps['randomforestclassifier']
        importances = rf_model.feature_importances_
    except KeyError:
        logger.error(
            "Não foi possível encontrar o passo 'randomforestclassifier' no pipeline. Verifique o nome usado."
        )
        raise typer.Exit(code=1)
    
    # C) Criar um df com as features e valores
    df_importance = pd.DataFrame(
        {'Feature': features_names, 'Importance': importances}
    ).sort_values(by='Importance', ascending=False)

    # D) Criar gráfico
    logger.info('Gerando gráfico de feature importance.')

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data = df_importance, 
        x='Importance', 
        y = 'Feature', 
        hue='Feature',
        palette='viridis'
    )
    plt.title(
        'Feature Importance',
        fontsize=13,
        pad=15,
    )
    plt.xlabel('Importância', fontsize=11)
    plt.ylabel('Vairável', fontsize=11)
    plt.grid(axis='x', linestyle='--', alpha=0.5)


    # Salva a imagem com alta resolução para o seu Readme do GitHub
    plt.savefig(output_path_features_importance, dpi=300, bbox_inches='tight')
    plt.close()
    logger.success(f'Gráfico de feature importance salvo.')
   

if __name__ == '__main__':
    app()
