from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "features_train_cluster.csv",
    output_path: Path = FIGURES_DIR / "plot_cotovelo.png",
    # -----------------------------------------
):
    #---------------------------------------------------------------------
    # 1. Carregar os dados
    #---------------------------------------------------------------------
    import pandas as pd

    try:
        df= pd.read_csv(input_path)
        logger.info(f'Dataset Processed carregado com sucesso. Formato inicial: {df.shape}')
    except Exception as e:
        logger.error(f'Erro a ler arquivo {e}')
        # forçar o encerramento e dizer ao sistema que tem algo errado
        raise typer.Exit(code=1)

    #---------------------------------------------------------------------
    # 2. Testar faixa de valores de cluster (k)
    #---------------------------------------------------------------------
    from sklearn.cluster import KMeans


    inercias = []

    faixa_k = range(1,11)

    logger.info("Testando faixa de valores.")

    for k in faixa_k:
        # treinar para cada valor k
        kmeans_teste = KMeans(
            n_clusters=k, random_state=42, n_init=10
        )
        kmeans_teste.fit(df)

        inercias.append(
            kmeans_teste.inertia_
        )

    #---------------------------------------------------------------------
    # 3. Plotar gráfico
    #---------------------------------------------------------------------
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(faixa_k, inercias, 'bx-') # 'bx-' cria uma linha azul com marcações em X
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inércia (Compactação dos Grupos)')
    plt.title('Método do Cotovelo para Escolha do K Ideal')
    plt.xticks(faixa_k) # Garante que todos os números de 1 a 10 apareçam no eixo X
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    logger.success("Gráfico de K gerado.")


if __name__ == "__main__":
    app()
