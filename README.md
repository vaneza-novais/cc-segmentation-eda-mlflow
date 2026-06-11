# Credit Card Clustering: From Exploratory Data Analysis to MLflow Experiment Tracking

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

An end-to-end customer segmentation project using K-Means and PCA on credit card usage data, fully tracked with MLflow.

# Sobre o projeto  
O objetivo desse projeto é segmentar a base de usuários de cartão de crédito, seguido da predição de classificação de novos usuários.  
A finalidade original do dataset é definir estratégias de marketing para cada cluster de clientes. Oportuno frisar, que podem haver outras finalidades.
O dataset utilizado se encontra disponivel no [kaggle](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata).

## Fluxo do Projeto
Todo o projeto foi estruturado de forma facionada e registrada no MLflow, de modo resumido:  
Dados Brutos ➔ Tratamento/Pipelines ➔ Clusterização (K-Means) ➔ Cross Validation de Classificação (Regressão Logística vs. Random Forest) ➔ MLflow

### Dicionário das variáveis
**CUST_ID** : ID do titular - Categórica  
**BALANCE** : Saldo devedor atual  
**BALANCE_FREQUENCY** : Com que frequencia o saldo devedor é atualizado? Entre 0 e 1 (1 = frequently updated, 0 = not frequently updated)  
**PURCHASES** : Valor de compras feitas na conta  
**ONEOFF_PURCHASES** : Valor máximo de compra realizado a vista  
**INSTALLMENTS_PURCHASES** : Valor de compra feita em parcelas  
**CASH_ADVANCE** : Valor retirado em pagamentos antecipados (emprestimos, saques)  
**PURCHASES_FREQUENCY** : Com que frequencia as compras tem sendo feitas? Entre 0 e 1 (1 = frequently purchased, 0 = not frequently purchased)  
**ONEOFFPURCHASESFREQUENCY** : Com que frequencia as compras são a vista? (1 = frequently purchased, 0 = not frequently purchased)  
**PURCHASESINSTALLMENTSFREQUENCY** : Com que frequencia as compras parceladas são feitas (1 = frequently done, 0 = not frequently done)  
**CASHADVANCEFREQUENCY** : Com que frequencia o emprestimo/saque é feito  
**CASHADVANCETRX** : Qntd de emprestimos/saque  
**PURCHASES_TRX** : Qntd de transacoẽs de compras feitas  
**CREDIT_LIMIT** : Limite do cartão de credito para o usuário  
**PAYMENTS** : Valor do pagamento tt  
**MINIMUM_PAYMENTS** : Valor minimo da fatura  
**PRCFULLPAYMENT** : Percental de pagamento tt (1=sempre quitou, 0 = nunca quitou tudo)  
**TENURE** : Tempo de contrato (meses)  

Algumas variáveis foram removidas através da tecnica VIF, quando seu resultado é de valor maior que 5.

# Cluster descobertos
Com a aplicação do K-Means, obteve-se a seguinte segmentação dos clientes:  
| Cluster | Nome do Perfil | Estratégia de Abordagem |
| :---: | :--- | :--- |
| **0** | **Clientes Conservadores** | Faça uma comunicação transparente e prática, apresentando cases de sucesso, destacando o custo beneficio e confiabilidade do serviço de cartão de crédito. |
| **1** | **Clientes Dependentes de Crédito/Empréstimo** | Ofereça aprovação facilitada, focada na baixa burocracia assim como o atendimento prático para renegociação de dívidas e ferramentas de controle total do limite. |
| **2** | **Clientes de Alto Valor** | Fidelize esse cliente com programas de pontuação, benefícios de cliente VIP, categorias de cartão exclusivas e sempre ofereça um limite alto. |
| **3** | **Clientes Engajados** | Ofereça campanhas de incentivo para uso do cartão, assim como vantagens no parcelamento e pagamento antecipado. |
| **4** | **Clientes em Risco Financeiro** | Realize cobranças preventivas como lembrete em paralelo ao envio de conteúdos acerca de educação financeira e incentivo de pagamento correto, e aplique a redução sutil do limite conforme o risco aumenta. |
| **5** | **Clientes de Baixo Engajamento** | Desenvolva campanhas de descontos e pagamentos simplificados, implante a comunicação direta além de redução severa do limite ao mínimo possível, para evitar novas compras. |

<img width="2090" height="1451" alt="image" src="https://github.com/user-attachments/assets/dd20c97b-7918-496d-be78-111a850c487b" />


# Aplicação em produção
Para definição de cluster de novos clientes, será utilizado o modelo de Regressão Logística, no qual a variável target é o cluster já descoberto.
Regressão Logística foi o modelo escolhido dado que a sua comparação de desempenho com o Random Forest obteve melhor resultado.
| Modelo | F1-Score | Precisão |
| :--- | :---: | :---: |
| **Logistic Regression** | 99,37% | 99,40% |
| **Random Forest** | 95,68% | 95,88% |  


O Modelo de Regressão Logistica apresentou um desempenho excelente, dado que houve a aplicação de um modelo não supervisionado e as variáveis foram escalonadas com as transformações Logarítmicas e Quantil.
<img width="1645" height="880" alt="image" src="https://github.com/user-attachments/assets/15a23ce9-0dfd-4bd6-9c77-18b667519b12" />


Além disso, analisando o gráfico de **Feature Importance**, consegue-se entender como as variáveis predominam na decisão do modelo:
<img width="3520" height="1684" alt="features_importance" src="https://github.com/user-attachments/assets/b69549a7-df0e-4761-ac3d-a0f63fe81440" />
* A Frequencia de Compras predomina na decisão do modelo, assim como quase tudo que é relacionado a compras,
* A caracteristica de menos influencia é o percentual de pagamento total

# Como executar:
1. git clone
2. pip install -r requirements.txt
3. mlflow server
4. http://127.0.0.1:5000 (Link do servidor)


### Project Organization - CookieCutter

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         credit_card_clustering:_from_exploratory_data_analysis_to_mlflow_experiment_tracking and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── credit_card_clustering:_from_exploratory_data_analysis_to_mlflow_experiment_tracking   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes credit_card_clustering:_from_exploratory_data_analysis_to_mlflow_experiment_tracking a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

