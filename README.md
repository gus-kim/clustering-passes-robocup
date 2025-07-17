# Análise de Clustering de Passes de Futebol de Robôs

Este projeto realiza a análise de dados de partidas de futebol de robôs (provavelmente da competição RoboCup), com o objetivo de agrupar (clusterizar) os passes em diferentes categorias com base em suas características.

## Estrutura de Arquivos

```
/
├─── src/                     # Contém os scripts Python
│   ├─── clustering_analysis.py # Script principal que executa o clustering.
│   └─── feature_extraction.py  # Script para pré-processar e filtrar os dados de passe.
├─── CSV_Completo/            # Contém os dados brutos das partidas convertidos para CSV.
├─── CSV_filtrado/            # Contém os dados de passes filtrados, prontos para análise.
├─── data_bruto/              # Contém os logs originais das partidas (.rcg.gz) e o conversor.
├─── .gitignore               # Arquivos e diretórios a serem ignorados pelo Git.
├─── README.md                # Este arquivo.
├─── requirements.txt         # Dependências do Python.
└─── scripts/
    └─── run_analysis.sh      # Script para automatizar a análise.
```

## Como Usar

### 1. Pré-requisitos

- Python 3.x
- Git

### 2. Instalação

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/gus-kim/clustering-passes-robocup.git
cd clustering-passes-robocup
pip install -r requirements.txt
```

### 3. Configuração dos Dados

Este repositório não armazena os arquivos de dados brutos ou processados. Para executar a análise, você precisa criar a estrutura de diretórios e adicionar os arquivos de dados manualmente.

1.  **Crie os diretórios de dados:**

    ```bash
    mkdir -p data_bruto CSV_Completo
    ```

2.  **Adicione seus dados:**
    *   Coloque seus arquivos de log de partida (`.rcg.gz`) no diretório `data_bruto/`.
    *   Converta os logs para o formato CSV e coloque os arquivos resultantes no diretório `CSV_Completo/`.

    **Nota:** O script `run_analysis.sh` espera encontrar o arquivo CSV de entrada (por exemplo, `Helios_vs_Cyrus_2023.csv`) dentro da pasta `CSV_Completo/`.

### 4. Executando a Análise

O script `scripts/run_analysis.sh` automatiza todo o processo. Basta fornecer o nome do arquivo CSV de entrada da pasta `CSV_Completo/`.

**Exemplo:**

```bash
./run_analysis.sh Helios_vs_Cyrus_2023.csv
```

O script irá:

1.  Executar `src/feature_extraction.py` para pré-processar os dados.
2.  Executar `src/clustering_analysis.py` para realizar a análise de clustering.

Os resultados, incluindo gráficos e tabelas, serão exibidos na tela.