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
git clone <URL_DO_SEU_REPOSITORIO_NO_GITHUB>
cd T2_pass_clustering
pip install -r requirements.txt
```

### 3. Executando a Análise

O script `scripts/run_analysis.sh` automatiza todo o processo. Basta fornecer o nome do arquivo CSV de entrada da pasta `CSV_Completo/`.

**Exemplo:**

```bash
./run_analysis.sh Helios_vs_Cyrus_2023.csv
```

O script irá:

1.  Executar `src/feature_extraction.py` para pré-processar os dados.
2.  Executar `src/clustering_analysis.py` para realizar a análise de clustering.

Os resultados, incluindo gráficos e tabelas, serão exibidos na tela.