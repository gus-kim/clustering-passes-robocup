# Análise de Clustering de Passes de Futebol de Robôs

Este projeto realiza a análise de dados de partidas de futebol de robôs (provavelmente da competição RoboCup), com o objetivo de agrupar (clusterizar) os passes em diferentes categorias com base em suas características.

## Estrutura do Repositório

```
/
├─── src/
│   ├─── Arquivo_final2.py       # Script principal que executa o clustering.
│   └─── filter.py               # Script para pré-processar e filtrar os dados de passe.
├─── CSV_Completo/           # Contém os dados brutos das partidas convertidos para CSV.
├─── CSV_filtrado/           # Contém os dados de passes filtrados, prontos para análise.
├─── data_bruto/             # Contém os logs originais das partidas (.rcg.gz) e o conversor.
├─── README.md               # Este arquivo.
└─── requirements.txt        # Dependências do Python.
```

## Como Executar

### 1. Instalar Dependências

Antes de começar, instale todas as bibliotecas Python necessárias:
```bash
pip install -r requirements.txt
```

### 2. Filtragem dos Dados

Use o script `src/filter.py` para processar um CSV da pasta `CSV_Completo/` e extrair as características dos passes. O resultado será salvo em `CSV_filtrado/`.

**Exemplo:**
```bash
python src/filter.py CSV_Completo/Helios_vs_Cyrus_2023.csv CSV_filtrado/pass_features_helios_cyrus.csv
```

### 3. Execução do Clustering

Use o script `src/Arquivo_final2.py` para carregar os dados filtrados e realizar a análise de clustering.

**Exemplo:**
```bash
python src/Arquivo_final2.py CSV_filtrado/pass_features_helios_cyrus.csv
```

## Automação

Para facilitar, você pode usar o script `run_analysis.sh` para executar o pipeline completo para um arquivo de entrada específico. Veja o arquivo para mais detalhes.
