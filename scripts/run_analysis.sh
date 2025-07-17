#!/bin/bash

# Script para automatizar a análise de clustering de passes.
# Uso: ./run_analysis.sh <nome_do_arquivo_csv_de_entrada>
# Exemplo: ./run_analysis.sh Helios_vs_Cyrus_2023.csv

# --- Configuração ---
INPUT_CSV_NAME=$1
BASE_NAME=$(basename ${INPUT_CSV_NAME} .csv)

INPUT_FILE="CSV_Completo/${INPUT_CSV_NAME}"
FILTERED_FILE="CSV_filtrado/features_${BASE_NAME}.csv"

# --- Validação ---
if [ -z "$1" ]; then
    echo "Erro: Você precisa fornecer o nome do arquivo CSV de entrada."
    echo "Uso: ./run_analysis.sh <nome_do_arquivo_csv_de_entrada>"
    echo "Exemplo: ./run_analysis.sh Helios_vs_Cyrus_2023.csv"
    exit 1
fi

if [ ! -f "${INPUT_FILE}" ]; then
    echo "Erro: Arquivo de entrada não encontrado em ${INPUT_FILE}"
    exit 1
fi

# --- Pipeline ---

echo "[PASSO 1/2] Iniciando a filtragem de passes..."
python3 src/feature_extraction.py "${INPUT_FILE}" "${FILTERED_FILE}"

# Verifica se a filtragem foi bem-sucedida
if [ $? -ne 0 ]; then
    echo "Erro durante a execução do script de filtragem."
    exit 1
fi

echo "
[PASSO 2/2] Iniciando a análise de clustering..."
python3 src/clustering_analysis.py "${FILTERED_FILE}"

if [ $? -ne 0 ]; then
    echo "Erro durante a execução do script de análise."
    exit 1
fi

echo "
Análise completa!"
