"""
Autores: Pedro Henrique Siqueira de Paula Peixoto,
         Rafael Dayvid de Melo Freitas,
         Thales Henrique Marques Lázaro,
         Vinícius Duarte de Carvalho.

Processamento dos Microdados do ENEM (2022/2023) com arquivo CSV final reestruturado e ordenado.
"""

import argparse
import string
import sys
from typing import NamedTuple
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# CONFIGURAÇÕES E CONSTANTES
# -----------------------------------------------------------------------------

ARQUIVOS_PADRAO = {
    2022: "MICRODADOS_ENEM_2022.csv",
    2023: "MICRODADOS_ENEM_2023.csv",
}

# Questões socioeconômicas Q001 a Q025
COLUNAS_QUESTOES = [f"Q{i:03d}" for i in range(1, 26)]

# Colunas brutas a serem lidas do CSV do Inep
COLUNAS_DEMOGRAFICAS = ["NU_INSCRICAO", "TP_FAIXA_ETARIA", "TP_SEXO", "TP_COR_RACA", "TP_ESCOLA"]
COLUNAS = COLUNAS_DEMOGRAFICAS + COLUNAS_QUESTOES

COLUNAS_FILTRO = [
    "IN_TREINEIRO",
    "NU_NOTA_REDACAO",
    "TP_PRESENCA_CN",
    "TP_PRESENCA_CH",
    "TP_PRESENCA_LC",
    "TP_PRESENCA_MT",
]

# ORDENAÇÃO FINAL DAS COLUNAS NO CSV DE SAÍDA
COLUNAS_ORDENADAS_SAIDA = (
    ["NU_INSCRICAO", "NU_ANO"]                  # Bloco 1: Identificação
    + COLUNAS_DEMOGRAFICAS[1:]                   # Bloco 2: Perfil do Participante
    + ["SAL_PER_CAPITA_MIN", "SAL_PER_CAPITA_MAX"] # Bloco 3: Renda Per Capita Calculada
    + COLUNAS_QUESTOES                           # Bloco 4: Questionário Socioeconômico (Q001-Q025)
)

SALARIO_MINIMO = {
    2022: 1212.00,
    2023: 1320.00,
}

LIMITES_RENDA = {
    2022: [1212, 1818, 2424, 3030, 3636, 4848, 6060, 7272, 8484, 9696, 10908, 12120, 14544, 18180, 24240],
    2023: [1320, 1980, 2640, 3300, 3960, 5280, 6600, 7920, 9240, 10560, 11880, 13200, 15840, 19800, 26400],
}

SAIDA = "enem_treino.csv"
SEP, ENCODING = ";", "latin-1"
CHUNK_SIZE = 200_000


# -----------------------------------------------------------------------------
# ESTRUTURA E CRIAÇÃO DAS FAIXAS DE RENDA
# -----------------------------------------------------------------------------

class IntervaloRenda(NamedTuple):
    minimo: float
    maximo: float


def construir_tabela_faixas(ano: int) -> dict[str, IntervaloRenda]:
    limites = LIMITES_RENDA[ano]
    letras_intermediarias = string.ascii_uppercase[1:16]  # Letras 'B' até 'P'
    
    tabela_faixas = {"A": IntervaloRenda(minimo=0.0, maximo=0.0)}
    piso_atual = 0.0

    for letra, teto in zip(letras_intermediarias, limites):
        teto_float = float(teto)
        tabela_faixas[letra] = IntervaloRenda(minimo=piso_atual, maximo=teto_float)
        piso_atual = teto_float

    tabela_faixas["Q"] = IntervaloRenda(minimo=piso_atual, maximo=np.nan)
    return tabela_faixas


def exibir_tabela_faixas(ano: int) -> None:
    sm = SALARIO_MINIMO[ano]
    tabela = construir_tabela_faixas(ano)
    
    print(f"\n{'='*65}")
    print(f" TABELA DE FAIXAS DE RENDA - ENEM {ano} (Salário Mínimo: R$ {sm:,.2f})")
    print(f"{'='*65}")
    print(f"{'Letra':<6} | {'Piso (R$)':<12} | {'Teto (R$)':<14} | {'Em Salários Mínimos (SM)'}")
    print(f"{'-'*65}")

    for letra, faixa in tabela.items():
        min_sm = faixa.minimo / sm
        if np.isnan(faixa.maximo):
            teto_str = "Sem limite"
            sm_str = f"Acima de {min_sm:.1f} SM"
        else:
            max_sm = faixa.maximo / sm
            teto_str = f"R$ {faixa.maximo:,.2f}"
            sm_str = f"{min_sm:.1f} a {max_sm:.1f} SM"

        print(f" {letra:<5} | R$ {faixa.minimo:<9,.2f} | {teto_str:<14} | {sm_str}")
    print(f"{'='*65}\n")


def calcular_renda_per_capita(df: pd.DataFrame, ano: int) -> pd.DataFrame:
    sm = SALARIO_MINIMO[ano]
    tabela_faixas = construir_tabela_faixas(ano)

    mapa_minimo = {letra: faixa.minimo for letra, faixa in tabela_faixas.items()}
    mapa_maximo = {letra: faixa.maximo for letra, faixa in tabela_faixas.items()}

    renda_minima_familia = df["Q006"].map(mapa_minimo)
    renda_maxima_familia = df["Q006"].map(mapa_maximo)

    num_moradores = pd.to_numeric(df["Q005"], errors="coerce")
    num_moradores = num_moradores.where(num_moradores > 0)

    df["SAL_PER_CAPITA_MIN"] = (renda_minima_familia / sm / num_moradores).round(4)
    df["SAL_PER_CAPITA_MAX"] = (renda_maxima_familia / sm / num_moradores).round(4)

    return df


# -----------------------------------------------------------------------------
# PROCESSAMENTO DE DADOS EM LOTES
# -----------------------------------------------------------------------------

def processar_microdados(ano: int, caminho_csv: str, eh_primeiro_bloco: bool) -> int:
    cols_leitura = COLUNAS + COLUNAS_FILTRO

    cabecalho = pd.read_csv(caminho_csv, sep=SEP, encoding=ENCODING, nrows=0).columns
    faltando = [c for c in cols_leitura if c not in cabecalho]
    if faltando:
        raise ValueError(f"Colunas não encontradas em {caminho_csv}: {faltando}")

    exibir_tabela_faixas(ano)

    total_lidas = 0
    total_salvas = 0

    for chunk in pd.read_csv(caminho_csv, sep=SEP, encoding=ENCODING, usecols=cols_leitura, chunksize=CHUNK_SIZE):
        total_lidas += len(chunk)

        # Filtros de presença, redação válida e não-treineiro
        presentes = (chunk[["TP_PRESENCA_CN", "TP_PRESENCA_CH", "TP_PRESENCA_LC", "TP_PRESENCA_MT"]] == 1).all(axis=1)
        redacao_valida = chunk["NU_NOTA_REDACAO"].notna() & (chunk["NU_NOTA_REDACAO"] > 0)
        nao_treineiros = chunk["IN_TREINEIRO"] == 0

        chunk_filtrado = chunk[presentes & redacao_valida & nao_treineiros].copy()

        if chunk_filtrado.empty:
            continue

        # Inserção do ano e cálculo da renda per capita
        chunk_filtrado["NU_ANO"] = ano
        chunk_filtrado = calcular_renda_per_capita(chunk_filtrado, ano)

        # Formatação de tipo para garantir inteiros limpos no CSV
        chunk_filtrado["Q005"] = pd.to_numeric(chunk_filtrado["Q005"], errors="coerce").astype("Int64")

        # Reordenação estruturada das colunas de saída
        chunk_estruturado = chunk_filtrado[COLUNAS_ORDENADAS_SAIDA].copy()

        # Gravação no CSV
        modo = "w" if eh_primeiro_bloco else "a"
        chunk_estruturado.to_csv(
            SAIDA, 
            sep=";", 
            index=False, 
            encoding="utf-8", 
            mode=modo, 
            header=eh_primeiro_bloco,
            float_format="%.4f"
        )

        eh_primeiro_bloco = False
        total_salvas += len(chunk_estruturado)

    print(f"ENEM {ano}: {total_lidas:,} linhas lidas -> {total_salvas:,} salvas em {SAIDA}")
    return total_salvas


# -----------------------------------------------------------------------------
# EXECUÇÃO PRINCIPAL
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processa microdados do ENEM 2022 e 2023.")
    parser.add_argument("--path2022", type=str, default=ARQUIVOS_PADRAO[2022], help="Caminho do CSV de 2022")
    parser.add_argument("--path2023", type=str, default=ARQUIVOS_PADRAO[2023], help="Caminho do CSV de 2023")
    args = parser.parse_args()

    arquivos = {2022: args.path2022, 2023: args.path2023}
    total_linhas = 0
    primeiro = True

    for ano, caminho in arquivos.items():
        total_linhas += processar_microdados(ano, caminho, eh_primeiro_bloco=primeiro)
        primeiro = False

    print(f"\nProcessamento concluído com sucesso!")
    print(f"Arquivo final: {SAIDA} contendo {total_linhas:,} linhas.")
