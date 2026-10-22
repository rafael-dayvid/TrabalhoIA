"""
Autores: Pedro Henrique Siqueira de Paula Peixoto,
         Rafael Dayvid de Melo Freitas,
         Thales Henrique Marques Lázaro,
         Vinícius Duarte de Carvalho.

Divisão da base de dados do ENEM em Treino, Validação e Teste mantendo a estratificação.
"""

import argparse
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

CATEGORICAS = ["TP_SEXO"] + [f"Q{i:03d}" for i in range(1, 26) if i != 5]
SEP = ";"

def gravar(df: pd.DataFrame, caminho: str) -> None:
    df.to_csv(caminho, sep=SEP, index=False, encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Divide a base do ENEM em treino, validação e teste.")
    parser.add_argument("--principal", type=str, default="enem_treino.csv", help="Caminho da base principal")
    parser.add_argument("--validacao", type=str, default="enem_validacao.csv", help="Caminho da base de validação")
    parser.add_argument("--teste", type=str, default="enem_teste.csv", help="Caminho da base de teste")
    parser.add_argument("--n-val", type=int, default=200_000, help="Número de amostras para validação")
    parser.add_argument("--n-test", type=int, default=200_000, help="Número de amostras para teste")
    parser.add_argument("--semente", type=int, default=42, help="Semente aleatória (seed)")
    
    args = parser.parse_args()

    # Trava de segurança contra re-execução
    for arq in (args.validacao, args.teste):
        if os.path.exists(arq):
            sys.exit(f"ERRO: O arquivo '{arq}' já existe. Apague os arquivos gerados caso queira refazer o split.")

    if not os.path.exists(args.principal):
        sys.exit(f"ERRO: O arquivo principal '{args.principal}' não foi encontrado.")

    # Tipagem para economia de memória
    dtypes = {c: "category" for c in CATEGORICAS}
    dtypes["Q005"] = "Int64"
    if "NU_ANO" in dtypes:
        dtypes["NU_ANO"] = "int16"

    print(f"Lendo {args.principal}...")
    df = pd.read_csv(args.principal, sep=SEP, encoding="utf-8", dtype=dtypes)
    total = len(df)
    
    n_total_solicitado = args.n_val + args.n_test
    if total <= n_total_solicitado:
        sys.exit(f"ERRO: O arquivo tem {total:,} linhas, insuficiente para retirar {args.n_val:,} + {args.n_test:,}.")

    print(f"Base carregada: {total:,} linhas.\n")

    # Amostragem Estratificada por 'NU_ANO' (se a coluna existir)
    coluna_estratificacao = "NU_ANO" if "NU_ANO" in df.columns else None

    # 1. Separa o grupo de Validação
    frac_val = args.n_val / len(df)
    df_restante, df_val = train_test_split(
        df,
        test_size=frac_val,
        random_state=args.semente,
        stratify=df[coluna_estratificacao] if coluna_estratificacao else None
    )

    # 2. Separa o grupo de Teste
    frac_test = args.n_test / len(df_restante)
    df_treino, df_test = train_test_split(
        df_restante,
        test_size=frac_test,
        random_state=args.semente,
        stratify=df_restante[coluna_estratificacao] if coluna_estratificacao else None
    )

    # Gravação dos arquivos divididos
    gravar(df_val, args.validacao)
    print(f"Validação salvos: {len(df_val):,} linhas -> {args.validacao}")

    gravar(df_test, args.teste)
    print(f"Teste salvos: {len(df_test):,} linhas -> {args.teste}")

    # Atualização da base principal (Treino) via escrita atômica
    temporario = args.principal + ".tmp"
    gravar(df_treino, temporario)
    os.replace(temporario, args.principal)
    print(f"Base principal atualizada (Treino): {len(df_treino):,} linhas -> {args.principal}\n")

    # Conferência de integridade
    set_val = set(df_val["NU_INSCRICAO"])
    set_test = set(df_test["NU_INSCRICAO"])
    set_treino = set(df_treino["NU_INSCRICAO"])

    repetidos = len(set_val & set_test) + len(set_val & set_treino) + len(set_test & set_treino)
    soma_linhas = len(df_val) + len(df_test) + len(df_treino)

    print(f"--- Conferência ---")
    print(f"Inscrições repetidas: {repetidos}")
    print(f"Soma das linhas geradas: {soma_linhas:,} (Original: {total:,})")

if __name__ == "__main__":
    main()
