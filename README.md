# Processamento e Análise de Dados do ENEM (2022–2023)

Projeto desenvolvido para a disciplina de **Inteligência Artificial**, focado no pré-processamento, engenharia de atributos (feature engineering) e preparação de dados dos microdados do ENEM 2022 e 2023 para treinamento de modelos preditivos.

---

## 👥 Autores

* **Pedro Henrique Siqueira de Paula Peixoto**
* **Rafael Dayvid de Melo Freitas**
* **Thales Henrique Marques Lázaro**
* **Vinícius Duarte de Carvalho**

---

## 📌 Visão Geral do Projeto

O objetivo principal deste projeto é construir uma base consolidada e limpa de dados socioeconômicos e demográficos de participantes do ENEM, permitindo o desenvolvimento de modelos de Inteligência Artificial voltados para a análise de perfil, equidade e desempenho dos estudantes.

### 📊 Fluxo de Tratamento dos Dados:
1. **Filtragem de Participantes:** Remoção de inscritos ausentes em qualquer dia de prova, treineiros (`IN_TREINEIRO == 1`) e candidatos com nota de redação zerada ou nula.
2. **Engenharia de Features:** Mapeamento da questão socioeconômica $Q006$ (renda familiar) e $Q005$ (quantidade de moradores) para calcular as estimativas de renda *per capita* mínima e máxima (`SAL_PER_CAPITA_MIN` e `SAL_PER_CAPITA_MAX`), expressas em múltiplos do salário mínimo do ano correspondente.
3. **Divisão de Conjuntos (Data Splitting):** Separação estratificada e reproduzível dos dados em bases disjuntas de **Treino**, **Validação** e **Teste**.

---

## 📁 Estrutura do Repositório

```text
TrabalhoIA/
├── data/                      # Diretório reservado para dados locais (ignorado no git)
├── scripts/
│   ├── processar_enem.py      # Processamento inicial dos microdados do Inep
│   └── separar_conjuntos.py   # Divisão em treino, validação e teste
├── .gitignore
├── README.md
└── requirements.txt
