
<img width="6358" height="1896" alt="image" src="https://github.com/user-attachments/assets/a7856e56-e60c-4f7c-b155-e85703d4f41b" />

Programa de Pós Graduação em Engenharia de Software CIN - UFPE 

Aluna : Maria Gabriela Alves Zuppardo 

Orientador : Vinicius Cardoso Garcia

# Pipeline Multi-Agente para Engenharia de Requisitos

Proposta de avaliação empírica de um sistema multi-agente baseado em LLMs para execução integrada de elicitação, classificação e priorização de requisitos de software, com protocolo automatizado e replicado.

## Estrutura do Projeto

```
mas_re/
├── agents/                    # Agentes especializados
│   ├── elicitador.py          # Agente de elicitação de requisitos
│   ├── classificador.py       # Agente de classificação FR/NFR
│   └── priorizador.py         # Agente de priorização (AHP/MoSCoW)
│
├── pipeline/                  # Pipeline multi-agente (LangGraph)
│   ├── graph.py               # Definição do grafo LangGraph
│   ├── state.py               # Estado compartilhado entre agentes
│   ├── handoff.py             # Protocolos de handoff entre agentes
│   └── nodes/                 # Nós do grafo
│
├── baseline/                  # Agente único para comparação
│   └── single_agent.py        # Baseline: agente único sequencial
│
├── evaluation/                # Framework de avaliação automatizado
│   ├── protocol.py            # Protocolo de avaliação reprodutível
│   ├── runner.py              # Executor de experimentos controlados
│   └── metrics/
│       ├── classification.py  # F1, Precision, Recall (FR/NFR)
│       └── prioritization.py  # Kendall τ, NDCG
│
├── datasets/                  # Datasets anotados públicos
│   ├── promise_nfr/           # PROMISE NFR Dataset
│   └── nfric/                 # NFRIC Dataset
│
├── prompts/                   # Templates de prompts por agente
│   ├── elicitacao.py
│   ├── classificacao.py
│   └── priorizacao.py
│
├── config/                    # Configurações do sistema
│   ├── settings.py
│   └── models.py
│
├── experiments/               # Resultados e logs dos experimentos
│   ├── results/
│   └── logs/
│
├── tests/                     # Testes unitários e de integração
│   ├── unit/
│   └── integration/
│
├── notebooks/                 # Análise exploratória e visualizações
├── docs/                      # Documentação
│   ├── architecture/          # Diagramas de arquitetura MAS
│   └── protocols/             # Protocolos de handoff documentados
│
└── requirements.txt
```

## Perguntas de Pesquisa

**PQ:** Em que medida um pipeline multi-agente baseado em LLMs é capaz de executar de forma integrada a classificação e priorização de requisitos de software, e qual a qualidade do output gerado em comparação a abordagens de agente único?

- **SQ1 – Eficácia por etapa:** Qualidade do output de cada agente individualmente contra datasets PROMISE NFR e NFRIC.
- **SQ2 – Ganho de colaboração:** Pipeline multi-agente vs. agente único em qualidade e consistência.

## Métricas de Avaliação

| Tarefa | Métricas |
|--------|----------|
| Classificação FR/NFR | F1, Precision, Recall |
| Priorização | Kendall τ, NDCG |

## Setup

```bash
cp .env.example .env
pip install -r requirements.txt
```
