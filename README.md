# Previsão Vibecodada da Copa

> Motor preditivo Bayesiano de futebol que roda no terminal, consome CSV, cospe probabilidade — e não pede desculpa.

Sim, é **vibecodado**. Um humano teve uma ideia, um LLM ajudou a escrever metade do código, e o resultado é um ensemble estatístico de verdade escondido atrás de uma CLI bonitinha com `rich`. Não tem React. Não tem Kubernetes. Não tem fila RabbitMQ. Tem Python, pandas, e a audácia de achar que Poisson explica o futebol melhor que o seu cunhado no churrasco.

---

## O que é isso?

Uma ferramenta **local** para prever resultados de jogos entre seleções. Você roda no terminal, ela lê arquivos, treina um modelo, e te diz:

- Probabilidade de vitória / empate / derrota
- Gols esperados (λ)
- Os 3 placares exatos mais prováveis

Foco atual: **Copa do Mundo 2026**, mas o histórico vai desde 1990 (~32k jogos) ou 1872 (~49k) se você quiser sofrer.

---

## Por que "vibecodado"?

Porque nasceu de uma sessão de vibe coding com Cursor, inspirado nos melhores modelos públicos (onthepitch, wc2026, playmobil), mas implementado num fim de semana com:

- Estrutura flat (`cli.py` + `model.py` + `components/`)
- Zero microsserviço
- Zero banco de dados
- Zero API web
- 100% "roda na minha máquina Pop!_OS e me obedece"

A estatística é séria. O packaging é caótico. **Vibecode com responsabilidade.**

---

## Como funciona (a parte que impressiona o nerd do grupo)

O modelo combina **3 componentes** num ensemble uniforme (média 1/3 cada — evidência empírica diz que stacking fancy perde pro simples):

| Componente | O que faz |
|------------|-----------|
| **Elo** | Rating clássico, anti-overfit, âncora estável |
| **Dixon-Coles** | Poisson bivariado com correção para 0-0, 1-1, etc. |
| **Hierárquico** | Ataque/defesa com shrinkage por confederação FIFA |

Jogos **mais recentes pesam mais**. Meia-vida de **5 anos** — um jogo de 1990 pesa ~0,6%; um jogo da Copa 2026 pesa 100%. O CSV tem histórico antigo, mas o modelo não trata 1990 como se fosse ontem.

```
P(final) = média(P_elo, P_dixon_coles, P_hierárquico)
         → calibração por tier (amistoso / eliminatória / torneio)
         → Monte Carlo com 50k–100k simulações
         → tabelas bonitas no terminal
```

---

## Instalação

```bash
git clone https://github.com/SEU_USER/prever-copa.git
cd prever-copa

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ funciona. O plano original pedia 3.11; a vida é curta.

---

## Uso rápido (3 comandos e você já é analista)

### 1. Baixar dados e montar a base

```bash
# Histórico completo desde 1990 + Elo do eloratings.net
python cli.py bootstrap-data --fetch-elo

# OU: sincronizar só a Copa 2026 (recomendado durante o torneio)
python cli.py sync-wc2026 --cutoff 2026-06-30 --retrain
```

### 2. Treinar o modelo

```bash
python cli.py train --fast
```

Gera `model_weights.json` com os pesos de ataque/defesa de cada seleção.

### 3. Prever um jogo

```bash
# Brasil vs Noruega (oitavas, campo neutro)
python cli.py predict --team-a Brazil --team-b Norway --neutral --sims 100000

# Funciona em português também
python cli.py predict --team-a Brasil --team-b Argentina --neutral
```

### Extras

```bash
python cli.py rank --top 20          # ranking global por força
python cli.py backtest               # stub — Fase 3 ainda não chegou
python cli.py --help                 # a Bíblia
```

---

## Onde estão os jogos?

| Arquivo | Conteúdo |
|---------|----------|
| `data/matches.csv` | Base normalizada — **o modelo lê daqui** |
| `data/raw/results.csv` | Download bruto do [martj42](https://github.com/martj42/international_results) |
| `data/wc2026_patches.csv` | Placares manuais que o martj42 ainda não tinha |
| `data/teams_context.csv` | Elo, GDP, cultura futebolística, etc. |
| `model_weights.json` | Pesos treinados (gerado pelo `train`) |

Para ver só a Copa 2026:

```bash
python3 -c "
import pandas as pd
wc = pd.read_csv('data/matches.csv')
print(wc[(wc.tournament=='FIFA World Cup') & (wc.date>='2026-06-11')][['date','home_team','away_team','home_goals','away_goals']])
"
```

---

## Estrutura do projeto

```
prever-copa/
├── cli.py                 # Interface terminal (Typer + Rich)
├── model.py               # Orquestrador train/predict/rank
├── bootstrap_data.py        # Download martj42 + patches Copa 2026
├── features.py            # Elo, decay temporal, índice cultural
├── ensemble.py            # Fusão dos 3 componentes
├── calibration.py         # Calibração por tier
├── components/
│   ├── elo_model.py
│   ├── dixon_coles.py
│   └── hierarchical.py
└── data/
    ├── matches.csv
    ├── teams_context.csv
    ├── wc2026_patches.csv
    └── ...
```

Flat. Simples. Como seu time joga quando precisa ganhar.

---

## Atualizar jogos da Copa em tempo real

Quando sair um resultado novo:

1. Se o [martj42](https://github.com/martj42/international_results) já tiver atualizado → só rode:
   ```bash
   python cli.py sync-wc2026 --cutoff 2026-07-04 --retrain
   ```

2. Se ainda não tiver → adicione uma linha em `data/wc2026_patches.csv`:
   ```csv
   date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
   2026-07-01,Mexico,Ecuador,2,1,FIFA World Cup,Mexico City,Mexico,FALSE
   ```
   E rode o `sync-wc2026` de novo.

---

## Fontes de dados

- **[martj42/international_results](https://github.com/martj42/international_results)** — ~49k jogos internacionais desde 1872
- **[eloratings.net](https://www.eloratings.net/)** — Elo ratings ao vivo (TSV)
- **World Bank / socio_economic.csv** — GDP, população (curado manualmente)
- **Patches manuais** — porque nem todo dataset atualiza na hora que o Haaland marca

---

## Roadmap (o que ainda não vibecodamos)

- [x] Fase 1 — CLI + bootstrap + ensemble heurístico + Copa 2026
- [ ] Fase 2 — Dixon-Coles MLE real + calibração isotônica + backtest walk-forward
- [ ] Fase 3 — PyMC NUTS hierárquico com posterior completa
- [ ] Fase 4 — Simulador de chaves (grupos → mata-mata) + recalibração pós-jogo

---

## Disclaimer

Isso **não é recomendação de aposta**. É um experimento estatístico vibecodado para diversão e aprendizado. O futebol é irracional por design — o modelo disse que Brasil ganhava e o VAR disse "segura aí". Use com humor e um grão de sal do tamanho do Maracanã.

---

## Licença

MIT — faça o que quiser, mas se acertar a zebra, manda um print.

---

<p align="center">
  <i>Feito com Python, café, inferência Bayesiana e excesso de confiança.</i><br>
  <b>Previsão Vibecodada da Copa</b> — porque microserviço é coisa de quem não confia no terminal.
</p>
