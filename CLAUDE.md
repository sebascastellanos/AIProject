## Approach
- Read existing files before writing. Don't re-read unless changed.
- Thorough in reasoning, concise in output.
- Skip files over 100KB unless required.
- No sycophantic openers or closing fluff.
- No emojis or em-dashes.
- Do not guess APIs, versions, flags, commit SHAs, or package names. Verify by reading code or docs before asserting.

# Contexto del Proyecto AI - Connect 4

## Estructura Estricta del Repositorio (Tree)
Siempre que crees archivos, analices el código o sugieras cambios, debes respetar este árbol de directorios:

```text
ProyectoAI/
  ├── main.py
  ├── tournament.py         <- Ejecución del torneo general
  ├── CLAUDE.md                     <- (Este archivo de configuración)       
  ├── connect4/             <- Código base del profesor (No se modifica)
  │    ├── policy.py
  │    ├── connect_state.py
  │    ├── environment_state.py
  │    ├── utils.py
  │    └── dtos.py
  ├── groups/               <- Carpetas individuales por integrante
  │    ├── Group A/policy.py <-- Carpeta de Mari con fvmc
  │    ├── Group B/policy.py <-- Carpeta de Vale para hacer mcts
  │    └── Sebastian_ADP/   <- Carpeta de Sebas (Rama: ADP)
  │         ├── policy.py   <- Agente definitivo ADP (Ecuación de Bellman)
  │         └── entrega.ipynb <- Notebook de experimentos de Sebas
  └── versus/               <- Logs de partidas guardados automáticamente en .json


