# MCTSVale — Agente Connect-4

Agente basado en **Monte Carlo Tree Search (MCTS) + UCB1** para el torneo de Connect-4.

Valentina López - Universidad de la Sabana 2026-1

---

## Qué es MCTS y cómo funciona

MCTS construye un árbol de búsqueda en tiempo real simulando partidas completas desde el estado actual, sin necesidad de aprender ni guardar una tabla de valores.

**Fase de precalentamiento (`mount`):**

1. Juega simulaciones desde el tablero vacío durante `warmup_time` segundos, construyendo un árbol de apertura reutilizable.
2. Cada nodo del árbol guarda `visits` y `total_reward` acumulados desde la perspectiva del jugador raíz.
3. La selección de ramas usa **UCB1**: `Q(s,a)/N + c · √(ln(N_padre) / N)`, balanceando exploración y explotación.

**Fase de decisión (`act`):**

Las jugadas se evalúan en orden de prioridad:

1. Ganar de inmediato.
2. Bloquear amenaza inmediata del rival.
3. Crear un **fork** propio (dos amenazas simultáneas que el rival no puede bloquear ambas).
4. Bloquear un fork del rival.
5. Si ninguna aplica, corre **MCTS + UCB1** y elige la columna más visitada.

El tablero se representa internamente como dos enteros de 64 bits (**bitboard**), lo que hace las verificaciones de victoria ~10× más rápidas que con NumPy.

---

## Requisitos e instalación

El proyecto usa la librería estándar de Python más:

```
numpy
matplotlib   # solo para el notebook de experimentos
```

Instalar dependencias:

```bash
pip install numpy matplotlib
```

No se requieren librerías externas adicionales. El código base `connect4/` ya está incluido en el repositorio.

---

## Cómo ejecutar el torneo

Desde la raíz del proyecto:

```bash
python main.py
```

`main.py` descubre automáticamente todos los agentes en `groups/` e inscribe a `MCTSVale` junto con el resto de participantes.

---

## Cómo correr el notebook de experimentos

Desde la raíz del proyecto:

```bash
jupyter notebook groups/mctsVale/entrega.ipynb
```

O con JupyterLab:

```bash
jupyter lab groups/mctsVale/entrega.ipynb
```

El notebook asume que el directorio de trabajo es la raíz del proyecto (ajusta el `sys.path` automáticamente). Los experimentos incluidos son:

| Experimento | Qué mide |
|---|---|
| 1 | Win-rate vs jugador aleatorio para distintos `n_simulations` |
| 2 | Desempeño por color asignado (Rojo vs Amarillo) |
| 3 | Rollout guiado vs rollout aleatorio (`use_heuristic`) en enfrentamiento directo |
| 4 | Self-play: MCTS(200) vs versiones con menos simulaciones |
| 5 | Latencia de `act()` vs número de simulaciones |

---

## Parámetros configurables

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `n_simulations` | `500` | Simulaciones MCTS por movimiento. Más simulaciones = mejores decisiones = mayor tiempo de cómputo (relación aproximadamente lineal). Con 200 se logra 100% de win-rate vs aleatorio. |
| `warmup_time` | `4.0` s | Segundos de precalentamiento del árbol en `mount()`. Construye un árbol de apertura reutilizable antes de la primera partida. |
| `c` | `1.4` | Constante de exploración UCB1. Controla el balance entre explorar ramas nuevas y explotar las que ya funcionaron. |
| `rollout_limit` | `100` | Pasos máximos del rollout por simulación. Limita partidas muy largas. |
| `use_heuristic` | `True` | `True` → el rollout prefiere ganar o bloquear si puede. `False` → rollout completamente aleatorio. |

Ejemplo de uso con parámetros personalizados:

```python
from groups.mctsVale.policy import MCTSVale

agente = MCTSVale(n_simulations=200, warmup_time=4.0, c=1.4, use_heuristic=True)
agente.mount()
accion = agente.act(tablero)
```

---

## Estructura de archivos

```
groups/mctsVale/
  ├── policy.py        # Implementación del agente MCTSVale
  ├── entrega.ipynb    # Notebook con experimentos y análisis
  └── README.md        # Este archivo
```
