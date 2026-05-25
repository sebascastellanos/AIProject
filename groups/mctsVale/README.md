# MCTS Connect-4 Agent

**Valentina López **  
Universidad de La Sabana — Inteligencia Artificial 2026-1

---

## Idea principal

Este agente juega Connect-4 mediante **Monte Carlo Tree Search (MCTS)** con
**UCB1** como criterio de selección y **RAVE** (Rapid Action Value Estimation)
para acelerar la convergencia del árbol de búsqueda.

A diferencia de los agentes basados en tablas o modelos aprendidos offline,
este agente construye su árbol de decisión de forma online durante cada
movimiento, usando simulaciones para estimar el valor de cada posición.
El árbol se precalienta en `mount()` y se complementa con simulaciones
adicionales en cada llamada a `act()`.

---

## Cómo funciona

El agente combina cuatro componentes principales:

**1. Bitboard**  
El tablero se representa como dos enteros de 64 bits (uno por jugador).
Las operaciones de victoria y movimiento se realizan con operaciones
bitwise, lo que resulta aproximadamente 10 veces más rápido que
operar directamente sobre arrays de numpy.

**2. MCTS con UCB1**  
Cada simulación sigue cuatro fases:

- Seleccion: desciende por el árbol eligiendo el hijo con mayor UCB1+RAVE
- Expansion: agrega un nodo no explorado al árbol
- Simulacion: juega hasta el final con heurística (ganar/bloquear) y evalúa el tablero
- Retropropagacion: actualiza visitas y victorias hacia la raíz

**3. RAVE (Rapid Action Value Estimation)**  
Además de las estadísticas clásicas por nodo, el agente mantiene
estadísticas globales por acción. Cuando un nodo tiene pocas visitas,
RAVE guía la búsqueda con información de toda la partida. A medida
que las visitas crecen, UCB1 toma el control. Esto acelera la
convergencia significativamente con el mismo número de simulaciones.

**4. Evaluación de tablero**  
En lugar de simular partidas completas al azar, el agente juega un
rollout corto con heurística y luego evalúa la posición con una
función que analiza todas las ventanas de 4 celdas del tablero,
asignando puntos por fichas propias y penalizaciones por amenazas
del rival.

El flujo de decisión en `act()` sigue estas prioridades:

1. Ganar inmediatamente si hay columna ganadora
2. Bloquear victoria inmediata del oponente
3. Crear una doble amenaza propia (fork)
4. Bloquear una doble amenaza del oponente
5. MCTS + UCB1 + RAVE con árbol precalentado

---

## Parametros configurables

| Parametro | Default | Descripcion |
|---|---|---|
| `n_simulations` | 300 | Simulaciones online por movimiento |
| `warmup_time` | 4.0 | Segundos de precalentamiento en `mount()` |
| `c` | sqrt(2) | Constante de exploracion UCB1 |
| `k_rave` | 300.0 | Balance entre RAVE y UCB1 |
| `use_heuristic` | True | Rollout con heuristica vs completamente aleatorio |

---

## Cómo ejecutarlo

Desde la raíz del proyecto:

```
python main.py
```

El agente se encuentra en:

```
groups/mctsVale/policy.py
```

Los archivos de esta entrega son:

```
groups/mctsVale/
├── policy.py
├── entrega.ipynb
└── README.md
```

No se requiere ningún archivo de datos adicional. El agente no guarda
modelos en disco — todo el procesamiento ocurre en memoria durante
`mount()` y `act()`.

---

## Diferencia respecto a los otros agentes del grupo

El agente de Sebastian (ADP) aprende un modelo del MDP offline
(transiciones y recompensas) y aplica value iteration para obtener
Q-values. Es efectivo en estados visitados durante el entrenamiento.

Este agente no modela transiciones. En cambio, construye un árbol de
búsqueda directamente sobre el estado actual del juego, evaluando
posiciones mediante simulaciones guiadas. Esto lo hace más robusto
en estados no vistos durante el precalentamiento.
