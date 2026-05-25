# SebastianADP — Agente Connect 4

Agente basado en **Approximate Dynamic Programming (ADP)** para el torneo de Connect 4.

---

## Que es ADP y como funciona

ADP aprende un modelo empírico del MDP durante la fase `mount()` y luego actua de forma greedy sobre los Q-values resultantes.

**Fase de entrenamiento (`mount`):**

1. Juega `n_trials` partidas contra un oponente aleatorio, usando una política win/block + exploración aleatoria para cubrir variedad de estados.
2. Estima el modelo a partir de las transiciones observadas:
   - **P̂(s'|s,a)**: frecuencias de transición normalizadas por par `(estado, accion)`.
   - **R̂(s,a)**: recompensa media inmediata (+1 victoria, -1 derrota, 0 resto).
3. Ejecuta `vi_iters` sweeps de **value iteration** sobre el modelo estimado:
   `Q(s,a) = R̂(s,a) + γ · Σ P̂(s'|s,a) · V(s')`

Los estados se normalizan (`mis_piezas = +1`) antes de calcular la clave, lo que hace el modelo invariante al color asignado.

**Fase de decision (`act`):**

Las jugadas se evaluan en orden de prioridad:
1. Ganar de inmediato.
2. Bloquear amenaza inmediata del rival.
3. Entre las jugadas "seguras" (no abren amenaza al rival), elegir la de mayor Q-value.
4. Si no hay jugadas seguras, minimizar las amenazas que abre al rival y romper empates con Q-value.

Para estados no vistos en entrenamiento se aplica una heuristica de ventanas de 4 celdas.

---

## Requisitos e instalacion

El proyecto usa la libreria estandar de Python mas:

```
numpy
matplotlib   # solo para el notebook de experimentos
```

Instalar dependencias:

```bash
pip install numpy matplotlib
```

No se requieren librerias externas adicionales. El codigo base `connect4/` ya esta incluido en el repositorio.

---

## Como ejecutar el torneo

Desde la raiz del proyecto (`ProyectoAI/`):

```bash
python main.py
```

`main.py` descubre automaticamente todos los agentes en `groups/` e inscribe a `SebastianADP` junto con el resto de participantes. Al terminar imprime el campeon del torneo.

---

## Como correr el notebook de experimentos

Desde la raiz del proyecto:

```bash
jupyter notebook groups/Sebastian_ADP/entrega.ipynb
```

O con JupyterLab:

```bash
jupyter lab groups/Sebastian_ADP/entrega.ipynb
```

El notebook asume que el directorio de trabajo es la raiz del proyecto (ajusta el `sys.path` automaticamente). Los experimentos incluidos son:

| Experimento | Que mide |
|---|---|
| 1 | Win-rate vs jugador aleatorio para distintos `n_trials` |
| 1b | Desempeno por color asignado (Rojo vs Amarillo) |
| 1c | Learning curve y reward promedio vs episodios |
| 2 | ADP vs ADP con distintos `n_trials` |
| 3 | Tamano del modelo aprendido (estados, pares s-a) vs `n_trials` |
| 4 | Exploracion/explotacion, heuristica y latencia de `act()` |

---

## Parametros configurables

| Parametro | Valor por defecto | Descripcion |
|---|---|---|
| `n_trials` | `500` | Numero de partidas de entrenamiento. Mas partidas = modelo mas preciso = mayor tiempo de entrenamiento (aprox. lineal). Con 500 se logra ~99% de win-rate vs aleatorio en ~5 s. |
| `gamma` | `0.95` | Factor de descuento en la ecuacion de Bellman. Controla cuanto peso tiene el valor futuro respecto a la recompensa inmediata. |
| `vi_iters` | `30` | Numero de sweeps de value iteration sobre el modelo estimado. Valores bajos pueden dar Q-values sin converger; 30 es suficiente para los estados visitados en entrenamiento. |

Ejemplo de uso con parametros personalizados:

```python
from groups.Sebastian_ADP.policy import SebastianADP

agente = SebastianADP(n_trials=1000, gamma=0.99, vi_iters=50)
agente.mount()
accion = agente.act(tablero)
```

---

## Estructura de archivos

```
groups/Sebastian_ADP/
  ├── policy.py        # Implementacion del agente SebastianADP
  ├── entrega.ipynb    # Notebook con experimentos y analisis
  └── README.md        # Este archivo
```
