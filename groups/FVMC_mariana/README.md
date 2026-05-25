# First-Visit Monte Carlo - Agente Connect-4

**Mariana Valle Moreno**  
**Universidad de La Sabana**  
**Inteligencia Artificial - 2026-2**

---

## Qué hace el agente

Este agente juega Connect-4 utilizando un enfoque basado en **First-Visit Monte Carlo (FVMC)**.

La idea principal es que el agente aprende jugando muchas partidas simuladas contra un oponente aleatorio. A partir de esas partidas construye una **tabla Q**, donde guarda el valor estimado de tomar una acción en un estado específico del tablero.

Durante una partida real, el agente no entrena desde cero. En cambio, usa el conocimiento previamente aprendido y guardado en `q_values.pkl` para tomar mejores decisiones.

El agente combina:

- Reglas tácticas simples para ganar o bloquear jugadas inmediatas (**heurística**).
- Una tabla Q aprendida mediante First-Visit Monte Carlo.
- Una estrategia base que prioriza las columnas centrales si el estado actual no fue aprendido.

---

## Cómo funciona

El flujo de decisión del agente es el siguiente:

1. Revisa las columnas disponibles.
2. Identifica qué ficha debe jugar en el turno actual.
3. Si puede ganar inmediatamente, juega esa columna.
4. Si el oponente puede ganar en el siguiente turno, bloquea esa columna.
5. Si no hay una jugada inmediata, consulta la tabla Q aprendida con FVMC.
6. Si el estado no existe en la tabla Q, usa una estrategia base que prioriza el centro del tablero.

Como se mencionó anteriormente, la tabla Q se guarda y se puede encontrar específicamente en la siguiente ruta desde la raíz del proyecto:

```text
groups/FVMC_mariana/q_values.pkl
```

Esta tabla ya fue entrenada localmente con:

```python
self.train_fvmc(num_episodes=10000, epsilon=0.2, agent_piece=-1)
self.train_fvmc(num_episodes=10000, epsilon=0.2, agent_piece=1)
```

Es decir, se entrenó con **10000 partidas como jugador rojo** y **10000 partidas como jugador amarillo**.

Para generar nuevamente la tabla de manera local, se puede crear un archivo auxiliar en la raíz del proyecto. Por ejemplo:

```text
train_q_values.py
```

Con el siguiente contenido:

```python
from groups.FVMC_mariana.policy import FVMCConnect4Agent

agent = FVMCConnect4Agent()
agent.train_and_save_full_q_values()

print("Tabla Q generada correctamente en groups/FVMC_mariana/q_values.pkl")
```

Luego se ejecuta desde la raíz del proyecto:

```bash
py train_q_values.py
```

Si se usa otro nombre para el archivo auxiliar, el comando sería:

```bash
py nombre_del_archivo.py
```

Si el archivo `q_values.pkl` ya existe, el agente simplemente lo carga y no vuelve a entrenar.

Si el archivo `q_values.pkl` no se encuentra, el agente realiza un entrenamiento auxiliar más liviano de emergencia:

```python
self.train_fvmc(num_episodes=500, epsilon=0.2, agent_piece=-1)
self.train_fvmc(num_episodes=500, epsilon=0.2, agent_piece=1)
```

Esto permite que el agente pueda ejecutarse incluso si la tabla Q no está disponible, aunque el rendimiento esperado será menor que usando la tabla entrenada localmente con 10000 partidas por color.

---

## Cómo ejecutarlo

Para ejecutar el torneo desde la raíz del proyecto:

```bash
py main.py
```

El agente se encuentra en:

```text
groups/FVMC_mariana/policy.py
```

Los archivos principales de esta entrega son:

```text
groups/FVMC_mariana/
├── policy.py
├── q_values.pkl
├── entrega.ipynb
└── README.md
```

**Nota:** El archivo `q_values.pkl` debe mantenerse en la misma carpeta que `policy.py`, ya que el agente lo busca automáticamente en esa ubicación.
