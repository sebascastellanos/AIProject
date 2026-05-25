import sys
sys.path.insert(0, '.')

import numpy as np
from groups.mctsVale.policy import MCTSVale
from connect4.connect_state import ConnectState

class RandomPolicy:
    def mount(self): pass
    def act(self, s):
        available = [c for c in range(7) if s[0, c] == 0]
        return int(np.random.choice(available))

def play_game(agent, random, mcts_is_red=True):
    state = ConnectState()
    while not state.is_final():
        if state.player == -1:
            col = agent.act(state.board) if mcts_is_red else random.act(state.board)
        else:
            col = random.act(state.board) if mcts_is_red else agent.act(state.board)
        state = state.transition(col)
    return state.get_winner()

# Configuración
N = 20  # número de partidas (bájalo si es muy lento)
agent = MCTSVale(n_simulations=100)
agent.mount()
random = RandomPolicy()

# MCTS como rojo (-1)
wins, losses, draws = 0, 0, 0
for _ in range(N):
    w = play_game(agent, random, mcts_is_red=True)
    if w == -1: wins += 1
    elif w == 1: losses += 1
    else: draws += 1
print(f"MCTS como ROJO   → V:{wins} D:{losses} E:{draws} de {N} partidas")

# MCTS como amarillo (1)
wins, losses, draws = 0, 0, 0
for _ in range(N):
    w = play_game(agent, random, mcts_is_red=False)
    if w == 1: wins += 1
    elif w == -1: losses += 1
    else: draws += 1
print(f"MCTS como AMARILLO → V:{wins} D:{losses} E:{draws} de {N} partidas")