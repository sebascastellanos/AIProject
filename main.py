import numpy as np
from connect4.policy import Policy
from connect4.utils import find_importable_classes
from tournament import run_tournament, play 

# 1. Leer los 4 participantes
participants = find_importable_classes("groups", Policy)
players = list(participants.items())

print("Participantes reales encontrados:", [p[0] for p in players])

# 2. Correr el torneo atrapando los 5 argumentos con *args
champion = run_tournament(
    players=players,
    play=lambda *args: play(*args), # <-- ¡ESTA ES LA MAGIA! Recibe los 5 y pasa los 5
    best_of=7,
    first_player_distribution=0.5,
    shuffle=True, 
    seed=911
)

print("\n🏆 ¡EL CAMPEÓN DEL TORNEO OFICIAL ES:", champion[0])