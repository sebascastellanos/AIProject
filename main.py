from connect4.policy import Policy
from connect4.utils import find_importable_classes
from tournament import run_tournament, play

# Read all files within subfolder of "groups"
participants = find_importable_classes("groups", Policy)

# Build a participant list (name, class)
#players = list(participants.items())

# Solo prueba
ignored_players = ["Group C"]

# Build a participant list (name, class)
players = [
    participant
    for participant in participants.items()
    if participant[0] not in ignored_players
]

# Run the tournament
champion = run_tournament(
    players,
    play,  # You could also create your own play function for testing purposes
    shuffle=True,
)
print("Champion:", champion)
