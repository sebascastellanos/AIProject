import numpy as np
from connect4.policy import Policy
from typing import override
import os
import pickle


class FVMCConnect4Agent(Policy):

    def __init__(self):
        self.max_depth = 4
        self.Q = {}
        self.returns_count = {}

        self.q_values_path = os.path.join(
            os.path.dirname(__file__),
            "q_values.pkl"
        )

    @override
    def mount(self, *args, **kwargs) -> None:
        """
        Se ejecuta antes de iniciar una partida.

        El autocalificador puede llamar mount(timeout), por eso se aceptan
        *args y **kwargs.

        Si existe la tabla Q, la carga.
        Si no existe, entrena una versión ligera con 500 partidas por ficha.
        """

        if os.path.exists(self.q_values_path):
            self.load_q_values()
        else:
            self.train_fvmc(num_episodes=500, epsilon=0.2, agent_piece=-1)
            self.train_fvmc(num_episodes=500, epsilon=0.2, agent_piece=1)
            self.save_q_values()

    @override
    def act(self, s: np.ndarray) -> int:
        """
        Recibe el tablero actual y retorna la columna donde el agente quiere jugar.

        El tablero tiene tamaño 6x7:
        - 0 significa casilla vacía
        - -1 significa ficha roja
        - 1 significa ficha amarilla
        """

        available_cols = self.get_available_columns(s)

        # Seguridad: si por alguna razón no hay columnas disponibles
        if len(available_cols) == 0:
            return 0

        # Identificar qué ficha soy en este turno
        my_piece = self.get_current_player(s)
        opponent_piece = -my_piece

        # Uso de heurísticas y FVMC para decidir la mejor jugada
        # 1. Si puedo ganar en este turno, juego esa columna
        winning_move = self.find_winning_move(s, my_piece, available_cols)
        if winning_move is not None:
            return winning_move

        # 2. Si el oponente puede ganar en su próximo turno, bloqueo esa columna
        opponent_winning_move = self.find_winning_move(
            s,
            opponent_piece,
            available_cols
        )
        if opponent_winning_move is not None:
            return opponent_winning_move

        # 3. Si no hay jugada inmediata, intento usar lo aprendido por FVMC
        learned_action = self.choose_best_learned_action(s, available_cols)
        if learned_action is not None:
            return learned_action

        # 4. Si el estado no fue aprendido, uso estrategia base
        return self.choose_strategic_column(available_cols)

    # ============================================================
    # Heurísticas
    # ============================================================

    def find_winning_move(
        self,
        board: np.ndarray,
        piece: int,
        available_cols: list[int]
    ) -> int | None:
        """
        Revisa si poniendo una ficha en alguna columna disponible,
        el jugador gana inmediatamente.
        """
        for col in available_cols:
            temp_board = board.copy()
            row = self.get_next_open_row(temp_board, col)

            if row is not None:
                temp_board[row, col] = piece

                if self.is_winning_board(temp_board, piece):
                    return col

        return None

    def choose_strategic_column(self, available_cols: list[int]) -> int:
        """
        Se prioriza el centro porque en Connect-4 suele dar más posibilidades
        de crear líneas horizontales y diagonales.
        """

        preferred_order = [3, 2, 4, 1, 5, 0, 6]

        for col in preferred_order:
            if col in available_cols:
                return col

        return available_cols[0]

    # ============================================================
    # FVMC
    # ============================================================

    def train_and_save_full_q_values(self) -> None:
        """
        Entrenamiento local recomendado para crear una tabla Q más fuerte.

        Esta función se puede ejecutar localmente para generar q_values.pkl
        con 10000 partidas como rojo y 10000 partidas como amarillo.
        """

        self.Q = {}
        self.returns_count = {}

        self.train_fvmc(num_episodes=10000, epsilon=0.2, agent_piece=-1)
        self.train_fvmc(num_episodes=10000, epsilon=0.2, agent_piece=1)

        self.save_q_values()

    def train_fvmc(
        self,
        num_episodes: int,
        epsilon: float,
        agent_piece: int
    ) -> None:
        """
        Entrena el agente usando First-Visit Monte Carlo.

        num_episodes: cantidad de partidas simuladas.
        epsilon: probabilidad de explorar.
        agent_piece: ficha del agente durante el entrenamiento (-1 o 1).
        """

        for _ in range(num_episodes):
            episode, reward = self.simulate_episode(
                epsilon=epsilon,
                agent_piece=agent_piece
            )

            self.update_q_first_visit(episode, reward)

    def simulate_episode(
        self,
        epsilon: float,
        agent_piece: int
    ) -> tuple[list[tuple[tuple, int]], float]:
        """
        Simula una partida completa de Connect-4.

        Retorna:
        - episode: lista de pares (estado, acción) visitados por el agente.
        - reward: recompensa final desde la perspectiva del agente.
        """

        board = self.create_empty_board()
        episode = []

        current_piece = -1  # Rojo empieza

        while True:
            available_cols = self.get_available_columns(board)

            if len(available_cols) == 0:
                reward = self.get_reward(winner=0, agent_piece=agent_piece)
                return episode, reward

            if current_piece == agent_piece:
                # Turno del agente: usa epsilon-greedy
                state_key = self.board_to_key(board)
                action = self.choose_epsilon_greedy_action(board, epsilon)
                episode.append((state_key, action))
            else:
                # Turno del oponente: jugador aleatorio
                action = self.choose_random_action(board)

            board = self.apply_action(board, action, current_piece)

            winner = self.get_winner(board)

            if winner != 0:
                reward = self.get_reward(winner=winner, agent_piece=agent_piece)
                return episode, reward

            if self.is_draw(board):
                reward = self.get_reward(winner=0, agent_piece=agent_piece)
                return episode, reward

            current_piece = -current_piece

    def update_q_first_visit(
        self,
        episode: list[tuple[tuple, int]],
        reward: float
    ) -> None:
        """
        Actualiza la tabla Q usando First-Visit Monte Carlo.

        Solo se actualiza la primera aparición de cada par (estado, acción)
        dentro del episodio.
        """

        visited = set()

        for state_action in episode:
            if state_action not in visited:
                visited.add(state_action)

                old_count = self.returns_count.get(state_action, 0)
                old_value = self.Q.get(state_action, 0.0)

                new_count = old_count + 1

                # Promedio incremental:
                # nuevo_valor = viejo_valor + (retorno - viejo_valor) / cantidad
                new_value = old_value + (reward - old_value) / new_count

                self.returns_count[state_action] = new_count
                self.Q[state_action] = new_value

    def get_reward(self, winner: int, agent_piece: int) -> float:
        """
        Retorna la recompensa final desde la perspectiva del agente.

        +1 si gana el agente
         0 si hay empate
        -1 si pierde el agente
        """
        if winner == agent_piece:
            return 1.0
        elif winner == 0:
            return 0.0
        else:
            return -1.0

    # ============================================================
    # Selección de acciones
    # ============================================================

    def choose_random_action(self, board: np.ndarray) -> int:
        """
        Escoge una acción aleatoria entre las columnas disponibles.
        """
        available_cols = self.get_available_columns(board)
        rng = np.random.default_rng()
        return int(rng.choice(available_cols))

    def choose_epsilon_greedy_action(
        self,
        board: np.ndarray,
        epsilon: float
    ) -> int:
        """
        Política epsilon-greedy para entrenamiento FVMC.

        - Con probabilidad epsilon, explora una acción aleatoria.
        - Con probabilidad 1 - epsilon, usa la mejor acción conocida en Q.
        """

        available_cols = self.get_available_columns(board)
        rng = np.random.default_rng()

        # Exploración
        if rng.random() < epsilon:
            return int(rng.choice(available_cols))

        # Explotación
        state_key = self.board_to_key(board)

        best_action = None
        best_value = float("-inf")

        for col in available_cols:
            value = self.Q.get((state_key, col), 0.0)

            if value > best_value:
                best_value = value
                best_action = col

        if best_action is not None:
            return int(best_action)

        return self.choose_strategic_column(available_cols)

    def choose_best_learned_action(
        self,
        board: np.ndarray,
        available_cols: list[int]
    ) -> int | None:
        """
        Busca en la tabla Q la mejor acción aprendida para el estado actual.

        Si el agente conoce valores para este estado, retorna la columna con mayor valor.
        Si no conoce el estado, retorna None.
        """

        state_key = self.board_to_key(board)

        best_action = None
        best_value = float("-inf")
        found_learned_value = False

        for col in available_cols:
            state_action = (state_key, col)

            if state_action in self.Q:
                found_learned_value = True
                value = self.Q[state_action]

                if value > best_value:
                    best_value = value
                    best_action = col

        if found_learned_value:
            return int(best_action)

        return None

    # ============================================================
    # Funciones del tablero
    # ============================================================

    def get_available_columns(self, board: np.ndarray) -> list[int]:
        """
        Retorna las columnas que todavía tienen espacio.
        """
        return [c for c in range(7) if board[0, c] == 0]

    def get_current_player(self, board: np.ndarray) -> int:
        """
        Determina qué jugador debe mover según la cantidad de fichas
        en el tablero (-1: rojo, 1: amarillo).
        """
        red_count = np.sum(board == -1)
        yellow_count = np.sum(board == 1)

        if red_count == yellow_count:
            return -1
        else:
            return 1

    def get_next_open_row(self, board: np.ndarray, col: int) -> int | None:
        """
        Retorna la fila donde caería la ficha en una columna.
        """
        for row in range(5, -1, -1):
            if board[row, col] == 0:
                return row

        return None

    def apply_action(self, board: np.ndarray, col: int, piece: int) -> np.ndarray:
        """
        Aplica una jugada en una copia del tablero y retorna el nuevo tablero.
        """
        new_board = board.copy()
        row = self.get_next_open_row(new_board, col)

        if row is not None:
            new_board[row, col] = piece

        return new_board

    def is_winning_board(self, board: np.ndarray, piece: int) -> bool:
        """
        Verifica si el jugador indicado tiene 4 fichas conectadas.
        """

        rows = 6
        cols = 7

        # Horizontal
        for r in range(rows):
            for c in range(cols - 3):
                if (
                    board[r, c] == piece and
                    board[r, c + 1] == piece and
                    board[r, c + 2] == piece and
                    board[r, c + 3] == piece
                ):
                    return True

        # Vertical
        for r in range(rows - 3):
            for c in range(cols):
                if (
                    board[r, c] == piece and
                    board[r + 1, c] == piece and
                    board[r + 2, c] == piece and
                    board[r + 3, c] == piece
                ):
                    return True

        # Diagonal positiva /
        for r in range(3, rows):
            for c in range(cols - 3):
                if (
                    board[r, c] == piece and
                    board[r - 1, c + 1] == piece and
                    board[r - 2, c + 2] == piece and
                    board[r - 3, c + 3] == piece
                ):
                    return True

        # Diagonal negativa \
        for r in range(rows - 3):
            for c in range(cols - 3):
                if (
                    board[r, c] == piece and
                    board[r + 1, c + 1] == piece and
                    board[r + 2, c + 2] == piece and
                    board[r + 3, c + 3] == piece
                ):
                    return True

        return False

    def get_winner(self, board: np.ndarray) -> int:
        """
        Retorna:
        -1 si gana rojo
         1 si gana amarillo
         0 si todavía no hay ganador
        """
        if self.is_winning_board(board, -1):
            return -1

        if self.is_winning_board(board, 1):
            return 1

        return 0

    def is_draw(self, board: np.ndarray) -> bool:
        """
        Retorna True si el tablero está lleno y no hay más columnas disponibles.
        """
        return len(self.get_available_columns(board)) == 0

    def create_empty_board(self) -> np.ndarray:
        """
        Crea un tablero vacío.
        """
        return np.zeros((6, 7), dtype=int)

    def board_to_key(self, board: np.ndarray) -> tuple:
        """
        Convierte el tablero en una clave inmutable para poder usarlo en diccionarios.

        Ideal para guardar valores Q[(estado, accion)] = valor_estimado.
        """
        return tuple(board.flatten())

    # ============================================================
    # Persistencia de la tabla Q
    # ============================================================

    def save_q_values(self) -> None:
        """
        Guarda la tabla Q y el contador de retornos en un archivo.
        """

        data = {
            "Q": self.Q,
            "returns_count": self.returns_count
        }

        with open(self.q_values_path, "wb") as file:
            pickle.dump(data, file)

    def load_q_values(self) -> None:
        """
        Carga la tabla Q y el contador de retornos desde un archivo.
        """

        with open(self.q_values_path, "rb") as file:
            data = pickle.load(file)

        self.Q = data.get("Q", {})
        self.returns_count = data.get("returns_count", {})