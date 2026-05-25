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
        

        if os.path.exists(self.q_values_path):
            self.load_q_values()
        else:
            self.train_fvmc(num_episodes=500, epsilon=0.2, agent_piece=-1)
            self.train_fvmc(num_episodes=500, epsilon=0.2, agent_piece=1)
            self.save_q_values()

    @override
    def act(self, s: np.ndarray) -> int:
        

        available_cols = self.get_available_columns(s)

        if len(available_cols) == 0:
            return 0

        my_piece = self.get_current_player(s)
        opponent_piece = -my_piece

        winning_move = self.find_winning_move(s, my_piece, available_cols)
        if winning_move is not None:
            return winning_move

        opponent_winning_move = self.find_winning_move(
            s,
            opponent_piece,
            available_cols
        )
        if opponent_winning_move is not None:
            return opponent_winning_move

        learned_action = self.choose_best_learned_action(s, available_cols)
        if learned_action is not None:
            return learned_action

        return self.choose_strategic_column(available_cols)

   

    def find_winning_move(
        self,
        board: np.ndarray,
        piece: int,
        available_cols: list[int]
    ) -> int | None:
        
        for col in available_cols:
            temp_board = board.copy()
            row = self.get_next_open_row(temp_board, col)

            if row is not None:
                temp_board[row, col] = piece

                if self.is_winning_board(temp_board, piece):
                    return col

        return None

    def choose_strategic_column(self, available_cols: list[int]) -> int:
       

        preferred_order = [3, 2, 4, 1, 5, 0, 6]

        for col in preferred_order:
            if col in available_cols:
                return col

        return available_cols[0]

   

    def train_and_save_full_q_values(self) -> None:
        

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
        

        board = self.create_empty_board()
        episode = []

        current_piece = -1  

        while True:
            available_cols = self.get_available_columns(board)

            if len(available_cols) == 0:
                reward = self.get_reward(winner=0, agent_piece=agent_piece)
                return episode, reward

            if current_piece == agent_piece:
                
                state_key = self.board_to_key(board)
                action = self.choose_epsilon_greedy_action(board, epsilon)
                episode.append((state_key, action))
            else:
                
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
        

        visited = set()

        for state_action in episode:
            if state_action not in visited:
                visited.add(state_action)

                old_count = self.returns_count.get(state_action, 0)
                old_value = self.Q.get(state_action, 0.0)

                new_count = old_count + 1

                
                new_value = old_value + (reward - old_value) / new_count

                self.returns_count[state_action] = new_count
                self.Q[state_action] = new_value

    def get_reward(self, winner: int, agent_piece: int) -> float:
        
        if winner == agent_piece:
            return 1.0
        elif winner == 0:
            return 0.0
        else:
            return -1.0

    

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

    

    def get_available_columns(self, board: np.ndarray) -> list[int]:
        
        return [c for c in range(7) if board[0, c] == 0]

    def get_current_player(self, board: np.ndarray) -> int:
        
        red_count = np.sum(board == -1)
        yellow_count = np.sum(board == 1)

        if red_count == yellow_count:
            return -1
        else:
            return 1

    def get_next_open_row(self, board: np.ndarray, col: int) -> int | None:
        
        for row in range(5, -1, -1):
            if board[row, col] == 0:
                return row

        return None

    def apply_action(self, board: np.ndarray, col: int, piece: int) -> np.ndarray:
        
        new_board = board.copy()
        row = self.get_next_open_row(new_board, col)

        if row is not None:
            new_board[row, col] = piece

        return new_board

    def is_winning_board(self, board: np.ndarray, piece: int) -> bool:
        

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
        
        if self.is_winning_board(board, -1):
            return -1

        if self.is_winning_board(board, 1):
            return 1

        return 0

    def is_draw(self, board: np.ndarray) -> bool:
        
        return len(self.get_available_columns(board)) == 0

    def create_empty_board(self) -> np.ndarray:
        
        return np.zeros((6, 7), dtype=int)

    def board_to_key(self, board: np.ndarray) -> tuple:
        
        return tuple(board.flatten())

    # ============================================================
    # Persistencia de la tabla Q
    # ============================================================

    def save_q_values(self) -> None:
        

        data = {
            "Q": self.Q,
            "returns_count": self.returns_count
        }

        with open(self.q_values_path, "wb") as file:
            pickle.dump(data, file)

    def load_q_values(self) -> None:
        

        with open(self.q_values_path, "rb") as file:
            data = pickle.load(file)

        self.Q = data.get("Q", {})
        self.returns_count = data.get("returns_count", {})