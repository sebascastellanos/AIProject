import numpy as np
from connect4.policy import Policy


class FVMCConnect4Agent(Policy):

    def __init__(self):
        self.max_depth = 4

    def mount(self, *args, **kwargs) -> None:
        """
        El autocalificador puede llamar mount(timeout).
        Por eso se aceptan *args y **kwargs.
        """
        self.max_depth = 4

    def act(self, s: np.ndarray) -> int:
        available_cols = self.get_available_columns(s)

        if len(available_cols) == 0:
            return 0

        my_piece = self.get_current_player(s)
        opponent_piece = -my_piece

        # 1. Si puedo ganar inmediatamente, juego ahí.
        winning_move = self.find_winning_move(s, my_piece, available_cols)
        if winning_move is not None:
            return int(winning_move)

        # 2. Si el rival puede ganar inmediatamente, bloqueo.
        blocking_move = self.find_winning_move(s, opponent_piece, available_cols)
        if blocking_move is not None:
            return int(blocking_move)

        # 3. Evito jugadas que le regalen victoria inmediata al rival.
        safe_cols = self.filter_safe_actions(
            s,
            available_cols,
            my_piece,
            opponent_piece
        )

        if len(safe_cols) > 0:
            candidate_cols = safe_cols
        else:
            candidate_cols = available_cols

        # 4. Uso minimax con poda alfa-beta.
        best_col = self.choose_minimax_action(
            board=s,
            available_cols=candidate_cols,
            my_piece=my_piece
        )

        return int(best_col)

    # ============================================================
    # Minimax
    # ============================================================

    def choose_minimax_action(
        self,
        board: np.ndarray,
        available_cols: list,
        my_piece: int
    ) -> int:
        best_score = float("-inf")
        best_col = self.choose_strategic_column(available_cols)

        ordered_cols = [c for c in [3, 2, 4, 1, 5, 0, 6] if c in available_cols]

        for col in ordered_cols:
            new_board = self.apply_action(board, col, my_piece)

            score = self.minimax(
                board=new_board,
                depth=self.max_depth - 1,
                current_piece=-my_piece,
                agent_piece=my_piece,
                alpha=float("-inf"),
                beta=float("inf")
            )

            if score > best_score:
                best_score = score
                best_col = col

        return best_col

    def minimax(
        self,
        board: np.ndarray,
        depth: int,
        current_piece: int,
        agent_piece: int,
        alpha: float,
        beta: float
    ) -> float:
        winner = self.get_winner(board)

        if winner == agent_piece:
            return 1000000 + depth

        if winner == -agent_piece:
            return -1000000 - depth

        if depth == 0 or self.is_draw(board):
            return self.evaluate_board(board, agent_piece)

        available_cols = self.get_available_columns(board)
        ordered_cols = [c for c in [3, 2, 4, 1, 5, 0, 6] if c in available_cols]

        if current_piece == agent_piece:
            value = float("-inf")

            for col in ordered_cols:
                new_board = self.apply_action(board, col, current_piece)

                value = max(
                    value,
                    self.minimax(
                        board=new_board,
                        depth=depth - 1,
                        current_piece=-current_piece,
                        agent_piece=agent_piece,
                        alpha=alpha,
                        beta=beta
                    )
                )

                alpha = max(alpha, value)

                if alpha >= beta:
                    break

            return value

        value = float("inf")

        for col in ordered_cols:
            new_board = self.apply_action(board, col, current_piece)

            value = min(
                value,
                self.minimax(
                    board=new_board,
                    depth=depth - 1,
                    current_piece=-current_piece,
                    agent_piece=agent_piece,
                    alpha=alpha,
                    beta=beta
                )
            )

            beta = min(beta, value)

            if alpha >= beta:
                break

        return value

    # ============================================================
    # Heurística
    # ============================================================

    def evaluate_board(self, board: np.ndarray, piece: int) -> float:
        score = 0.0
        opponent_piece = -piece

        # Control del centro
        center_array = list(board[:, 3])
        center_count = center_array.count(piece)
        score += center_count * 6

        # Horizontal
        for r in range(6):
            row_array = list(board[r, :])
            for c in range(4):
                window = row_array[c:c + 4]
                score += self.evaluate_window(window, piece, opponent_piece)

        # Vertical
        for c in range(7):
            col_array = list(board[:, c])
            for r in range(3):
                window = col_array[r:r + 4]
                score += self.evaluate_window(window, piece, opponent_piece)

        # Diagonal positiva /
        for r in range(3, 6):
            for c in range(4):
                window = [
                    board[r, c],
                    board[r - 1, c + 1],
                    board[r - 2, c + 2],
                    board[r - 3, c + 3]
                ]
                score += self.evaluate_window(window, piece, opponent_piece)

        # Diagonal negativa \
        for r in range(3):
            for c in range(4):
                window = [
                    board[r, c],
                    board[r + 1, c + 1],
                    board[r + 2, c + 2],
                    board[r + 3, c + 3]
                ]
                score += self.evaluate_window(window, piece, opponent_piece)

        return score

    def evaluate_window(
        self,
        window: list,
        piece: int,
        opponent_piece: int
    ) -> float:
        score = 0.0

        piece_count = window.count(piece)
        opponent_count = window.count(opponent_piece)
        empty_count = window.count(0)

        if piece_count == 4:
            score += 100000

        elif piece_count == 3 and empty_count == 1:
            score += 100

        elif piece_count == 2 and empty_count == 2:
            score += 10

        if opponent_count == 3 and empty_count == 1:
            score -= 120

        elif opponent_count == 2 and empty_count == 2:
            score -= 15

        return score

    # ============================================================
    # Reglas tácticas
    # ============================================================

    def find_winning_move(
        self,
        board: np.ndarray,
        piece: int,
        available_cols: list
    ):
        for col in available_cols:
            temp_board = self.apply_action(board, col, piece)

            if self.is_winning_board(temp_board, piece):
                return col

        return None

    def is_safe_action(
        self,
        board: np.ndarray,
        action: int,
        my_piece: int,
        opponent_piece: int
    ) -> bool:
        new_board = self.apply_action(board, action, my_piece)
        opponent_available_cols = self.get_available_columns(new_board)

        opponent_winning_move = self.find_winning_move(
            new_board,
            opponent_piece,
            opponent_available_cols
        )

        return opponent_winning_move is None

    def filter_safe_actions(
        self,
        board: np.ndarray,
        available_cols: list,
        my_piece: int,
        opponent_piece: int
    ) -> list:
        safe_cols = []

        for col in available_cols:
            if self.is_safe_action(board, col, my_piece, opponent_piece):
                safe_cols.append(col)

        return safe_cols

    def choose_strategic_column(self, available_cols: list) -> int:
        preferred_order = [3, 2, 4, 1, 5, 0, 6]

        for col in preferred_order:
            if col in available_cols:
                return col

        return available_cols[0]

    # ============================================================
    # Funciones del tablero
    # ============================================================

    def get_available_columns(self, board: np.ndarray) -> list:
        return [c for c in range(7) if board[0, c] == 0]

    def get_current_player(self, board: np.ndarray) -> int:
        red_count = np.sum(board == -1)
        yellow_count = np.sum(board == 1)

        if red_count == yellow_count:
            return -1

        return 1

    def get_next_open_row(self, board: np.ndarray, col: int):
        for row in range(5, -1, -1):
            if board[row, col] == 0:
                return row

        return None

    def apply_action(
        self,
        board: np.ndarray,
        col: int,
        piece: int
    ) -> np.ndarray:
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