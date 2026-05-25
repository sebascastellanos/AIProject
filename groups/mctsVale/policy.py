import numpy as np
import math
import time
from connect4.policy import Policy
from typing import override

# ==============================================================================
# MCTS + UCB1 + RAVE + Evaluación de tablero para Connect-4
# Autora: Vale
#
# Algoritmo: Monte Carlo Tree Search con UCB1, RAVE y función de evaluación
#
# Mejoras sobre MCTS básico:
#   1. Bitboard        → tablero como enteros de 64 bits, ~10x más rápido
#   2. RAVE            → Rapid Action Value Estimation
#   3. Función eval    → evalúa el tablero con heurística en vez de rollout puro
#   4. Warm-up offline → mount() precalienta el árbol durante N segundos
#   5. Fork detection  → detecta y crea/bloquea dobles amenazas
#   6. Orden columnas  → centro primero
#
# Parámetros configurables:
#   n_simulations : int   → simulaciones online por movimiento
#   warmup_time   : float → segundos de precalentamiento en mount()
#   c             : float → constante UCB1
#   k_rave        : float → peso RAVE
#   use_heuristic : bool  → usa función de evaluación (True) o rollout puro (False)
# ==============================================================================

ROWS = 6
COLS = 7
COL_ORDER = [3, 2, 4, 1, 5, 0, 6]
_COL_SHIFT = [c * (ROWS + 1) for c in range(COLS)]

_SCORE_TABLE  = {1: 1, 2: 5, 3: 50}
_THREAT_TABLE = {1: -1, 2: -8, 3: -100}


def _build_win_masks() -> list:
    masks = []
    def _bit(r, c):
        return 1 << (_COL_SHIFT[c] + r)
    for r in range(ROWS):
        for c in range(COLS - 3):
            masks.append(_bit(r,c)|_bit(r,c+1)|_bit(r,c+2)|_bit(r,c+3))
    for r in range(ROWS - 3):
        for c in range(COLS):
            masks.append(_bit(r,c)|_bit(r+1,c)|_bit(r+2,c)|_bit(r+3,c))
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            masks.append(_bit(r,c)|_bit(r+1,c+1)|_bit(r+2,c+2)|_bit(r+3,c+3))
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            masks.append(_bit(r,c)|_bit(r-1,c+1)|_bit(r-2,c+2)|_bit(r-3,c+3))
    return masks

WIN_MASKS = _build_win_masks()


def _build_windows() -> list:
    windows = []
    for r in range(ROWS):
        for c in range(COLS - 3):
            windows.append([(r, c+i) for i in range(4)])
    for r in range(ROWS - 3):
        for c in range(COLS):
            windows.append([(r+i, c) for i in range(4)])
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            windows.append([(r+i, c+i) for i in range(4)])
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            windows.append([(r-i, c+i) for i in range(4)])
    return windows

WINDOWS = _build_windows()


def _evaluate_board(board: np.ndarray, player: int) -> float:
    """
    Evalúa el tablero desde la perspectiva de player.
    Analiza cada ventana de 4 celdas y suma puntos por fichas propias
    y resta por fichas del rival. Retorna valor entre -1 y 1.
    """
    opp   = -player
    score = 0.0
    for window in WINDOWS:
        cells  = [board[r][c] for r, c in window]
        mine   = cells.count(player)
        theirs = cells.count(opp)
        if theirs == 0 and mine > 0:
            score += _SCORE_TABLE.get(mine, 0)
        elif mine == 0 and theirs > 0:
            score += _THREAT_TABLE.get(theirs, 0)
    max_score = 4 * len(WINDOWS)
    return max(min(score / max_score, 1.0), -1.0)


class BitBoard:
    """
    Tablero como dos enteros de 64 bits (uno por jugador).
    Operaciones bitwise ~10x más rápidas que numpy.
    """
    __slots__ = ("bb", "heights", "player", "n_moves")

    def __init__(self):
        self.bb      = [0, 0]
        self.heights = list(_COL_SHIFT)
        self.player  = -1
        self.n_moves = 0

    def copy(self) -> "BitBoard":
        b = BitBoard.__new__(BitBoard)
        b.bb      = self.bb[:]
        b.heights = self.heights[:]
        b.player  = self.player
        b.n_moves = self.n_moves
        return b

    def is_valid_col(self, col: int) -> bool:
        top_bit = _COL_SHIFT[col] + ROWS - 1
        return not (((self.bb[0] | self.bb[1]) >> top_bit) & 1)

    def drop(self, col: int) -> None:
        idx = 0 if self.player == -1 else 1
        self.bb[idx] |= (1 << self.heights[col])
        self.heights[col] += 1
        self.player  = -self.player
        self.n_moves += 1

    def is_winner(self, player: int) -> bool:
        idx = 0 if player == -1 else 1
        bb  = self.bb[idx]
        return any((bb & m) == m for m in WIN_MASKS)

    def is_draw(self) -> bool:
        return self.n_moves == ROWS * COLS

    def is_terminal(self) -> bool:
        return self.is_winner(-1) or self.is_winner(1) or self.is_draw()

    def get_winner(self) -> int:
        if self.is_winner(-1): return -1
        if self.is_winner(1):  return  1
        return 0

    def free_cols(self) -> list:
        return [c for c in COL_ORDER if self.is_valid_col(c)]

    def to_numpy(self) -> np.ndarray:
        board = np.zeros((ROWS, COLS), dtype=int)
        for c in range(COLS):
            for r in range(ROWS):
                bit = _COL_SHIFT[c] + r
                if (self.bb[0] >> bit) & 1:
                    board[ROWS - 1 - r, c] = -1
                elif (self.bb[1] >> bit) & 1:
                    board[ROWS - 1 - r, c] = 1
        return board

    @staticmethod
    def from_numpy(board: np.ndarray) -> "BitBoard":
        b = BitBoard()
        n_red    = int(np.sum(board == -1))
        n_yellow = int(np.sum(board == 1))
        b.player  = -1 if n_red == n_yellow else 1
        b.n_moves = n_red + n_yellow
        for c in range(COLS):
            h = 0
            for r in range(ROWS - 1, -1, -1):
                p = board[r, c]
                if p != 0:
                    idx = 0 if p == -1 else 1
                    b.bb[idx] |= (1 << (_COL_SHIFT[c] + h))
                    h += 1
            b.heights[c] = _COL_SHIFT[c] + h
        return b


def _would_win_bb(bb: BitBoard, col: int, player: int) -> bool:
    if not bb.is_valid_col(col):
        return False
    tmp = bb.copy()
    tmp.player = player
    tmp.drop(col)
    return tmp.is_winner(player)


def _count_threats(bb: BitBoard, player: int) -> int:
    return sum(1 for c in range(COLS)
               if bb.is_valid_col(c) and _would_win_bb(bb, c, player))


class _Node:
    """
    Nodo MCTS con UCB1 + RAVE.
    RAVE acelera la convergencia usando estadísticas globales por acción.
    """
    __slots__ = ("bb","parent","action","children","visits","wins",
                 "_untried","_terminal","rave_wins","rave_visits")

    def __init__(self, bb: BitBoard, parent=None, action=None):
        self.bb        = bb
        self.parent    = parent
        self.action    = action
        self.children  : list["_Node"] = []
        self.visits    = 0
        self.wins      = 0.0
        self._terminal = bb.is_terminal()
        self._untried  = bb.free_cols() if not self._terminal else []
        self.rave_wins   : dict = {}
        self.rave_visits : dict = {}

    def ucb1_rave(self, c: float, k_rave: float) -> float:
        """
        UCB1 + RAVE:
          score = (1-β)·Q_mcts + β·Q_rave + c·sqrt(ln(N_padre)/N)
          β = sqrt(k / (3N + k))
        """
        if self.visits == 0:
            return float("inf")
        q_mcts  = self.wins / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        rv = self.rave_visits.get(self.action, 0)
        if rv > 0:
            q_rave     = self.rave_wins.get(self.action, 0.0) / rv
            beta       = math.sqrt(k_rave / (3 * self.visits + k_rave))
            q_combined = (1 - beta) * q_mcts + beta * q_rave
        else:
            q_combined = q_mcts
        return q_combined + explore

    def is_fully_expanded(self) -> bool:
        return len(self._untried) == 0

    def best_child(self, c: float, k_rave: float) -> "_Node":
        return max(self.children, key=lambda ch: ch.ucb1_rave(c, k_rave))

    def expand(self, rng: np.random.Generator) -> "_Node":
        idx    = int(rng.integers(len(self._untried)))
        action = self._untried.pop(idx)
        new_bb = self.bb.copy()
        new_bb.drop(action)
        child  = _Node(new_bb, parent=self, action=action)
        self.children.append(child)
        return child


class MCTSVale(Policy):
    """
    Agente Connect-4: MCTS + UCB1 + RAVE + Evaluación de tablero + Warm-up.

    Parámetros
    ----------
    n_simulations : int   → simulaciones online por movimiento [50, 100, 200, 500]
    warmup_time   : float → segundos de precalentamiento en mount() [1, 2, 4, 8]
    c             : float → constante UCB1, sqrt(2) es el óptimo teórico
    k_rave        : float → balance RAVE, valores típicos 100-500
    use_heuristic : bool  → rollout con evaluación (True) o aleatorio (False)
    """

    def __init__(
        self,
        n_simulations: int = 300,
        warmup_time: float = 4.0,
        c: float = math.sqrt(2),
        k_rave: float = 300.0,
        use_heuristic: bool = True,
    ):
        self.n_simulations = n_simulations
        self.warmup_time   = warmup_time
        self.c             = c
        self.k_rave        = k_rave
        self.use_heuristic = use_heuristic
        self._rng          = np.random.default_rng(42)
        self._root         : _Node | None = None

    @override
    def mount(self) -> None:
        """
        Precalienta el árbol de apertura durante warmup_time segundos.
        Equivalente al entrenamiento offline de ADP pero con MCTS.
        """
        self._rng = np.random.default_rng(42)
        root      = _Node(BitBoard())
        deadline  = time.time() + self.warmup_time
        while time.time() < deadline:
            node = self._select(root)
            if not node._terminal:
                node = node.expand(self._rng)
            reward, actions = self._simulate(node.bb, -1)
            self._backpropagate(node, reward, actions)
        self._root = root

    def _select(self, node: _Node) -> _Node:
        while not node._terminal:
            if not node.is_fully_expanded():
                return node
            node = node.best_child(self.c, self.k_rave)
        return node

    def _simulate(self, bb: BitBoard, root_player: int) -> tuple:
        """
        Rollout corto con heurística + evaluación del tablero.
        Más informativo que el rollout aleatorio puro.
        """
        MAX_DEPTH = 12
        current   = bb.copy()
        actions   = []
        depth     = 0

        while not current.is_terminal() and depth < MAX_DEPTH:
            free = current.free_cols()
            if self.use_heuristic:
                action = None
                for col in free:
                    if _would_win_bb(current, col, current.player):
                        action = col
                        break
                if action is None:
                    opp = -current.player
                    for col in free:
                        if _would_win_bb(current, col, opp):
                            action = col
                            break
                if action is None:
                    action = int(self._rng.choice(free))
            else:
                action = int(self._rng.choice(free))
            actions.append(action)
            current.drop(action)
            depth += 1

        if current.is_terminal():
            winner = current.get_winner()
            reward = 1.0 if winner == root_player else (-1.0 if winner != 0 else 0.0)
        else:
            board_np = current.to_numpy()
            reward   = _evaluate_board(board_np, root_player)

        return reward, actions

    def _backpropagate(self, node: _Node, reward: float, actions: list) -> None:
        current = node
        while current is not None:
            current.visits += 1
            current.wins   += reward
            if current.parent is not None:
                for a in actions:
                    current.parent.rave_visits[a] = current.parent.rave_visits.get(a, 0) + 1
                    current.parent.rave_wins[a]   = current.parent.rave_wins.get(a, 0.0) + reward
            reward  = -reward
            current = current.parent

    def _find_fork(self, bb: BitBoard, player: int) -> int | None:
        for col in bb.free_cols():
            tmp = bb.copy()
            tmp.player = player
            tmp.drop(col)
            if _count_threats(tmp, player) >= 2:
                return col
        return None

    def _run_mcts(self, bb: BitBoard) -> int:
        root_player = bb.player
        root        = _Node(bb)
        if self._root is not None and self._root.rave_visits:
            root.rave_visits = dict(self._root.rave_visits)
            root.rave_wins   = dict(self._root.rave_wins)
        for _ in range(self.n_simulations):
            node = self._select(root)
            if not node._terminal:
                node = node.expand(self._rng)
            reward, actions = self._simulate(node.bb, root_player)
            self._backpropagate(node, reward, actions)
        if not root.children:
            return bb.free_cols()[0]
        best = max(root.children, key=lambda ch: ch.visits)
        return best.action

    @override
    def act(self, s: np.ndarray) -> int:
        """
        Prioridades:
        1. Ganar inmediatamente.
        2. Bloquear victoria inmediata del oponente.
        3. Crear fork propio.
        4. Bloquear fork del oponente.
        5. MCTS + UCB1 + RAVE + evaluación de tablero.
        """
        bb  = BitBoard.from_numpy(s)
        our = bb.player
        opp = -our

        for col in bb.free_cols():
            if _would_win_bb(bb, col, our):
                return col

        for col in bb.free_cols():
            if _would_win_bb(bb, col, opp):
                return col

        fork = self._find_fork(bb, our)
        if fork is not None:
            return fork

        opp_fork = self._find_fork(bb, opp)
        if opp_fork is not None:
            return opp_fork

        return self._run_mcts(bb)