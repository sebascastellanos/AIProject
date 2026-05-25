import numpy as np
import math
from connect4.policy import Policy
from typing import override



ROWS = 6
COLS = 7


COL_ORDER = [3, 2, 4, 1, 5, 0, 6]



_COL_SHIFT = [c * (ROWS + 1) for c in range(COLS)]

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


class BitBoard:
   

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
        """Coloca ficha del jugador actual en la columna."""
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
        """Columnas libres ordenadas centro primero."""
        return [c for c in COL_ORDER if self.is_valid_col(c)]

    @staticmethod
    def from_numpy(board: np.ndarray) -> "BitBoard":
        """Convierte un tablero numpy al formato bitboard."""
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
    """True si colocar en col gana inmediatamente para player."""
    if not bb.is_valid_col(col):
        return False
    tmp = bb.copy()
    tmp.player = player
    tmp.drop(col)
    return tmp.is_winner(player)


def _count_threats(bb: BitBoard, player: int) -> int:
    """Cuenta cuántas columnas dan victoria inmediata a player."""
    return sum(1 for c in range(COLS)
               if bb.is_valid_col(c) and _would_win_bb(bb, c, player))



class _Node:
    

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
        
        if self.visits == 0:
            return float("inf")
        q_mcts  = self.wins / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        rv = self.rave_visits.get(self.action, 0)
        if rv > 0:
            q_rave = self.rave_wins.get(self.action, 0.0) / rv
            beta   = math.sqrt(k_rave / (3 * self.visits + k_rave))
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
    

    def __init__(
        self,
        n_simulations: int = 800,
        c: float = math.sqrt(2),
        k_rave: float = 300.0,
        use_heuristic: bool = True,
    ):
        self.n_simulations = n_simulations
        self.c             = c
        self.k_rave        = k_rave
        self.use_heuristic = use_heuristic
        self._rng          = np.random.default_rng(42)

    @override
    def mount(self) -> None:
        """MCTS es online → mount() solo reinicia el RNG."""
        self._rng = np.random.default_rng(42)

    def _select(self, node: _Node) -> _Node:
        while not node._terminal:
            if not node.is_fully_expanded():
                return node
            node = node.best_child(self.c, self.k_rave)
        return node

    def _rollout(self, bb: BitBoard, root_player: int) -> tuple:
        
        current = bb.copy()
        actions = []
        while not current.is_terminal():
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
        winner = current.get_winner()
        reward = 1.0 if winner == root_player else (-1.0 if winner != 0 else 0.0)
        return reward, actions

    def _backpropagate(self, node: _Node, reward: float, rollout_actions: list) -> None:
        
        current = node
        while current is not None:
            current.visits += 1
            current.wins   += reward
            if current.parent is not None:
                for a in rollout_actions:
                    current.parent.rave_visits[a] = current.parent.rave_visits.get(a, 0) + 1
                    current.parent.rave_wins[a]   = current.parent.rave_wins.get(a, 0.0) + reward
            reward  = -reward
            current = current.parent

    def _run_mcts(self, bb: BitBoard) -> int:
        """Ejecuta n_simulations iteraciones y retorna la columna más visitada."""
        root_player = bb.player
        root        = _Node(bb)
        for _ in range(self.n_simulations):
            node = self._select(root)
            if not node._terminal:
                node = node.expand(self._rng)
            reward, actions = self._rollout(node.bb, root_player)
            self._backpropagate(node, reward, actions)
        best = max(root.children, key=lambda ch: ch.visits)
        return best.action

    def _find_fork(self, bb: BitBoard, player: int) -> int | None:
        
        for col in bb.free_cols():
            tmp = bb.copy()
            tmp.player = player
            tmp.drop(col)
            if _count_threats(tmp, player) >= 2:
                return col
        return None

    @override
    def act(self, s: np.ndarray) -> int:
        
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