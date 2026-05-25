import numpy as np
import math
import time
from connect4.policy import Policy
from typing import override

# ==============================================================================
# MCTS + UCB1 para Connect-4
# Autora: Vale
#
# Basado en la arquitectura del MCTSAgent ejemplo, con mejoras:
#   1. Bitboard     → tablero como enteros de 64 bits, ~10x más rápido
#   2. Fork detect  → detecta/crea dobles amenazas antes de MCTS
#   3. Warm-up      → mount() precalienta el árbol N segundos
#   4. Centro first → columnas del centro primero
#   5. Rollout real → simula hasta el final (como el ejemplo)
#
# Parámetros configurables:
#   n_simulations : int   → simulaciones por movimiento
#   warmup_time   : float → segundos de precalentamiento en mount()
#   c             : float → constante UCB1
#   rollout_limit : int   → pasos máximos del rollout
#   use_heuristic : bool  → versión con/sin rollout hasta el final
# ==============================================================================

ROWS = 6
COLS = 7
COL_ORDER  = [3, 2, 4, 1, 5, 0, 6]
_COL_SHIFT = [c * (ROWS + 1) for c in range(COLS)]


def _build_win_masks():
    masks = []
    def _bit(r, c): return 1 << (_COL_SHIFT[c] + r)
    for r in range(ROWS):
        for c in range(COLS-3):
            masks.append(_bit(r,c)|_bit(r,c+1)|_bit(r,c+2)|_bit(r,c+3))
    for r in range(ROWS-3):
        for c in range(COLS):
            masks.append(_bit(r,c)|_bit(r+1,c)|_bit(r+2,c)|_bit(r+3,c))
    for r in range(ROWS-3):
        for c in range(COLS-3):
            masks.append(_bit(r,c)|_bit(r+1,c+1)|_bit(r+2,c+2)|_bit(r+3,c+3))
    for r in range(3, ROWS):
        for c in range(COLS-3):
            masks.append(_bit(r,c)|_bit(r-1,c+1)|_bit(r-2,c+2)|_bit(r-3,c+3))
    return masks

WIN_MASKS = _build_win_masks()


# ==============================================================================
# BitBoard
# ==============================================================================

class BitBoard:
    """
    Tablero como dos enteros de 64 bits (uno por jugador).
    Operaciones bitwise ~10x más rápidas que numpy para verificar victorias.
    """
    __slots__ = ("bb", "heights", "player", "n_moves")

    def __init__(self):
        self.bb      = [0, 0]
        self.heights = list(_COL_SHIFT)
        self.player  = -1
        self.n_moves = 0

    def copy(self):
        b = BitBoard.__new__(BitBoard)
        b.bb      = self.bb[:]
        b.heights = self.heights[:]
        b.player  = self.player
        b.n_moves = self.n_moves
        return b

    def is_valid_col(self, col):
        return not (((self.bb[0]|self.bb[1]) >> (_COL_SHIFT[col]+ROWS-1)) & 1)

    def drop(self, col):
        idx = 0 if self.player == -1 else 1
        self.bb[idx] |= (1 << self.heights[col])
        self.heights[col] += 1
        self.player  = -self.player
        self.n_moves += 1

    def is_winner(self, player):
        idx = 0 if player == -1 else 1
        bb  = self.bb[idx]
        return any((bb & m) == m for m in WIN_MASKS)

    def is_draw(self):
        return self.n_moves == ROWS * COLS

    def is_terminal(self):
        return self.is_winner(-1) or self.is_winner(1) or self.is_draw()

    def get_winner(self):
        if self.is_winner(-1): return -1
        if self.is_winner(1):  return  1
        return 0

    def free_cols(self):
        """Columnas disponibles ordenadas centro primero."""
        return [c for c in COL_ORDER if self.is_valid_col(c)]

    @staticmethod
    def from_numpy(board: np.ndarray):
        b = BitBoard()
        n_red    = int(np.sum(board == -1))
        n_yellow = int(np.sum(board == 1))
        b.player  = -1 if n_red == n_yellow else 1
        b.n_moves = n_red + n_yellow
        for c in range(COLS):
            h = 0
            for r in range(ROWS-1, -1, -1):
                p = board[r, c]
                if p != 0:
                    idx = 0 if p == -1 else 1
                    b.bb[idx] |= (1 << (_COL_SHIFT[c]+h))
                    h += 1
            b.heights[c] = _COL_SHIFT[c] + h
        return b


# ==============================================================================
# Funciones auxiliares
# ==============================================================================

def _would_win_bb(bb, col, player):
    if not bb.is_valid_col(col): return False
    tmp = bb.copy()
    tmp.player = player
    tmp.drop(col)
    return tmp.is_winner(player)

def _count_threats(bb, player):
    return sum(1 for c in range(COLS)
               if bb.is_valid_col(c) and _would_win_bb(bb, c, player))


# ==============================================================================
# Nodo MCTS
# ==============================================================================

class _Node:
    """
    Nodo del árbol MCTS con UCB1.

    Estadísticas guardadas desde la perspectiva del jugador ROOT
    (no del jugador en ese nodo). Igual que el ejemplo MCTSAgent.

    UCB1 = total_reward/visits + c * sqrt(ln(N_padre) / visits)
    """
    __slots__ = ("bb","parent","action","children","visits","total_reward","_terminal")

    def __init__(self, bb, parent=None, action=None):
        self.bb           = bb
        self.parent       = parent
        self.action       = action
        self.children     = {}        # col -> _Node
        self.visits       = 0
        self.total_reward = 0.0
        self._terminal    = bb.is_terminal()

    def uct_score(self, c):
        if self.visits == 0:
            return float("inf")
        exploit = self.total_reward / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits + 1) / self.visits)
        return exploit + explore

    def best_uct_child(self, c):
        return max(self.children.values(), key=lambda ch: ch.uct_score(c))

    def is_fully_expanded(self, free_cols):
        return len(self.children) == len(free_cols)

    def untried_cols(self, free_cols):
        return [col for col in free_cols if col not in self.children]


# ==============================================================================
# Agente principal
# ==============================================================================

class MCTSVale(Policy):
    """
    Agente Connect-4: MCTS + UCB1 + Bitboard + Fork detection + Warm-up.

    Arquitectura basada en MCTSAgent (ejemplo del curso) con mejoras:
    - Bitboard para operaciones ~10x más rápidas
    - Detección de fork (doble amenaza) antes de MCTS
    - Warm-up offline en mount() equivalente al de ADP
    - Columnas del centro exploradas primero

    El rollout simula hasta el final del juego (igual que el ejemplo),
    lo que da señales más precisas que una función heurística.

    Parámetros
    ----------
    n_simulations : int
        Simulaciones MCTS por movimiento. Variable principal del análisis.
    warmup_time : float
        Segundos de precalentamiento en mount(). Equivalente al
        tiempo de entrenamiento de ADP.
    c : float
        Constante UCB1. 1.4 es el valor del ejemplo; sqrt(2)≈1.414 es teórico.
    rollout_limit : int
        Pasos máximos del rollout. 100 como en el ejemplo.
    use_heuristic : bool
        True  → rollout guiado (gana/bloquea si puede).
        False → rollout completamente aleatorio.
    """

    def __init__(
        self,
        n_simulations: int = 500,
        warmup_time:   float = 4.0,
        c:             float = 1.4,
        rollout_limit: int   = 100,
        use_heuristic: bool  = True,
    ):
        self.n_simulations = n_simulations
        self.warmup_time   = warmup_time
        self.c             = c
        self.rollout_limit = rollout_limit
        self.use_heuristic = use_heuristic
        self._rng          = np.random.default_rng(42)

    @override
    def mount(self) -> None:
        """
        Precalienta el árbol de apertura durante warmup_time segundos.
        Usa el mismo tiempo que ADP para entrenar offline.
        """
        self._rng  = np.random.default_rng(42)
        root       = _Node(BitBoard())
        deadline   = time.time() + self.warmup_time
        root_player = -1

        while time.time() < deadline:
            node, path = self._select(root)
            if not node._terminal:
                node = self._expand(node)
            reward = self._rollout(node.bb, root_player)
            self._backpropagate(node, reward)

        self._warmup_root = root

    # ------------------------------------------------------------------
    # Fases MCTS
    # ------------------------------------------------------------------

    def _select(self, root):
        node = root
        while True:
            if node._terminal:
                return node, None
            free = node.bb.free_cols()
            if not node.is_fully_expanded(free):
                return node, None
            node = node.best_uct_child(self.c)

    def _expand(self, node):
        free    = node.bb.free_cols()
        untried = node.untried_cols(free)
        col     = int(self._rng.choice(untried))
        new_bb  = node.bb.copy()
        new_bb.drop(col)
        child   = _Node(new_bb, parent=node, action=col)
        node.children[col] = child
        return child

    def _rollout(self, bb, root_player):
        """
        Simula una partida completa desde el estado dado.
        Igual que el ejemplo: juega hasta el final o rollout_limit pasos.

        Con use_heuristic=True: gana si puede, bloquea si puede, si no azar.
        Con use_heuristic=False: completamente aleatorio.

        Retorna:
            1.0  si gana root_player
            0.0  si pierde
            0.5  si empate o límite alcanzado
        """
        current = bb.copy()
        steps   = 0

        while steps < self.rollout_limit:
            # Verificar resultado
            if current.is_winner(root_player):  return 1.0
            if current.is_winner(-root_player): return 0.0
            free = current.free_cols()
            if not free: return 0.5

            if self.use_heuristic:
                # Ganar ahora
                action = None
                for col in free:
                    if _would_win_bb(current, col, current.player):
                        action = col
                        break
                # Bloquear
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

            current.drop(action)
            steps += 1

        return 0.5

    def _backpropagate(self, node, reward):
        """
        Propaga el reward hacia la raíz.
        El reward es desde la perspectiva del jugador ROOT,
        igual que en el ejemplo MCTSAgent.
        """
        current = node
        while current is not None:
            current.visits       += 1
            current.total_reward += reward
            current = current.parent

    # ------------------------------------------------------------------
    # MCTS principal
    # ------------------------------------------------------------------

    def _run_mcts(self, bb):
        root_player = bb.player
        root        = _Node(bb)

        for _ in range(self.n_simulations):
            node, _ = self._select(root)
            if not node._terminal:
                node = self._expand(node)
            reward = self._rollout(node.bb, root_player)
            self._backpropagate(node, reward)

        if not root.children:
            return bb.free_cols()[0]

        # Hijo más visitado (más robusto que mayor win rate)
        best_col    = None
        best_visits = -1
        center      = COLS // 2
        for col, child in root.children.items():
            if child.visits > best_visits:
                best_visits = child.visits
                best_col    = col
            elif child.visits == best_visits:
                # Desempata hacia el centro
                if abs(col - center) < abs(best_col - center):
                    best_col = col

        return best_col

    # ------------------------------------------------------------------
    # Fork detection
    # ------------------------------------------------------------------

    def _find_fork(self, bb, player):
        """Busca movimiento que cree 2+ amenazas simultáneas."""
        for col in bb.free_cols():
            tmp = bb.copy()
            tmp.player = player
            tmp.drop(col)
            if _count_threats(tmp, player) >= 2:
                return col
        return None

    # ------------------------------------------------------------------
    # Decisión final
    # ------------------------------------------------------------------

    @override
    def act(self, s: np.ndarray) -> int:
        """
        Prioridades:
        1. Ganar inmediatamente.
        2. Bloquear victoria inmediata del rival.
        3. Crear fork propio (doble amenaza).
        4. Bloquear fork del rival.
        5. MCTS + UCB1.
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
