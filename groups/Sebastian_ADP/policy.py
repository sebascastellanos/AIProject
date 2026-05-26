import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState
from typing import override
from collections import defaultdict


def _board_key(norm_board: np.ndarray) -> tuple:
    return tuple(norm_board.flatten().tolist())


def _current_player(board: np.ndarray) -> int:
    
    n_minus = int(np.sum(board == -1))
    n_plus = int(np.sum(board == 1))
    return -1 if n_minus == n_plus else 1


def _would_win(board: np.ndarray, col: int, player: int) -> bool:
    
    state = ConnectState(board, player)
    if not state.is_applicable(col):
        return False
    return state.transition(col).get_winner() == player


class SebastianADP(Policy):
    

    def __init__(self, n_trials: int = 500, gamma: float = 0.95, vi_iters: int = 30):
        self.n_trials = n_trials
        self.gamma = gamma
        self.vi_iters = vi_iters

        
        self._outcomes: dict = defaultdict(list)
       
        self.P_hat: dict = {}  
        self.R_hat: dict = {}   
        
        self.V: dict = {}

    @override
    def mount(self) -> None:
        self._outcomes.clear()
        self.P_hat.clear()
        self.R_hat.clear()
        self.V.clear()
        self.episode_rewards: list[float] = []  
        rng = np.random.default_rng(42)
        for _ in range(self.n_trials):
            r = self._run_trial(rng, explore=True)
            self.episode_rewards.append(r)       
        self._build_model()
        self._value_iteration()

    

    def _smart_action(self, board: np.ndarray, our: int, avail: list,
                      rng: np.random.Generator, explore: bool) -> int:
        opp = -our
        for col in avail:
            if _would_win(board, col, our):
                return col
        for col in avail:
            if _would_win(board, col, opp):
                return col
        if explore or not self.V:
            return int(rng.choice(avail))
        norm = board * our
        key = _board_key(norm)
        return max(avail, key=lambda c: self._get_q(key, c, norm))

    def _run_trial(self, rng: np.random.Generator, explore: bool = True) -> float:
        
        our = int(rng.choice([-1, 1]))
        state = ConnectState()

        
        if state.player != our:
            avail = state.get_free_cols()
            state = state.transition(int(rng.choice(avail)))
            if state.is_final():
                return 0.0

        while not state.is_final():
            # --- OUR TURN ---
            board = state.board
            norm = board * our                      
            key = _board_key(norm)
            avail = state.get_free_cols()
            action = self._smart_action(board, our, avail, rng, explore)

            state = state.transition(action)

            if state.is_final():
                w = state.get_winner()
                r = 1.0 if w == our else (0.0 if w == 0 else -1.0)
                self._outcomes[(key, action)].append((None, r))
                return r

            # --- OPPONENT TURN (random) ---
            opp_avail = state.get_free_cols()
            state = state.transition(int(rng.choice(opp_avail)))

            if state.is_final():
                w = state.get_winner()
                r = 1.0 if w == our else (0.0 if w == 0 else -1.0)
                self._outcomes[(key, action)].append((None, r))
                return r
            else:
                next_key = _board_key(state.board * our)
                self._outcomes[(key, action)].append((next_key, 0.0))
        return 0.0

    

    def _build_model(self) -> None:
        
        for sa, outcomes in self._outcomes.items():
            n = len(outcomes)
            cnt: dict = defaultdict(int)
            total_r = 0.0
            for nk, r in outcomes:
                cnt[nk] += 1
                total_r += r
            self.P_hat[sa] = {k: v / n for k, v in cnt.items()}
            self.R_hat[sa] = total_r / n

    

    def _value_iteration(self) -> None:
        
        state_actions: dict = defaultdict(list)
        for (key, action), dist in self.P_hat.items():
            state_actions[key].append(action)
            if key not in self.V:
                self.V[key] = 0.0
            for nk in dist:
                if nk is not None and nk not in self.V:
                    self.V[nk] = 0.0

        for _ in range(self.vi_iters):
            for key, actions in state_actions.items():
                best_q = -float('inf')
                for action in actions:
                    sa = (key, action)
                    # Q(s,a) = R̂(s,a) + γ · Σ_{s'} P̂(s'|s,a)·V(s')
                    exp_v = sum(
                        p * (self.V.get(nk, 0.0) if nk is not None else 0.0)
                        for nk, p in self.P_hat[sa].items()
                    )
                    q = self.R_hat[sa] + self.gamma * exp_v
                    if q > best_q:
                        best_q = q
                if best_q > -float('inf'):
                    self.V[key] = best_q

    

    def _heuristic(self, norm_board: np.ndarray, col: int) -> float:
        score = -abs(col - 3) * 0.04          
        for r in range(6):
            for c in range(7):
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    window = []
                    for i in range(4):
                        nr, nc = r + dr * i, c + dc * i
                        if 0 <= nr < 6 and 0 <= nc < 7:
                            window.append(norm_board[nr, nc])
                    if len(window) == 4:
                        mine = window.count(1)
                        opp = window.count(-1)
                        if opp == 0 and mine > 0:
                            score += (0.02, 0.10, 0.60)[mine - 1]
                        elif mine == 0 and opp > 0:
                            score -= (0.02, 0.15, 0.80)[opp - 1]
        return score

    def _get_q(self, key: tuple, col: int, norm_board: np.ndarray) -> float:
        sa = (key, col)
        if sa in self.P_hat:
            exp_v = sum(
                p * (self.V.get(nk, 0.0) if nk is not None else 0.0)
                for nk, p in self.P_hat[sa].items()
            )
            return self.R_hat[sa] + self.gamma * exp_v
        return self._heuristic(norm_board, col)

    

    def _is_two_ply_safe(self, s: np.ndarray, col: int, our: int, opp: int) -> bool:
        state1 = ConnectState(s, our)
        if not state1.is_applicable(col):
            return False
        next1 = state1.transition(col)
        if next1.is_final():
            return True  
        for opp_col in next1.get_free_cols():
            if _would_win(next1.board, opp_col, opp):
                return False
        return True

    @override
    def act(self, s: np.ndarray) -> int:
        our = _current_player(s)
        opp = -our
        available = [c for c in range(7) if s[0, c] == 0]
        norm_s = s * our
        key = _board_key(norm_s)

        for col in available:
            if _would_win(s, col, our):
                return col

        opp_now = [col for col in available if _would_win(s, col, opp)]
        if len(opp_now) == 1:
            return opp_now[0]
        if len(opp_now) >= 2:
            return max(opp_now, key=lambda c: self._get_q(key, c, norm_s))

        safe = [col for col in available if self._is_two_ply_safe(
            s, col, our, opp)]

        if safe:
            best_q, best_col = -float('inf'), safe[len(safe) // 2]
            for col in safe:
                q = self._get_q(key, col, norm_s)
                if q > best_q:
                    best_q, best_col = q, col
            return best_col

        def n_opp_threats_after(col: int) -> int:
            state1 = ConnectState(s, our)
            if not state1.is_applicable(col):
                return 99
            next1 = state1.transition(col)
            if next1.is_final():
                return 0
            return sum(
                1 for c2 in next1.get_free_cols()
                if _would_win(next1.board, c2, opp)
            )

        best_col = min(
            available,
            key=lambda c: (n_opp_threats_after(
                c), -self._get_q(key, c, norm_s))
        )
        return best_col

