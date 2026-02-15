import collections
import heapq
from nurikabe_model import NurikabeModel
from typing import Optional, List, Tuple, Set


class IslandSteinerPropagator:
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    MAX_EXACT_TERMINALS = 8  # Threshold for switching between exact DP and best-effort heuristic

    def __init__(self, model: NurikabeModel, island_id: int):
        self.model = model
        self.island_id = island_id
        self.island_bit = 1 << (island_id - 1)
        self.target_size = model.islands[island_id - 1].clue
        self.inf = (model.rows * model.cols) + 10
        
        self._owners = None
        self._is_core = None
        self._costs = None

    def run(self, mandatory_groups: List[Set[Tuple[int, int]]] = None) -> Tuple[List[Tuple[int, int]], bool]:
        if mandatory_groups is None:
            mandatory_groups = []
        self._prepare_context()
        core_set = self._get_core_set()
        core_components = self.get_connected_components(core_set)

        valid_groups = self._filter_groups(mandatory_groups)
        if valid_groups is None:
            return [], True

        budget = self.target_size - len(core_set)
        if budget < 0: return [], True

        reductions = set()
        
        # 1. Always try to prune based on Core Steiner Tree if possible (strong pruning)
        num_cores = len(core_components)
        if num_cores > 0:
            if num_cores == 1:
                r_core, contra = self._solve_single_terminal(core_components[0], budget)
            elif num_cores <= self.MAX_EXACT_TERMINALS:
                r_core, contra = self._solve_exact(core_components, budget)
            else:
                # Should we do something here? best_effort on cores is same as _solve_best_effort below
                r_core, contra = [], False
            
            if contra: return [], True
            reductions.update(r_core)

        # 2. Now consider mandatory groups
        if not valid_groups:
            return list(reductions), False

        terminals = core_components + valid_groups
        k = len(terminals)
        
        if k <= self.MAX_EXACT_TERMINALS:
            r_all, contra = self._solve_exact(terminals, budget)
        else:
            r_all, contra = self._solve_best_effort(terminals, budget)
            
        if contra: return [], True
        reductions.update(r_all)

        return list(reductions), False

    def _prepare_context(self):
        rows, cols = self.model.rows, self.model.cols
        self._owners = [[0] * cols for _ in range(rows)]
        self._is_core = [[False] * cols for _ in range(rows)]
        self._costs = [[1] * cols for _ in range(rows)]

        clue_pos = self.model.islands[self.island_id - 1].pos
        
        for r in range(rows):
            for c in range(cols):
                cell = self.model.cells[r][c]
                o_val = cell.owners.bits
                if cell.is_sea:
                    o_val = 0
                self._owners[r][c] = o_val
                
                if (r, c) == clue_pos or (cell.is_land and cell.owners.get_singleton_id() == self.island_id):
                    self._is_core[r][c] = True
                    self._costs[r][c] = 0

    def _get_core_set(self) -> Set[Tuple[int, int]]:
        return {(r, c) for r in range(self.model.rows) 
                for c in range(self.model.cols) if self._is_core[r][c]}

    def _filter_groups(self, groups: List[Set[Tuple[int, int]]]) -> Optional[List[Set[Tuple[int, int]]]]:
        valid_groups = []
        for g in groups:
            filtered = {(r, c) for (r, c) in g if self._is_traversable(r, c)}
            if not filtered: return None
            valid_groups.append(filtered)
        return valid_groups

    def _is_traversable(self, r: int, c: int) -> bool:
        return 0 <= r < self.model.rows and 0 <= c < self.model.cols and \
               (self._owners[r][c] & self.island_bit) != 0

    def _solve_single_terminal(self, terminal: Set[Tuple[int, int]], budget: int) -> Tuple[List[Tuple[int, int]], bool]:
        dists = self.compute_dijkstra(terminal)
        reductions = []
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if not self._is_traversable(r, c): continue
                if self._is_core[r][c]: continue
                if dists[r][c] > budget:
                    reductions.append((r, c))
        return reductions, False

    def _solve_exact(self, terminals: List[Set[Tuple[int, int]]], budget: int) -> Tuple[List[Tuple[int, int]], bool]:
        rows, cols = self.model.rows, self.model.cols
        k = len(terminals)
        full_mask = (1 << k) - 1
        dp = [[[self.inf] * cols for _ in range(rows)] for _ in range(1 << k)]
        active_cells = [set() for _ in range(1 << k)]

        for i, term_set in enumerate(terminals):
            m = 1 << i
            for r, c in term_set:
                dp[m][r][c] = self._costs[r][c]
                active_cells[m].add((r, c))
            self._dijkstra_closure(dp[m], active_cells[m])

        for mask in range(1, 1 << k):
            if (mask & (mask - 1)) == 0: continue

            sub = (mask - 1) & mask
            while sub > (mask ^ sub):
                other = mask ^ sub
                for r in range(rows):
                    for c in range(cols):
                        if dp[sub][r][c] == self.inf or dp[other][r][c] == self.inf:
                            continue
                        v = dp[sub][r][c] + dp[other][r][c] - self._costs[r][c]
                        if v < dp[mask][r][c]:
                            dp[mask][r][c] = v
                            active_cells[mask].add((r, c))
                sub = (sub - 1) & mask

            if active_cells[mask]:
                self._dijkstra_closure(dp[mask], active_cells[mask])

        reductions = []
        min_v = min((dp[full_mask][r][c] for r in range(rows) for c in range(cols) if self._is_traversable(r, c)), default=self.inf)
        
        if min_v > budget:
            return [], True
            
        for r in range(rows):
            for c in range(cols):
                if not self._is_traversable(r, c): continue
                if self._is_core[r][c]: continue
                if dp[full_mask][r][c] > budget:
                    reductions.append((r, c))
        
        return reductions, False

    def _dijkstra_closure(self, mask_dp: List[List[int]], support: Set[Tuple[int, int]]):
        pq = [(mask_dp[r][c], r, c) for r, c in support]
        heapq.heapify(pq)

        while pq:
            d, r, c = heapq.heappop(pq)
            if d != mask_dp[r][c]: continue
            for dr, dc in self.DIRECTIONS:
                nr, nc = r + dr, c + dc
                if self._is_traversable(nr, nc):
                    nd = d + self._costs[nr][nc]
                    if nd < mask_dp[nr][nc]:
                        mask_dp[nr][nc] = nd
                        support.add((nr, nc))
                        heapq.heappush(pq, (nd, nr, nc))

    def _solve_best_effort(self, terminals, budget) -> Tuple[List[Tuple[int, int]], bool]:
        # terminals[0] is at least one core component. 
        # Actually, let's use all core components for the distance base.
        core_indices = [i for i, t in enumerate(terminals) if any(self._is_core[r][c] for r, c in t)]
        if not core_indices:
            # Should not happen as core is always added first
            return [], False
            
        core_combined = set().union(*(terminals[i] for i in core_indices))
        core_dists = self.compute_dijkstra(core_combined)
        
        # 1. Contradiction check: can each terminal reach the core?
        max_reach_dist = 0
        for i in range(len(terminals)):
            if i in core_indices: continue
            best = min((core_dists[r][c] for r, c in terminals[i]), default=self.inf)
            if best == self.inf: return [], True
            max_reach_dist = max(max_reach_dist, best)
        
        # This is a very loose lower bound for Steiner Tree, but safe.
        if max_reach_dist > budget:
            return [], True
            
        # 2. Pruning
        # A cell is only useful if it's within budget of the core 
        # AND it's not too far from SOME terminal.
        all_terminals_set = set().union(*terminals)
        combined_dists = self.compute_dijkstra(all_terminals_set)
        
        reductions = []
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if not self._is_traversable(r, c): continue
                if self._is_core[r][c]: continue
                
                # If it's too far from the core, it's useless
                if core_dists[r][c] > budget:
                    reductions.append((r, c))
                    continue
                    
                # If it's too far from ANY terminal, it's also useless
                if combined_dists[r][c] > budget:
                    reductions.append((r, c))
                    
        return list(set(reductions)), False

    def get_connected_components(self, cells_set: Set[Tuple[int, int]]) -> List[Set[Tuple[int, int]]]:
        visited = set()
        components = []
        for r, c in cells_set:
            if (r, c) in visited: continue
            comp, q = set(), collections.deque([(r, c)])
            visited.add((r, c))
            while q:
                cr, cc = q.popleft()
                comp.add((cr, cc))
                for dr, dc in self.DIRECTIONS:
                    nr, nc = cr + dr, cc + dc
                    if (nr, nc) in cells_set and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append((nr, nc))
            components.append(comp)
        return components

    def compute_dijkstra(self, sources: Set[Tuple[int, int]]) -> List[List[int]]:
        rows, cols = self.model.rows, self.model.cols
        dists = [[self.inf] * cols for _ in range(rows)]
        pq = []
        for r, c in sources:
            if self._is_traversable(r, c):
                dists[r][c] = self._costs[r][c]
                heapq.heappush(pq, (dists[r][c], r, c))
        
        while pq:
            d, r, c = heapq.heappop(pq)
            if d != dists[r][c]: continue
            for dr, dc in self.DIRECTIONS:
                nr, nc = r + dr, c + dc
                if self._is_traversable(nr, nc):
                    nd = d + self._costs[nr][nc]
                    if nd < dists[nr][nc]:
                        dists[nr][nc] = nd
                        heapq.heappush(pq, (nd, nr, nc))
        return dists