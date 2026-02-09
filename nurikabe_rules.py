from typing import Optional, Set, List, Tuple, Callable
from nurikabe_model import NurikabeModel, StepResult, CellState

# Global registry for rules: list of (priority, func, name)
_RULES = []

def solver_rule(priority: int, name: str, message: str = "") -> Callable:
    """Decorator to register a solver rule with a priority and a descriptive name."""
    def decorator(func: Callable) -> Callable:
        func._rule_name = name
        func._rule_message = message
        _RULES.append((priority, func, name))
        return func
    return decorator

class NurikabeSolver:
    RULE_NAMES: List[str] = []

    def __init__(self, model: NurikabeModel) -> None:
        self.model = model

    def step(self) -> Optional[StepResult]:
        # Iterate over rules sorted by priority (lowest number first)
        for _, func, name in sorted(_RULES, key=lambda x: x[0]):
            res = func(self)
            if res:
                if not res.rule:
                    res.rule = name
                
                # If there's a rule message template and no explicit message in result
                if hasattr(func, '_rule_message') and func._rule_message and not res.message:
                    if res.format_args:
                        try:
                            res.message = func._rule_message % res.format_args
                        except TypeError:
                            # Fallback if formatting fails
                            res.message = func._rule_message
                    else:
                         res.message = func._rule_message

                self.model.last_step = res
                
                # Check if the step broke any rules
                is_ok, err_msg = self.model.puzzle_correct_so_far()
                if not is_ok:
                    res.message = f"!!! CONTRADICTION DETECTED !!! {err_msg}. (Rule: {res.rule})"
                    # We might want to stop here or signal error. 
                    # For now, we just update the message so the UI shows it.
                
                return res

        self.model.last_step = StepResult([], "No applicable rule found.", "None")
        is_ok, err_msg = self.model.puzzle_correct_so_far()
        if not is_ok:
            self.model.last_step.message = f"!!! CONTRADICTION DETECTED !!! {err_msg}. (No rule applied)"
        return self.model.last_step

    @solver_rule(priority=4, name="G1b Separation: match neighbor owners",
                 message="Restricted potential owners of (%d,%d) to match adjacent Land cells.")
    def try_rule_neighbor_of_fixed_restriction(self) -> Optional[StepResult]:
        """
        G1b Separation (Proactive): 
        If a cell (r,c) is adjacent to a Land cell (nr,nc), then (r,c) cannot belong to 
        an island 'k' unless (nr,nc) can also belong to 'k'.
        Logic: owners[r][c] &= owners[nr][nc] (for all Land neighbors)
        """
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if not self.model.cells[r][c].is_unknown:
                    continue
                
                # Combine masks from all LAND neighbors
                has_restriction = False
                combined_neighbor_mask = (1 << len(self.model.islands)) - 1
                
                for nr, nc in self.model.neighbors4(r, c):
                    if self.model.is_land_certain(nr, nc):
                        combined_neighbor_mask &= self.model.cells[nr][nc].owners
                        has_restriction = True
                
                if has_restriction:
                    if self.model.restrict_owners_intersection(r, c, combined_neighbor_mask):
                        return StepResult(
                            changed_cells=[(r, c)],
                            format_args=(r, c)
                        )
        return None

    @solver_rule(priority=1, name="G4 Anti-2x2: 3 blacks -> land",
                 message="Forced land to avoid a 2x2 black pool at block (%d,%d).")
    def try_rule_anti_2x2_force_land(self) -> Optional[StepResult]:
        for r in range(self.model.rows - 1):
            for c in range(self.model.cols - 1):
                cells = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
                blacks = [(rr, cc) for (rr, cc) in cells if self.model.is_black_certain(rr, cc)]
                if len(blacks) == 3:
                    # the remaining cell forced land
                    for rr, cc in cells:
                        if not self.model.is_black_certain(rr, cc):
                            if self.model.is_clue(rr, cc):
                                continue
                            # If not certain black, and we need to avoid 2x2, we force Land.
                            if not self.model.cells[rr][cc].is_black:
                                if self.model.force_land(rr, cc):
                                    return StepResult(
                                        changed_cells=[(rr, cc)],
                                        format_args=(r, c)
                                    )
        return None

    @solver_rule(priority=2, name="G1 Separation: neighbor of >=2 fixed owners -> black",
                 message="Forced black because cell touches multiple distinct fixed islands %s.")
    def try_rule_common_neighbor_two_fixed_owners_black(self) -> Optional[StepResult]:
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if self.model.is_clue(r, c) or self.model.is_black_certain(r, c):
                    continue
                fixed: Set[int] = set()
                for rr, cc in self.model.neighbors4(r, c):
                    fo = self.model.fixed_owner(rr, cc)
                    if fo is not None:
                        fixed.add(fo)
                if len(fixed) >= 2:
                    # touches at least two distinct fixed islands -> must be black
                    changed = self.model.force_black(r, c)
                    if changed:
                        ids = sorted(list(fixed))[:3]
                        return StepResult(
                            changed_cells=[(r, c)],
                            format_args=(ids,)
                        )
        return None

    @solver_rule(priority=3, name="G2 Unification: land cluster domain intersection",
                 message="Unified land cluster at %s with intersection of potential owners.")
    def try_rule_land_cluster_unification(self) -> Optional[StepResult]:
        visited = [[False for _ in range(self.model.cols)] for _ in range(self.model.rows)]
        K = len(self.model.islands)
        full_mask = (1 << K) - 1
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if self.model.is_land_certain(r, c) and not visited[r][c]:
                    # Find the connected component of land cells
                    component = []
                    q = [(r, c)]
                    visited[r][c] = True
                    common_mask = full_mask
                    idx = 0
                    while idx < len(q):
                        curr_r, curr_c = q[idx]
                        idx += 1
                        component.append((curr_r, curr_c))
                        common_mask &= self.model.cells[curr_r][curr_c].owners
                        for nr, nc in self.model.neighbors4(curr_r, curr_c):
                            if self.model.is_land_certain(nr, nc) and not visited[nr][nc]:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                    # Apply the intersection mask to all cells in this cluster
                    for cr, cc in component:
                        if self.model.cells[cr][cc].owners != common_mask:
                            self.model.cells[cr][cc].owners = common_mask
                            return StepResult(
                                changed_cells=[(cr, cc)],
                                format_args=(component[0],)
                            )
        return None

    @solver_rule(priority=5, name="G3 Closure: complete island forces black neighbors",
                 message="Island %d is complete (%d/%d); neighbor (%d,%d) must be black.")
    def try_rule_close_complete_island(self) -> Optional[StepResult]:
        for isl in self.model.islands:
            iid = isl.island_id
            clue = isl.clue
            bit = self.model.bit(iid)
            
            # 1. Get all cells definitively belonging to this island
            island_cells = []
            for r in range(self.model.rows):
                for c in range(self.model.cols):
                    cell = self.model.cells[r][c]
                    if cell.is_land and cell.owners == bit:
                        island_cells.append((r, c))
            
            # 2. If the island is complete, neighbors that are NOT in the island must be black
            if len(island_cells) == clue:
                for r, c in island_cells:
                    for rr, cc in self.model.neighbors4(r, c):
                        # ONLY target neighbors that are NOT already part of this island
                        if (rr, cc) not in island_cells:
                            if self.model.is_clue(rr, cc):
                                continue
                            if not self.model.is_black_certain(rr, cc):
                                self.model.force_black(rr, cc)
                                return StepResult(
                                    changed_cells=[(rr, cc)],
                                    format_args=(iid, clue, clue, rr, cc)
                                )
        return None

    @solver_rule(priority=6, name="G6 Empty domain -> black", 
                 message="Owners domain became empty; forced black (no island can own this cell).")
    def try_rule_empty_owners_becomes_black(self) -> Optional[StepResult]:
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if self.model.is_clue(r, c):
                    continue
                cell = self.model.cells[r][c]
                # If owners domain is empty and it's not already black (and not land), force black.
                if cell.owners == 0 and not cell.is_land and cell.state != CellState.BLACK:
                    self.model.force_black(r, c)
                    return StepResult(changed_cells=[(r, c)])
        return None

    @solver_rule(priority=7, name="G7 Generic Mandatory Expansion",
                 message="Island %d must include (%d,%d) to reach size %d (bottleneck detected).")
    def try_rule_island_mandatory_expansion(self) -> Optional[StepResult]:
        for isl in self.model.islands:
            iid = isl.island_id
            clue = isl.clue
            bit = self.model.bit(iid)
            
            # 1. Identify current connected component for this island
            sr, sc = isl.pos
            component = {(sr, sc)}
            stack = [(sr, sc)]
            while stack:
                r, c = stack.pop()
                for nr, nc in self.model.neighbors4(r, c):
                    if (nr, nc) not in component:
                        n_cell = self.model.cells[nr][nc]
                        # Cell belongs to island iid if it's land and uniquely owned by bit
                        if n_cell.is_land and n_cell.owners == bit:
                            component.add((nr, nc))
                            stack.append((nr, nc))
            
            if len(component) >= clue:
                continue
                
            # 2. Find all cells this island could potentially occupy
            potential = set()
            for r in range(self.model.rows):
                for c in range(self.model.cols):
                    if not self.model.is_black_certain(r, c) and (self.model.cells[r][c].owners & bit):
                        potential.add((r, c))
            
            # 3. Check neighbors of the current component for bottlenecks
            neighbors = set()
            for r, c in component:
                for nr, nc in self.model.neighbors4(r, c):
                    if (nr, nc) in potential and (nr, nc) not in component:
                        neighbors.add((nr, nc))
            
            for n in neighbors:
                # Test if island can reach 'clue' size WITHOUT using neighbor 'n'
                test_potential = potential - {n}
                q = list(component)
                seen = set(component)
                reached_count = 0
                idx = 0
                while idx < len(q):
                    curr = q[idx]
                    idx += 1
                    reached_count += 1
                    if reached_count >= clue: break
                    for nn in self.model.neighbors4(*curr):
                        if nn in test_potential and nn not in seen:
                            seen.add(nn)
                            q.append(nn)
                
                if reached_count < clue:
                    # 'n' is mandatory for island iid to ever reach its size
                    tr, tc = n
                    self.model.force_land(tr, tc)
                    self.model.cells[tr][tc].owners = bit
                    return StepResult(
                        changed_cells=[(tr, tc)],
                        format_args=(iid, tr, tc, clue)
                    )
        return None

    def _is_connected(self, cells: Set[Tuple[int, int]]) -> bool:
        if not cells:
            return True
        start = next(iter(cells))
        q = [start]
        seen = {start}
        count = 0
        while q:
            curr = q.pop()
            count += 1
            for n in self.model.neighbors4(*curr):
                if n in cells and n not in seen:
                    seen.add(n)
                    q.append(n)
        return count == len(cells)

    def analyze_island_extensions(self, isl) -> Optional[Tuple[Set[Tuple[int, int]], Set[Tuple[int, int]]]]:
        """
        Performs a brute-force search for all valid shapes of the given island.
        Returns:
            (union_of_extensions, intersection_of_extensions)
            union: Set of cells that appear in AT LEAST ONE valid completion (excluding the fixed core).
            intersection: Set of cells that appear in ALL valid completions (excluding the fixed core).
        """
        MAX_STATES = 50000
        iid = isl.island_id
        clue = isl.clue
        bit = self.model.bit(iid)

        # 1. Identify fixed core and potential cells
        fixed_core = set()
        potential = set()
        
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                cell = self.model.cells[r][c]
                if cell.is_land and cell.owners == bit:
                    fixed_core.add((r, c))
                if not self.model.is_black_certain(r, c) and (cell.owners & bit):
                    potential.add((r, c))

        # Filter potential: Must not touch any other island's fixed cells
        # This prevents the shape from violating the "islands must not touch" rule.
        # We only filter cells that are NOT in the fixed_core (to avoid breaking the core itself in case of existing contradiction).
        filtered_potential = set()
        for r, c in potential:
            if (r, c) in fixed_core:
                filtered_potential.add((r, c))
                continue
            
            touches_other = False
            for nr, nc in self.model.neighbors4(r, c):
                # Check if neighbor is Land and CANNOT be this island
                # If it's Land and doesn't include our bit, it belongs to others.
                n_cell = self.model.cells[nr][nc]
                if self.model.is_land_certain(nr, nc) and (n_cell.owners & bit) == 0:
                    touches_other = True
                    break
            
            if not touches_other:
                filtered_potential.add((r, c))
        potential = filtered_potential

        current_size = len(fixed_core)
        if current_size >= clue:
            return set(), set()
        
        needed = clue - current_size
        candidates = list(potential - fixed_core)
        if len(candidates) < needed:
            return set(), set()

        # Adjacency map for potential cells
        adj = {p: [] for p in potential}
        for r, c in potential:
            for nr, nc in self.model.neighbors4(r, c):
                if (nr, nc) in potential:
                    adj[(r, c)].append((nr, nc))

        # Initial frontier: neighbors of fixed_core in potential
        initial_frontier = set()
        for r, c in fixed_core:
            for nr, nc in adj[(r, c)]:
                if (nr, nc) not in fixed_core:
                    initial_frontier.add((nr, nc))

        stack = [(frozenset(), frozenset(initial_frontier))]
        visited_states = {frozenset()}
        
        common_added = None
        union_added = set()
        
        states_visited = 0
        too_complex = False
        
        while stack:
            states_visited += 1
            if states_visited > MAX_STATES:
                too_complex = True
                break
            
            curr_added, curr_frontier = stack.pop()
            
            if len(curr_added) == needed:
                if not self._is_connected(fixed_core | curr_added):
                    continue

                if common_added is None:
                    common_added = set(curr_added)
                else:
                    common_added &= curr_added
                
                union_added.update(curr_added)
                continue
            
            frontier_list = sorted(list(curr_frontier))
            for cell in frontier_list:
                new_added = set(curr_added)
                new_added.add(cell)
                new_added_frozen = frozenset(new_added)
                
                if new_added_frozen in visited_states:
                    continue
                
                new_frontier = set(curr_frontier)
                new_frontier.remove(cell)
                for n in adj[cell]:
                    if n not in fixed_core and n not in new_added:
                        new_frontier.add(n)
                
                visited_states.add(new_added_frozen)
                stack.append((new_added_frozen, frozenset(new_frontier)))

        if too_complex:
            return None
            
        if common_added is None:
            return set(), set()
            
        return union_added, common_added

    @solver_rule(priority=8, name="G7b Global Brute Force Intersection & Pruning",
                 message="Brute force analysis for Island %d: pruned unreachable candidates / enforced bottlenecks.")
    def try_rule_island_global_bottleneck(self) -> Optional[StepResult]:
        # "Brute Force" Intersection Rule
        for isl in self.model.islands:
            result = self.analyze_island_extensions(isl)
            if result is None:
                continue
            union_set, common_set = result
            
            changed_list = []
            bit = self.model.bit(isl.island_id)
            
            fixed_core = set()
            for r in range(self.model.rows):
                for c in range(self.model.cols):
                    cell = self.model.cells[r][c]
                    if cell.is_land and cell.owners == bit:
                        fixed_core.add((r, c))
            
            for r in range(self.model.rows):
                for c in range(self.model.cols):
                    if (r, c) in fixed_core:
                        continue
                    if self.model.cells[r][c].owners & bit:
                        if (r, c) not in union_set:
                            # This cell allows 'bit' but is not in any valid extension -> prune it
                            if self.model.remove_owner(r, c, isl.island_id):
                                changed_list.append((r, c))

            # 2. Bottleneck: Force cells that are in ALL valid extensions
            if common_set:
                for r, c in common_set:
                    if self.model.cells[r][c].state == CellState.BLACK: 
                        continue
                    
                    if self.model.force_land(r, c):
                        if (r, c) not in changed_list:
                             changed_list.append((r, c))
                    
                    if self.model.cells[r][c].owners != bit:
                        self.model.cells[r][c].owners = bit
                        if (r, c) not in changed_list:
                            changed_list.append((r, c))

            if changed_list:
                return StepResult(
                    changed_cells=changed_list,
                    format_args=(isl.island_id,)
                )

        return None

    @solver_rule(priority=9, name="G9 Global Black Connectivity (Articulation Point)",
                 message="Global Sea Connectivity: Cell (%d,%d) is a mandatory articulation point.")
    def try_rule_black_mandatory_expansion(self) -> Optional[StepResult]:
        # 1. Identify all connected components of current black cells
        visited = [[False for _ in range(self.model.cols)] for _ in range(self.model.rows)]
        black_components = []
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if self.model.is_black_certain(r, c) and not visited[r][c]:
                    comp = []
                    q = [(r, c)]
                    visited[r][c] = True
                    while q:
                        curr_r, curr_c = q.pop(0)
                        comp.append((curr_r, curr_c))
                        for nr, nc in self.model.neighbors4(curr_r, curr_c):
                            if self.model.is_black_certain(nr, nc) and not visited[nr][nc]:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                    black_components.append(comp)

        if len(black_components) < 2:
            return None

        # 2. Potential sea consists of all cells that are not certain land
        potential_sea = set()
        candidates = []
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if not self.model.is_land_certain(r, c):
                    potential_sea.add((r, c))
                    if not self.model.is_black_certain(r, c):
                        candidates.append((r, c))

        if not candidates:
            return None

        # Map each black cell to its component index for fast lookup
        cell_to_comp_idx = {}
        for idx, comp in enumerate(black_components):
            for cell in comp:
                cell_to_comp_idx[cell] = idx

        # Helper BFS to find reachable component INDICES from a start cell
        def get_reachable_component_indices(start_cell: Tuple[int, int], forbidden: Optional[Tuple[int, int]]) -> Set[int]:
            reached_comp_indices = set()
            
            # Optimization: If start_cell itself belongs to a component, add it immediately
            if start_cell in cell_to_comp_idx:
                reached_comp_indices.add(cell_to_comp_idx[start_cell])

            q = [start_cell]
            seen = {start_cell}
            if forbidden:
                seen.add(forbidden)
            
            idx = 0
            while idx < len(q):
                curr = q[idx]
                idx += 1
                
                # If we hit a black cell, record its component
                if curr in cell_to_comp_idx:
                    reached_comp_indices.add(cell_to_comp_idx[curr])
                
                for nr, nc in self.model.neighbors4(*curr):
                    if (nr, nc) in potential_sea and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            return reached_comp_indices

        # 3. Group components into partitions (sets of component indices that can reach each other)
        comp_partitions = [] # List of sets of component indices
        assigned_comps = set()
        
        for i in range(len(black_components)):
            if i in assigned_comps:
                continue
            
            # BFS from component i to find all connected components
            reached = get_reachable_component_indices(black_components[i][0], None)
            comp_partitions.append(reached)
            assigned_comps.update(reached)

        # Filter partitions that have < 2 components (no connectivity to protect)
        active_partitions = [p for p in comp_partitions if len(p) >= 2]
        
        if not active_partitions:
            return None

        # 4. Iterate over ALL unknown potential sea cells
        # For each candidate, check if it breaks ANY active partition
        for cand in candidates:
            # Let's just iterate partitions.
            for partition in active_partitions:
                # Pick a representative component from the partition
                start_comp_idx = next(iter(partition))
                start_cell = black_components[start_comp_idx][0]
                
                # Check reachability with 'cand' blocked
                reachable_indices = get_reachable_component_indices(start_cell, cand)
                
                # Check if we lost any components in this partition
                original_count = len(partition)
                found_count = len(reachable_indices & partition)
                
                if found_count < original_count:
                    # Split detected!
                    tr, tc = cand
                    if self.model.cells[tr][tc].state != CellState.BLACK:
                        self.model.force_black(tr, tc)
                        return StepResult(
                            changed_cells=[(tr, tc)],
                            format_args=(tr, tc)
                        )
        return None

    @solver_rule(priority=10, name="G10 Island Completion: common neighbor of last candidates -> black",
                 message="Island %d needs 1 cell; either %s or %s will complete it. Their common neighbor (%d,%d) must be black.")
    def try_rule_island_completion_common_neighbor_black(self) -> Optional[StepResult]:
        for isl in self.model.islands:
            iid = isl.island_id
            clue = isl.clue
            bit = self.model.bit(iid)
            
            sr, sc = isl.pos
            component = {(sr, sc)}
            stack = [(sr, sc)]
            while stack:
                r, c = stack.pop()
                for nr, nc in self.model.neighbors4(r, c):
                    if (nr, nc) not in component:
                        n_cell = self.model.cells[nr][nc]
                        if n_cell.is_land and n_cell.owners == bit:
                            component.add((nr, nc))
                            stack.append((nr, nc))
            
            if len(component) != clue - 1:
                continue
            
            candidates = set()
            for r, c in component:
                for nr, nc in self.model.neighbors4(r, c):
                    if (nr, nc) not in component and not self.model.is_black_certain(nr, nc):
                        if not self.model.is_clue(nr, nc) and (self.model.cells[nr][nc].owners & bit):
                            candidates.add((nr, nc))
            
            if len(candidates) == 2:
                c_list = list(candidates)
                n1 = set(self.model.neighbors4(*c_list[0]))
                n2 = set(self.model.neighbors4(*c_list[1]))
                common = n1 & n2
                
                for xr, xc in common:
                    if (xr, xc) not in component and (xr, xc) not in candidates:
                        cell_x = self.model.cells[xr][xc]
                        if not cell_x.is_land and cell_x.state != CellState.BLACK:
                            self.model.force_black(xr, xc)
                            return StepResult(
                                changed_cells=[(xr, xc)],
                                format_args=(iid, c_list[0], c_list[1], xr, xc)
                            )
        return None

    @solver_rule(priority=0, name="G5 Distance Pruning (Enhanced)",
                 message="Pruned potential owners based on reachability from current island components.")
    def try_rule_distance_pruning(self) -> Optional[StepResult]:
        changed_cells = []
        
        # Pre-compute fixed cells for each island to optimize
        fixed_cells_by_island = {isl.island_id: [] for isl in self.model.islands}
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                cell = self.model.cells[r][c]
                if cell.is_land and cell.owners != 0:
                    # Check if singleton owner
                    iid = self.model.owners_singleton(r, c)
                    if iid is not None:
                         if iid in fixed_cells_by_island:
                             fixed_cells_by_island[iid].append((r, c))

        for isl in self.model.islands:
            iid = isl.island_id
            clue = isl.clue
            bit = self.model.bit(iid)
            
            # Start BFS from ALL cells currently fixed to this island
            # Note: The clue cell itself is always fixed owner, so it's included.
            current_fixed = fixed_cells_by_island[iid]
            current_size = len(current_fixed)
            
            remaining = clue - current_size
            if remaining < 0:
                continue
                
            reachable = set(current_fixed)
            queue = [(r, c, 0) for r, c in current_fixed]
            
            idx = 0
            while idx < len(queue):
                r, c, d = queue[idx]
                idx += 1
                
                if d < remaining:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if self.model.in_bounds(nr, nc):
                            if (nr, nc) not in reachable:
                                # Obstacles: clues of OTHER islands, certain black cells, or cells owned by OTHER islands
                                is_obstacle = False
                                if self.model.is_black_certain(nr, nc):
                                    is_obstacle = True
                                elif (self.model.cells[nr][nc].owners & bit) == 0:
                                    is_obstacle = True

                                if not is_obstacle:
                                    reachable.add((nr, nc))
                                    queue.append((nr, nc, d + 1))
            
            # Prune this island bit from all cells it cannot reach
            for r in range(self.model.rows):
                for c in range(self.model.cols):
                    if (r, c) not in reachable:
                        if self.model.cells[r][c].owners & bit:
                            self.model.cells[r][c].owners &= ~bit
                            changed_cells.append((r, c))
        
        if changed_cells:
            return StepResult(changed_cells=list(set(changed_cells)))
        return None

    @solver_rule(priority=20, name="G11 Hypothetical Sea Connectivity",
                 message="Hypothetical island expansion at (%d,%d) disconnects sea.")
    def try_rule_hypothetical_sea_connectivity(self) -> Optional[StepResult]:
        # Optimization: identifying black cells once
        black_cells = []
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if self.model.is_black_certain(r, c):
                    black_cells.append((r, c))
        
        if len(black_cells) < 2:
            return None # Cannot disconnect < 2 cells

        black_cells_set = set(black_cells)

        # Pre-calculate fixed core for each island to speed up lookups
        island_cores = {}
        for isl in self.model.islands:
            island_cores[isl.island_id] = []
        
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if self.model.is_land_certain(r, c):
                    singleton = self.model.owners_singleton(r, c)
                    if singleton is not None:
                        island_cores[singleton].append((r, c))

        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if not self.model.cells[r][c].is_unknown:
                    continue
                
                owners_mask = self.model.cells[r][c].owners
                if owners_mask == 0: continue

                impossible_owners = []
                potential_owners = self.model.bitset_to_ids(owners_mask)
                
                for iid in potential_owners:
                    core_cells = island_cores[iid]
                    if not core_cells: continue

                    bit = self.model.bit(iid)
                    
                    # 1. BFS to get distance map from (r,c)
                    dist_map = {(r, c): 0}
                    q_bfs = [(r, c)]
                    idx = 0
                    
                    min_dist = float('inf')
                    
                    while idx < len(q_bfs):
                        curr = q_bfs[idx]
                        idx += 1
                        d = dist_map[curr]
                        
                        is_core = False
                        for cr, cc in core_cells:
                            if (cr, cc) == curr:
                                is_core = True
                                break
                        
                        if is_core:
                            if d < min_dist:
                                min_dist = d
                            continue 

                        if d >= min_dist:
                            continue

                        for nr, nc in self.model.neighbors4(*curr):
                            if (nr, nc) not in dist_map:
                                cell_n = self.model.cells[nr][nc]
                                if not cell_n.is_black and (cell_n.owners & bit):
                                    dist_map[(nr, nc)] = d + 1
                                    q_bfs.append((nr, nc))
                    
                    if min_dist == float('inf'):
                        impossible_owners.append(iid)
                        continue

                    # 2. DFS to reconstruct ALL shortest paths and check connectivity
                    targets = [cell for cell in core_cells if dist_map.get(cell) == min_dist]
                    
                    MAX_PATHS = 50
                    paths_checked = 0
                    found_valid_path = False
                    
                    stack = []
                    for t in targets:
                        stack.append((t, {t}))
                    
                    while stack:
                        curr, path_so_far = stack.pop()
                        
                        if curr == (r, c):
                            paths_checked += 1
                            obstacles = path_so_far.union(core_cells)
                            
                            q_sea = [black_cells[0]]
                            seen_sea = {black_cells[0]}
                            reached_blacks_count = 0
                            sea_idx = 0
                            
                            while sea_idx < len(q_sea):
                                curr_sea = q_sea[sea_idx]
                                sea_idx += 1
                                if curr_sea in black_cells_set:
                                    reached_blacks_count += 1
                                for nr, nc in self.model.neighbors4(*curr_sea):
                                    if (nr, nc) not in obstacles:
                                        if not self.model.is_land_certain(nr, nc):
                                            if (nr, nc) not in seen_sea:
                                                seen_sea.add((nr, nc))
                                                q_sea.append((nr, nc))
                            
                            if reached_blacks_count == len(black_cells):
                                found_valid_path = True
                                break
                            
                            if paths_checked >= MAX_PATHS:
                                found_valid_path = True 
                                break
                            continue

                        current_d = dist_map[curr]
                        for nr, nc in self.model.neighbors4(*curr):
                            if dist_map.get((nr, nc)) == current_d - 1:
                                new_path = path_so_far.copy()
                                new_path.add((nr, nc))
                                stack.append(((nr, nc), new_path))
                    
                    if not found_valid_path:
                        impossible_owners.append(iid)

                if len(impossible_owners) == len(potential_owners):
                    self.model.force_black(r, c)
                    msg = "Every hypothetical expansion of (%d,%d) disconnects sea; forced Black." % (r, c)
                    return StepResult(
                        changed_cells=[(r, c)],
                        message=msg
                    )
                elif len(impossible_owners) > 0:
                    changed = False
                    removed_ids = []
                    for iid in impossible_owners:
                        if self.model.remove_owner(r, c, iid):
                            changed = True
                            removed_ids.append(iid)
                    if changed:
                        ids_str = ", ".join(map(str, sorted(removed_ids)))
                        msg = "Removed impossible owners %s for cell (%d,%d) because their shortest paths disconnect sea." % (ids_str, r, c)
                        return StepResult(
                            changed_cells=[(r, c)],
                            message=msg
                        )

        return None

# Dynamically generate the list of rule names sorted by priority
NurikabeSolver.RULE_NAMES = [r[2] for r in sorted(_RULES, key=lambda x: x[0])]
