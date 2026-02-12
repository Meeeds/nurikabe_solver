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
        # Check for contradictions at the start of the step
        is_ok, err_msg = self.model.puzzle_correct_so_far()
        if not is_ok:
            prev_rule = self.model.last_step.rule if self.model.last_step else "None"
            res = StepResult([], f"!!! CONTRADICTION DETECTED !!! {err_msg}. (Rule: {prev_rule})", "BROKEN_NURIKABE_RULES")
            self.model.last_step = res
            return res

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
                return res

        self.model.last_step = StepResult([], "No applicable rule found.", "None")
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
        land_components = self.model.get_all_components(self.model.is_land_certain)
        K = len(self.model.islands)
        full_mask = (1 << K) - 1
        
        for component in land_components:
            common_mask = full_mask
            for cr, cc in component:
                common_mask &= self.model.cells[cr][cc].owners
            
            # Apply the intersection mask to all cells in this cluster
            for cr, cc in component:
                if self.model.cells[cr][cc].owners != common_mask:
                    self.model.cells[cr][cc].owners = common_mask
                    return StepResult(
                        changed_cells=[(cr, cc)],
                        format_args=(list(component)[0],)
                    )
        return None

    @solver_rule(priority=5, name="G3 Closure: complete island forces black neighbors",
                 message="Island %d is complete (%d/%d); neighbor (%d,%d) must be black.")
    def try_rule_close_complete_island(self) -> Optional[StepResult]:
        for isl in self.model.islands:
            iid = isl.island_id
            clue = isl.clue
            
            # 1. Get all cells definitively belonging to this island (globally)
            island_cells = self.model.get_island_core_cells(iid)
            
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
            component = self.model.get_connected_component(isl.pos[0], isl.pos[1],
                lambda r, c: self.model.is_land_certain(r, c) and self.model.cells[r][c].owners == bit)
            
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
                if self.model.is_mandatory_for_reach_size(n, component, clue, potential):
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
        start = list(cells)[0]
        seen = self.model.get_connected_component(start[0], start[1], lambda rr, cc: (rr, cc) in cells)
        return len(seen) == len(cells)

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

        # 1. Identify fixed core (globally) and potential cells
        fixed_core = self.model.get_island_core_cells(iid)
        
        potential = set()
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                cell = self.model.cells[r][c]
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
            
            # Use get_island_core_cells for global discovery
            fixed_core = self.model.get_island_core_cells(isl.island_id)
            
            for r in range(self.model.rows):
                for c in range(self.model.cols):
                    if (r, c) in fixed_core:
                        continue
                    if self.model.can_be_land(r, c, isl.island_id):
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
        black_components = self.model.get_all_components(self.model.is_black_certain)

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

        # Helper to find reachable component INDICES from a start cell
        def get_reachable_component_indices(start_cell: Tuple[int, int]) -> Set[int]:
            reachable_cells = self.model.get_connected_component(start_cell[0], start_cell[1], lambda r, c: (r, c) in potential_sea)
            reached_comp_indices = set()
            for cell in reachable_cells:
                if cell in cell_to_comp_idx:
                    reached_comp_indices.add(cell_to_comp_idx[cell])
            return reached_comp_indices

        # 3. Group components into partitions (sets of component indices that can reach each other)
        comp_partitions = [] # List of sets of component indices
        assigned_comps = set()
        
        for i in range(len(black_components)):
            if i in assigned_comps:
                continue
            
            # BFS from component i to find all connected components
            reached = get_reachable_component_indices(list(black_components[i])[0])
            comp_partitions.append(reached)
            assigned_comps.update(reached)

        # Filter partitions that have < 2 components (no connectivity to protect)
        active_partitions = [p for p in comp_partitions if len(p) >= 2]
        
        if not active_partitions:
            return None

        # 4. Iterate over ALL unknown potential sea cells
        for cand in candidates:
            for partition in active_partitions:
                # Pick a representative component from the partition
                start_comp_idx = next(iter(partition))
                start_cell = list(black_components[start_comp_idx])[0]
                other_targets = [list(black_components[i])[0] for i in partition if i != start_comp_idx]
                
                if self.model.is_mandatory_for_connectivity(cand, start_cell, other_targets, potential_sea):
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
            
            # Identify current connected component for this island
            component = self.model.get_connected_component(isl.pos[0], isl.pos[1],
                lambda r, c: self.model.is_fixed_to(r, c, iid))
            
            if len(component) != clue - 1:
                continue
            
            candidates = set()
            for r, c in component:
                for nr, nc in self.model.neighbors4(r, c):
                    if (nr, nc) not in component and self.model.can_be_land(nr, nc, iid):
                        if not self.model.is_clue(nr, nc):
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
        
        # Get all fixed cells for each island in a single pass to optimize and maintain determinism
        all_cores = self.model.get_all_island_core_cells()

        for isl in self.model.islands:
            iid = isl.island_id
            clue = isl.clue
            bit = self.model.bit(iid)
            
            # Start BFS from ALL cells currently fixed to this island
            current_fixed = all_cores[iid]
            current_size = len(current_fixed)
            
            remaining = clue - current_size
            if remaining < 0:
                continue
            
            def obstacle_predicate(nr, nc):
                # Obstacles: certain black cells, or cells that CANNOT be owned by this island
                return not self.model.can_be_land(nr, nc, iid)

            reachable = self.model.get_reachable_cells(current_fixed, remaining, obstacle_predicate)
            
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

# Dynamically generate the list of rule names sorted by priority
NurikabeSolver.RULE_NAMES = [r[2] for r in sorted(_RULES, key=lambda x: x[0])]
