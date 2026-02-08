from typing import Optional, Set, List, Tuple
from nurikabe_model import NurikabeModel, StepResult, BLACK, CellState

class NurikabeSolver:
    def __init__(self, model: NurikabeModel) -> None:
        self.model = model

    def step(self) -> Optional[StepResult]:
        # Priority order:
        # 0) Distance pruning (G5)
        res = self.try_rule_distance_pruning()
        if res:
            self.model.last_step = res
            return res

        # 1) Anti-2x2: 3 blacks -> force land (G4)
        res = self.try_rule_anti_2x2_force_land()
        if res:
            self.model.last_step = res
            return res

        # 2) Neighbor of two fixed different owners -> force black (G1 strong)
        res = self.try_rule_common_neighbor_two_fixed_owners_black()
        if res:
            self.model.last_step = res
            return res

        # 3) Neighbor of Land must share potential owners (Separation)
        res = self.try_rule_neighbor_of_fixed_restriction()
        if res:
            self.model.last_step = res
            return res

        # 4) Land adjacency with fixed owner propagates owner (G2 safe form)
        res = self.try_rule_land_cluster_unification()
        if res:
            self.model.last_step = res
            return res

        # 5) Closure of complete islands: remove owner from neighbors (G3)
        res = self.try_rule_close_complete_island()
        if res:
            self.model.last_step = res
            return res

        # 6) Empty owners -> black certain (G6) (as an explicit step)
        res = self.try_rule_empty_owners_becomes_black()
        if res:
            self.model.last_step = res
            return res

        # 7) Mandatory expansion (bottleneck or unique neighbor)
        res = self.try_rule_island_mandatory_expansion()
        if res:
            self.model.last_step = res
            return res

        # 8) Global Bottleneck: check all potential cells, not just neighbors
        res = self.try_rule_island_global_bottleneck()
        if res:
            self.model.last_step = res
            return res

        # 9) Sea connectivity: mandatory bridge for black components (G9)
        res = self.try_rule_black_mandatory_expansion()
        if res:
            self.model.last_step = res
            return res

        # 10) Island near completion: common neighbor of 2 last candidates -> black (G10)
        res = self.try_rule_island_completion_common_neighbor_black()
        if res:
            self.model.last_step = res
            return res

        self.model.last_step = StepResult([], "No applicable rule found.", "None")
        return self.model.last_step

    def try_rule_neighbor_of_fixed_restriction(self) -> Optional[StepResult]:
        """
        G1b Separation (Proactive): 
        If a cell (r,c) is adjacent to a Land cell (nr,nc), then (r,c) cannot belong to 
        an island 'k' unless (nr,nc) can also belong to 'k'.
        Logic: owners[r][c] &= owners[nr][nc] (for all Land neighbors)
        """
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if self.model.is_clue(r, c) or self.model.is_black_certain(r, c):
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
                            message=f"Restricted potential owners of ({r},{c}) to match adjacent Land cells.",
                            rule="G1b Separation: match neighbor owners"
                        )
        return None

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
                            # (Implicitly checking if Land is possible? The model allows forcing land unless it is already black)
                            if not self.model.cells[rr][cc].is_black:
                                if self.model.force_land(rr, cc):
                                    return StepResult(
                                        changed_cells=[(rr, cc)],
                                        message=f"Forced land to avoid a 2x2 black pool at block ({r},{c}).",
                                        rule="G4 Anti-2x2: 3 blacks -> land"
                                    )
        return None

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
                            message=f"Forced black because cell touches multiple distinct fixed islands {ids}.",
                            rule="G1 Separation: neighbor of >=2 fixed owners -> black"
                        )
        return None

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
                                message=f"Unified land cluster at {component[0]} with intersection of potential owners.",
                                rule="G2 Unification: land cluster domain intersection"
                            )
        return None

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
                                    message=f"Island {iid} is complete ({clue}/{clue}); neighbor ({rr},{cc}) must be black.",
                                    rule="G3 Closure: complete island forces black neighbors"
                                )
        return None

    def try_rule_empty_owners_becomes_black(self) -> Optional[StepResult]:
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if self.model.is_clue(r, c):
                    continue
                cell = self.model.cells[r][c]
                # If owners domain is empty and it's not already black (and not land), force black.
                if cell.owners == 0 and not cell.is_land and cell.state != BLACK:
                    self.model.force_black(r, c)
                    return StepResult(
                        changed_cells=[(r, c)],
                        message="Owners domain became empty; forced black (no island can own this cell).",
                        rule="G6 Empty domain -> black"
                    )
        return None

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
                        message=f"Island {iid} must include ({tr},{tc}) to reach size {clue} (bottleneck detected).",
                        rule="G7 Generic Mandatory Expansion"
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
                    if self.model.cells[r][c].state == BLACK: 
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
                    message=f"Brute force analysis for Island {isl.island_id}: pruned unreachable candidates / enforced bottlenecks.",
                    rule="G7b Global Brute Force Intersection & Pruning"
                )

        return None

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
                    if self.model.cells[tr][tc].state != BLACK:
                        self.model.force_black(tr, tc)
                        return StepResult(
                            changed_cells=[(tr, tc)],
                            message=f"Global Sea Connectivity: Cell ({tr},{tc}) is a mandatory articulation point.",
                            rule="G9 Global Black Connectivity (Articulation Point)"
                        )
        return None

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
                        if not cell_x.is_land and cell_x.state != BLACK:
                            self.model.force_black(xr, xc)
                            return StepResult(
                                changed_cells=[(xr, xc)],
                                message=f"Island {iid} needs 1 cell; either {c_list[0]} or {c_list[1]} will complete it. Their common neighbor ({xr},{xc}) must be black.",
                                rule="G10 Island Completion: common neighbor of last candidates -> black"
                            )
        return None

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
            return StepResult(
                changed_cells=list(set(changed_cells)),
                message="Pruned potential owners based on reachability from current island components.",
                rule="G5 Distance Pruning (Enhanced)"
            )
        return None
