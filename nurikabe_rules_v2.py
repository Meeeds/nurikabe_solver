from typing import Optional, List
from nurikabe_model import NurikabeModel, StepResult
from IslandSteinerPropagator import IslandSteinerPropagator


class NurikabeSolverV2:
    RULE_NAMES: List[str] = [
        "B0 Distance Pruning (Steiner Tree)",
        "R0 Empty domain -> Sea",
        "R1 Anti 2x2",
        "R2 Neighbor Domain Restriction",
        "R3 Mandatory Expansion",
        "R4 Global Sea Connectivity",
        "R5 Generalized Theorem"
    ]

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

        # Run rules in order of priority
        for full_name in self.RULE_NAMES:
            code = full_name.split()[0] # B0, R0, etc.
            method_name = f"try_{code}"
            method = getattr(self, method_name, None)
            if method:
                res = method()
                if res:
                    if not res.rule:
                        res.rule = full_name
                    self.model.last_step = res
                    return res
        
        return StepResult([], "No more V2 rules", "None")

    def try_B0(self) -> Optional[StepResult]:
        """B0 - Distance pruning and global consistency (Steiner Tree)."""
        changed_cells = []
        for isl in self.model.islands:
            iid = isl.island_id
            exclusive_2x2 = self.model.get_exclusive_2x2_pools(iid)
            propagator = IslandSteinerPropagator(self.model, iid)
            reductions, contradiction = propagator.run(mandatory_groups=exclusive_2x2)
            
            if contradiction:
                return StepResult([], f"Contradiction: Island {iid} cannot connect its components, reach 2x2 responsibilities, or reach size {isl.clue}", "BROKEN_NURIKABE_RULES")
            
            for r, c in reductions:
                if self.model.remove_owner(r, c, iid):
                    changed_cells.append((r, c))
                                
        if changed_cells:
            return StepResult(list(set(changed_cells)), "Pruned potential owners")
        return None

    def try_R0(self) -> Optional[StepResult]:
        """R0 - Empty domain implies sea."""
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if self.model.is_clue(r, c) or self.model.is_sea_certain(r, c):
                    continue
                if self.model.cells[r][c].owners.is_empty():
                    if self.model.force_sea(r, c):
                        return StepResult([(r, c)], f"Forced sea at ({r},{c}) because no island can reach it")
        return None

    def try_R1(self) -> Optional[StepResult]:
        """R1 - Anti 2x2 pool."""
        for r in range(self.model.rows - 1):
            for c in range(self.model.cols - 1):
                cells = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
                seas = [(rr, cc) for rr, cc in cells if self.model.is_sea_certain(rr, cc)]
                if len(seas) == 3:
                    for rr, cc in cells:
                        if not self.model.is_sea_certain(rr, cc):
                            if not self.model.is_land_certain(rr, cc):
                                if self.model.force_land(rr, cc):
                                    return StepResult([(rr, cc)], f"Forced land at ({rr},{cc}) to avoid 2x2 sea pool")
        return None

    def try_R2(self) -> Optional[StepResult]:
        """R2 - Domain restriction by neighbors."""
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if self.model.is_sea_certain(r, c):
                    continue
                
                changed = False
                for nr, nc in self.model.neighbors4(r, c):
                    if self.model.is_land_certain(nr, nc):
                        # Owners of this cell must be a subset of the land neighbor's owners
                        if self.model.restrict_owners_intersection(r, c, self.model.cells[nr][nc].owners):
                            changed = True
                
                if changed:
                    return StepResult([(r, c)], f"Restricted owners of ({r},{c}) to match adjacent land")
        return None

    def try_R3(self) -> Optional[StepResult]:
        """R3 - Mandatory expansion."""
        for isl in self.model.islands:
            iid = isl.island_id
            clue = isl.clue
            core = list(self.model.get_island_core_cells(iid))
            if not core: continue
            
            # Identify 2x2 pools that ONLY this island can save
            exclusive_2x2 = self.model.get_exclusive_2x2_pools(iid)
            
            potential_area = set()
            for r in range(self.model.rows):
                for c in range(self.model.cols):
                    if self.model.can_be_land(r, c, iid):
                        potential_area.add((r, c))
            
            candidates = [p for p in potential_area if self.model.cells[p[0]][p[1]].is_unknown]
            
            for p in candidates:
                # 1. Mandatory for size
                if self.model.is_mandatory_for_reach_size(p, core, clue, potential_area):
                    changed_land = self.model.force_land(p[0], p[1])
                    changed_owner = self.model.force_owner(p[0], p[1], iid)
                    if changed_land or changed_owner:
                        return StepResult([p], f"R3: Forced land/owner at {p} for island {iid} to reach size {clue}")
                
                # 2. Mandatory for connectivity of core cells
                if len(core) > 1:
                    if self.model.is_mandatory_for_connectivity(p, core[0], core[1:], potential_area):
                        changed_land = self.model.force_land(p[0], p[1])
                        changed_owner = self.model.force_owner(p[0], p[1], iid)
                        if changed_land or changed_owner:
                            return StepResult([p], f"R3: Forced land/owner at {p} to connect core of island {iid}")
                
                # 3. Mandatory for connectivity to exclusive 2x2 pools
                for block_set in exclusive_2x2:
                    if self.model.is_mandatory_for_connectivity_to_set(p, core[0], block_set, potential_area):
                        changed_land = self.model.force_land(p[0], p[1])
                        changed_owner = self.model.force_owner(p[0], p[1], iid)
                        if changed_land or changed_owner:
                            return StepResult([p], f"R3: Forced land/owner at {p} for island {iid} to prevent exclusive 2x2 pool")
        return None

    def try_R4(self) -> Optional[StepResult]:
        """R4 - Global sea connectivity."""
        sea_components = self.model.get_all_components(self.model.is_sea_certain)
        if len(sea_components) < 2:
            return None
        
        potential_sea = set()
        candidates = []
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if not self.model.is_land_certain(r, c):
                    potential_sea.add((r, c))
                    if self.model.cells[r][c].is_unknown:
                        candidates.append((r, c))
        
        targets = [list(comp)[0] for comp in sea_components]
        for cand in candidates:
            if self.model.is_mandatory_for_connectivity(cand, targets[0], targets[1:], potential_sea):
                if self.model.force_sea(cand[0], cand[1]):
                    return StepResult([cand], f"Forced sea at {cand} to preserve global sea connectivity")
        return None

    def try_R5(self) -> Optional[StepResult]:
        """R5 - Generalized Theorem."""
        for isl in self.model.islands:
            iid = isl.island_id
            clue = isl.clue
            core = self.model.get_island_core_cells(iid)
            if len(core) >= clue:
                continue
            
            potential = set()
            for r in range(self.model.rows):
                for c in range(self.model.cols):
                    if self.model.can_be_land(r, c, iid):
                        potential.add((r, c))
            
            frontier = set()
            for r, c in core:
                for nr, nc in self.model.neighbors4(r, c):
                    if (nr, nc) in potential and (nr, nc) not in core:
                        frontier.add((nr, nc))
            
            if not frontier:
                continue
            
            # Find a candidate X (unknown, not in core/frontier) that neighbors all of frontier E
            first_e = list(frontier)[0]
            for xr, xc in self.model.neighbors4(first_e[0], first_e[1]):
                if (xr, xc) in core or (xr, xc) in frontier:
                    continue
                if (xr, xc) in potential: # Important: skip if it could belong to THIS island
                    continue
                if self.model.is_clue(xr, xc) or self.model.is_sea_certain(xr, xc):
                    continue
                
                x_neighbors = set(self.model.neighbors4(xr, xc))
                if frontier.issubset(x_neighbors):
                    if self.model.force_sea(xr, xc):
                        return StepResult([(xr, xc)], f"R5: Forced sea at ({xr},{xc}) because it neighbors all expansion candidates {list(frontier)}")
        return None
