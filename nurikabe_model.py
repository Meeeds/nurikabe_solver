from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from enum import IntEnum

# ----------------------------
# Domain model
# ----------------------------

class CellState(IntEnum):
    UNKNOWN = 0
    BLACK = 1  # Sea
    LAND = 2

RuleName = str

@dataclass
class Island:
    island_id: int
    clue: int
    pos: Tuple[int, int]  # (r,c)

@dataclass
class Cell:
    state: CellState = CellState.UNKNOWN
    owners: int = 0  # bitmask of potential island owners

    @property
    def is_land(self) -> bool:
        return self.state == CellState.LAND

    @property
    def is_black(self) -> bool:
        return self.state == CellState.BLACK

    @property
    def is_unknown(self) -> bool:
        return self.state == CellState.UNKNOWN

@dataclass
class StepResult:
    changed_cells: List[Tuple[int, int]]
    message: str = ""
    rule: RuleName = ""
    format_args: Tuple[Any, ...] = field(default_factory=tuple)


class NurikabeModel:
    def __init__(self) -> None:
        self.rows = 0
        self.cols = 0
        self.clues: List[List[int]] = []
        self.islands: List[Island] = []
        self.island_by_pos: Dict[Tuple[int, int], int] = {}

        # The grid of Cells
        self.cells: List[List[Cell]] = []

        # UI/log support
        self.last_step: Optional[StepResult] = None

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def neighbors4(self, r: int, c: int) -> List[Tuple[int, int]]:
        out = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if self.in_bounds(rr, cc):
                out.append((rr, cc))
        return out

    def is_clue(self, r: int, c: int) -> bool:
        return self.clues[r][c] > 0

    def bit(self, island_id: int) -> int:
        # island_id is 1..K
        return 1 << (island_id - 1)

    def bitset_to_ids(self, bits: int) -> List[int]:
        ids = []
        i = 1
        b = bits
        while b:
            if b & 1:
                ids.append(i)
            b >>= 1
            i += 1
        return ids

    def get_cell(self, r: int, c: int) -> Cell:
        return self.cells[r][c]

    def owners_empty(self, r: int, c: int) -> bool:
        return self.cells[r][c].owners == 0

    def owners_singleton(self, r: int, c: int) -> Optional[int]:
        bits = self.cells[r][c].owners
        if bits != 0 and (bits & (bits - 1)) == 0:
            return bits.bit_length()
        return None

    def is_black_certain(self, r: int, c: int) -> bool:
        return self.cells[r][c].state == CellState.BLACK

    def is_land_certain(self, r: int, c: int) -> bool:
        return self.cells[r][c].state == CellState.LAND

    def fixed_owner(self, r: int, c: int) -> Optional[int]:
        if self.is_land_certain(r, c):
            return self.owners_singleton(r, c)
        return None

    def force_black(self, r: int, c: int) -> bool:
        """Force cell to BLACK state."""
        if self.is_clue(r, c):
            return False
        cell = self.cells[r][c]
        if cell.state == CellState.BLACK:
            return False
        
        # If it was Land, this is a contradiction, but we enforce the new state
        cell.state = CellState.BLACK
        cell.owners = 0  # Black cells have no owners
        return True

    def force_land(self, r: int, c: int) -> bool:
        """Force cell to LAND state."""
        if self.is_clue(r, c):
            return False
        cell = self.cells[r][c]
        if cell.state == CellState.LAND:
            return False
        
        cell.state = CellState.LAND
        # Ensure it has potential owners if none were set (recovery/init)
        if cell.owners == 0:
            cell.owners = self.get_potential_owners_mask(r, c)
        return True

    def restrict_owners_intersection(self, r: int, c: int, mask: int) -> bool:
        """Owners := Owners & mask"""
        if self.is_clue(r, c):
            return False
        cell = self.cells[r][c]
        before = cell.owners
        after = before & mask
        if after != before:
            cell.owners = after
            # If owners becomes empty, it implies it cannot be Land.
            # If it was Land or Unknown, it should become Black (G6 logic).
            # However, this method only restricts owners. Rule G6 handles the state transition.
            return True
        return False

    def remove_owner(self, r: int, c: int, island_id: int) -> bool:
        """Owners := Owners without island_id"""
        if self.is_clue(r, c):
            return False
        cell = self.cells[r][c]
        b = self.bit(island_id)
        before = cell.owners
        after = before & (~b)
        if after != before:
            cell.owners = after
            return True
        return False

    def get_potential_owners_mask(self, r: int, c: int) -> int:
        mask = 0
        for isl in self.islands:
            # Simple Manhattan distance check for initial potential
            dist = abs(r - isl.pos[0]) + abs(c - isl.pos[1])
            if dist < isl.clue:
                mask |= self.bit(isl.island_id)
        return mask

    def parse_puzzle_text(self, text: str) -> Tuple[bool, str]:
        lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip() != ""]
        if not lines:
            return False, "Empty input."

        grid: List[List[int]] = []
        for ln in lines:
            if " " in ln.strip():
                toks = [t for t in ln.strip().split(" ") if t != ""]
                row: List[int] = []
                for t in toks:
                    if t in [".", "0"]:
                        row.append(0)
                    else:
                        try:
                            v = int(t)
                            if v < 0:
                                return False, "Negative clue not allowed."
                            row.append(v)
                        except ValueError:
                            return False, f"Bad token: {t}"
                grid.append(row)
            else:
                row = []
                for ch in ln.strip():
                    if ch == "." or ch == "0":
                        row.append(0)
                    elif ch.isdigit():
                        row.append(int(ch))
                    else:
                        return False, f"Bad character: {ch}"
                grid.append(row)

        cols = len(grid[0])
        if any(len(r) != cols for r in grid):
            return False, "Ragged rows: all rows must have the same number of columns."
        self.load_grid(grid)
        return True, "Loaded."

    def load_grid(self, grid: List[List[int]]) -> None:
        self.clues = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

        self.islands = []
        self.island_by_pos = {}
        island_id = 0
        for r in range(self.rows):
            for c in range(self.cols):
                if self.clues[r][c] > 0:
                    island_id += 1
                    isl = Island(island_id=island_id, clue=self.clues[r][c], pos=(r, c))
                    self.islands.append(isl)
                    self.island_by_pos[(r, c)] = island_id

        # Init cells
        self.cells = [[Cell() for _ in range(self.cols)] for _ in range(self.rows)]

        # Initialize owners domains
        K = len(self.islands)
        all_mask = (1 << K) - 1
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.cells[r][c]
                if self.is_clue(r, c):
                    iid = self.island_by_pos[(r, c)]
                    cell.owners = self.bit(iid)
                    cell.state = CellState.LAND
                else:
                    cell.owners = all_mask
                    cell.state = CellState.UNKNOWN

        # Clues cannot be adjacent orthogonally to another clue (validation)
        for isl in self.islands:
            r, c = isl.pos
            for rr, cc in self.neighbors4(r, c):
                if self.is_clue(rr, cc):
                    self.last_step = StepResult(
                        changed_cells=[],
                        message="Invalid puzzle: orthogonally adjacent clues.",
                        rule="Validate"
                    )

    def apply_distance_pruning_all(self) -> None:
        for isl in self.islands:
            iid = isl.island_id
            limit = isl.clue
            sr, sc = isl.pos
            bit = self.bit(iid)
            reachable = {(sr, sc)}
            queue = [(sr, sc, 0)]
            idx = 0
            while idx < len(queue):
                r, c, d = queue[idx]
                idx += 1
                if d < limit - 1:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            if (nr, nc) not in reachable:
                                cell = self.cells[nr][nc]
                                # Obstacle if it's strictly Black, or a clue of another island
                                is_obstacle = (cell.state == CellState.BLACK)
                                if self.is_clue(nr, nc) and (nr, nc) != (sr, sc):
                                    is_obstacle = True
                                
                                if not is_obstacle:
                                    reachable.add((nr, nc))
                                    queue.append((nr, nc, d + 1))
            
            for r in range(self.rows):
                for c in range(self.cols):
                    if (r, c) not in reachable:
                        self.cells[r][c].owners &= ~bit

    def reset_domains_from_manual(self) -> None:
        """Rebuild domains from scratch but keep current states."""
        # Backup current states
        states = [[self.cells[r][c].state for c in range(self.cols)] for r in range(self.rows)]
        
        # Reload grid to reset everything
        self.load_grid([row[:] for row in self.clues])
        
        # Restore states
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_clue(r, c): 
                    continue
                old_state = states[r][c]
                if old_state == CellState.BLACK:
                    self.force_black(r, c)
                elif old_state == CellState.LAND:
                    self.force_land(r, c)
                # If UNKNOWN, we leave it as initialized by load_grid (UNKNOWN with full mask)

    def island_size_assigned(self, island_id: int) -> int:
        """Count land certain cells whose owner singleton is island_id."""
        cnt = 0
        b = self.bit(island_id)
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.cells[r][c]
                if cell.state == CellState.LAND and cell.owners == b:
                    cnt += 1
        return cnt

    def cycle_state(self, r: int, c: int, forward: bool = True) -> None:
        if self.is_clue(r, c):
            return
        cell = self.cells[r][c]
        # UNKNOWN -> LAND -> BLACK -> UNKNOWN
        cycle = [CellState.UNKNOWN, CellState.BLACK, CellState.LAND] if forward else [CellState.UNKNOWN, CellState.LAND, CellState.BLACK]
        # Note order: UI requested Unknown -> Land -> Black -> Unknown
        # Let's match UI request: U -> L -> B -> U
        cycle = [CellState.UNKNOWN, CellState.LAND, CellState.BLACK]
        
        idx = cycle.index(cell.state)
        new_state = cycle[(idx + 1) % len(cycle)]
        
        if new_state == CellState.UNKNOWN:
            cell.state = CellState.UNKNOWN
            if cell.owners == 0:
                cell.owners = self.get_potential_owners_mask(r, c)
        elif new_state == CellState.BLACK:
            self.force_black(r, c)
        elif new_state == CellState.LAND:
            self.force_land(r, c)

    def snapshot(self) -> Dict[str, object]:
        last_step = None
        if self.last_step is not None:
            last_step = {
                "changed_cells": list(self.last_step.changed_cells),
                "message": self.last_step.message,
                "rule": self.last_step.rule,
            }
        
        # Serialize cells
        # We store lists [state, owners] for JSON compatibility
        cells_data = []
        for r in range(self.rows):
            row_data = []
            for c in range(self.cols):
                cell = self.cells[r][c]
                row_data.append([int(cell.state), cell.owners])
            cells_data.append(row_data)

        return {
            "clues": [row[:] for row in self.clues],
            "cells": cells_data,
            "last_step": last_step,
        }

    def puzzle_correct_so_far(self) -> Tuple[bool, str]:
        """
        Checks if the current definitive state (LAND, BLACK, clues) 
        respects the basic rules of Nurikabe.
        Returns (is_correct, error_message).
        """
        # 1. No 2x2 black blocks
        for r in range(self.rows - 1):
            for c in range(self.cols - 1):
                if (self.is_black_certain(r, c) and 
                    self.is_black_certain(r + 1, c) and 
                    self.is_black_certain(r, c + 1) and 
                    self.is_black_certain(r + 1, c + 1)):
                    return False, f"2x2 sea block at ({r},{c})"

        # 2. Island checks (connectivity, clues, size)
        visited = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        land_components = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_land_certain(r, c) and not visited[r][c]:
                    comp = []
                    q = [(r, c)]
                    visited[r][c] = True
                    while q:
                        curr_r, curr_c = q.pop(0)
                        comp.append((curr_r, curr_c))
                        for nr, nc in self.neighbors4(curr_r, curr_c):
                            if self.is_land_certain(nr, nc) and not visited[nr][nc]:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                    land_components.append(comp)

        for comp in land_components:
            comp_clues = [(rr, cc) for (rr, cc) in comp if self.is_clue(rr, cc)]
            
            # Multiple clues in one island
            if len(comp_clues) > 1:
                clue_vals = [self.clues[rr][cc] for (rr, cc) in comp_clues]
                return False, f"Island contains multiple clues: {clue_vals} at {comp_clues[0]}"
            
            # Size check
            if len(comp_clues) == 1:
                cr, cc = comp_clues[0]
                clue_val = self.clues[cr][cc]
                if len(comp) > clue_val:
                    return False, f"Island at ({cr},{cc}) is too large: {len(comp)} > {clue_val}"
                
                # If surrounded by sea, it must be exactly the right size
                is_sealed = True
                for rr, cc in comp:
                    for nr, nc in self.neighbors4(rr, cc):
                        if not self.is_land_certain(nr, nc) and not self.is_black_certain(nr, nc):
                            is_sealed = False
                            break
                    if not is_sealed: break
                
                if is_sealed and len(comp) < clue_val:
                    return False, f"Island at ({cr},{cc}) is sealed but too small: {len(comp)} < {clue_val}"
            else:
                # No clues in this land component
                # Check if it's sealed
                is_sealed = True
                for rr, cc in comp:
                    for nr, nc in self.neighbors4(rr, cc):
                        if not self.is_land_certain(nr, nc) and not self.is_black_certain(nr, nc):
                            is_sealed = False
                            break
                    if not is_sealed: break
                if is_sealed:
                    return False, f"Land component at {comp[0]} has no clue and is sealed"

        # 3. Check for completion
        is_complete = True
        for r in range(self.rows):
            for c in range(self.cols):
                if self.cells[r][c].is_unknown:
                    is_complete = False
                    break
            if not is_complete: break
        
        if is_complete:
            # All land cells must have been visited (already checked by component analysis)
            # All sea cells must be connected
            sea_cells = [(r, c) for r in range(self.rows) for c in range(self.cols) if self.is_black_certain(r, c)]
            if sea_cells:
                start_sea = sea_cells[0]
                q = [start_sea]
                seen_sea = {start_sea}
                while q:
                    curr_r, curr_c = q.pop(0)
                    for nr, nc in self.neighbors4(curr_r, curr_c):
                        if self.is_black_certain(nr, nc) and (nr, nc) not in seen_sea:
                            seen_sea.add((nr, nc))
                            q.append((nr, nc))
                if len(seen_sea) != len(sea_cells):
                    return False, "Sea is not connected"
            
            # Every clue must have an island (already checked if all land is assigned)
            # and every island must have a clue (already checked)
            # and sizes must match (already checked)

        return True, "OK"

    def restore(self, state: Dict[str, object]) -> None:
        clues = state.get("clues")
        if not isinstance(clues, list):
            raise ValueError("Invalid snapshot: 'clues' missing.")
        
        self.load_grid([[int(v) for v in row] for row in clues])

        cells_data = state.get("cells")
        if isinstance(cells_data, list):
            for r in range(self.rows):
                for c in range(self.cols):
                    if r < len(cells_data) and c < len(cells_data[r]):
                        s_val, o_val = cells_data[r][c]
                        self.cells[r][c].state = CellState(s_val)
                        self.cells[r][c].owners = o_val

        last_step = state.get("last_step")
        if last_step and isinstance(last_step, dict):
            self.last_step = StepResult(
                changed_cells=[(int(r), int(c)) for (r, c) in last_step.get("changed_cells", [])],
                message=str(last_step.get("message", "")),
                rule=str(last_step.get("rule", "")),
            )
        else:
            self.last_step = None

