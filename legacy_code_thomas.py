"""
Nurikabe Assistant (Pygame)

Features:
- Paste / type a puzzle grid in-app (multiline).
- Play manually: toggle cell states (unknown / black / land).
- Click "Step" to apply ONE automatic deduction step.
- The app prints and shows which rule fired and what changed.

Puzzle input format:
- One row per line.
- Use '.' or '0' for empty (no clue).
- Use integers (e.g. 1,2,12) for clues.
- Separate tokens by spaces OR write them tightly (e.g. "..3.1").
Examples:
  . . 3 . 1
  . 2 . . .
Or:
  ..3.1
  .2...

Controls:
- Left click on a non-clue cell: cycle Unknown -> Black -> Land -> Unknown
- Right click on a non-clue cell: cycle Unknown -> Land -> Black -> Unknown
- Mouse wheel over text box: scroll text input
- Buttons: Load, Step, Reset (rebuild domains from current manual marks)
"""

import pygame
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set

# ----------------------------
# Domain model
# ----------------------------

UNKNOWN = 0
BLACK = 1
LAND = 2

RuleName = str


@dataclass
class Island:
    island_id: int
    clue: int
    pos: Tuple[int, int]  # (r,c)


@dataclass
class StepResult:
    changed_cells: List[Tuple[int, int]]
    message: str
    rule: RuleName


class NurikabeModel:
    def __init__(self) -> None:
        self.rows = 0
        self.cols = 0
        self.clues: List[List[int]] = []
        self.islands: List[Island] = []
        self.island_by_pos: Dict[Tuple[int, int], int] = {}

        # For each cell:
        # - black_possible: bool
        # - owners: bitset (int) where bit i means island i (1..K) is possible owner
        # - manual_mark: UNKNOWN/BLACK/LAND affects black_possible / owners
        self.black_possible: List[List[bool]] = []
        self.owners: List[List[int]] = []
        self.manual_mark: List[List[int]] = []

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

    def owners_empty(self, r: int, c: int) -> bool:
        return self.owners[r][c] == 0

    def owners_singleton(self, r: int, c: int) -> Optional[int]:
        bits = self.owners[r][c]
        if bits != 0 and (bits & (bits - 1)) == 0:
            # index of the single bit
            return (bits.bit_length())
        return None

    def is_black_certain(self, r: int, c: int) -> bool:
        # In our model: black certain iff owners is empty AND black_possible is True
        return self.owners[r][c] == 0 and self.black_possible[r][c]

    def is_land_certain(self, r: int, c: int) -> bool:
        return not self.black_possible[r][c]

    def fixed_owner(self, r: int, c: int) -> Optional[int]:
        if self.is_land_certain(r, c):
            return self.owners_singleton(r, c)
        return None

    def force_black(self, r: int, c: int) -> bool:
        """Force cell to black certain by emptying owners."""
        if self.is_clue(r, c):
            return False
        changed = False
        if self.owners[r][c] != 0:
            self.owners[r][c] = 0
            changed = True
        if not self.black_possible[r][c]:
            # black is now forced, but if black was impossible, this is a contradiction.
            # We handle contradictions by allowing it, but note it.
            self.black_possible[r][c] = True
            changed = True
        self.manual_mark[r][c] = BLACK
        return changed

    def force_land(self, r: int, c: int) -> bool:
        if self.is_clue(r, c):
            return False
        changed = False
        if self.black_possible[r][c]:
            self.black_possible[r][c] = False
            changed = True
        # Policy: A land cell must ALWAYS have potential owners
        if self.owners[r][c] == 0:
            self.owners[r][c] = self.get_potential_owners_mask(r, c)
            changed = True
            
        self.manual_mark[r][c] = LAND
        return changed

    def restrict_owners_intersection(self, r: int, c: int, mask: int) -> bool:
        """Owners := Owners & mask"""
        if self.is_clue(r, c):
            return False
        before = self.owners[r][c]
        after = before & mask
        if after != before:
            self.owners[r][c] = after
            # If owners empties, this implies black certain (G6), unless black is impossible.
            if after == 0:
                # keep black_possible as-is; if black_possible false => contradiction.
                pass
            return True
        return False

    def remove_owner(self, r: int, c: int, island_id: int) -> bool:
        """Owners := Owners without island_id"""
        if self.is_clue(r, c):
            return False
        b = self.bit(island_id)
        before = self.owners[r][c]
        after = before & (~b)
        if after != before:
            self.owners[r][c] = after
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

        # tokenized if spaces exist, else treat as char-grid with digits possibly multi-digit? (hard)
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
                # char grid: allow '.' and digits 1..9 only
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

        # init arrays
        self.black_possible = [[True for _ in range(self.cols)] for _ in range(self.rows)]
        self.owners = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.manual_mark = [[UNKNOWN for _ in range(self.cols)] for _ in range(self.rows)]

        # initialize owners domains
        K = len(self.islands)
        all_mask = (1 << K) - 1
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_clue(r, c):
                    iid = self.island_by_pos[(r, c)]
                    self.owners[r][c] = self.bit(iid)
                    self.black_possible[r][c] = False
                    self.manual_mark[r][c] = LAND
                else:
                    self.owners[r][c] = all_mask
                    self.black_possible[r][c] = True
                    self.manual_mark[r][c] = UNKNOWN

        # apply initial pruning by distance (G5)
        self.apply_distance_pruning_all()

        # clues cannot be adjacent orthogonally to another clue (validation)
        # we won't fail hard; we'll just record a last_step message.
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
                                if not self.is_black_certain(nr, nc) and not (self.is_clue(nr, nc) and (nr, nc) != (sr, sc)):
                                    reachable.add((nr, nc))
                                    queue.append((nr, nc, d + 1))
            for r in range(self.rows):
                for c in range(self.cols):
                    if (r, c) not in reachable:
                        self.owners[r][c] &= ~bit

    def reset_domains_from_manual(self) -> None:
        """Rebuild domains from scratch but keep manual marks (black/land/unknown)."""
        grid = [row[:] for row in self.clues]
        self.load_grid(grid)
        # re-apply manual marks on non-clue cells
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_clue(r, c):
                    continue
                mark = self.manual_mark[r][c]
                if mark == BLACK:
                    self.force_black(r, c)
                elif mark == LAND:
                    self.force_land(r, c)
        # Re-run pruning to account for new black walls and land constraints
        self.apply_distance_pruning_all()

    # ----------------------------
    # Rule application (single-step)
    # ----------------------------

    def island_size_assigned(self, island_id: int) -> int:
        """Count land certain cells whose owner singleton is island_id."""
        cnt = 0
        b = self.bit(island_id)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.black_possible[r][c]:
                    continue
                if self.owners[r][c] == b:
                    cnt += 1
        return cnt

    def step(self) -> Optional[StepResult]:
        # Priority order:
        # 1) Anti-2x2: 3 blacks -> force land (G4)
        res = self.try_rule_anti_2x2_force_land()
        if res:
            self.last_step = res
            return res

        # 2) Neighbor of two fixed different owners -> force black (G1 strong)
        res = self.try_rule_common_neighbor_two_fixed_owners_black()
        if res:
            self.last_step = res
            return res

        # 3) Land adjacency with fixed owner propagates owner (G2 safe form)
        res = self.try_rule_land_cluster_unification()
        if res:
            self.last_step = res
            return res

        # 4) Closure of complete islands: remove owner from neighbors (G3)
        res = self.try_rule_close_complete_island()
        if res:
            self.last_step = res
            return res

        # 5) Empty owners -> black certain (G6) (as an explicit step)
        res = self.try_rule_empty_owners_becomes_black()
        if res:
            self.last_step = res
            return res

        # 6) Mandatory expansion (bottleneck or unique neighbor)
        res = self.try_rule_island_mandatory_expansion()
        if res:
            self.last_step = res
            return res

        # 8) Sea connectivity: mandatory bridge for black components (G9)
        res = self.try_rule_black_mandatory_expansion()
        if res:
            self.last_step = res
            return res

        # 9) Island near completion: common neighbor of 2 last candidates -> black (G10)
        res = self.try_rule_island_completion_common_neighbor_black()
        if res:
            self.last_step = res
            return res

        self.last_step = StepResult([], "No applicable rule found.", "None")
        return self.last_step

    def try_rule_anti_2x2_force_land(self) -> Optional[StepResult]:
        for r in range(self.rows - 1):
            for c in range(self.cols - 1):
                cells = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
                blacks = [(rr, cc) for (rr, cc) in cells if self.is_black_certain(rr, cc)]
                if len(blacks) == 3:
                    # the remaining cell forced land
                    for rr, cc in cells:
                        if not self.is_black_certain(rr, cc):
                            if self.is_clue(rr, cc):
                                continue
                            if self.black_possible[rr][cc]:
                                self.force_land(rr, cc)
                                return StepResult(
                                    changed_cells=[(rr, cc)],
                                    message=f"Forced land to avoid a 2x2 black pool at block ({r},{c}).",
                                    rule="G4 Anti-2x2: 3 blacks -> land"
                                )
        return None

    def try_rule_common_neighbor_two_fixed_owners_black(self) -> Optional[StepResult]:
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_clue(r, c) or self.is_black_certain(r, c):
                    continue
                fixed: Set[int] = set()
                for rr, cc in self.neighbors4(r, c):
                    fo = self.fixed_owner(rr, cc)
                    if fo is not None:
                        fixed.add(fo)
                if len(fixed) >= 2:
                    # touches at least two distinct fixed islands -> must be black
                    changed = self.force_black(r, c)
                    if changed:
                        ids = sorted(list(fixed))[:3]
                        return StepResult(
                            changed_cells=[(r, c)],
                            message=f"Forced black because cell touches multiple distinct fixed islands {ids}.",
                            rule="G1 Separation: neighbor of >=2 fixed owners -> black"
                        )
        return None

    def try_rule_land_cluster_unification(self) -> Optional[StepResult]:
        visited = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        K = len(self.islands)
        full_mask = (1 << K) - 1
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_land_certain(r, c) and not visited[r][c]:
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
                        common_mask &= self.owners[curr_r][curr_c]
                        for nr, nc in self.neighbors4(curr_r, curr_c):
                            if self.is_land_certain(nr, nc) and not visited[nr][nc]:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                    # Apply the intersection mask to all cells in this cluster
                    for cr, cc in component:
                        if self.owners[cr][cc] != common_mask:
                            self.owners[cr][cc] = common_mask
                            return StepResult(
                                changed_cells=[(cr, cc)],
                                message=f"Unified land cluster at {component[0]} with intersection of potential owners.",
                                rule="G2 Unification: land cluster domain intersection"
                            )
        return None

    def try_rule_close_complete_island(self) -> Optional[StepResult]:
        for isl in self.islands:
            iid = isl.island_id
            clue = isl.clue
            bit = self.bit(iid)
            
            # 1. Get all cells definitively belonging to this island
            island_cells = []
            for r in range(self.rows):
                for c in range(self.cols):
                    if not self.black_possible[r][c] and self.owners[r][c] == bit:
                        island_cells.append((r, c))
            
            # 2. If the island is complete, neighbors that are NOT in the island must be black
            if len(island_cells) == clue:
                for r, c in island_cells:
                    for rr, cc in self.neighbors4(r, c):
                        # ONLY target neighbors that are NOT already part of this island
                        if (rr, cc) not in island_cells:
                            if self.is_clue(rr, cc):
                                continue
                            if not self.is_black_certain(rr, cc):
                                self.force_black(rr, cc)
                                return StepResult(
                                    changed_cells=[(rr, cc)],
                                    message=f"Island {iid} is complete ({clue}/{clue}); neighbor ({rr},{cc}) must be black.",
                                    rule="G3 Closure: complete island forces black neighbors"
                                )
        return None

    def try_rule_empty_owners_becomes_black(self) -> Optional[StepResult]:
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_clue(r, c):
                    continue
                if self.owners[r][c] == 0 and self.black_possible[r][c] and self.manual_mark[r][c] != BLACK:
                    self.manual_mark[r][c] = BLACK
                    return StepResult(
                        changed_cells=[(r, c)],
                        message="Owners domain became empty; forced black (no island can own this cell).",
                        rule="G6 Empty domain -> black"
                    )
        return None

    def try_rule_island_mandatory_expansion(self) -> Optional[StepResult]:
        for isl in self.islands:
            iid = isl.island_id
            clue = isl.clue
            bit = self.bit(iid)
            
            # 1. Identify current connected component for this island
            sr, sc = isl.pos
            component = {(sr, sc)}
            stack = [(sr, sc)]
            while stack:
                r, c = stack.pop()
                for nr, nc in self.neighbors4(r, c):
                    if (nr, nc) not in component:
                        # Cell belongs to island iid if it's land and uniquely owned by bit
                        if not self.black_possible[nr][nc] and self.owners[nr][nc] == bit:
                            component.add((nr, nc))
                            stack.append((nr, nc))
            
            if len(component) >= clue:
                continue
                
            # 2. Find all cells this island could potentially occupy
            potential = set()
            for r in range(self.rows):
                for c in range(self.cols):
                    if not self.is_black_certain(r, c) and (self.owners[r][c] & bit):
                        potential.add((r, c))
            
            # 3. Check neighbors of the current component for bottlenecks
            neighbors = set()
            for r, c in component:
                for nr, nc in self.neighbors4(r, c):
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
                    for nn in self.neighbors4(*curr):
                        if nn in test_potential and nn not in seen:
                            seen.add(nn)
                            q.append(nn)
                
                if reached_count < clue:
                    # 'n' is mandatory for island iid to ever reach its size
                    tr, tc = n
                    self.force_land(tr, tc)
                    self.owners[tr][tc] = bit
                    return StepResult(
                        changed_cells=[(tr, tc)],
                        message=f"Island {iid} must include ({tr},{tc}) to reach size {clue} (bottleneck detected).",
                        rule="G7 Generic Mandatory Expansion"
                    )
        return None

    def try_rule_black_mandatory_expansion(self) -> Optional[StepResult]:
        # 1. Identify all connected components of current black cells
        visited = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        black_components = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.is_black_certain(r, c) and not visited[r][c]:
                    comp = []
                    q = [(r, c)]
                    visited[r][c] = True
                    while q:
                        curr_r, curr_c = q.pop(0)
                        comp.append((curr_r, curr_c))
                        for nr, nc in self.neighbors4(curr_r, curr_c):
                            if self.is_black_certain(nr, nc) and not visited[nr][nc]:
                                visited[nr][nc] = True
                                q.append((nr, nc))
                    black_components.append(comp)

        if not black_components:
            return None

        # 2. Potential sea consists of all cells that are not certain land
        potential_sea = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if not self.is_land_certain(r, c):
                    potential_sea.add((r, c))

        # 3. For each component, check if it's forced to use a specific cell to connect to the rest
        for comp in black_components:
            comp_set = set(comp)
            others = potential_sea - comp_set
            if not others:
                continue

            # Find candidates: neighbors of the component that could be black
            candidates = set()
            for r, c in comp:
                for nr, nc in self.neighbors4(r, c):
                    if (nr, nc) in others:
                        candidates.add((nr, nc))

            for cand in candidates:
                # Test connectivity if 'cand' was removed from potential sea
                remaining_potential = others - {cand}
                
                # BFS starting from the component to see if it can reach ANY other potential cell
                q = [comp[0]]
                seen = {comp[0]}
                can_reach_rest = False
                idx = 0
                while idx < len(q):
                    curr = q[idx]
                    idx += 1
                    if curr in remaining_potential:
                        can_reach_rest = True
                        break
                    for nr, nc in self.neighbors4(*curr):
                        if (nr, nc) not in seen and ((nr, nc) in comp_set or (nr, nc) in remaining_potential):
                            seen.add((nr, nc))
                            q.append((nr, nc))
                
                if not can_reach_rest:
                    # 'cand' is a mandatory bridge for this component to stay connected
                    tr, tc = cand
                    if self.manual_mark[tr][tc] != BLACK:
                        self.force_black(tr, tc)
                        return StepResult(
                            changed_cells=[(tr, tc)],
                            message=f"Sea component at {comp[0]} must include ({tr},{tc}) to stay connected to the rest of the potential sea.",
                            rule="G9 Black Connectivity: mandatory bridge"
                        )
        return None

    def try_rule_island_completion_common_neighbor_black(self) -> Optional[StepResult]:
        for isl in self.islands:
            iid = isl.island_id
            clue = isl.clue
            bit = self.bit(iid)
            
            sr, sc = isl.pos
            component = {(sr, sc)}
            stack = [(sr, sc)]
            while stack:
                r, c = stack.pop()
                for nr, nc in self.neighbors4(r, c):
                    if (nr, nc) not in component:
                        if not self.black_possible[nr][nc] and self.owners[nr][nc] == bit:
                            component.add((nr, nc))
                            stack.append((nr, nc))
            
            if len(component) != clue - 1:
                continue
            
            candidates = set()
            for r, c in component:
                for nr, nc in self.neighbors4(r, c):
                    if (nr, nc) not in component and not self.is_black_certain(nr, nc):
                        if not self.is_clue(nr, nc) and (self.owners[nr][nc] & bit):
                            candidates.add((nr, nc))
            
            if len(candidates) == 2:
                c_list = list(candidates)
                n1 = set(self.neighbors4(*c_list[0]))
                n2 = set(self.neighbors4(*c_list[1]))
                common = n1 & n2
                
                for xr, xc in common:
                    if (xr, xc) not in component and (xr, xc) not in candidates:
                        if self.black_possible[xr][xc] and self.manual_mark[xr][xc] != BLACK:
                            self.force_black(xr, xc)
                            return StepResult(
                                changed_cells=[(xr, xc)],
                                message=f"Island {iid} needs 1 cell; either {c_list[0]} or {c_list[1]} will complete it. Their common neighbor ({xr},{xc}) must be black.",
                                rule="G10 Island Completion: common neighbor of last candidates -> black"
                            )
        return None
# ----------------------------
# UI widgets
# ----------------------------

class Button:
    def __init__(self, rect: pygame.Rect, text: str) -> None:
        self.rect = rect
        self.text = text

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, mouse_pos: Tuple[int, int]) -> None:
        hover = self.rect.collidepoint(mouse_pos)
        color = (210, 210, 210) if hover else (190, 190, 190)
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, (80, 80, 80), self.rect, 2, border_radius=8)
        surf = font.render(self.text, True, (0, 0, 0))
        screen.blit(surf, (self.rect.x + 10, self.rect.y + (self.rect.height - surf.get_height()) // 2))

    def clicked(self, event: pygame.event.Event) -> bool:
        return event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos)


class TextBox:
    def __init__(self, rect: pygame.Rect, initial_text: str = "") -> None:
        self.rect = rect
        self.text = initial_text
        self.active = False
        self.scroll = 0  # vertical scroll in pixels

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.active = self.rect.collidepoint(event.pos)

        if event.type == pygame.MOUSEWHEEL:
            # scroll only if mouse is over the box
            mx, my = pygame.mouse.get_pos()
            if self.rect.collidepoint((mx, my)):
                self.scroll = max(0, self.scroll - event.y * 20)

        if not self.active:
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                self.text += "\n"
            elif event.key == pygame.K_TAB:
                self.text += "    "
            else:
                if event.unicode:
                    self.text += event.unicode

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        pygame.draw.rect(screen, (245, 245, 245), self.rect, border_radius=8)
        pygame.draw.rect(screen, (80, 80, 80), self.rect, 2, border_radius=8)

        # Render multiline with scroll
        lines = self.text.splitlines() or [""]
        y = self.rect.y + 8 - self.scroll
        for ln in lines:
            surf = font.render(ln, True, (0, 0, 0))
            if y + surf.get_height() >= self.rect.y and y <= self.rect.y + self.rect.height:
                screen.blit(surf, (self.rect.x + 8, y))
            y += surf.get_height() + 2

        # caret indicator
        if self.active:
            pygame.draw.circle(screen, (0, 0, 0), (self.rect.right - 12, self.rect.y + 14), 3)


# ----------------------------
# Main app
# ----------------------------

def toggle_cell_manual(model: NurikabeModel, r: int, c: int, forward: bool = True) -> None:
    if model.is_clue(r, c):
        return
    mark = model.manual_mark[r][c]
    cycle = [UNKNOWN, BLACK, LAND] if forward else [UNKNOWN, LAND, BLACK]
    idx = cycle.index(mark)
    new_mark = cycle[(idx + 1) % len(cycle)]
    model.manual_mark[r][c] = new_mark

    # Apply to domain/black_possible immediately (local "manual constraint")
    if new_mark == UNKNOWN:
        # "unknown" means revert to permissive: black possible, owners stay as-is
        model.black_possible[r][c] = True
        # owners unchanged; if it was emptied by black, keep empty (user can Reset to rebuild)
    elif new_mark == BLACK:
        model.force_black(r, c)
    elif new_mark == LAND:
        model.force_land(r, c)


def draw_grid(
    screen: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    model: NurikabeModel,
    top_left: Tuple[int, int],
    cell_size: int,
    highlight: Set[Tuple[int, int]],
) -> None:
    ox, oy = top_left
    for r in range(model.rows):
        for c in range(model.cols):
            x = ox + c * cell_size
            y = oy + r * cell_size
            rect = pygame.Rect(x, y, cell_size, cell_size)

            # base color
            if model.is_clue(r, c):
                color = (235, 235, 255)
            else:
                if model.is_black_certain(r, c) or model.manual_mark[r][c] == BLACK:
                    color = (20, 20, 20)
                elif not model.black_possible[r][c] or model.manual_mark[r][c] == LAND:
                    color = (255, 255, 255)
                else:
                    color = (220, 220, 220)

            pygame.draw.rect(screen, color, rect)

            # highlight
            if (r, c) in highlight:
                pygame.draw.rect(screen, (255, 200, 0), rect, 4)
            else:
                pygame.draw.rect(screen, (90, 90, 90), rect, 1)

            # clue text
            if model.is_clue(r, c):
                # draw clue number
                val = model.clues[r][c]
                surf = font.render(str(val), True, (0, 0, 0))
                screen.blit(surf, (rect.x + (cell_size - surf.get_width()) // 2,
                                   rect.y + (cell_size - surf.get_height()) // 2))
                # draw island ID in top-left (same as land cells)
                iid = model.island_by_pos[(r, c)]
                id_surf = small_font.render(str(iid), True, (0, 0, 200))
                screen.blit(id_surf, (rect.x + 3, rect.y + 2))
            else:
                # show singleton owner as a tiny id (optional but very helpful)
                if not model.black_possible[r][c] and model.owners[r][c] != 0:
                    single = model.owners_singleton(r, c)
                    if single is not None:
                        surf = small_font.render(str(single), True, (0, 0, 120))
                        screen.blit(surf, (x + 3, y + 3))


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Nurikabe Assistant")

    W, H = 1200, 780
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 20)
    small_font = pygame.font.SysFont("consolas", 14)
    big_font = pygame.font.SysFont("consolas", 24)

    # UI layout
    textbox = TextBox(
        pygame.Rect(20, 20, 420, 240),
        initial_text="2........2\n......2...\n.2..7.....\n..........\n......3.3.\n..2....3..\n2..4......\n..........\n.1....2.4.\n",
    )
    btn_load = Button(pygame.Rect(20, 270, 130, 40), "Load")
    btn_step = Button(pygame.Rect(160, 270, 130, 40), "Step")
    btn_reset = Button(pygame.Rect(300, 270, 140, 40), "Reset Domains")

    msg_rect = pygame.Rect(20, 320, 420, 440)

    model = NurikabeModel()
    ok, status = model.parse_puzzle_text(textbox.text)
    if not ok:
        model.last_step = StepResult([], status, "Parse")

    # Grid drawing area
    grid_origin = (480, 20)
    cell_size = 32

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        highlight_cells: Set[Tuple[int, int]] = set()
        if model.last_step:
            highlight_cells = set(model.last_step.changed_cells)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            textbox.handle_event(event)

            if btn_load.clicked(event):
                ok, status = model.parse_puzzle_text(textbox.text)
                if not ok:
                    model.last_step = StepResult([], status, "Parse")
                else:
                    model.last_step = StepResult([], "Puzzle loaded. Domains initialized with distance pruning.", "Load")

            if btn_reset.clicked(event):
                model.reset_domains_from_manual()
                model.last_step = StepResult([], "Domains rebuilt (distance pruning) and manual marks re-applied.", "Reset")

            if btn_step.clicked(event):
                model.step()

            # Grid clicks (manual play)
            if event.type == pygame.MOUSEBUTTONDOWN:
                # avoid clicking through UI
                if textbox.rect.collidepoint(event.pos) or btn_load.rect.collidepoint(event.pos) or btn_step.rect.collidepoint(event.pos) or btn_reset.rect.collidepoint(event.pos) or msg_rect.collidepoint(event.pos):
                    continue

                gx, gy = grid_origin
                mx, my = event.pos
                if mx >= gx and my >= gy and model.rows > 0 and model.cols > 0:
                    c = (mx - gx) // cell_size
                    r = (my - gy) // cell_size
                    if 0 <= r < model.rows and 0 <= c < model.cols:
                        if event.button == 1:
                            toggle_cell_manual(model, r, c, forward=True)
                        elif event.button == 3:
                            toggle_cell_manual(model, r, c, forward=False)

        # draw background
        screen.fill((250, 250, 250))

        # draw UI
        textbox.draw(screen, font)
        btn_load.draw(screen, font, mouse_pos)
        btn_step.draw(screen, font, mouse_pos)
        btn_reset.draw(screen, font, mouse_pos)

        # message panel
        pygame.draw.rect(screen, (245, 245, 245), msg_rect, border_radius=8)
        pygame.draw.rect(screen, (80, 80, 80), msg_rect, 2, border_radius=8)
        title = big_font.render("Last action / explanation", True, (0, 0, 0))
        screen.blit(title, (msg_rect.x + 10, msg_rect.y + 10))

        if model.last_step:
            rule = font.render(f"Rule: {model.last_step.rule}", True, (0, 0, 0))
            screen.blit(rule, (msg_rect.x + 10, msg_rect.y + 50))

            # wrap message
            msg = model.last_step.message
            words = msg.split(" ")
            lines = []
            cur = ""
            for w in words:
                test = (cur + " " + w).strip()
                if font.size(test)[0] < msg_rect.width - 20:
                    cur = test
                else:
                    lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)

            y = msg_rect.y + 85
            for ln in lines[:14]:
                surf = font.render(ln, True, (0, 0, 0))
                screen.blit(surf, (msg_rect.x + 10, y))
                y += surf.get_height() + 6

        # draw grid
        if model.rows > 0:
            draw_grid(screen, font, small_font, model, grid_origin, cell_size, highlight_cells)

            # small legend
            lx, ly = grid_origin[0], grid_origin[1] + model.rows * cell_size + 10
            legend = [
                "Legend:",
                "Unknown: gray",
                "Black (sea): black",
                "Land (island cell): white",
                "Small blue number: singleton owner id",
            ]
            y = ly
            for ln in legend:
                surf = small_font.render(ln, True, (0, 0, 0))
                screen.blit(surf, (lx, y))
                y += surf.get_height() + 2

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
