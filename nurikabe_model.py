from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

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

    def cycle_manual_mark(self, r: int, c: int, forward: bool = True) -> None:
        if self.is_clue(r, c):
            return
        mark = self.manual_mark[r][c]
        cycle = [UNKNOWN, BLACK, LAND] if forward else [UNKNOWN, LAND, BLACK]
        idx = cycle.index(mark)
        new_mark = cycle[(idx + 1) % len(cycle)]
        self.manual_mark[r][c] = new_mark

        # Apply to domain/black_possible immediately (local "manual constraint")
        if new_mark == UNKNOWN:
            # "unknown" means revert to permissive: black possible
            self.black_possible[r][c] = True
            # if owners was emptied (due to being black), we should try to restore potential owners
            # otherwise it stays "impossible" which isn't what "unknown" implies for the user.
            if self.owners[r][c] == 0:
                 self.owners[r][c] = self.get_potential_owners_mask(r, c)
        elif new_mark == BLACK:
            self.force_black(r, c)
        elif new_mark == LAND:
            self.force_land(r, c)

    def snapshot(self) -> Dict[str, object]:
        """Return a fully self-contained, pickle-friendly snapshot of the current state.

        Intended for:
        - Undo/redo stacks (store snapshots)
        - Background solving (send snapshots across processes)
        """
        last_step = None
        if self.last_step is not None:
            last_step = {
                "changed_cells": list(self.last_step.changed_cells),
                "message": self.last_step.message,
                "rule": self.last_step.rule,
            }
        return {
            "clues": [row[:] for row in self.clues],
            "manual_mark": [row[:] for row in self.manual_mark],
            "black_possible": [row[:] for row in self.black_possible],
            "owners": [row[:] for row in self.owners],
            "last_step": last_step,
        }

    def restore(self, state: Dict[str, object]) -> None:
        """Restore a state previously produced by snapshot()."""
        clues = state.get("clues")
        if not isinstance(clues, list) or not clues or not isinstance(clues[0], list):
            raise ValueError("Invalid snapshot: missing/invalid 'clues'.")

        self.load_grid([[int(v) for v in row] for row in clues])

        manual_mark = state.get("manual_mark")
        black_possible = state.get("black_possible")
        owners = state.get("owners")

        if not (isinstance(manual_mark, list) and isinstance(black_possible, list) and isinstance(owners, list)):
            raise ValueError("Invalid snapshot: missing arrays.")

        self.manual_mark = [[int(v) for v in row] for row in manual_mark]
        self.black_possible = [[bool(v) for v in row] for row in black_possible]
        self.owners = [[int(v) for v in row] for row in owners]

        last_step = state.get("last_step")
        if last_step is None:
            self.last_step = None
        else:
            if not isinstance(last_step, dict):
                raise ValueError("Invalid snapshot: 'last_step' must be dict or None.")
            self.last_step = StepResult(
                changed_cells=[(int(r), int(c)) for (r, c) in last_step.get("changed_cells", [])],
                message=str(last_step.get("message", "")),
                rule=str(last_step.get("rule", "")),
            )
