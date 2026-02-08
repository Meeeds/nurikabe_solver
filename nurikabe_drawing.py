import math
import pygame
from dataclasses import dataclass
from typing import Tuple, Optional, List
from nurikabe_model import NurikabeModel, BLACK, LAND
import grid_style

@dataclass
class Camera:
    offset_x: float = 0.0
    offset_y: float = 0.0
    zoom: float = 1.0

    def screen_to_world(self, sx: float, sy: float) -> Tuple[float, float]:
        return (sx - self.offset_x) / self.zoom, (sy - self.offset_y) / self.zoom

    def world_to_screen(self, wx: float, wy: float) -> Tuple[float, float]:
        return wx * self.zoom + self.offset_x, wy * self.zoom + self.offset_y

    def zoom_at(self, mouse_pos: Tuple[int, int], zoom_factor: float, min_zoom: float, max_zoom: float) -> None:
        mx, my = mouse_pos
        wx, wy = self.screen_to_world(mx, my)

        new_zoom = self.zoom * zoom_factor
        new_zoom = max(min_zoom, min(max_zoom, new_zoom))
        if abs(new_zoom - self.zoom) < 1e-9:
            return

        self.zoom = new_zoom
        self.offset_x = mx - wx * self.zoom
        self.offset_y = my - wy * self.zoom

def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def draw_grid(
    screen: pygame.Surface,
    model: NurikabeModel,
    camera: Camera,
    base_cell_size: int,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    highlight: Optional[Tuple[int, int]] = None,
    affected_cells: Optional[List[Tuple[int, int]]] = None
) -> None:
    rows, cols = model.rows, model.cols
    if rows == 0 or cols == 0:
        return

    cell_size = base_cell_size * camera.zoom
    if cell_size < 2:
        return

    sw, sh = screen.get_size()
    left = -cell_size
    top = -cell_size
    right = sw + cell_size
    bottom = sh + cell_size

    wl, wt = camera.screen_to_world(left, top)
    wr, wb = camera.screen_to_world(right, bottom)
    c0 = clamp_int(int(math.floor(wl / base_cell_size)), 0, cols - 1)
    r0 = clamp_int(int(math.floor(wt / base_cell_size)), 0, rows - 1)
    c1 = clamp_int(int(math.ceil(wr / base_cell_size)), 0, cols - 1)
    r1 = clamp_int(int(math.ceil(wb / base_cell_size)), 0, rows - 1)

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            wx = c * base_cell_size
            wy = r * base_cell_size
            sx, sy = camera.world_to_screen(wx, wy)
            rect = pygame.Rect(int(sx), int(sy), int(cell_size), int(cell_size))

            if model.is_clue(r, c):
                pygame.draw.rect(screen, grid_style.COLOR_CLUE, rect)
            else:
                if model.is_black_certain(r, c):
                    pygame.draw.rect(screen, grid_style.COLOR_BLACK, rect)
                elif model.is_land_certain(r, c):
                    pygame.draw.rect(screen, grid_style.COLOR_LAND, rect)
                else:
                    pygame.draw.rect(screen, grid_style.COLOR_UNKNOWN, rect)

            pygame.draw.rect(screen, grid_style.COLOR_GRID_LINES, rect, 1)

            if highlight is not None and (r, c) == highlight:
                pygame.draw.rect(screen, grid_style.COLOR_EDITOR_HIGHLIGHT, rect, 3)

            if affected_cells and (r, c) in affected_cells:
                pygame.draw.rect(screen, grid_style.COLOR_SOLVER_HIGHLIGHT, rect, 4)

            if model.is_clue(r, c):
                txt = str(model.clues[r][c])
                surf = font.render(txt, True, grid_style.COLOR_TEXT_CLUE)
                screen.blit(
                    surf,
                    (rect.x + (rect.width - surf.get_width()) // 2, rect.y + (rect.height - surf.get_height()) // 2)
                )
                if camera.zoom >= 1.0:
                    iid = model.island_by_pos.get((r, c))
                    if iid is not None:
                        id_surf = small_font.render(str(iid), True, grid_style.COLOR_TEXT_DEBUG)
                        screen.blit(id_surf, (rect.x + 3, rect.y + 2))
            else:
                if camera.zoom >= 1.0:
                    ids = model.bitset_to_ids(model.cells[r][c].owners)
                    if len(ids) == 0:
                        txt = "-"
                    elif len(ids) > 3:
                        txt = "*"
                    else:
                        txt = ",".join(str(x) for x in ids)
                    surf = small_font.render(txt, True, grid_style.COLOR_TEXT_DEBUG)
                    screen.blit(surf, (rect.x + 3, rect.y + 2))

def pick_cell_from_mouse(model: NurikabeModel, camera: Camera, base_cell_size: int, mouse_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    if model.rows == 0 or model.cols == 0:
        return None
    mx, my = mouse_pos
    wx, wy = camera.screen_to_world(mx, my)
    c = int(wx // base_cell_size)
    r = int(wy // base_cell_size)
    if 0 <= r < model.rows and 0 <= c < model.cols:
        return (r, c)
    return None
