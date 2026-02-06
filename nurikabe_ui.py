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
from typing import Tuple, Set
from nurikabe_model import NurikabeModel, StepResult, UNKNOWN, BLACK, LAND
from nurikabe_rules import NurikabeSolver

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
    solver = NurikabeSolver(model)
    
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
                solver.step()

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
                            model.cycle_manual_mark(r, c, forward=True)
                        elif event.button == 3:
                            model.cycle_manual_mark(r, c, forward=False)

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
