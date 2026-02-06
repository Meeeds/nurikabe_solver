"""
Nurikabe Assistant (Pygame)

Features:
- Editor Mode: Set grid size, click cells to type clues (numbers), Load to solver.
- Main Mode: Play manually (toggle cell states), Step (auto-solve), Reset.

Controls (Main):
- Left click: cycle Unknown -> Black -> Land -> Unknown
- Right click: cycle Unknown -> Land -> Black -> Unknown
- Buttons: Edit Grid, Step, Reset

Controls (Editor):
- Click "Rows" or "Cols" fields to type size.
- Click grid cells to select. Type numbers to set clues. 0/Backspace clears.
- Button: Load to Solver
"""

import pygame
import os
from typing import Tuple, Set, Optional, List
from nurikabe_model import NurikabeModel, StepResult, UNKNOWN, BLACK, LAND
from nurikabe_rules import NurikabeSolver

# ----------------------------
# UI Constants & Enums
# ----------------------------
MODE_MAIN = 0
MODE_EDITOR = 1

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


class NumberInput:
    def __init__(self, rect: pygame.Rect, initial_val: int) -> None:
        self.rect = rect
        self.text = str(initial_val)
        self.val = initial_val
        self.active = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Returns True if value changed."""
        changed = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
            else:
                self.active = False
        
        if self.active and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
                if not self.text:
                    self.val = 0
                else:
                    self.val = int(self.text)
                changed = True
            elif event.unicode.isdigit():
                if len(self.text) < 2: # Limit to 2 digits
                    self.text += event.unicode
                    self.val = int(self.text)
                    changed = True
        return changed

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        color = (255, 255, 255) if self.active else (230, 230, 230)
        pygame.draw.rect(screen, color, self.rect, border_radius=4)
        pygame.draw.rect(screen, (0, 0, 0) if self.active else (100, 100, 100), self.rect, 2, border_radius=4)
        surf = font.render(self.text, True, (0, 0, 0))
        # Center text
        text_rect = surf.get_rect(center=self.rect.center)
        screen.blit(surf, text_rect)

class TextInput:
    def __init__(self, rect: pygame.Rect, initial_text: str = "") -> None:
        self.rect = rect
        self.text = initial_text
        self.active = False

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
            else:
                self.active = False
        
        if self.active and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_RETURN:
                self.active = False # Deactivate on enter
            else:
                if event.unicode and len(self.text) < 30: # Limit length
                    self.text += event.unicode

    def draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        color = (255, 255, 255) if self.active else (230, 230, 230)
        pygame.draw.rect(screen, color, self.rect, border_radius=4)
        pygame.draw.rect(screen, (0, 0, 0) if self.active else (100, 100, 100), self.rect, 2, border_radius=4)
        
        # Render text with clipping if too long
        surf = font.render(self.text, True, (0, 0, 0))
        if surf.get_width() > self.rect.width - 10:
             # simple clipping: show end
             # Ideally we'd scroll, but this is simple
             pass 
        screen.blit(surf, (self.rect.x + 5, self.rect.y + (self.rect.height - surf.get_height()) // 2))

class EditorState:
    def __init__(self, rows: int = 10, cols: int = 10) -> None:
        self.rows = rows
        self.cols = cols
        self.clues = [[0 for _ in range(cols)] for _ in range(rows)]
        self.selected_cell: Optional[Tuple[int, int]] = None
        self.message = ""
        
    def resize(self, r: int, c: int) -> None:
        # Create new grid, copy old values where they fit
        new_clues = [[0 for _ in range(c)] for _ in range(r)]
        for i in range(min(self.rows, r)):
            for j in range(min(self.cols, c)):
                new_clues[i][j] = self.clues[i][j]
        self.rows = r
        self.cols = c
        self.clues = new_clues
        self.selected_cell = None

    def from_model(self, model: NurikabeModel) -> None:
        if model.rows > 0 and model.cols > 0:
            self.rows = model.rows
            self.cols = model.cols
            self.clues = [row[:] for row in model.clues]
        else:
            self.rows = 10
            self.cols = 10
            self.clues = [[0 for _ in range(10)] for _ in range(10)]
        self.selected_cell = None
        self.message = ""

# ----------------------------
# Helpers
# ----------------------------

def save_to_file(filename: str, state: EditorState) -> str:
    try:
        with open(filename, 'w') as f:
            for r in range(state.rows):
                line = []
                for c in range(state.cols):
                    val = state.clues[r][c]
                    if val == 0:
                        line.append(".")
                    else:
                        line.append(str(val))
                f.write(" ".join(line) + "\n")
        return f"Saved to {filename}"
    except Exception as e:
        return f"Error saving: {e}"

def load_from_file(filename: str, state: EditorState) -> str:
    if not os.path.exists(filename):
        return f"File not found: {filename}"
    try:
        with open(filename, 'r') as f:
            content = f.read()
            
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        if not lines:
            return "Empty file."
            
        grid = []
        for ln in lines:
            row = []
            tokens = ln.replace(".", "0").split()
            # If split didn't work (e.g. tight packing "..3.1"), handling that is complex here 
            # without regex or parser logic from model. 
            # Let's rely on model's parser logic logic by creating a temporary model?
            # Or just implementing basic space-sep parsing which our save produces.
            if len(tokens) == 0:
                # Try char by char?
                for ch in ln:
                    if ch == '.' or ch == '0': row.append(0)
                    elif ch.isdigit(): row.append(int(ch))
            else:
                for t in tokens:
                    if t == '.' or t == '0': row.append(0)
                    elif t.isdigit(): row.append(int(t))
            grid.append(row)
            
        if not grid: return "No data found."
        
        # Verify rectangle
        cols = len(grid[0])
        rows = len(grid)
        if any(len(r) != cols for r in grid):
            return "Ragged rows in file."
            
        state.rows = rows
        state.cols = cols
        state.clues = grid
        state.selected_cell = None
        return f"Loaded {filename}"
    except Exception as e:
        return f"Error loading: {e}"

# ----------------------------
# Rendering Helpers
# ----------------------------

def draw_main_grid(
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
                val = model.clues[r][c]
                surf = font.render(str(val), True, (0, 0, 0))
                screen.blit(surf, (rect.x + (cell_size - surf.get_width()) // 2,
                                   rect.y + (cell_size - surf.get_height()) // 2))
                iid = model.island_by_pos[(r, c)]
                id_surf = small_font.render(str(iid), True, (0, 0, 200))
                screen.blit(id_surf, (rect.x + 3, rect.y + 2))
            else:
                if not model.black_possible[r][c] and model.owners[r][c] != 0:
                    single = model.owners_singleton(r, c)
                    if single is not None:
                        surf = small_font.render(str(single), True, (0, 0, 120))
                        screen.blit(surf, (x + 3, y + 3))

def draw_editor_grid(
    screen: pygame.Surface,
    font: pygame.font.Font,
    state: EditorState,
    top_left: Tuple[int, int],
    cell_size: int
) -> None:
    ox, oy = top_left
    for r in range(state.rows):
        for c in range(state.cols):
            x = ox + c * cell_size
            y = oy + r * cell_size
            rect = pygame.Rect(x, y, cell_size, cell_size)
            
            is_selected = (state.selected_cell == (r, c))
            
            bg_color = (255, 255, 220) if is_selected else (255, 255, 255)
            pygame.draw.rect(screen, bg_color, rect)
            
            if is_selected:
                pygame.draw.rect(screen, (0, 0, 255), rect, 3)
            else:
                pygame.draw.rect(screen, (150, 150, 150), rect, 1)
            
            val = state.clues[r][c]
            if val > 0:
                surf = font.render(str(val), True, (0, 0, 0))
                screen.blit(surf, (rect.x + (cell_size - surf.get_width()) // 2,
                                   rect.y + (cell_size - surf.get_height()) // 2))

# ----------------------------
# Main app
# ----------------------------

def main() -> None:
    pygame.init()
    pygame.display.set_caption("Nurikabe Assistant")

    W, H = 1200, 800
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 20)
    small_font = pygame.font.SysFont("consolas", 14)
    big_font = pygame.font.SysFont("consolas", 24)

    # Application state
    mode = MODE_MAIN
    
    # Model
    model = NurikabeModel()
    # Initialize with default small puzzle
    

    model.parse_puzzle_text("2........2\n......2...\n.2..7.....\n..........\n......3.3.\n..2....3..\n2..4......\n..........\n.1....2.4.\n") 
    solver = NurikabeSolver(model)
    
    # Editor State
    editor_state = EditorState(10, 10)
    
    # UI Elements (Main)
    btn_edit_grid = Button(pygame.Rect(20, 20, 140, 40), "Setup Grid")
    btn_step = Button(pygame.Rect(180, 20, 130, 40), "Step")
    btn_reset = Button(pygame.Rect(330, 20, 160, 40), "Reset Domains")
    
    msg_rect = pygame.Rect(20, 80, 470, 600)
    
    # UI Elements (Editor)
    lbl_rows = font.render("Rows:", True, (0,0,0))
    inp_rows = NumberInput(pygame.Rect(80, 20, 50, 30), 10)
    
    lbl_cols = font.render("Cols:", True, (0,0,0))
    inp_cols = NumberInput(pygame.Rect(200, 20, 50, 30), 10)
    
    btn_editor_load = Button(pygame.Rect(300, 15, 180, 40), "Load to Solver")

    # File I/O
    lbl_file = font.render("File:", True, (0,0,0))
    inp_filename = TextInput(pygame.Rect(80, 70, 200, 30), "grid.txt")
    btn_save = Button(pygame.Rect(300, 70, 80, 30), "Save")
    btn_load_file = Button(pygame.Rect(390, 70, 80, 30), "Load")
    
    grid_origin = (510, 20)
    cell_size = 32

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        events = pygame.event.get()
        
        for event in events:
            if event.type == pygame.QUIT:
                running = False
        
        screen.fill((240, 240, 240))
        
        if mode == MODE_MAIN:
            # Event handling for MAIN
            for event in events:
                if btn_edit_grid.clicked(event):
                    mode = MODE_EDITOR
                    editor_state.from_model(model)
                    inp_rows.text = str(editor_state.rows)
                    inp_rows.val = editor_state.rows
                    inp_cols.text = str(editor_state.cols)
                    inp_cols.val = editor_state.cols
                    editor_state.message = ""
                    
                if btn_step.clicked(event):
                    solver.step()
                if btn_reset.clicked(event):
                    model.reset_domains_from_manual()
                    model.last_step = StepResult([], "Domains rebuilt.", "Reset")
                    
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # check grid clicks
                    gx, gy = grid_origin
                    mx, my = event.pos
                    if mx >= gx and my >= gy and model.rows > 0:
                        c = (mx - gx) // cell_size
                        r = (my - gy) // cell_size
                        if 0 <= r < model.rows and 0 <= c < model.cols:
                            if event.button == 1:
                                model.cycle_manual_mark(r, c, forward=True)
                            elif event.button == 3:
                                model.cycle_manual_mark(r, c, forward=False)

            # Draw MAIN
            btn_edit_grid.draw(screen, font, mouse_pos)
            btn_step.draw(screen, font, mouse_pos)
            btn_reset.draw(screen, font, mouse_pos)
            
            # Message box
            pygame.draw.rect(screen, (255, 255, 255), msg_rect, border_radius=8)
            pygame.draw.rect(screen, (100, 100, 100), msg_rect, 2, border_radius=8)
            screen.blit(big_font.render("Log / Info", True, (0,0,0)), (msg_rect.x+10, msg_rect.y+10))
            
            if model.last_step:
                rule_txt = font.render(f"Rule: {model.last_step.rule}", True, (0,0,150))
                screen.blit(rule_txt, (msg_rect.x+10, msg_rect.y+50))
                
                # word wrap msg
                y = msg_rect.y + 80
                words = model.last_step.message.split(' ')
                cur_line = ""
                for w in words:
                    test_line = cur_line + " " + w if cur_line else w
                    if font.size(test_line)[0] < msg_rect.width - 20:
                        cur_line = test_line
                    else:
                        screen.blit(font.render(cur_line, True, (0,0,0)), (msg_rect.x+10, y))
                        y += 25
                        cur_line = w
                if cur_line:
                    screen.blit(font.render(cur_line, True, (0,0,0)), (msg_rect.x+10, y))

            # Draw Grid
            highlight_cells = set(model.last_step.changed_cells) if model.last_step else set()
            draw_main_grid(screen, font, small_font, model, grid_origin, cell_size, highlight_cells)

        elif mode == MODE_EDITOR:
            # Event handling for EDITOR
            for event in events:
                r_changed = inp_rows.handle_event(event)
                c_changed = inp_cols.handle_event(event)
                inp_filename.handle_event(event)
                
                if r_changed or c_changed:
                    # Enforce limits 1-50
                    r = max(1, min(50, inp_rows.val))
                    c = max(1, min(50, inp_cols.val))
                    if r != editor_state.rows or c != editor_state.cols:
                        editor_state.resize(r, c)
                
                if btn_editor_load.clicked(event):
                    model.load_grid(editor_state.clues)
                    model.last_step = StepResult([], "Grid loaded from editor.", "Load")
                    mode = MODE_MAIN
                    solver = NurikabeSolver(model) # Re-init solver

                if btn_save.clicked(event):
                    msg = save_to_file(inp_filename.text, editor_state)
                    editor_state.message = msg

                if btn_load_file.clicked(event):
                    msg = load_from_file(inp_filename.text, editor_state)
                    editor_state.message = msg
                    if "Loaded" in msg:
                        # Update inputs
                        inp_rows.text = str(editor_state.rows)
                        inp_rows.val = editor_state.rows
                        inp_cols.text = str(editor_state.cols)
                        inp_cols.val = editor_state.cols
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    gx, gy = grid_origin
                    mx, my = event.pos
                    # Check grid click
                    if mx >= gx and my >= gy:
                        c = (mx - gx) // cell_size
                        r = (my - gy) // cell_size
                        if 0 <= r < editor_state.rows and 0 <= c < editor_state.cols:
                            editor_state.selected_cell = (r, c)
                        else:
                            editor_state.selected_cell = None
                    # If clicked outside input boxes and grid, deselect
                    elif not inp_rows.rect.collidepoint(event.pos) and not inp_cols.rect.collidepoint(event.pos) and not inp_filename.rect.collidepoint(event.pos):
                         editor_state.selected_cell = None
                         
                if event.type == pygame.KEYDOWN and editor_state.selected_cell and not inp_filename.active:
                    r, c = editor_state.selected_cell
                    if event.unicode.isdigit():
                        # Append digit? Or replace? Let's say we append unless 0. 
                        # To keep it simple: if 0, clear. If digit, append if < 99.
                        current_val = editor_state.clues[r][c]
                        new_digit = int(event.unicode)
                        if current_val == 0:
                            editor_state.clues[r][c] = new_digit
                        else:
                            combined = int(str(current_val) + str(new_digit))
                            if combined <= 999: # limit clue size
                                editor_state.clues[r][c] = combined
                    elif event.key == pygame.K_BACKSPACE:
                        current_val = editor_state.clues[r][c]
                        s = str(current_val)
                        if len(s) > 1:
                            editor_state.clues[r][c] = int(s[:-1])
                        else:
                            editor_state.clues[r][c] = 0
                    elif event.key == pygame.K_DELETE or event.key == pygame.K_0:
                        editor_state.clues[r][c] = 0

            # Draw EDITOR
            screen.blit(lbl_rows, (20, 25))
            inp_rows.draw(screen, font)
            
            screen.blit(lbl_cols, (140, 25))
            inp_cols.draw(screen, font)
            
            btn_editor_load.draw(screen, font, mouse_pos)
            
            screen.blit(lbl_file, (20, 75))
            inp_filename.draw(screen, font)
            btn_save.draw(screen, font, mouse_pos)
            btn_load_file.draw(screen, font, mouse_pos)

            if editor_state.message:
                msg_surf = font.render(editor_state.message, True, (200, 0, 0))
                screen.blit(msg_surf, (20, 110))
            
            info_txt = font.render("Editor Mode: Click cell to type number. 0 or Backspace to clear.", True, (50, 50, 50))
            screen.blit(info_txt, (20, 150))
            
            draw_editor_grid(screen, font, editor_state, grid_origin, cell_size)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
