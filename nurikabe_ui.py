"""
Nurikabe Assistant (Pygame)

Features:
- Editor Mode: Set grid size, click cells to type clues (numbers), Load to solver.
- Main Mode: Play manually (toggle cell states), Step (auto-solve), Reset.

Controls (Main):
- Left click: cycle Unknown -> Land -> Black -> Unknown
- Right click: Display debug info (owners/black possible)
- Buttons: Edit Grid, Step, Reset

Controls (Editor):
- Click "Rows" or "Cols" fields to type size.
- Click grid cells to select. Type numbers to set clues. 0/Backspace clears.
- Button: Load to Solver
"""

import os
import glob
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any

import pygame
import pygame_gui

from nurikabe_model import NurikabeModel
from nurikabe_rules import NurikabeSolver
from nurikabe_worker import SolverWorker, WorkerCommand
from nurikabe_drawing import Camera, draw_grid, pick_cell_from_mouse, clamp_int
import grid_style


# ----------------------------
# UI Constants & Enums
# ----------------------------
MODE_MAIN = 0
MODE_EDITOR = 1

DRAG_THRESHOLD_PX = 6


# ----------------------------
# App state
# ----------------------------

@dataclass
class EditorState:
    rows: int = 7
    cols: int = 7
    grid: Optional[List[List[int]]] = None
    selected: Optional[Tuple[int, int]] = None
    message: str = ""
    puzzle_files: Optional[List[str]] = None
    selected_file: Optional[str] = None
    
    # Tree view state
    tree_roots: List['TreeNode'] = field(default_factory=list)
    visible_nodes: List['TreeNode'] = field(default_factory=list)

@dataclass
class TreeNode:
    name: str
    path: str
    is_dir: bool
    children: List['TreeNode'] = field(default_factory=list)
    expanded: bool = False
    level: int = 0

@dataclass
class MainState:
    mode: int = MODE_MAIN
    model: NurikabeModel = NurikabeModel()
    solver: NurikabeSolver = NurikabeSolver(NurikabeModel())
    debug_cell: Optional[Tuple[int, int]] = None
    last_step_msg: str = ""
    affected_cells: List[Tuple[int, int]] = field(default_factory=list)


# ----------------------------
# Helpers
# ----------------------------

def build_tree_from_dir(path: str, level: int = 0) -> Optional[TreeNode]:
    if not os.path.isdir(path):
        return None
    
    node = TreeNode(name=os.path.basename(path), path=path, is_dir=True, level=level, expanded=False)
    
    try:
        entries = sorted(os.listdir(path))
        dirs = []
        files = []
        for entry in entries:
            if entry.startswith('.'): 
                continue
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                dirs.append(entry)
            elif entry.endswith('.txt'):
                files.append(entry)
        
        for d in dirs:
            child = build_tree_from_dir(os.path.join(path, d), level + 1)
            if child and (child.children or any(f.endswith('.txt') for f in os.listdir(child.path))):
                node.children.append(child)
        for f in files:
            node.children.append(TreeNode(name=f, path=os.path.join(path, f), is_dir=False, level=level+1))
    except OSError:
        pass
        
    return node

def refresh_file_list(editor: EditorState, files_list_ui: pygame_gui.elements.UISelectionList) -> None:
    """Flattens the tree and updates the UI list."""
    items = []
    nodes = []
    
    # Helper to traverse
    def traverse(node_list: List[TreeNode]):
        for node in node_list:
            # Create display string
            indent = "    " * node.level
            icon = ("[-] " if node.expanded else "[+] ") if node.is_dir else "    "
            
            display_text = f"{indent}{icon}{node.name}"
            # Minimal zero-width spaces for uniqueness
            while display_text in items:
                display_text += "\u200b"
            
            items.append(display_text)
            nodes.append(node)
            
            if node.is_dir and node.expanded:
                traverse(node.children)

    traverse(editor.tree_roots)
    
    editor.visible_nodes = nodes
    files_list_ui.set_item_list(items)

def load_puzzle_files_to_tree() -> List[TreeNode]:
    roots = []
    for p in ["puzzles", "tests"]:
        if os.path.exists(p):
            t = build_tree_from_dir(p)
            if t:
                # Default expand top level
                t.expanded = True 
                roots.append(t)
    return roots

def grid_default(rows: int, cols: int) -> List[List[int]]:
    return [[0 for _ in range(cols)] for _ in range(rows)]


def cycle_mark(model: NurikabeModel, r: int, c: int) -> None:
    model.cycle_state(r, c)


def format_debug_cell(model: NurikabeModel, r: int, c: int) -> str:
    if not model.in_bounds(r, c):
        return "Out of bounds."
    if model.is_clue(r, c):
        return f"Cell ({r},{c}) clue={model.clues[r][c]} island_id={model.island_by_pos.get((r,c), '-')}"
    cell = model.cells[r][c]
    owners_bits = cell.owners
    owners_ids = model.bitset_to_ids(owners_bits)
    return (
        f"Cell ({r},{c}) state={cell.state.name}\n"
        f"owners_bits={owners_bits} owners_ids={owners_ids}\n"
        f"is_black={cell.is_black} is_land={cell.is_land}"
    )


def html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&#39;")
    )

def find_first_file(nodes: List[TreeNode]) -> Optional[str]:
    for node in nodes:
        if not node.is_dir:
            return node.path
        res = find_first_file(node.children)
        if res:
            return res
    return None

def update_selected_cell_info(editor: EditorState, lbl_selected_cell_info: pygame_gui.elements.UILabel) -> None:
    if editor.selected is not None and editor.grid is not None:
        r, c = editor.selected
        if 0 <= r < editor.rows and 0 <= c < editor.cols:
            value = editor.grid[r][c]
            lbl_selected_cell_info.set_text(f"Selected: ({r},{c}) Value: {value}")
        else:
            lbl_selected_cell_info.set_text("Selected: Invalid cell")
    else:
        lbl_selected_cell_info.set_text("Selected: None")

def save_editor_grid_to_file(
    editor: EditorState,
    filename: str,
    log_append: Any, # type: ignore
    files_list: pygame_gui.elements.UISelectionList
) -> None:
    if not filename:
        log_append("Save failed: Filename cannot be empty.")
        return

    filepath = os.path.join("puzzles", filename)

    if editor.grid is None:
        log_append("Save failed: Editor grid is empty.")
        return

    try:
        grid_str = []
        for r in range(editor.rows):
            grid_str.append(" ".join(str(editor.grid[r][c]) for c in range(editor.cols)))
        content = "\n".join(grid_str)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        log_append(f"Grid saved to {filepath}")

        # Reload tree and update the list
        editor.tree_roots = load_puzzle_files_to_tree()
        refresh_file_list(editor, files_list)

    except Exception as e:
        log_append(f"Save failed: {e}")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    pygame.init()
    pygame.display.set_caption("Nurikabe Assistant")

    screen = pygame.display.set_mode((1200, 800), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("arial", 20)
    small_font = pygame.font.SysFont("arial", 14)

    ui_manager = pygame_gui.UIManager(screen.get_size())

    ui_manager.get_theme().load_theme({
        "selection_list.#item_list_item": {
            "misc": {
                "text_horiz_alignment": "left"
            }
        }
    })


    controls_win = pygame_gui.elements.UIWindow(
        pygame.Rect(20, 20, 260, 320),
        ui_manager,
        window_display_title="Controls",
        resizable=True
    )
    controls_win.close_window_button.hide()
    log_win = pygame_gui.elements.UIWindow(
        pygame.Rect(20, 360, 520, 360),
        ui_manager,
        window_display_title="Log",
        resizable=True
    )
    log_win.close_window_button.hide()
    editor_win = pygame_gui.elements.UIWindow(
        pygame.Rect(300, 20, 340, 600),
        ui_manager,
        window_display_title="Editor",
        visible=False,
        resizable=True
    )
    editor_win.close_window_button.hide()

    controls_win.set_minimum_dimensions((260, 320))
    log_win.set_minimum_dimensions((520, 260))
    editor_win.set_minimum_dimensions((340, 600))

    btn_edit = pygame_gui.elements.UIButton(pygame.Rect(10, 10, 240, 36), "Edit Grid", ui_manager, container=controls_win)
    btn_step = pygame_gui.elements.UIButton(pygame.Rect(10, 56, 240, 36), "Step", ui_manager, container=controls_win)
    btn_next_cell = pygame_gui.elements.UIButton(pygame.Rect(10, 102, 240, 36), "Next Cell", ui_manager, container=controls_win)
    btn_10steps = pygame_gui.elements.UIButton(pygame.Rect(10, 148, 240, 36), "10Steps", ui_manager, container=controls_win)
    btn_reset = pygame_gui.elements.UIButton(pygame.Rect(10, 194, 240, 36), "Reset", ui_manager, container=controls_win)

    lbl_hint = pygame_gui.elements.UILabel(
        pygame.Rect(10, 240, 240, 60),
        "Pan: drag with LMB\nZoom: mouse wheel\nClick: release without drag",
        ui_manager,
        container=controls_win
    )

    log_box = pygame_gui.elements.UITextBox(
        html_text="",
        relative_rect=pygame.Rect(10, 10, 500, 260),
        manager=ui_manager,
        container=log_win,
        anchors={"left": "left", "right": "right", "top": "top", "bottom": "bottom"}
    )
    btn_clear_log = pygame_gui.elements.UIButton(
        pygame.Rect(10, -40, 120, 30),
        "Clear",
        ui_manager,
        container=log_win,
        anchors={"left": "left", "bottom": "bottom"}
    )

    lbl_rows = pygame_gui.elements.UILabel(pygame.Rect(10, 10, 60, 26), "Rows:", ui_manager, container=editor_win)
    inp_rows = pygame_gui.elements.UITextEntryLine(pygame.Rect(70, 10, 80, 26), ui_manager, container=editor_win)
    inp_rows.set_text("7")
    lbl_cols = pygame_gui.elements.UILabel(pygame.Rect(160, 10, 60, 26), "Cols:", ui_manager, container=editor_win)
    inp_cols = pygame_gui.elements.UITextEntryLine(pygame.Rect(220, 10, 80, 26), ui_manager, container=editor_win)
    inp_cols.set_text("7")

    btn_new = pygame_gui.elements.UIButton(pygame.Rect(10, 46, 100, 32), "New Grid", ui_manager, container=editor_win)
    btn_clear_clues = pygame_gui.elements.UIButton(pygame.Rect(115, 46, 100, 32), "Clear Clues", ui_manager, container=editor_win)
    btn_load_to_solver = pygame_gui.elements.UIButton(pygame.Rect(220, 46, 100, 32), "Load To Solver", ui_manager, container=editor_win)

    lbl_selected_cell_info = pygame_gui.elements.UILabel(
        pygame.Rect(10, 88, 310, 24),
        "Selected: None",
        ui_manager,
        container=editor_win
    )

    lbl_files = pygame_gui.elements.UILabel(pygame.Rect(10, 122, 290, 24), "Puzzle files (click to load):", ui_manager, container=editor_win)
    files_list = pygame_gui.elements.UISelectionList(
        relative_rect=pygame.Rect(10, 150, 310, 200),
        item_list=[],
        manager=ui_manager,
        container=editor_win,
        anchors={"left": "left", "right": "right", "top": "top"}
    )
    inp_filename = pygame_gui.elements.UITextEntryLine(
        pygame.Rect(10, 360, 310, 32),
        ui_manager,
        container=editor_win,
        anchors={"left": "left", "right": "right", "top": "top"}
    )
    inp_filename.set_text("new_puzzle.nu.txt")
    btn_save = pygame_gui.elements.UIButton(
        pygame.Rect(10, 396, 310, 32),
        "Save Puzzle",
        ui_manager,
        container=editor_win,
        anchors={"left": "left", "right": "right", "top": "top"}
    )
    btn_close_editor = pygame_gui.elements.UIButton(
        pygame.Rect(10, 432, 310, 32),
        "Close Editor",
        ui_manager,
        container=editor_win,
        anchors={"left": "left", "right": "right", "top": "top"}
    )

    state = MainState()
    state.model = NurikabeModel()
    state.solver = NurikabeSolver(state.model)

    editor = EditorState()
    editor.grid = grid_default(editor.rows, editor.cols)
    # Load Tree
    editor.tree_roots = load_puzzle_files_to_tree()
    refresh_file_list(editor, files_list)

    camera = Camera()
    base_cell_size = 48

    def center_camera_on_model(model: NurikabeModel) -> None:
        if model.rows == 0 or model.cols == 0:
            return
        sw, sh = screen.get_size()
        grid_w = model.cols * base_cell_size
        grid_h = model.rows * base_cell_size
        camera.zoom = 1.0
        camera.offset_x = (sw - grid_w) * 0.5
        camera.offset_y = (sh - grid_h) * 0.5

    log_lines: List[str] = []

    def log_append(msg: str) -> None:
        if not msg:
            return
        
        # Force wrapping by ensuring spaces around common delimiters
        for char in [',', ';', ']', ')', '}', ':', '>', '[', '(', '{', '<', '=', '|']:
            msg = msg.replace(char, ' ' + char + ' ')
        
        # Handle multi-line messages
        for line in msg.splitlines():
            line = line.strip()
            if not line:
                continue
                
            # Split by any whitespace and rejoin to normalize
            words = line.split()
            processed_words = []
            for word in words:
                # If a word is extremely long (no spaces), force a break by inserting a space.
                # Threshold reduced to 30 for safer wrapping in the UI.
                while len(word) > 30:
                    processed_words.append(word[:30])
                    word = word[30:]
                if word:
                    processed_words.append(word)
            
            log_lines.append(' '.join(processed_words))
        
        # Keep a reasonable history
        max_log_lines = 100
        if len(log_lines) > max_log_lines:
            del log_lines[0:len(log_lines) - max_log_lines]
            
        html = "<br>".join(html_escape(ln) for ln in log_lines)
        log_box.set_text(html)
        
        # Auto-scroll to bottom
        if log_box.scroll_bar is not None:
            log_box.scroll_bar.set_scroll_from_start_percentage(1.0)

    def log_clear() -> None:
        log_lines.clear()
        log_box.set_text("")

    worker = SolverWorker()
    worker.start()

    undo_stack: List[Dict[str, object]] = []
    max_undo = 200

    def push_undo() -> None:
        undo_stack.append(state.model.snapshot())
        if len(undo_stack) > max_undo:
            del undo_stack[:20]

    def sync_worker() -> None:
        worker.send(WorkerCommand(kind="sync_state", payload={"state": state.model.snapshot()}))
        log_append("Synced UI state to worker.")

    def apply_worker_state(snap: Dict[str, object]) -> None:
        try:
            state.model.restore(snap)
        except Exception as e:
            log_append(f"UI restore failed: {e}")

    def is_over_ui(pos: Tuple[int, int]) -> bool:
        for w in (controls_win, log_win, editor_win):
            if w.visible and w.get_abs_rect().collidepoint(pos):
                return True
        return False

    # Auto-load the first puzzle found in the tree
    first_puzzle = find_first_file(editor.tree_roots)
    if first_puzzle:
        try:
            with open(first_puzzle, "r", encoding="utf-8") as f:
                txt = f.read()
            tmp = NurikabeModel()
            ok, msg = tmp.parse_puzzle_text(txt)
            if ok:
                editor.grid = [row[:] for row in tmp.clues]
                editor.rows = tmp.rows
                editor.cols = tmp.cols
                inp_rows.set_text(str(editor.rows))
                inp_cols.set_text(str(editor.cols))
                log_append(f"Auto-loaded: {first_puzzle}")
            else:
                log_append(f"Auto-load failed: {msg}")
        except Exception as e:
            log_append(f"Auto-load failed: {e}")

    state.model.load_grid(editor.grid)
    state.solver = NurikabeSolver(state.model)
    center_camera_on_model(state.model)
    sync_worker()
    log_append("Ready.")
    update_selected_cell_info(editor, lbl_selected_cell_info)

    panning = False
    pan_last: Optional[Tuple[int, int]] = None
    lmb_down_pos: Optional[Tuple[int, int]] = None
    lmb_dragging = False
    lmb_down_over_ui = False
    lmb_down_mode = MODE_MAIN

    model_for_editor = NurikabeModel()

    running = True
    while running:
        time_delta = clock.tick(60) / 1000.0

        while True:
            res = worker.try_recv()
            if res is None:
                break
            if res.kind == "error":
                log_append(res.payload.get("message", "Worker error."))
                continue
            if "state" in res.payload:
                snap = res.payload["state"]
                if isinstance(snap, dict):
                    apply_worker_state(snap)
            step = res.payload.get("step_result")
            if isinstance(step, dict):
                msg = step.get("message", "")
                rule = step.get("rule", "")
                if msg or rule:
                    log_append(f"[{rule}] {msg}".strip())
                state.affected_cells = step.get("changed_cells", [])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                ui_manager.set_window_resolution(event.size)

            ui_manager.process_events(event)

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == btn_clear_log:
                    log_clear()

                elif event.ui_element == btn_edit:
                    editor_win.show()
                    editor_win.close_window_button.hide()
                    state.mode = MODE_EDITOR
                    update_selected_cell_info(editor, lbl_selected_cell_info)
                    log_append("Editor opened.")

                elif event.ui_element == btn_close_editor:
                    editor_win.hide()
                    state.mode = MODE_MAIN
                    log_append("Editor closed.")

                elif event.ui_element == btn_new:
                    try:
                        r = int(inp_rows.get_text().strip())
                        c = int(inp_cols.get_text().strip())
                        r = clamp_int(r, 1, 60)
                        c = clamp_int(c, 1, 60)
                        editor.rows, editor.cols = r, c
                        editor.grid = grid_default(r, c)
                        editor.selected = None
                        model_for_editor.load_grid(editor.grid)
                        center_camera_on_model(model_for_editor)
                        update_selected_cell_info(editor, lbl_selected_cell_info)
                        log_append(f"New grid {r}x{c}.")
                    except ValueError:
                        log_append("Invalid rows/cols.")

                elif event.ui_element == btn_clear_clues:
                    if editor.grid is not None:
                        for r_idx in range(editor.rows):
                            for c_idx in range(editor.cols):
                                editor.grid[r_idx][c_idx] = 0
                        editor.selected = None
                        update_selected_cell_info(editor, lbl_selected_cell_info)
                        log_append("All clues cleared in editor grid.")
                    else:
                        log_append("No editor grid to clear.")

                elif event.ui_element == btn_load_to_solver:
                    if editor.grid is not None:
                        push_undo()
                        state.model.load_grid(editor.grid)
                        state.solver = NurikabeSolver(state.model)
                        center_camera_on_model(state.model)
                        sync_worker()
                        log_append("Grid loaded into solver.")

                        # Close editor
                        editor_win.hide()
                        state.mode = MODE_MAIN
                        log_append("Editor closed.")
                    else:
                        log_append("No editor grid.")

                elif event.ui_element == btn_save:
                    save_editor_grid_to_file(editor, inp_filename.get_text(), log_append, files_list)

                elif event.ui_element == btn_step:
                    push_undo()
                    worker.send(WorkerCommand(kind="step"))

                elif event.ui_element == btn_next_cell:
                    push_undo()
                    worker.send(WorkerCommand(kind="next_cell"))

                elif event.ui_element == btn_10steps:
                    push_undo()
                    for _ in range(10):
                        worker.send(WorkerCommand(kind="step"))

                elif event.ui_element == btn_reset:
                    if undo_stack:
                        snap0 = undo_stack[0]
                        undo_stack.clear()
                        apply_worker_state(snap0)
                        state.solver = NurikabeSolver(state.model)
                        state.affected_cells.clear()
                        sync_worker()
                        log_append("Reset to first undo snapshot.")
                    else:
                        log_append("Nothing to reset (undo empty).")

                else:
                    # Check if it's a button in our files_list
                    idx = -1
                    # item_list property returns the list of item data
                    for i, item in enumerate(files_list.item_list):
                        if item['button_element'] == event.ui_element:
                            idx = i
                            break
                    
                    if idx != -1 and idx < len(editor.visible_nodes):
                        node = editor.visible_nodes[idx]
                        
                        if node.is_dir:
                            node.expanded = not node.expanded
                            refresh_file_list(editor, files_list)
                        else:
                            # Preview on click
                            try:
                                with open(node.path, "r", encoding="utf-8") as f:
                                    txt = f.read()
                                tmp = NurikabeModel()
                                ok, msg = tmp.parse_puzzle_text(txt)
                                if ok:
                                    editor.grid = [row[:] for row in tmp.clues]
                                    editor.rows = tmp.rows
                                    editor.cols = tmp.cols
                                    inp_rows.set_text(str(editor.rows))
                                    inp_cols.set_text(str(editor.cols))
                                    model_for_editor.load_grid(editor.grid)
                                    center_camera_on_model(model_for_editor)
                                    update_selected_cell_info(editor, lbl_selected_cell_info)
                                    log_append(f"Previewed: {node.name}")
                                else:
                                    log_append(f"Preview failed: {msg}")
                            except Exception as e:
                                log_append(f"File read failed: {e}")

            if event.type == pygame.MOUSEWHEEL:
                if not is_over_ui(pygame.mouse.get_pos()):
                    if event.y > 0:
                        camera.zoom_at(pygame.mouse.get_pos(), 1.1, 0.2, 6.0)
                    elif event.y < 0:
                        camera.zoom_at(pygame.mouse.get_pos(), 1.0 / 1.1, 0.2, 6.0)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 2:
                    if not is_over_ui(event.pos):
                        panning = True
                        pan_last = event.pos

                if event.button == 1:
                    lmb_down_pos = event.pos
                    lmb_dragging = False
                    lmb_down_over_ui = is_over_ui(event.pos)
                    lmb_down_mode = state.mode
                    if not lmb_down_over_ui:
                        pan_last = event.pos
                        panning = False

                if event.button == 3:
                    if not is_over_ui(event.pos):
                        cell = pick_cell_from_mouse(state.model, camera, base_cell_size, event.pos)
                        if cell is not None:
                            r, c = cell
                            log_append(format_debug_cell(state.model, r, c))

            if event.type == pygame.MOUSEMOTION:
                if panning and pan_last is not None:
                    mx, my = event.pos
                    lx, ly = pan_last
                    dx, dy = mx - lx, my - ly
                    camera.offset_x += dx
                    camera.offset_y += dy
                    pan_last = event.pos

                if lmb_down_pos is not None and not lmb_down_over_ui:
                    mx, my = event.pos
                    sx, sy = lmb_down_pos
                    if not lmb_dragging:
                        if abs(mx - sx) >= DRAG_THRESHOLD_PX or abs(my - sy) >= DRAG_THRESHOLD_PX:
                            lmb_dragging = True
                            panning = True
                            pan_last = event.pos
                    else:
                        if panning and pan_last is not None:
                            lx, ly = pan_last
                            dx, dy = mx - lx, my - ly
                            camera.offset_x += dx
                            camera.offset_y += dy
                            pan_last = event.pos

            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    panning = False
                    pan_last = None

                if event.button == 1:
                    if lmb_down_pos is not None and not lmb_down_over_ui:
                        if not lmb_dragging:
                            if lmb_down_mode == MODE_MAIN:
                                cell = pick_cell_from_mouse(state.model, camera, base_cell_size, event.pos)
                                if cell is not None:
                                    r, c = cell
                                    push_undo()
                                    cycle_mark(state.model, r, c)
                                    state.affected_cells.clear()
                                    sync_worker()
                            else:
                                if editor.grid is not None:
                                    model_for_editor.load_grid(editor.grid)
                                    cell = pick_cell_from_mouse(model_for_editor, camera, base_cell_size, event.pos)
                                    if cell is not None:
                                        editor.selected = cell
                                        update_selected_cell_info(editor, lbl_selected_cell_info)

                    lmb_down_pos = None
                    lmb_dragging = False
                    lmb_down_over_ui = False
                    panning = False
                    pan_last = None

            if event.type == pygame.KEYDOWN and state.mode == MODE_EDITOR:
                if editor.selected is not None and editor.grid is not None:
                    r, c = editor.selected
                    if 0 <= r < editor.rows and 0 <= c < editor.cols:
                        if event.key in (pygame.K_BACKSPACE, pygame.K_DELETE, pygame.K_0):
                            editor.grid[r][c] = 0
                            update_selected_cell_info(editor, lbl_selected_cell_info)
                        else:
                            ch = event.unicode
                            if ch.isdigit():
                                cur = editor.grid[r][c]
                                d = int(ch)
                                nxt = cur * 10 + d
                                nxt = clamp_int(nxt, 0, 999)
                                editor.grid[r][c] = nxt
                                update_selected_cell_info(editor, lbl_selected_cell_info)

        if state.mode == MODE_EDITOR:
            btn_step.disable()
            btn_next_cell.disable()
            btn_10steps.disable()
            btn_reset.disable()
        else:
            btn_step.enable()
            btn_next_cell.enable()
            btn_10steps.enable()
            btn_reset.enable()

        ui_manager.update(time_delta)

        screen.fill(grid_style.COLOR_BG)

        highlight_cell = None
        if state.mode == MODE_EDITOR:
            highlight_cell = editor.selected
            if editor.grid is not None:
                model_for_editor.load_grid(editor.grid)
            draw_grid(screen, model_for_editor, camera, base_cell_size, font, small_font, highlight=highlight_cell)
        else:
            draw_grid(
                screen, state.model, camera, base_cell_size, font, small_font,
                highlight=highlight_cell,
                affected_cells=state.affected_cells
            )

        ui_manager.draw_ui(screen)

        if state.mode == MODE_EDITOR:
            help_surf = small_font.render("Editor: click a cell then type digits; Backspace/0 clears.", True, (220, 220, 220))
            screen.blit(help_surf, (screen.get_width() - help_surf.get_width() - 12, 12))

        pygame.display.flip()

    worker.stop()
    pygame.quit()


if __name__ == "__main__":
    main()