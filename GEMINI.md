This is a nurikabe project, a famous game where you have to fill in land and see cells into a grid with clues. The project has a UI and a solver. 

The UI let's you both edit, load, save, and solve a nurikabe grid
The solver is not a standard solver. It aims at implementation small rules understandable by human.
Here is the overview of the code : 

1. `nurikabe_model.py` 
    - Defines the core data structures (`NurikabeModel`, `Island`, `StepResult`) and manages the grid state, including cell owners and validity checks.
1. `nurikabe_rules.py`
    - Implements the `NurikabeSolver` logic with various rule-based heuristics (e.g., connectivity, separation, global bottlenecks) to iteratively solve the puzzle.
1. `nurikabe_worker.py`
    - Handles the background solver thread (SolverWorker).
    - Contains worker-related data structures (WorkerCommand, WorkerResult).
1. `nurikabe_drawing.py`
    - Handles grid rendering (draw_grid).
    - Manages the camera/view transformation (Camera).
    - Contains coordinate utilities (pick_cell_from_mouse, clamp_int).
1. `nurikabe_ui.py` (Updated)
    - Remains the entry point (main).
    - Manages UI state (EditorState, MainState).
    - Handles events and Pygame GUI integration.
    - Imports necessary components from the new files.

