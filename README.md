Use python 3.11+

Enable a pyenv of your preference

## Install dependencies
```bash
pip install -r requirement.txt
```

## Run the program 
```bash
python nurikabe_ui.py
```

## High level description of the code : 

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


## Unit test

Unit test has been first generated using `unittest/generate_rule_tests.py`

How to run them
```
pytest
pytest unittest/test_rule_G1b_Separation__match_neighbor_owners.py 
```

### Generating Test Images

The `unittest/generate_test_images.py` script creates visual representations of the "before" and "after" states for each generated unit test. This is useful for quickly understanding the effect of each rule application. The "after" image highlights the cells that were changed by the rule.

To generate these images:

```bash
python unittest/generate_test_images.py
```

The images will be saved in the `unittest/images/` directory.


## Test Framework

The `test.py` script allows for automated regression testing of the solver:

Run all tests : 
```
python tests/test.py --all
```

Run one Test : 
- **Generate Mode**: Runs the solver on a grid and saves the final state and rule counts as a reference JSON.
  ```bash
  python tests/test.py <grid_file> --mode generate
  ```
- **Test Mode**: Compares the solver's current output against an existing reference JSON to detect regressions.
  ```bash
  python tests/test.py <grid_file> --mode test
  ```
