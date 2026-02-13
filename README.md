Use python 3.11+

Enable a pyenv of your preference

## Install dependencies
```bash
pip install -r requirement.txt
```

## Run the program 
```bash
# Default (V1)
python nurikabe_ui.py

# Using V2 Solver
python nurikabe_ui.py --model v2
```

## High level description of the code : 

1. `nurikabe_model.py` 
    - Defines the core data structures (`NurikabeModel`, `Island`, `StepResult`) and manages the grid state, including cell owners and validity checks.
1. `nurikabe_rules.py`
    - Implements the original `NurikabeSolver` (V1) with heuristic rules (G1-G10).
1. `nurikabe_rules_v2.py`
    - Implements `NurikabeSolverV2`, a more streamlined rule-based solver (B0, R0-R5).
1. `nurikabe_worker.py`
    - Handles the background solver thread (SolverWorker). Supports multiple solver classes.
1. `nurikabe_drawing.py`
    - Handles grid rendering (draw_grid).
    - Manages the camera/view transformation (Camera).
1. `nurikabe_ui.py`
    - Entry point. Manages UI state and events. Supports `--model [v1|v2]`.


## Unit test

Unit tests are generated based on rule applications seen during a solve.

### Generating Unit Tests
```bash
# Generate for V1 (output to unittest/)
python unittest/generate_rule_tests.py --model v1

# Generate for V2 (output to unittest_v2/)
python unittest/generate_rule_tests.py --model v2
```

### Running with Pytest
```bash
# Run V1 tests
pytest unittest/

# Run V2 tests
pytest unittest_v2/
```

### Generating Test Images

Side-by-side PNG images of "before" and "after" states for each generated unit test.

```bash
# For V1
python unittest/generate_test_images.py --model v1

# For V2
python unittest/generate_test_images.py --model v2
```

The images are saved in `unittest/RULE_NAME/images/` or `unittest_v2/RULE_NAME/images/`.


## Test Framework (Regression Testing)

The `tests/test.py` script performs automated regression testing against reference solutions.

```bash
# Run all tests for V1
python tests/test.py --mode test --model v1 -v tests/

# Run main_tests for V2 (uses .reference.v2.json)
python tests/test.py --mode test --model v2 -v tests/main_tests

# Run all test for V2 (uses .reference.v2.json)
python tests/test.py --mode test --model v2 -v tests/
```

### Individual Puzzle commands:
- **Generate Mode**: Saves the final state as a reference.
  ```bash
  python tests/test.py <grid_file> --model v2 --mode generate
  ```
- **Test Mode**: Compares output against the reference.
  ```bash
  python tests/test.py <grid_file> --model v2 --mode test
  ```


## Cell state
Summary of Representation from class CellState and OwnerMask
  ┌─────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────┐
  │ Condition                                               │ Representation in Cell                                        │
  ├─────────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────┤
  │ Possible States: Sea OR Island X                        │ state = CellState.UNKNOWN, owners = {Island X}                │
  │ Definitively Sea                                        │ state = CellState.SEA, owners = {} (empty)                    │
  │ Definitively Island X                                   │ state = CellState.LAND, owners = {Island X}                   │
  │ Definitively Land, Island Unknown                       │ state = CellState.LAND, owners = {Island X, Island Y, ...}    │
  │ Possible States: Sea OR {Island X, Island Y, ...}.      │ state = CellState.UNKNOWN, owners = {Island X, Island Y, ...} │
  └─────────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────┘

