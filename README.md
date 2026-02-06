Use python 3.11+

Enable a pyenv of your preference

# Install dependencies
```bash
pip install -r requirement.txt
```

# Run the program 
```bash
python nurikabe_ui.py
```

# High level description of the code : 

- **nurikabe_model.py**: Defines the core data structures (`NurikabeModel`, `Island`, `StepResult`) and manages the grid state, including cell owners and validity checks.
- **nurikabe_rules.py**: Implements the `NurikabeSolver` logic with various rule-based heuristics (e.g., connectivity, separation, global bottlenecks) to iteratively solve the puzzle.
- **nurikabe_ui.py**: Provides an interactive Pygame interface for editing grids, visualizing the solver's progress step-by-step, and debugging specific logic like island extensions.

# Test Framework

The `test.py` script allows for automated regression testing of the solver:

- **Generate Mode**: Runs the solver on a grid and saves the final state and rule counts as a reference JSON.
  ```bash
  python test.py <grid_file> --mode generate
  ```
- **Test Mode**: Compares the solver's current output against an existing reference JSON to detect regressions.
  ```bash
  python test.py <grid_file> --mode test
  ```
