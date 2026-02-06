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