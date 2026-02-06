import json
import argparse
import sys
from collections import Counter
from typing import Dict, Any, List

from nurikabe_model import NurikabeModel
from nurikabe_rules import NurikabeSolver

def serialize_grid(model: NurikabeModel) -> List[List[str]]:
    """Converts the final grid state into a list of strings for JSON serialization."""
    grid_state = []
    for r in range(model.rows):
        row_state = []
        for c in range(model.cols):
            if model.is_clue(r, c):
                row_state.append(f"CLUE({model.clues[r][c]})")
            elif model.is_black_certain(r, c):
                row_state.append("BLACK")
            elif model.is_land_certain(r, c):
                # Distinguish if it has an assigned owner or just land
                owners = model.bitset_to_ids(model.owners[r][c])
                if len(owners) == 1:
                    row_state.append(f"LAND({owners[0]})")
                else:
                    row_state.append("LAND(MULTIPLE)")
            else:
                row_state.append("UNKNOWN")
        grid_state.append(row_state)
    return grid_state

def run_solver(grid_path: str) -> Dict[str, Any]:
    """Loads a grid, runs the solver to completion, and returns the result stats."""
    model = NurikabeModel()
    
    try:
        with open(grid_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{grid_path}' not found.")
        sys.exit(1)

    success, msg = model.parse_puzzle_text(content)
    if not success:
        print(f"Error parsing grid: {msg}")
        sys.exit(1)

    solver = NurikabeSolver(model)
    
    rule_counts = Counter()
    steps_taken = 0
    
    while True:
        result = solver.step()
        
        # The solver returns rule="None" when no more rules can be applied
        if result.rule == "None":
            break
        
        rule_name = result.rule
        rule_counts[rule_name] += 1
        steps_taken += 1

    # Determine if solved (no unknowns left)
    is_solved = True
    for r in range(model.rows):
        for c in range(model.cols):
            if not model.is_black_certain(r, c) and not model.is_land_certain(r, c):
                is_solved = False
                break
        if not is_solved: break

    return {
        "rules_triggered": dict(rule_counts),
        "steps_total": steps_taken,
        "is_fully_solved": is_solved,
        "final_grid": serialize_grid(model)
    }

def generate_reference(grid_path: str):
    """Runs solver and saves the result as a reference JSON."""
    result = run_solver(grid_path)
    ref_path = grid_path + ".reference.json"
    
    with open(ref_path, 'w') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    
    print(f"Success: Reference generated and saved to '{ref_path}'")
    print(f"Solved: {result['is_fully_solved']}, Steps: {result['steps_total']}")

def check_regression(grid_path: str):
    """Runs solver and compares result with existing reference JSON."""
    current_result = run_solver(grid_path)
    ref_path = grid_path + ".reference.json"
    
    try:
        with open(ref_path, 'r') as f:
            reference_result = json.load(f)
    except FileNotFoundError:
        print(f"Error: Reference file '{ref_path}' not found. Run in 'generate' mode first.")
        sys.exit(1)
    
    # Comparison Logic
    # 1. Compare Final Grid (Critical)
    grid_match = current_result["final_grid"] == reference_result["final_grid"]
    
    # 2. Compare Rule Counts (Indicative of logic changes)
    rules_match = current_result["rules_triggered"] == reference_result["rules_triggered"]
    
    if grid_match and rules_match:
        print("TEST PASSED: Output matches reference exactly.")
    else:
        print("TEST FAILED: Output mismatch.")
        
        if not grid_match:
            print("CRITICAL: Final grid state differs!")
            # Basic diff output
            # (Could be improved to show specific cells)
        
        if not rules_match:
            print("WARNING: Rule usage counts differ (logic path changed).")
            print("Reference Rules:", reference_result["rules_triggered"])
            print("Current Rules:  ", current_result["rules_triggered"])
            
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nurikabe Solver Test Runner")
    parser.add_argument("grid_file", help="Path to the .nu.txt grid file")
    parser.add_argument("--mode", choices=["generate", "test"], default="test", 
                        help="Mode: 'generate' to create reference JSON, 'test' to compare against it.")
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        generate_reference(args.grid_file)
    else:
        check_regression(args.grid_file)
