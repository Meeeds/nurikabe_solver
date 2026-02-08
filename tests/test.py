import os
import json
import argparse
import sys
from collections import Counter
from typing import Dict, Any, List

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
                owners = model.bitset_to_ids(model.cells[r][c].owners)
                if len(owners) == 1:
                    row_state.append(f"LAND({owners[0]})")
                else:
                    row_state.append("LAND(MULTIPLE)")
            else:
                row_state.append("UNKNOWN")
        grid_state.append(row_state)
    return grid_state

def run_solver(grid_path: str) -> tuple[Dict[str, Any] | None, str | None]:
    """Loads a grid, runs the solver to completion, and returns the result stats."""
    model = NurikabeModel()
    
    try:
        with open(grid_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return None, f"Error: File '{grid_path}' not found."

    success, msg = model.parse_puzzle_text(content)
    if not success:
        return None, f"Error parsing grid: {msg}"

    solver = NurikabeSolver(model)
    
    rule_counts = Counter()
    steps_taken = 0
    
    while True:
        result = solver.step()
        print(f"Step {steps_taken + 1}: Rule applied: {result.rule}, Message: {result.message}, Changed Cells: {len(result.changed_cells)}")
        
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
    }, None

def generate_reference(grid_path: str) -> bool:
    """Runs solver and saves the result as a reference JSON."""
    result, error_msg = run_solver(grid_path)
    if error_msg:
        print(f"Error for {grid_path}: {error_msg}")
        return False
        
    ref_path = grid_path + ".reference.json"
    
    with open(ref_path, 'w') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    
    print(f"Success: Reference generated for '{grid_path}' and saved to '{ref_path}'")
    print(f"Solved: {result['is_fully_solved']}, Steps: {result['steps_total']}")
    return True

def check_regression(grid_path: str) -> bool:
    """Runs solver and compares result with existing reference JSON."""
    current_result, error_msg = run_solver(grid_path)
    if error_msg:
        print(f"Error for {grid_path}: {error_msg}")
        return False

    ref_path = grid_path + ".reference.json"
    
    try:
        with open(ref_path, 'r') as f:
            reference_result = json.load(f)
    except FileNotFoundError:
        print(f"Error for '{grid_path}': Reference file '{ref_path}' not found. Run in 'generate' mode first.")
        return False
    except json.JSONDecodeError:
        print(f"Error for '{grid_path}': Reference file '{ref_path}' is not a valid JSON.")
        return False
    
    # Comparison Logic
    grid_match = current_result["final_grid"] == reference_result["final_grid"]
    rules_match = current_result["rules_triggered"] == reference_result["rules_triggered"]
    
    if grid_match and rules_match:
        print(f"TEST PASSED: '{grid_path}' matches reference exactly.")
        return True
    else:
        print(f"TEST FAILED: '{grid_path}' output mismatch.")
        
        if not grid_match:
            print("  CRITICAL: Final grid state differs!")
            # Basic diff output
        
        if not rules_match:
            print("  WARNING: Rule usage counts differ (logic path changed).")
            print("    Reference Rules:", reference_result["rules_triggered"])
            print("    Current Rules:  ", current_result["rules_triggered"])
            
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nurikabe Solver Test Runner")
    parser.add_argument("grid_file", nargs='?', help="Path to the .nu.txt grid file (optional if --all is used)")
    parser.add_argument("--mode", choices=["generate", "test"], default="test", 
                        help="Mode: 'generate' to create reference JSON, 'test' to compare against it.")
    parser.add_argument("--all", action="store_true", 
                        help="Run all tests on *.nu.txt files in the test directory.")
    
    args = parser.parse_args()

    if args.all:
        if args.grid_file:
            print("Error: Cannot use --all and specify a grid_file simultaneously.")
            sys.exit(1)
        test_dir = os.path.dirname(os.path.abspath(__file__))
        test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".nu.txt")]
        
        if not test_files:
            print(f"No .nu.txt files found in the test directory: {test_dir}")
            sys.exit(0)

        all_tests_passed = True
        print(f"Running all tests in {args.mode} mode for files in {test_dir}:")
        for test_file in sorted(test_files):
            print(f"\n--- Processing {os.path.basename(test_file)} ---")
            if args.mode == "generate":
                if not generate_reference(test_file):
                    all_tests_passed = False
            else: # mode == "test"
                if not check_regression(test_file):
                    all_tests_passed = False
        
        if all_tests_passed:
            print("\nAll selected tests PASSED!")
            sys.exit(0)
        else:
            print("\nSome tests FAILED!")
            sys.exit(1)
    elif args.grid_file:
        if args.mode == "generate":
            generate_reference(args.grid_file)
        else:
            check_regression(args.grid_file)
    else:
        print("Error: No grid file specified and --all option not used.")
        parser.print_help()
        sys.exit(1)
