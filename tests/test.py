import os
import json
import argparse
import sys
import time
import glob
from typing import Dict, Any, List

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nurikabe_model import NurikabeModel
from nurikabe_rules import NurikabeSolver
from nurikabe_rules_v2 import NurikabeSolverV2
from tests.test_utils import print_test_comparison_summary, print_execution_times

SOLVER_TIMEOUT = 30 # Seconds

# Global variable to store the selected solver class
SOLVER_CLASS = NurikabeSolver

NEW_CELLS_FOUND = 0
CELLS_NOW_NOT_FOUND = 0
EXCLUDED_FILES = []

def print_global_stats():
    """Prints a summary of cell discovery across all tests."""
    print("\n" + "="*100)
    print(f"{'GLOBAL EXECUTION SUMMARY':^100}")
    print("="*100)
    
    # Print cumulative cell discovery stats
    print(f"    {'Global Cell Discovery Delta':<85} | {'Value':>10}")
    print(f"    {'-'*85}-+-{'-'*10}")
    print(f"    {'New cells found (IMPROVEMENT)':<85} | {NEW_CELLS_FOUND:>10}")
    print(f"    {'Cells now not found (REGRESSION)':<85} | {CELLS_NOW_NOT_FOUND:>10}")
    print(f"    {'-'*85}-+-{'-'*10}")

    if EXCLUDED_FILES:
        print(f"\n    {'Excluded Files (Timed out > 30s)':^100}")
        print(f"    {'-'*100}")
        for f in sorted(EXCLUDED_FILES):
            print(f"    {f}")
        print(f"    {'-'*100}")

    print("="*100 + "\n")

def serialize_grid(model: NurikabeModel) -> List[List[str]]:
    """Converts the final grid state into a list of strings for JSON serialization."""
    grid_state = []
    for r in range(model.rows):
        row_state = []
        for c in range(model.cols):
            if model.is_clue(r, c):
                row_state.append(f"CLUE({model.clues[r][c]})")
            elif model.is_sea_certain(r, c):
                row_state.append("SEA")
            elif model.is_land_certain(r, c):
                # Distinguish if it has an assigned owner or just land
                owners = model.cells[r][c].owners.to_ids()
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

    solver = SOLVER_CLASS(model)
    
    start_time = time.time()
    while True:
        result = solver.step()
        
        if result.rule == "BROKEN_NURIKABE_RULES":
            return None, f"CRITICAL ERROR: Contradiction detected! {result.message}"

        # The solver returns rule="None" when no more rules can be applied
        if result.rule == "None":
            break
        
        if time.time() - start_time > SOLVER_TIMEOUT:
            return None, "TIMEOUT"


    # Determine if solved (no unknowns left) and count definitive cells
    is_solved = True
    cells_found = 0
    for r in range(model.rows):
        for c in range(model.cols):
            if model.is_sea_certain(r, c) or model.is_land_certain(r, c):
                cells_found += 1
            else:
                is_solved = False

    return {
        "is_fully_solved": is_solved,
        "number_of_cell_found": cells_found,
        "final_grid": serialize_grid(model)
    }, None

def get_reference_path(grid_path: str, model_name: str) -> str:
    """Returns the reference file path based on the model name."""
    if model_name == "v2":
        return grid_path + ".reference.v2.json"
    return grid_path + ".reference.json"

def generate_reference(grid_path: str, model_name: str) -> bool:
    """Runs solver and saves the result as a reference JSON."""
    result, error_msg = run_solver(grid_path)
    if error_msg:
        print(f"Error for {grid_path}: {error_msg}")
        return False
        
    ref_path = get_reference_path(grid_path, model_name)
    
    with open(ref_path, 'w') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    
    print(f"Success: Reference generated for '{grid_path}' and saved to '{ref_path}'")
    print(f"Solved: {result['is_fully_solved']}, Cells found: {result['number_of_cell_found']}")
    return True

def run_and_print_stats(grid_path: str) -> bool:
    """Runs solver and prints result stats without saving."""
    result, error_msg = run_solver(grid_path)
    if error_msg:
        print(f"Error for {grid_path}: {error_msg}")
        return False
    
    print(f"Results for '{grid_path}':")
    print(f"Solved: {result['is_fully_solved']}, Cells found: {result['number_of_cell_found']}")
    return True

def check_regression_with_result(grid_path: str, current_result: Dict[str, Any], model_name: str) -> tuple[bool, Dict[str, Any] | None, Dict[str, int]]:
    """
    Compares current solver result with existing reference JSON. 
    Returns (passed, reference_result, diff_metrics).
    """
    global NEW_CELLS_FOUND, CELLS_NOW_NOT_FOUND

    ref_path = get_reference_path(grid_path, model_name)
    metrics = {'common': 0, 'cur_only': 0, 'ref_only': 0}
    
    try:
        with open(ref_path, 'r') as f:
            reference_result = json.load(f)
    except FileNotFoundError:
        print(f"Error for '{grid_path}': Reference file '{ref_path}' not found. Run in 'generate' mode first.")
        return False, None, metrics
    except json.JSONDecodeError:
        print(f"Error for '{grid_path}': Reference file '{ref_path}' is not a valid JSON.")
        return False, None, metrics
    
    # Cell-by-cell comparison for detailed metrics
    grid1 = reference_result["final_grid"]
    grid2 = current_result["final_grid"]
    rows = len(grid1)
    cols = len(grid1[0])
    
    for r in range(rows):
        for c in range(cols):
            s1 = grid1[r][c]
            s2 = grid2[r][c]
            
            # Normalize LAND(X) to LAND for comparison
            type1 = "LAND" if s1.startswith("LAND") else s1
            type2 = "LAND" if s2.startswith("LAND") else s2
            
            if type1 != "UNKNOWN" and type2 == "UNKNOWN":
                metrics['ref_only'] += 1
            elif type1 == "UNKNOWN" and type2 != "UNKNOWN":
                metrics['cur_only'] += 1
            elif type1 != "UNKNOWN" and type2 != "UNKNOWN":
                metrics['common'] += 1

    # Comparison Logic
    grid_match = current_result["final_grid"] == reference_result["final_grid"]
    cells_found_match = current_result.get("number_of_cell_found") == reference_result.get("number_of_cell_found")
    
    if grid_match and cells_found_match:
        print(f"TEST PASSED: '{grid_path}' matches reference exactly.")
        return True, reference_result, metrics
    else:
        print(f"TEST FAILED: '{grid_path}' output mismatch.")
        
        if not grid_match:
            print("  WARNING Final grid state differs!")
            if reference_result.get('is_fully_solved') and not current_result['is_fully_solved']:
                print("CRITICAL REGRESSION, grid was previously solved, now it is not")
            elif not reference_result.get('is_fully_solved') and current_result['is_fully_solved']:
                print("IMPROVEMENT, grid was previously not solved, now it is solved :)")
                
            print(f" reference_result[is_fully_solved] = {reference_result.get('is_fully_solved')} current_result[is_fully_solved] = {current_result['is_fully_solved']}")
        
        if not cells_found_match:
            
            if current_result['number_of_cell_found'] > reference_result.get('number_of_cell_found'):
                print(f"  IMPROVEMENT: Number of cells found differs! Ref: {reference_result.get('number_of_cell_found')}, Cur: {current_result['number_of_cell_found']}")
                NEW_CELLS_FOUND += current_result['number_of_cell_found'] - reference_result.get('number_of_cell_found')
            else:
                print(f"  CRITICAL REGRESSION: Number of cells found differs! Ref: {reference_result.get('number_of_cell_found')}, Cur: {current_result['number_of_cell_found']}")
                CELLS_NOW_NOT_FOUND += reference_result.get('number_of_cell_found') - current_result['number_of_cell_found']
        
        return False, reference_result, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nurikabe Solver Test Runner")
    parser.add_argument("path", nargs='?', help="Path to a .txt grid file OR a directory containing .txt files. (Defaults to 'tests/' directory)")
    parser.add_argument("--mode", choices=["generate", "test", "stats"], default="test", 
                        help="Mode: 'generate' to create reference JSON, 'test' to compare against it, 'stats' to just run and show stats.")
    parser.add_argument("--model", choices=["v1", "v2"], default="v1", help="Solver model to use.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print summary of execution times and global rule statistics.")
    
    args = parser.parse_args()

    if args.model == "v2":
        SOLVER_CLASS = NurikabeSolverV2
    else:
        SOLVER_CLASS = NurikabeSolver
    
    files_to_process = []

    # Determine the target directory or file
    target_path = args.path if args.path else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
    
    if os.path.isdir(target_path):
        # It's a directory, find all .txt files recursively
        files_to_process = sorted(glob.glob(os.path.join(target_path, "**", "*.txt"), recursive=True))
        if not files_to_process:
                print(f"No .txt files found in the directory: {target_path}")
                sys.exit(0)
        print(f"Running all tests in {args.mode} mode for {len(files_to_process)} files in {target_path}:")
    elif os.path.isfile(target_path):
        # It's a single file
        files_to_process = [target_path]
    else:
        print(f"Error: Path '{target_path}' does not exist.")
        sys.exit(1)

    # Process files
    all_tests_passed = True
    test_results = []
    for test_file in sorted(files_to_process):
        test_name = os.path.basename(test_file)
        print(f"\n--- Processing {test_name} ---")
        start_time = time.time()
        
        passed = False
        res_data = {
            'name': test_name,
            'cur_cells': None,
            'ref_cells': None,
            'status': 'FAIL'
        }

        if args.mode == "generate":
            current_result, error_msg = run_solver(test_file)
            if error_msg == "TIMEOUT":
                print(f"EXCLUDED (TIMEOUT > {SOLVER_TIMEOUT}s): {test_file}")
                EXCLUDED_FILES.append(test_file)
                passed = True
                res_data['status'] = 'TIMEOUT'
            elif error_msg:
                print(f"Error for {test_file}: {error_msg}")
                passed = False
            else:
                ref_path = get_reference_path(test_file, args.model)
                with open(ref_path, 'w') as f:
                    json.dump(current_result, f, indent=2, sort_keys=True)
                print(f"Success: Reference generated for '{test_file}' and saved to '{ref_path}'")
                passed = True
                res_data['status'] = 'GEN'
                res_data['cur_cells'] = current_result['number_of_cell_found']
        elif args.mode == "stats":
            result, error_msg = run_solver(test_file)
            if error_msg == "TIMEOUT":
                print(f"EXCLUDED (TIMEOUT > {SOLVER_TIMEOUT}s): {test_file}")
                EXCLUDED_FILES.append(test_file)
                passed = True
                res_data['status'] = 'TIMEOUT'
            elif error_msg:
                print(f"Error for {test_file}: {error_msg}")
                passed = False
            else:
                print(f"Results for '{test_file}':")
                print(f"Solved: {result['is_fully_solved']}, Cells found: {result['number_of_cell_found']}")
                passed = True
                res_data['status'] = 'STATS'
                res_data['cur_cells'] = result['number_of_cell_found']
        else: # mode == "test"
            current_result, error_msg = run_solver(test_file)
            if error_msg == "TIMEOUT":
                print(f"EXCLUDED (TIMEOUT > {SOLVER_TIMEOUT}s): {test_file}")
                EXCLUDED_FILES.append(test_file)
                passed = True
                res_data['status'] = 'TIMEOUT'
            elif error_msg:
                print(f"Error for {test_file}: {error_msg}")
                passed = False
            else:
                passed, reference_result, metrics = check_regression_with_result(test_file, current_result, args.model)
                res_data['cur_cells'] = current_result['number_of_cell_found']
                if reference_result:
                    res_data['ref_cells'] = reference_result.get('number_of_cell_found')
                res_data['common'] = metrics['common']
                res_data['cur_only'] = metrics['cur_only']
                res_data['ref_only'] = metrics['ref_only']
                res_data['status'] = 'PASS' if passed else 'FAIL'
        
        elapsed = time.time() - start_time
        res_data['elapsed'] = elapsed
        test_results.append(res_data)

        if args.verbose:
            print(f"Elapsed time: {elapsed:.3f}s")
        
        if not passed:
            all_tests_passed = False

    if args.verbose:
        execution_times = [(r['name'], r['elapsed']) for r in test_results]
        if args.mode == "test":
            print_test_comparison_summary(test_results)
        else:
            print_execution_times(execution_times)

        print_global_stats()

    if args.mode == "test" and len(files_to_process) > 1:
        if all_tests_passed:
            print("\nAll selected tests PASSED!")
            sys.exit(0)
        else:
            print("\nSome tests FAILED!")
            sys.exit(1)
