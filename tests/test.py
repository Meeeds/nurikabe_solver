import os
import json
import argparse
import sys
import time
from collections import Counter
from typing import Dict, Any, List

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nurikabe_model import NurikabeModel
from nurikabe_rules import NurikabeSolver

# Master list of rules defined in the solver and model
KNOWN_RULES = sorted(set(NurikabeSolver.RULE_NAMES))

GLOBAL_RULE_COUNTS = Counter()
NEW_CELLS_FOUND = 0
CELLS_NOW_NOT_FOUND = 0

def print_global_stats():
    """Prints a summary of rule applications across all tests."""
    print("\n" + "="*100)
    print(f"{'GLOBAL EXECUTION SUMMARY':^100}")
    print("="*100)
    
    # Print cumulative cell discovery stats
    print(f"    {'Global Cell Discovery Delta':<85} | {'Value':>10}")
    print(f"    {'-'*85}-+-{'-'*10}")
    print(f"    {'New cells found (IMPROVEMENT)':<85} | {NEW_CELLS_FOUND:>10}")
    print(f"    {'Cells now not found (REGRESSION)':<85} | {CELLS_NOW_NOT_FOUND:>10}")
    print(f"    {'-'*85}-+-{'-'*10}")

    print(f"\n    {'Rule Application Statistics':^100}")
    print(f"    {'Rule Name':<85} | {'Applications':>10}")
    print(f"    {'-'*85}-+-{'-'*10}")
    
    # Sort by application count descending, then by name
    sorted_stats = sorted(
        [(name, GLOBAL_RULE_COUNTS[name]) for name in KNOWN_RULES],
        key=lambda x: (-x[1], x[0])
    )
    
    for name, count in sorted_stats:
        status = "" if count > 0 else "  [NEVER TRIGGERED]"
        print(f"    {name:<85} | {count:>10}{status}")
    
    # Also check for any rules that appeared but aren't in KNOWN_RULES (safety)
    unknown_rules = set(GLOBAL_RULE_COUNTS.keys()) - set(KNOWN_RULES) - {"None"}
    if unknown_rules:
        print(f"    {'-'*85}-+-{'-'*10}")
        for name in sorted(unknown_rules):
            print(f"    {name:<85}*| {GLOBAL_RULE_COUNTS[name]:>10}  [UNKNOWN RULE]")
            
    print("="*100 + "\n")

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
        # print(f"Step {steps_taken + 1}: Rule applied: {result.rule}, Message: {result.message}, Changed Cells: {len(result.changed_cells)}")
        
        # The solver returns rule="None" when no more rules can be applied
        if result.rule == "None":
            break
        
        rule_name = result.rule
        rule_counts[rule_name] += 1
        GLOBAL_RULE_COUNTS[rule_name] += 1
        steps_taken += 1

    # Determine if solved (no unknowns left) and count definitive cells
    is_solved = True
    cells_found = 0
    for r in range(model.rows):
        for c in range(model.cols):
            if model.is_black_certain(r, c) or model.is_land_certain(r, c):
                cells_found += 1
            else:
                is_solved = False

    return {
        "rules_triggered": dict(rule_counts),
        "steps_total": steps_taken,
        "is_fully_solved": is_solved,
        "number_of_cell_found": cells_found,
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
    print(f"Solved: {result['is_fully_solved']}, Cells found: {result['number_of_cell_found']}, Steps: {result['steps_total']}")
    return True

def check_regression(grid_path: str) -> bool:
    """Runs solver and compares result with existing reference JSON."""
    global NEW_CELLS_FOUND, CELLS_NOW_NOT_FOUND
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
    cells_found_match = current_result.get("number_of_cell_found") == reference_result.get("number_of_cell_found")
    
    if grid_match and rules_match and cells_found_match:
        print(f"TEST PASSED: '{grid_path}' matches reference exactly.")
        return True
    else:
        print(f"TEST FAILED: '{grid_path}' output mismatch.")
        
        if not grid_match:
            print("  CRITICAL: Final grid state differs!")
            print(f" reference_result[is_fully_solved] = {reference_result.get('is_fully_solved')} current_result[is_fully_solved] = {current_result['is_fully_solved']}")
        
        if not cells_found_match:
            print(f"  CRITICAL: Number of cells found differs! Ref: {reference_result.get('number_of_cell_found')}, Cur: {current_result['number_of_cell_found']}")
            if current_result['number_of_cell_found'] > reference_result.get('number_of_cell_found'):
                NEW_CELLS_FOUND += current_result['number_of_cell_found'] - reference_result.get('number_of_cell_found')
            else:
                CELLS_NOW_NOT_FOUND += reference_result.get('number_of_cell_found') - current_result['number_of_cell_found']
        
        if not rules_match:
            print("  WARNING: Rule usage counts differ (logic path changed).")
            ref_rules = reference_result["rules_triggered"]
            cur_rules = current_result["rules_triggered"]
            all_rule_names = sorted(set(ref_rules.keys()) | set(cur_rules.keys()))
            
            print(f"    {'Rule Name':<60} | {'Ref':>5} | {'Cur':>5} | {'Diff':>5}")
            print(f"    {'-'*80}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}")
            for name in all_rule_names:
                ref_val = ref_rules.get(name, 0)
                cur_val = cur_rules.get(name, 0)
                diff = cur_val - ref_val
                diff_str = f"{diff:+d}" if diff != 0 else "0"
                print(f"    {name:<80} | {ref_val:>5} | {cur_val:>5} | {diff_str:>5}")
            
            print(f"    {'-'*80}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}")
            ref_total = reference_result.get('steps_total', sum(ref_rules.values()))
            cur_total = current_result['steps_total']
            print(f"    {'TOTAL STEPS':<80} | {ref_total:>5} | {cur_total:>5} | {cur_total - ref_total:>+5}")
            
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nurikabe Solver Test Runner")
    parser.add_argument("path", nargs='?', help="Path to a .txt grid file OR a directory containing .txt files. (Optional if --all is used)")
    parser.add_argument("--mode", choices=["generate", "test"], default="test", 
                        help="Mode: 'generate' to create reference JSON, 'test' to compare against it.")
    parser.add_argument("--all", action="store_true", 
                        help="Run all tests on *.txt files in the test directory (deprecated behavior, prefer providing a directory path).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print summary of execution times and global rule statistics.")
    
    args = parser.parse_args()

    files_to_process = []

    # Determine the target directory or file
    if args.path:
        if os.path.isdir(args.path):
            # It's a directory, find all .txt files
            files_to_process = [os.path.join(args.path, f) for f in os.listdir(args.path) if f.endswith(".txt")]
            if not files_to_process:
                 print(f"No .txt files found in the directory: {args.path}")
                 sys.exit(0)
            print(f"Running all tests in {args.mode} mode for {len(files_to_process)} files in {args.path}:")
        elif os.path.isfile(args.path):
            # It's a single file
            files_to_process = [args.path]
        else:
            print(f"Error: Path '{args.path}' does not exist.")
            sys.exit(1)
    elif args.all:
        # Fallback to legacy behavior: current script directory
        test_dir = os.path.dirname(os.path.abspath(__file__))
        files_to_process = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".txt")]
        if not files_to_process:
            print(f"No .txt files found in the test directory: {test_dir}")
            sys.exit(0)
        print(f"Running all tests in {args.mode} mode for files in {test_dir}:")
    else:
        print("Error: No path specified and --all option not used.")
        parser.print_help()
        sys.exit(1)

    # Process files
    all_tests_passed = True
    execution_times = []
    for test_file in sorted(files_to_process):
        test_name = os.path.basename(test_file)
        print(f"\n--- Processing {test_name} ---")
        start_time = time.time()
        
        if args.mode == "generate":
            passed = generate_reference(test_file)
        else: # mode == "test"
            passed = check_regression(test_file)
        
        elapsed = time.time() - start_time
        execution_times.append((test_name, elapsed))
        if args.verbose:
            print(f"Elapsed time: {elapsed:.3f}s")
        
        if not passed:
            all_tests_passed = False

    if args.verbose:
        # Print summary of execution times
        print("\n" + "="*70)
        print(f"{'TESTS SORTED BY ELAPSED TIME':^70}")
        print("="*70)
        print(f"    {'Test File':<50} | {'Duration':>10}")
        print(f"    {'-'*50}-+-{'-'*10}")
        for name, elapsed in sorted(execution_times, key=lambda x: x[1], reverse=True):
            print(f"    {name:<50} | {elapsed:>9.3f}s")
        print("="*70 + "\n")

        print_global_stats()

    if args.mode == "test" and len(files_to_process) > 1:
        if all_tests_passed:
            print("\nAll selected tests PASSED!")
            sys.exit(0)
        else:
            print("\nSome tests FAILED!")
            sys.exit(1)
