import os
import sys
import json
import random
import glob
from collections import defaultdict

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nurikabe_model import NurikabeModel
from nurikabe_rules import NurikabeSolver, _RULES

def get_puzzle_files():
    # Assuming puzzles are in puzzles/ folder relative to project root
    puzzle_dir = os.path.join(os.path.dirname(__file__), '..', 'tests')
    return sorted(glob.glob(os.path.join(puzzle_dir, '*.txt')))

def sanitize_filename(name):
    # Replace non-alphanumeric characters with underscores
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")

def collect_samples():
    samples = defaultdict(list)
    puzzle_files = get_puzzle_files()
    
    print(f"Found {len(puzzle_files)} puzzle files.")
    
    for p_file in puzzle_files:
        print(f"Processing {os.path.basename(p_file)}...")
        model = NurikabeModel()
        try:
            with open(p_file, 'r') as f:
                content = f.read()
            success, msg = model.parse_puzzle_text(content)
            if not success:
                print(f"Skipping {p_file}: {msg}")
                continue
        except Exception as e:
            print(f"Error reading {p_file}: {e}")
            continue

        solver = NurikabeSolver(model)
        
        while True:
            # Capture state before
            before_snapshot = model.snapshot()
            
            # Step
            res = solver.step()
            
            if res.rule == "None" or res.rule == "BROKEN_NURIKABE_RULES":
                break
            
            # Capture state after
            after_snapshot = model.snapshot()
            
            # Store sample
            samples[res.rule].append({
                "rule_applied": res.rule,
                "grid_before": before_snapshot,
                "grid_after": after_snapshot,
                "source_puzzle": os.path.basename(p_file)
            })
            
    return samples

def generate_tests(samples):
    unittest_dir = os.path.dirname(__file__)
    
    for rule_name, instances in samples.items():
        print(f"Generating tests for rule: {rule_name} ({len(instances)} instances found)")
        
        # Select up to 20 random instances
        selected = random.sample(instances, min(len(instances), 20))
        
        sanitized_rule_name = sanitize_filename(rule_name)
        
        rule_dir = os.path.join(unittest_dir, sanitized_rule_name)
        output_dir_data = os.path.join(rule_dir, 'data')
        
        # Ensure directories exist
        os.makedirs(output_dir_data, exist_ok=True)

        json_filenames = []
        
        for i, data in enumerate(selected):
            # Save JSON data
            json_filename = f"rule_{sanitized_rule_name}_{i}.json"
            json_path = os.path.join(output_dir_data, json_filename)
            
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Store relative path for the python test (relative to project root)
            json_filenames.append(os.path.join('unittest', sanitized_rule_name, 'data', json_filename))
            
        # Generate Python Test File
        test_filename = f"test_rule_{sanitized_rule_name}.py"
        test_path = os.path.join(unittest_dir, test_filename)
        
        test_content = f"""
import pytest
import json
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nurikabe_model import NurikabeModel
from nurikabe_rules import NurikabeSolver, _RULES

def get_rule_func(rule_name):
    for _, func, name in _RULES:
        if name == rule_name:
            return func
    return None

TEST_DATA_FILES = [
    {', '.join([repr(f) for f in json_filenames])}
]

@pytest.mark.parametrize("data_file", TEST_DATA_FILES)
def test_{sanitized_rule_name}(data_file):
    # Resolve absolute path to data file
    # Assuming test is run from project root or unittest folder
    # We try to locate the file relative to the project root
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    abs_path = os.path.join(project_root, data_file)
    
    if not os.path.exists(abs_path):
        pytest.fail(f"Data file not found: {{abs_path}}")

    with open(abs_path, 'r') as f:
        data = json.load(f)
    
    assert data['grid_before'] != data['grid_after'], "grid is the same before and after"
    model = NurikabeModel()
    # Restore 'before' state
    model.restore(data['grid_before'])
    
    solver = NurikabeSolver(model)
    rule_name = data['rule_applied']
    
    rule_func = get_rule_func(rule_name)
    assert rule_func is not None, f"Rule {{rule_name}} not found in solver registry."
    
    # Apply the rule
    res = rule_func(solver)
    
    assert res is not None, f"Rule {{rule_name}} expected to apply but returned None."
    
    # Restore 'after' state for comparison
    expected_model = NurikabeModel()
    expected_model.restore(data['grid_after'])
    
    # Compare cells (state and owners)
    # Using snapshot 'cells' structure: List[List[Tuple[state, owners]]]
    current_snapshot = model.snapshot()
    
    # Compare structure
    assert current_snapshot['cells'] == data['grid_after']['cells'], "Grid state (cells/owners) mismatch after rule application."
    
    # Optionally check changed cells if recorded
    if res.changed_cells:
        # Check that the reported changed cells actually changed
        pass 
"""
        with open(test_path, 'w') as f:
            f.write(test_content.strip())

if __name__ == "__main__":
    random.seed(42) # Reproducibility
    samples = collect_samples()
    generate_tests(samples)
