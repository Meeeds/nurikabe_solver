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
    'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_0.json', 'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_1.json', 'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_2.json', 'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_3.json', 'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_4.json', 'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_5.json', 'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_6.json', 'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_7.json', 'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_8.json', 'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_9.json', 'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_10.json', 'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_11.json', 'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_12.json', 'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_13.json', 'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_14.json', 'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_15.json', 'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_16.json', 'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_17.json', 'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_18.json', 'unittest/G2_Unification__land_cluster_domain_intersection/data/rule_G2_Unification__land_cluster_domain_intersection_19.json'
]

@pytest.mark.parametrize("data_file", TEST_DATA_FILES)
def test_G2_Unification__land_cluster_domain_intersection(data_file):
    # Resolve absolute path to data file
    # Assuming test is run from project root or unittest folder
    # We try to locate the file relative to the project root
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    abs_path = os.path.join(project_root, data_file)
    
    if not os.path.exists(abs_path):
        pytest.fail(f"Data file not found: {abs_path}")

    with open(abs_path, 'r') as f:
        data = json.load(f)
    
    assert data['grid_before'] != data['grid_after'], "grid is the same before and after"
    model = NurikabeModel()
    # Restore 'before' state
    model.restore(data['grid_before'])
    
    solver = NurikabeSolver(model)
    rule_name = data['rule_applied']
    
    rule_func = get_rule_func(rule_name)
    assert rule_func is not None, f"Rule {rule_name} not found in solver registry."
    
    # Apply the rule
    res = rule_func(solver)
    
    assert res is not None, f"Rule {rule_name} expected to apply but returned None."
    
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