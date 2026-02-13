import pytest
import json
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nurikabe_model import NurikabeModel
from nurikabe_rules_v2 import NurikabeSolverV2


def get_rule_func(solver, rule_name):
    code = rule_name.split()[0]
    func = getattr(solver, f"try_{code}", None)
    return func


TEST_DATA_FILES = [
    'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_0.json', 'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_1.json', 'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_2.json', 'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_3.json', 'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_4.json', 'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_5.json', 'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_6.json', 'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_7.json', 'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_8.json', 'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_9.json', 'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_10.json', 'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_11.json', 'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_12.json', 'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_13.json', 'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_14.json', 'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_15.json', 'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_16.json', 'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_17.json', 'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_18.json', 'unittest_v2/B0_Distance_Pruning/data/rule_B0_Distance_Pruning_19.json'
]

@pytest.mark.parametrize("data_file", TEST_DATA_FILES)
def test_B0_Distance_Pruning(data_file):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    abs_path = os.path.join(project_root, data_file)
    
    if not os.path.exists(abs_path):
        pytest.fail(f"Data file not found: {abs_path}")

    with open(abs_path, 'r') as f:
        data = json.load(f)
    
    assert data['grid_before'] != data['grid_after'], "grid is the same before and after"
    model = NurikabeModel()
    model.restore(data['grid_before'])
    
    solver = NurikabeSolverV2(model)
    rule_name = data['rule_applied']
    
    rule_func = get_rule_func(solver, rule_name)
    assert rule_func is not None, f"Rule {rule_name} not found in solver."
    
    res = rule_func()
    
    assert res is not None, f"Rule {rule_name} expected to apply but returned None."
    
    current_snapshot = model.snapshot()
    assert current_snapshot['cells'] == data['grid_after']['cells'], "Grid state (cells/owners) mismatch after rule application."