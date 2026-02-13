import os
import sys
import json
import glob
import pygame
import argparse

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nurikabe_model import NurikabeModel
from nurikabe_drawing import Camera, draw_grid
import grid_style

def get_rule_func(solver, rule_name, model_name):
    if model_name == "v1":
        from nurikabe_rules import _RULES
        for _, func, name in _RULES:
            if name == rule_name:
                return lambda: func(solver)
    else:
        code = rule_name.split()[0]
        func = getattr(solver, f"try_{code}", None)
        return func
    return None

def generate_images(model_name):
    """
    Generates side-by-side PNG images of the 'before' and 'after' grid states
    for each test case JSON file.
    """
    # Pygame setup
    pygame.init()
    # Try to find a system font, fallback to default
    try:
        font = pygame.font.SysFont("arial", 24)
        small_font = pygame.font.SysFont("arial", 14)
    except:
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 14)

    # Constants
    BASE_CELL_SIZE = 32
    PADDING = 20
    TITLE_HEIGHT = 40

    # Dirs
    unittest_root = "unittest" if model_name == "v1" else "unittest_v2"
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    search_dir = os.path.join(project_root, unittest_root)

    if model_name == "v1":
        from nurikabe_rules import NurikabeSolver as SolverClass
    else:
        from nurikabe_rules_v2 import NurikabeSolverV2 as SolverClass

    # Find all .json files in selected unittest root
    json_files = sorted(glob.glob(os.path.join(search_dir, '**', '*.json'), recursive=True))
    print(f"Found {len(json_files)} test data files in {unittest_root}. Generating images...")

    for json_file in json_files:
        print(f"  - Processing {os.path.basename(json_file)}...")
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Create model_before
            model_before = NurikabeModel()
            model_before.restore(data['grid_before'])
            
            # Create model_after by applying the rule
            model_after = NurikabeModel()
            model_after.restore(data['grid_before'])
            solver = SolverClass(model_after)
            
            rule_name = data['rule_applied']
            rule_func = get_rule_func(solver, rule_name, model_name)
            if not rule_func:
                print(f"    WARNING: Rule '{rule_name}' not found. Skipping.")
                continue

            res = rule_func()
            affected_cells = res.changed_cells if res else []

            # Target images directory
            data_dir = os.path.dirname(json_file)
            rule_dir = os.path.dirname(data_dir)
            images_dir = os.path.join(rule_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)

            # Assumes both grids have same dimensions
            rows, cols = model_before.rows, model_before.cols
            if rows == 0 or cols == 0: continue

            grid_width = cols * BASE_CELL_SIZE
            grid_height = rows * BASE_CELL_SIZE
            img_width = 2 * grid_width + 3 * PADDING
            img_height = grid_height + PADDING + TITLE_HEIGHT

            # Create surface
            surface = pygame.Surface((img_width, img_height))
            surface.fill(grid_style.COLOR_BG)
            
            # Draw "Before"
            camera_before = Camera(offset_x=PADDING, offset_y=TITLE_HEIGHT, zoom=1.0)
            draw_grid(surface, model_before, camera_before, BASE_CELL_SIZE, font, small_font)
            surface.blit(font.render("Before", True, (220, 220, 220)), (PADDING, 5))

            # Draw "After"
            camera_after = Camera(offset_x=grid_width + 2 * PADDING, offset_y=TITLE_HEIGHT, zoom=1.0)
            draw_grid(surface, model_after, camera_after, BASE_CELL_SIZE, font, small_font, affected_cells=affected_cells)
            
            rule_name_short = rule_name[:40] + '...' if len(rule_name) > 40 else rule_name
            surface.blit(font.render(f"After: {rule_name_short}", True, (220, 220, 220)), (grid_width + 2 * PADDING, 5))

            # Save image
            image_path = os.path.join(images_dir, os.path.basename(json_file).replace('.json', '.png'))
            pygame.image.save(surface, image_path)

        except Exception as e:
            print(f"    ERROR processing {os.path.basename(json_file)}: {e}")

    pygame.quit()
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["v1", "v2"], default="v1")
    args = parser.parse_args()
    generate_images(args.model)
