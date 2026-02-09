import os
import sys
import json
import glob
import pygame

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nurikabe_model import NurikabeModel
from nurikabe_rules import NurikabeSolver, _RULES
from nurikabe_drawing import Camera, draw_grid
import grid_style

def get_rule_func(rule_name):
    for _, func, name in _RULES:
        if name == rule_name:
            return func
    return None

def sanitize_filename(name):
    return "".join(c if c.isalnum() else "_" for c in name).strip("_")

def generate_images():
    """
    Generates side-by-side PNG images of the 'before' and 'after' grid states
    for each test case JSON file.
    The images are saved in rule-specific subdirectories within unittest/.
    """
    # Pygame setup
    pygame.init()
    font = pygame.font.SysFont("arial", 24)
    small_font = pygame.font.SysFont("arial", 14)

    # Constants
    BASE_CELL_SIZE = 32
    PADDING = 20
    TITLE_HEIGHT = 40

    # Dirs
    unittest_dir = os.path.dirname(__file__)

    # Find all .json files in any subfolder of unittest/
    json_files = sorted(glob.glob(os.path.join(unittest_dir, '**', '*.json'), recursive=True))
    print(f"Found {len(json_files)} test data files. Generating images...")

    for json_file in json_files:
        print(f"  - Processing {json_file}...")
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            rule_name = data['rule_applied']
            rule_func = get_rule_func(rule_name)
            if not rule_func:
                print(f"    WARNING: Rule '{rule_name}' not found. Skipping.")
                continue

            # Target images directory is peer to the 'data' directory containing the json
            data_dir = os.path.dirname(json_file)
            rule_dir = os.path.dirname(data_dir)
            images_dir = os.path.join(rule_dir, 'images')
            
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
                print(f"Created directory: {images_dir}")

            # Create model_before
            model_before = NurikabeModel()
            model_before.restore(data['grid_before'])
            
            # Create model_after by applying the rule
            model_after = NurikabeModel()
            model_after.restore(data['grid_before']) # Start from same state
            solver = NurikabeSolver(model_after)
            res = rule_func(solver) # Apply rule

            affected_cells = res.changed_cells if res else []

            # Assumes both grids have same dimensions
            rows = model_before.rows
            cols = model_before.cols

            if rows == 0 or cols == 0:
                print(f"    Skipping {os.path.basename(json_file)} due to empty grid.")
                continue

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
            
            before_text = font.render("Before", True, (220, 220, 220))
            surface.blit(before_text, (PADDING, 5))

            # Draw "After" with highlights
            camera_after = Camera(offset_x=grid_width + 2 * PADDING, offset_y=TITLE_HEIGHT, zoom=1.0)
            draw_grid(surface, model_after, camera_after, BASE_CELL_SIZE, font, small_font, affected_cells=affected_cells)
            
            rule_name_short = rule_name[:40] + '...' if len(rule_name) > 40 else rule_name
            after_text = font.render(f"After: {rule_name_short}", True, (220, 220, 220))
            surface.blit(after_text, (grid_width + 2 * PADDING, 5))

            # Save image
            image_filename = os.path.basename(json_file).replace('.json', '.png')
            image_path = os.path.join(images_dir, image_filename)
            pygame.image.save(surface, image_path)

        except Exception as e:
            print(f"    ERROR processing {os.path.basename(json_file)}: {e}")


    pygame.quit()
    print(f"Done. Images saved in {images_dir}")

if __name__ == "__main__":
    generate_images()