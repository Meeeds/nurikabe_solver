import json
import glob
import os
import sys
import pygame

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nurikabe_model import NurikabeModel, CellState
from nurikabe_drawing import Camera, draw_grid
import grid_style

def restore_model_from_string_grid(grid_strings):
    """Restores a NurikabeModel from a list of lists of strings."""
    rows = len(grid_strings)
    cols = len(grid_strings[0])
    
    # 1. Extract clues
    clues = [[0 for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            s = grid_strings[r][c]
            if s.startswith("CLUE("):
                clues[r][c] = int(s[5:-1])
                
    model = NurikabeModel()
    model.load_grid(clues)
    
    # 2. Set states and owners
    for r in range(rows):
        for c in range(cols):
            s = grid_strings[r][c]
            cell = model.cells[r][c]
            if s == "SEA":
                model.force_sea(r, c)
            elif s.startswith("LAND("):
                model.force_land(r, c)
                id_str = s[5:-1]
                if id_str != "MULTIPLE":
                    island_id = int(id_str)
                    model.force_owner(r, c, island_id)
                # else: multiple owners, already set by force_land
            elif s.startswith("CLUE("):
                # load_grid already set it to LAND with correct owner
                pass
            elif s == "UNKNOWN":
                cell.state = CellState.UNKNOWN
                # owners are already full from load_grid
                
    return model

def save_comparison_image(name, grid1, grid2, conflicts=None):
    """Generates a side-by-side comparison image of V1 and V2 results."""
    # Pygame setup
    if not pygame.get_init():
        pygame.init()
    
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

    # Create models
    model1 = restore_model_from_string_grid(grid1)
    model2 = restore_model_from_string_grid(grid2)

    rows, cols = model1.rows, model1.cols
    if rows == 0 or cols == 0: return

    grid_width = cols * BASE_CELL_SIZE
    grid_height = rows * BASE_CELL_SIZE
    img_width = 2 * grid_width + 3 * PADDING
    img_height = grid_height + PADDING + TITLE_HEIGHT

    # Create surface
    surface = pygame.Surface((img_width, img_height))
    surface.fill(grid_style.COLOR_BG)
    
    # Highlight conflicts on both grids if provided
    affected_cells = []
    if conflicts:
        for c_str in conflicts:
            # Parse "(r,c): ..."
            try:
                coords = c_str.split(":")[0].strip("()").split(",")
                affected_cells.append((int(coords[0]), int(coords[1])))
            except:
                pass

    # Draw V1
    camera1 = Camera(offset_x=PADDING, offset_y=TITLE_HEIGHT, zoom=1.0)
    draw_grid(surface, model1, camera1, BASE_CELL_SIZE, font, small_font, affected_cells=affected_cells)
    surface.blit(font.render(f"V1: {name}", True, (220, 220, 220)), (PADDING, 5))

    # Draw V2
    camera2 = Camera(offset_x=grid_width + 2 * PADDING, offset_y=TITLE_HEIGHT, zoom=1.0)
    draw_grid(surface, model2, camera2, BASE_CELL_SIZE, font, small_font, affected_cells=affected_cells)
    surface.blit(font.render(f"V2: {name}", True, (220, 220, 220)), (grid_width + 2 * PADDING, 5))

    # Target images directory
    images_dir = os.path.join('tests', 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Save image
    safe_name = name.replace(os.sep, '_').replace('/', '_')
    image_path = os.path.join(images_dir, f"{safe_name}.png")
    pygame.image.save(surface, image_path)
    return image_path

def compare_references():
    v2_refs = sorted(glob.glob("tests/**/*.reference.v2.json", recursive=True))
    
    results = []
    total_conflicts = 0

    print("Comparing V1 and V2 references...")

    for f2 in v2_refs:
        f1 = f2.replace(".reference.v2.json", ".reference.json")
        # Get relative path for display
        rel_path = os.path.relpath(f2, "tests").replace(".reference.v2.json", "")
        
        if not os.path.exists(f1):
            continue

        with open(f1, 'r') as j1, open(f2, 'r') as j2:
            data1 = json.load(j1)
            data2 = json.load(j2)

        grid1 = data1['final_grid']
        grid2 = data2['final_grid']
        
        rows = len(grid1)
        cols = len(grid1[0])
        
        conflicts = []
        v1_only = 0
        v2_only = 0
        common = 0
        
        for r in range(rows):
            for c in range(cols):
                s1 = grid1[r][c]
                s2 = grid2[r][c]
                
                # Normalize LAND(X) to LAND for comparison
                type1 = "LAND" if s1.startswith("LAND") else s1
                type2 = "LAND" if s2.startswith("LAND") else s2
                
                if type1 != "UNKNOWN" and type2 == "UNKNOWN":
                    v1_only += 1
                elif type1 == "UNKNOWN" and type2 != "UNKNOWN":
                    v2_only += 1
                elif type1 != "UNKNOWN" and type2 != "UNKNOWN":
                    common += 1
                    if type1 != type2:
                        conflicts.append(f"({r},{c}): V1={s1} vs V2={s2}")

        v1_count = data1.get('number_of_cell_found', 0)
        v2_count = data2.get('number_of_cell_found', 0)
        total_conflicts += len(conflicts)

        has_difference = (v1_count != v2_count) or (len(conflicts) > 0)
        image_generated = False
        if has_difference:
            try:
                save_comparison_image(rel_path, grid1, grid2, conflicts=conflicts)
                image_generated = True
            except Exception as e:
                print(f"Error generating image for {rel_path}: {e}")

        results.append({
            'name': rel_path,
            'v1': v1_count,
            'v2': v2_count,
            'v1_only': v1_only,
            'v2_only': v2_only,
            'common': common,
            'conflicts': len(conflicts),
            'conflict_details': conflicts,
            'image_generated': image_generated
        })

    # Display final table
    header = f"{'Puzzle':<50} | {'V1':>5} | {'V2':>5} | {'Com.':>5} | {'V1!':>5} | {'V2!':>5} | {'Delta':>6} | {'Status':<12} | {'Img'}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for res in results:
        delta = res['v2'] - res['v1']
        delta_str = f"{delta:+d}" if delta != 0 else "0"
        status = "OK" if res['conflicts'] == 0 else f"{res['conflicts']} CONFLICTS"
        
        # Color coding simulation for terminal if delta > 0 (Improvement)
        # Using simple markers since we are in a text-only output
        imp_marker = " (+)" if delta > 0 else ""
        img_marker = " [X]" if res['image_generated'] else ""
        
        print(f"{res['name']:<50} | {res['v1']:>5} | {res['v2']:>5} | {res['common']:>5} | {res['v1_only']:>5} | {res['v2_only']:>5} | {delta_str:>6} | {status:<12}{imp_marker} | {img_marker}")
        if res['conflict_details']:
            for detail in res['conflict_details'][:2]:
                print(f"    CONFLICT: {detail}")
            if len(res['conflict_details']) > 2:
                print(f"    ...")

    print("-" * len(header))

    total_v1 = sum(r['v1'] for r in results)
    total_v2 = sum(r['v2'] for r in results)
    total_common = sum(r['common'] for r in results)
    total_v1_only = sum(r['v1_only'] for r in results)
    total_v2_only = sum(r['v2_only'] for r in results)
    total_delta = total_v2 - total_v1
    total_delta_str = f"{total_delta:+d}" if total_delta != 0 else "0"

    print(f"{'TOTAL':<50} | {total_v1:>5} | {total_v2:>5} | {total_common:>5} | {total_v1_only:>5} | {total_v2_only:>5} | {total_delta_str:>6} |")
    print("-" * len(header))
    print(f"Total Puzzles Compared: {len(results)}")
    print(f"Total Contradictions Found: {total_conflicts}")
    
    better_v2 = [r['name'] for r in results if r['v2'] > r['v1']]
    if better_v2:
        print(f"Puzzles where V2 found more cells: {len(better_v2)}")
    
    if total_conflicts == 0:
        print("\nConsistency check PASSED: No contradictions found between V1 and V2.")
    else:
        print(f"\nConsistency check FAILED: {total_conflicts} cell contradictions found.")
    
    if any(r['image_generated'] for r in results):
        print(f"Images generated in 'tests/images/' for differences.")

    if pygame.get_init():
        pygame.quit()

if __name__ == "__main__":
    compare_references()
