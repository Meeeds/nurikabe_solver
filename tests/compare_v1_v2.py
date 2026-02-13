import json
import glob
import os

def compare_references():
    v2_refs = sorted(glob.glob("tests/**/*.reference.v2.json", recursive=True))
    
    total_checked = 0
    total_conflicts = 0
    puzzles_with_conflicts = []

    print(f"{'Puzzle':<40} | {'Status':<15} | {'V1 Found':>8} | {'V2 Found':>8}")
    print("-" * 80)

    for f2 in v2_refs:
        f1 = f2.replace(".reference.v2.json", ".reference.json")
        puzzle_name = os.path.basename(f2).replace(".reference.v2.json", "")
        
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
        for r in range(rows):
            for c in range(cols):
                s1 = grid1[r][c]
                s2 = grid2[r][c]
                
                # Normalize LAND(X) to LAND for comparison
                type1 = "LAND" if s1.startswith("LAND") else s1
                type2 = "LAND" if s2.startswith("LAND") else s2
                
                if type1 != "UNKNOWN" and type2 != "UNKNOWN" and type1 != type2:
                    conflicts.append(f"({r},{c}): V1={s1} vs V2={s2}")

        total_checked += 1
        v1_count = data1.get('number_of_cell_found', 0)
        v2_count = data2.get('number_of_cell_found', 0)

        if conflicts:
            total_conflicts += len(conflicts)
            puzzles_with_conflicts.append(puzzle_name)
            print(f"{puzzle_name:<40} | {len(conflicts):>3} CONFLICTS | {v1_count:>8} | {v2_count:>8}")
            for cmd in conflicts[:3]: # Show first 3 conflicts
                print(f"    {cmd}")
            if len(conflicts) > 3:
                print(f"    ...")
        else:
            print(f"{puzzle_name:<40} | OK              | {v1_count:>8} | {v2_count:>8}")

    print("-" * 80)
    print(f"Total Puzzles Compared: {total_checked}")
    print(f"Total Contradictions Found: {total_conflicts}")
    if puzzles_with_conflicts:
        print(f"Puzzles with errors: {', '.join(puzzles_with_conflicts)}")
    else:
        print("Consistency check PASSED: No contradictions found between V1 and V2.")

if __name__ == "__main__":
    compare_references()
