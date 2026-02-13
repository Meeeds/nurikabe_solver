import json
import glob
import os

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
        for r in range(rows):
            for c in range(cols):
                s1 = grid1[r][c]
                s2 = grid2[r][c]
                
                # Normalize LAND(X) to LAND for comparison
                type1 = "LAND" if s1.startswith("LAND") else s1
                type2 = "LAND" if s2.startswith("LAND") else s2
                
                if type1 != "UNKNOWN" and type2 != "UNKNOWN" and type1 != type2:
                    conflicts.append(f"({r},{c}): V1={s1} vs V2={s2}")

        v1_count = data1.get('number_of_cell_found', 0)
        v2_count = data2.get('number_of_cell_found', 0)
        total_conflicts += len(conflicts)

        results.append({
            'name': rel_path,
            'v1': v1_count,
            'v2': v2_count,
            'conflicts': len(conflicts),
            'conflict_details': conflicts
        })

    # Display final table
    header = f"{'Puzzle':<50} | {'V1':>6} | {'V2':>6} | {'Delta':>6} | {'Status':<12}"
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
        
        print(f"{res['name']:<50} | {res['v1']:>6} | {res['v2']:>6} | {delta_str:>6} | {status:<12}{imp_marker}")
        if res['conflict_details']:
            for detail in res['conflict_details'][:2]:
                print(f"    CONFLICT: {detail}")
            if len(res['conflict_details']) > 2:
                print(f"    ...")

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

if __name__ == "__main__":
    compare_references()
