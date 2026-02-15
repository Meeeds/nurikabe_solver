from typing import List, Dict, Any

def print_execution_times(execution_times: List[tuple]):
    """Prints a summary of execution times sorted by duration."""
    print("\n" + "="*70)
    print(f"{'TESTS SORTED BY ELAPSED TIME':^70}")
    print("="*70)
    print(f"    {'Test File':<50} | {'Duration':>10}")
    print(f"    {'-'*50}-+-{'-'*10}")
    for name, elapsed in sorted(execution_times, key=lambda x: x[1], reverse=True):
        print(f"    {name:<50} | {elapsed:>9.3f}s")
    print("="*70 + "\n")

def print_test_comparison_summary(results: List[Dict[str, Any]]):
    """
    Prints a detailed table comparing current results with references.
    Expected keys in results: 'name', 'cur_cells', 'ref_cells', 'common', 'cur_only', 'ref_only', 'elapsed', 'status'
    """
    header = f"{'Puzzle':<50} | {'Cur':>5} | {'Ref':>5} | {'Com.':>5} | {'Ref!':>5} | {'Cur!':>5} | {'Delta':>6} | {'Time':>9} | {'Status':<12}"
    width = len(header)
    print("\n" + "=" * width)
    title = 'TEST COMPARISON SUMMARY (CURRENT VS REFERENCE)'
    print(f"{title:^{width}}")
    print("=" * width)
    print(header)
    print("-" * width)
    
    for res in sorted(results, key=lambda x: x['elapsed'], reverse=True):
        name = res['name']
        cur = res.get('cur_cells', '-')
        ref = res.get('ref_cells', '-')
        common = res.get('common', '-')
        ref_only = res.get('ref_only', '-')
        cur_only = res.get('cur_only', '-')
        
        delta_str = "-"
        if isinstance(cur, int) and isinstance(ref, int):
            delta = cur - ref
            delta_str = f"{delta:+d}" if delta != 0 else "0"
        
        elapsed_val = res.get('elapsed', 0)
        elapsed_str = f"{elapsed_val:>8.3f}s"
        status = res.get('status', 'FAIL')
        
        # Add improvement marker if delta > 0
        imp_marker = " (+)" if isinstance(cur, int) and isinstance(ref, int) and cur > ref else ""
        status_display = f"{status}{imp_marker}"
        
        print(f"{name:<50} | {cur:>5} | {ref:>5} | {common:>5} | {ref_only:>5} | {cur_only:>5} | {delta_str:>6} | {elapsed_str} | {status_display:<12}")
    print("-" * width)
