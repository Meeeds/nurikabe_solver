import json
import os

def extract_puzzles():
    input_file = 'puzzles/database/Nurikabe_dataset.json'
    output_dir = 'puzzles/database/'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find input file {input_file}")
        return

    puzzles = data.get('data', {})
    
    # Sort keys to ensure sequential ordering matches the dataset structure (01_10x10, 02_10x10, etc.)
    sorted_keys = sorted(puzzles.keys())
    
    for i, key in enumerate(sorted_keys, 1):
        puzzle_data = puzzles[key]
        problem_str = puzzle_data.get('problem', '')
        
        if not problem_str:
            print(f"Warning: No problem data for {key}")
            continue
            
        lines = problem_str.strip().split('\n')
        
        # The first line contains dimensions (e.g., "10 10"), which we skip based on the requirements
        # We also need to replace '-' with '.' to match the target format
        grid_lines = []
        if len(lines) > 0:
            # Check if first line is dimensions (contains digits)
            first_line_parts = lines[0].split()
            if len(first_line_parts) == 2 and first_line_parts[0].isdigit() and first_line_parts[1].isdigit():
                grid_content = lines[1:]
            else:
                grid_content = lines
            
            for line in grid_content:
                # Replace '-' with '.'
                formatted_line = line.replace('-', '.')
                grid_lines.append(formatted_line)
        
        output_content = '\n'.join(grid_lines)
        
        # Generate filename: puzzlekit-dataset-001.txt, etc.
        filename = f"puzzlekit-dataset-{i:03d}.txt"
        file_path = os.path.join(output_dir, filename)
        
        with open(file_path, 'w') as f:
            f.write(output_content)
            f.write('\n') # Ensure file ends with a newline
            
        print(f"Generated {filename} from {key}")

    print(f"Extraction complete. {len(sorted_keys)} files created in {output_dir}")

if __name__ == "__main__":
    extract_puzzles()
