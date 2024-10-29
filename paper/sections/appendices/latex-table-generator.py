import os
from pathlib import Path
import re

def escape_latex_path(path):
    """Escape backslashes in paths for LaTeX."""
    return str(path).replace('\\', '/')

def parse_png_filename(filename):
    """Parse PNG filename to extract dataset, window size, and iteration."""
    stem = Path(filename).stem
    try:
        dataset, window_size, iteration, fourth = stem.split('_')
        return dataset + window_size, iteration, fourth
    except: 
        return stem.split('_')

def format_window_size(size):
    """Format window size for section title."""
    return f"{size}$\\times${size} Window"

def generate_latex_tables(dataset_names, base_directory):
    """Generate LaTeX tables for each dataset's PNG files, grouped by window size."""
    # Template for each table with caption
    table_template = r'''
\begin{table}
    \caption{%s}
    \includegraphics[width=\linewidth]{%s}
    \label{tab:%s}
\end{table}
'''
    
    for dataset in dataset_names:
        try:
            # Create Path object for the directory
            directory = Path(base_directory) / dataset
            
            if not directory.exists():
                print(f"Warning: Directory not found: {directory}")
                continue
                
            latex_code = ''
            # Get PNG files and parse their names
            png_files = []
            for f in directory.glob("*.png"):
                dataset_name, window_size, iteration = parse_png_filename(f.name)
                if all(x is not None for x in (dataset_name, window_size, iteration)):
                    png_files.append((f, window_size, iteration))
            
            if not png_files:
                print(f"Warning: No valid PNG files found in {directory}")
                continue
            
            # Sort files by window size and iteration
            png_files.sort(key=lambda x: (x[1], x[2]))
            
            current_window_size = None
            
            for png_file, window_size, iteration in png_files:
                # Add subsubsection when window size changes
                if window_size != current_window_size:
                    latex_code += f"\n\\subsubsection*{{{format_window_size(window_size)}}}\n"
                    current_window_size = window_size
                
                # Get the file name without extension for label
                label = png_file.stem
                
                # Create a caption without the iteration number
                caption = f"{dataset.split('_')[0]} Results - {format_window_size(window_size)}"
                
                
                # Escape the path for LaTeX
                image_path = escape_latex_path(png_file)
                
                # Add table to the LaTeX code
                latex_code += table_template % (caption, image_path, label)
            
            # Write the LaTeX code to a file
            output_file = Path(f'{base_directory}/{dataset}-tables.tex')
            with output_file.open('w', encoding='utf-8') as f:
                f.write(latex_code)

            
            print(f"Successfully generated {output_file}")
            
        except Exception as e:
            print(f"Error processing {dataset}: {str(e)}")

# Usage
datasets = ['caltech_30', 'INRIA', 'PnPLO']
base_dir = "/Users/adamsam/repos/ee/Pedestrian-Detection/code/paper/tables"

generate_latex_tables(datasets, base_dir)