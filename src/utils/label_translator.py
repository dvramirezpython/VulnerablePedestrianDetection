'''
Transform labels to keep only one label VP
1. Load each txt file from the labels directory for training and validation
2. Split each line by space
3. If the first element is 0 or 1 or 3 make it 1 (vulnerable pedestrian) otherwise 0 (pedestrian)
4. Save the modified lines back to the same txt file overwriting the original file
'''

from pathlib import Path
import os
def transform_labels(labels_dir):
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory {labels_dir} does not exist.")
    
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'r') as file:
            lines = file.readlines()
        
        modified_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            label = int(parts[0])
            if label in [0, 1]:
                parts[0] = '1'  # Change to label 1
            else:
                parts[0] = '0'
            modified_lines.append(' '.join(parts))
        if not modified_lines:
            print(f"No valid labels found in {label_file}, skipping.")
        else:
            with open(label_file, 'w') as file:
                file.write('\n'.join(modified_lines) + '\n')

# Example usage 
transform_labels('dataset/Vulnerable Detection.v3i.yolov12_labels_checked/valid/labels')
print("Labels' transformation complete!") 