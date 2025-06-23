#!/usr/bin/env python3
"""
Convert sample annotations from custom format to YOLO format.
Custom format: line_number|class_id x_center y_center width height
YOLO format: class_id x_center y_center width height

Class mapping conversion:
Old: 0=helmet, 1=person
New: 0=helmet, 1=reflective_jacket, 2=person
"""

import os
import glob
from pathlib import Path

def convert_annotation_file(input_file, output_file):
    """Convert a single annotation file from custom format to YOLO format."""
    yolo_lines = []
    
    # Class mapping: old -> new
    # Old: 0=helmet, 1=person
    # New: 0=helmet, 1=reflective_jacket, 2=person
    class_mapping = {0: 0, 1: 2}  # helmet stays 0, person becomes 2
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse direct YOLO format: class_id x_center y_center width height
            parts = line.split()
            if len(parts) == 5:
                old_class_id = int(parts[0])
                new_class_id = class_mapping.get(old_class_id, old_class_id)
                
                # Create new annotation with mapped class ID
                new_annotation = f"{new_class_id} {' '.join(parts[1:])}"
                yolo_lines.append(new_annotation)
    
    # Write to YOLO format file
    with open(output_file, 'w') as f:
        for yolo_line in yolo_lines:
            f.write(yolo_line + '\n')
    
    print(f"Converted {input_file} -> {output_file} ({len(yolo_lines)} annotations)")

def main():
    # Define paths
    sample_dir = Path("data/sample_annotations")
    train_labels_dir = Path("data/labels/train")
    
    # Ensure output directory exists
    train_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .txt files in sample_annotations
    annotation_files = list(sample_dir.glob("*.txt"))
    
    if not annotation_files:
        print("No annotation files found in data/sample_annotations/")
        return
    
    print(f"Found {len(annotation_files)} annotation files to convert:")
    
    for input_file in annotation_files:
        # Create corresponding output filename
        output_file = train_labels_dir / input_file.name
        convert_annotation_file(input_file, output_file)
    
    print(f"\nConversion complete! YOLO labels saved to {train_labels_dir}")
    print("\nClass mapping:")
    print("0: helmet")
    print("1: reflective_jacket") 
    print("2: person")

if __name__ == "__main__":
    main()
