#!/usr/bin/env python3
"""
Clean up label files by removing comment lines and empty lines
"""

import os
from pathlib import Path
import argparse

def clean_label_file(label_path):
    """Clean a single label file by removing comments and empty lines"""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    # Filter out comment lines and empty lines
    clean_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            clean_lines.append(line)
    
    # Write back the cleaned content
    with open(label_path, 'w') as f:
        for line in clean_lines:
            f.write(line + '\n')
    
    return len(clean_lines)

def clean_labels_directory(labels_dir):
    """Clean all label files in a directory"""
    labels_path = Path(labels_dir)
    
    if not labels_path.exists():
        print(f"Directory does not exist: {labels_dir}")
        return
    
    label_files = list(labels_path.glob("*.txt"))
    
    if not label_files:
        print(f"No .txt files found in {labels_dir}")
        return
    
    total_annotations = 0
    cleaned_files = 0
    
    for label_file in label_files:
        try:
            annotation_count = clean_label_file(label_file)
            total_annotations += annotation_count
            cleaned_files += 1
            print(f"Cleaned {label_file.name}: {annotation_count} annotations")
        except Exception as e:
            print(f"Error cleaning {label_file.name}: {e}")
    
    print(f"\nCleaning complete:")
    print(f"  Files processed: {cleaned_files}")
    print(f"  Total annotations: {total_annotations}")

def main():
    parser = argparse.ArgumentParser(description='Clean YOLO label files')
    parser.add_argument('--labels_dir', default='data/enhanced_labels',
                       help='Directory containing label files to clean')
    
    args = parser.parse_args()
    
    print(f"🧹 Cleaning label files in: {args.labels_dir}")
    clean_labels_directory(args.labels_dir)

if __name__ == "__main__":
    main()
