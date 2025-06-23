#!/usr/bin/env python3
"""
Manual Labelling Guide and Quick Annotation Tool

This script provides:
1. A comprehensive guide for manual annotation
2. A quick tool to add helmet/jacket annotations to existing person labels
3. Templates and examples for different safety scenarios
4. Validation tools for annotation quality
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import os
import random
from typing import List, Dict, Tuple

class ManualLabellingGuide:
 """Guide and tools for manual annotation of safety equipment"""

 def __init__(self):
 self.setup_class_info()

 def setup_class_info(self):
 """Setup class information and guidelines"""
 self.classes = {
 0: {
 'name': 'helmet',
 'description': 'Safety helmet, hard hat, construction helmet',
 'guidelines': [
 'Include full helmet shape even if partially occluded',
 'Tight bounding box around helmet outline',
 'Do not include face or neck area',
 'Label even if helmet is tilted or at angle',
 'Include helmet visor but not loose straps'
 ],
 'common_mistakes': [
 'Box too large including face/shoulders',
 'Missing helmets in background/side of frame',
 'Confusing hard hats with caps/hats',
 'Not labeling partially visible helmets'
 ]
 },
 1: {
 'name': 'reflective_jacket',
 'description': 'High-visibility vest, reflective jacket, safety vest',
 'guidelines': [
 'Include the full visible torso area covered by vest',
 'Include reflective strips and logos',
 'Label even if jacket is open or loose',
 'Include sleeves if present',
 'Distinguish from regular clothing'
 ],
 'common_mistakes': [
 'Missing jackets worn over other clothing',
 'Confusing regular bright clothing with hi-vis',
 'Box too tight missing reflective strips',
 'Not labeling partially visible vests'
 ]
 },
 2: {
 'name': 'person',
 'description': 'Worker, human, person in the scene',
 'guidelines': [
 'Include full visible body from head to feet',
 'Tight box around person silhouette',
 'Include tools/equipment being held',
 'Label even if partially occluded',
 'Include person even if very small in distance'
 ],
 'common_mistakes': [
 'Box too loose including background',
 'Missing people in background',
 'Not labeling partially visible people',
 'Confusing mannequins or statues with people'
 ]
 }
 }

 def print_labelling_guide(self):
 """Print comprehensive labelling guidelines"""
 print("=" * 80)
 print("THERMAL POWER PLANT SAFETY DETECTION - LABELLING GUIDE")
 print("=" * 80)

 print("\n GENERAL PRINCIPLES:")
 print("• Accuracy over Speed: Take time to ensure correct labels")
 print("• Consistency: Apply same standards across all images")
 print("• Completeness: Label ALL visible instances, even small ones")
 print("• Context: Consider thermal plant environment specifics")

 print(f"\n CLASS DEFINITIONS:")
 for class_id, info in self.classes.items():
 print(f"\n{class_id}. {info['name'].upper()}")
 print(f" Description: {info['description']}")
 print(" Guidelines:")
 for guideline in info['guidelines']:
 print(f" • {guideline}")
 print(" Common Mistakes to Avoid:")
 for mistake in info['common_mistakes']:
 print(f" {mistake}")

 print(f"\n THERMAL POWER PLANT SPECIFIC CONSIDERATIONS:")

 environments = {
 "Boiler Area": {
 "requirements": "Helmet + Reflective Jacket mandatory",
 "challenges": "Steam effects, high contrast lighting, metal reflections",
 "tips": "Look for PPE even in steamy/bright conditions"
 },
 "Control Room": {
 "requirements": "Variable (check plant policy)",
 "challenges": "Indoor lighting, computer screens",
 "tips": "Focus on entry/exit areas for PPE compliance"
 },
 "Switchyard": {
 "requirements": "Helmet + Reflective Jacket + Arc flash protection",
 "challenges": "Distance shots, electrical equipment occlusion",
 "tips": "Workers may be small but still need labeling"
 },
 "Coal Handling": {
 "requirements": "Helmet + Reflective Jacket + Dust protection",
 "challenges": "Dust clouds, dark conditions, bulk material",
 "tips": "Look for silhouettes and reflective strips"
 }
 }

 for area, details in environments.items():
 print(f"\n {area}:")
 print(f" Requirements: {details['requirements']}")
 print(f" Challenges: {details['challenges']}")
 print(f" Tips: {details['tips']}")

 print(f"\n ANNOTATION FORMAT (YOLO):")
 print("Format: class_id x_center y_center width height")
 print("• All coordinates normalized to [0, 1]")
 print("• x_center, y_center: center point of bounding box")
 print("• width, height: width and height of bounding box")
 print("• One annotation per line")

 print(f"\n QUALITY CHECKLIST:")
 quality_checks = [
 " All persons in image labeled?",
 " All helmets labeled (even partially visible)?",
 " All reflective jackets labeled?",
 " Bounding boxes tight but complete?",
 " No overlapping labels for same object?",
 " Coordinates within [0, 1] range?",
 " Class IDs correct (0=helmet, 1=jacket, 2=person)?",
 " File saved with same name as image?"
 ]

 for check in quality_checks:
 print(f" {check}")

 print(f"\n RECOMMENDED TOOLS:")
 tools = [
 "LabelImg: https://github.com/heartexlabs/labelImg",
 "CVAT: https://github.com/openvinotoolkit/cvat",
 "Label Studio: https://labelstud.io/",
 "Roboflow: https://roboflow.com/",
 "VGG Image Annotator (VIA): https://www.robots.ox.ac.uk/~vgg/software/via/"
 ]

 for tool in tools:
 print(f" • {tool}")

 def create_annotation_examples(self, output_dir="data/annotation_examples"):
 """Create visual examples of good annotations"""

 Path(output_dir).mkdir(parents=True, exist_ok=True)

 # Example scenarios for thermal power plant
 examples = {
 "compliant_worker_boiler.txt": {
 "description": "Worker in boiler area with full PPE",
 "annotations": [
 "2 0.45 0.65 0.35 0.7", # person (center-left, full body)
 "0 0.45 0.28 0.18 0.22", # helmet (head area)
 "1 0.45 0.52 0.32 0.45" # reflective_jacket (torso)
 ],
 "scenario": "compliant"
 },

 "violation_worker_control.txt": {
 "description": "Worker in control room without helmet",
 "annotations": [
 "2 0.6 0.55 0.28 0.8", # person (right side)
 "1 0.6 0.45 0.25 0.4" # reflective_jacket only (missing helmet)
 ],
 "scenario": "violation"
 },

 "multiple_workers_maintenance.txt": {
 "description": "Multiple workers with varying compliance",
 "annotations": [
 # Worker 1 - Compliant
 "2 0.25 0.6 0.22 0.65", # person 1
 "0 0.25 0.32 0.15 0.18", # helmet 1
 "1 0.25 0.48 0.2 0.35", # jacket 1

 # Worker 2 - Missing helmet
 "2 0.55 0.7 0.25 0.6", # person 2
 "1 0.55 0.58 0.22 0.32", # jacket 2 (no helmet)

 # Worker 3 - Distant but compliant
 "2 0.8 0.45 0.15 0.35", # person 3 (smaller, background)
 "0 0.8 0.32 0.08 0.1", # helmet 3 (smaller)
 "1 0.8 0.42 0.12 0.2" # jacket 3 (smaller)
 ],
 "scenario": "mixed_compliance"
 },

 "challenging_steam_conditions.txt": {
 "description": "Workers in steamy boiler area - challenging visibility",
 "annotations": [
 "2 0.4 0.55 0.3 0.7", # person (partially obscured by steam)
 "0 0.4 0.28 0.16 0.2", # helmet (visible through steam)
 "1 0.4 0.48 0.28 0.4" # reflective jacket (strips visible)
 ],
 "scenario": "challenging_conditions"
 },

 "distant_switchyard_workers.txt": {
 "description": "Workers at distance in switchyard",
 "annotations": [
 "2 0.15 0.75 0.12 0.3", # distant worker 1
 "0 0.15 0.65 0.06 0.08", # small helmet 1
 "1 0.15 0.72 0.1 0.15", # small jacket 1

 "2 0.85 0.68 0.1 0.25", # distant worker 2
 "0 0.85 0.6 0.05 0.07" # small helmet 2 (missing jacket)
 ],
 "scenario": "distant_workers"
 }
 }

 # Save examples with detailed explanations
 for filename, example in examples.items():
 filepath = Path(output_dir) / filename

 with open(filepath, 'w') as f:
 f.write(f"# {example['description']}\n")
 f.write(f"# Scenario: {example['scenario']}\n")
 f.write("# Format: class_id x_center y_center width height\n")
 f.write("# Classes: 0=helmet, 1=reflective_jacket, 2=person\n\n")

 for annotation in example['annotations']:
 f.write(f"{annotation}\n")

 # Create explanation file
 explanation_file = Path(output_dir) / "examples_explanation.md"
 with open(explanation_file, 'w') as f:
 f.write("# Annotation Examples Explanation\n\n")
 f.write("This directory contains example annotations for different scenarios in thermal power plants.\n\n")

 for filename, example in examples.items():
 f.write(f"## {filename}\n")
 f.write(f"**Description:** {example['description']}\n\n")
 f.write(f"**Scenario Type:** {example['scenario']}\n\n")
 f.write("**Annotations:**\n")
 for i, annotation in enumerate(example['annotations'], 1):
 parts = annotation.split()
 class_id = int(parts[0])
 class_name = self.classes[class_id]['name']
 f.write(f"{i}. `{annotation}` - {class_name}\n")
 f.write("\n")

 print(f"Annotation examples created in {output_dir}")
 print(f"See {explanation_file} for detailed explanations")

 def quick_safety_equipment_annotator(self, person_labels_dir, output_dir, sample_size=10):
 """
 Quick tool to add helmet and jacket annotations to existing person labels
 This is a semi-automated approach for rapid dataset enhancement
 """

 print(" Quick Safety Equipment Annotator")
 print("This tool helps add helmet/jacket annotations to existing person labels")

 person_labels_path = Path(person_labels_dir)
 output_path = Path(output_dir)
 output_path.mkdir(parents=True, exist_ok=True)

 # Get all label files
 label_files = list(person_labels_path.glob("*.txt"))

 if not label_files:
 print(f"No label files found in {person_labels_dir}")
 return

 # Sample random files for manual annotation
 sample_files = random.sample(label_files, min(sample_size, len(label_files)))

 print(f"Selected {len(sample_files)} files for annotation enhancement")

 for i, label_file in enumerate(sample_files, 1):
 print(f"\n Processing {i}/{len(sample_files)}: {label_file.name}")

 # Read existing annotations
 with open(label_file, 'r') as f:
 lines = f.readlines()

 persons = []
 for line in lines:
 line = line.strip()
 if line and not line.startswith('#'):
 parts = line.split()
 if len(parts) == 5 and int(parts[0]) == 2: # person class
 persons.append({
 'class_id': int(parts[0]),
 'x_center': float(parts[1]),
 'y_center': float(parts[2]),
 'width': float(parts[3]),
 'height': float(parts[4])
 })

 if not persons:
 print(f" No persons found in {label_file.name}")
 continue

 # Create enhanced annotations
 enhanced_annotations = []

 for j, person in enumerate(persons):
 print(f" Person {j+1}: center=({person['x_center']:.3f}, {person['y_center']:.3f})")

 # Add person annotation
 enhanced_annotations.append(
 f"2 {person['x_center']:.6f} {person['y_center']:.6f} {person['width']:.6f} {person['height']:.6f}"
 )

 # Generate likely helmet position (above person center)
 helmet_scenarios = self._generate_helmet_scenarios(person)
 jacket_scenarios = self._generate_jacket_scenarios(person)

 # For automation, we'll create multiple scenario files
 # In practice, this would be manual selection
 scenarios = [
 ("compliant", helmet_scenarios[0], jacket_scenarios[0]),
 ("missing_helmet", None, jacket_scenarios[0]),
 ("missing_jacket", helmet_scenarios[0], None),
 ("violation", None, None)
 ]

 # Create a compliant scenario by default (most common in good operations)
 scenario_name, helmet, jacket = scenarios[0] # compliant

 if helmet:
 enhanced_annotations.append(
 f"0 {helmet['x_center']:.6f} {helmet['y_center']:.6f} {helmet['width']:.6f} {helmet['height']:.6f}"
 )

 if jacket:
 enhanced_annotations.append(
 f"1 {jacket['x_center']:.6f} {jacket['y_center']:.6f} {jacket['width']:.6f} {jacket['height']:.6f}"
 )

 # Save enhanced annotations
 output_file = output_path / f"enhanced_{label_file.name}"
 with open(output_file, 'w') as f:
 f.write("# Enhanced with helmet and jacket annotations\n")
 f.write("# Classes: 0=helmet, 1=reflective_jacket, 2=person\n")
 f.write("# NOTE: This is auto-generated and requires manual review!\n\n")
 for annotation in enhanced_annotations:
 f.write(f"{annotation}\n")

 print(f" Enhanced annotations saved to {output_file}")

 print(f"\n Enhancement complete!")
 print(f" Enhanced files saved in: {output_path}")
 print(f" IMPORTANT: These are auto-generated annotations that need manual review!")
 print(f" Use a labeling tool to verify and correct the helmet/jacket positions.")

 def _generate_helmet_scenarios(self, person):
 """Generate likely helmet positions for a person"""
 # Helmet typically at top 15-25% of person bounding box
 helmet_height_ratio = 0.15 # helmet height relative to person height
 helmet_width_ratio = 0.12 # helmet width relative to person width

 # Calculate helmet position (top of person)
 helmet_y_offset = -person['height'] * 0.35 # slightly above person center

 helmet = {
 'x_center': person['x_center'],
 'y_center': person['y_center'] + helmet_y_offset,
 'width': person['width'] * helmet_width_ratio,
 'height': person['height'] * helmet_height_ratio
 }

 # Ensure coordinates are within bounds
 helmet['x_center'] = max(0, min(1, helmet['x_center']))
 helmet['y_center'] = max(0, min(1, helmet['y_center']))
 helmet['width'] = max(0.01, min(0.5, helmet['width']))
 helmet['height'] = max(0.01, min(0.5, helmet['height']))

 return [helmet] # Return as list for consistency

 def _generate_jacket_scenarios(self, person):
 """Generate likely reflective jacket positions for a person"""
 # Jacket typically covers torso (middle section of person)
 jacket_height_ratio = 0.4 # jacket height relative to person height
 jacket_width_ratio = 0.8 # jacket width relative to person width

 # Calculate jacket position (center-upper torso of person)
 jacket_y_offset = -person['height'] * 0.1 # slightly above person center

 jacket = {
 'x_center': person['x_center'],
 'y_center': person['y_center'] + jacket_y_offset,
 'width': person['width'] * jacket_width_ratio,
 'height': person['height'] * jacket_height_ratio
 }

 # Ensure coordinates are within bounds
 jacket['x_center'] = max(0, min(1, jacket['x_center']))
 jacket['y_center'] = max(0, min(1, jacket['y_center']))
 jacket['width'] = max(0.01, min(1, jacket['width']))
 jacket['height'] = max(0.01, min(1, jacket['height']))

 return [jacket] # Return as list for consistency

def main():
 parser = argparse.ArgumentParser(description='Manual Labelling Guide and Tools')
 parser.add_argument('--mode', choices=['guide', 'examples', 'enhance'],
 default='guide', help='Operation mode')
 parser.add_argument('--person_labels_dir', default='data/auto_labels',
 help='Directory with existing person labels')
 parser.add_argument('--output_dir', default='data/enhanced_labels',
 help='Output directory for enhanced labels')
 parser.add_argument('--sample_size', type=int, default=10,
 help='Number of files to enhance (for testing)')

 args = parser.parse_args()

 guide = ManualLabellingGuide()

 if args.mode == 'guide':
 guide.print_labelling_guide()

 elif args.mode == 'examples':
 guide.create_annotation_examples()

 elif args.mode == 'enhance':
 guide.quick_safety_equipment_annotator(
 args.person_labels_dir,
 args.output_dir,
 args.sample_size
 )

if __name__ == "__main__":
 main()
