#!/usr/bin/env python3
"""
Automated Labelling Assistant for Safety Detection System

This script helps with the critical labelling gap by:
1. Using a pre-trained YOLO model to generate initial labels
2. Providing a framework for manual annotation review
3. Creating batch annotation tools
4. Generating statistics on labelling progress

Note: This provides a starting point but manual review is essential for accuracy.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import json
import argparse
import os
from tqdm import tqdm
import logging

class AutoLabelAssistant:
    """Automated labelling assistant for safety equipment detection"""
    
    def __init__(self, confidence_threshold=0.3):
        """
        Initialize the auto-labelling assistant
        
        Args:
            confidence_threshold: Lower threshold for initial detection suggestions
        """
        self.confidence_threshold = confidence_threshold
        self.setup_logging()
        
        # Load pre-trained YOLO model for initial suggestions
        self.model = YOLO('yolov8n.pt')  # Start with general object detection
        
        # Class mapping for safety equipment
        # COCO classes that might correspond to our safety classes
        self.coco_to_safety_mapping = {
            0: 2,   # person -> person (class 2)
            # Note: COCO doesn't have helmet or reflective_jacket classes
            # We'll need manual annotation for these
        }
        
        # Our safety classes
        self.safety_classes = {
            0: 'helmet',
            1: 'reflective_jacket', 
            2: 'person'
        }
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def generate_initial_labels(self, images_dir, output_dir, batch_size=50):
        """
        Generate initial label suggestions using pre-trained YOLO
        
        Args:
            images_dir: Directory containing images to label
            output_dir: Directory to save generated labels
            batch_size: Number of images to process at once
        """
        images_path = Path(images_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_path.glob(f"*{ext}"))
            image_files.extend(images_path.glob(f"*{ext.upper()}"))
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        # Statistics
        stats = {
            'total_images': len(image_files),
            'images_with_persons': 0,
            'images_processed': 0,
            'labels_generated': 0
        }
        
        # Process images in batches
        for i in tqdm(range(0, len(image_files), batch_size), desc="Processing batches"):
            batch_files = image_files[i:i + batch_size]
            
            for img_file in tqdm(batch_files, desc=f"Batch {i//batch_size + 1}", leave=False):
                try:
                    # Run YOLO detection
                    results = self.model(str(img_file), conf=self.confidence_threshold, verbose=False)
                    
                    # Process detections
                    annotations = self._process_detections(results[0], img_file)
                    
                    # Save annotations if any found
                    if annotations:
                        label_file = output_path / f"{img_file.stem}.txt"
                        self._save_annotations(annotations, label_file)
                        stats['labels_generated'] += 1
                        
                        # Check if persons were detected
                        if any(ann['class_id'] == 2 for ann in annotations):
                            stats['images_with_persons'] += 1
                    
                    stats['images_processed'] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error processing {img_file}: {e}")
                    continue
        
        # Save statistics
        stats_file = output_path / "labelling_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Auto-labelling complete!")
        self.logger.info(f"Images processed: {stats['images_processed']}")
        self.logger.info(f"Labels generated: {stats['labels_generated']}")
        self.logger.info(f"Images with persons: {stats['images_with_persons']}")
        
        return stats
        
    def _process_detections(self, result, img_file):
        """Process YOLO detection results"""
        annotations = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            # Get image dimensions for normalization
            img = cv2.imread(str(img_file))
            img_height, img_width = img.shape[:2]
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                coco_class_id = int(cls_id)
                
                # Map COCO class to our safety classes
                if coco_class_id in self.coco_to_safety_mapping:
                    safety_class_id = self.coco_to_safety_mapping[coco_class_id]
                    
                    # Convert to YOLO format (normalized coordinates)
                    x_center = (box[0] + box[2]) / 2 / img_width
                    y_center = (box[1] + box[3]) / 2 / img_height
                    width = (box[2] - box[0]) / img_width
                    height = (box[3] - box[1]) / img_height
                    
                    annotation = {
                        'class_id': safety_class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'confidence': float(conf),
                        'source': 'auto_generated'
                    }
                    
                    annotations.append(annotation)
        
        return annotations
        
    def _save_annotations(self, annotations, label_file):
        """Save annotations in YOLO format"""
        with open(label_file, 'w') as f:
            for ann in annotations:
                line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n"
                f.write(line)
                
    def create_labelling_plan(self, images_dir, output_file="labelling_plan.json"):
        """
        Create a comprehensive labelling plan for the dataset
        """
        images_path = Path(images_dir)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_path.glob(f"*{ext}"))
            image_files.extend(images_path.glob(f"*{ext.upper()}"))
        
        # Analyze current labelling status
        labels_dir = Path("data/labels/train")
        labeled_images = set()
        if labels_dir.exists():
            for label_file in labels_dir.glob("*.txt"):
                labeled_images.add(label_file.stem)
        
        # Create labelling plan
        total_images = len(image_files)
        labeled_count = len(labeled_images)
        unlabeled_count = total_images - labeled_count
        
        plan = {
            'dataset_summary': {
                'total_images': total_images,
                'labeled_images': labeled_count,
                'unlabeled_images': unlabeled_count,
                'labelling_progress': (labeled_count / total_images * 100) if total_images > 0 else 0
            },
            'labelling_strategy': {
                'phase_1': {
                    'description': "Auto-generate initial person detections",
                    'target_images': min(unlabeled_count, 500),
                    'method': "YOLO pre-trained model",
                    'priority': "High"
                },
                'phase_2': {
                    'description': "Manual annotation of safety equipment (helmets, jackets)",
                    'target_images': 200,
                    'method': "Manual annotation tools (LabelImg, CVAT)",
                    'priority': "Critical"
                },
                'phase_3': {
                    'description': "Review and correct auto-generated labels", 
                    'target_images': 300,
                    'method': "Manual review and correction",
                    'priority': "High"
                },
                'phase_4': {
                    'description': "Quality assurance and validation",
                    'target_images': 100,
                    'method': "Random sampling and verification",
                    'priority': "Medium"
                }
            },
            'recommended_tools': [
                "LabelImg - https://github.com/heartexlabs/labelImg",
                "CVAT - https://github.com/openvinotoolkit/cvat", 
                "Label Studio - https://labelstud.io/",
                "Roboflow - https://roboflow.com/"
            ],
            'minimum_viable_dataset': {
                'total_required': 100,
                'per_class_minimum': {
                    'helmet': 50,
                    'reflective_jacket': 30,
                    'person': 80
                }
            }
        }
        
        # Save plan
        with open(output_file, 'w') as f:
            json.dump(plan, f, indent=2)
        
        self.logger.info(f"Labelling plan saved to {output_file}")
        self.logger.info(f"Current progress: {plan['dataset_summary']['labelling_progress']:.1f}%")
        
        return plan
        
    def create_annotation_template(self, sample_images_dir, template_count=10):
        """
        Create annotation templates for common scenarios in thermal power plants
        """
        template_dir = Path("data/annotation_templates")
        template_dir.mkdir(parents=True, exist_ok=True)
        
        # Common thermal power plant scenarios
        scenarios = {
            'boiler_area_compliant': {
                'description': 'Worker in boiler area with full PPE',
                'required_equipment': ['helmet', 'reflective_jacket'],
                'sample_annotation': [
                    "2 0.5 0.6 0.3 0.8",    # person
                    "0 0.5 0.25 0.15 0.2",  # helmet
                    "1 0.5 0.5 0.25 0.4"    # reflective_jacket
                ]
            },
            'control_room_entry': {
                'description': 'Worker entering control room',
                'required_equipment': ['helmet'],
                'sample_annotation': [
                    "2 0.5 0.6 0.3 0.8",    # person
                    "0 0.5 0.25 0.15 0.2"   # helmet
                ]
            },
            'maintenance_crew': {
                'description': 'Multiple workers performing maintenance',
                'required_equipment': ['helmet', 'reflective_jacket'],
                'sample_annotation': [
                    "2 0.3 0.6 0.25 0.7",   # worker 1
                    "0 0.3 0.25 0.12 0.15", # helmet 1
                    "1 0.3 0.5 0.2 0.3",    # jacket 1
                    "2 0.7 0.6 0.25 0.7",   # worker 2
                    "0 0.7 0.25 0.12 0.15"  # helmet 2 (missing jacket - violation)
                ]
            },
            'safety_violation': {
                'description': 'Worker without required PPE',
                'required_equipment': [],
                'sample_annotation': [
                    "2 0.5 0.6 0.3 0.8"     # person only (violation)
                ]
            }
        }
        
        # Create template files
        for scenario_name, scenario_data in scenarios.items():
            template_file = template_dir / f"{scenario_name}_template.txt"
            with open(template_file, 'w') as f:
                f.write(f"# {scenario_data['description']}\n")
                f.write(f"# Required equipment: {', '.join(scenario_data['required_equipment'])}\n")
                f.write("# Format: class_id x_center y_center width height\n")
                f.write("# Classes: 0=helmet, 1=reflective_jacket, 2=person\n\n")
                for annotation in scenario_data['sample_annotation']:
                    f.write(f"{annotation}\n")
        
        self.logger.info(f"Annotation templates created in {template_dir}")
        
    def validate_existing_labels(self, labels_dir):
        """
        Validate existing label files for correctness
        """
        labels_path = Path(labels_dir)
        if not labels_path.exists():
            self.logger.warning(f"Labels directory {labels_dir} does not exist")
            return
        
        validation_results = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'errors': []
        }
        
        for label_file in labels_path.glob("*.txt"):
            validation_results['total_files'] += 1
            
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                file_valid = True
                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    parts = line.split()
                    if len(parts) != 5:
                        validation_results['errors'].append(f"{label_file.name}:{i} - Invalid format: {line}")
                        file_valid = False
                        continue
                    
                    # Validate class ID
                    try:
                        class_id = int(parts[0])
                        if class_id not in [0, 1, 2]:
                            validation_results['errors'].append(f"{label_file.name}:{i} - Invalid class ID: {class_id}")
                            file_valid = False
                    except ValueError:
                        validation_results['errors'].append(f"{label_file.name}:{i} - Non-numeric class ID: {parts[0]}")
                        file_valid = False
                    
                    # Validate coordinates
                    try:
                        coords = [float(x) for x in parts[1:]]
                        for j, coord in enumerate(coords):
                            if not 0 <= coord <= 1:
                                validation_results['errors'].append(f"{label_file.name}:{i} - Coordinate out of range [0,1]: {coord}")
                                file_valid = False
                    except ValueError:
                        validation_results['errors'].append(f"{label_file.name}:{i} - Non-numeric coordinates")
                        file_valid = False
                
                if file_valid:
                    validation_results['valid_files'] += 1
                else:
                    validation_results['invalid_files'] += 1
                    
            except Exception as e:
                validation_results['errors'].append(f"{label_file.name} - File error: {e}")
                validation_results['invalid_files'] += 1
        
        # Save validation results
        results_file = Path("data/label_validation_results.json")
        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        self.logger.info(f"Label validation complete:")
        self.logger.info(f"  Total files: {validation_results['total_files']}")
        self.logger.info(f"  Valid files: {validation_results['valid_files']}")
        self.logger.info(f"  Invalid files: {validation_results['invalid_files']}")
        self.logger.info(f"  Errors: {len(validation_results['errors'])}")
        
        return validation_results

def main():
    parser = argparse.ArgumentParser(description='Auto-labelling Assistant for Safety Detection')
    parser.add_argument('--mode', choices=['plan', 'auto_label', 'validate', 'templates'], 
                       default='plan', help='Operation mode')
    parser.add_argument('--images_dir', default='data/images/train',
                       help='Directory containing images')
    parser.add_argument('--labels_dir', default='data/labels/train', 
                       help='Directory containing labels')
    parser.add_argument('--output_dir', default='data/auto_labels',
                       help='Output directory for auto-generated labels')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold for auto-labelling')
    
    args = parser.parse_args()
    
    assistant = AutoLabelAssistant(confidence_threshold=args.confidence)
    
    if args.mode == 'plan':
        plan = assistant.create_labelling_plan(args.images_dir)
        print(f"\n=== LABELLING PLAN ===")
        print(f"Total images: {plan['dataset_summary']['total_images']}")
        print(f"Labeled: {plan['dataset_summary']['labeled_images']}")
        print(f"Unlabeled: {plan['dataset_summary']['unlabeled_images']}")
        print(f"Progress: {plan['dataset_summary']['labelling_progress']:.1f}%")
        
    elif args.mode == 'auto_label':
        stats = assistant.generate_initial_labels(args.images_dir, args.output_dir)
        
    elif args.mode == 'validate':
        results = assistant.validate_existing_labels(args.labels_dir)
        
    elif args.mode == 'templates':
        assistant.create_annotation_template(args.images_dir)

if __name__ == "__main__":
    main()
