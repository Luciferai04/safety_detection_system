#!/usr/bin/env python3
"""
Dataset Organization Script for Safety Detection System

This script properly organizes the dataset by:
1. Creating train/validation/test splits
2. Copying corresponding images and labels
3. Creating dataset.yaml for YOLO training
4. Generating dataset statistics
"""

import os
import shutil
import random
import yaml
import json
from pathlib import Path
import argparse
from collections import defaultdict
import logging

class DatasetOrganizer:
    """Organizes the safety detection dataset for YOLO training"""
    
    def __init__(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """
        Initialize dataset organizer
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation  
            test_ratio: Proportion of data for testing
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
            
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def organize_dataset(self, images_dir, labels_dir, output_base_dir):
        """
        Organize dataset into train/val/test splits
        
        Args:
            images_dir: Directory containing training images
            labels_dir: Directory containing corresponding labels
            output_base_dir: Base directory for organized dataset
        """
        images_path = Path(images_dir)
        labels_path = Path(labels_dir)
        output_path = Path(output_base_dir)
        
        # Create output directory structure
        for split in ['train', 'val', 'test']:
            (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_path.glob(f"*{ext}"))
            image_files.extend(images_path.glob(f"*{ext.upper()}"))
        
        # Filter images that have corresponding labels
        labeled_images = []
        unlabeled_images = []
        
        for img_file in image_files:
            label_file = labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                labeled_images.append(img_file)
            else:
                unlabeled_images.append(img_file)
        
        self.logger.info(f"Found {len(image_files)} total images")
        self.logger.info(f"  - {len(labeled_images)} images with labels")
        self.logger.info(f"  - {len(unlabeled_images)} images without labels")
        
        if len(labeled_images) == 0:
            self.logger.error("No labeled images found! Cannot proceed with dataset organization.")
            return None
        
        # Shuffle labeled images for random splits
        random.shuffle(labeled_images)
        
        # Calculate split sizes
        total_labeled = len(labeled_images)
        train_size = int(total_labeled * self.train_ratio)
        val_size = int(total_labeled * self.val_ratio)
        test_size = total_labeled - train_size - val_size
        
        # Create splits
        splits = {
            'train': labeled_images[:train_size],
            'val': labeled_images[train_size:train_size + val_size],
            'test': labeled_images[train_size + val_size:]
        }
        
        self.logger.info(f"Dataset splits:")
        self.logger.info(f"  - Train: {len(splits['train'])} images")
        self.logger.info(f"  - Validation: {len(splits['val'])} images")
        self.logger.info(f"  - Test: {len(splits['test'])} images")
        
        # Copy files to respective directories
        dataset_stats = {}
        for split_name, files in splits.items():
            self.logger.info(f"Processing {split_name} split...")
            
            split_stats = self._copy_split_files(
                files, labels_path, output_path, split_name
            )
            dataset_stats[split_name] = split_stats
        
        # Create dataset.yaml for YOLO
        dataset_yaml_path = self._create_dataset_yaml(output_path)
        
        # Generate comprehensive dataset statistics
        overall_stats = self._generate_dataset_statistics(
            output_path, dataset_stats, unlabeled_images
        )
        
        # Save statistics
        stats_file = output_path / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(overall_stats, f, indent=2)
        
        self.logger.info(f"Dataset organization complete!")
        self.logger.info(f"Dataset saved to: {output_path}")
        self.logger.info(f"Dataset YAML: {dataset_yaml_path}")
        self.logger.info(f"Statistics: {stats_file}")
        
        return overall_stats
        
    def _copy_split_files(self, image_files, labels_path, output_path, split_name):
        """Copy image and label files for a specific split"""
        split_stats = {
            'image_count': 0,
            'annotation_count': 0,
            'class_distribution': defaultdict(int),
            'files_copied': []
        }
        
        for img_file in image_files:
            # Copy image
            dest_img_dir = output_path / 'images' / split_name
            dest_img_path = dest_img_dir / img_file.name
            shutil.copy2(img_file, dest_img_path)
            split_stats['image_count'] += 1
            
            # Copy corresponding label
            label_file = labels_path / f"{img_file.stem}.txt"
            if label_file.exists():
                dest_label_dir = output_path / 'labels' / split_name
                dest_label_path = dest_label_dir / label_file.name
                shutil.copy2(label_file, dest_label_path)
                
                # Count annotations and class distribution
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                split_stats['class_distribution'][class_id] += 1
                                split_stats['annotation_count'] += 1
                
                split_stats['files_copied'].append({
                    'image': str(dest_img_path),
                    'label': str(dest_label_path)
                })
        
        return split_stats
        
    def _create_dataset_yaml(self, output_path):
        """Create YOLO dataset configuration file"""
        
        dataset_config = {
            'path': str(output_path.resolve()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            
            # Number of classes
            'nc': 3,
            
            # Class names (in order of class IDs)
            'names': {
                0: 'helmet',
                1: 'reflective_jacket',
                2: 'person'
            }
        }
        
        yaml_path = output_path / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
        
        self.logger.info(f"Created dataset.yaml: {yaml_path}")
        return yaml_path
        
    def _generate_dataset_statistics(self, output_path, dataset_stats, unlabeled_images):
        """Generate comprehensive dataset statistics"""
        
        # Class names for reference
        class_names = {0: 'helmet', 1: 'reflective_jacket', 2: 'person'}
        
        # Calculate totals
        total_images = sum(stats['image_count'] for stats in dataset_stats.values())
        total_annotations = sum(stats['annotation_count'] for stats in dataset_stats.values())
        
        # Overall class distribution
        overall_class_dist = defaultdict(int)
        for split_stats in dataset_stats.values():
            for class_id, count in split_stats['class_distribution'].items():
                overall_class_dist[class_id] += count
        
        # Convert to named distribution
        class_distribution_named = {}
        for class_id, count in overall_class_dist.items():
            class_name = class_names.get(class_id, f"unknown_class_{class_id}")
            class_distribution_named[class_name] = count
        
        # Calculate class distribution percentages
        class_percentages = {}
        for class_name, count in class_distribution_named.items():
            percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
            class_percentages[class_name] = round(percentage, 1)
        
        # Dataset quality metrics
        avg_annotations_per_image = total_annotations / total_images if total_images > 0 else 0
        
        # Split statistics with named classes
        split_stats_named = {}
        for split_name, stats in dataset_stats.items():
            split_class_dist = {}
            for class_id, count in stats['class_distribution'].items():
                class_name = class_names.get(class_id, f"unknown_class_{class_id}")
                split_class_dist[class_name] = count
            
            split_stats_named[split_name] = {
                'image_count': stats['image_count'],
                'annotation_count': stats['annotation_count'],
                'class_distribution': split_class_dist,
                'avg_annotations_per_image': stats['annotation_count'] / stats['image_count'] if stats['image_count'] > 0 else 0
            }
        
        overall_stats = {
            'dataset_summary': {
                'total_labeled_images': total_images,
                'total_unlabeled_images': len(unlabeled_images),
                'total_annotations': total_annotations,
                'avg_annotations_per_image': round(avg_annotations_per_image, 2),
                'number_of_classes': len(class_distribution_named)
            },
            'class_distribution': class_distribution_named,
            'class_percentages': class_percentages,
            'split_distribution': {
                'train': {
                    'images': dataset_stats['train']['image_count'],
                    'percentage': round(dataset_stats['train']['image_count'] / total_images * 100, 1) if total_images > 0 else 0
                },
                'val': {
                    'images': dataset_stats['val']['image_count'],
                    'percentage': round(dataset_stats['val']['image_count'] / total_images * 100, 1) if total_images > 0 else 0
                },
                'test': {
                    'images': dataset_stats['test']['image_count'],
                    'percentage': round(dataset_stats['test']['image_count'] / total_images * 100, 1) if total_images > 0 else 0
                }
            },
            'split_statistics': split_stats_named,
            'data_quality': {
                'labelling_completeness': round(total_images / (total_images + len(unlabeled_images)) * 100, 1) if (total_images + len(unlabeled_images)) > 0 else 0,
                'min_images_per_class_recommended': 50,
                'current_min_images_per_class': min(class_distribution_named.values()) if class_distribution_named else 0,
                'dataset_ready_for_training': min(class_distribution_named.values()) >= 10 if class_distribution_named else False
            },
            'recommendations': self._generate_recommendations(class_distribution_named, total_images, len(unlabeled_images))
        }
        
        return overall_stats
        
    def _generate_recommendations(self, class_dist, labeled_count, unlabeled_count):
        """Generate recommendations for improving the dataset"""
        recommendations = []
        
        # Check minimum samples per class
        min_recommended = 50
        for class_name, count in class_dist.items():
            if count < min_recommended:
                recommendations.append(f"Increase {class_name} samples to at least {min_recommended} (currently {count})")
        
        # Check overall dataset size
        if labeled_count < 100:
            recommendations.append(f"Increase total labeled images to at least 100 (currently {labeled_count})")
        
        # Check unlabeled images
        if unlabeled_count > labeled_count:
            recommendations.append(f"Label more images: {unlabeled_count} unlabeled images available")
        
        # Check class balance
        if class_dist:
            max_count = max(class_dist.values())
            min_count = min(class_dist.values())
            if max_count > min_count * 3:  # More than 3x imbalance
                recommendations.append("Consider balancing class distribution - significant class imbalance detected")
        
        # Training readiness
        if labeled_count < 50:
            recommendations.append("Dataset too small for reliable training - aim for at least 50-100 labeled images")
        elif labeled_count < 200:
            recommendations.append("Dataset adequate for initial training but could benefit from more samples")
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(description='Organize Safety Detection Dataset')
    parser.add_argument('--images_dir', default='data/images/train',
                       help='Directory containing training images')
    parser.add_argument('--labels_dir', default='data/labels/train',
                       help='Directory containing training labels')
    parser.add_argument('--output_dir', default='data/organized_dataset',
                       help='Output directory for organized dataset')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Proportion of data for training (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Proportion of data for validation (default: 0.2)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Proportion of data for testing (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible splits')
    
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    random.seed(args.seed)
    
    # Initialize organizer
    organizer = DatasetOrganizer(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # Organize dataset
    stats = organizer.organize_dataset(
        args.images_dir,
        args.labels_dir,
        args.output_dir
    )
    
    if stats:
        print(f"\n=== DATASET ORGANIZATION SUMMARY ===")
        print(f"Total labeled images: {stats['dataset_summary']['total_labeled_images']}")
        print(f"Total unlabeled images: {stats['dataset_summary']['total_unlabeled_images']}")
        print(f"Total annotations: {stats['dataset_summary']['total_annotations']}")
        print(f"Average annotations per image: {stats['dataset_summary']['avg_annotations_per_image']}")
        
        print(f"\n=== CLASS DISTRIBUTION ===")
        for class_name, count in stats['class_distribution'].items():
            percentage = stats['class_percentages'][class_name]
            print(f"{class_name}: {count} ({percentage}%)")
        
        print(f"\n=== RECOMMENDATIONS ===")
        for recommendation in stats['recommendations']:
            print(f"- {recommendation}")
        
        print(f"\nDataset ready for training: {'Yes' if stats['data_quality']['dataset_ready_for_training'] else 'No'}")

if __name__ == "__main__":
    main()
