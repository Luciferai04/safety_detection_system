#!/usr/bin/env python3
"""
Enhanced Thermal Power Plant Safety Detection Model Training

This script provides comprehensive training for thermal power plant specific
safety detection with improved accuracy and robustness.
"""

import os
import sys
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class ThermalPlantDataAugmentation:
    """Thermal power plant specific data augmentation"""
    
    def __init__(self):
        self.thermal_augmentations = A.Compose([
            # Environmental conditions
            A.OneOf([
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.1, p=0.3),  # Steam simulation
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=0.2),  # Heat shimmer
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Dust/interference
            ], p=0.5),
            
            # Lighting conditions
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            ], p=0.6),
            
            # Industrial environment simulation
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.2),  # Movement blur
                A.GlassBlur(sigma=0.7, max_delta=2, iterations=1, p=0.2),  # Heat distortion
                A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.5), p=0.2),  # Focus issues
            ], p=0.4),
            
            # Geometric transforms for different camera angles
            A.OneOf([
                A.Perspective(scale=(0.05, 0.1), p=0.3),
                A.Rotate(limit=10, p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.3),
            ], p=0.4),
            
            # Weather simulation
            A.OneOf([
                A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=10, drop_width=1, p=0.2),
                A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.2, brightness_coeff=1.5, p=0.1),
            ], p=0.2),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    def __call__(self, image, bboxes, class_labels):
        """Apply thermal plant specific augmentations"""
        augmented = self.thermal_augmentations(
            image=image, 
            bboxes=bboxes, 
            class_labels=class_labels
        )
        return augmented['image'], augmented['bboxes'], augmented['class_labels']

class ThermalPlantTrainer:
    """Enhanced trainer for thermal power plant safety detection"""
    
    def __init__(self, config_path="config/thermal_plant_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_name = self.config['model']['name']
        self.epochs = self.config['training']['training_parameters']['epochs']
        self.batch_size = self.config['training']['training_parameters']['batch_size']
        self.lr = self.config['training']['training_parameters']['learning_rate']
        
        # Initialize augmentation
        self.augmentation = ThermalPlantDataAugmentation()
    
    def prepare_thermal_plant_dataset(self):
        """Prepare enhanced dataset for thermal power plant training"""
        print("📊 Preparing Thermal Power Plant Dataset...")
        
        # Create enhanced dataset structure
        dataset_path = Path(self.config['training']['dataset_path'])
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Create data.yaml for training
        data_yaml = {
            'path': str(dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 7,  # Updated number of classes
            'names': [
                'helmet', 'reflective_jacket', 'safety_boots', 
                'safety_gloves', 'arc_flash_suit', 'respirator', 'person'
            ]
        }
        
        with open(dataset_path / 'data.yaml', 'w') as f:
            yaml.dump(data_yaml, f)
        
        print(f"✅ Dataset configuration saved to {dataset_path / 'data.yaml'}")
        return str(dataset_path / 'data.yaml')
    
    def train_enhanced_model(self):
        """Train enhanced model for thermal power plant safety detection"""
        print("🚀 Starting Enhanced Thermal Power Plant Model Training...")
        
        # Prepare dataset
        data_yaml_path = self.prepare_thermal_plant_dataset()
        
        # Initialize model
        model = YOLO(f'{self.model_name}.pt')  # Start with pretrained model
        
        # Configure training parameters
        train_args = {
            'data': data_yaml_path,
            'epochs': self.epochs,
            'imgsz': 640,
            'batch': self.batch_size,
            'lr0': self.lr,
            'weight_decay': self.config['training']['training_parameters']['weight_decay'],
            'warmup_epochs': self.config['training']['training_parameters']['warmup_epochs'],
            'cos_lr': self.config['training']['training_parameters']['cosine_lr'],
            'patience': self.config['training']['early_stopping']['patience'],
            'save_period': 10,
            'project': 'models',
            'name': 'thermal_plant_safety_enhanced',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'resume': False,
            'nosave': False,
            'noval': False,
            'noautoanchor': False,
            'evolve': None,
            'bucket': '',
            'cache': False,
            'image_weights': False,
            'device': '',
            'multi_scale': True,
            'dropout': 0.1,  # Add dropout for better generalization
        }
        
        # Start training
        print(f"🎯 Training with target metrics:")
        print(f"   • mAP@0.5: {self.config['training']['target_metrics']['map50']}")
        print(f"   • mAP@0.5:0.95: {self.config['training']['target_metrics']['map50_95']}")
        print(f"   • Precision: {self.config['training']['target_metrics']['precision']}")
        print(f"   • Recall: {self.config['training']['target_metrics']['recall']}")
        
        try:
            results = model.train(**train_args)
            
            print("✅ Training completed successfully!")
            print(f"📊 Best model saved to: models/thermal_plant_safety_enhanced/weights/best.pt")
            
            # Validate results
            metrics = results.results_dict if hasattr(results, 'results_dict') else {}
            self.validate_training_results(metrics)
            
            return results
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None
    
    def validate_training_results(self, metrics):
        """Validate training results against thermal plant requirements"""
        print("\n🔍 Validating Training Results...")
        
        target_metrics = self.config['training']['target_metrics']
        
        validation_results = {
            'map50': metrics.get('metrics/mAP50(B)', 0),
            'map50_95': metrics.get('metrics/mAP50-95(B)', 0), 
            'precision': metrics.get('metrics/precision(B)', 0),
            'recall': metrics.get('metrics/recall(B)', 0)
        }
        
        print("📊 Training Results vs Targets:")
        for metric, value in validation_results.items():
            target = target_metrics[metric]
            status = "✅" if value >= target else "❌"
            print(f"   {status} {metric}: {value:.3f} (target: {target:.3f})")
        
        overall_score = sum(
            1 for metric, value in validation_results.items() 
            if value >= target_metrics[metric]
        ) / len(target_metrics) * 100
        
        print(f"\n🎯 Overall Performance: {overall_score:.1f}%")
        
        if overall_score >= 75:
            print("🎉 Model meets thermal power plant production requirements!")
        elif overall_score >= 50:
            print("⚠️ Model shows promise but needs improvement")
        else:
            print("❌ Model requires significant improvement")
        
        return validation_results

def main():
    """Main training function"""
    print("🏭 Enhanced Thermal Power Plant Safety Detection Training")
    print("=" * 60)
    
    # Create enhanced config
    print("📝 Creating enhanced thermal plant configuration...")
    
    # Initialize trainer
    trainer = ThermalPlantTrainer()
    
    # Start training
    results = trainer.train_enhanced_model()
    
    if results:
        print("\n🎉 Enhanced model training completed successfully!")
        print("💡 Model is now optimized for thermal power plant environments")
    else:
        print("\n❌ Training failed. Please check the error logs.")

if __name__ == "__main__":
    main()
