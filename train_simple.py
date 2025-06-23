#!/usr/bin/env python3
"""
Simple Training Script for Safety Detection Model

This script provides a straightforward way to train the YOLO model
with our organized dataset structure.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml
import logging

def setup_logging():
 """Setup basic logging"""
 logging.basicConfig(
 level=logging.INFO,
 format='%(asctime)s - %(levelname)s - %(message)s'
 )
 return logging.getLogger(__name__)

def train_safety_model(dataset_yaml_path, epochs=50, batch_size=8, model_size='n'):
 """
 Train safety detection model

 Args:
 dataset_yaml_path: Path to dataset YAML file
 epochs: Number of training epochs
 batch_size: Training batch size
 model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
 """
 logger = setup_logging()

 # Validate dataset YAML
 if not Path(dataset_yaml_path).exists():
 raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml_path}")

 # Load and validate dataset configuration
 with open(dataset_yaml_path, 'r') as f:
 dataset_config = yaml.safe_load(f)

 logger.info(f"Dataset configuration loaded from {dataset_yaml_path}")
 logger.info(f"Dataset path: {dataset_config['path']}")
 logger.info(f"Number of classes: {dataset_config['nc']}")
 logger.info(f"Classes: {dataset_config['names']}")

 # Check if dataset directories exist
 dataset_path = Path(dataset_config['path'])
 train_images = dataset_path / dataset_config['train']
 val_images = dataset_path / dataset_config['val']

 if not train_images.exists():
 raise FileNotFoundError(f"Training images directory not found: {train_images}")
 if not val_images.exists():
 raise FileNotFoundError(f"Validation images directory not found: {val_images}")

 # Count images
 train_count = len(list(train_images.glob("*.jpg"))) + len(list(train_images.glob("*.png")))
 val_count = len(list(val_images.glob("*.jpg"))) + len(list(val_images.glob("*.png")))

 logger.info(f"Training images: {train_count}")
 logger.info(f"Validation images: {val_count}")

 if train_count == 0:
 raise ValueError("No training images found!")

 # Initialize YOLO model
 model_name = f"yolov8{model_size}.pt"
 logger.info(f"Loading pre-trained model: {model_name}")

 model = YOLO(model_name)

 # Training configuration optimized for safety detection
 train_config = {
 'data': dataset_yaml_path,
 'epochs': epochs,
 'batch': batch_size,
 'imgsz': 640,
 'lr0': 0.01,
 'momentum': 0.937,
 'weight_decay': 0.0005,
 'warmup_epochs': 3,
 'warmup_momentum': 0.8,
 'warmup_bias_lr': 0.1,
 'device': 'cpu', # Use CPU (change to 'cuda' or '0' for GPU)
 'patience': 10,
 'save': True,
 'save_period': 10,
 'project': 'models',
 'name': 'safety_detection_training',
 'exist_ok': True,
 'pretrained': True,
 'optimizer': 'Adam',
 'verbose': True,

 # Data augmentation for safety equipment
 'hsv_h': 0.015, # Hue augmentation (lighting conditions)
 'hsv_s': 0.7, # Saturation augmentation
 'hsv_v': 0.4, # Value/brightness augmentation
 'degrees': 10, # Rotation (helmets can be at angles)
 'translate': 0.1, # Translation augmentation
 'scale': 0.5, # Scale augmentation
 'shear': 0.0, # No shearing (preserve helmet/jacket shapes)
 'perspective': 0.0, # No perspective (preserve equipment shapes)
 'flipud': 0.0, # No vertical flip (helmet orientation matters)
 'fliplr': 0.5, # Horizontal flip OK
 'mosaic': 1.0, # Mosaic augmentation
 'mixup': 0.0, # No mixup (preserve distinct objects)
 'copy_paste': 0.0, # No copy-paste for this use case
 }

 logger.info("Starting training with configuration:")
 for key, value in train_config.items():
 logger.info(f" {key}: {value}")

 # Start training
 logger.info("Training started...")

 try:
 results = model.train(**train_config)

 logger.info("Training completed successfully!")

 # Get the save directory from the model trainer results
 save_dir = Path("models/safety_detection_training") # Default location
 logger.info(f"Training results saved to: {save_dir}")

 # Save final model to models directory
 models_dir = Path("models")
 models_dir.mkdir(exist_ok=True)

 final_model_path = models_dir / "safety_detection_best.pt"
 best_weights = save_dir / "weights" / "best.pt"

 if best_weights.exists():
 import shutil
 shutil.copy2(best_weights, final_model_path)
 logger.info(f"Best model copied to: {final_model_path}")

 return results

 except Exception as e:
 logger.error(f"Training failed: {e}")
 raise

def main():
 parser = argparse.ArgumentParser(description='Train Safety Detection Model')
 parser.add_argument('--dataset', type=str, required=True,
 help='Path to dataset YAML file')
 parser.add_argument('--epochs', type=int, default=50,
 help='Number of training epochs (default: 50)')
 parser.add_argument('--batch', type=int, default=8,
 help='Batch size (default: 8, reduce if GPU memory issues)')
 parser.add_argument('--model', type=str, choices=['n', 's', 'm', 'l', 'x'], default='n',
 help='YOLO model size: n(ano), s(mall), m(edium), l(arge), x(tra) (default: n)')

 args = parser.parse_args()

 print(" Safety Detection Model Training")
 print("=" * 50)
 print(f"Dataset: {args.dataset}")
 print(f"Epochs: {args.epochs}")
 print(f"Batch size: {args.batch}")
 print(f"Model size: yolov8{args.model}")
 print("=" * 50)

 try:
 results = train_safety_model(
 dataset_yaml_path=args.dataset,
 epochs=args.epochs,
 batch_size=args.batch,
 model_size=args.model
 )

 print("\n Training completed successfully!")
 print(f" Results saved to: {results.save_dir}")
 print(f" Check training metrics: {results.save_dir}/results.png")
 print(f" Best model: models/safety_detection_best.pt")

 except Exception as e:
 print(f"\n Training failed: {e}")
 print(" Tips:")
 print(" - Reduce batch size if GPU memory issues")
 print(" - Check dataset paths and annotations")
 print(" - Ensure sufficient disk space")
 return 1

 return 0

if __name__ == "__main__":
 exit(main())
