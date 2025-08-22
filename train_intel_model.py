#!/usr/bin/env python3
"""
Intel-Powered Tablet Defect Detection Model Training Script

This script trains a VGG16-based model optimized with Intel Extension for PyTorch
to detect tablet defects (Edge Defects vs Normal tablets).

Usage:
    python train_intel_model.py

Requirements:
    - Intel Extension for PyTorch
    - Defective Pill Classifier dataset
    - CUDA-capable GPU (optional, will use CPU if not available)
"""

import os
import sys
import time
from datetime import datetime

def main():
    print("=" * 60)
    print("Intel-Powered Tablet Defect Detection Model Training")
    print("=" * 60)
    print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if dataset exists
    dataset_path = "Defective Pill Classifier"
    if not os.path.exists(dataset_path):
        print("❌ Error: Dataset not found!")
        print(f"Expected dataset at: {os.path.abspath(dataset_path)}")
        print("Please ensure the Defective Pill Classifier dataset is in the project directory.")
        return False
    
    # Check if training data exists
    train_csv = os.path.join(dataset_path, "train", "_classes.csv")
    if not os.path.exists(train_csv):
        print("❌ Error: Training data not found!")
        print(f"Expected training CSV at: {os.path.abspath(train_csv)}")
        return False
    
    print("✅ Dataset found successfully!")
    print(f"Dataset location: {os.path.abspath(dataset_path)}")
    print()
    
    try:
        # Import Intel detector module
        from intel_tablet_detector import train_intel_model
        
        print("🚀 Starting Intel model training...")
        print("This may take several minutes depending on your hardware.")
        print()
        
        # Train the model
        start_time = time.time()
        model_path = train_intel_model()
        training_time = time.time() - start_time
        
        print()
        print("=" * 60)
        print("✅ Training completed successfully!")
        print("=" * 60)
        print(f"Model saved to: {os.path.abspath(model_path)}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Test the model
        print("🧪 Testing the trained model...")
        from intel_tablet_detector import initialize_intel_detector
        
        detector = initialize_intel_detector(model_path)
        print("✅ Model loaded and ready for inference!")
        print()
        
        print("📋 Next steps:")
        print("1. The model is now integrated with your Flask application")
        print("2. Upload images through the web interface to test detection")
        print("3. The system will use both Intel VGG16 and OpenCV detection")
        print("4. Check the image analysis results for Intel predictions")
        print()
        
        return True
        
    except ImportError as e:
        print("❌ Error: Required dependencies not found!")
        print(f"Missing: {e}")
        print()
        print("Please install the required dependencies:")
        print("pip install intel-extension-for-pytorch torch torchvision pandas")
        return False
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print()
        print("Troubleshooting tips:")
        print("1. Ensure you have sufficient disk space")
        print("2. Check if you have enough RAM (recommended: 8GB+)")
        print("3. If using GPU, ensure CUDA is properly installed")
        print("4. Try running with CPU only if GPU issues occur")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 