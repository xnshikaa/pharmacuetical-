import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import intel_extension_for_pytorch as ipex
import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image
import time
from datetime import datetime

# Configuration
INPUT_IMG_SIZE = (224, 224)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 0.001
EPOCHS = 50
BATCH_SIZE = 32
TARGET_TRAINING_ACCURACY = 0.95

class CustomVGG(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        self.feature_extractor = models.vgg16(pretrained=True).features[:-1]
        self.classification_head = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(
                kernel_size=(INPUT_IMG_SIZE[0] // 2 ** 5, INPUT_IMG_SIZE[1] // 2 ** 5)
            ),
            nn.Flatten(),
            nn.Linear(
                in_features=self.feature_extractor[-2].out_channels,
                out_features=n_classes,
            ),
        )
        # self._freeze_params()

    def _freeze_params(self):
        for param in self.feature_extractor[:23].parameters():
            param.requires_grad = False

    def forward(self, x_in):
        """
        forward
        """
        feature_maps = self.feature_extractor(x_in)
        scores = self.classification_head(feature_maps)

        if self.training:
            return scores

        probs = nn.functional.softmax(scores, dim=-1)

        weights = self.classification_head[3].weight
        weights = (
            weights.unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(0)
            .repeat(
                (
                    x_in.size(0),
                    1,
                    1,
                    INPUT_IMG_SIZE[0] // 2 ** 4,
                    INPUT_IMG_SIZE[0] // 2 ** 4,
                )
            )
        )
        feature_maps = feature_maps.unsqueeze(1).repeat((1, probs.size(1), 1, 1, 1))
        location = torch.mul(weights, feature_maps).sum(axis=2)
        location = F.interpolate(location, size=INPUT_IMG_SIZE, mode="bilinear")

        maxs, _ = location.max(dim=-1, keepdim=True)
        maxs, _ = maxs.max(dim=-2, keepdim=True)
        mins, _ = location.min(dim=-1, keepdim=True)
        mins, _ = mins.min(dim=-2, keepdim=True)
        norm_location = (location - mins) / (maxs - mins)

        return probs, norm_location

class TabletDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Load CSV file
        self.df = pd.read_csv(csv_file)
        
        # Define class mapping
        self.classes = ['Edge Defect', 'Normal']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        
        # Load image
        img_path = os.path.join(self.data_dir, filename)
        image = Image.open(img_path).convert('RGB')
        
        # Get label (Edge Defect = 0, Normal = 1)
        label = 1 if row['Normal'] == 1 else 0
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms():
    """Get training and validation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize(INPUT_IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(INPUT_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train(train_loader, model, optimizer, criterion, epochs, device, target_accuracy):
    """Training function"""
    model.train()
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Accuracy: {100*correct/total:.2f}%')
        
        epoch_accuracy = 100 * correct / total
        print(f'Epoch {epoch+1} completed. Accuracy: {epoch_accuracy:.2f}%')
        
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            # Save best model
            torch.save(model.state_dict(), 'best_tablet_model.pth')
        
        if epoch_accuracy >= target_accuracy * 100:
            print(f'Target accuracy {target_accuracy*100}% reached!')
            break
    
    return model

class IntelTabletDefectDetector:
    def __init__(self, model_path=None):
        self.device = DEVICE
        self.model = CustomVGG(n_classes=2)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded pre-trained model from {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Intel optimization
        try:
            self.model = ipex.optimize(self.model)
            print("Intel optimization applied successfully")
        except Exception as e:
            print(f"Intel optimization failed: {e}")
        
        self.transform = transforms.Compose([
            transforms.Resize(INPUT_IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def detect_defects(self, image_path):
        """Detect tablet defects using Intel-optimized model"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                probs, attention_map = self.model(input_tensor)
            
            # Get predictions
            probabilities = probs.cpu().numpy()[0]
            predicted_class = np.argmax(probabilities)
            confidence = probabilities[predicted_class]
            
            # Class labels
            classes = ['Edge Defect', 'Normal']
            predicted_label = classes[predicted_class]
            
            # Process attention map for defect localization
            attention_map = attention_map.cpu().numpy()[0, predicted_class]
            attention_map = cv2.resize(attention_map, (image.size[0], image.size[1]))
            
            # Find defect regions
            defect_regions = self._find_defect_regions(attention_map, predicted_class)
            
            return {
                'defects': defect_regions,
                'total_defects': len(defect_regions),
                'predicted_class': predicted_label,
                'confidence': float(confidence),
                'attention_map': attention_map,
                'analysis_timestamp': datetime.now().isoformat(),
                'model_type': 'Intel-VGG16',
                'image_processed': True
            }
            
        except Exception as e:
            print(f"Error in Intel defect detection: {e}")
            return {"error": f"Intel detection failed: {str(e)}"}
    
    def _find_defect_regions(self, attention_map, predicted_class):
        """Find specific defect regions from attention map"""
        defects = []
        
        if predicted_class == 0:  # Edge Defect
            # Threshold the attention map
            threshold = np.percentile(attention_map, 85)
            binary_map = (attention_map > threshold).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 50:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    defects.append({
                        'type': 'edge_defect',
                        'confidence': float(np.mean(attention_map[y:y+h, x:x+w])),
                        'description': f'Edge defect detected in region {i+1}',
                        'severity': 'high' if area > 200 else 'medium',
                        'bbox': (x, y, w, h),
                        'area': area
                    })
        
        return defects
    
    def generate_defect_summary(self, defects_analysis):
        """Generate summary for Intel-based defect detection"""
        if 'error' in defects_analysis:
            return f"Intel detection failed: {defects_analysis['error']}"
        
        predicted_class = defects_analysis.get('predicted_class', 'Unknown')
        confidence = defects_analysis.get('confidence', 0)
        total_defects = defects_analysis.get('total_defects', 0)
        
        if predicted_class == 'Normal':
            return f"Intel VGG16 Model: Tablet appears NORMAL (Confidence: {confidence:.1%})"
        else:
            summary = f"Intel VGG16 Model: {predicted_class} detected (Confidence: {confidence:.1%})\n"
            summary += f"Total defect regions: {total_defects}\n\n"
            
            defects = defects_analysis.get('defects', [])
            for defect in defects:
                severity_emoji = "🔴" if defect['severity'] == 'high' else "🟡"
                summary += f"{severity_emoji} {defect['description']} (Confidence: {defect['confidence']:.1%})\n"
            
            return summary

def train_intel_model():
    """Train the Intel-optimized model"""
    print("Starting Intel model training...")
    
    # Data paths
    train_dir = "Defective Pill Classifier/train"
    train_csv = "Defective Pill Classifier/train/_classes.csv"
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Create datasets
    train_dataset = TabletDataset(train_dir, train_csv, train_transform)
    
    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Calculate class weights
    class_counts = [0, 0]
    for _, label in train_dataset:
        class_counts[label] += 1
    
    total_samples = sum(class_counts)
    class_weight = [total_samples / (2 * count) for count in class_counts]
    
    # Initialize model
    model = CustomVGG(n_classes=2)
    class_weight_tensor = torch.tensor(class_weight).type(torch.FloatTensor).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Intel optimization
    try:
        model, optimizer = ipex.optimize(model=model, optimizer=optimizer, dtype=torch.float32)
        print("Intel optimization applied for training")
    except Exception as e:
        print(f"Intel optimization failed: {e}")
    
    # Train model
    start_time = time.time()
    trained_model = train(train_loader, model=model, optimizer=optimizer, criterion=criterion, 
                         epochs=EPOCHS, device=DEVICE, target_accuracy=TARGET_TRAINING_ACCURACY)
    train_time = time.time() - start_time
    
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Save model
    model_path = "intel_tablet_model.h5"
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    return model_path

# Global instance for integration
intel_detector = None

def initialize_intel_detector(model_path="intel_tablet_model.h5"):
    """Initialize the Intel detector globally"""
    global intel_detector
    if intel_detector is None:
        intel_detector = IntelTabletDefectDetector(model_path)
    return intel_detector

if __name__ == "__main__":
    # Train the model if it doesn't exist
    if not os.path.exists("intel_tablet_model.h5"):
        print("Training Intel model...")
        train_intel_model()
    
    # Test the detector
    detector = initialize_intel_detector()
    print("Intel Tablet Defect Detector initialized successfully!") 