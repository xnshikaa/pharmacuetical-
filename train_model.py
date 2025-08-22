#!/usr/bin/env python3
"""
Training script for tablet defect detection model.
This script allows users to train the model with their own data.
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
from image_processor import TabletDefectDetector
import argparse

class ModelTrainer:
    def __init__(self):
        self.detector = TabletDefectDetector()
        self.training_data = []
        
    def load_training_data(self, data_dir):
        """Load training data from directory structure"""
        training_data = []
        
        # Expected directory structure:
        # data_dir/
        #   ├── cracks/
        #   ├── discoloration/
        #   ├── contamination/
        #   ├── deformation/
        #   └── coating_defects/
        
        defect_types = ['cracks', 'discoloration', 'contamination', 'deformation', 'coating_defects']
        
        for defect_type in defect_types:
            defect_dir = os.path.join(data_dir, defect_type)
            if os.path.exists(defect_dir):
                for filename in os.listdir(defect_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        filepath = os.path.join(defect_dir, filename)
                        training_data.append({
                            'filepath': filepath,
                            'defect_type': defect_type.replace('_', ''),
                            'expected_defects': [defect_type.replace('_', '')]
                        })
        
        return training_data
    
    def evaluate_model(self, training_data):
        """Evaluate current model performance on training data"""
        results = {
            'total_images': len(training_data),
            'correct_detections': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'accuracy': 0.0,
            'detailed_results': []
        }
        
        for item in training_data:
            # Analyze image with current model
            analysis = self.detector.detect_defects(item['filepath'])
            
            detected_defects = [defect['type'] for defect in analysis.get('defects', [])]
            expected_defects = item['expected_defects']
            
            # Calculate metrics
            correct = len(set(detected_defects) & set(expected_defects))
            false_positives = len(set(detected_defects) - set(expected_defects))
            false_negatives = len(set(expected_defects) - set(detected_defects))
            
            if correct > 0:
                results['correct_detections'] += 1
            
            results['false_positives'] += false_positives
            results['false_negatives'] += false_negatives
            
            results['detailed_results'].append({
                'filepath': item['filepath'],
                'expected': expected_defects,
                'detected': detected_defects,
                'correct': correct,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            })
        
        if results['total_images'] > 0:
            results['accuracy'] = results['correct_detections'] / results['total_images']
        
        return results
    
    def generate_training_report(self, results):
        """Generate a detailed training report"""
        report = f"""
TABLET DEFECT DETECTION MODEL TRAINING REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL PERFORMANCE:
- Total Images: {results['total_images']}
- Correct Detections: {results['correct_detections']}
- False Positives: {results['false_positives']}
- False Negatives: {results['false_negatives']}
- Overall Accuracy: {results['accuracy']:.2%}

DETAILED RESULTS:
"""
        
        for result in results['detailed_results']:
            filename = os.path.basename(result['filepath'])
            report += f"""
File: {filename}
Expected Defects: {', '.join(result['expected'])}
Detected Defects: {', '.join(result['detected'])}
Correct: {result['correct']}, False Positives: {result['false_positives']}, False Negatives: {result['false_negatives']}
"""
        
        return report
    
    def save_results(self, results, output_file):
        """Save training results to file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def print_recommendations(self, results):
        """Print recommendations for improving the model"""
        print("\n" + "="*60)
        print("RECOMMENDATIONS FOR MODEL IMPROVEMENT")
        print("="*60)
        
        if results['accuracy'] < 0.7:
            print("WARNING: Model accuracy is below 70%. Consider:")
            print("   - Adding more training images")
            print("   - Ensuring images are well-lit and clear")
            print("   - Balancing training data across defect types")
        
        if results['false_positives'] > results['correct_detections']:
            print("WARNING: High false positive rate. Consider:")
            print("   - Adjusting detection thresholds")
            print("   - Adding more negative training samples")
            print("   - Improving image preprocessing")
        
        if results['false_negatives'] > results['correct_detections']:
            print("WARNING: High false negative rate. Consider:")
            print("   - Adding more positive training samples")
            print("   - Improving defect detection algorithms")
            print("   - Adjusting detection sensitivity")
        
        print(f"\nCurrent Performance Metrics:")
        print(f"   Accuracy: {results['accuracy']:.2%}")
        print(f"   Precision: {results['correct_detections'] / max(1, results['correct_detections'] + results['false_positives']):.2%}")
        print(f"   Recall: {results['correct_detections'] / max(1, results['correct_detections'] + results['false_negatives']):.2%}")

def main():
    parser = argparse.ArgumentParser(description='Train tablet defect detection model')
    parser.add_argument('--data_dir', required=True, help='Directory containing training data')
    parser.add_argument('--output', default='training_results.json', help='Output file for results')
    parser.add_argument('--report', default='training_report.txt', help='Output file for training report')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory '{args.data_dir}' does not exist")
        return
    
    print("Starting tablet defect detection model training...")
    
    trainer = ModelTrainer()
    
    # Load training data
    print(f"Loading training data from {args.data_dir}...")
    training_data = trainer.load_training_data(args.data_dir)
    
    if not training_data:
        print("ERROR: No training data found. Please ensure your directory structure is:")
        print("   data_dir/")
        print("   ├── cracks/")
        print("   ├── discoloration/")
        print("   ├── contamination/")
        print("   ├── deformation/")
        print("   └── coating_defects/")
        return
    
    print(f"SUCCESS: Loaded {len(training_data)} training images")
    
    # Evaluate model
    print("Evaluating model performance...")
    results = trainer.evaluate_model(training_data)
    
    # Generate and save report
    report = trainer.generate_training_report(results)
    
    with open(args.report, 'w') as f:
        f.write(report)
    
    trainer.save_results(results, args.output)
    
    print(f"SUCCESS: Training evaluation complete!")
    print(f"Results saved to: {args.output}")
    print(f"Report saved to: {args.report}")
    
    # Print recommendations
    trainer.print_recommendations(results)
    
    print(f"\nNext Steps:")
    print(f"   1. Review the detailed report in {args.report}")
    print(f"   2. Add more training images to improve accuracy")
    print(f"   3. Adjust detection parameters in image_processor.py")
    print(f"   4. Re-run training to measure improvements")

if __name__ == "__main__":
    main() 