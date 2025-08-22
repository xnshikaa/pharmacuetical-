import cv2
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime

# Import Intel detector (optional)
try:
    from intel_tablet_detector import initialize_intel_detector
    INTEL_AVAILABLE = True
except ImportError:
    INTEL_AVAILABLE = False
    print("Intel detector not available. Using OpenCV-based detection only.")

class TabletDefectDetector:
    def __init__(self, use_intel=False):
        self.defect_types = {
            'crack': 'Crack or fracture in tablet surface',
            'discoloration': 'Abnormal color variation or staining',
            'contamination': 'Foreign particles or debris on surface',
            'deformation': 'Irregular shape or size deviation',
            'surface_damage': 'Scratches, chips, or surface irregularities',
            'coating_defect': 'Uneven or damaged tablet coating',
            'size_variation': 'Significant size differences from standard',
            'texture_anomaly': 'Unusual surface texture or roughness',
            'mixed_tablets': 'Multiple different tablets detected - CRITICAL QUALITY ISSUE',
            'edge_defect': 'Edge defect detected by Intel VGG16 model'
        }
        
        self.use_intel = use_intel and INTEL_AVAILABLE
        self.intel_detector = None
        
        if self.use_intel:
            try:
                self.intel_detector = initialize_intel_detector()
                print("Intel detector initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Intel detector: {e}")
                self.use_intel = False
    
    def preprocess_image(self, image_path):
        """Preprocess the image for analysis"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize for consistent processing
            height, width = image_rgb.shape[:2]
            max_dim = 800
            if max(height, width) > max_dim:
                scale = max_dim / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_rgb = cv2.resize(image_rgb, (new_width, new_height))
            
            return image_rgb
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def _detect_defects_opencv(self, image_path):
        """Original OpenCV-based defect detection"""
        image = self.preprocess_image(image_path)
        if image is None:
            return {"error": "Failed to process image"}
        
        defects = []
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 1. Detect cracks and surface damage
        cracks = self._detect_cracks(gray)
        if cracks['detected']:
            defects.append({
                'type': 'crack',
                'confidence': cracks['confidence'],
                'description': 'Crack or fracture detected on tablet surface',
                'severity': 'high' if cracks['confidence'] > 0.7 else 'medium'
            })
        
        # 2. Detect discoloration
        discoloration = self._detect_discoloration(hsv)
        if discoloration['detected']:
            defects.append({
                'type': 'discoloration',
                'confidence': discoloration['confidence'],
                'description': 'Abnormal color variation detected',
                'severity': 'high' if discoloration['confidence'] > 0.8 else 'medium'
            })
        
        # 3. Detect contamination
        contamination = self._detect_contamination(gray)
        if contamination['detected']:
            defects.append({
                'type': 'contamination',
                'confidence': contamination['confidence'],
                'description': 'Foreign particles or contamination detected',
                'severity': 'high' if contamination['confidence'] > 0.8 else 'medium'
            })
        
        # 4. Detect shape deformation
        deformation = self._detect_deformation(gray)
        if deformation['detected']:
            defects.append({
                'type': 'deformation',
                'confidence': deformation['confidence'],
                'description': 'Irregular shape or size deviation detected',
                'severity': 'high' if deformation['confidence'] > 0.8 else 'medium'
            })
        
        # 5. Detect coating defects
        coating_defect = self._detect_coating_defects(gray)
        if coating_defect['detected']:
            defects.append({
                'type': 'coating_defect',
                'confidence': coating_defect['confidence'],
                'description': 'Uneven or damaged tablet coating detected',
                'severity': 'high' if coating_defect['confidence'] > 0.8 else 'medium'
            })
        
        return {
            'defects': defects,
            'total_defects': len(defects),
            'analysis_timestamp': datetime.now().isoformat(),
            'image_processed': True,
            'critical_error': False,
            'mixed_tablets_detected': False,
            'detection_method': 'OpenCV'
        }
    
    def detect_mixed_tablets(self, image_path):
        """Detect if multiple different tablets are present - CRITICAL QUALITY ISSUE"""
        image = self.preprocess_image(image_path)
        if image is None:
            return {"error": "Failed to process image"}
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # 1. Detect individual tablets
        tablets = self._detect_individual_tablets(gray)
        
        # 2. Analyze tablet variety
        variety_analysis = self._analyze_tablet_variety(tablets, hsv)
        
        # 3. Check for mixed tablet issue
        mixed_tablet_issue = self._check_mixed_tablets(tablets, variety_analysis)
        
        return {
            'tablets_detected': len(tablets),
            'tablet_variety': variety_analysis,
            'mixed_tablets_detected': mixed_tablet_issue['detected'],
            'critical_error': mixed_tablet_issue['detected'],
            'error_message': mixed_tablet_issue['message'],
            'analysis_timestamp': datetime.now().isoformat(),
            'image_processed': True
        }
    
    def _detect_individual_tablets(self, gray_image):
        """Detect individual tablets in the image"""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Use adaptive thresholding to handle varying lighting
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tablets = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter by reasonable tablet size (adjust based on your images)
            if 500 < area < 50000:  # Adjust these thresholds as needed
                x, y, w, h = cv2.boundingRect(contour)
                tablets.append({
                    'contour': contour,
                    'area': area,
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2)
                })
        
        return tablets
    
    def _analyze_tablet_variety(self, tablets, hsv_image):
        """Analyze the variety of tablets based on color, size, and shape"""
        if len(tablets) < 2:
            return {'variety_score': 0, 'different_colors': 0, 'different_sizes': 0, 'different_shapes': 0}
        
        # Analyze colors
        colors = []
        sizes = []
        shapes = []
        
        for tablet in tablets:
            x, y, w, h = tablet['bbox']
            
            # Extract color information from tablet region
            tablet_region = hsv_image[y:y+h, x:x+w]
            if tablet_region.size > 0:
                mean_hue = np.mean(tablet_region[:, :, 0])
                mean_sat = np.mean(tablet_region[:, :, 1])
                colors.append((mean_hue, mean_sat))
            
            # Size analysis
            sizes.append(tablet['area'])
            
            # Shape analysis (aspect ratio)
            aspect_ratio = w / h if h > 0 else 1
            shapes.append(aspect_ratio)
        
        # Calculate variety metrics
        different_colors = self._count_different_colors(colors)
        different_sizes = self._count_different_sizes(sizes)
        different_shapes = self._count_different_shapes(shapes)
        
        # Overall variety score
        variety_score = (different_colors + different_sizes + different_shapes) / 3.0
        
        return {
            'variety_score': variety_score,
            'different_colors': different_colors,
            'different_sizes': different_sizes,
            'different_shapes': different_shapes,
            'total_tablets': len(tablets)
        }
    
    def _count_different_colors(self, colors):
        """Count how many different color groups exist"""
        if len(colors) < 2:
            return 0
        
        # Group similar colors
        color_groups = []
        for hue, sat in colors:
            grouped = False
            for group in color_groups:
                # Check if color is similar to existing group
                if any(abs(hue - g[0]) < 20 and abs(sat - g[1]) < 30 for g in group):
                    group.append((hue, sat))
                    grouped = True
                    break
            if not grouped:
                color_groups.append([(hue, sat)])
        
        return len(color_groups)
    
    def _count_different_sizes(self, sizes):
        """Count how many different size groups exist"""
        if len(sizes) < 2:
            return 0
        
        # Group similar sizes
        size_groups = []
        for size in sizes:
            grouped = False
            for group in size_groups:
                # Check if size is similar to existing group (within 30% difference)
                if any(abs(size - s) / s < 0.3 for s in group):
                    group.append(size)
                    grouped = True
                    break
            if not grouped:
                size_groups.append([size])
        
        return len(size_groups)
    
    def _count_different_shapes(self, shapes):
        """Count how many different shape groups exist"""
        if len(shapes) < 2:
            return 0
        
        # Group similar shapes
        shape_groups = []
        for shape in shapes:
            grouped = False
            for group in shape_groups:
                # Check if shape is similar to existing group (within 20% difference)
                if any(abs(shape - s) / s < 0.2 for s in group):
                    group.append(shape)
                    grouped = True
                    break
            if not grouped:
                shape_groups.append([shape])
        
        return len(shape_groups)
    
    def _check_mixed_tablets(self, tablets, variety_analysis):
        """Check if multiple different tablets are present"""
        if len(tablets) < 2:
            return {
                'detected': False,
                'message': 'Single tablet detected - no mixing issue'
            }
        
        # Check for variety indicators
        variety_score = variety_analysis['variety_score']
        different_colors = variety_analysis['different_colors']
        different_sizes = variety_analysis['different_sizes']
        different_shapes = variety_analysis['different_shapes']
        
        # Determine if this is a mixed tablet issue
        is_mixed = False
        reasons = []
        
        if different_colors >= 3:
            is_mixed = True
            reasons.append(f"{different_colors} different colors detected")
        
        if different_sizes >= 3:
            is_mixed = True
            reasons.append(f"{different_sizes} different sizes detected")
        
        if different_shapes >= 2:
            is_mixed = True
            reasons.append(f"{different_shapes} different shapes detected")
        
        if variety_score > 0.6:
            is_mixed = True
            reasons.append("High variety score indicates mixed tablets")
        
        if is_mixed:
            message = f"🚨 CRITICAL QUALITY ISSUE: Multiple different tablets detected! ({len(tablets)} tablets found)\n"
            message += f"Reasons: {', '.join(reasons)}\n"
            message += "This indicates a serious pharmaceutical quality control failure.\n"
            message += "IMMEDIATE ACTION REQUIRED: Batch quarantine and investigation needed."
        else:
            message = f"Multiple tablets detected ({len(tablets)}) but appear to be the same type.\n"
            message += f"Variety analysis: {variety_score:.2f} (low variety = same tablets)"
        
        return {
            'detected': is_mixed,
            'message': message,
            'variety_score': variety_score,
            'tablet_count': len(tablets)
        }
    
    def detect_defects(self, image_path):
        """Analyze image for tablet defects using both OpenCV and Intel methods"""
        # First, check for mixed tablets (CRITICAL ISSUE)
        mixed_tablet_analysis = self.detect_mixed_tablets(image_path)
        
        # If mixed tablets detected, return critical error
        if mixed_tablet_analysis.get('critical_error', False):
            return {
                'defects': [{
                    'type': 'mixed_tablets',
                    'confidence': 0.95,
                    'description': 'Multiple different tablets detected - CRITICAL QUALITY ISSUE',
                    'severity': 'critical',
                    'error_message': mixed_tablet_analysis['error_message']
                }],
                'total_defects': 1,
                'analysis_timestamp': datetime.now().isoformat(),
                'image_processed': True,
                'critical_error': True,
                'mixed_tablets_detected': True,
                'detection_method': 'OpenCV + Multi-tablet detection'
            }
        
        # Use Intel detection if available and enabled
        if self.use_intel and self.intel_detector:
            try:
                intel_results = self.intel_detector.detect_defects(image_path)
                if 'error' not in intel_results:
                    # Combine Intel results with OpenCV results
                    opencv_results = self._detect_defects_opencv(image_path)
                    
                    # Merge results
                    combined_defects = intel_results.get('defects', [])
                    combined_defects.extend(opencv_results.get('defects', []))
                    
                    return {
                        'defects': combined_defects,
                        'total_defects': len(combined_defects),
                        'analysis_timestamp': datetime.now().isoformat(),
                        'image_processed': True,
                        'critical_error': False,
                        'mixed_tablets_detected': False,
                        'detection_method': 'Intel VGG16 + OpenCV',
                        'intel_prediction': intel_results.get('predicted_class'),
                        'intel_confidence': intel_results.get('confidence'),
                        'attention_map': intel_results.get('attention_map')
                    }
            except Exception as e:
                print(f"Intel detection failed, falling back to OpenCV: {e}")
        
        # Fall back to OpenCV-only detection
        return self._detect_defects_opencv(image_path)
    
    def _detect_cracks(self, gray_image):
        """Detect cracks using edge detection and morphological operations"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Edge detection with lower thresholds for better crack detection
        edges = cv2.Canny(blurred, 30, 100)
        
        # Morphological operations to connect crack lines
        kernel = np.ones((2, 2), np.uint8)  # Smaller kernel
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Analyze contours for crack-like features
        crack_score = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 1000:  # Adjusted area range
                # Check if contour is elongated (crack-like)
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / max(1, min(w, h))
                if aspect_ratio > 2.5:  # Lower threshold for crack detection
                    crack_score += 1
                    # Additional scoring for very thin lines
                    if aspect_ratio > 5:
                        crack_score += 0.5
        
        confidence = min(crack_score / 8.0, 1.0)  # Adjusted normalization
        return {
            'detected': confidence > 0.2,  # Lower threshold
            'confidence': confidence
        }
    
    def _detect_discoloration(self, hsv_image):
        """Detect color variations that might indicate discoloration"""
        # Analyze color distribution
        h, s, v = cv2.split(hsv_image)
        
        # Calculate color variance
        h_var = np.var(h)
        s_var = np.var(s)
        
        # High variance in hue or saturation might indicate discoloration
        confidence = min((h_var + s_var) / 1000.0, 1.0)
        
        return {
            'detected': confidence > 0.4,
            'confidence': confidence
        }
    
    def _detect_contamination(self, gray_image):
        """Detect foreign particles or contamination"""
        # Apply threshold to find dark spots (potential contamination)
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find dark regions
        dark_regions = cv2.bitwise_not(thresh)
        
        # Find contours of dark regions
        contours, _ = cv2.findContours(dark_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contamination_score = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 500:  # Reasonable size for contamination
                contamination_score += 1
        
        confidence = min(contamination_score / 5.0, 1.0)
        return {
            'detected': confidence > 0.3,
            'confidence': confidence
        }
    
    def _detect_deformation(self, gray_image):
        """Detect shape irregularities"""
        # Find tablet outline
        _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'detected': False, 'confidence': 0}
        
        # Find the largest contour (assumed to be the tablet)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate shape metrics
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Circularity (perfect circle = 1.0)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # Low circularity indicates deformation
        confidence = 1.0 - circularity
        return {
            'detected': confidence > 0.3,
            'confidence': confidence
        }
    
    def _detect_coating_defects(self, gray_image):
        """Detect coating irregularities"""
        # Apply gradient filter to detect surface irregularities
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # High gradient variance indicates coating defects
        gradient_variance = np.var(gradient_magnitude)
        
        # Also check for texture irregularities
        # Apply Laplacian for edge detection
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        laplacian_var = np.var(laplacian)
        
        # Combine both metrics
        combined_score = (gradient_variance + laplacian_var) / 2000.0
        confidence = min(combined_score, 1.0)
        
        return {
            'detected': confidence > 0.3,  # Lower threshold
            'confidence': confidence
        }
    
    def generate_defect_summary(self, defects_analysis):
        """Generate a summary of detected defects for report generation"""
        if 'error' in defects_analysis:
            return "Image analysis failed. Please ensure the image is clear and well-lit."
        
        # Check for critical mixed tablet error first
        if defects_analysis.get('critical_error', False):
            critical_defect = defects_analysis['defects'][0]
            return f"🚨 CRITICAL ERROR: {critical_defect['error_message']}"
        
        defects = defects_analysis.get('defects', [])
        detection_method = defects_analysis.get('detection_method', 'Unknown')
        
        # Add Intel prediction info if available
        intel_info = ""
        if 'intel_prediction' in defects_analysis:
            intel_info = f"\n🤖 Intel VGG16 Prediction: {defects_analysis['intel_prediction']} (Confidence: {defects_analysis['intel_confidence']:.1%})"
        
        if not defects:
            return f"No defects detected. Tablet appears to be in good condition.{intel_info}\n\nDetection Method: {detection_method}"
        
        summary = f"Analysis detected {len(defects)} potential defect(s):{intel_info}\n\n"
        
        for defect in defects:
            severity_emoji = "🔴" if defect['severity'] == 'high' else "🟡" if defect['severity'] == 'medium' else "🟢"
            summary += f"{severity_emoji} {defect['description']} (Confidence: {defect['confidence']:.1%})\n"
        
        summary += f"\nDetection Method: {detection_method}"
        return summary

# Global instance with Intel support
defect_detector = TabletDefectDetector(use_intel=True) 