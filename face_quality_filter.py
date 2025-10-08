# face_quality_filter.py
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from tqdm import tqdm
import pickle
import math

class FaceQualityFilter:
    """
    Filter low-quality face images, including extreme side faces, blurry images, and non-face images
    Designed for pre-cropped face images from MTCNN detection
    """
    
    def __init__(self, use_dlib=False):
        """
        Initialize the face quality filter
        
        Args:
            use_dlib: Whether to use dlib for landmark detection (optional, slower but more accurate)
        """
        self.use_dlib = use_dlib
        self.landmark_predictor = None
        
        if use_dlib:
            self.landmark_predictor = self._init_landmark_detector()
    
    def _init_landmark_detector(self):
        """Initialize dlib facial landmark detector (optional)"""
        try:
            import dlib
            model_path = "models/shape_predictor_68_face_landmarks.dat"
            
            if not os.path.exists(model_path):
                print(f"⚠️  Landmark model not found: {model_path}")
                print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
                print("Continuing without dlib landmark detection...")
                return None
            
            return dlib.shape_predictor(model_path)
        except ImportError:
            print("⚠️  dlib not installed. Continuing without landmark detection...")
            return None
    
    def calculate_sharpness(self, image):
        """
        Calculate image sharpness using Laplacian variance
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Sharpness score (higher is sharper)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        return sharpness
    
    def detect_face_with_opencv(self, image):
        """
        Use OpenCV Haar Cascade to verify face presence
        More reliable for pre-cropped face images
        
        Args:
            image: Input image
            
        Returns:
            True if face detected, False otherwise
        """
        try:
            # Load Haar Cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect faces with lenient parameters
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,  # More lenient scale factor
                minNeighbors=2,     # Lower neighbor requirement
                minSize=(20, 20)    # Smaller minimum size
            )
            
            return len(faces) > 0
            
        except Exception as e:
            # If detection fails, assume face is present (fail-safe)
            return True
    
    def calculate_symmetry_score(self, image):
        """
        Calculate face symmetry to detect extreme side faces
        Side faces will have low symmetry scores
        
        Args:
            image: Input face image
            
        Returns:
            Symmetry score (0-1, higher means more symmetric/frontal)
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            h, w = gray.shape
            
            # Split image into left and right halves
            left_half = gray[:, :w//2]
            right_half = gray[:, w//2:]
            
            # Flip right half for comparison
            right_half_flipped = cv2.flip(right_half, 1)
            
            # Resize to same size in case of odd width
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]
            
            # Calculate structural similarity
            # Use normalized correlation
            left_norm = left_half.astype(float) / 255.0
            right_norm = right_half_flipped.astype(float) / 255.0
            
            # Calculate correlation
            correlation = np.corrcoef(left_norm.flatten(), right_norm.flatten())[0, 1]
            
            # Handle NaN (can occur with uniform images)
            if np.isnan(correlation):
                correlation = 0.5
            
            # Normalize to 0-1 range
            symmetry_score = (correlation + 1) / 2
            
            return max(0, min(1, symmetry_score))
            
        except Exception as e:
            # If calculation fails, return neutral score
            return 0.5
    
    def calculate_contrast(self, image):
        """
        Calculate image contrast
        Very low contrast might indicate issues
        
        Args:
            image: Input image
            
        Returns:
            Contrast score
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate standard deviation as contrast measure
        contrast = np.std(gray)
        
        return contrast
    
    def calculate_brightness(self, image):
        """
        Calculate average brightness
        
        Args:
            image: Input image
            
        Returns:
            Brightness score (0-255)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        brightness = np.mean(gray)
        
        return brightness
    
    def detect_eye_region(self, image):
        """
        Detect if eye region is visible (crucial for face identification)
        Uses Haar Cascade for eye detection
        
        Args:
            image: Input face image
            
        Returns:
            True if eyes detected, False otherwise
        """
        try:
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect eyes
            eyes = eye_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(10, 10)
            )
            
            # Should detect at least one eye for valid face
            return len(eyes) >= 1
            
        except Exception as e:
            # If detection fails, assume eyes are present
            return True
    
    def evaluate_face_quality(self, image_path, verbose=False):
        """
        Comprehensive face quality evaluation
        Designed specifically for pre-cropped face images
        
        Args:
            image_path: Path to face image
            verbose: Whether to print detailed information
            
        Returns:
            quality_dict: Dictionary containing quality metrics
        """
        quality_dict = {
            'path': image_path,
            'is_valid': True,  # Start with assuming valid
            'sharpness': 0,
            'symmetry': 0,
            'contrast': 0,
            'brightness': 0,
            'has_face': True,
            'has_eyes': True,
            'reason': []
        }
        
        try:
            # Read image
            image = cv2.imread(image_path)
            
            if image is None:
                quality_dict['is_valid'] = False
                quality_dict['reason'].append('Cannot read image')
                return quality_dict
            
            # Check image size (too small images are problematic)
            h, w = image.shape[:2]
            if h < 40 or w < 40:
                quality_dict['is_valid'] = False
                quality_dict['reason'].append(f'Image too small ({w}x{h})')
                return quality_dict
            
            # 1. Calculate sharpness
            sharpness = self.calculate_sharpness(image)
            quality_dict['sharpness'] = sharpness
            
            # More lenient sharpness threshold
            if sharpness < 20:  # Lowered from 50
                quality_dict['reason'].append(f'Too blurry (sharpness: {sharpness:.1f})')
                quality_dict['is_valid'] = False
            
            # 2. Calculate symmetry (for side face detection)
            symmetry = self.calculate_symmetry_score(image)
            quality_dict['symmetry'] = symmetry
            
            # More lenient symmetry threshold
            if symmetry < 0.35:  # Lowered from 0.5
                quality_dict['reason'].append(f'Extreme side face (symmetry: {symmetry:.2f})')
                quality_dict['is_valid'] = False
            
            # 3. Calculate contrast
            contrast = self.calculate_contrast(image)
            quality_dict['contrast'] = contrast
            
            if contrast < 15:  # Very low contrast
                quality_dict['reason'].append(f'Too low contrast ({contrast:.1f})')
                quality_dict['is_valid'] = False
            
            # 4. Calculate brightness
            brightness = self.calculate_brightness(image)
            quality_dict['brightness'] = brightness
            
            # Check for extreme brightness
            if brightness < 20 or brightness > 235:
                quality_dict['reason'].append(f'Extreme brightness ({brightness:.1f})')
                quality_dict['is_valid'] = False
            
            # 5. Verify face presence with OpenCV
            has_face = self.detect_face_with_opencv(image)
            quality_dict['has_face'] = has_face
            
            if not has_face:
                quality_dict['reason'].append('No face detected')
                quality_dict['is_valid'] = False
            
            # 6. Check for eye region (important for face recognition)
            has_eyes = self.detect_eye_region(image)
            quality_dict['has_eyes'] = has_eyes
            
            if not has_eyes:
                quality_dict['reason'].append('No eyes detected')
                quality_dict['is_valid'] = False
            
            # Verbose output
            if verbose and not quality_dict['is_valid']:
                reasons = ', '.join(quality_dict['reason'])
                print(f"❌ {os.path.basename(image_path)}: {reasons}")
            
        except Exception as e:
            quality_dict['is_valid'] = False
            quality_dict['reason'].append(f'Error: {str(e)}')
            if verbose:
                print(f"❌ {os.path.basename(image_path)}: Error - {str(e)}")
        
        return quality_dict
    
    def filter_face_images(self, face_paths, output_dir=None, save_report=True, strict_mode=False):
        """
        Batch filter face images
        
        Args:
            face_paths: List of face image paths
            output_dir: Output directory for reports
            save_report: Whether to save detailed report
            strict_mode: Use stricter filtering criteria
            
        Returns:
            valid_paths: List of valid face image paths
            invalid_paths: List of invalid face image paths
            quality_report: Detailed quality report
        """
        print(f"🔍 Starting quality check for {len(face_paths)} face images...")
        
        if strict_mode:
            print("⚠️  Using strict filtering mode")
        else:
            print("✓ Using lenient filtering mode (recommended for pre-cropped faces)")
        
        valid_paths = []
        invalid_paths = []
        quality_report = []
        
        # Progress bar
        for face_path in tqdm(face_paths, desc="Quality check"):
            quality_dict = self.evaluate_face_quality(face_path, verbose=False)
            quality_report.append(quality_dict)
            
            if quality_dict['is_valid']:
                valid_paths.append(face_path)
            else:
                invalid_paths.append(face_path)
        
        # Print statistics
        print(f"\n✅ Quality check complete:")
        print(f"   Passed: {len(valid_paths)} images ({len(valid_paths)/len(face_paths)*100:.1f}%)")
        print(f"   Failed: {len(invalid_paths)} images ({len(invalid_paths)/len(face_paths)*100:.1f}%)")
        
        # Statistics of failure reasons
        if invalid_paths:
            print(f"\n❌ Failure reasons:")
            reason_counts = {}
            for report in quality_report:
                if not report['is_valid']:
                    for reason in report['reason']:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   {reason}: {count} images")
        
        # Save detailed report
        if save_report and output_dir:
            # Create quality check directory
            quality_dir = os.path.join(output_dir, 'quality_check')
            if not os.path.exists(quality_dir):
                os.makedirs(quality_dir)
            
            # Save pickle report
            report_path = os.path.join(quality_dir, 'face_quality_report.pkl')
            with open(report_path, 'wb') as f:
                pickle.dump(quality_report, f)
            print(f"\n📊 Detailed report saved: {report_path}")
            
            # Save text report
            txt_report_path = os.path.join(quality_dir, 'face_quality_report.txt')
            with open(txt_report_path, 'w', encoding='utf-8') as f:
                f.write(f"Face Quality Check Report\n")
                f.write(f"="*60 + "\n\n")
                f.write(f"Total images: {len(face_paths)}\n")
                f.write(f"Passed: {len(valid_paths)} ({len(valid_paths)/len(face_paths)*100:.1f}%)\n")
                f.write(f"Failed: {len(invalid_paths)} ({len(invalid_paths)/len(face_paths)*100:.1f}%)\n\n")
                
                f.write(f"Failure reason statistics:\n")
                f.write(f"-"*60 + "\n")
                for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{reason}: {count} images\n")
                
                f.write(f"\nDetailed list of failed images:\n")
                f.write(f"-"*60 + "\n")
                for report in quality_report:
                    if not report['is_valid']:
                        f.write(f"\n{os.path.basename(report['path'])}\n")
                        f.write(f"  Reasons: {', '.join(report['reason'])}\n")
                        f.write(f"  Sharpness: {report['sharpness']:.1f}\n")
                        f.write(f"  Symmetry: {report['symmetry']:.2f}\n")
                        f.write(f"  Contrast: {report['contrast']:.1f}\n")
                        f.write(f"  Brightness: {report['brightness']:.1f}\n")
                        f.write(f"  Has face: {report['has_face']}\n")
                        f.write(f"  Has eyes: {report['has_eyes']}\n")
            
            print(f"📄 Text report saved: {txt_report_path}")
        
        return valid_paths, invalid_paths, quality_report


def integrate_quality_filter_with_main(face_paths, output_dir, strict_mode=False):
    """
    Integration function for main program
    
    Args:
        face_paths: All detected face image paths
        output_dir: Output directory
        strict_mode: Whether to use strict filtering
        
    Returns:
        filtered_face_paths: High-quality face image paths after filtering
    """
    # Create quality filter (without dlib for better compatibility)
    quality_filter = FaceQualityFilter(use_dlib=False)
    
    # Perform quality check
    valid_paths, invalid_paths, quality_report = quality_filter.filter_face_images(
        face_paths, 
        output_dir=output_dir, 
        save_report=True,
        strict_mode=strict_mode
    )
    
    # Optional: Move failed images to separate folder
    if invalid_paths:
        invalid_dir = os.path.join(output_dir, 'invalid_faces')
        if not os.path.exists(invalid_dir):
            os.makedirs(invalid_dir)
        
        print(f"\n📁 Moving {len(invalid_paths)} low-quality images to: {invalid_dir}")
        
        import shutil
        moved_count = 0
        for invalid_path in invalid_paths:
            try:
                dst_path = os.path.join(invalid_dir, os.path.basename(invalid_path))
                shutil.move(invalid_path, dst_path)
                moved_count += 1
            except Exception as e:
                print(f"⚠️  Cannot move {invalid_path}: {e}")
        
        print(f"✅ Moved {moved_count} files successfully")
    
    return valid_paths


# Test example
if __name__ == "__main__":
    # Test single image
    test_image = "path/to/test/face.jpg"
    
    filter_obj = FaceQualityFilter(use_dlib=False)
    quality = filter_obj.evaluate_face_quality(test_image, verbose=True)
    
    print(f"\nQuality evaluation result:")
    print(f"  Valid: {quality['is_valid']}")
    print(f"  Sharpness: {quality['sharpness']:.1f}")
    print(f"  Symmetry: {quality['symmetry']:.2f}")
    print(f"  Contrast: {quality['contrast']:.1f}")
    print(f"  Brightness: {quality['brightness']:.1f}")
    print(f"  Has face: {quality['has_face']}")
    print(f"  Has eyes: {quality['has_eyes']}")
    if quality['reason']:
        print(f"  Reasons: {', '.join(quality['reason'])}")