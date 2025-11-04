#!/usr/bin/env python3
"""
Text Orientation Detection Script (No OCR Required)

This script detects the orientation of text in an image using computer vision
techniques without requiring Tesseract or any OCR engine.

Requirements:
    pip install opencv-python numpy pillow --break-system-packages

Method: Uses edge detection, morphological operations, and geometric analysis
to detect text regions and their orientation.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, List
import argparse


class TextOrientationDetectorCV:
    """Detects text orientation using computer vision without OCR."""
    
    def __init__(self):
        """Initialize the detector."""
        self.angles = [0, 90, 180, 270]
    
    def rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """
        Rotate image by specified angle.
        
        Args:
            image: Input image as numpy array
            angle: Rotation angle (0, 90, 180, or 270)
            
        Returns:
            Rotated image
        """
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for text detection.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        return denoised
    
    def detect_text_regions(self, image: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Detect potential text regions using edge detection and morphology.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Tuple of (binary mask, list of contours)
        """
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Use morphological operations to connect text components
        # Horizontal kernel to connect characters in lines
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        dilated_h = cv2.dilate(binary, kernel_h, iterations=1)
        
        # Vertical kernel for small connections
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        dilated = cv2.dilate(dilated_h, kernel_v, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        return dilated, contours
    
    def analyze_text_orientation(self, image: np.ndarray) -> Dict:
        """
        Analyze text orientation based on geometric features.
        
        Args:
            image: Preprocessed grayscale image
            
        Returns:
            Dictionary with orientation metrics
        """
        # Detect text regions
        mask, contours = self.detect_text_regions(image)
        
        if not contours:
            return {
                'score': 0,
                'num_regions': 0,
                'horizontal_score': 0,
                'aspect_ratio_score': 0,
                'alignment_score': 0
            }
        
        # Filter contours by size
        min_area = 100
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if not valid_contours:
            return {
                'score': 0,
                'num_regions': 0,
                'horizontal_score': 0,
                'aspect_ratio_score': 0,
                'alignment_score': 0
            }
        
        # Analyze contours
        horizontal_count = 0
        aspect_ratios = []
        y_positions = []
        heights = []
        
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if region is horizontally oriented (width > height)
            if w > h * 1.5:
                horizontal_count += 1
            
            # Store aspect ratio
            aspect_ratio = w / max(h, 1)
            aspect_ratios.append(aspect_ratio)
            
            # Store vertical position for alignment analysis
            y_positions.append(y)
            heights.append(h)
        
        num_regions = len(valid_contours)
        
        # Calculate horizontal orientation score (0-1)
        horizontal_score = horizontal_count / num_regions if num_regions > 0 else 0
        
        # Calculate aspect ratio score (prefer wider rectangles for text lines)
        avg_aspect_ratio = np.mean(aspect_ratios) if aspect_ratios else 0
        # Ideal text line has aspect ratio > 3
        aspect_ratio_score = min(avg_aspect_ratio / 5.0, 1.0)
        
        # Calculate alignment score (text lines should be horizontally aligned)
        alignment_score = 0
        if len(y_positions) > 1:
            y_positions = np.array(y_positions)
            heights = np.array(heights)
            
            # Calculate standard deviation of y positions (normalized by height)
            avg_height = np.mean(heights)
            y_std = np.std(y_positions)
            alignment_score = max(0, 1.0 - (y_std / (avg_height * 3)))
        
        # Combined score
        score = (
            horizontal_score * 0.4 + 
            aspect_ratio_score * 0.4 + 
            alignment_score * 0.2
        )
        
        return {
            'score': score * 100,  # Convert to 0-100 scale
            'num_regions': num_regions,
            'horizontal_score': horizontal_score * 100,
            'aspect_ratio_score': aspect_ratio_score * 100,
            'alignment_score': alignment_score * 100
        }
    
    def detect_orientation(self, image_path: str, verbose: bool = False) -> Dict:
        """
        Detect the orientation of text in an image.
        
        Args:
            image_path: Path to the image file
            verbose: Print detailed information
            
        Returns:
            Dictionary with detection results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        results = {}
        best_angle = 0
        best_score = 0
        
        if verbose:
            print("\nTesting different orientations...")
            print("-" * 70)
        
        # Test each rotation angle
        for angle in self.angles:
            rotated = self.rotate_image(image, angle)
            preprocessed = self.preprocess_image(rotated)
            analysis = self.analyze_text_orientation(preprocessed)
            
            results[angle] = analysis
            
            if verbose:
                print(f"Angle {angle:3d}°:")
                print(f"  Overall Score:      {analysis['score']:6.2f}")
                print(f"  Text Regions:       {analysis['num_regions']:6d}")
                print(f"  Horizontal Score:   {analysis['horizontal_score']:6.2f}")
                print(f"  Aspect Ratio Score: {analysis['aspect_ratio_score']:6.2f}")
                print(f"  Alignment Score:    {analysis['alignment_score']:6.2f}")
                print()
            
            # Update best result
            if analysis['score'] > best_score:
                best_score = analysis['score']
                best_angle = angle
        
        # Determine if rotation is needed
        rotation_needed = best_angle != 0
        correction_angle = best_angle
        
        return {
            'current_angle': 0,
            'correct_angle': best_angle,
            'rotation_needed': rotation_needed,
            'correction_angle': correction_angle,
            'confidence': best_score,
            'num_text_regions': results[best_angle]['num_regions'],
            'all_results': results
        }
    
    def correct_orientation(self, image_path: str, output_path: str = None) -> str:
        """
        Detect orientation and save corrected image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save corrected image (optional)
            
        Returns:
            Path to corrected image
        """
        # Detect orientation
        result = self.detect_orientation(image_path)
        
        if not result['rotation_needed']:
            print("Image is already correctly oriented!")
            return image_path
        
        # Load and rotate image
        image = cv2.imread(image_path)
        corrected = self.rotate_image(image, result['correction_angle'])
        
        # Save corrected image
        if output_path is None:
            base_name = image_path.rsplit('.', 1)[0]
            extension = image_path.rsplit('.', 1)[1] if '.' in image_path else 'jpg'
            output_path = f"{base_name}_corrected.{extension}"
        
        cv2.imwrite(output_path, corrected)
        print(f"Corrected image saved to: {output_path}")
        
        return output_path
    
    def visualize_text_detection(self, image_path: str, angle: int = 0, 
                                 output_path: str = None) -> str:
        """
        Visualize detected text regions for debugging.
        
        Args:
            image_path: Path to input image
            angle: Rotation angle to test
            output_path: Path to save visualization
            
        Returns:
            Path to visualization image
        """
        # Load and rotate image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        rotated = self.rotate_image(image, angle)
        preprocessed = self.preprocess_image(rotated)
        
        # Detect text regions
        mask, contours = self.detect_text_regions(preprocessed)
        
        # Draw contours on original image
        visualization = rotated.copy()
        
        # Filter and draw valid contours
        min_area = 100
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(visualization, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Save visualization
        if output_path is None:
            base_name = image_path.rsplit('.', 1)[0]
            output_path = f"{base_name}_debug_{angle}deg.jpg"
        
        cv2.imwrite(output_path, visualization)
        print(f"Visualization saved to: {output_path}")
        
        return output_path


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Detect text orientation using computer vision (no OCR)'
    )
    parser.add_argument('image', help='Path to the image file')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show detailed information about detection')
    parser.add_argument('-c', '--correct', action='store_true',
                        help='Save a corrected version of the image')
    parser.add_argument('-o', '--output', help='Output path for corrected image')
    parser.add_argument('--debug', action='store_true',
                        help='Create visualization of detected text regions')
    
    args = parser.parse_args()
    
    # Create detector
    detector = TextOrientationDetectorCV()
    
    # Detect orientation
    result = detector.detect_orientation(args.image, verbose=args.verbose)
    
    # Print summary
    print("\n" + "="*70)
    print("TEXT ORIENTATION DETECTION RESULTS (Computer Vision Method)")
    print("="*70)
    print(f"Image: {args.image}")
    print(f"Current orientation: {result['current_angle']}°")
    print(f"Correct orientation: {result['correct_angle']}°")
    print(f"Rotation needed: {'YES' if result['rotation_needed'] else 'NO'}")
    
    if result['rotation_needed']:
        print(f"Required rotation: {result['correction_angle']}° clockwise")
    
    print(f"Detection confidence: {result['confidence']:.1f}%")
    print(f"Text regions detected: {result['num_text_regions']}")
    
    # Correct if requested
    if args.correct and result['rotation_needed']:
        print("\nCorrecting orientation...")
        detector.correct_orientation(args.image, args.output)
    
    # Create debug visualization if requested
    if args.debug:
        print("\nCreating debug visualizations...")
        for angle in [0, 90, 180, 270]:
            detector.visualize_text_detection(args.image, angle)


if __name__ == "__main__":
    main()