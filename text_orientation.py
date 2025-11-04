#!/usr/bin/env python3
"""
Text Orientation Detection Script

This script detects the orientation of text in an image and determines
if the image needs to be rotated to correctly orient the text.

Requirements:
    pip install pillow pytesseract opencv-python numpy --break-system-packages

You'll also need tesseract-ocr installed on your system:
    sudo apt-get install tesseract-ocr
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import Tuple, Dict
import argparse


class TextOrientationDetector:
    """Detects text orientation and suggests rotation corrections."""
    
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
    
    def get_text_confidence(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Extract text and confidence score from image using OCR.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (extracted_text, average_confidence)
        """
        # Convert to PIL Image for tesseract
        pil_image = Image.fromarray(image)
        
        # Get detailed OCR data
        ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        
        # Calculate average confidence for detected text
        confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
        text_items = [text for text in ocr_data['text'] if text.strip()]
        
        if not confidences or not text_items:
            return "", 0.0
        
        avg_confidence = sum(confidences) / len(confidences)
        full_text = " ".join(text_items)
        
        return full_text, avg_confidence
    
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
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = {}
        best_angle = 0
        best_confidence = 0
        best_text = ""
        
        if verbose:
            print("\nTesting different orientations...")
            print("-" * 60)
        
        # Test each rotation angle
        for angle in self.angles:
            rotated = self.rotate_image(image_rgb, angle)
            text, confidence = self.get_text_confidence(rotated)
            
            results[angle] = {
                'text': text,
                'confidence': confidence,
                'text_length': len(text.strip())
            }
            
            if verbose:
                print(f"Angle {angle:3d}°: Confidence={confidence:5.1f}%, "
                      f"Text length={len(text.strip()):4d} chars")
                if text.strip():
                    preview = text.strip()[:60] + "..." if len(text.strip()) > 60 else text.strip()
                    print(f"           Preview: {preview}")
                print()
            
            # Update best result (prioritize confidence, then text length)
            if (confidence > best_confidence or 
                (confidence == best_confidence and len(text) > len(best_text))):
                best_confidence = confidence
                best_angle = angle
                best_text = text
        
        # Determine if rotation is needed
        rotation_needed = best_angle != 0
        correction_angle = best_angle
        
        return {
            'current_angle': 0,
            'correct_angle': best_angle,
            'rotation_needed': rotation_needed,
            'correction_angle': correction_angle,
            'confidence': best_confidence,
            'detected_text': best_text,
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
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        corrected = self.rotate_image(image_rgb, result['correction_angle'])
        corrected_bgr = cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)
        
        # Save corrected image
        if output_path is None:
            base_name = image_path.rsplit('.', 1)[0]
            extension = image_path.rsplit('.', 1)[1] if '.' in image_path else 'jpg'
            output_path = f"{base_name}_corrected.{extension}"
        
        cv2.imwrite(output_path, corrected_bgr)
        print(f"Corrected image saved to: {output_path}")
        
        return output_path


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Detect text orientation in images and suggest corrections'
    )
    parser.add_argument('image', help='Path to the image file')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show detailed information about detection')
    parser.add_argument('-c', '--correct', action='store_true',
                        help='Save a corrected version of the image')
    parser.add_argument('-o', '--output', help='Output path for corrected image')
    
    args = parser.parse_args()
    
    # Create detector
    detector = TextOrientationDetector()
    
    # Detect orientation
    result = detector.detect_orientation(args.image, verbose=args.verbose)
    
    # Print summary
    print("\n" + "="*60)
    print("TEXT ORIENTATION DETECTION RESULTS")
    print("="*60)
    print(f"Image: {args.image}")
    print(f"Current orientation: {result['current_angle']}°")
    print(f"Correct orientation: {result['correct_angle']}°")
    print(f"Rotation needed: {'YES' if result['rotation_needed'] else 'NO'}")
    
    if result['rotation_needed']:
        print(f"Required rotation: {result['correction_angle']}° clockwise")
    
    print(f"Detection confidence: {result['confidence']:.1f}%")
    
    if result['detected_text'].strip():
        print(f"\nDetected text preview:")
        preview = result['detected_text'].strip()[:200]
        if len(result['detected_text'].strip()) > 200:
            preview += "..."
        print(f"  {preview}")
    
    # Correct if requested
    if args.correct and result['rotation_needed']:
        print("\nCorrecting orientation...")
        detector.correct_orientation(args.image, args.output)


if __name__ == "__main__":
    main()