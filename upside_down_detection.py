#!/usr/bin/env python3
"""
Image Orientation Detector
Detects if an image is upside down using face detection and text orientation.
"""

import cv2
import numpy as np
from PIL import Image
import sys
import os
import argparse

def detect_faces(image_path):
    """
    Detect faces in the image and return their positions.
    Returns a list of face rectangles and the image.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load OpenCV's pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    return faces, img

def detect_faces_at_rotation(img, rotation_angle):
    """
    Detect faces in an image at a specific rotation.
    
    Args:
        img: OpenCV image
        rotation_angle: Angle to rotate (0, 90, 180, 270)
    
    Returns:
        Number of faces detected at this rotation
    """
    # Rotate image if needed
    if rotation_angle == 90:
        rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_angle == 180:
        rotated = cv2.rotate(img, cv2.ROTATE_180)
    elif rotation_angle == 270:
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:  # 0 degrees
        rotated = img.copy()
    
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    
    # Load OpenCV's pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    return len(faces) if faces is not None else 0

def check_orientation_by_face(image_path):
    """
    Check image orientation by detecting faces at all 4 rotations.
    Returns: dict with results for all rotations
    """
    faces_normal, img = detect_faces(image_path)
    
    if img is None:
        return {'error': 'Could not load image'}
    
    # Test all 4 rotations
    rotations = [0, 90, 180, 270]
    results = {}
    
    for angle in rotations:
        face_count = detect_faces_at_rotation(img, angle)
        results[angle] = face_count
    
    # Find the rotation with most faces
    if all(count == 0 for count in results.values()):
        return {
            'status': 'unknown',
            'results': results,
            'best_rotation': None,
            'message': 'No faces detected at any rotation'
        }
    
    best_rotation = max(results, key=results.get)
    
    return {
        'status': 'success',
        'results': results,
        'best_rotation': best_rotation,
        'confidence': results[best_rotation]
    }

def analyze_image_features(image_path):
    """
    Analyze general image features that might indicate orientation.
    This uses edge detection to see if more detail is at top or bottom.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 'error'
    
    # Split image into top and bottom halves
    height = img.shape[0]
    top_half = img[:height//2, :]
    bottom_half = img[height//2:, :]
    
    # Detect edges
    edges_top = cv2.Canny(top_half, 100, 200)
    edges_bottom = cv2.Canny(bottom_half, 100, 200)
    
    # Count edge pixels (more edges usually at top for natural images)
    edge_density_top = np.sum(edges_top) / edges_top.size
    edge_density_bottom = np.sum(edges_bottom) / edges_bottom.size
    
    # If bottom has significantly more edges, might be upside down
    if edge_density_bottom > edge_density_top * 1.3:
        return 'possibly_upside_down'
    else:
        return 'likely_upright'

def rotate_and_save_image(image_path, rotation_angle, output_path=None):
    """
    Rotate an image and save it.
    
    Args:
        image_path: Input image path
        rotation_angle: Angle to rotate (0, 90, 180, 270)
        output_path: Output path (if None, will create a new filename)
    
    Returns:
        Path to saved image
    """
    img = cv2.imread(image_path)
    
    if img is None:
        return None
    
    # Rotate the image
    if rotation_angle == 90:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
    elif rotation_angle == 270:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        rotated_img = img
    
    # Generate output filename if not provided
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_corrected{ext}"
    
    # Save the rotated image
    cv2.imwrite(output_path, rotated_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return output_path

def main(image_path, auto_correct=False, output_path=None):
    """
    Main function to determine image orientation across all 4 rotations.
    """
    print(f"Analyzing: {image_path}\n")
    print("=" * 60)
    
    # Method 1: Face Detection at all rotations
    print("Method 1: Face Detection at All Rotations")
    result = check_orientation_by_face(image_path)
    
    if 'error' in result:
        print("❌ Error: Could not load image")
        return
    
    if result['status'] == 'unknown':
        print(f"⚠ Result: {result['message']}")
    else:
        print("\nFaces detected at each rotation:")
        for angle in [0, 90, 180, 270]:
            count = result['results'][angle]
            is_best = (angle == result['best_rotation'])
            marker = "⭐" if is_best else "  "
            print(f"{marker} {angle:3d}° - {count} face(s) detected")
        
        print("\n" + "-" * 60)
        print(f"\n🎯 BEST ORIENTATION: {result['best_rotation']}°")
        print(f"   Faces detected: {result['confidence']}")
        
        best_angle = result['best_rotation']
        if best_angle == 0:
            print("\n✅ Image is UPRIGHT (correct orientation)")
        elif best_angle == 180:
            print("\n🔄 Image is UPSIDE DOWN (needs 180° rotation)")
        elif best_angle == 90:
            print("\n🔄 Image is rotated 90° COUNTERCLOCKWISE")
            print("   (needs 90° clockwise rotation to correct)")
        elif best_angle == 270:
            print("\n🔄 Image is rotated 90° CLOCKWISE")
            print("   (needs 90° counterclockwise rotation to correct)")
    
    print("\n" + "=" * 60)
    
    # Method 2: Edge Analysis (keeping for reference)
    print("\nMethod 2: Edge Distribution Analysis")
    feature_result = analyze_image_features(image_path)
    print(f"✓ Result: {feature_result.replace('_', ' ').title()}")
    
    print("\n" + "=" * 60)
    
    # Final recommendation
    print("\n📊 FINAL ASSESSMENT:")
    if result['status'] == 'success':
        best_angle = result['best_rotation']
        if best_angle == 0:
            print("✓ The image is UPRIGHT")
        else:
            print(f"🔄 The image needs {best_angle}° rotation to be upright")
            if best_angle == 180:
                print("   (Image is upside down)")
            elif best_angle == 90:
                print("   (Image is rotated 90° counterclockwise)")
            elif best_angle == 270:
                print("   (Image is rotated 90° clockwise)")
            
            # Auto-correct if requested
            if auto_correct:
                print("\n" + "=" * 60)
                print("🔧 AUTO-CORRECTING IMAGE")
                print("=" * 60)
                
                output_file = rotate_and_save_image(image_path, best_angle, output_path)
                if output_file:
                    print(f"\n✅ Corrected image saved to: {output_file}")
                    print(f"   Rotation applied: {best_angle}°")
                else:
                    print(f"\n❌ Error: Could not save corrected image")
            elif best_angle != 0:
                print(f"\n💡 To automatically correct the orientation, run:")
                print(f"   python {sys.argv[0]} \"{image_path}\" --correct")
    else:
        print(f"❓ Cannot determine with certainty")
        print(f"   Edge analysis suggests: {feature_result.replace('_', ' ')}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Detect image orientation using face detection (tests all 4 rotations)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Just detect orientation
  python upside_down_detector.py photo.jpg
  
  # Detect and auto-correct
  python upside_down_detector.py photo.jpg --correct
  
  # Specify output filename
  python upside_down_detector.py photo.jpg --correct --output fixed.jpg
        """
    )
    
    parser.add_argument('image', help='Path to the image file')
    parser.add_argument('--correct', '-c', action='store_true', 
                       help='Automatically rotate and save the corrected image')
    parser.add_argument('--output', '-o', help='Output filename for corrected image')
    
    args = parser.parse_args()
    
    main(args.image, auto_correct=args.correct, output_path=args.output)