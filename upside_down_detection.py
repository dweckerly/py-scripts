#!/usr/bin/env python3
"""
Image Orientation Detector
Detects if an image is upside down using face detection and text orientation.
"""

import cv2
import numpy as np
from PIL import Image
import sys

def detect_faces(image_path):
    """
    Detect faces in the image and return their positions.
    Returns a list of face rectangles.
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

def check_orientation_by_face(image_path):
    """
    Check if image is upside down by detecting faces in original and rotated versions.
    Returns: 'upright', 'upside_down', or 'unknown'
    """
    faces_normal, img = detect_faces(image_path)
    
    if img is None:
        return 'error', 0, 0
    
    # Rotate image 180 degrees
    img_rotated = cv2.rotate(img, cv2.ROTATE_180)
    rotated_path = '/tmp/rotated_temp.jpg'
    cv2.imwrite(rotated_path, img_rotated)
    
    faces_rotated, _ = detect_faces(rotated_path)
    
    faces_normal_count = len(faces_normal) if faces_normal is not None else 0
    faces_rotated_count = len(faces_rotated) if faces_rotated is not None else 0
    
    if faces_normal_count == 0 and faces_rotated_count == 0:
        return 'unknown', 0, 0
    elif faces_rotated_count > faces_normal_count:
        return 'upside_down', faces_normal_count, faces_rotated_count
    else:
        return 'upright', faces_normal_count, faces_rotated_count

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

def main(image_path):
    """
    Main function to determine if an image is upside down.
    """
    print(f"Analyzing: {image_path}\n")
    print("=" * 60)
    
    # Method 1: Face Detection
    print("Method 1: Face Detection")
    orientation, normal_faces, rotated_faces = check_orientation_by_face(image_path)
    
    if orientation == 'error':
        print("❌ Error: Could not load image")
        return
    elif orientation == 'upside_down':
        print(f"✓ Result: Image appears to be UPSIDE DOWN")
        print(f"  - Faces detected in normal orientation: {normal_faces}")
        print(f"  - Faces detected when rotated 180°: {rotated_faces}")
    elif orientation == 'upright':
        print(f"✓ Result: Image appears to be UPRIGHT")
        print(f"  - Faces detected in normal orientation: {normal_faces}")
        print(f"  - Faces detected when rotated 180°: {rotated_faces}")
    else:
        print(f"⚠ Result: Cannot determine (no faces detected)")
    
    print("\n" + "-" * 60)
    
    # Method 2: Edge Analysis
    print("\nMethod 2: Edge Distribution Analysis")
    feature_result = analyze_image_features(image_path)
    print(f"✓ Result: {feature_result.replace('_', ' ').title()}")
    
    print("\n" + "=" * 60)
    
    # Final recommendation
    print("\n📊 FINAL ASSESSMENT:")
    if orientation == 'upside_down':
        print("🔄 The image is UPSIDE DOWN")
    elif orientation == 'upright':
        print("✓ The image is UPRIGHT")
    else:
        print(f"❓ Cannot determine with certainty")
        print(f"   Edge analysis suggests: {feature_result.replace('_', ' ')}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python upside_down_detector.py <image_path>")
        print("Example: python upside_down_detector.py photo.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    main(image_path)