import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

class BoxDrawer:
    """A tool for drawing boxes on images using coordinate arrays."""
    
    def __init__(self):
        pass
    
    def draw_boxes_pil(self, image_path, coordinates, output_path=None, 
                       box_color='red', line_width=3, show_image=True):
        """
        Draw boxes on an image using PIL/Pillow.
        
        Args:
            image_path (str): Path to the input image
            coordinates (list): List of coordinate arrays. Each array should contain
                               [x1, x2, y1, y2] where (x1,y1) and (x2,y2) define the box
            output_path (str, optional): Path to save the output image
            box_color (str): Color of the box outline
            line_width (int): Width of the box outline
            show_image (bool): Whether to display the image
        
        Returns:
            PIL.Image: Image with boxes drawn
        """
        # Open the image
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        
        # Draw each box
        for coord in coordinates:
            if len(coord) != 4:
                raise ValueError("Each coordinate array must contain exactly 4 values: [x1, x2, y1, y2]")
            
            x1, x2, y1, y2 = coord
            
            # Ensure we have the correct order (top-left to bottom-right)
            left = min(x1, x2)
            right = max(x1, x2)
            top = min(y1, y2)
            bottom = max(y1, y2)
            
            # Draw rectangle
            draw.rectangle([left, top, right, bottom], 
                         outline=box_color, width=line_width)
        
        # Save if output path provided
        if output_path:
            image.save(output_path)
        
        # Display if requested
        if show_image:
            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            plt.axis('off')
            plt.title('Image with Boxes')
            plt.show()
        
        return image
    
    def draw_boxes_opencv(self, image_path, coordinates, output_path=None,
                         box_color=(0, 0, 255), line_width=3, show_image=True):
        """
        Draw boxes on an image using OpenCV.
        
        Args:
            image_path (str): Path to the input image
            coordinates (list): List of coordinate arrays. Each array should contain
                               [x1, x2, y1, y2] where (x1,y1) and (x2,y2) define the box
            output_path (str, optional): Path to save the output image
            box_color (tuple): Color of the box outline in BGR format (B, G, R)
            line_width (int): Width of the box outline
            show_image (bool): Whether to display the image
        
        Returns:
            numpy.ndarray: Image array with boxes drawn
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Draw each box
        for coord in coordinates:
            if len(coord) != 4:
                raise ValueError("Each coordinate array must contain exactly 4 values: [x1, x2, y1, y2]")
            
            x1, x2, y1, y2 = coord
            
            # Ensure we have the correct order
            left = min(x1, x2)
            right = max(x1, x2)
            top = min(y1, y2)
            bottom = max(y1, y2)
            
            # Draw rectangle
            cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), 
                         box_color, line_width)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, image)
        
        # Display if requested
        if show_image:
            # Convert BGR to RGB for matplotlib display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 8))
            plt.imshow(image_rgb)
            plt.axis('off')
            plt.title('Image with Boxes')
            plt.show()
        
        return image

# Example usage
def example_usage():
    """Example of how to use the BoxDrawer tool."""
    
    # Initialize the box drawer
    drawer = BoxDrawer()
    
    # Example coordinate arrays - each contains [x1, x2, y1, y2]
    coordinates = [
        [100, 300, 50, 200],   # First box
        [400, 600, 100, 250],  # Second box
        [200, 500, 300, 450]   # Third box
    ]
    
    # Using PIL method
    try:
        image_with_boxes = drawer.draw_boxes_pil(
            image_path='your_image.jpg',
            coordinates=coordinates,
            output_path='output_pil.jpg',
            box_color='red',
            line_width=3
        )
        print("PIL method completed successfully!")
    except Exception as e:
        print(f"PIL method error: {e}")
    
    # Using OpenCV method
    try:
        image_with_boxes = drawer.draw_boxes_opencv(
            image_path='your_image.jpg',
            coordinates=coordinates,
            output_path='output_opencv.jpg',
            box_color=(0, 255, 0),  # Green in BGR
            line_width=3
        )
        print("OpenCV method completed successfully!")
    except Exception as e:
        print(f"OpenCV method error: {e}")

# Quick function for simple usage
def draw_boxes_simple(image_path, coordinates, output_path=None):
    """
    Simple function to quickly draw boxes on an image.
    
    Args:
        image_path (str): Path to input image
        coordinates (list): List of [x1, x2, y1, y2] arrays
        output_path (str, optional): Path to save output
    """
    drawer = BoxDrawer()
    return drawer.draw_boxes_pil(image_path, coordinates, output_path)

if __name__ == "__main__":
    # Run example (uncomment to test)
    # example_usage()
    
    # Or use the simple function:
    # coordinates = [[100, 300, 50, 200], [400, 600, 100, 250]]
    # draw_boxes_simple('your_image.jpg', coordinates, 'output.jpg')
    pass