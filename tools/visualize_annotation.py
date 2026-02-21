import json
import cv2
import numpy as np
import argparse
from pathlib import Path

def visualize_annotation(image_path, annotation_path, output_path, arrow_length=50):
    """
    Visualize annotations by drawing arrows on the image.
    
    Args:
        image_path (str): Path to the input image
        annotation_path (str): Path to the JSON annotation file
        output_path (str): Path to save the output visualization
        arrow_length (int): Length of the arrow in pixels, default 50
        
    Returns:
        None: Saves the visualization to output_path
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    with open(annotation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    detections = data.get('detections', [])
    
    print(f"Visualizing {len(detections)} detections...")
    
    for idx, detection in enumerate(detections):
        center = detection['center']
        x = int(center[0])
        y = int(center[1])
        
        direction = detection['direction']
        cos_theta = float(direction['cos'])
        sin_theta = float(direction['sin'])
        
        end_x = int(x + cos_theta * arrow_length)
        end_y = int(y + sin_theta * arrow_length)
        
        cv2.arrowedLine(image, (x, y), (end_x, end_y), 
                       (0, 0, 255), 3, tipLength=0.3)
        
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1) 

    cv2.imwrite(str(output_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    print(f"Visualization saved to: {output_path}")
    print(f"Image size: {image.shape[1]} x {image.shape[0]}")


def main():

    parser = argparse.ArgumentParser(
        description='Visualize hair follicle annotations with arrows',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python visualize_annotation.py --image sample.jpg --annotation sample.json --output result.png
  
  # Custom arrow length
  python visualize_annotation.py --image sample.jpg --annotation sample.json --output result.png --arrow_length 100
        """
    )
    
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Path to input image file')
    parser.add_argument('--annotation', '-a', type=str, required=True,
                       help='Path to JSON annotation file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Path to save output visualization (PNG format)')
    parser.add_argument('--arrow_length', '-l', type=int, default=50,
                       help='Length of direction arrows in pixels (default: 50)')
    
    args = parser.parse_args()
    
    image_path = Path(args.image)
    annotation_path = Path(args.annotation)
    
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return
    
    if not annotation_path.exists():
        print(f"Error: Annotation file not found: {annotation_path}")
        return
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        visualize_annotation(
            image_path=image_path,
            annotation_path=annotation_path,
            output_path=output_path,
            arrow_length=args.arrow_length
        )
        print("✓ Visualization completed successfully!")
    except Exception as e:
        print(f"✗ Error during visualization: {e}")
        raise


if __name__ == '__main__':
    main()
