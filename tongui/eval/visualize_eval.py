from PIL import Image, ImageDraw, ImageFont
import json
import os
from typing import Dict, List, Tuple
import math

class Mind2WebVisualizer:
    def __init__(self, json_path: str, base_img_dir: str):
        """
        Initialize the visualizer
        
        Args:
            json_path: Path to the JSON file containing annotations
            base_img_dir: Base directory containing the images
        """
        self.base_img_dir = base_img_dir
        
        # Load annotations
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        # Colors for different action types
        self.action_colors = {
            'CLICK': (255, 0, 0),  # Red
            'TYPE': (0, 255, 0),   # Green
        }

    def draw_annotation(self, img: Image.Image, anno: Dict) -> Image.Image:
        """
        Draw annotation on the image
        
        Args:
            img: PIL Image object
            anno: Annotation dictionary
        
        Returns:
            Annotated PIL Image
        """
        draw = ImageDraw.Draw(img)
        predict_action = anno["sentence"][0].split("Action:")[1].strip() if "Action:" in anno["sentence"][0] else anno["sentence"][0].strip()
        predict_action = json.loads(predict_action)
        # Get action details
        action = anno['meta']['answer']['action']
        value = anno['meta']['answer']['value']
        position = anno['meta']['answer']['position']
        print("Predict action", predict_action)
        print("Answer", anno['meta']['answer'])
        # Convert normalized coordinates to pixel coordinates
        img_w, img_h = img.size
        x = position[0] * img_w
        y = position[1] * img_h
        
        # Draw circle at action position
        radius = 10
        draw.ellipse(
            [(x-radius, y-radius), (x+radius, y+radius)],
            outline=self.action_colors.get(action, (0, 0, 255)),
            width=2
        )
        
        # Draw action text
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
            
        text = f"GT: {action}: {value}"
        print(text)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        # Position text above the circle
        text_x = max(0, min(x - text_w/2, img_w - text_w))
        text_y = max(0, y - radius - text_h - 5)
        
        # Draw white background for text
        padding = 2
        draw.rectangle(
            [
                (text_x - padding, text_y - padding),
                (text_x + text_w + padding, text_y + text_h + padding)
            ],
            fill=(255, 255, 255)
        )
        
        # Draw text
        draw.text(
            (text_x, text_y),
            text,
            fill=self.action_colors.get(action, (0, 0, 255)),
            font=font
        )
        
        # Draw predict action
        predict_action_text = f"Predict: {predict_action['action']}: {predict_action['value']}"
        predict_point = predict_action['position']
        predict_point_x = predict_point[0] * img_w
        predict_point_y = predict_point[1] * img_h
        draw.text(
            (predict_point_x, predict_point_y),
            predict_action_text,
            fill=self.action_colors.get(predict_action['action'], (0, 0, 255)),
            font=font
        )
        draw.ellipse(
            [(predict_point_x-radius, predict_point_y-radius), (predict_point_x+radius, predict_point_y+radius)],
            outline=self.action_colors.get(predict_action['action'], (0, 0, 255)),
            width=2
        )
        
        return img

    def visualize_sequence(self, split: str, sequence_id: int) -> List[Image.Image]:
        """
        Visualize a sequence of actions
        
        Args:
            split: Dataset split name
            sequence_id: Sequence ID
            
        Returns:
            List of annotated PIL Images
        """
        print("visualize_sequence", split, sequence_id)
        sequence = self.data[split][str(sequence_id)]
        annotated_images = []
        
        for anno in sequence:
            # Load image
            img_path = os.path.join(self.base_img_dir, anno['img_path'])
            try:
                img = Image.open(img_path)
            except FileNotFoundError:
                print(f"Image not found: {img_path}")
                continue
                
            # Draw annotation
            annotated_img = self.draw_annotation(img.copy(), anno)
            annotated_images.append(annotated_img)
            
        return annotated_images

def display_grid(images: List[Image.Image], max_cols: int = 3) -> Image.Image:
    """
    Display images in a grid
    
    Args:
        images: List of PIL Images
        max_cols: Maximum number of columns
        
    Returns:
        PIL Image containing the grid
    """
    if not images:
        return None
        
    n = len(images)
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)
    
    cell_width = max(img.width for img in images)
    cell_height = max(img.height for img in images)
    
    grid = Image.new('RGB', (cell_width * cols, cell_height * rows))
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid.paste(img, (col * cell_width, row * cell_height))
        
    return grid

def main():
    # Initialize visualizer
    json_path = "tmp/mind2web_epo0_tmp_dict.json"
    base_img_dir = "."  # Adjust this to your image directory
    
    visualizer = Mind2WebVisualizer(json_path, base_img_dir)
    
    # Visualize first sequence
    annotated_images = []
    for i in range(10):
        annotated_images += visualizer.visualize_sequence("test_task", i)
    print(len(annotated_images))
    if annotated_images:
        # Display as grid
        grid = display_grid(annotated_images)
        if grid:
            # grid.show()
            # Optionally save
            grid.save("visualization.png")
    else:
        print("No images were successfully processed")

if __name__ == "__main__":
    main()