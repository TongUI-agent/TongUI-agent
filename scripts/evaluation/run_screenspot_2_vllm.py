import base64
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import openai
from PIL import Image, ImageDraw
from tongui.data.template.screenspot import _SCREENSPOT_SYSTEM, _SYSTEM_point, GROUNDING
from tqdm import tqdm
from io import BytesIO

from utils.qwen import perform_gui_grounding_with_api

def load_screenspot_data(
        dataset_dir,
        limit=-1, 
        filter_by_split: str | None = None
    ):
    # Recreating the dataset loading logic without using the class
    base_image_dir = os.path.join(dataset_dir, "ScreenSpot-v2")
    meta_dir = base_image_dir
    img_dir = os.path.join(base_image_dir, "screenspotv2_image")

    # Load the JSON data
    json_data = []
    for filename in os.listdir(meta_dir):
        if filename.endswith(".json"):
            with open(os.path.join(meta_dir, filename)) as f:
                file_data = json.load(f)
                for item in file_data:
                    # Add the filename to the item
                    item['filename'] = filename.removeprefix("screenspot_").removesuffix("_v2.json")
                    json_data.append(item)

    print(f"Dataset: Screenspot; # samples: {len(json_data)}")
    if limit > 0:
        json_data = json_data[:limit]
    # if filter_by_split:
    #     json_data = [item for item in json_data if item['split'] == filter_by_split]
    return json_data, img_dir

def predict_point(client: openai.OpenAI, image_path: str, task: str, model: str = "tongui-3b"):
    # Read the image file and encode it as base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Get image dimensions
 
    buffered = BytesIO()
    with Image.open(image_path).convert("RGB") as img:
        # img = resize_image(
        #     img,
        #     max_pixels=2500*28*28
        # )
        img_width, img_height = img.size
        # img.save(buffered, format="JPEG")
        # encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    prompt = _SCREENSPOT_SYSTEM + ' ' + _SYSTEM_point
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": 
                [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    },
                    {"type": "text", "text": task}
                ]
            },
        ],
        max_completion_tokens=64,
        temperature=1e-6,
        top_p=0.95,
        extra_body={
            "top_k": 50
        }
    )
    
    # Extract the point from the response
    prediction_text = response.choices[0].message.content
    print(f"Raw prediction: {prediction_text}; Usage: {response.usage.total_tokens}")
    
    # Try to parse the point from the response
    try:
        # The model might return a point as [x, y] or a bounding box as [x1, y1, x2, y2]
        coords = json.loads(prediction_text)
        
        if len(coords) == 2:  # It's a point [x, y]
            # Convert from normalized coordinates (0-1) to pixel coordinates
            x = int(coords[0] * img_width)
            y = int(coords[1] * img_height)
            return [x, y]
            
        elif len(coords) == 4:  # It's a bounding box [x1, y1, x2, y2]
            # Convert from normalized coordinates (0-1) to pixel coordinates
            x1 = int(coords[0] * img_width)
            y1 = int(coords[1] * img_height)
            x2 = int(coords[2] * img_width)
            y2 = int(coords[3] * img_height)
            
            # Calculate the center point of the bounding box
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            return [int(x), int(y)]
            
        else:
            print(f"Unexpected format: {coords}")
            return None
            
    except json.JSONDecodeError:
        print(f"Error parsing coordinates: {prediction_text}")
        return None
    
def predict_point_by_uitars(client: openai.OpenAI, image_path: str, task: str, model: str = "tongui-3b"):
    # Read the image file and encode it as base64
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Get image dimensions
 
    buffered = BytesIO()
    with Image.open(image_path).convert("RGB") as img:
        # img = resize_image(
        #     img,
        #     max_pixels=2500*28*28
        # )
        img_width, img_height = img.size
        # img.save(buffered, format="JPEG")
        # encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    prompt = GROUNDING + task
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": 
                [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    },
                    {"type": "text", "text": prompt},
                ]
            },
        ],
        max_completion_tokens=64,
        temperature=1e-6,
        top_p=0.95,
        extra_body={
            "top_k": 50
        }
    )
    
    # Extract the point from the response
    prediction_text = response.choices[0].message.content
    print(f"Raw prediction: {prediction_text}; Usage: {response.usage.total_tokens}")
    
    # Try to parse the point from the response
    try:
        # The model might return a point as [x, y] or a bounding box as [x1, y1, x2, y2]
        # coords = json.loads(prediction_text)
        coords =  [int(num) / 1000 for num in prediction_text.strip("()").split(',')]
        
        if len(coords) == 2:  # It's a point [x, y]
            # Convert from normalized coordinates (0-1) to pixel coordinates
            x = int(coords[0] * img_width)
            y = int(coords[1] * img_height)
            return [x, y]
            
        # elif len(coords) == 4:  # It's a bounding box [x1, y1, x2, y2]
        #     # Convert from normalized coordinates (0-1) to pixel coordinates
        #     x1 = int(coords[0] * img_width)
        #     y1 = int(coords[1] * img_height)
        #     x2 = int(coords[2] * img_width)
        #     y2 = int(coords[3] * img_height)
            
        #     # Calculate the center point of the bounding box
        #     x = (x1 + x2) / 2
        #     y = (y1 + y2) / 2
        #     return [int(x), int(y)]
            
        else:
            print(f"Unexpected format: {coords}")
            return None
            
    except Exception as e:
        print(f"Error parsing coordinates: {prediction_text}; Error: {e}")
        return None
    
def predict_point_by_qwen(client: openai.OpenAI, image_path: str, task: str, model: str = "tongui-3b"):
    # Read the image file and encode it as base64
    
    # Extract the point from the response
    prediction_text = perform_gui_grounding_with_api(
        image_path,
        task,
        model,
        client
    )
    print(f"Raw prediction: {prediction_text}")
    
    # Try to parse the point from the response
    try:
        # The model might return a point as [x, y] or a bounding box as [x1, y1, x2, y2]
        # coords = json.loads(prediction_text)
        coords = prediction_text
        
        if len(coords) == 2:  # It's a point [x, y]
            # Convert from normalized coordinates (0-1) to pixel coordinates
            x = int(coords[0])
            y = int(coords[1])
            return [x, y]
            
        else:
            print(f"Unexpected format: {coords}")
            return None
            
    except Exception as e:
        print(f"Error parsing coordinates: {prediction_text}; Error: {e}")
        return None

def is_point_in_box(point, box):
    """
    Check if a point [x, y] is inside a bounding box [x1, y1, x2, y2]
    """
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def display_point_on_image(image_path, point, gt_box=None, save_path=None):
    """
    Display a point on an image, optionally with a ground truth box
    point format: [x, y]
    gt_box format: [x1, y1, x2, y2]
    """
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Draw the point (as a small circle)
    point_radius = 5
    x, y = point
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)], 
        fill="red", outline="red"
    )
    
    # Draw the ground truth box if provided
    if gt_box:
        draw.rectangle([(gt_box[0], gt_box[1]), (gt_box[2], gt_box[3])], outline="green", width=2)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(np.array(img))  # Convert PIL Image to numpy array
    plt.axis('off')
    
    # Add a title indicating if the point is in the box
    if gt_box:
        is_in_box = is_point_in_box(point, gt_box)
        plt.title(f"Point in box: {is_in_box}", fontsize=14)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()

def print_accuracy_metrics(group_metrics):
    """
    Print accuracy metrics for different groups
    """
    print("\n===== ACCURACY METRICS =====")
    
    # 1. Overall accuracy
    total_correct = sum(metrics['correct'] for metrics in group_metrics.values())
    total_samples = sum(metrics['total'] for metrics in group_metrics.values())
    overall_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0
    print(f"\n1. Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{total_samples})")
    
    # 2. Accuracy by split and data_type
    print("\n2. Accuracy by Split and Data Type:")
    
    # Group by split
    split_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    for (split, data_type), metrics in group_metrics.items():
        split_metrics[split]['correct'] += metrics['correct']
        split_metrics[split]['total'] += metrics['total']
    
    print("  By Split:")
    for split, metrics in split_metrics.items():
        accuracy = (metrics['correct'] / metrics['total'] * 100) if metrics['total'] > 0 else 0
        print(f"    - {split}: {accuracy:.2f}% ({metrics['correct']}/{metrics['total']})")
    
    # Group by data_type
    data_type_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
    for (split, data_type), metrics in group_metrics.items():
        data_type_metrics[data_type]['correct'] += metrics['correct']
        data_type_metrics[data_type]['total'] += metrics['total']
    
    print("  By Data Type:")
    for data_type, metrics in data_type_metrics.items():
        accuracy = (metrics['correct'] / metrics['total'] * 100) if metrics['total'] > 0 else 0
        print(f"    - {data_type}: {accuracy:.2f}% ({metrics['correct']}/{metrics['total']})")
    
    # Detailed breakdown by split and data_type
    print("  By Split and Data Type:")
    for (split, data_type), metrics in group_metrics.items():
        accuracy = (metrics['correct'] / metrics['total'] * 100) if metrics['total'] > 0 else 0
        print(f"    - {split}, {data_type}: {accuracy:.2f}% ({metrics['correct']}/{metrics['total']})")
    
    # 3. Average accuracy across groups
    print("\n3. Average Accuracy Across Groups:")
    
    # Average by split
    split_avg = {}
    for split, metrics in split_metrics.items():
        split_avg[split] = (metrics['correct'] / metrics['total'] * 100) if metrics['total'] > 0 else 0
    
    avg_split_accuracy = sum(split_avg.values()) / len(split_avg) if split_avg else 0
    print(f"  Average across splits: {avg_split_accuracy:.2f}%")
    
    # Average by data_type
    data_type_avg = {}
    for data_type, metrics in data_type_metrics.items():
        data_type_avg[data_type] = (metrics['correct'] / metrics['total'] * 100) if metrics['total'] > 0 else 0
    
    avg_data_type_accuracy = sum(data_type_avg.values()) / len(data_type_avg) if data_type_avg else 0
    print(f"  Average across data types: {avg_data_type_accuracy:.2f}%")
    
    # Average across all groups
    group_accuracies = []
    for (split, data_type), metrics in group_metrics.items():
        if metrics['total'] > 0:
            group_accuracies.append(metrics['correct'] / metrics['total'] * 100)
    
    avg_group_accuracy = sum(group_accuracies) / len(group_accuracies) if group_accuracies else 0
    print(f"  Average across all groups: {avg_group_accuracy:.2f}%")

def main():
    # Set to True to display points on images
    DISPLAY_IMAGES = True
    LIMIT = -1
    # MODEL = "tongui-3b"
    # MODEL = "tongui-7b"
    # MODEL = "ui-tars-2b"
    # MODEL = "ui-tars-7b"
    # MODEL = "qwen2.5-7b"
    MODEL = "qwen2.5-3b"
    ENDPOINT = "http://localhost:50004/v1"
    FILTER_BY_SPLIT = None
    # Initialize OpenAI client
    client = openai.OpenAI(
        api_key="EMPTY",
        base_url=ENDPOINT,
    )
    # Load the dataset
    json_data, img_dir = load_screenspot_data(
        "dataset",
        limit=LIMIT,
        filter_by_split=FILTER_BY_SPLIT
    )

    # Track metrics by group (split and data_type)
    group_metrics = defaultdict(lambda: {'correct': 0, 'total': 0})

    # Process each sample
    for idx, item in tqdm(enumerate(json_data), total=len(json_data)):
        # print(f"\nProcessing sample {idx+1}/{len(json_data)}")
        # print(f"Item: {item}")
        
        # Get image path
        image_path = os.path.join(img_dir, item['img_filename'])
        
        # Skip if image doesn't exist
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        # Get task description
        task = item['instruction']
        
        # Get ground truth bbox
        gt_bbox = item['bbox']  # [x, y, width, height]
        # Convert to [x1, y1, x2, y2] format
        gt_bbox_xyxy = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]
        print(f"Ground truth bbox [x1, y1, x2, y2]: {gt_bbox_xyxy}")
        
        # Make prediction
        # pred_point = predict_point(client, image_path, task, model=MODEL)
        if MODEL.startswith("ui-tars"):
            pred_point = predict_point_by_uitars(client, image_path, task, model=MODEL)
        elif MODEL.startswith("qwen"):
            pred_point = predict_point_by_qwen(client, image_path, task, model=MODEL)
        else:
            pred_point = predict_point(client, image_path, task, model=MODEL)
        print(f"Predicted point [x, y]: {pred_point}")
        
        # Check if point is in box
        if pred_point:
            is_correct = is_point_in_box(pred_point, gt_bbox_xyxy)
            print(f"Point in box: {is_correct}")
            
            # Get the group key (split and data_type)
            split = item.get('filename', 'unknown')
            data_type = item.get('data_type', 'unknown')
            group_key = (split, data_type)
            
            # Update metrics
            group_metrics[group_key]['total'] += 1
            if is_correct:
                group_metrics[group_key]['correct'] += 1
            
            # Display the point on the image if requested
            if DISPLAY_IMAGES:
                output_dir = "point_visualizations"
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"sample_{idx}_{os.path.basename(item['img_filename'])}")
                display_point_on_image(image_path, pred_point, gt_bbox_xyxy, save_path)
    
    # Print accuracy metrics
    print_accuracy_metrics(group_metrics)
    with open(f"results_{MODEL}_screenspot_pro.json", "w") as f:
        json.dump(group_metrics, f, indent=4)

if __name__ == "__main__":
    main()
