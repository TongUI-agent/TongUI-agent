import json
import argparse
from PIL import Image, ImageDraw
import os
parser = argparse.ArgumentParser()
parser.add_argument("--data-path", type=str, required=True)
args = parser.parse_args()
def visualize_bbox(image_root, image_url, raw_bbox, predictions):
    
    image_path = os.path.join(image_root, image_url)
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    print(raw_bbox)
    x1, y1, w, h = raw_bbox
    x2 = x1 + w
    y2 = y1 + h
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    draw.ellipse([predictions[0] * img_size[0], predictions[1] * img_size[1], predictions[0] * img_size[0] + 10, predictions[1] * img_size[1] + 10], fill="blue")
data_path = args.data_path

with open(data_path, "r") as f:
    data = json.load(f)
image_root = "training_data/ScreenSpot/images"

import dataclasses
class Stats:
    total: int = 0
    correct: int = 0
data_type_stats = {
    "text": Stats(),
    "icon": Stats(),
    "total": Stats(),
}
for item in data:
    task = item["task"]
    image_url = item["img_url"]
    img_size = item["img_size"]
    bbox = item["bbox"]
    raw_bbox = item["bbox"]
    predictions = item["predictions"]
    print(predictions)
    # predictions is a point scale 0-1, bbox is [x1, y1, w, h]

    bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[0] / img_size[0] + bbox[2] / img_size[0], bbox[1] / img_size[1] + bbox[3] / img_size[1]]
    # check if point is in the bbox
    print(predictions, bbox)
    correct = 0
    if predictions[0] < bbox[0] or predictions[0] > bbox[2] or predictions[1] < bbox[1] or predictions[1] > bbox[3]:
        print("point is not in the bbox")
    else:
        print("point is in the bbox")
        correct = 1
    data_type = item["data_type"]
    data_type_stats[data_type].total += 1
    data_type_stats[data_type].correct += correct
    data_type_stats["total"].total += 1
    data_type_stats["total"].correct += correct
        


print("Stats:")
for data_type, stats in data_type_stats.items():
    print(f"{data_type}: {stats.correct / stats.total}")




