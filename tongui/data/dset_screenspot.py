import os
import cv2
import pdb
import json
import random
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F


import sys
sys.path.append('.')
from src.dataset_preprocess.data.template import screenspot_to_qwen, batch_add_answer

class ScreenSpotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir,
        dataset,
        json_data,
        processor,
        inference=False,
        args_dict={},
    ):
        self.processor = processor
        self.min_pixels = processor.image_processor.min_pixels
        self.max_pixels = processor.image_processor.max_pixels
        self.inference = inference

        self.base_image_dir = os.path.join(dataset_dir, 'ScreenSpot')
        META_DIR = os.path.join(self.base_image_dir, "metadata")
        self.IMG_DIR = os.path.join(self.base_image_dir, "images")
        with open(os.path.join(META_DIR, "{}.json".format(json_data))) as f:
            self.json_data = json.load(f)

        self.samples_per_epoch = args_dict.get('samples_per_epoch', 1)
        self.xy_int = args_dict.get('xy_int', False)

        print(f"Dataset: Screenspot; Split: {json_data}; # samples: {len(self.json_data)}")

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        return self.get_sample(idx)

    def get_sample(self, idx):
        item = self.json_data[idx]
        print(item)
        if 'img_url' in item.keys():
            image_path = os.path.join(self.IMG_DIR, item["img_url"])
            image_list = [Image.open(image_path).convert("RGB")]
        else:
            image_path = ""
            image_list = None
        item['img_url_abs'] = image_path

        task = item['task']
        #print(image_path)
        img_dict = {
            'type': 'image', 
            'min_pixels': self.min_pixels, 
            'max_pixels': self.max_pixels,
            "image": f"file://{image_path}"}
        source = screenspot_to_qwen(task, img_dict, self.xy_int)
        # print(source)
        prompt = self.processor.apply_chat_template(source, tokenize=False, add_generation_prompt=True)
        data_dict_q = self.processor(text=prompt, images=image_list, return_tensors="pt",
                                        training=not self.inference)

        if 'labels' not in data_dict_q:
            data_dict_q['labels'] = data_dict_q['input_ids']

        data_dict = dict(
            input_ids=data_dict_q["input_ids"][0],
            pixel_values=data_dict_q["pixel_values"],
            image_sizes=data_dict_q["image_grid_thw"],
            labels=data_dict_q["labels"][0],
        )

        # Prepare elements for ShowUI
        for key in ['select_mask', 'patch_pos', 'patch_assign', 'patch_assign_len']:
            if key in data_dict_q:
                data_dict[key] = data_dict_q[key]

        return (
            data_dict,
            item,
        )

if __name__ == '__main__':
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
                                            # "Qwen/Qwen2-VL-2B-Instruct", 
                                            "/mnt/bofeidisk2/Qwen2.5-VL-3B-Instruct/",
                                            min_pixels=256*28*28, 
                                            max_pixels=1344*28*28,
                                            model_max_length=8196,
                                            )

    dataset = ScreenSpotDataset(
        "evaluation_data",
        "ScreenSpot",
        "hf_test_full",
        processor,
        inference=True
    )

    for i in range(len(dataset)):
        data = dataset.__getitem__(i)
        data_size = str(data[1]['img_size'])
        print(i, len(data[0]['input_ids']))