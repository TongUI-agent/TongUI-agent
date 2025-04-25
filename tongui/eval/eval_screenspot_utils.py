import os
import re
import ast
import sys
import pdb
import json
import torch
import wandb
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
import torch.distributed as dist
from accelerate.utils import gather_object
from tongui.data.data_utils import AverageMeter, ProgressMeter, Summary, dict_to_cuda
from tongui.utils.utils import save_json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
logging.basicConfig(level=logging.INFO)

def broadcast_value(value, src=0, local_rank=0):
    tensor = torch.tensor([value], dtype=torch.float32).to(f'cuda:{local_rank}')
    dist.broadcast(tensor, src=src)
    return tensor.item()

def get_bbox(bbox, img_size, xy_int):
    x1, y1, w, h = bbox
    weight, height = img_size

    # x1y1wh to x1y1x2y2
    bbox = [x1, y1, x1 + w, y1 + h]

    # normalisation
    bbox = [bbox[0] / weight, bbox[1] / height, 
            bbox[2] / weight, bbox[3] / height]
    if xy_int:
        bbox = [int(item * 1000) for item in bbox]
    return bbox

def pointinbbox(pred_point, gt_bbox):
    # pred_point: [x, y] in [0, 1]
    # gt_bbox: [x1, y1, x2, y2] in [0, 1]
    if (gt_bbox[0] <= pred_point[0] <= gt_bbox[2]) and (gt_bbox[1] <= pred_point[1] <= gt_bbox[3]):
        return True
    else:
        return False

def draw_point_bbox(image_path, point=None, bbox=None, radius=5, line=3):
    if type(image_path) is list and len(image_path) == 1:
        image_path = image_path[0]
    elif type(image_path) is str:
        pass
    else:
        raise ValueError(f'image_path {type(image_path)} and len {len(image_path)}')
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    if point is not None:
        x, y = point[0] * width, point[1] * height
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='blue', outline='blue')
    if bbox is not None:
        x1, y1, x2, y2 = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
        draw.rectangle([x1, y1, x2, y2], outline='red', width=line)

    image_draw = np.array(image)
    return image_draw

def calculate_screenspot_metrics(results):
    metrics = {}
    for type_ in results:
        num_step = 0
        num_success = 0

        for step in results[type_]:
            num_step += 1
            num_success += step["acc"]

        metrics[f"{type_} Success Rate"] = num_success / num_step

    for key, value in metrics.items():
        logging.info(f"[{key}]: {value}")
    return metrics

def convert_list_tensor_to_number(input_list):
    return [item.item() if isinstance(item, torch.Tensor) else item for item in input_list]
        
def validate_screenspot(val_loader, model_engine, processor, epoch, global_step, writer, args, media=True):
    model_engine.eval()

    answers_unique = []
    generated_texts_unique = []
    outputs_unique = []

    results = {}
    metric = 0
    for i, input_dict in enumerate(tqdm(val_loader)):
        if args.debug and i == 100:
            break
        torch.cuda.empty_cache()
        
        try:
            input_dict, item = input_dict
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["pixel_values"] = input_dict["pixel_values"].half()
            elif args.precision == "bf16":
                input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()
            else:
                input_dict["pixel_values"] = input_dict["pixel_values"].float()

            forward_dict = dict(
                pixel_values=input_dict["pixel_values"],
                input_ids=input_dict["input_ids"],
                labels=input_dict["labels"],
            )

            # Add optional inputs if they exist
            forward_dict.update(image_grid_thw=input_dict["image_sizes"].squeeze(dim=0)) if "image_sizes" in input_dict else None
            forward_dict.update(patch_assign=input_dict["patch_assign"]) if "patch_assign" in input_dict else None
            forward_dict.update(patch_assign_len=input_dict["patch_assign_len"]) if "patch_assign_len" in input_dict else None
            forward_dict.update(patch_pos=input_dict["patch_pos"]) if "patch_pos" in input_dict else None
            forward_dict.update(select_mask=input_dict["select_mask"]) if "select_mask" in input_dict else None

            generate_ids = model_engine.generate(
                **forward_dict,
                max_new_tokens=128,
                eos_token_id=processor.tokenizer.eos_token_id,
                do_sample=False
            )
            
            generate_ids = generate_ids[:, input_dict['input_ids'].shape[1]:]
            generated_texts = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            meta = item
            
            split_i = meta['split']
            if isinstance(split_i, list):
                split_i = split_i[0]
            
            if split_i not in results:
                results[split_i] = {}

            type_i = meta['data_type']
            if isinstance(type_i, list):
                type_i = type_i[0]
            
            if type_i not in results[split_i]:
                results[split_i][type_i] = []

            step_result = {
                "split": split_i,
                'data_type': type_i,
                "anno_id": meta['id'],
                "img_path": meta['img_url_abs'],
                "instruction": meta['task'],
                "sentence": generated_texts,
                "bbox": meta['bbox'],
                "meta": meta
            }

            img_size = meta['img_size']
            gt_bbox = get_bbox(meta['bbox'], img_size, args.xy_int)
            step_result['gt_bbox'] = convert_list_tensor_to_number(gt_bbox)
            step_result['bbox'] = convert_list_tensor_to_number(step_result["bbox"])
            step_result["meta"]["img_size"] = convert_list_tensor_to_number(img_size)
            step_result["meta"]["bbox"] = convert_list_tensor_to_number(step_result["meta"]["bbox"])

            try:
                pred_point = ast.literal_eval(generated_texts)
                step_result['pred_point'] = pred_point

                if pointinbbox(pred_point, gt_bbox):
                    step_result["acc"] = 1
                else:
                    step_result["acc"] = 0
                    
            except Exception as e:
                logging.warning(f"Error parsing prediction: {e}")
                step_result["acc"] = 0

            results[split_i][type_i].append(step_result)

        except Exception as e:
            logging.warning(f"Error processing sample {i}: {str(e)}")
            continue

    # Calculate metrics
    eval_dict = {}
    for split in results.keys():
        logging.info("==="*10)
        logging.info(f"{split}")
        logging.info("==="*10)
        eval_dict[split] = calculate_screenspot_metrics(results[split])

    if wandb.run is None:
        wandb.init(project="AgentNetEval", name="screenspot")
    
    if not args.debug:
        for split in eval_dict.keys():
            for key, value in eval_dict[split].items():
                if isinstance(value, list):
                    continue
                wandb.log({f"metrics/screenspot/{split}/{key}": value}, step=global_step)

    score_all = [value for split in eval_dict.values() for value in split.values()]
    metric = sum(score_all) / len(score_all)
    eval_dict['Avg Success Rate'] = metric
    wandb.log({"metrics/screenspot/Avg Success Rate": metric}, step=global_step)

    if media:
        images_list = []
        for split in results.keys():
            for type_ in results[split].keys():
                sample = random.choice(results[split][type_])
                img_anno = sample['anno_id']
                img_url = sample['img_path']
                img_inst = sample['instruction']
                gt_bbox = sample['gt_bbox']
                if 'pred_point' in sample:
                    pred_point = sample['pred_point']
                    img_array = draw_point_bbox(img_url, pred_point, gt_bbox, radius=5, line=3)
                else:
                    img_array = draw_point_bbox(img_url, None, gt_bbox)
                images = wandb.Image(img_array, caption=f"{split}/{type_}/{img_anno}_{img_inst}")
                images_list.append(images)
        wandb.log({"examples": images_list}, step=global_step)

    print(results)
    print("#" * 100)
    print(eval_dict)
    save_json(results, os.path.join(args.tmp_dir, f'screenspot_epo{epoch}_tmp_dict.json'))
    save_json(eval_dict, os.path.join(args.tmp_dir, f'screenspot_epo{epoch}_res_dict.json'))

    return metric