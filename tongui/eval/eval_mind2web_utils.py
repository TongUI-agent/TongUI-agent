import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate.utils import gather_object
from tqdm import tqdm

from tongui.data.data_utils import dict_to_cuda
from tongui.eval.aitw_utils import pred2json
from tongui.utils.utils import save_json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import logging

logging.basicConfig(level=logging.INFO)

def broadcast_value(value, src=0, local_rank=0):
    tensor = torch.tensor([value], dtype=torch.float32).to(f'cuda:{local_rank}')
    dist.broadcast(tensor, src=src)
    return tensor.item()

def calculate_f1(pred, label):
    pred = set(pred.strip().split())
    label = set(label.strip().split())
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def get_bbox(meta):
    image_size = meta['img_size']
    action = meta['step']

    bbox = [action["bbox"]["x"], action["bbox"]["y"], action["bbox"]["x"] + action["bbox"]["width"],
            action["bbox"]["y"] + action["bbox"]["height"]]
    bbox = [bbox[0] / image_size[0], bbox[1] / image_size[1], bbox[2] / image_size[0], bbox[3] / image_size[1]]
    bbox = [round(item, 3) for item in bbox]
    return bbox

def calculate_mind2web_metrics(results):
    num_step = 0
    num_episode = 0
    num_op = 0
    num_ele = 0
    op_f1 = {'CLICK': [], 'TYPE': [], 'SELECT': []}
    macro_ele_acc = {}
    macro_step_acc = {}
    macro_action_f1 = {}
    num_step_success = 0
    num_episode_success = 0

    for i, (annot_id, item) in enumerate(results.items()):
        macro_ele_acc[i] = []
        macro_step_acc[i] = []
        macro_action_f1[i] = []
        num_episode += 1
        episode_success = True
        for step_result in item:
            num_step += 1

            if step_result["Op_match"]:
                num_op += 1

            if step_result["Ele_match"]:
                num_ele += 1
                macro_ele_acc[i].append(1)
            else:
                macro_ele_acc[i].append(0)

            if step_result["Op_F1"][1] in op_f1:
                op_f1[step_result["Op_F1"][1]].append(step_result["Op_F1"][0])
            macro_action_f1[i].append(step_result["Op_F1"][0])

            if step_result["Op_F1"][0] == 1.0 and step_result["Ele_match"]:
                num_step_success += 1
                macro_step_acc[i].append(1)
            else:
                macro_step_acc[i].append(0)
                episode_success = False

        if episode_success:
            num_episode_success += 1

    marco_op_f1 = np.mean([np.mean(x) for x in op_f1.values()])
    macro_ele_acc = np.mean([np.mean(x) for x in macro_ele_acc.values()])
    macro_step_acc = np.mean([np.mean(x) for x in macro_step_acc.values()])
    macro_action_f1 = np.mean([np.mean(x) for x in macro_action_f1.values()])

    logging.info("[Operation F1]: " + str(marco_op_f1))
    logging.info("[Element Acc]: " + str(num_ele / num_step))
    logging.info("[Step Success]: " + str(num_step_success / num_step))
    logging.info("[Episode Success]: " + str(num_episode_success / num_episode))
    logging.info("[Operation F1 cate]: " + str([np.mean(x) for x in op_f1.values()]))

    logging.info("[Macro Ele Acc]: " + str(macro_ele_acc))
    logging.info("[Macro Op F1]: " + str(macro_action_f1))
    logging.info("[Macro Step SR]: " + str(macro_step_acc))

    metrics = {
        "Operation F1": marco_op_f1,
        "Element Accuracy": num_ele / num_step,
        "Step Success": num_step_success / num_step,
        "Episode Success": num_episode_success / num_episode,
        "Operation F1 categories": [np.mean(x) for x in op_f1.values()],
        "Macro Element Accuracy": macro_ele_acc,
        "Macro Operation F1": macro_action_f1,
        "Macro Step Success Rate": macro_step_acc
    }
    return metrics

@torch.no_grad()
def validate_mind2web(val_loader, model_engine, processor, epoch, global_step, writer, args):
    model_engine.eval()

    answers_unique = []
    generated_texts_unique = []
    outputs_unique = []

    global_rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    metric = 0
    for i, input_dict in tqdm(enumerate(val_loader), total=len(val_loader)):
        torch.cuda.empty_cache()
        if args.debug and i > 100:
            break
        input_dict, item = input_dict
        input_dict = dict_to_cuda(input_dict, device=f'cuda:{local_rank}')
        if args.precision == "fp16":
            input_dict["pixel_values"] = input_dict["pixel_values"].half()
        elif args.precision == "bf16":
            input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()
        else:
            input_dict["pixel_values"] = input_dict["pixel_values"].float()

        with torch.no_grad():
            forward_dict = dict(
                pixel_values=input_dict["pixel_values"],
                input_ids=input_dict["input_ids"],
                labels=input_dict["labels"],
                )
            forward_dict.update(image_grid_thw=input_dict["image_sizes"].squeeze(dim=0))
            forward_dict.update(patch_assign=input_dict["patch_assign"]) if "patch_assign" in input_dict else None
            forward_dict.update(patch_assign_len=input_dict["patch_assign_len"]) if "patch_assign_len" in input_dict else None
            forward_dict.update(patch_pos=input_dict["patch_pos"]) if "patch_pos" in input_dict else None
            forward_dict.update(select_mask=input_dict["select_mask"]) if "select_mask" in input_dict else None

            generate_ids = model_engine.generate(**forward_dict, 
                                    max_new_tokens=128, 
                                    eos_token_id=processor.tokenizer.eos_token_id)

            generate_ids = generate_ids[:, input_dict['input_ids'].shape[1]:]
            generated_texts = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            print("generated_texts: \n", generated_texts)
            meta = item[0]
            # print("meta: \n", meta)
            outputs = {"split": meta['split'],
                "anno_id": meta['anno_id'], "img_path": meta['img_url_abs'], "instruction": meta['task'], "sentence": generated_texts,
                "Op_match": False, "Ele_match": False, "Op_F1": [0, meta['answer']["action"]],
                "meta": meta}

            generated_texts_unique.extend(generated_texts)
            answers_unique.append(meta['answer'])
            outputs_unique.append(outputs)

    answers_unique = gather_object(answers_unique)
    generated_texts_unique = gather_object(generated_texts_unique)
    outputs_unique = gather_object(outputs_unique)

    if global_rank == 0:
        # align the settings with SeeClick
        action2id = {'CLICK': 4, 'SELECT': 2, 'TYPE': 3}

        results = {}
        for pred_i, ans_i, output_i in tqdm(zip(generated_texts_unique, answers_unique, outputs_unique)):
            split_i = output_i['split']
            if split_i not in results:
                results[split_i] = {}

            anno_id = output_i['anno_id']
            if anno_id not in results[split_i]:
                results[split_i][anno_id] = []
            
            step_result = output_i.copy()
            try:
                # action_pred = ast.literal_eval(pred_i)
                action_pred = pred2json(pred_i, version=args.version)
                answer = ans_i

                if action_pred["action"] == answer["action"]:
                    step_result["Op_match"] = True

                click_point = action_pred["position"]

                bbox_ref = get_bbox(output_i['meta'])
                if (bbox_ref[0] <= click_point[0] <= bbox_ref[2]) and (bbox_ref[1] <= click_point[1] <= bbox_ref[3]):
                    step_result["Ele_match"] = True
                print("action_pred: ", action_pred)
                print("answer: ", answer)
                print("click_point: ", click_point)
                print("bbox_ref: ", bbox_ref)
                action_pred_idx = action2id[action_pred["action"]]
                pred_str = str(action_pred_idx)
                if action_pred["action"] in ['TYPE', 'SELECT']:
                    pred_str += ' '
                    pred_str += action_pred["value"].lower()
                    
                action_ref_idx = action2id[answer["action"]]
                ref_str = str(action_ref_idx)
                if answer["action"] in ['TYPE', 'SELECT']:
                    ref_str += ' '
                    ref_str += answer["value"].lower()
                    
                op_f1 = calculate_f1(pred_str, ref_str)
                print("Compute op_f1:", pred_str, ref_str, op_f1)
                step_result["Op_F1"][0] = op_f1

            except Exception as e:
                print(e)
                print(f"format wrong with {anno_id}'s prediction: {pred_i}")

            results[split_i][anno_id].append(step_result)
            
        
        if not args.debug and wandb.run is None:
            wandb.init(project="AgentNetEval", name="Mind2Web", config=vars(args))
        
        eval_dict = {}
        for split in results.keys():
            logging.info("==="*10)
            logging.info(f"{split}")
            logging.info("==="*10)
            eval_dict[split] = calculate_mind2web_metrics(results[split])

        if not args.debug:
            for split in eval_dict.keys():
                for key, value in eval_dict[split].items():
                    if isinstance(value, list):
                        continue
                    # writer.add_scalar(f"metrics/mind2web/{split}/{key}", value, epoch)
                    wandb.log({f"metrics/mind2web/{split}/{key}": value}, step=global_step)

        metric = sum([x["Macro Step Success Rate"] for x in eval_dict.values()]) / len(eval_dict)
        if not args.debug:
            # writer.add_scalar("metrics/mind2web/Avg Macro Step Success Rate", metric, epoch)
            wandb.log({"metrics/mind2web/Avg Macro Step Success Rate": metric}, step=global_step)

        save_json(results, os.path.join(args.tmp_dir, f'mind2web_epo{epoch}_tmp_dict.json'))
        save_json(eval_dict, os.path.join(args.tmp_dir, f'mind2web_epo{epoch}_res_dict.json'))

    
    if world_size > 1:
        metric = broadcast_value(metric, src=0, local_rank=local_rank)
    return metric