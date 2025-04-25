import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import wandb
from accelerate.utils import gather_object
from tqdm import tqdm

from tongui.data.data_utils import dict_to_cuda
from tongui.eval.aitw_utils import (
    action2json,
    check_actions_match,
    is_tap_action,
    pred2json_post,
)
from tongui.utils.utils import save_json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# to avoid error;
# NUM_HISTORY = 4

import logging

logging.basicConfig(level=logging.INFO)

def broadcast_value(value, src=0, local_rank=0):
    tensor = torch.tensor([value], dtype=torch.float32).to(f'cuda:{local_rank}')
    dist.broadcast(tensor, src=src)
    return tensor.item()

def calculate_aitw_metrics(results):
    corr_action=0
    corr_type=0
    num_text=0
    corr_text=0
    num_scroll=0
    corr_scroll=0
    num_click=0
    corr_click=0
    num_both_click=0
    corr_both_click=0
    num_wrong_format=0
    num=0
    for episode in results.keys():
        for step in results[episode]:
            corr_action += step['corr_action']
            corr_type += step['corr_type']
            num_text += step['num_text']
            corr_text += step['corr_text']
            num_scroll += step['num_scroll']
            corr_scroll += step['corr_scroll']
            num_click += step['num_click']
            corr_click += step['corr_click']
            num_both_click += step['num_both_click']
            corr_both_click += step['corr_both_click']
            num_wrong_format += step['num_wrong_format']
            num += 1
    
    logging.info("[Score]: " + str(corr_action/num))
    logging.info("[Valid]: " + str(num_wrong_format/num))
    metrics = {
        "Score": corr_action / num,
        "Num Corr Action": corr_action,
        "Num Corr Type": corr_type,
        
        "Num Text": num_text,
        "Num Corr Text": corr_text,
        
        "Num Scroll": num_scroll,
        "Num Corr Scroll": corr_scroll,

        "Num Click": num_click,
        "Num Corr Click": corr_click,

        "Num Both Click": num_both_click,
        "Num Corr Both Click": corr_both_click,

        "Num Wrong Format": num_wrong_format,
        "Num": num,
    }
    return metrics

@torch.no_grad()
def validate_aitw(val_loader, model_engine, processor, epoch, global_step, writer, args):
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
        if args.debug and i > 10:
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
            # for k, v in forward_dict.items():
            #     print(k, v.shape)
            # exit()
            generate_ids = model_engine.generate(**forward_dict, 
                                    max_new_tokens=128, 
                                    eos_token_id=processor.tokenizer.eos_token_id)

            generate_ids = generate_ids[:, input_dict['input_ids'].shape[1]:]
            generated_texts = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            print("Generated texts: ", generated_texts)
            assert type(item) == list and len(item) == 1
            meta = item[0]
            # print("meta: ", meta)
            outputs = {"domain": meta['domain'],
                "anno_id": meta['anno_id'],
                "ep_id": meta['ep_id'], "img_path": meta['img_url'], "instruction": meta['task'], "sentence": generated_texts,
                "meta": meta, "annot_position": meta['annot_position']
                }
            outputs.update(dict(
                corr_action=0,
                corr_type=0,
                num_text=0,
                corr_text=0,
                num_scroll=0,
                corr_scroll=0,
                num_click=0,
                corr_click=0,
                num_both_click=0,
                corr_both_click=0,
                num_wrong_format=0,
                # num=0
            ))

            generated_texts_unique.extend(generated_texts)
            answers_unique.append(meta['answer'])
            outputs_unique.append(outputs)

    answers_unique = gather_object(answers_unique)
    generated_texts_unique = gather_object(generated_texts_unique)
    outputs_unique = gather_object(outputs_unique)

    if global_rank == 0:
        # align the settings with SeeClick
        results = {}
        for pred_i, ans_i, output_i in tqdm(zip(generated_texts_unique, answers_unique, outputs_unique)):
            domain_i = output_i['domain']
            if domain_i not in results:
                results[domain_i] = {}

            ep_id = output_i['ep_id']
            if ep_id not in results[domain_i]:
                results[domain_i][ep_id] = []
            
            # step_result = output_i.copy()
            try:
                NUM_HISTORY = args.num_history
                anno_id = output_i['anno_id']
                # print("ast.literal_eval", pred_i, type(pred_i))
                # pred_i = ast.literal_eval(pred_i)
                if args.version == "v1":
                    pred_i = json.loads(pred_i)
                elif args.version == "v2":
                    pred_i = pred_i.split("Action:")[1].strip()
                    pred_i = json.loads(pred_i)
                else:
                    raise ValueError(f"Invalid version: {args.version}")
                # print("pred2json_post", pred_i)
                action_pred = pred2json_post(pred_i)
                if action_pred is None:
                    output_i['num_wrong_format'] += 1
                    continue
                step_i = output_i['meta']['step']
                # print("action2json", step_i)
                action_ref = action2json(step_i)
                # print("annotaion_positions", output_i["annot_position"], len(output_i["annot_position"]))
                NUM_HISTORY = 4
                annot_position = np.array([output_i["annot_position"][i:i + NUM_HISTORY]    \
                                            for i in range(0, len(output_i["annot_position"]), NUM_HISTORY)])
                # print("annot_position", annot_position, len(annot_position))
                #exit()
                check_match = check_actions_match(action_pred["touch_point"], 
                action_pred["lift_point"],
                action_pred["action_type"], 
                action_ref["touch_point"],
                action_ref["lift_point"],                           action_ref["action_type"],
                annot_position)
                print("check_match", check_match)
                # step accuracy
                if check_match == True:
                    output_i['corr_action'] += 1
                    match_label = 1
                    # logging.info("Step: " + str(j) + " right")
                else:
                    match_label = 0
                    # logging.info("Step: " + str(j) + " wrong")

                # type accuracy
                if action_pred["action_type"] == action_ref["action_type"]:
                    output_i['corr_type'] += 1

                # text accuracy
                if action_ref["action_type"] == 3:
                    output_i['num_text'] += 1
                    if (action_pred["typed_text"] == action_ref["typed_text"]) or (
                            action_pred["typed_text"] in action_ref["typed_text"]) or (
                            action_ref["typed_text"] in action_pred["typed_text"]):
                        output_i['corr_text'] += 1

                if action_ref["action_type"] == 4:
                    # click accuracy
                    if is_tap_action(action_ref["touch_point"], action_ref["lift_point"]):
                        output_i['num_click'] += 1
                        if match_label:
                            output_i['corr_click'] += 1
                    # scroll accuracy
                    else:
                        output_i['num_scroll'] += 1
                        if match_label:
                            output_i['corr_scroll'] += 1
                    if (action_pred["action_type"] == 4) and is_tap_action(action_ref["touch_point"],
                                                                            action_ref[
                                                                                "lift_point"]) and is_tap_action(
                            action_pred["touch_point"], action_pred["lift_point"]):
                        output_i['num_both_click'] += 1
                        if match_label:
                            output_i['corr_both_click'] += 1

            except Exception as e:
                print("Format Action met error: ", e)
                output_i['num_wrong_format'] += 1
                print(f"format wrong with {anno_id}'s prediction: {pred_i}")

            results[domain_i][ep_id].append(output_i)

        eval_dict = {}
        for domain in results.keys():
            logging.info("==="*10)
            logging.info(f"Domain: {domain}")
            logging.info("==="*10)
            eval_dict[domain] = calculate_aitw_metrics(results[domain])
        if wandb.run is None:
            wandb.init(project="AgentNetEval", name="AITW", config=vars(args))
        
        if not args.debug:
            for domain in eval_dict.keys():
                for key, value in eval_dict[domain].items():
                    # if isinstance(value, list):
                    if key not in ['Score']:
                        continue
                    # writer.add_scalar(f"metrics/aitw/{domain}/{key}", value, epoch)
                    wandb.log({f"metrics/aitw/{domain}/{key}": value}, step=global_step)

        metric = sum([x["Score"] for x in eval_dict.values()]) / len(eval_dict)
        logging.info("==="*10)
        logging.info(f"[Avg Score]: {metric}")
        logging.info("==="*10)
        if not args.debug:
            # writer.add_scalar("metrics/aitw/Avg Score", metric, epoch)
            wandb.log({"metrics/aitw/Avg Score": metric}, step=global_step)

        save_json(results, os.path.join(args.tmp_dir, f'aitw_epo{epoch}_tmp_dict.json'))
        save_json(eval_dict, os.path.join(args.tmp_dir, f'aitw_epo{epoch}_res_dict.json'))

    if world_size > 1:
        metric = broadcast_value(metric, src=0, local_rank=local_rank)
    return metric