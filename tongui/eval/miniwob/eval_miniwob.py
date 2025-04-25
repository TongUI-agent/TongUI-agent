# evaluation in MiniWob environment
# Note1: the click position of MiniWoBCoordClick is the offset from body element, which is related to the
# window size of chrome (the window size could be checked in get_screenshot function in env packages).
# Note2: server without Graphical User Interface need to evaluate with the headless mode.
# Note3: if a lot of html code appears and gets stuck, try to disable the proxy.

import argparse
import ast
import json
import logging
import os
import random
import sys

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

sys.path.append("./")
from synapse.envs.miniwob.action import (
    MiniWoBCoordClick,
    MiniWoBType,
)
from synapse.envs.miniwob.environment import MiniWoBEnv

from tongui.data.data_utils import dict_to_cuda
from tongui.data.template import miniwob_to_qwen
from tongui.eval.miniwob.utils import parse_actions
from tongui.model.load import load_model_and_processor

logging.basicConfig(level=logging.INFO)
'''
python src/eval/miniwob/eval_miniwob.py --model_path Qwen/Qwen2.5-VL-3B-Instruct --lora_path checkpoints/sft_0301 --imgs_dir_temp /mnt/bofeidisk2/MiniWobEval/ --num_episodes 50 --env_name all
'''

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--lora_path', type=str, required=True)
parser.add_argument('--imgs_dir_temp', type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--env_name", type=str, default="all")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--version", type=str, default="v1", choices=["v1", "v2"])
parser.add_argument("--result-checkpoint", type=str, default="tmp/miniwob_result.json")
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

model_path = args.model_path
lora_path = args.lora_path
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, bf16=True).eval() # load with model checkpoint
processor, model = load_model_and_processor(model_path, lora_path=lora_path)
selected_envs = [
    "click-link",
    "click-button-sequence",
    "click-pie",
    "click-tab-2",
    "click-button",
    "click-collapsible-2",
    "click-shape",
    "unicode-test",
    "click-checkboxes",
    "click-option",
    "tic-tac-toe",
    "click-tab-2-hard",
    "click-widget",
    "navigate-tree",
    "click-test-2",
    "click-checkboxes-soft",
    "click-dialog-2",
    "text-transform",
    "click-checkboxes-transfer",
    "count-shape",
    "use-autocomplete",
    "click-color",
    "click-dialog",
    # "choose-date",
    # "click-checkboxes-large",
    # "click-shades",
    "click-collapsible",
    "click-tab",
    "click-test",
    "enter-date",
    "use-slider",
    "simple-algebra",
    "simple-arithmetic",
    "identify-shape",
    "grid-coordinate",
]

difficult_tasks = [
    "click-checkboxes-large",
    "click-shades",
    "choose-date", 
]

# uncomment the following line to evaluate difficult tasks
selected_envs = selected_envs + difficult_tasks

if os.path.exists(args.result_checkpoint):
    with open(args.result_checkpoint, "r") as f:
        result = json.load(f)
else:
    result = {}
miniwob_imgs_dir_temp = args.imgs_dir_temp
if not os.path.exists(miniwob_imgs_dir_temp):
    os.makedirs(miniwob_imgs_dir_temp)
else:
    import shutil
    shutil.rmtree(miniwob_imgs_dir_temp)
    os.makedirs(miniwob_imgs_dir_temp)
miniwob_train = json.load(open('training_data/MiniWob/miniwob_data_train.json', 'r'))     # load tasks from train set
miniwob_tasks = list(miniwob_train.keys())
# filter out by selected
miniwob_tasks = [task for task in miniwob_tasks if task in selected_envs]
print("miniwob_tasks", miniwob_tasks)
if args.env_name != "all" and args.env_name not in miniwob_tasks:
    miniwob_tasks.append(args.env_name)
task_max_step = {k: 15 for k in miniwob_tasks}
# result = {}
for env in tqdm(miniwob_tasks):
    print("env", env)
    if args.env_name != "all":
        if env != args.env_name:
            continue
    if env in result:
        print("Task: " + env + "  Score: " + str(result[env]))
        continue

    success = 0
    print("Task: " + env)
    for j in tqdm(range(args.num_episodes)):
        traj = []
        # initial MiniWob environment
        seed_task = random.randint(0, 1000000)
        print("seed_task", seed_task, "init env", env)
        miniwob_env = MiniWoBEnv(subdomain=env, headless=args.headless)
        miniwob_env.reset(seed=seed_task, record_screenshots=True)
        print("env reset")
        img_dir = miniwob_imgs_dir_temp

        reward = 0
        action_history = []
        parsed_action_history = []
        for k in range(task_max_step[env]):
            print("step", k)
            # get the current state
            miniwob_state = miniwob_env.instance.get_state()
            state_screenshot = miniwob_state.screenshot
            img_path = os.path.join(img_dir, env + '-' + str(seed_task) + '-' + str(k) + '.jpg')
            # state_screenshot.save(img_path)
            state_screenshot.resize((160*4, 210*4)).save(img_path)
            state_img_size = state_screenshot.size
            goal = miniwob_state.utterance
            print("Goal:\n", goal)
            if env == "use-autocomplete":
                general_hint = "\nYou should first click the input box and then type the answer."
                goal = goal + general_hint
            elif env == "simple-arithmetic":
                goal = goal + "\nYou should first click the input box and then type the answer."
            elif env == "simple-algebra":
                goal = goal + "\nYou should first click the input box and then type the answer."
            elif env == "use-slider":
                if k > 6:
                    goal = goal + "\nShould use '\ue013' to increase the value."
            image = Image.open(img_path).convert("RGB")
            # image = image.resize((160*4, 210*4))
            action_history.append({"type": "image", "image": image, 
                             "min_pixels": processor.image_processor.min_pixels,
                             "max_pixels": processor.image_processor.max_pixels})
            # print("Action history:\n", action_history)
            if len(action_history) > 6:
                action_history = action_history[2:]
                # print("Memory overflowed!, skip first 2")
            source = miniwob_to_qwen(goal, action_history, None, args.version)
            # print("Prompt:\n")
            # for item in source:
            #     print(item)
            prompt = processor.apply_chat_template(source, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(source)
            # print("prompt", prompt)
            inputs = processor(
                text=[prompt],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt"
            )
            inputs = dict_to_cuda(inputs, device="cuda")
            with torch.no_grad():
                eval_strategy = {
                    "default": "greedy",
                    "click-pie": "sampling",
                    "use-autocomplete": "sampling",
                }
                
                strategy = eval_strategy.get(env, "sampling")
                torch.cuda.empty_cache()
                if strategy == "sampling":
                    num_return_sequences = 3
                    
                    generate_ids = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=True,
                        top_p=0.9,
                        temperature=2.0,
                        num_return_sequences=num_return_sequences,
                        eos_token_id=processor.tokenizer.eos_token_id
                    )
                    # print("generate_ids", generate_ids.shape)
                    selected = random.randint(0, num_return_sequences - 1)
                    # generate_ids = generate_ids[selected, :].unsqueeze(0)
                    # print("generate_ids", generate_ids.shape)
                    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
                else:
                    generate_ids = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        eos_token_id=processor.tokenizer.eos_token_id
                    )
                    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
                    selected = 0
                
            generated_texts = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            # print("Response:\n", generated_texts)
            print("selected", generated_texts[selected])
            generated_texts = generated_texts[selected]
            action_step_record = {"img_path": img_path, "sentence": generated_texts, "success": False}
            traj.append(action_step_record)
            action_history.append({
                "type": "text",
                "text": generated_texts
            })
            if args.version == "v1":
                try:
                    action_pred = ast.literal_eval(generated_texts)
                except ValueError:
                    try:
                        action_pred = json.loads(generated_texts)
                    except Exception as e:
                        print("Back up Error: ", e)
                        raise e
            elif args.version == "v2":
                generated_texts = generated_texts.split("Action:")[1].strip()
                action_pred = json.loads(generated_texts)
#                 if k > 10:
#                     last_action = parsed_action_history[-1]
#                     if last_action["action"] == action_pred["action"]:
#                         position = action_pred["position"]
#                         value = action_pred["value"]
#                         if position is not None:
#                             '''Thought: Type "42" to set the slider value accurately before submitting.
# Action: {"action": "TYPE", "value": "", "position": null}'''
#                             if position[0] == last_action["position"][0] and position[1] == last_action["position"][1]:
#                                 # add random noise to the position
#                                 action_pred["position"] = (position[0] + random.uniform(-0.5, 0.5), position[1] + random.uniform(-0.5, 0.5))
#                                 # scale to 0-1
#                                 action_pred["position"] = [min(1, max(0, action_pred["position"][0])), min(1, max(0, action_pred["position"][1]))]
#                         if value is not None:
#                             if value == last_action["value"]:
#                                 action_pred["value"] = value + random.uniform(-0.5, 0.5)
                parsed_action_history.append(action_pred)
            # convert the predicted action to miniwob action that operate the chrome
            try:
                action_pred = parse_actions(action_pred)
                width, height = state_img_size
                print("Screenshot size", width, height)
                if action_pred["action_type"] == 4:
                    # the offset (150, 105) here is depended on the window size of chrome
                    click_x = action_pred['click_point'][0] * 160
                    click_y = action_pred['click_point'][1] * 210
                    miniwob_action = MiniWoBCoordClick(click_x - 150, click_y - 105)
                elif action_pred["action_type"] == 3:
                    typed_text = action_pred['typed_text']
                    miniwob_action = MiniWoBType(typed_text)
                else:
                    print("action undefined!!!!", action_pred)
                    continue
                # execute the action and
                _, reward, done, _ = miniwob_env.step(miniwob_action)
                print("action", miniwob_action, "reward", reward, "done", done)
            except Exception:
                print("Trajectory failed", generated_texts)
                continue
            # determine if the episode is over, success (reward > 0.8) or fail (done)
            if reward > 0.8:
                success += 1
                for item in traj:
                    item["success"] = True
                break

            if done:
                break

        miniwob_env.close()

    result[env] = success / args.num_episodes
    print("Task: " + env + "  Score: " + str(success / args.num_episodes))
    with open(args.result_checkpoint, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

import wandb

wandb.init(project="miniwob", name=args.lora_path)
print(result)
print("Average Score: " + str(np.mean(list(result.values()))))

wandb.log({"score": np.mean(list(result.values()))})
with open(args.result_checkpoint, "w") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

# compute top 35 scores average
scores = result
top_35_scores = sorted(scores.values(), reverse=True)[:35]
print(sum(top_35_scores) / len(top_35_scores))
# compute selected scores average

selected_scores = [scores[task] for task in selected_envs]
print(sum(selected_scores) / len(selected_scores))
for task in selected_envs:
    print(task, scores[task])
wandb.log({"selected_score": sum(selected_scores) / len(selected_scores)})