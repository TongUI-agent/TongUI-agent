import base64
import json
from collections import defaultdict
import numpy as np
import logging

import openai
from transformers.models.auto.processing_auto import AutoProcessor
from tongui.eval.eval_mind2web_utils import get_bbox, calculate_f1

logging.basicConfig(level=logging.INFO)

def parse_source_to_payload(source):
    """
    Parse the source field into a payload for OpenAI client
    """
    messages = []
    for item in source:
        if item['role'] == 'user':
            for content in item['content']:
                if content['type'] == 'text':
                    messages.append({"type": "text", "text": content['text']})
                elif content['type'] == 'image':
                    # Read and encode the image
                    with open(content['image'], "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                    messages.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    })
    return messages

def parse_model_response(response_text):
    """
    Parse the model's response into thought and action
    """
    try:
        # Split the response into thought and action parts
        parts = response_text.split('\nAction:')
        if len(parts) != 2:
            return None, None
            
        thought = parts[0].replace('Thought:', '').strip()
        action_str = parts[1].strip()
        
        # Parse the action JSON
        action = json.loads(action_str)
        return thought, action
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None, None

def predict_action(client: openai.OpenAI, source, model: str = "tongui-3b"):
    """
    Make prediction using OpenAI client
    """
    messages = parse_source_to_payload(source)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": messages}
        ],
        max_completion_tokens=256,
        temperature=1e-6,
        top_p=0.95,
        extra_body={
            "top_k": 50
        }
    )
    
    prediction_text = response.choices[0].message.content
    print(f"Raw prediction: {prediction_text}; usage: {response.usage.total_tokens}")
    
    thought, action = parse_model_response(prediction_text)
    return thought, action

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

def main():
    # Configuration
    MODEL = "tongui-7b"
    ENDPOINT = "http://localhost:50005/v1"
    LIMIT = -1
    
    # Initialize OpenAI client
    client = openai.OpenAI(
        api_key="EMPTY",
        base_url=ENDPOINT,
    )
    
    # Initialize processor and dataset
    processor = AutoProcessor.from_pretrained("Bofeee5675/TongUI-3B")
    dataset_name = "task"
    dataset_dir = "evaluation_data"
    version = "v2"
    dataset_name = f"hf_test_{dataset_name}_with_thoughts"
    
    from tongui.data.dset_mind2web import Mind2WebDataset
    dataset = Mind2WebDataset(
        dataset_dir,
        "Mind2Web",
        dataset_name,
        processor,
        inference=True,
        args_dict={'num_history': 2, 'interleaved_history': 'vtvt', 'version': version}
    )
    
    # Track results
    results = defaultdict(lambda: defaultdict(list))
    
    # Process each sample
    for idx, item in enumerate(dataset):
        if LIMIT > 0 and idx >= LIMIT:
            break
            
        data_dict, item = item
        print(f"\nProcessing sample {idx+1}")
        
        # Get source and make prediction
        source = item['source']
        thought, action = predict_action(client, source, model=MODEL)
        
        if action:
            # Compare with ground truth
            gt_action = item['answer']
            op_match = action['action'] == gt_action['action']
            
            # Calculate element match
            bbox_ref = get_bbox(item)
            click_point = action['position']
            ele_match = (bbox_ref[0] <= click_point[0] <= bbox_ref[2]) and (bbox_ref[1] <= click_point[1] <= bbox_ref[3])
            
            # Calculate operation F1
            action2id = {'CLICK': 4, 'SELECT': 2, 'TYPE': 3}
            action_pred_idx = action2id[action['action']]
            pred_str = str(action_pred_idx)
            if action['action'] in ['TYPE', 'SELECT']:
                pred_str += ' ' + action['value'].lower()
                
            action_ref_idx = action2id[gt_action['action']]
            ref_str = str(action_ref_idx)
            if gt_action['action'] in ['TYPE', 'SELECT']:
                ref_str += ' ' + gt_action['value'].lower()
                
            op_f1 = calculate_f1(pred_str, ref_str)
            
            # Store results
            step_result = {
                "Op_match": op_match,
                "Ele_match": ele_match,
                "Op_F1": [op_f1, gt_action['action']],
                "meta": item
            }
            
            split = item.get('split', 'unknown')
            anno_id = item.get('anno_id', str(idx))
            results[split][anno_id].append(step_result)
            
            print(f"Predicted: {action}")
            print(f"Ground truth: {gt_action}")
            print(f"Op match: {op_match}, Ele match: {ele_match}, Op F1: {op_f1}")
    
    # Calculate and print metrics
    print("\n===== EVALUATION METRICS =====")
    for split in results.keys():
        print(f"\n{split}")
        print("="*30)
        metrics = calculate_mind2web_metrics(results[split])
        for metric_name, value in metrics.items():
            if isinstance(value, list):
                print(f"{metric_name}: {value}")
            else:
                print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()
