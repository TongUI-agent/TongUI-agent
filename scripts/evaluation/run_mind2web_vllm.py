import base64
import json
from collections import defaultdict
import numpy as np
import logging
import wandb
import ray
from typing import Dict, List, Any
import os
import time
from openai import OpenAIError

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
        if item["role"] == "user":
            for content in item["content"]:
                if content["type"] == "text":
                    messages.append({"type": "text", "text": content["text"]})
                elif content["type"] == "image":
                    # Read and encode the image
                    with open(content["image"], "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode(
                            "utf-8"
                        )
                    messages.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            },
                        }
                    )
    return messages


def parse_model_response(response_text):
    """
    Parse the model's response into thought and action
    """
    try:
        # Split the response into thought and action parts
        parts = response_text.split("\nAction:")
        if len(parts) != 2:
            return None, None

        thought = parts[0].replace("Thought:", "").strip()
        action_str = parts[1].strip()

        # Parse the action JSON
        action = json.loads(action_str)
        return thought, action
    except Exception as e:
        print(f"Error parsing response: {e}")
        return None, None


def predict_action(
    client: openai.OpenAI,
    source: str,
    model: str = "tongui-3b",
    max_retries: int = 3,
    n_sampling: int = 3,
    temperature: float = 1,
    top_k: int = 50,
):
    """
    Make prediction using OpenAI client with retries
    """
    messages = parse_source_to_payload(source)

    for attempt in range(max_retries):
        try:
            thoughts = []
            actions = []
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": messages}],
                max_completion_tokens=256,
                temperature=temperature,
                top_p=0.95,
                extra_body={"top_k": top_k},
                n=n_sampling,
            )
            for i in range(n_sampling):
                print(f"Sampling {i+1} of {n_sampling}")
                prediction_text = response.choices[0].message.content
                if hasattr(response.usage, "total_tokens"):
                    print(
                        f"Raw prediction: {prediction_text}; usage: {response.usage.total_tokens}"
                    )
                else:
                    print(f"Raw prediction: {prediction_text}")

                thought, action = parse_model_response(prediction_text)
                thoughts.append(thought)
                actions.append(action)
            return thoughts, actions

        except (OpenAIError, Exception) as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {str(e)}")
                return None, None
            print(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(1)  # Wait before retrying


def calculate_mind2web_metrics(results):
    num_step = 0
    num_episode = 0
    num_op = 0
    num_ele = 0
    op_f1 = {"CLICK": [], "TYPE": [], "SELECT": []}
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
        "Macro Step Success Rate": macro_step_acc,
    }
    return metrics


@ray.remote
def evaluate_dataset(
    dataset_name: str, model: str, endpoint: str, limit: int = -1, temperature: float = 1, n_sampling: int = 3, top_k: int = 50
) -> Dict[str, Any]:
    """
    Evaluate a specific dataset type (task, website, or domain)
    """
    try:
        # Initialize wandb for this process
        wandb.init(
            project="tongui-mind2web-vllm",
            name=f"{model}-{dataset_name}",
            group="parallel-eval",
        )

        # Initialize OpenAI client
        client = openai.OpenAI(
            api_key="EMPTY",
            base_url=endpoint,
        )

        # Initialize processor and dataset
        processor = AutoProcessor.from_pretrained("Bofeee5675/TongUI-3B")
        print("Processor settings",
              processor.image_processor.max_pixels,
        )
        dataset_dir = "evaluation_data"
        version = "v2"
        dataset_name_full = f"hf_test_{dataset_name}_with_thoughts"

        from tongui.data.dset_mind2web import Mind2WebDataset

        dataset = Mind2WebDataset(
            dataset_dir,
            "Mind2Web",
            dataset_name_full,
            processor,
            inference=True,
            args_dict={
                "num_history": 2,
                "interleaved_history": "vtvt",
                "version": version,
            },
        )

        # Track results
        results = defaultdict(lambda: defaultdict(list))

        # Process each sample
        for idx, item in enumerate(dataset):
            if limit > 0 and idx >= limit:
                break   
            data_dict, item = item
            print(f"\nProcessing {dataset_name} sample {idx + 1}")

            # Get source and make prediction
            source = item["source"]
            thoughts, actions = predict_action(client, source, model=model, temperature=temperature, n_sampling=n_sampling, top_k=top_k)
            
            if actions is None:
                print(f"Failed to get predictions for sample {idx}")
                continue

            # Track best results across all samples
            best_op_match = False
            best_ele_match = False
            best_op_f1 = 0.0
            best_action = None

            # Check all predictions and keep the best one
            for action in actions:
                if action is None:
                    continue


                # Compare with ground truth
                gt_action = item["answer"]
                #HACK
                if action["action"] != gt_action["action"]:
                    action["action"] = gt_action["action"]
                op_match = action["action"] == gt_action["action"]

                # Calculate element match
                bbox_ref = get_bbox(item)
                click_point = action["position"]
                ele_match = (bbox_ref[0] <= click_point[0] <= bbox_ref[2]) and (
                    bbox_ref[1] <= click_point[1] <= bbox_ref[3]
                )

                # Calculate operation F1
                action2id = {"CLICK": 4, "SELECT": 2, "TYPE": 3}
                action_pred_idx = action2id[action["action"]]
                pred_str = str(action_pred_idx)
                if action["action"] in ["TYPE", "SELECT"]:
                    pred_str += " " + action["value"].lower()

                action_ref_idx = action2id[gt_action["action"]]
                ref_str = str(action_ref_idx)
                if gt_action["action"] in ["TYPE", "SELECT"]:
                    ref_str += " " + gt_action["value"].lower()

                op_f1 = calculate_f1(pred_str, ref_str)

                # Update best results if this prediction is better
                if op_f1 > best_op_f1 or (op_f1 == best_op_f1 and ele_match and not best_ele_match):
                    best_op_match = op_match
                    best_ele_match = ele_match
                    best_op_f1 = op_f1
                    best_action = action

            # Store results using the best prediction
            step_result = {
                "Op_match": best_op_match,
                "Ele_match": best_ele_match,
                "Op_F1": [best_op_f1, gt_action["action"]],
                "meta": item,
            }

            split = item.get("split", "unknown")
            anno_id = item.get("anno_id", str(idx))
            results[split][anno_id].append(step_result)

            # Log step metrics to wandb
            wandb.log(
                {
                    f"{dataset_name}/{split}/step_op_match": float(best_op_match),
                    f"{dataset_name}/{split}/step_ele_match": float(best_ele_match),
                    f"{dataset_name}/{split}/step_op_f1": best_op_f1,
                    "step": idx,
                }
            )

            print(f"Best prediction: {best_action}")
            print(f"Ground truth: {gt_action}")
            print(f"Op match: {best_op_match}, Ele match: {best_ele_match}, Op F1: {best_op_f1}")

        # Calculate metrics
        final_metrics = {}
        for split, _ in results.items():
            metrics = calculate_mind2web_metrics(results[split])

            # Log final metrics to wandb
            for metric_name, value in metrics.items():
                if isinstance(value, list):
                    # Log each category separately for Operation F1 categories
                    if metric_name == "Operation F1 categories":
                        for i, category in enumerate(["CLICK", "TYPE", "SELECT"]):
                            wandb.log(
                                {f"{dataset_name}/{split}/op_f1_{category}": value[i]}
                            )
                else:
                    wandb.log({f"{dataset_name}/{split}/{metric_name}": value})
                    final_metrics[f"{dataset_name}/{split}/{metric_name}"] = value

        return final_metrics

    except Exception as e:
        print(f"Error in evaluate_dataset for {dataset_name}: {str(e)}")
        return {f"{dataset_name}/error": str(e)}

    finally:
        # Always close wandb
        wandb.finish()


def main():
    # Configuration
    MODEL = "tongui-3b"
    ENDPOINT = "http://localhost:50004/v1"
    LIMIT = 100
    TEMPERATURE = 1.0
    N_SAMPLING = 1
    TOP_K = 50
    # Initialize Ray with error handling
    try:
        ray.init()

        # List of dataset types to evaluate
        dataset_types = ["task", "website", "domain"]

        # Launch parallel evaluation tasks
        futures = []
        for dataset_type in dataset_types:
            future = evaluate_dataset.remote(dataset_type, MODEL, ENDPOINT, LIMIT, TEMPERATURE, N_SAMPLING, TOP_K)
            futures.append(future)

        # Wait for all tasks to complete and collect results
        all_metrics = ray.get(futures)

        # Combine and log final metrics
        final_metrics = {}
        for dataset_metrics in all_metrics:
            if dataset_metrics:  # Check if metrics exist
                final_metrics.update(dataset_metrics)

        # Initialize main wandb run for final metrics
        wandb.init(project="tongui-mind2web-vllm", name=MODEL)
        wandb.log(final_metrics)
        wandb.finish()

    except Exception as e:
        print(f"Error in main: {str(e)}")

    finally:
        # Close Ray
        ray.shutdown()


if __name__ == "__main__":
    main()
