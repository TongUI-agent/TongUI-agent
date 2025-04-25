import json
import os
import sys

from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

sys.path.append(".")
from tongui.data.template.shared_grounding import grounding_to_qwen

with open("evaluation_data/ScreenSpot/metadata/hf_test_full.json", "r") as f:
    data = json.load(f)

model_id = "/mnt/bofeidisk2/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, device_map="cuda")
processor = AutoProcessor.from_pretrained(model_id)

# Define LoRA configuration
peft_model_id = "saves/qwen2_5vl-3b/lora/sft"
# Apply LoRA config to the model
model = PeftModel.from_pretrained(model, peft_model_id)
model.merge_and_unload()
model.eval()
predictions = []
for i in tqdm(range(len(data))):
    task = data[i]['task']
    image_url = data[i]['img_url']
    image_root = "evaluation_data/ScreenSpot/images"
    image_path = os.path.join(image_root, image_url)
    image = Image.open(image_path)
    image_dict = {
        'type': 'image', 
        'min_pixels': 3136, 
        'max_pixels': 1003520,
        "image_url": image_path
    }

    source = grounding_to_qwen(task, image_dict)
    # Initialize base model and processor
    # generate
    messages = source
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda")

    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    grounding_result = json.loads(output_text)
    item = data[i]
    item["predictions"] = grounding_result
    predictions.append(item)
    if len(predictions) % 100 == 0:
        os.makedirs("evaluations/ScreenSpot", exist_ok=True)
        with open("evaluations/ScreenSpot/hf_test_full_predictions.json", "w") as f:
            json.dump(predictions, f, indent=4, ensure_ascii=False)
        
os.makedirs("evaluations/ScreenSpot", exist_ok=True)
with open("evaluations/ScreenSpot/hf_test_full_predictions.json", "w") as f:
    json.dump(predictions, f, indent=4, ensure_ascii=False)