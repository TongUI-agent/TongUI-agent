import torch
import json
from PIL import Image, ImageDraw
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from tongui.utils import draw_point_on_image

def load_model_and_processor(model_path):
    """
    Load the Qwen2.5-VL model and processor with optional LoRA weights.

    Args:
        args: Arguments containing:
            - model_path: Path to the base model
            - precision: Model precision ("fp16", "bf16", or "fp32")
            - lora_path: Path to LoRA weights (optional)
            - merge_lora: Boolean indicating whether to merge LoRA weights

    Returns:
        tuple: (processor, model) - The initialized processor and model
    """
    # Initialize processor
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=256 * 28 * 28,
            max_pixels=1344 * 28 * 28,
            model_max_length=8196,
        )
    except Exception as e:
        print(f"Error loading processor: {e}")
        processor = None
        config = AutoConfig.from_pretrained(model_path)
        print(config)
        raise e
    # Initialize base model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    model.eval()

    return processor, model


def main():
    processor, model = load_model_and_processor(
        model_path="Bofeee5675/TongUI-3B",
    )
    image_path = "assets/safari_google.png"
    _SYSTEM = "Based on the screenshot of the page, I give a text description and you give its corresponding location. The coordinate represents a clickable location [x, y] for an element, which is a relative coordinate on the screenshot, scaled from 0 to 1."
    MIN_PIXELS = 256 * 28 * 28
    MAX_PIXELS = 1344 * 28 * 28
    img_dict = {
        "type": "image",
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "image": f"file://{image_path}",
    }
    user_content = [
        {"type": "text", "text": _SYSTEM},
        img_dict,
        {"type": "text", "text": "search box"},
    ]
    prompt = [{"role": "user", "content": user_content}]
    inputs = processor.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    data_dict = processor(
        text=inputs,
        images=[Image.open(image_path).convert("RGB")],
        return_tensors="pt",
        training=False,
    )
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = v.to(model.device)
    generate_ids = model.generate(
        **data_dict,
        max_new_tokens=128,
        eos_token_id=processor.tokenizer.eos_token_id,
        do_sample=False,
    )
    
    generate_ids = generate_ids[:, data_dict['input_ids'].shape[1]:]
    generated_texts = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    print(generated_texts)
    points = json.loads(generated_texts)
    draw_point_on_image(image_path, points)
if __name__ == "__main__":
    main()
