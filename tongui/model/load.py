import torch
from peft.peft_model import PeftModel
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration


def load_model_and_processor(model_path, precision="bf16", lora_path=None, merge_lora=True):
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
            min_pixels=256*28*28,
            max_pixels=1344*28*28,
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
        device_map="auto",
        torch_dtype=torch.float16 if precision == "fp16" else torch.bfloat16 if precision == "bf16" else torch.float32,
        attn_implementation="flash_attention_2",
    )
    
    # Load LoRA weights if path is provided
    if lora_path is not None and len(lora_path) > 0:
        print(f"Loading LoRA weights from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        
        if merge_lora:
            print("Merging LoRA weights into base model")
            model = model.merge_and_unload()
    
    model.eval()
    
    return processor, model