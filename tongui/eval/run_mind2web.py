import os
import sys
import torch
import logging
import argparse
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from peft import PeftModel
from torch.utils.data._utils.collate import default_collate

sys.path.append('.')
from tongui.data.dset_mind2web import Mind2WebDataset
from tongui.eval.eval_mind2web_utils import validate_mind2web

logging.basicConfig(level=logging.INFO)

'''
python src/eval/run_mind2web.py \
    --model_path /mnt/buffer/zhangbofei/Qwen2.5-VL-3B-Instruct/ \
    --dataset_dir evaluation_data \
    --lora_path saves/qwen2_5vl-3b/lora/sft
'''
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on Mind2Web dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to the LoRA weights")
    parser.add_argument("--dataset_dir", type=str, default="evaluation_data", help="Path to dataset directory")
    parser.add_argument("--dataset_name", type=str, default="task", choices=["task", "domain", "website"], help="Name of the dataset")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save results")
    parser.add_argument("--tmp_dir", type=str, default="tmp", help="Directory for temporary files")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--num_history", type=int, default=2, help="Number of history steps")
    parser.add_argument("--version", type=str, default="v1", choices=["v1", "v2"], help="Version of the dataset")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()

def collate_fn(batch):
    """Custom collate function to handle None values and tuple returns from dataset"""
    # Filter out None values
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    # If the batch contains tuples (data_dict, meta), handle them separately
    if isinstance(batch[0], tuple):
        data_dicts, metas = zip(*batch)
        # Collate data_dicts (tensors)
        collated_data = default_collate(data_dicts)
        # Keep metas as a list
        return collated_data, list(metas)
    
    # Regular collation for non-tuple batch items
    return default_collate(batch)

def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Initialize processor and model
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=256*28*28,
        max_pixels=1344*28*28,
        model_max_length=8196,
    )
    
    from transformers import Qwen2_5_VLForConditionalGeneration
    # Initialize base model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16 if args.precision == "fp16" else torch.bfloat16 if args.precision == "bf16" else torch.float32,
        attn_implementation="flash_attention_2",
    )
    
    # Load and merge LoRA weights
    peft_model_id = args.lora_path
    if peft_model_id is not None and len(peft_model_id) > 0:
        model = PeftModel.from_pretrained(model, peft_model_id)
        model = model.merge_and_unload()
    model.eval()
    
    # Initialize dataset and dataloader
    dataset_name = f"hf_test_{args.dataset_name}"
    if args.version == "v2":
        dataset_name = f"hf_test_{args.dataset_name}_with_thoughts"
    dataset = Mind2WebDataset(
        args.dataset_dir,
        "Mind2Web",
        dataset_name,
        processor,
        inference=True,
        args_dict={'num_history': args.num_history, 'interleaved_history': 'vtvt', 'version': args.version}
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_fn
    )
    print("batch size", args.batch_size)
    assert args.batch_size == 1
    
    # Prepare model and dataloader
    model, dataloader = accelerator.prepare(model, dataloader)
    
    # Run evaluation
    results = validate_mind2web(
        dataloader,
        model,
        processor,
        epoch=0,
        global_step=0,
        writer=None,
        args=args
    )

if __name__ == "__main__":
    main()
