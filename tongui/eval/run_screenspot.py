import os
import sys
import torch
import logging
import argparse
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM
# from tensorboardX import SummaryWriter
from peft import PeftModel

sys.path.append('.')
from tongui.data.dset_screenspot import ScreenSpotDataset
from tongui.eval.eval_screenspot_utils import validate_screenspot
# from src.utils.utils import set_seed

logging.basicConfig(level=logging.INFO)

'''
CUDA_VISIBLE_DEVICES=0 python src/eval/run_screenspot.py \
    --model_path /mnt/buffer/zhangbofei/Qwen2.5-VL-3B-Instruct/ \
    --dataset_dir evaluation_data \
    --lora_path saves/sft_0219

'''
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on ScreenSpot dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to the LoRA weights")
    parser.add_argument("--dataset_dir", type=str, default="evaluation_data", help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save results")
    parser.add_argument("--tmp_dir", type=str, default="tmp", help="Directory for temporary files")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--xy_int", action="store_true", help="Use integer coordinates")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()

def main():
    args = parse_args()
    #set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Initialize tensorboard writer
    # writer = SummaryWriter(args.output_dir)
    
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
    dataset = ScreenSpotDataset(
        args.dataset_dir,
        "ScreenSpot",
        "hf_test_full",
        processor,
        inference=True,
        args_dict={'xy_int': args.xy_int}
    )
    # if args.debug:
    #     dataset = dataset[:100]
        
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    print("batch size", args.batch_size)
    assert args.batch_size == 1
    # exit()
    # Prepare model and dataloader
    model, dataloader = accelerator.prepare(model, dataloader)
    
    # Run evaluation
    results = validate_screenspot(
        dataloader,
        model,
        processor,
        epoch=0,
        global_step=0,
        writer=None,
        args=args,
        media=True
    )
    

if __name__ == "__main__":
    main()
