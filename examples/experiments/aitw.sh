CUDA_VISIBLE_DEVICES=0 python tongui/eval/run_aitw.py \
    --model_path saves/qwen2_5vl-3b/Qwen2.5-VL-3B-Instruct_sft_0316/ \
    --dataset_dir evaluation_data \
    --lora_path saves/qwen2_5vl-3b/lora/sft_aitw/checkpoint-200 \
    --version v2