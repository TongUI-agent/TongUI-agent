#!/bin/bash

# Print GPU information if nvidia-smi is available
echo
echo "=== GPU Information ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv,noheader
else
    echo "nvidia-smi not found - No GPU information available"
fi

# Print Python environment
echo
echo "=== Python Environment ==="
if command -v python3 &> /dev/null; then
    echo "Python Path: $(which python3)"
    echo "Python Version: $(python3 --version)"
else
    echo "Python3 not found"
fi


FORCE_TORCHRUN=1 NNODES=$nnodes NODE_RANK=$node_rank MASTER_ADDR=$master_addr MASTER_PORT=$master_port llamafactory-cli train configs/training/sft_tiny_32b.yaml

