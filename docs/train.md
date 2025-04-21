# 🔧 Training Pipeline
## Install environment
> [!IMPORTANT]
> Installation is mandatory.

We use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to train the model. You can use `uv` or `pip` to install the training dependencies.

```bash
uv sync --group=train
# or 
pip install -e ".[train]"
```
## Training

### Training with a tiny dataset.
> [!TIP]
> To help you understand how the dataset is organized, we provide a tiny subset of our entire dataset.

First, download the dataset from [HF datasets Bofeee5675/TongUI-Tiny](https://huggingface.co/datasets/Bofeee5675/TongUI-Tiny). We recommend you to download the dataset using `huggingface-cli`.
```bash
huggingface-cli download Bofeee5675/TongUI-Tiny --local-dir . --repo-type dataset
```

Unzip the `training_data.zip` into a folder `training_data`. Move the `tongui-tiny.json` into `data` folder. You will have a folder structure like this:
```bash
├── configs
│   └── training
│       └── sft_tiny.yaml
├── data
│   ├── dataset_info.json
│   └── tongui-tiny.json
├── training_data 
└── uv.lock
```

This dataset follows the format of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for training Vision-Language Model. You can check the JSON file to understand the format.

To start training, you can use the following command:
```bash
uv run llamafactory-cli train configs/training/sft_tiny.yaml
```
You can use a 24GB GPU to fine-tune `Qwen2.5-VL-3B-Instruct` with this tiny dataset. This gives you a good starting point to prepare for the full training or prepare for the own dataset.

### Training with the full dataset.

First download the dataset from [HF datasets Bofeee5675/TongUI-143K](https://huggingface.co/datasets/Bofeee5675/TongUI-143K) and [HF datasets Bofeee5675/TongUI-1M](https://huggingface.co/datasets/Bofeee5675/TongUI-1M). The difference between the two datasets is summarized in the following table:

| Dataset | # of training data | Description |
|---------|--------------------|----------------------|
| TongUI-143K | 143k + 240k(from previous work) | The version of the dataset on which our model trained upon deadline of paper submission. |
| TongUI-1M | ~1M(TongUI collected only) | The full version of the dataset. |

After you setup the folder structure mentioned in the [Training with a tiny dataset](#training-with-a-tiny-dataset) section, you can start training with the full dataset.

```bash
uv run llamafactory-cli train configs/training/sft_3b.yaml
```
The config file `configs/training/sft_3b.yaml` contains all necessary hyper-parameters to reproduce the results in the paper.

- Check out `data/dataset_info.json` to see the detailed configuration of the dataset.
    ```yaml
    datasets: gui_video,guiact_smartphone_thought,guiact_web_single_thought,guiact_web_multi_thought,showui-desktop-augmented,showui-web,amex-ele,amex-func,aitw_with_thoughts,miniwob_with_thoughts,mind2web_with_thoughts,wikihow_v2,baidu_jingyan_train
    ```

- To train with the TongUI-1M dataset, just change the `dataset` field in the config file to:
    ```yaml
    datasets: gui_video_full,baidu_experience_full,guiact_smartphone_thought,guiact_web_single_thought,guiact_web_multi_thought,showui-desktop-augmented,showui-web,amex-ele,amex-func,mind2web_with_thoughtsx3,wikihow_v3
    ```