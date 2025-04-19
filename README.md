# TongUI

Training Vision-Language-Action(VLA) Model for GUI & Computer Use tasks by watching online tutorials. Fully open-sourced dataset, model and training pipeline. Cost efficient solution for GUI task data generation.

针对图形操作界面任务设计的VLA模型和智能体框架。

<p align="center">
        &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2504.12679">Paper</a> &nbsp&nbsp 
        | 🤗 <a href="https://huggingface.co/collections/Bofeee5675/tongui-67f611e2d48b2b6e0d2ba3ee">Hugging Collections (Models & Datasets)</a>&nbsp&nbsp 
        | &nbsp&nbsp 🤗 <a href="https://huggingface.co/spaces/Bofeee5675/TongUI">Spaces Demo</a> &nbsp&nbsp | &nbsp&nbsp 🌐 <a href="https://tongui-agent.github.io/">Webpage</a>
</p>

> [**TongUI: Building Generalized GUI Agents by Learning from Multimodal Web Tutorials**](https://arxiv.org/abs/2504.12679)<br>
> [Bofei Zhang*](https://bofei5675.github.io/), [Zirui Shan*](), [Zhi Gao*](https://zhigao2017.github.io/), [Wang Zhang](), [Rui Xi](), [Xiaojian Ma](https://jeasinema.github.io/), [Yuan Tao](https://i.yt.sb/), [Xinxiao Wu](), [Song-Chun Zhu](https://www.zhusongchun.net/), [Qing Li✉](https://liqing.io/)

## 🌟 Updates
- [ ] Release all experiments scripts.
- [ ] Release Training pipeline.
- [ ] Release 1M version of TongUI dataset.
- [ ] Release TongUI-7B.
- [x] [2025.04.17] Release **TongUI-3B** model and **143K** training dataset.  


## 👋 Getting Started
We use [uv](https://docs.astral.sh/uv/getting-started/) to manage the dependencies.
```bash
uv sync --all-groups
```
To using `conda` and `pip` to install the dependencies.
```bash
conda create -n tongui python=3.12
conda activate tongui
pip install -e .
```

To execute any script by `uv`, you can use the following command.
```bash
uv run <script_name>.py
```
Just replace `uv` with python if you are using `conda` or `pip` to install the dependencies.
```bash
python <script_name>.py
```

### Gradio Demo (Local or Online)
We host an online Gradio Demo on [Hugging Face Spaces](https://huggingface.co/spaces/Bofeee5675/TongUI). Please feel free to try it. We also open source the code for this demo. Feel free to run it locally.
```bash
git clone https://huggingface.co/spaces/Bofeee5675/TongUI
cd TongUI
uv run app.py
```

### API Calling
You can programatically call the TongUI API by using the following code.
```bash
uv run examples/api.py
```

### Serve Model By vLLM
You can serve the model by `vLLM`.
```bash
uv run vllm serve Bofeee5675/TongUI-3B --port 8000 --served-model-name tongui-3b --limit-mm-per-prompt image=3
```

Then, you can use openai compatible API to call the model. Checkout `examples/call_vllm.py` for more details.
```bash
uv run examples/call_vllm.py
```
### Local Model
Checkout `examples/inference.py` for local inference.
```bash
uv run examples/inference.py
```
<!-- ## Advanced Example
Above examples are for basic usage of TongUI, which demonstrates a simple task for GUI element grounding. To address multi-turn navigation tasks, Checkout examples: -->

## 🔧 Training Details
For detailed information about model training, including hyperparameters, data preprocessing, and training configurations, please refer to our [Training Documentation](docs/train.md).

## 📚 Experiments
For comprehensive experimental results, ablation studies, and evaluation details, please check our [Experiments Documentation](docs/experiments.md).

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=TongUI-agent/TongUI-agent&type=Date)](https://www.star-history.com/#TongUI-agent/TongUI-agent&Date)

# Acknowledgement
We thank the following projects for their wonderful works.
- We adopt experiments, data preprocessing pipeline from  [ShowUI](https://github.com/showlab/ShowUI)
- We train our model by using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main?tab=readme-ov-file)

