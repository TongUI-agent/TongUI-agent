# Experiments Documentation

To get the evaluation data, checkout this [HF repo](https://huggingface.co/datasets/Bofeee5675/GUI-Net-Benchmark).

## Grounding

We test with ScreenSpot, ScreenSpot-V2, ScreenSpot-Pro. Currently, we only put ScreenSpot data in the HF repo. This demonstrates how to setup the evaluation data.

For evaluation code, you can checkout
* `tongui/eval/run_screenspot.py`: using HF transformers to evaluate ScreenSpot
* `scripts/evaluation/run_screenspot_vllm.py`: using vllm to evaluate ScreenSpot. In this folder, you will find the code to evaluate ScreenSpot-V2 and ScreenSpot-Pro.


## Navigation

We test our model performance on AITW, Mind2Web, and Baidu Experience(we check the data quality and annotate the bounding boxes). To reproduce the results, you should first download the data from the HF repo, and checkout the code:
* `tongui/eval/run_aitw.py`: using HF transformers to evaluate AITW
* `tongui/eval/run_mind2web.py`: using HF transformers to evaluate Mind2Web
* `tongui/eval/run_baidu.py`: using HF transformers to evaluate Baidu Experience

For online evaluation such as MiniWob, we follow the documentation from [SeeClick](https://github.com/njucckevin/SeeClick/blob/main/agent_tasks/readme_agent.md) to evaluate the model.

