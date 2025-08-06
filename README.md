# RepoDeepSearch
In this work, we improve the repo deep search performance of LLMs  by introducing ToolTrain, a tool-interactive training approach to enhance their tool-use and reasoning.  

## Training
ToolTrain first utilizes supervised fine-tuning to warm up the model with our lightweight agent, RepoSearcher, and then employs tool-interactive reinforcement learning to teach the model how to effectively navigate code repositories.
### SFT
We use the SFT module of verl to finetune the LLM, and the training script is as follows.
```
bash scripts/tooltrain_sft.sh
```
### RL
We use the RL module of verl to further improve the LLM, and the training script is as follows.
```
bash scripts/tooltrain_rl.sh
```

## Inference
We evaluated RepoSearcher with the ToolTrain model on SWE-Bench-Verified, comparing it with the various baselines.
### Localization
The localization inference script is as follows.
```
bash scripts/loc_inference.sh
```

### Patch Generation
The patch generation inference script is as follows.
```
bash scripts/patch_inference.sh
```

## Evaluation

## Localization Evaluation
The evaluation script for localization results is as follows.
```
python evaluation/FLEval.py --loc_file <loc_file_path>
```

## Patch Generation Evaluation
We utilize the official evaluation script provided by the SWE-Bench-Verified, which can be found at https://github.com/SWE-bench/SWE-bench.

## Citation

```bibtex
@misc{ma2025toolintegratedreinforcementlearningrepo,
      title={Tool-integrated Reinforcement Learning for Repo Deep Search}, 
      author={Zexiong Ma and Chao Peng and Qunhong Zeng and Pengfei Gao and Yanzhen Zou and Bing Xie},
      year={2025},
      eprint={2508.03012},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2508.03012}, 
}
```


## Acknowledgement
* [Verl](https://github.com/volcengine/verl)
* [Agentless](https://github.com/OpenAutoCoder/Agentless/tree/main)
* [SWE-Bench](https://github.com/swe-bench/SWE-bench.git)
* [CoSIL](https://github.com/ZhonghaoJiang/CoSIL)