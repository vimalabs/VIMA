# VIMA: General Robot Manipulation with Multimodal Prompts
<div align="center">

[[Website]](https://vimalabs.github.io/)
[[arXiv]](https://arxiv.org/abs/2210.03094)
[[PDF]](https://vimalabs.github.io/assets/vima_paper.pdf)
[[Pretrained Models]](#Pretrained-Models)
[[VIMA-Bench]](https://github.com/vimalabs/VimaBench)
[[Training Data]](https://huggingface.co/datasets/VIMA/VIMA-Data)
[[Model Card]](model-card.md)

[![Python Version](https://img.shields.io/badge/Python-3.9-blue.svg)](https://github.com/vimalabs/VIMA)
[<img src="https://img.shields.io/badge/Framework-PyTorch-red.svg"/>](https://pytorch.org/)
[![GitHub license](https://img.shields.io/github/license/vimalabs/VIMA)](https://github.com/vimalabs/VIMA/blob/main/LICENSE)
______________________________________________________________________
![](images/pull.png)
</div>

Prompt-based learning has emerged as a successful paradigm in natural language processing, where a single general-purpose language model can be instructed to perform any task specified by input prompts. However, different robotics tasks are still tackled by specialized models. This work shows that we can express a wide spectrum of robot manipulation tasks with *multimodal prompts*, interleaving textual and visual tokens.
We introduce VIMA (**Vi**suo**M**otor **A**ttention model, reads "_v-eye-ma_"), a novel scalable multi-task robot learner with a uniform sequence IO interface achieved through multimodal prompts. The architecture follows the encoder-decoder transformer design proven to be effective and scalable in NLP. VIMA encodes an input sequence of interleaving textual and visual prompt tokens with a [pretrained](https://www.deepmind.com/publications/multimodal-few-shot-learning-with-frozen-language-models) [language model](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html), and decodes robot control actions autoregressively for each environment interaction step. The transformer decoder is conditioned on the prompt via cross-attention layers that alternate with the usual causal self-attention. Instead of operating on raw pixels, VIMA adopts an object-centric approach. We parse all images in the prompt or observation into objects by [off-the-shelf detectors](https://arxiv.org/abs/1703.06870), and flatten them into sequences of object tokens. All these design choices combined deliver a conceptually simple architecture with strong model and data scaling properties.

In this repo, we provide VIMA model code, pre-trained checkpoints covering a spectrum of model sizes, and demo and eval scripts. This codebase is under [MIT License](LICENSE).

# Installation
VIMA requires Python ≥ 3.9. We have tested on Ubuntu 20.04. Installing VIMA codebase is as simple as:

```bash
pip install git+https://github.com/vimalabs/VIMA
```

# Pretrained Models
We host pretrained models covering a spectrum of model capacity on [HuggingFace](https://huggingface.co/VIMA/VIMA). Download links are listed below.

| [200M](https://huggingface.co/VIMA/VIMA/resolve/main/200M.ckpt) | [92M](https://huggingface.co/VIMA/VIMA/resolve/main/92M.ckpt) | [43M](https://huggingface.co/VIMA/VIMA/resolve/main/43M.ckpt) | [20M](https://huggingface.co/VIMA/VIMA/resolve/main/20M.ckpt) | [9M](https://huggingface.co/VIMA/VIMA/resolve/main/9M.ckpt) | [4M](https://huggingface.co/VIMA/VIMA/resolve/main/4M.ckpt) | [2M](https://huggingface.co/VIMA/VIMA/resolve/main/2M.ckpt)    |
|-----------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------|-------------------------------------------------------------|-------------------------------------------------------------|-----|

# Demo
To run the live demonstration, first follow the [instruction](https://github.com/vimalabs/VimaBench/tree/main#installation) to install [VIMA-Bench](https://github.com/vimalabs/VimaBench).Then we can run a live demo through

```bash
python3 scripts/example.py --ckpt={ckpt_path} --device={device} --partition={eval_level} --task={task}
```

Here `eval_level` means one out of four evaluation levels and can be chosen from `placement_generalization`, `combinatorial_generalization`, `novel_object_generalization`, and `novel_task_generalization`. `task` means a specific meta-task. Please refer to [task suite](https://github.com/vimalabs/VimaBench/tree/main#task-suite) and [benchmark](https://github.com/vimalabs/VimaBench/tree/main#evaluation-benchmark) for more details.

After running the above command, we should see a PyBullet GUI pop up, alongside a small window showing the multimodal prompt. Then a robot arm should move to complete the corresponding task. Note that this demo may not work on headless machines since the PyBullet GUI requires a display.

# Paper and Citation

Our paper is posted on [arXiv](https://arxiv.org/abs/2210.03094). If you find our work useful, please consider citing us! 

```bibtex
@article{jiang2022vima,
  title   = {VIMA: General Robot Manipulation with Multimodal Prompts},
  author  = {Yunfan Jiang and Agrim Gupta and Zichen Zhang and Guanzhi Wang and Yongqiang Dou and Yanjun Chen and Li Fei-Fei and Anima Anandkumar and Yuke Zhu and Linxi Fan},
  year    = {2022},
  journal = {arXiv preprint arXiv: Arxiv-2210.03094}
}
```
