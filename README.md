# Graph-Native_World_Modeling

This repository contains the implementation of **WorldGraph**, a graph world
model for learning and predicting graph transitions at three levels:

- **T1 — node level:** node addition, deletion, and property change;
- **T2 — edge level:** edge addition, deletion, and property change;
- **T3 — graph level:** structural changes of local subgraphs.

WorldGraph combines a multi-granularity graph encoder, a history-aware state
encoder, a transition Controller, and transition-aware reinforcement learning.

## Installation

Use Python 3.10 or newer and install the dependencies with:

```bash
pip install -r requirements.txt
```

## Data

The processed datasets are available from the [GWM-Zero Hugging Face dataset repository](https://huggingface.co/datasets/sanxun7/GWM-Zero). Download the required `.pt` files and place them under `data/processed/`, preserving their filenames.

## Training

Run commands from the repository root. The default command trains five seeds
and uses the validation split for model selection.

### Node-level task

```bash
python main.py --task T1 --dataset trade --device cuda:0
```

Available datasets: `trade`, `genre`, and `reddit`.

### Edge-level task

```bash
python main.py --task T2 --dataset un_vote --device cuda:0
```

Available datasets: `trade`, `un_vote`, `contact`, and `socialevo`.

### Graph-level task

```bash
python main.py --task T3 --dataset flights --device cuda:0
```

Available datasets: `flights`, `contact`, and `enron`.

## Citation
If you find our work useful in your research, please consider citing our paper, **[WorldGraph: Graph-Native World Modeling](https://arxiv.org/abs/2609.34159)**. Thank you!

```bibtex
@misc{ding2026worldgraphgraphnativeworldmodeling,
  title         = {WorldGraph: Graph-Native World Modeling},
  author        = {Zezhong Ding and Yipeng Li and Xike Xie},
  year          = {2026},
  eprint        = {2609.34159},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.34159}
}
```
