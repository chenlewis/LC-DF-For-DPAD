## Overview
This is the implementation of the method proposed in "Peiquan Li, Changsheng Chen, Yulia Chernyshova, Dmitry Nikolaev, Shunquan Tan, Vladimir Arlazarov, "Disentangling Moiré and Texture: Towards Robust Display-Recapture Detection for Document Images," 2025 IEEE International Workshop on Information Forensics and Security (WIFS)

## Environment
This code was implemented with Python 3.7 and PyTorch 1.10.1. You can install all the requirements via:
```bash
pip install -r requirements.txt
```

## Quick Start

## Train
```bash
CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.run --master_port 29501 --nproc_per_node=2 train.py --config configs/train/clas/train_bac2.yaml
```

## Test
```bash
python test.py --config [CONFIG_PATH] --model [MODEL_PATH] --prompt [PROMPT_PATH]
```

## Models and Results

Please find the pre-trained models[here](https://pan.baidu.com/s/1k1XmQj4n6xFGMRED1BjgGg).

