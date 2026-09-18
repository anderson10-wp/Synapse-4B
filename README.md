# Synapse-4B

Synapse-4B is a custom ~4B parameter decoder-only language model implemented from scratch in PyTorch. The project targets constrained training environments such as Kaggle and includes GPU and TPU training scripts plus tools for local inference and Hugging Face conversion.

Created by Anderson Luan.

## Model

| Component | Configuration |
|---|---|
| Parameters | ~4.0B |
| Vocabulary | 32,768 |
| Layers | 40 |
| Hidden size | 3,072 |
| Attention heads | 24 Q / 4 KV |
| Intermediate size | 8,192 |
| Context length | 2,048 |
| Architecture | RMSNorm + RoPE + GQA + SwiGLU |
| Weight tying | Enabled |

The implementation follows a LLaMA-style decoder architecture so converted checkpoints can be loaded through the Hugging Face Transformers ecosystem.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── training/
│   ├── train_gpu.py
│   └── train_tpu.py
└── tools/
    ├── converter_hf.py
    └── chat_local.py
```

Training scripts are kept separate from inference/conversion utilities so the repository is easier to navigate and extend.

## Installation

For local inference and conversion:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Local inference

The model weights and tokenizer are not stored in this repository.

After converting a checkpoint, run:

```bash
python tools/chat_local.py --model-dir ./models/Synapse-4B-HF
```

You can also point the script to any other compatible converted model directory:

```bash
python tools/chat_local.py --model-dir "C:\path\to\Synapse-4B-HF"
```

## Checkpoint conversion

Convert a raw Synapse checkpoint to a Hugging Face-compatible directory:

```bash
python tools/converter_hf.py \
  --checkpoint ./checkpoints/synapse_4b_latest.pt \
  --output-dir ./models/Synapse-4B-HF
```

Optionally copy the tokenizer into the generated directory:

```bash
python tools/converter_hf.py \
  --checkpoint ./checkpoints/synapse_4b_latest.pt \
  --output-dir ./models/Synapse-4B-HF \
  --tokenizer ./tokenizer.json
```

## Training

### Multi-GPU

`training/train_gpu.py` is designed for distributed training with PyTorch FSDP on 2x NVIDIA T4-class GPUs. It uses activation checkpointing and an 8-bit AdamW optimizer to reduce memory pressure.

The script expects to run inside a suitable Kaggle/PyTorch environment and automatically searches `/kaggle/input` for the tokenizer and latest checkpoint.

Start it with the distributed launcher required by the environment, for example:

```bash
torchrun --nproc_per_node=2 training/train_gpu.py
```

### TPU

`training/train_tpu.py` targets a TPU v5e-style Kaggle environment through PyTorch XLA and uses a memory-efficient Adafactor implementation.

The TPU environment should provide a compatible PyTorch/XLA installation.

## Dataset

The training scripts stream `pinzhenchen/alpaca-cleaned-pt`. The dataset is fetched at runtime rather than vendored into this repository.

Because cloud environments and package versions change, exact training throughput and memory use may vary between sessions.

## Project status

This repository contains the model implementation, training code, conversion tooling, and local inference entry point. Large checkpoints and generated model files are intentionally excluded from Git history.

## Notes

- Keep checkpoints outside Git; use Git LFS or an external model registry for large artifacts.
- Keep the tokenizer used for training together with the checkpoint when publishing a runnable model.
- Record training-step, dataset, tokenizer, and environment information with each released checkpoint for reproducibility.
