# 🧠 Synapse-4B 

**Synapse-4B** is a custom 4-Billion parameter Large Language Model (LLM) built entirely from scratch in PyTorch. It features a modern architecture (SwiGLU, RMSNorm, RoPE, and Weight Tying) and is designed to be trained in highly constrained environments like Kaggle (2x T4 GPUs or TPU v5e-8) without paying for expensive cloud instances.

Created by **Anderson Luan**, this project demonstrates how to train a massive neural network by pushing free hardware to its absolute limits using cutting-edge distributed training techniques.

---

## 🏗️ Architecture

The model architecture maps mathematically 1:1 to LLaMA, allowing for easy conversion to HuggingFace formats.
- **Parameters:** ~4.00B
- **Vocabulary:** 32,768 (Custom BPE Tokenizer)
- **Layers:** 40
- **Heads:** 24 (Query) / 4 (Key/Value) - *Grouped Query Attention*
- **Embedding Size:** 3072
- **Intermediate Size (MLP):** 8192
- **Context Length:** 2048

---

## 🚀 Training Techniques

Training a 4B parameter model usually requires massive GPU clusters (like 8x A100s). We successfully trained it on Kaggle using:

### 1. Multi-GPU (2x T4 15GB) - `train_gpu.py`
- **FSDP (Fully Sharded Data Parallel):** Shards the model states across both T4 GPUs.
- **8-Bit AdamW (bitsandbytes):** Optimizes memory by keeping the optimizer states in 8-bit precision, saving 16GB of VRAM.
- **Activation Checkpointing:** Frees up massive VRAM during the forward pass by recalculating activations in the backward pass.
- **Float16 + ShardedGradScaler:** Prevents Float16 gradient underflow (frozen loss) while avoiding the memory cost of Float32.

### 2. TPU v5e-8 (Supercomputer) - `train_tpu.py`
- **PyTorch XLA:** Native XLA compilation for Google's TPUs.
- **In-Place Adafactor:** A memory-efficient optimizer modified to run without float32 casting, saving gigabytes of HBM.
- **Relative Warmup & Checkpoint Deletion:** Bulletproofed logic to prevent gradient shock upon resuming and avoid Kaggle's 20GB disk limit.

---

## 🛠️ Repository Structure

- `train_gpu.py` -> The definitive script for Kaggle Multi-GPU (T4) training.
- `train_tpu.py` -> The definitive script for Kaggle TPU v5e-8 training.
- `tools/converter_hf.py` -> Converts the raw PyTorch `.pt` file into HuggingFace universal `LlamaForCausalLM` format.
- `tools/chat_local.py` -> An interactive terminal script to chat with the model locally.

---

## 💡 How to Use

1. **Train on Kaggle:** Upload `train_tpu.py` or `train_gpu.py` to a Kaggle Notebook and run it. The script automatically handles resuming from previous checkpoints.
2. **Convert to HuggingFace:**
   ```bash
   python tools/converter_hf.py
   ```
3. **Chat Locally:**
   ```bash
   python tools/chat_local.py
   ```

## 🏆 Journey
This model was trained in segments by leveraging "Account Relaying" to bypass Kaggle time limits. Starting from babbling subwords, it is continuously trained toward achieving full fluency.

*Built with passion and no fine-tuning of third-party weights. 100% Sovereign AI.*
