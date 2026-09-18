import os
import sys
import time
import math
import subprocess
import glob
from pathlib import Path

# Instala bibliotecas necessárias se não existirem
try:
    import tokenizers
    from datasets import load_dataset
    import bitsandbytes as bnb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tokenizers", "datasets", "bitsandbytes"])
    from datasets import load_dataset
    import bitsandbytes as bnb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tokenizers import Tokenizer

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
from functools import partial

# ==============================================================================
# RADAR: LOCALIZA OS ARQUIVOS ANTES DE INICIAR
# ==============================================================================
rank = int(os.environ.get("LOCAL_RANK", 0))

if rank == 0:
    print("=" * 60)
    print("🔍 RADAR: Localizando os arquivos do modelo...")

todos = glob.glob("/kaggle/input/**/*", recursive=True)

TOKENIZER_PATH = None
for f in todos:
    if os.path.isfile(f) and "tokenizer" in os.path.basename(f).lower() and f.endswith(".json"):
        TOKENIZER_PATH = f
        break

CHECKPOINT_PATH = None
for f in todos:
    if os.path.isfile(f) and "synapse_4b_latest" in os.path.basename(f):
        CHECKPOINT_PATH = f
        break

if not CHECKPOINT_PATH and os.path.exists("/kaggle/working/checkpoints_4b/synapse_4b_latest.pt"):
    CHECKPOINT_PATH = "/kaggle/working/checkpoints_4b/synapse_4b_latest.pt"

if rank == 0:
    print(f"✅ Tokenizer:  {TOKENIZER_PATH}")
    print(f"✅ Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 60)

# ==============================================================================
# 1. CONFIGURAÇÃO E ARQUITETURA
# ==============================================================================
class SynapseNextConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 32768)
        self.max_seq_len = kwargs.get("max_seq_len", 2048)
        self.n_embd = kwargs.get("n_embd", 3072)
        self.n_layer = kwargs.get("n_layer", 40)
        self.n_head = kwargs.get("n_head", 24)
        self.n_kv_head = kwargs.get("n_kv_head", 4)
        self.intermediate_size = kwargs.get("intermediate_size", 8192)
        self.norm_eps = kwargs.get("norm_eps", 1e-5)
        self.rope_theta = kwargs.get("rope_theta", 10000.0)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = torch.mean(x * x, dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(norm + self.eps))

def precompute_freqs_cos_sin(dim: int, end: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end)
    frequencies = torch.outer(t, inv_freq)
    angles = torch.cat((frequencies, frequencies), dim=-1)
    return angles.cos(), angles.sin()

def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)

def apply_rotary_emb(xq, xk, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(2).to(xq.dtype)
    sin = sin.unsqueeze(0).unsqueeze(2).to(xq.dtype)
    return (xq * cos) + (rotate_half(xq) * sin), (xk * cos) + (rotate_half(xk) * sin)

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.n_kv_head = cfg.n_head, cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.q_proj(x).view(B, T, self.n_head, self.head_dim), self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim), self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        q, k = apply_rotary_emb(q, k, cos, sin)
        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(y.transpose(1, 2).contiguous().view(B, T, C))

class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.n_embd, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.n_embd, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm, self.attention = RMSNorm(cfg.n_embd, eps=cfg.norm_eps), Attention(cfg)
        self.mlp_norm, self.mlp = RMSNorm(cfg.n_embd, eps=cfg.norm_eps), SwiGLU(cfg)
    def forward(self, x, cos, sin):
        x = x + self.attention(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))

class SynapseNextModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        cos, sin = precompute_freqs_cos_sin(cfg.n_embd // cfg.n_head, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        x, cos, sin = self.token_embedding(tokens), self.freqs_cos[:T], self.freqs_sin[:T]
        
        for block in self.blocks:
            x = block(x, cos, sin)
            
        logits = self.lm_head(self.norm(x))
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1)) if targets is not None else None
        return logits, loss

# ==============================================================================
# LOOP DE TREINAMENTO DISTRIBUÍDO VIA TORCHRUN
# ==============================================================================
def setup(rank):
    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)

def train_loop(rank, world_size, tokenizer_path, checkpoint_path):
    setup(rank)
    CKPT_DIR = Path("/kaggle/working/checkpoints_4b")
    if rank == 0: CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(tokenizer_path) if tokenizer_path else None

    torch.set_default_dtype(torch.float16)
    cfg, model = SynapseNextConfig(), SynapseNextModel(SynapseNextConfig())
    torch.set_default_dtype(torch.float32)

    start_step = 0
    if checkpoint_path and rank == 0:
        ckpt = torch.load(checkpoint_path, map_location="cpu", mmap=True)
        model.load_state_dict(ckpt["model_state_dict"])
        start_step = ckpt.get("step", 0)
        del ckpt
        print(f"🔄 Checkpoint carregado! Retomando passo {start_step}!")

    step_tensor = torch.tensor([start_step], dtype=torch.long, device=rank)
    dist.broadcast(step_tensor, src=0)
    start_step = step_tensor.item()

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🧠 Synapse-4B Pronta para GPU T4: {total_params:,} neurônios")

    model = FSDP(
        model.to(rank), 
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Block}), 
        mixed_precision=MixedPrecision(param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16), 
        device_id=rank, 
        sync_module_states=True, 
        use_orig_params=True
    )
    
    apply_activation_checkpointing(model, check_fn=lambda submodule: isinstance(submodule, Block))
    
    # OTIMIZADOR DE 8-BITS
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2.0e-4, weight_decay=0.05)
    
    iter_ds = iter(load_dataset("pinzhenchen/alpaca-cleaned-pt", split="train", streaming=True))
    if rank > 0: [next(iter_ds, None) for _ in range(rank * 2000)]

    SEQ_LEN, BATCH_SIZE, GRAD_ACCUM, MAX_STEPS, WARMUP, token_buffer = 1024, 1, 4, 5000, 200, []
    
    def get_batch():
        nonlocal iter_ds, token_buffer
        needed = BATCH_SIZE * (SEQ_LEN + 1)
        while len(token_buffer) < needed:
            try:
                item = next(iter_ds)
                p = f'{item.get("instruction", "")}\n{item.get("input", "")}'.strip() if item.get("input", "") else item.get("instruction", "")
                texto = f"<sistema>\nVocê é a Synapse, uma inteligência artificial avançada e prestativa.\nPense em <pensamento> antes de responder.\n</sistema>\n<usuario> {p} <synapse> <pensamento>\n1. Analisar pedido.\n2. Responder com precisão.\n</pensamento>\n{item.get('output', '')} <fim>\n"
                token_buffer.extend(tokenizer.encode(texto).ids)
            except StopIteration:
                iter_ds = iter(load_dataset("pinzhenchen/alpaca-cleaned-pt", split="train", streaming=True))
        x_list, y_list = [token_buffer[i*(SEQ_LEN+1) : (i+1)*(SEQ_LEN+1)][:SEQ_LEN] for i in range(BATCH_SIZE)], [token_buffer[i*(SEQ_LEN+1) : (i+1)*(SEQ_LEN+1)][1:SEQ_LEN+1] for i in range(BATCH_SIZE)]
        token_buffer = token_buffer[needed:]
        return torch.tensor(x_list, dtype=torch.long, device=rank), torch.tensor(y_list, dtype=torch.long, device=rank)

    if rank == 0: print("🚀 INICIANDO TREINAMENTO COM TORCHRUN NAS 2x GPUs T4...")
    model.train()
    start_time, accum_loss = time.time(), 0.0
    
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
    scaler = ShardedGradScaler()
    
    for step in range(start_step + 1, MAX_STEPS + 1):
        
        # FIX DO CONGELAMENTO: O WARMUP deve iniciar DO ZERO em relação ao start_step!
        for pg in optimizer.param_groups: 
            pg["lr"] = 2.0e-4 * min(1.0, (step - start_step) / WARMUP)
            
        optimizer.zero_grad()
        for micro in range(GRAD_ACCUM):
            x, y = get_batch()
            _, loss = model(x, y)
            
            # LUPA MATEMÁTICA: Escala a perda para não virar ZERO no float16
            scaler.scale(loss / GRAD_ACCUM).backward()
            
            accum_loss += loss.item()
            
        # FIX DE PROTEÇÃO: Desfaz a escala e trava matemática para não explodir!
        scaler.unscale_(optimizer)
        model.clip_grad_norm_(1.0)
        
        # Otimizador dá o passo através do Scaler
        scaler.step(optimizer)
        scaler.update()
        
        if step % 20 == 0 or step == MAX_STEPS:
            loss_tensor = torch.tensor(accum_loss / 20, device=rank)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            if rank == 0:
                print(f"Passo {step:04d}/{MAX_STEPS} | Perda: {loss_tensor.item():.4f} | PPL: {math.exp(min(loss_tensor.item(), 20)):.2f} | {(20 * 2 * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN) / max(time.time() - start_time, 1e-5):.0f} tok/s")
                start_time = time.time()
            accum_loss = 0.0
            
        if step % 100 == 0 or step == MAX_STEPS:
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                cpu_state_dict = model.state_dict()
            if rank == 0:
                # FIX DE DISCO: Apaga o checkpoint anterior para não lotar o HD de 20GB do Kaggle!
                old_ckpt = CKPT_DIR / f"synapse_4b_latest.pt"
                if old_ckpt.exists(): old_ckpt.unlink()
                
                torch.save({"step": step, "model_state_dict": cpu_state_dict, "config": cfg.to_dict()}, str(CKPT_DIR / "synapse_4b_latest.pt"))

    dist.destroy_process_group()

if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        print("⚠️ ERRO: Execute usando o comando !torchrun na célula abaixo!")
        sys.exit(1)
    train_loop(int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"]), TOKENIZER_PATH, CHECKPOINT_PATH)
