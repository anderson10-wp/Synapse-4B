import os
os.environ["XLA_USE_BF16"] = "1"
os.environ["PJRT_DEVICE"] = "TPU"
os.environ.pop("TPU_PROCESS_ADDRESSES", None)
os.environ.pop("CLOUD_TPU_TASK_ID", None)

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
    import torch_xla
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tokenizers", "datasets"])
    from datasets import load_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# ==============================================================================
# RADAR: LOCALIZA OS ARQUIVOS ANTES DE ACIONAR A TPU
# ==============================================================================
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
        x_normed = x * torch.rsqrt(norm + self.eps)
        return self.weight * x_normed

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
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out, xk_out

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        q, k = apply_rotary_emb(q, k, cos, sin)
        k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=2)
        v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)

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
        self.attn_norm = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.attention = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.mlp = SwiGLU(cfg)
    def forward(self, x, cos, sin):
        x = x + self.attention(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x

# ==============================================================================
# FUNÇÃO DE CHECKPOINT CUSTOMIZADA (BYPASS NOS BUGS DO PYTORCH XLA)
# ==============================================================================
class CheckpointXLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, block, x, cos, sin):
        ctx.block = block
        ctx.save_for_backward(x, cos, sin)
        with torch.no_grad():
            return block(x, cos, sin)

    @staticmethod
    def backward(ctx, grad_output):
        x, cos, sin = ctx.saved_tensors
        x = x.detach().requires_grad_(True)
        with torch.enable_grad():
            y = ctx.block(x, cos, sin)
        torch.autograd.backward(y, grad_output)
        return None, x.grad, None, None

class SynapseNextModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight # Weight Tying
        
        cos, sin = precompute_freqs_cos_sin(cfg.n_embd // cfg.n_head, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        cos = self.freqs_cos[:T]
        sin = self.freqs_sin[:T]
        x = self.token_embedding(tokens)
        
        for block in self.blocks:
            x = CheckpointXLA.apply(block, x, cos, sin)
                
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

# ==============================================================================
# 2. OTIMIZADOR ADAFACTOR (IN-PLACE E SEM QUEBRAS DE GRAFO XLA)
# ==============================================================================
import torch.optim as optim

class Adafactor(optim.Optimizer):
    def __init__(self, params, lr=1e-3, eps=(1e-30, 1e-3), clip_threshold=1.0,
                 decay_rate=-0.8, beta1=None, weight_decay=0.0, scale_parameter=True,
                 relative_step=True, warmup_init=False):
        defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold, decay_rate=decay_rate,
                        beta1=beta1, weight_decay=weight_decay, scale_parameter=scale_parameter,
                        relative_step=relative_step, warmup_init=warmup_init)
        super(Adafactor, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                if group["weight_decay"] > 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])
                    
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    if len(p.shape) >= 2:
                        state["exp_avg_sq_row"] = torch.zeros(p.shape[:-1], dtype=torch.bfloat16, device=p.device)
                        state["exp_avg_sq_col"] = torch.zeros(p.shape[:-2] + (p.shape[-1],), dtype=torch.bfloat16, device=p.device)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p.data, dtype=torch.bfloat16)
                        
                state["step"] += 1
                step_t = state["step"]
                lr_t = group["lr"]
                beta2t = 1.0 - math.pow(step_t, group["decay_rate"])
                
                if len(p.shape) >= 2:
                    grad_sq = grad ** 2
                    state["exp_avg_sq_row"].mul_(beta2t).add_(grad_sq.mean(dim=-1), alpha=1.0 - beta2t)
                    state["exp_avg_sq_col"].mul_(beta2t).add_(grad_sq.mean(dim=-2), alpha=1.0 - beta2t)
                    row_factor = (state["exp_avg_sq_row"] / state["exp_avg_sq_row"].mean(dim=-1, keepdim=True)).rsqrt().unsqueeze(-1)
                    col_factor = state["exp_avg_sq_col"].rsqrt().unsqueeze(-2)
                    grad.mul_(row_factor).mul_(col_factor)
                else:
                    state["exp_avg_sq"].mul_(beta2t).add_(grad ** 2, alpha=1.0 - beta2t)
                    grad.mul_(state["exp_avg_sq"].rsqrt())
                    
                grad_norm = grad.norm()
                divisor = torch.clamp(grad_norm / group["clip_threshold"], min=1.0)
                grad.div_(divisor)
                
                p.data.add_(grad, alpha=-lr_t)
        return loss

# ==============================================================================
# 3. LOOP DE TREINAMENTO DISTRIBUÍDO (TPU 8-CORE PADRÃO)
# ==============================================================================
def train_loop(index, tokenizer_path, checkpoint_path):
    device = xm.xla_device()
    
    CKPT_DIR = Path("/kaggle/working/checkpoints_4b")
    if index == 0:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
        
    tokenizer = Tokenizer.from_file(tokenizer_path) if tokenizer_path else None

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    cfg = SynapseNextConfig()
    model = SynapseNextModel(cfg)
    torch.set_default_dtype(old_dtype)
    
    if checkpoint_path and index == 0:
        ckpt = torch.load(checkpoint_path, map_location="cpu", mmap=True)
        model.load_state_dict(ckpt["model_state_dict"])
        start_step = ckpt.get("step", 0)
        del ckpt
        print(f"🔄 Checkpoint carregado com sucesso! Retomando a partir do passo {start_step}!")
        
    step_tensor = torch.tensor([start_step if 'start_step' in locals() else 0], dtype=torch.long, device=device)
    step_tensor = xm.all_reduce('sum', step_tensor)
    start_step = step_tensor.item()
        
    if index == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🧠 Synapse-4B Pronta: {total_params:,} neurônios (~{total_params/1e9:.2f}B)")

    model = model.to(device)
    xm.broadcast_master_param(model)
    
    LR_MAX = 2.0e-4
    optimizer = Adafactor(model.parameters(), lr=LR_MAX, scale_parameter=False, relative_step=False, warmup_init=False, weight_decay=0.05)
    
    ds_stream = load_dataset("pinzhenchen/alpaca-cleaned-pt", split="train", streaming=True)
    iter_ds = iter(ds_stream)
    
    if index > 0:
        for _ in range(index * 2000):
            try:
                next(iter_ds)
            except StopIteration:
                pass

    SEQ_LEN = 1024
    BATCH_SIZE = 2
    GRAD_ACCUM = 1
    MAX_STEPS = 50000
    WARMUP = 200
    token_buffer = []
    
    def get_batch():
        nonlocal iter_ds, token_buffer
        needed = BATCH_SIZE * (SEQ_LEN + 1)
        while len(token_buffer) < needed:
            try:
                item = next(iter_ds)
                inst = item.get("instruction", "")
                inp = item.get("input", "")
                out = item.get("output", "")
                p = f"{inst}\n{inp}".strip() if inp else inst
                texto = f"<sistema>\nVocê é a Synapse, uma IA criada pelo Anderson.\nPense em <pensamento> antes de responder.\n</sistema>\n<usuario> {p} <synapse> <pensamento>\n1. Analisar pedido.\n2. Responder com precisão.\n</pensamento>\n{out} <fim>\n"
                token_buffer.extend(tokenizer.encode(texto).ids)
            except StopIteration:
                iter_ds = iter(load_dataset("pinzhenchen/alpaca-cleaned-pt", split="train", streaming=True))
        
        x_list, y_list = [], []
        for i in range(BATCH_SIZE):
            chunk = token_buffer[i*(SEQ_LEN+1) : (i+1)*(SEQ_LEN+1)]
            x_list.append(chunk[:SEQ_LEN])
            y_list.append(chunk[1:SEQ_LEN+1])
            
        token_buffer = token_buffer[needed:]
        x = torch.tensor(x_list, dtype=torch.long, device=device)
        y = torch.tensor(y_list, dtype=torch.long, device=device)
        return x, y

    if index == 0:
        print("🚀 INICIANDO TREINAMENTO DISTRIBUÍDO NA TPU v5e-8 (8 CORES)...")
        
    model.train()
    start_time = time.time()
    accum_loss_tensor = torch.zeros(1, device=device)
    
    for step in range(start_step + 1, MAX_STEPS + 1):
        # WARMUP RELATIVO AO INÍCIO!
        lr = LR_MAX * min(1.0, (step - start_step) / WARMUP)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
            
        optimizer.zero_grad(set_to_none=True)
        
        x, y = get_batch()
        _, loss = model(x, y)
        loss.backward()
            
        xm.optimizer_step(optimizer)
        
        step_loss = loss.detach()
        accum_loss_tensor += step_loss
        
        if step % 20 == 0 or step == MAX_STEPS:
            avg_loss = accum_loss_tensor / 20
            avg_loss_reduced = xm.mesh_reduce("loss_reduce", avg_loss, lambda x: sum(x)/len(x))
            
            if index == 0:
                el = time.time() - start_time
                tps = (20 * 8 * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN) / max(el, 1e-5)
                val_loss = avg_loss_reduced.item()
                ppl = math.exp(min(val_loss, 20))
                print(f"Passo {step:04d}/{MAX_STEPS} | Perda: {val_loss:.4f} | PPL: {ppl:.2f} | {tps:.0f} tok/s")
                start_time = time.time()
                
            accum_loss_tensor.zero_()
            
        if step % 100 == 0 or step == MAX_STEPS:
            state_dict = model.state_dict()
            
            if index == 0:
                # FIX: Deleta o checkpoint antigo para não estourar o disco de 20GB do Kaggle!
                old_ckpt = CKPT_DIR / "synapse_4b_latest.pt"
                if old_ckpt.exists(): old_ckpt.unlink()
                
                xm.save({
                    "step": step,
                    "model_state_dict": state_dict,
                    "config": cfg.to_dict(),
                }, str(old_ckpt), master_only=True)
                
                print(f"💾 Checkpoint salvo com sucesso no passo {step}")

if __name__ == "__main__":
    xmp.spawn(train_loop, args=(TOKENIZER_PATH, CHECKPOINT_PATH), nprocs=None, start_method="fork")
    print("🎉 SESSÃO DE TREINO DA SYNAPSE-4B CONCLUÍDA COM SUCESSO NA TPU!")
