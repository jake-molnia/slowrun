"""
LSRecurrent: Iterative-refinement language model using LSLinear as the core primitive.

Instead of transformer attention blocks, each layer performs K steps of iterative
refinement. At each step, the current token representation x is combined with a
hidden state h and a causally-shifted neighbor signal, then projected through
LSLinear layers to produce a residual update. This gives basic causal sequence
coupling without attention.

Architecture per block (repeated K times):
    x_norm = RMSNorm(x)
    shifted = causal_shift(x_norm, 1)
    inp = concat([x_norm, h, shifted], dim=-1)
    h = SiLU(W_in(inp))
    delta = W_out(h)
    x = x + delta

Training supports progressive iteration scheduling: start with K=1 and ramp up
to the target K over a configurable warmup fraction.

Usage:
    # Smoke test (single GPU, fast)
    torchrun --standalone --nproc_per_node=1 train_lsrecurrent.py --config smoke

    # Small config on 8 GPUs
    torchrun --standalone --nproc_per_node=8 train_lsrecurrent.py --config small

    # Medium config with custom LR
    torchrun --standalone --nproc_per_node=8 train_lsrecurrent.py --config medium --matrix-lr 0.02
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import math
import time
import json
import argparse
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
try:
    import wandb
except ImportError:
    wandb = None
import tiktoken

_script_start = time.time()

# =============================================================================
# CLI arguments
# =============================================================================

parser = argparse.ArgumentParser(description="Train LSRecurrent model")
parser.add_argument("--config", type=str, default=None, choices=["smoke", "small", "medium"],
                    help="Preset config (overrides individual arch args)")
parser.add_argument("--device-batch-size", type=int, default=None)
parser.add_argument("--num-epochs", type=int, default=None)
parser.add_argument("--patience", type=int, default=-1)
parser.add_argument("--run", type=str, default=None)
parser.add_argument("--scalar-lr", type=float, default=0.1)
parser.add_argument("--matrix-lr", type=float, default=0.04)
parser.add_argument("--weight-decay", type=float, default=1.3)
parser.add_argument("--total-batch-size", type=int, default=None)
parser.add_argument("--save-result", type=str, default="")
parser.add_argument("--output_json", type=str, default=None)
parser.add_argument("--wandb_group", type=str, default=None)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--lr_multiplier", type=float, default=0.25)
parser.add_argument("--input_bin", type=str, default=None)
parser.add_argument("--input_val_bin", type=str, default=None)
parser.add_argument("--warmdown-ratio", type=float, default=0.2)
parser.add_argument("--logit-cap", type=float, default=10.0)
parser.add_argument("--stoch-depth", type=float, default=0.0)
# LSLinear parameters
parser.add_argument("--ls-enabled", type=int, default=1,
                    help="Use LSLinear (1=on, 0=off for nn.Linear baseline)")
parser.add_argument("--ls-num-blocks", type=int, default=16)
parser.add_argument("--ls-rank", type=int, default=128)
# LSRecurrent architecture
parser.add_argument("--n_layer", type=int, default=None)
parser.add_argument("--n_embd", type=int, default=None)
parser.add_argument("--h_dim", type=int, default=None, help="Hidden state dimension (default: n_embd)")
parser.add_argument("--n_iter", type=int, default=None, help="Max refinement iterations per block")
# Iteration schedule
parser.add_argument("--iter-warmup-frac", type=float, default=0.3,
                    help="Fraction of training to ramp iterations from 1 to n_iter")
# Logit averaging
parser.add_argument("--logit-avg", type=int, default=3)
parser.add_argument("--logit-avg-dir", type=str, default="logit_avg_ckpts")
parser.add_argument("--logit-avg-mode", type=str, default="both",
                    choices=["equal", "weighted", "both"])
parser.add_argument("--eval-logit-avg", action="store_true")
parser.add_argument("--swa-last-epochs", type=int, default=3)
args = parser.parse_args()

# =============================================================================
# Config presets
# =============================================================================

CONFIGS = {
    "smoke": dict(n_layer=4, n_embd=256, h_dim=256, n_iter=2,
                  num_epochs=2, total_batch_size=8192, device_batch_size=4),
    "small": dict(n_layer=8, n_embd=512, h_dim=512, n_iter=3,
                  num_epochs=5, total_batch_size=65536, device_batch_size=4),
    "medium": dict(n_layer=12, n_embd=768, h_dim=768, n_iter=4,
                   num_epochs=8, total_batch_size=262144, device_batch_size=4),
}

def _resolve(arg_val, config_val, default):
    if arg_val is not None:
        return arg_val
    if config_val is not None:
        return config_val
    return default

cfg = CONFIGS.get(args.config, {})
N_LAYER = _resolve(args.n_layer, cfg.get("n_layer"), 12)
N_EMBD = _resolve(args.n_embd, cfg.get("n_embd"), 768)
H_DIM = _resolve(args.h_dim, cfg.get("h_dim"), N_EMBD)
N_ITER = _resolve(args.n_iter, cfg.get("n_iter"), 4)
NUM_EPOCHS = _resolve(args.num_epochs, cfg.get("num_epochs"), 8)
TOTAL_BATCH_SIZE = _resolve(args.total_batch_size, cfg.get("total_batch_size"), 262144)
DEVICE_BATCH_SIZE = _resolve(args.device_batch_size, cfg.get("device_batch_size"), 4)

if args.output_json and not args.save_result:
    args.save_result = args.output_json

# =============================================================================
# Hyperparameters
# =============================================================================

MAX_SEQ_LEN = 2048
EVAL_TOKENS = 10_000_000
DATA_DIR = "fineweb_data"

_lr_mult = args.lr_multiplier if args.lr_multiplier is not None else 1.0
MATRIX_LR = args.matrix_lr * _lr_mult
SCALAR_LR = args.scalar_lr * _lr_mult
EMBEDDING_LR = 0.15 * _lr_mult
UNEMBEDDING_LR = 0.002 * _lr_mult
WEIGHT_DECAY = args.weight_decay
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = args.warmdown_ratio
FINAL_LR_FRAC = 0.0
LOGIT_CAP = args.logit_cap

# =============================================================================
# Utilities
# =============================================================================

def get_dist_info():
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return True, int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    return False, 0, 0, 1

def print0(s="", **kwargs):
    if int(os.environ.get('RANK', 0)) == 0:
        print(s, flush=True, **kwargs)

class DummyWandb:
    def __init__(self): self.summary = {}
    def log(self, *a, **kw): pass
    def finish(self): pass
    def log_code(self, *a, **kw): pass

def load_state_dict_into_model(model, state_dict):
    for name, p in model.named_parameters():
        if name in state_dict:
            p.data.copy_(state_dict[name].to(p.device, dtype=p.dtype))

# =============================================================================
# LSLinear: Low-rank + Block-diagonal Sparse Linear Layer
# =============================================================================

class LSLinearCompiled(nn.Module):
    """Low-rank + Block-Diagonal linear: y = blockdiag(W)(x) + x @ B^T @ A^T."""
    def __init__(self, in_features, out_features, num_blocks, rank, bias=False):
        super().__init__()
        assert in_features % num_blocks == 0 and out_features % num_blocks == 0, \
            f"{in_features},{out_features} must divide by {num_blocks}"
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        self.rank = rank
        self.block_in = in_features // num_blocks
        self.block_out = out_features // num_blocks
        self.sparse_weight = nn.Parameter(torch.empty(num_blocks * self.block_out, self.block_in))
        self.A = nn.Parameter(torch.empty(out_features, rank))
        self.B = nn.Parameter(torch.empty(rank, in_features))
        self.register_parameter("bias", nn.Parameter(torch.empty(out_features)) if bias else None)

    def forward(self, x):
        shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        W = self.sparse_weight.reshape(self.num_blocks, self.block_out, self.block_in)
        x_b = x_flat.reshape(-1, self.num_blocks, self.block_in).transpose(0, 1)
        y = torch.bmm(W, x_b.transpose(-1, -2)).transpose(-1, -2)
        y = y.transpose(0, 1).reshape(-1, self.out_features)
        y = y + (x_flat @ self.B.t()) @ self.A.t()
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.out_features)

LS_ENABLED = bool(args.ls_enabled)
LS_NUM_BLOCKS = args.ls_num_blocks
LS_RANK = args.ls_rank

def make_linear(in_features, out_features, bias=False, use_ls=True):
    if LS_ENABLED and use_ls:
        nb = LS_NUM_BLOCKS
        while (in_features % nb != 0 or out_features % nb != 0) and nb > 1:
            nb -= 1
        return LSLinearCompiled(in_features, out_features, nb, LS_RANK, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)

# =============================================================================
# Initialization helpers
# =============================================================================

def _init_ls_uniform(layer, bound):
    W = layer.sparse_weight.data.reshape(layer.num_blocks, layer.block_out, layer.block_in)
    for i in range(layer.num_blocks):
        torch.nn.init.uniform_(W[i], -bound, bound)
    torch.nn.init.zeros_(layer.A)
    torch.nn.init.kaiming_uniform_(layer.B, a=math.sqrt(5))

def _init_ls_normal(layer, std):
    W = layer.sparse_weight.data.reshape(layer.num_blocks, layer.block_out, layer.block_in)
    for i in range(layer.num_blocks):
        torch.nn.init.normal_(W[i], mean=0.0, std=std)
    torch.nn.init.zeros_(layer.A)
    torch.nn.init.kaiming_uniform_(layer.B, a=math.sqrt(5))

def _init_layer(layer, mode, scale):
    if isinstance(layer, LSLinearCompiled):
        if mode == 'uniform':
            _init_ls_uniform(layer, scale)
        else:
            _init_ls_normal(layer, scale)
    else:
        if mode == 'uniform':
            torch.nn.init.uniform_(layer.weight, -scale, scale)
        else:
            torch.nn.init.normal_(layer.weight, mean=0.0, std=scale)

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

# =============================================================================
# LSRecurrent Model
# =============================================================================

@dataclass
class LSRecurrentConfig:
    sequence_len: int = MAX_SEQ_LEN
    vocab_size: int = 50257
    n_layer: int = N_LAYER
    n_embd: int = N_EMBD
    h_dim: int = H_DIM
    n_iter: int = N_ITER
    dropout: float = 0.0
    stoch_depth: float = 0.0


class LSRecurrentBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.h_dim = config.h_dim
        self.n_embd = config.n_embd
        in_dim = 2 * config.n_embd + config.h_dim
        self.w_in = make_linear(in_dim, config.h_dim)
        self.w_out = make_linear(config.h_dim, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.drop_prob = config.stoch_depth * (layer_idx / max(config.n_layer - 1, 1))

    def _iterate(self, x, active_k):
        B, T, D = x.shape
        h = x.new_zeros(B, T, self.h_dim)
        for _ in range(active_k):
            x_norm = norm(x)
            shifted = F.pad(x_norm[:, :-1], (0, 0, 1, 0))
            inp = torch.cat([x_norm, h, shifted], dim=-1)
            h = F.silu(self.w_in(inp))
            delta = self.w_out(h)
            x = x + self.resid_dropout(delta)
        return x

    def forward(self, x, active_k):
        if self.training and self.drop_prob > 0:
            keep = (torch.rand((), device=x.device) >= self.drop_prob).to(x.dtype)
            x_in = x
            x = self._iterate(x, active_k)
            x = x_in + keep * (x - x_in)
        else:
            x = self._iterate(x, active_k)
        return x


class LSRecurrentLM(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        padded_vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab}")
        self.padded_vocab = padded_vocab
        self.wte = nn.Embedding(padded_vocab, config.n_embd)
        self.wpe = nn.Embedding(config.sequence_len, config.n_embd)
        self.blocks = nn.ModuleList([LSRecurrentBlock(config, i) for i in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)
        self.active_k = config.n_iter

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.wpe.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        s = 3**0.5 * self.config.n_embd**-0.5
        out_std = self.config.n_embd ** -0.5
        for block in self.blocks:
            _init_layer(block.w_in, 'uniform', s)
            _init_layer(block.w_out, 'normal', out_std)

        if self.wte.weight.device.type == "cuda":
            self.wte.to(dtype=torch.bfloat16)
            self.wpe.to(dtype=torch.bfloat16)

    def get_device(self):
        return self.wte.weight.device

    def setup_optimizer(self):
        ddp, rank, local_rank, world_size = get_dist_info()

        embed_params = list(self.wte.parameters()) + list(self.wpe.parameters())
        lm_head_params = list(self.lm_head.parameters())
        matrix_params = list(self.blocks.parameters())

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=UNEMBEDDING_LR,
                 betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
            dict(kind='adamw', params=embed_params, lr=EMBEDDING_LR,
                 betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(kind='muon', params=group_params, lr=MATRIX_LR,
                                     momentum=0.95, ns_steps=5, beta2=0.95,
                                     weight_decay=WEIGHT_DECAY))

        optimizer = DistMuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()
        positions = torch.arange(T, device=idx.device).unsqueeze(0)
        x = norm(self.wte(idx) + self.wpe(positions))

        for block in self.blocks:
            x = block(x, self.active_k)

        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        if LOGIT_CAP > 0:
            logits = LOGIT_CAP * torch.tanh(logits / LOGIT_CAP)

        if targets is None:
            return logits

        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                               ignore_index=-1, reduction=loss_reduction)
        if loss_reduction != 'mean':
            return loss
        return loss, {'lm_loss': loss}

# =============================================================================
# Optimizer: DistMuonAdamW
# =============================================================================

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    p.add_(exp_avg / ((exp_avg_sq / bias2).sqrt() + eps_t), alpha=-(lr_t / bias1))

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            X = a * X + X @ (b * A + c * (A @ A))
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            X = a * X + (b * A + c * (A @ A)) @ X
    g = X
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class DistMuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0)
        self._adamw_lr_t = torch.tensor(0.0)
        self._adamw_beta1_t = torch.tensor(0.0)
        self._adamw_beta2_t = torch.tensor(0.0)
        self._adamw_eps_t = torch.tensor(0.0)
        self._adamw_wd_t = torch.tensor(0.0)
        self._muon_momentum_t = torch.tensor(0.0)
        self._muon_lr_t = torch.tensor(0.0)
        self._muon_wd_t = torch.tensor(0.0)
        self._muon_beta2_t = torch.tensor(0.0)

    def _reduce_adamw(self, group, world_size):
        infos = {}
        for p in group['params']:
            grad = p.grad
            if p.numel() < 1024:
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                assert grad.shape[0] % world_size == 0
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=infos)

    def _reduce_muon(self, group, world_size):
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        stacked_grads = torch.empty(padded, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(torch.stack([p.grad for p in params]))
        if len(params) < padded:
            stacked_grads[len(params):].zero_()
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()
        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group, info, gather_list, rank, world_size):
        for p in group['params']:
            pinfo = info['param_infos'][p]
            pinfo['future'].wait()
            state = self.state[p]
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p_slice, pinfo['grad_slice'], state['exp_avg'], state['exp_avg_sq'],
                           self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                           self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)
            if not pinfo['is_small']:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group, info, gather_list, rank):
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            s = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(s, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        updated = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        if num_owned > 0:
            owned = torch.stack([params[start_idx + i] for i in range(num_owned)])
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            muon_step_fused(info['grad_chunk'][:num_owned], owned,
                          state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                          self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                          group["ns_steps"], red_dim)
            updated[:num_owned].copy_(owned)
        if num_owned < chunk_size:
            updated[num_owned:].zero_()
        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    @torch.no_grad()
    def step(self):
        if not dist.is_available() or not dist.is_initialized():
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if group["kind"] == "adamw":
                        if "step" not in state:
                            state["step"] = 0
                            state["exp_avg"] = torch.zeros_like(p)
                            state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] += 1
                        grad = p.grad
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        state["exp_avg"].lerp_(grad, 1 - group["betas"][0])
                        state["exp_avg_sq"].lerp_(grad.square(), 1 - group["betas"][1])
                        bias1 = 1 - group["betas"][0] ** state["step"]
                        bias2 = 1 - group["betas"][1] ** state["step"]
                        p.add_(state["exp_avg"] / ((state["exp_avg_sq"] / bias2).sqrt() + group["eps"]), alpha=-(group["lr"] / bias1))
                    elif group["kind"] == "muon":
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        grad = p.grad
                        state["momentum_buffer"].lerp_(grad, 1 - group["momentum"])
                        g = grad.lerp_(state["momentum_buffer"], group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(g, alpha=-group["lr"])
            return
        rank, world_size = dist.get_rank(), dist.get_world_size()
        reduce_infos = []
        for group in self.param_groups:
            if group["kind"] == "adamw": reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group["kind"] == "muon": reduce_infos.append(self._reduce_muon(group, world_size))
        gather_list = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group["kind"] == "adamw": self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group["kind"] == "muon": self._compute_muon(group, info, gather_list, rank)
        for info in gather_list:
            info["future"].wait()
            if info.get("params") is not None:
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

# =============================================================================
# DataLoader
# =============================================================================

class DataLoader:
    def __init__(self, filepath, B, T, device="cuda"):
        data = torch.load(filepath, weights_only=True)
        chunks = data['chunks']
        valid_counts = data['valid_counts']
        file_B = data['batch_size']
        sequence_size = data['sequence_size']
        assert sequence_size == T + 1, f"Data sequence_size {sequence_size} != T+1={T+1}"

        all_seqs = []
        for chunk, vc in zip(chunks, valid_counts):
            rows = chunk.view(file_B, sequence_size)[:vc]
            all_seqs.append(rows)
        all_seqs = torch.cat(all_seqs, dim=0).long()

        _, rank, _, world_size = get_dist_info()
        seqs_per_step = B * world_size
        num_steps = len(all_seqs) // seqs_per_step
        usable = num_steps * seqs_per_step
        all_seqs = all_seqs[:usable].view(num_steps, world_size, B, sequence_size)

        self.rank_data = all_seqs[:, rank].contiguous()
        self.num_steps = num_steps
        self.total_tokens = usable * T
        self.device = device
        self.pos = 0
        self.epoch = 1

    def __iter__(self):
        return self

    def _shuffle(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        perm = torch.randperm(self.num_steps, generator=g)
        self.rank_data = self.rank_data[perm]

    def __next__(self):
        if self.pos >= self.num_steps:
            self.pos = 0
            self.epoch += 1
            print0(f"Starting epoch {self.epoch}")
            self._shuffle()
        batch = self.rank_data[self.pos].to(self.device, non_blocking=True)
        self.pos += 1
        return batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), self.epoch

# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_tokens = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y, _ = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none').view(-1)
        y = y.view(-1)
        mask = y != -1
        total_loss += loss2d[mask].sum()
        total_tokens += mask.sum()
        num_bytes2d = token_bytes[y]
        total_nats += (loss2d * (num_bytes2d > 0)).sum()
        total_bytes += num_bytes2d.sum()
    if dist.is_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    total_nats, total_bytes = total_nats.item(), total_bytes.item()
    total_loss, total_tokens = total_loss.item(), total_tokens.item()
    bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float('inf')
    loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return bpb, loss


@torch.no_grad()
def evaluate_bpb_logit_avg(eval_model, ckpt_paths, weights, steps):
    dev = orig_model.get_device()
    V = orig_model.config.vocab_size

    val_loader = build_val_loader()
    all_x, all_y = [], []
    for _ in range(steps):
        x, y, _ = next(val_loader)
        all_x.append(x.cpu())
        all_y.append(y.cpu())

    BT = all_y[0].numel()
    batch_target_probs = torch.zeros(steps, BT, dtype=torch.float32, device=dev)

    for path, w in zip(ckpt_paths, weights):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        load_state_dict_into_model(orig_model, ckpt)
        del ckpt
        for i, (x, y) in enumerate(zip(all_x, all_y)):
            y_flat = y.view(-1).to(dev)
            with autocast_ctx:
                logits = eval_model(x.to(dev))
            probs = torch.softmax(logits.view(BT, V).float(), dim=-1)
            tgt = probs[torch.arange(BT, device=dev), y_flat.clamp_min(0)]
            batch_target_probs[i].add_(tgt, alpha=w)

    total_nats = torch.tensor(0.0, dtype=torch.float64, device=dev)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=dev)
    total_loss = torch.tensor(0.0, dtype=torch.float64, device=dev)
    total_tokens = torch.tensor(0, dtype=torch.int64, device=dev)

    for i, y in enumerate(all_y):
        y_flat = y.view(-1).to(dev)
        mask = y_flat != -1
        log_probs = batch_target_probs[i].clamp_min(1e-40).log()
        num_bytes_batch = token_bytes[y_flat.clamp_min(0)]
        total_nats += (log_probs.neg() * (num_bytes_batch > 0)).sum().double()
        total_bytes += num_bytes_batch.sum()
        total_loss += log_probs[mask].neg().sum().double()
        total_tokens += mask.sum()

    del batch_target_probs

    if dist.is_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    bpb = total_nats.item() / (math.log(2) * total_bytes.item()) if total_bytes.item() > 0 else float('inf')
    loss = total_loss.item() / total_tokens.item() if total_tokens.item() > 0 else float('inf')
    return bpb, loss

# =============================================================================
# Iteration schedule
# =============================================================================

def get_active_iters(step, total_steps, max_iters, warmup_frac):
    if max_iters <= 1:
        return 1
    warmup_steps = int(total_steps * warmup_frac)
    if warmup_steps <= 0 or step >= warmup_steps:
        return max_iters
    progress = step / warmup_steps
    return 1 + int((max_iters - 1) * progress)

# =============================================================================
# Training
# =============================================================================

ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
master_process = ddp_rank == 0
torch.manual_seed(42)

if ddp and torch.cuda.is_available():
    device = torch.device("cuda", ddp_local_rank)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(42)
    dist.init_process_group(backend="nccl")
    dist.barrier()
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device_type = device.type
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

gpu_peak_flops = float('inf')
if device_type == "cuda":
    gpu_name = torch.cuda.get_device_name(0).lower()
    if "h100" in gpu_name: gpu_peak_flops = 989e12
    elif "a100" in gpu_name: gpu_peak_flops = 312e12
    elif "4090" in gpu_name: gpu_peak_flops = 165.2e12

run_name = args.run if args.run else f"lsrecurrent_{time.strftime('%Y%m%d_%H%M%S')}"
_wandb_kwargs = {"project": "slowrun", "name": run_name}
if args.wandb_group:
    _wandb_kwargs["group"] = args.wandb_group
wandb_run = DummyWandb() if (not master_process or wandb is None) else wandb.init(**_wandb_kwargs)
if master_process:
    wandb_run.log_code(".")

print0(f"--- LSRecurrent Hyperparameters ---")
print0(f"  config={args.config or 'custom'}")
print0(f"  n_layer={N_LAYER}, n_embd={N_EMBD}, h_dim={H_DIM}, n_iter={N_ITER}")
print0(f"  seq_len={MAX_SEQ_LEN}")
print0(f"  stoch_depth={args.stoch_depth}")
print0(f"  total_batch_size={TOTAL_BATCH_SIZE}, device_batch_size={DEVICE_BATCH_SIZE}")
print0(f"  matrix_lr={MATRIX_LR}, scalar_lr={SCALAR_LR}, embedding_lr={EMBEDDING_LR}, unembedding_lr={UNEMBEDDING_LR}")
print0(f"  weight_decay={WEIGHT_DECAY}, adam_betas={ADAM_BETAS}")
print0(f"  warmup_ratio={WARMUP_RATIO}, warmdown_ratio={WARMDOWN_RATIO}")
print0(f"  iter_warmup_frac={args.iter_warmup_frac}")
print0(f"  num_epochs={NUM_EPOCHS}, patience={args.patience}")
print0(f"  dropout={args.dropout}")
print0(f"  ls_enabled={LS_ENABLED}, ls_blocks={LS_NUM_BLOCKS}, ls_rank={LS_RANK}")
print0(f"----------------------------------")

encoder = tiktoken.get_encoding("gpt2")
vocab_size = encoder.n_vocab
print0(f"Vocab size: {vocab_size:,}")

eot_id = encoder._special_tokens['<|endoftext|>']
token_bytes_list = []
for i in range(vocab_size):
    if i == eot_id:
        token_bytes_list.append(0)
    else:
        token_bytes_list.append(len(encoder.decode_single_token_bytes(i)))
token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32, device=device)

config = LSRecurrentConfig(vocab_size=vocab_size, dropout=args.dropout,
                           stoch_depth=args.stoch_depth)
with torch.device("meta"):
    model = LSRecurrentLM(config)
model.to_empty(device=device)
model.init_weights()

param_counts = sum(p.numel() for p in model.parameters())
block_params = sum(p.numel() for p in model.blocks.parameters())
embed_params_count = sum(p.numel() for p in model.wte.parameters()) + sum(p.numel() for p in model.wpe.parameters())
lm_head_params_count = sum(p.numel() for p in model.lm_head.parameters())
print0(f"Parameters: {param_counts:,} (blocks: {block_params:,}, embeddings: {embed_params_count:,}, lm_head: {lm_head_params_count:,})")

ls_params = sum(m.sparse_weight.numel() + m.A.numel() + m.B.numel()
                for m in model.modules() if isinstance(m, LSLinearCompiled))
dense_linear_params = sum(p.numel() for m in model.modules()
                          if isinstance(m, nn.Linear) for p in m.parameters())
if ls_params > 0:
    print0(f"LSLinear: {ls_params:,} params | Dense nn.Linear: {dense_linear_params:,}")
    print0(f"LSLinear config: blocks={LS_NUM_BLOCKS}, rank={LS_RANK}")

# Estimate FLOPs per token (approximate: 6 * non-embedding params per token, times K iterations)
non_embed_params = param_counts - embed_params_count
num_flops_per_token = 6 * non_embed_params * N_ITER
print0(f"FLOPs per token (K={N_ITER}): {num_flops_per_token:e}")

orig_model = model
model = torch.compile(model, dynamic=False)

optimizer = model.setup_optimizer()

_train_path = args.input_bin if args.input_bin else os.path.join(DATA_DIR, "fineweb_train.pt")
_val_path = args.input_val_bin if args.input_val_bin else os.path.join(DATA_DIR, "fineweb_val.pt")
train_loader = DataLoader(_train_path, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, device=device)
build_val_loader = lambda: DataLoader(_val_path, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, device=device)
TOKENS_PER_EPOCH = train_loader.total_tokens
x, y, current_epoch = next(train_loader)

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN * ddp_world_size
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0, \
    f"total_batch_size ({TOTAL_BATCH_SIZE}) must be divisible by tokens_per_fwdbwd ({tokens_per_fwdbwd})"
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
num_iterations = round(TOKENS_PER_EPOCH * NUM_EPOCHS / TOTAL_BATCH_SIZE)
print0(f"Batch size: {TOTAL_BATCH_SIZE:,} tokens, grad accum: {grad_accum_steps} steps")
print0(f"Training for {NUM_EPOCHS} epoch(s) (~{num_iterations} steps estimated)")
print0(f"Eval set: {EVAL_TOKENS:,} tokens")

def get_lr_multiplier(it):
    warmup = round(WARMUP_RATIO * num_iterations)
    warmdown = round(WARMDOWN_RATIO * num_iterations)
    if warmup > 0 and it < warmup:
        return (it + 1) / warmup
    elif it <= num_iterations - warmdown:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown
        return progress + (1 - progress) * FINAL_LR_FRAC

def get_muon_momentum(it):
    return (1 - min(it / 300, 1)) * 0.85 + min(it / 300, 1) * 0.95

steps_per_epoch = num_iterations / NUM_EPOCHS
_swa_start_step = (num_iterations - args.swa_last_epochs * steps_per_epoch) if args.swa_last_epochs > 0 else -1

step = 0
min_val_bpb = float("inf")
min_val_loss = float("inf")
epochs_without_improvement = 0
smooth_train_loss = 0
total_training_time = 0
timed_steps = 0
timing_start_step = 4
eval_steps = EVAL_TOKENS // (DEVICE_BATCH_SIZE * MAX_SEQ_LEN * ddp_world_size)
current_active_k = 1 if N_ITER > 1 and args.iter_warmup_frac > 0 else N_ITER
orig_model.active_k = current_active_k

late_checkpoint_paths = []
logit_avg_count = args.logit_avg
if logit_avg_count > 0 and master_process:
    os.makedirs(args.logit_avg_dir, exist_ok=True)
if logit_avg_count > 0:
    print0(f"Logit averaging: saving last {logit_avg_count} epoch checkpoints to {args.logit_avg_dir}/")

if args.eval_logit_avg:
    print0("--eval-logit-avg set: skipping training, loading checkpoints from disk.")
else:
    model.eval()
    val_loader = build_val_loader()
    with autocast_ctx:
        val_bpb, val_loss = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
    print0(f"Step {step:05d} | Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f}")
    wandb_run.log({"step": step, "val/bpb": val_bpb, "val/loss": val_loss})
    min_val_bpb = val_bpb
    min_val_loss = val_loss
    model.train()

print0(f"Starting with active_k={current_active_k} (max={N_ITER})")

while not args.eval_logit_avg and current_epoch <= NUM_EPOCHS:
    # Iteration schedule: ramp K over training
    new_k = get_active_iters(step, num_iterations, N_ITER, args.iter_warmup_frac)
    if new_k != current_active_k:
        print0(f"\n=== Iteration schedule: K {current_active_k} -> {new_k} at step {step} ===")
        current_active_k = new_k
        orig_model.active_k = new_k
        model = torch.compile(orig_model, dynamic=False)
        timing_start_step = step + 4
        gc.enable(); gc.collect()

    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss, metrics = model(x, y)
        train_loss = loss.detach()
        (loss / grad_accum_steps).backward()
        x, y, epoch = next(train_loader)

    lrm = get_lr_multiplier(step)
    if _swa_start_step >= 0 and step >= _swa_start_step:
        cycle_pos = (step - _swa_start_step) % steps_per_epoch
        swa_base = max(lrm, 0.05)
        lrm = 0.05 + (swa_base - 0.05) * (1 + math.cos(math.pi * cycle_pos / steps_per_epoch)) / 2
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = get_muon_momentum(step)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()
    synchronize()
    dt = time.time() - t0

    step += 1

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased = smooth_train_loss / (1 - ema_beta**step)
    pct = 100 * step / num_iterations
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / (gpu_peak_flops * ddp_world_size)
    if step >= timing_start_step:
        total_training_time += dt
        timed_steps += 1
    eta_str = f" | eta: {(num_iterations - step) * total_training_time / timed_steps / 60:.1f}m" if timed_steps > 0 else ""
    print0(f"step {step:05d} ({pct:.2f}%) | loss: {debiased:.6f} | K={current_active_k} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f}%{eta_str}")
    wandb_run.log({"step": step, "train/loss": debiased, "train/mfu": mfu,
                   "train/active_k": current_active_k,
                   **{f"train/{k}": v.item() for k, v in metrics.items()}})

    if ddp:
        epoch_tensor = torch.tensor([epoch], dtype=torch.long, device=device)
        dist.all_reduce(epoch_tensor, op=dist.ReduceOp.MAX)
        epoch = epoch_tensor.item()

    if epoch != current_epoch:
        model.eval()
        val_loader = build_val_loader()
        with autocast_ctx:
            val_bpb, val_loss = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Epoch {current_epoch} | Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f}")
        wandb_run.log({"step": step, "epoch": current_epoch, "val/bpb": val_bpb, "val/loss": val_loss})

        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
            min_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.patience >= 0 and epochs_without_improvement >= args.patience:
                print0(f"Early stopping: no improvement for {args.patience} epoch(s)")
                break

        if logit_avg_count > 0:
            ckpt_path = os.path.join(args.logit_avg_dir, f"lsrecurrent_epoch_{current_epoch:03d}.pt")
            if master_process:
                ckpt = {name: p.data.float().cpu() for name, p in orig_model.named_parameters()}
                torch.save(ckpt, ckpt_path)
                del ckpt
            late_checkpoint_paths.append(ckpt_path)
            if len(late_checkpoint_paths) > logit_avg_count:
                old = late_checkpoint_paths.pop(0)
                if master_process and os.path.exists(old):
                    os.remove(old)
            print0(f"  Saved checkpoint {ckpt_path} ({len(late_checkpoint_paths)}/{logit_avg_count})")

        model.train()
        current_epoch = epoch

    if step == 1:
        gc.collect(); gc.freeze(); gc.disable()

# =============================================================================
# Post-training: logit average evaluation
# =============================================================================

if logit_avg_count > 0:
    if args.eval_logit_avg:
        import glob as _glob
        all_disk = sorted(_glob.glob(os.path.join(args.logit_avg_dir, "lsrecurrent_epoch_*.pt")))
        ckpt_paths_for_logit = all_disk[-logit_avg_count:]
    else:
        ckpt_paths_for_logit = late_checkpoint_paths

    if len(ckpt_paths_for_logit) >= 2:
        n = len(ckpt_paths_for_logit)
        print0(f"\n--- Evaluating logit avg ({n} checkpoints: {[os.path.basename(p) for p in ckpt_paths_for_logit]}) ---")

        la_model = torch.compile(orig_model, dynamic=False)
        la_model.eval()

        def _run_mode(label, weights):
            print0(f"  [{label}] weights: {[f'{w:.3f}' for w in weights]}")
            bpb, loss = evaluate_bpb_logit_avg(la_model, ckpt_paths_for_logit, weights, eval_steps)
            print0(f"  [{label}] Val BPB: {bpb:.6f} | Val Loss: {loss:.6f}")
            wandb_run.log({f"logit_avg_{label}/bpb": bpb, f"logit_avg_{label}/loss": loss})
            return bpb, loss

        equal_w = [1.0 / n] * n
        raw_w = list(range(1, n + 1))
        weighted_w = [w / sum(raw_w) for w in raw_w]

        if args.logit_avg_mode in ("equal", "both"):
            eq_bpb, eq_loss = _run_mode("equal", equal_w)
            if eq_loss < min_val_loss:
                min_val_loss, min_val_bpb = eq_loss, eq_bpb
                print0(f"  ** New best! (logit avg equal weights)")

        if args.logit_avg_mode in ("weighted", "both"):
            wt_bpb, wt_loss = _run_mode("weighted", weighted_w)
            if wt_loss < min_val_loss:
                min_val_loss, min_val_bpb = wt_loss, wt_bpb
                print0(f"  ** New best! (logit avg recency weights)")

# Summary
print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.2f} MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
final_train_loss = smooth_train_loss / (1 - 0.9**step) if step > 0 else float('inf')
print0(f"Final train loss: {final_train_loss:.6f}")
print0(f"Min val BPB: {min_val_bpb:.6f}")
print0(f"Min val Loss: {min_val_loss:.6f}")
wandb_run.summary["final_train_loss"] = final_train_loss
wandb_run.summary["best_val_loss"] = min_val_loss

if args.save_result and master_process:
    result = {
        "model": "lsrecurrent",
        "config": args.config or "custom",
        "n_layer": N_LAYER,
        "n_embd": N_EMBD,
        "h_dim": H_DIM,
        "n_iter": N_ITER,
        "matrix_lr": args.matrix_lr,
        "weight_decay": args.weight_decay,
        "num_epochs": NUM_EPOCHS,
        "val_loss": val_loss if 'val_loss' in dir() else min_val_loss,
        "best_val_loss": min_val_loss,
        "params": param_counts,
        "wandb_url": getattr(wandb_run, "url", None),
    }
    with open(args.save_result, "w") as f:
        json.dump(result, f, indent=2)
    print0(f"Result saved to {args.save_result}")

total_wall_time = time.time() - _script_start
print0(f"Total wall time: {total_wall_time:.2f}s ({total_wall_time/60:.2f}m)")

wandb_run.finish()
if dist.is_initialized():
    dist.destroy_process_group()
