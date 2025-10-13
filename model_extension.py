"""
LLaMA-style Transformer with RoPE implementation based on nanoGPT conventions.
Implements Rotary Position Embedding (RoPE) and LLaMA architecture patterns.
"""

import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


class RoPE(nn.Module):
    """Rotary Position Embedding implementation"""
    
    def __init__(self, dim, max_seq_len=8192, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for cos and sin values
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cos_sin_cache(self, seq_len, device, dtype):
        """Update cached cos/sin values if sequence length changed"""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
    
    def rotate_half(self, x):
        """Rotates half the hidden dims of the input"""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q, k, seq_len):
        """Apply rotary position embedding to query and key tensors"""
        self._update_cos_sin_cache(seq_len, q.device, q.dtype)
        cos = self._cos_cached[:seq_len, ...].unsqueeze(0).unsqueeze(0)  # (1, 1, T, hs)
        sin = self._sin_cached[:seq_len, ...].unsqueeze(0).unsqueeze(0)  # (1, 1, T, hs)

        # Apply to query (B, nh, T, hs)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        # Apply to key (B, nh, T, hs)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)

        return q_embed, k_embed


class LLaMAAttention(nn.Module):
    """LLaMA-style attention with RoPE"""
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # Query, key, value projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # RoPE
        self.rope = RoPE(self.head_dim, max_seq_len=config.block_size, base=config.rope_base)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Causal mask for non-flash attention
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()  # batch, sequence, embedding
        
        # Calculate query, key, value
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        
        # Apply RoPE to query and key
        q, k = self.rope.apply_rotary_pos_emb(q, k, T)
        
        # Causal self-attention
        if self.flash:
            # Use Flash Attention
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0, 
                is_causal=True
            )
        else:
            # Manual attention implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.o_proj(y))
        
        return y


class LLaMAMLP(nn.Module):
    """LLaMA-style MLP with SwiGLU activation"""
    
    def __init__(self, config):
        super().__init__()
        # LLaMA uses different intermediate size calculation
        intermediate_size = int(2 * config.n_embd * 4 / 3)
        # Round to nearest multiple of 256 for efficiency
        intermediate_size = ((intermediate_size + 255) // 256) * 256
        
        self.gate_proj = nn.Linear(config.n_embd, intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(config.n_embd, intermediate_size, bias=config.bias)  
        self.down_proj = nn.Linear(intermediate_size, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        # SwiGLU activation: SiLU(gate) * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization used in LLaMA"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class LLaMABlock(nn.Module):
    """LLaMA-style transformer block"""
    
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.n_embd)
        self.self_attn = LLaMAAttention(config)
        self.post_attention_layernorm = RMSNorm(config.n_embd)
        self.mlp = LLaMAMLP(config)
    
    def forward(self, x):
        # Pre-norm residual connections (LLaMA style)
        h = x + self.self_attn(self.input_layernorm(x))
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


@dataclass
class LLaMAConfig:
    """Configuration for LLaMA-style model"""
    block_size: int = 2048
    vocab_size: int = 32000  # LLaMA vocab size
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False  # LLaMA doesn't use bias
    rope_base: float = 10000.0
    

class LLaMA(nn.Module):
    """LLaMA-style transformer model with RoPE"""
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings only
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([LLaMABlock(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),  # Final RMS norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying (optional, depends on LLaMA variant)
        # self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply scaled init to output projections
        for pn, p in self.named_parameters():
            if pn.endswith('o_proj.weight') or pn.endswith('down_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # Forward the LLaMA model - no positional embeddings, RoPE handles positions
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)
        
        # Apply transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            # Training: calculate loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference: only forward the lm_head on the last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        
        return logits, loss
    
    def crop_block_size(self, block_size):
        """Crop the block size if necessary"""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        # Update RoPE cache for all attention layers
        for block in self.transformer.h:
            block.self_attn.rope.max_seq_len = block_size
            block.self_attn.rope._seq_len_cached = 0  # Reset cache
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Configure AdamW optimizer following the same pattern as GPT"""
        # Get all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # Create optim groups: 2D parameters get weight decay, others don't
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Use fused AdamW if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        
        return optimizer
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens using the trained model.
        Same interface as the original GPT generate method.
        """
        for _ in range(max_new_tokens):
            # Crop context if it gets too long
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # Forward pass
            logits, _ = self(idx_cond)
            # Get logits for the last token and apply temperature
            logits = logits[:, -1, :] / temperature
            # Optionally crop to top k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx