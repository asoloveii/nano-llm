import inspect
from typing import Optional, Tuple
from dataclasses import dataclass

import torch 
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NanoConfig:
    """
    Configuration for NanoLM.

    Attributes:
        max_batch_size (int): Maximum batch size supported.
        max_seq_len (int): Maximum sequence length supported.
        vocab_size (int): Vocabulary size.
        d_model (int): Embedding/hidden dimension size.
        d_hidden (int): Hidden layer for SwiGLU FFN.
        n_heads (int): Number of attention heads.
        n_layers (int): Number of transformer blocks.
        theta (float): RoPE frequency base.
        n_activated_experts (int): Number of experts activated per token.
        kv_latent_dim (int): Latent dimension for keys and values in MLA.
        q_latent_dim (int): Latent dimension for queries in MLA.
        qk_nope_head_dim (int): Dimension of query/key without positional encoding.
        qk_rope_head_dim (int): Dimension of query/key with rotary positional encoding.
        v_head_dim (int): Dimension of values per head.
    """
    max_batch_size: int = 32
    max_seq_len: int = 2048
    vocab_size: int = 50304
    d_model: int = 768
    d_hidden: int = 3072
    n_heads: int = 12
    n_layers: int = 20
    theta: float = 10000.0
    eps: float = 1e-5
    # MLA
    kv_latent_dim: int = 128
    q_latent_dim: int = 128
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128


class RMSNorm(nn.Module):

    def __init__(self, 
                 d_model: int, 
                 eps: float = 1e-5):
        '''
        RMSNorm module. 

        Args:
            d_model (int): Hidden dimension of the model
            eps (float): 1e-5 Epsilon value for numerical stability
        '''
        super().__init__()

        # initialize trained gain parameter
        self.gain = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        RMSNorm forward pass.
        
        Args:
            x (torch.Tensor): input tensor

        Returns:
            result (torch.Tensor): normalized x
        '''
        # save original dtype
        in_dtype = x.dtype 
        x = x.to(torch.float32) # upcast to prevent overflow

        # calculate the result
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = (x / rms) * self.gain  

        # return in original dtype
        return result.to(in_dtype)


class RoPE(nn.Module):
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        '''
        Rotary Positional Embedding Layer.

        Args:
            theta (float): Θ value for the RoPE
            d_k (int): dimension of query and key vectors
            max_seq_len (int): maximum sequence length that will be inputted
        '''
        super().__init__()

        positions = torch.arange(max_seq_len).unsqueeze(1)
        freqs = torch.arange(0, d_k, 2) / d_k
        inv_freq = 1.0 / (theta ** freqs)
        angles = positions * inv_freq

        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional embedding to input tensor.

        Args:
            x: Tensor of shape (..., seq_len, d_k)
            token_positions: Tensor of shape (..., seq_len)

        Returns:
            Tensor of same shape as input, with RoPE applied.
        """
        if token_positions.ndim == 1:  # [seq_len]
            cos_pos = self.cos[token_positions].unsqueeze(0)  # [1, seq_len, d_k//2]
            sin_pos = self.sin[token_positions].unsqueeze(0)
        else:  # [batch, seq_len]
            cos_pos = self.cos[token_positions]  # [batch, seq_len, d_k//2]
            sin_pos = self.sin[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rot_even = x_even * cos_pos - x_odd * sin_pos
        x_rot_odd = x_even * sin_pos + x_odd * cos_pos

        x_rot = torch.zeros_like(x)
        x_rot[..., 0::2] = x_rot_even
        x_rot[..., 1::2] = x_rot_odd

        return x_rot


class MLA(nn.Module):

    def __init__(self, args: NanoConfig):
        '''Multi-Head Latent Attention.'''
        super().__init__()
        # basic parameters
        self.d_model = args.d_model
        self.n_heads = args.n_heads

        # dimension after latent (compressed) projections
        self.kv_latent_dim = args.kv_latent_dim
        self.q_latent_dim = args.q_latent_dim

        # head dimensions for queries, keys and values
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        # linear projections into compressed latent vectors
        self.down_proj_kv = nn.Linear(self.d_model, self.kv_latent_dim)
        self.down_proj_q = nn.Linear(self.d_model, self.q_latent_dim)

        # up projections from latent space
        self.up_proj_k_nope = nn.Linear(self.kv_latent_dim, self.qk_nope_head_dim * self.n_heads)
        self.up_proj_v = nn.Linear(self.kv_latent_dim, self.v_head_dim * self.n_heads)
        self.up_proj_q_nope = nn.Linear(self.q_latent_dim, self.qk_nope_head_dim * self.n_heads)

        # query and key projections for rotary positional embeddings
        self.up_proj_q_rope = nn.Linear(self.q_latent_dim, self.qk_rope_head_dim * self.n_heads)
        self.up_proj_k_rope = nn.Linear(self.d_model, self.qk_rope_head_dim * self.n_heads)

        # final output projection
        self.out_proj = nn.Linear(self.n_heads * self.v_head_dim, self.d_model)

        # cache KV compressed and key rope 
        self.register_buffer("kv_latent", torch.zeros(args.max_batch_size, 
                                                      args.max_seq_len, 
                                                      args.kv_latent_dim), persistent=False)
        self.register_buffer("k_rope", torch.zeros(args.max_batch_size, 
                                                   args.max_seq_len,
                                                   self.qk_rope_head_dim * self.n_heads), persistent=False)

    def forward(self, 
                x: torch.Tensor, 
                rope: Optional[RoPE] = None,
                token_positions: Optional[torch.Tensor] = None,
                current_pos: int = 0,
                use_cache: bool = False) -> torch.Tensor:
        '''
        Forward pass for Multi-Head Latent Attention.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, d_model).
            rope (RoPE, optional): Rotary positional embedding module.
            token_positions (Tensor, optional): Position indices.
            current_pos (int): Current position.
            use_cache (bool): Using caching for faster computation.

        Returns:
            Tensor: Output tensor of shape (batch, seq_len, d_model).
        '''
        batch_size, seq_len, _ = x.shape
        
        # compressed projections
        C_kv = self.down_proj_kv(x) # (bsz, seq, kv_latent_dim)
        C_q = self.down_proj_q(x)   # (bsz, seq, q_latent_dim)

        if use_cache:
            self.kv_latent[:batch_size, current_pos:current_pos+seq_len] = C_kv.detach()

        # restore query, key and value matrices from latent vector
        q_nope = self.up_proj_q_nope(C_q)   # (bsz, seq, qk_nope_head_dim * n_heads)
        k_nope = self.up_proj_k_nope(C_kv)  # (bsz, seq, qk_nope_head_dim * n_heads) 
        v = self.up_proj_v(C_kv)            # (bsz, seq, v_head_dim * n_heads)

        # query and key projections, which will further be used for rope
        q_proj_rope = self.up_proj_q_rope(C_q)
        k_proj_rope = self.up_proj_k_rope(x)

        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            # apply rope to queries and keys, shape: (bsz, seq, qk_rope_head_dim * n_heads)
            q_rope = rope(q_proj_rope, token_positions) 
            k_rope = rope(k_proj_rope, token_positions)
            if use_cache:
                self.k_rope[:batch_size, current_pos:current_pos+seq_len] = k_rope.detach()

        # concatenate nope and rope queries and keys, shape: (bsz, seq, qk_head_dim * n_heads)
        q = torch.concat([q_nope, q_rope], dim=-1)
        k = torch.concat([k_nope, k_rope], dim=-1)

        q = q.view(batch_size, seq_len, self.n_heads, self.qk_head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.qk_head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.v_head_dim).transpose(1, 2)

        scores = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (bsz, heads, seq, v_dim)
        scores = scores.transpose(1, 2).contiguous()                      # (bsz, seq, heads, v_dim)
        scores = scores.view(batch_size, seq_len, self.n_heads * self.v_head_dim)
        out = self.out_proj(scores)

        return out


class SwiGLU(nn.Module):
    
    def __init__(self, d_model: int, d_hidden: int):
        '''
        Swish Gated Linear Unit, composed of a SiLU activation and a GLU.
        
        Args:
            d_model (int): Input/output dimension.
            d_hidden (int): Hidden dimension.
        '''
        super().__init__()

        self.w1 = nn.Linear(d_model, d_hidden)
        self.w2 = nn.Linear(d_model, d_hidden)
        self.w3 = nn.Linear(d_hidden, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Apply SwiGLU layer.
        
        Args:
            x (Tensor): Input tensor of shape (..., d_model).

        Returns:
            Tensor: Output tensor of shape (..., d_model).
        '''
        # first linear projection + SiLU activation
        proj1 = self.w1(x)
        silu = F.silu(proj1)

        # gated linear unit
        gate = self.w2(x)
        glu = silu * gate

        # apply final linear projection
        out = self.w3(glu)

        return out


class Block(nn.Module):

    def __init__(self, args: NanoConfig):
        '''
        A Transformer block contains two ‘sublayers’, one for the multihead latent attention,
        and another for the mixture of experts block.  
        norm -> mla -> norm -> moe
        ''' 
        super().__init__()

        self.norm1 = RMSNorm(args.d_model)
        self.rope = RoPE(args.theta, args.qk_rope_head_dim * args.n_heads, args.max_seq_len)
        self.mla = MLA(args)

        self.norm2 = RMSNorm(args.d_model)
        self.ffn = SwiGLU(args.d_model, args.d_hidden)

    def forward(self, 
                x: torch.Tensor, 
                token_positions: Optional[torch.Tensor] = None,
                current_pos: int = 0,
                use_cache: bool = False) -> torch.Tensor:
        '''
        Apply transformer block.
        
        Args:
            x (Tensor): Input tensor (batch, seq, d_model).
            token_positions (Tensor, optional): Position indices.
            current_pos (int): Current position.
            use_cache (bool): Using caching for faster computation.

        Returns:
            Tensor: Output tensor (batch, seq, d_model).
        '''
        x = x + self.mla(self.norm1(x), self.rope, token_positions, current_pos, use_cache)
        x = x + self.ffn(self.norm2(x))
        return x


class Nano(nn.Module):
    
    def __init__(self, args: NanoConfig):
        '''
        Nano language model.
        
        Args:
            args (NanoConfig): Model configuration.
        '''
        super().__init__()

        self.input_emb = nn.Embedding(args.vocab_size, args.d_model)

        self.blocks = nn.ModuleList([Block(args) for _ in range(args.n_layers)])

        self.final_norm = RMSNorm(args.d_model)
        self.out_proj = nn.Linear(args.d_model, args.vocab_size, bias=False)

        self._init_weights(self)

    def forward(self, ids: torch.LongTensor, past_len: int = 0, use_cache: bool = False) -> torch.Tensor:
        '''
        Forward pass for next token prediction.
        
        Args:
            ids (Tensor): Input token IDs (batch, seq).
            past_len (int, optional): Number of tokens already cached in KV memory from previous steps.
            use_cache (bool, optional): Using caching in MLA for faster computation. 

        Returns:
            Tensor: Logits (batch, seq, vocab_size).
        '''
        x = self.input_emb(ids)
        token_positions = torch.arange(past_len, past_len + ids.shape[1], device=ids.device)

        # iterate over all transformer blocks
        for block in self.blocks:
            x = block(x, token_positions, past_len, use_cache)

        # output predictions
        x = self.final_norm(x)
        out = self.out_proj(x)

        return out
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.inference_mode()
    def generate(self,
                 ids: torch.Tensor,
                 max_new_tokens: int = 50,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            ids (Tensor): initial tokens of shape (batch, seq).
            max_new_tokens (int): number of tokens to generate.
            temperature (float): sampling temperature (>0).
            top_k (int, optional): restrict sampling to top-k tokens.
            eos_token_id (int, optional): stop generation if this token is produced.

        Returns:
            Tensor: Generated sequence of shape (batch, seq + max_new_tokens).
        """
        idx = ids
        bsz = idx.shape[0]

        # Reset KV caches before generation
        for block in self.blocks:
            block.mla.kv_latent.zero_()
            block.mla.k_rope.zero_()

        for t in range(max_new_tokens):
            logits = self(idx[:, -1:], past_len=idx.shape[1]-1, use_cache=True)  # feed last token
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break

        return idx
    
    def configure_optimizer(self, 
                             weight_decay: float, 
                             learning_rate: float, 
                             betas: Tuple[float, float]) -> torch.optim.Optimizer:
        '''Configures AdamW optimizer.'''
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"Num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and torch.cuda.is_available()
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        
        print(f"using fused AdamW: {use_fused}")

        return optimizer
