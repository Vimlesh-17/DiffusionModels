import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # Combine Wq, Wk, and Wv into a single linear transformation for efficiency
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # Define the output projection (Wo)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        
        self.n_heads = n_heads
        # Dimension per head, assuming d_embed is divisible by n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        # x: Input tensor of shape (batch_size, seq_len, d_embed)

        # Store input dimensions
        batch_size, seq_len, d_embed = x.shape

        # Calculate query, key, and value by splitting the result of a single linear transformation
        # (batch_size, seq_len, 3 * d_embed) -> 3 tensors of shape (batch_size, seq_len, d_embed)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # Reshape tensors to prepare for multi-head processing
        # (batch_size, seq_len, d_embed) -> (batch_size, seq_len, n_heads, d_head)
        # Transpose for head-first attention computation: (batch_size, n_heads, seq_len, d_head)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention: (batch_size, n_heads, seq_len, d_head) @ (batch_size, n_heads, d_head, seq_len)
        # -> (batch_size, n_heads, seq_len, seq_len)
        attn_scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)

        # Apply causal mask if specified
        if causal_mask:
            # Create an upper triangular mask to prevent attending to future positions
            mask = torch.ones_like(attn_scores, dtype=torch.bool).triu(1)
            # Apply mask to the attention scores
            attn_scores.masked_fill_(mask, float('-inf'))

        # Apply softmax to compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute attention output
        # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, d_head)
        # -> (batch_size, n_heads, seq_len, d_head)
        output = attn_weights @ v

        # Rearrange to original tensor layout and combine heads
        # (batch_size, n_heads, seq_len, d_head) -> (batch_size, seq_len, n_heads, d_head) -> (batch_size, seq_len, d_embed)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, d_embed)

        # Apply the final linear projection
        return self.out_proj(output)

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # Linear projections for query, key, and value
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        # Output projection
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        
        self.n_heads = n_heads
        # Dimension per head
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        # x: Latent tensor (batch_size, seq_len_q, d_embed)
        # y: Context tensor (batch_size, seq_len_kv, d_cross)

        # Get input dimensions
        batch_size, seq_len_q, d_embed = x.shape

        # Project x and y to query, key, and value tensors
        q = self.q_proj(x)  # (batch_size, seq_len_q, d_embed)
        k = self.k_proj(y)  # (batch_size, seq_len_kv, d_embed)
        v = self.v_proj(y)  # (batch_size, seq_len_kv, d_embed)

        # Reshape for multi-head attention and transpose
        # (batch_size, seq_len, d_embed) -> (batch_size, seq_len, n_heads, d_head)
        q = q.view(batch_size, seq_len_q, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        # Calculate attention scores
        # (batch_size, n_heads, seq_len_q, d_head) @ (batch_size, n_heads, d_head, seq_len_kv) -> (batch_size, n_heads, seq_len_q, seq_len_kv)
        attn_scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)

        # Apply softmax to obtain attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Compute the weighted sum of values
        # (batch_size, n_heads, seq_len_q, seq_len_kv) @ (batch_size, n_heads, seq_len_kv, d_head)
        # -> (batch_size, n_heads, seq_len_q, d_head)
        output = attn_weights @ v

        # Reshape to merge heads back into the original dimensions
        # (batch_size, n_heads, seq_len_q, d_head) -> (batch_size, seq_len_q, n_heads, d_head) -> (batch_size, seq_len_q, d_embed)
        output = output.transpose(1, 2).reshape(batch_size, seq_len_q, d_embed)

        # Apply output projection
        return self.out_proj(output)
