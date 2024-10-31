import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        
        # Embedding layer for token indices to embeddings
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        # Learnable positional embeddings for sequence positions
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))
    
    def forward(self, tokens):
        # Convert token indices to embeddings
        # (batch_size, seq_len) -> (batch_size, seq_len, n_embd)
        x = self.token_embedding(tokens)
        # Add position embeddings to token embeddings
        # (batch_size, seq_len, n_embd)
        x += self.position_embedding
        
        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        
        # Layer normalization before attention
        self.layernorm_1 = nn.LayerNorm(n_embd)
        # Self-attention module
        self.attention = SelfAttention(n_head, n_embd)
        # Layer normalization before feedforward network (FFN)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        # First linear layer in the FFN with a higher dimension
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        # Second linear layer in the FFN back to the embedding dimension
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        # Preserve input for residual connection
        residue = x

        ### SELF-ATTENTION ###
        # Normalize input and apply self-attention with causal masking
        # (batch_size, seq_len, n_embd)
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        
        # Add residual connection
        x += residue

        ### FEEDFORWARD LAYER ###
        # Store intermediate result for residual connection
        residue = x
        # Normalize input
        x = self.layernorm_2(x)
        
        # Apply first linear layer and QuickGELU activation
        # (batch_size, seq_len, 4 * n_embd)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702 * x)  # QuickGELU activation function
        
        # Project back to embedding dimension
        # (batch_size, seq_len, n_embd)
        x = self.linear_2(x)
        
        # Add residual connection
        x += residue

        return x

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Embedding layer for input tokens
        self.embedding = CLIPEmbedding(49408, 768, 77)
        
        # Stack of transformer layers
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for _ in range(12)
        ])
        
        # Final layer normalization
        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # Ensure tokens are in the correct dtype
        tokens = tokens.type(torch.long)
        
        # Get embeddings with positional information
        # (batch_size, seq_len, n_embd)
        state = self.embedding(tokens)

        # Pass through each transformer layer
        for layer in self.layers:
            # (batch_size, seq_len, n_embd)
            state = layer(state)
        
        # Apply final normalization
        output = self.layernorm(state)
        
        return output
