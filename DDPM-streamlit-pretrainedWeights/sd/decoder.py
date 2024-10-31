import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        """
        Initializes an attention block with Group Normalization and Self-Attention layer.
        Args:
            channels (int): Number of input channels/features in the block.
        """
        super().__init__()
        # Normalizes across channels for stable training with smaller batch sizes
        self.groupnorm = nn.GroupNorm(32, channels)
        # Applies self-attention to learn relationships between spatial locations
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x):
        """
        Forward pass through the attention block.
        Args:
            x (Tensor): Input tensor of shape (Batch_Size, Features, Height, Width).
        Returns:
            Tensor: Output tensor after applying attention and adding the residual connection.
        """
        # Residual connection for combining original input with attention output
        residue = x

        # Normalize the input
        x = self.groupnorm(x)
        n, c, h, w = x.shape

        # Flatten spatial dimensions for attention mechanism
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)  # Transform into (Batch_Size, Height * Width, Features)

        # Apply self-attention without masking
        x = self.attention(x)

        # Reshape back to original dimensions
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        # Add the residual connection
        x += residue

        return x 

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initializes a residual block with Group Normalization, SiLU activation, and Convolutional layers.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        # Group normalization to stabilize training
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        # First convolution layer that preserves spatial dimensions
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        # Second convolution layer, also preserving spatial dimensions
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # If the input and output channels differ, use a 1x1 convolution to match dimensions
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        """
        Forward pass through the residual block.
        Args:
            x (Tensor): Input tensor of shape (Batch_Size, In_Channels, Height, Width).
        Returns:
            Tensor: Output tensor after residual connections and convolutions.
        """
        residue = x

        # Normalize and apply activation
        x = self.groupnorm_1(x)
        x = F.silu(x)

        # First convolution operation
        x = self.conv_1(x)

        # Normalize and apply activation again
        x = self.groupnorm_2(x)
        x = F.silu(x)

        # Second convolution operation
        x = self.conv_2(x)

        # Add residual connection, either using the identity or a convolution layer to match dimensions
        return x + self.residual_layer(residue)

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        """
        Initializes the decoder as a sequence of residual and attention blocks with upsampling operations.
        The decoder gradually upsamples the latent feature map back to the original image resolution.
        """
        super().__init__(
            # Scale back the initial feature map size with a 1x1 convolution
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            # Start with a 3x3 convolution to expand channels from 4 to 512
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            # Apply residual blocks and attention blocks, keeping channels constant at 512
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            # Upsample: doubles the spatial dimensions
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            # Additional residual blocks
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            # Second upsample operation
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            
            # Decrease channels from 512 to 256 and apply residual blocks
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            
            # Third upsample to reach the target resolution
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            
            # Further reduce channels to 128 and apply residual blocks
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            
            # Final group normalization and activation
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            
            # Output layer with 3 channels for RGB
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
        Forward pass through the decoder.
        Args:
            x (Tensor): Input tensor of shape (Batch_Size, 4, Height / 8, Width / 8).
        Returns:
            Tensor: Decoded image tensor of shape (Batch_Size, 3, Height, Width).
        """
        # Undo the scaling applied during encoding
        x /= 0.18215

        # Apply each layer in the decoder sequentially
        for module in self:
            x = module(x)

        return x
