import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

# TimeEmbedding class is responsible for transforming a time step into a higher-dimensional embedding.
# This embedding is used to condition the UNet on the diffusion time step.
class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # Two linear layers to project the input embedding to a higher-dimensional space.
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        # x: Time step embedding of shape (1, n_embd)

        # Project the input to a higher-dimensional space.
        x = self.linear_1(x)  # (1, n_embd) -> (1, 4 * n_embd)
        
        # Apply SiLU (Swish) activation function.
        x = F.silu(x) 
        
        # Further project the activated output.
        x = self.linear_2(x)  # (1, 4 * n_embd) -> (1, 4 * n_embd)

        # Return the transformed time embedding.
        return x

# UNET_ResidualBlock class implements a residual block with time conditioning.
# It processes feature maps and conditions them with time embeddings.
class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        # Group normalization and convolution layers for feature processing.
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Linear layer to transform the time embedding.
        self.linear_time = nn.Linear(n_time, out_channels)

        # Group normalization and convolution layers for merged features.
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Identity or convolution layer for residual connection.
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):
        # feature: Input feature map of shape (Batch_Size, In_Channels, Height, Width)
        # time: Time embedding of shape (1, n_time)

        # Save the input for the residual connection.
        residue = feature
        
        # Normalize and activate the feature map.
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        
        # Convolve the feature map.
        feature = self.conv_feature(feature)
        
        # Activate the time embedding.
        time = F.silu(time)
        
        # Transform the time embedding to match the feature map dimensions.
        time = self.linear_time(time)
        
        # Add the time embedding to the feature map.
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        # Normalize, activate, and convolve the merged output.
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        
        # Add the residual connection.
        return merged + self.residual_layer(residue)

# UNET_AttentionBlock class implements a block with self-attention, cross-attention, and feed-forward layers.
# It enhances feature representations by attending to relevant information.
class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        
        # Group normalization and convolution layers for input processing.
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        # Layer normalization and attention layers.
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        
        # Linear layers for the GeGLU feed-forward network.
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        # Convolution layer for the output.
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        # x: Feature map of shape (Batch_Size, Features, Height, Width)
        # context: Contextual information of shape (Batch_Size, Seq_Len, Dim)

        # Save the input for the final skip connection.
        residue_long = x

        # Normalize and process the input feature map.
        x = self.groupnorm(x)
        x = self.conv_input(x)
        
        # Reshape the feature map for attention operations.
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        
        # Self-attention with skip connection.
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short
        
        # Cross-attention with skip connection.
        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short
        
        # Feed-forward network with GeGLU and skip connection.
        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short
        
        # Reshape the output back to the original feature map dimensions.
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        # Final skip connection and output processing.
        return self.conv_output(x) + residue_long

# Upsample class is responsible for increasing the spatial dimensions of the input feature map.
class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Convolution layer to process the upsampled feature map.
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x: Feature map of shape (Batch_Size, Features, Height, Width)

        # Upsample the feature map using nearest neighbor interpolation.
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        # Convolve the upsampled feature map.
        return self.conv(x)

# SwitchSequential class is a custom sequential container that handles different types of layers.
# It applies layers in sequence, handling specific layer types differently.
class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

# UNET class implements a U-Net architecture with attention and residual blocks.
# It processes input data through a series of encoder and decoder layers.
class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the encoder layers consisting of convolution, residual, and attention blocks.
        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        # Define the bottleneck layers for processing the most compressed representation.
        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280), 
            UNET_AttentionBlock(8, 160), 
            UNET_ResidualBlock(1280, 1280), 
        )
        
        # Define the decoder layers consisting of residual, attention, and upsampling blocks.
        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        # x: Input feature map of shape (Batch_Size, 4, Height / 8, Width / 8)
        # context: Contextual information of shape (Batch_Size, Seq_Len, Dim)
        # time: Time embedding of shape (1, 1280)

        # List to store skip connections from encoder layers.
        skip_connections = []
        
        # Pass input through encoder layers, storing skip connections.
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        # Process through the bottleneck layers.
        x = self.bottleneck(x, context, time)

        # Pass through decoder layers, concatenating skip connections.
        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        # Return the final processed output.
        return x

# UNET_OutputLayer class implements the final output layer of the UNet.
# It processes the final feature map to produce the output image.
class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Group normalization and convolution layers for output processing.
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x: Feature map of shape (Batch_Size, 320, Height / 8, Width / 8)

        # Normalize and activate the feature map.
        x = self.groupnorm(x)
        x = F.silu(x)
        
        # Convolve to produce the output image.
        x = self.conv(x)
        
        # Return the final output.
        return x

# Diffusion class implements the entire diffusion model.
# It consists of a time embedding, UNet, and output layer.
class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize the time embedding, UNet, and output layer.
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent, context, time):
        # latent: Latent feature map of shape (Batch_Size, 4, Height / 8, Width / 8)
        # context: Contextual information of shape (Batch_Size, Seq_Len, Dim)
        # time: Time step embedding of shape (1, 320)

        # Transform the time step into a higher-dimensional embedding.
        time = self.time_embedding(time)
        
        # Process the latent feature map through the UNet.
        output = self.unet(latent, context, time)
        
        # Produce the final output image.
        output = self.final(output)
        
        # Return the final output.
        return output