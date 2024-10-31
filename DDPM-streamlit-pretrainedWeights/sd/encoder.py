import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        # Initialize the VAE_Encoder as a subclass of nn.Sequential
        # nn.Sequential is a container module that processes inputs through a sequence of modules.
        super().__init__(
            # The input is expected to have 3 channels (e.g., RGB image).
            # First layer: Convolutional layer to increase the number of channels to 128.
            # Kernel size is 3x3, padding is 1 to preserve spatial dimensions.
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            # Add a residual block with 128 input and output channels.
            VAE_ResidualBlock(128, 128),
            
            # Another residual block with 128 input and output channels.
            VAE_ResidualBlock(128, 128),
            
            # Convolutional layer with stride 2 to downsample the spatial dimensions by half.
            # Padding is 0, so the spatial dimensions will be reduced.
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # Add a residual block to increase the number of channels to 256.
            VAE_ResidualBlock(128, 256), 
            
            # Another residual block with 256 input and output channels.
            VAE_ResidualBlock(256, 256), 
            
            # Convolutional layer with stride 2 to further downsample the spatial dimensions.
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 
            
            # Add a residual block to increase the number of channels to 512.
            VAE_ResidualBlock(256, 512), 
            
            # Another residual block with 512 input and output channels.
            VAE_ResidualBlock(512, 512), 
            
            # Convolutional layer with stride 2 to further downsample the spatial dimensions.
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            
            # Several residual blocks with 512 input and output channels to add depth and complexity.
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            
            # Attention block to allow the model to focus on important features.
            VAE_AttentionBlock(512), 
            
            # Another residual block with 512 input and output channels.
            VAE_ResidualBlock(512, 512), 
            
            # Group normalization to stabilize the learning process.
            # Divides channels into groups and applies normalization within each group.
            nn.GroupNorm(32, 512), 
            
            # SiLU activation function (also known as Swish) to introduce non-linearity.
            nn.SiLU(), 

            # Convolutional layer to reduce the number of channels to 8.
            # Padding is 1 to preserve spatial dimensions.
            nn.Conv2d(512, 8, kernel_size=3, padding=1), 

            # Final convolutional layer with kernel size 1x1 to refine the features.
            nn.Conv2d(8, 8, kernel_size=1, padding=0), 
        )

    def forward(self, x, noise):
        # x: Input tensor with shape (Batch_Size, Channel, Height, Width)
        # noise: Random noise tensor with shape (Batch_Size, 4, Height / 8, Width / 8)

        for module in self:
            # Check if the module has a stride of (2, 2), indicating downsampling.
            if getattr(module, 'stride', None) == (2, 2):
                # Apply asymmetric padding to maintain the correct spatial dimensions.
                # Padding is applied on the right and bottom by 1 pixel.
                x = F.pad(x, (0, 1, 0, 1))
            
            # Pass the input through the current module.
            x = module(x)

        # Split the output tensor into two tensors along the channel dimension.
        # Each tensor will have half the channels of the original tensor (4 channels each).
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # Clamp the log variance to a range between -30 and 20.
        # This ensures that the variance remains within a reasonable range.
        log_variance = torch.clamp(log_variance, -30, 20)

        # Compute the variance by exponentiating the log variance.
        variance = log_variance.exp()

        # Compute the standard deviation by taking the square root of the variance.
        stdev = variance.sqrt()
        
        # Apply the reparameterization trick: Transform the noise to have the desired mean and standard deviation.
        # This is part of the VAE's stochastic nature, allowing for sampling from the latent space.
        x = mean + stdev * noise
        
        # Scale the output by a constant factor.
        # This scaling factor is often used to ensure the latent space has a certain scale.
        x *= 0.18215
        
        # Return the transformed tensor.
        return x