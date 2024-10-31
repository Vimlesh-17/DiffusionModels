import torch
import numpy as np

class DDPMSampler:
    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        # Initializing parameters for the sampler based on DDPM and Stable Diffusion setups.
        # "beta_start" and "beta_end" refer to initial and final noise levels for a diffusion process, respectively.
        
        # Linearly spaced betas squared to control the noise schedule
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32) ** 2
        # Alphas (1 - betas), which represent the amount of original signal retained
        self.alphas = 1.0 - self.betas
        # Cumulative product of alphas, storing the amount of signal left at each step
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # Constant tensor representing the number 1, used for edge cases in calculations
        self.one = torch.tensor(1.0)

        # Random generator for sampling
        self.generator = generator

        # Number of timesteps for training diffusion process and inference timesteps
        self.num_train_timesteps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50):
        """Set the timesteps for inference by subsampling training timesteps based on step ratio."""
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        # Generate inference timesteps by step ratio
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        """Calculate the previous timestep in the diffusion process based on the current step and step ratio."""
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        """Calculate the variance used for sampling noise at each timestep."""
        prev_t = self._get_previous_timestep(timestep)

        # Product of alphas up to timestep t and previous timestep
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one

        # Calculate beta for current timestep based on alphas' cumulative product
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # Variance calculation as per DDPM formulas; clamped to avoid very small values
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)

        return variance
    
    def set_strength(self, strength=1):
        """Adjust inference strength to control output closeness to the input image.
        Higher strength adds more noise and further deviates from the input.
        """
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        """Perform a single sampling step to generate the previous timestep's latents."""
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # Compute cumulative product of alpha for the current and previous timestep
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one

        # Beta products for current and previous timestep
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # Compute alpha and beta ratios for current timestep
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # Predicted original sample based on model output
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # Coefficients for computing predicted previous sample
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

        # Calculate previous latent sample
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # Add noise based on variance if t > 0
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise
        
        # Add noise to predicted sample
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample
    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """Adds noise to original samples following the diffusion process."""
        # Retrieve cumulative alphas for the specified timesteps
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        # Calculate sqrt(alpha) and sqrt(1 - alpha) at each timestep
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # Generate noise and create noisy samples
        noise = torch.randn(original_samples.shape, generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
