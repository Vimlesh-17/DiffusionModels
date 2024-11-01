import torch

class Diffusion:
    def __init__(self, noise_steps=500, beta_start=1e-3, beta_end=0.02, img_size=32, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]  # Index alpha_hat!
                beta = self.beta[t][:, None, None, None]
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                x = (1 / torch.sqrt(alpha)) * (x - (((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise)) + torch.sqrt(beta) * noise  # Corrected formula
        model.train()
        x = (x+1)/2
        x = x.clamp(0,1)
        x = (x*255).type(torch.uint8)
        return x
