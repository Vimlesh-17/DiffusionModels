import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

# Constants for image dimensions
WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    # Disable gradient calculation as we're not training
    with torch.no_grad():
        # Validate the 'strength' parameter to be within the range (0, 1]
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        # Define a function to move tensors to the idle device if specified
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize a random number generator for reproducibility
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()  # Random seed
        else:
            generator.manual_seed(seed)  # Use specified seed

        # Load and move the CLIP model to the specified device
        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            # Conditional generation with classifier-free guidance
            # Encode the prompt into token IDs and convert to tensor
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # Get the context (embeddings) for the conditional prompt
            cond_context = clip(cond_tokens)

            # Encode the unconditional prompt similarly
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # Get the context for the unconditional prompt
            uncond_context = clip(uncond_tokens)

            # Concatenate conditional and unconditional contexts
            context = torch.cat([cond_context, uncond_context])
        else:
            # Encode the prompt into token IDs and convert to tensor
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # Get the context (embeddings) for the prompt
            context = clip(tokens)

        # Move the CLIP model to the idle device if specified
        to_idle(clip)

        # Initialize the sampler based on the specified sampler name
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        # Define the shape of the latent space
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            # If an input image is provided, encode it into latent space
            encoder = models["encoder"]
            encoder.to(device)

            # Resize and convert the input image to a tensor
            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            # Generate random noise for the encoder
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # Encode the input image into latent space
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise to the encoded latents based on the strength
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            # Move the encoder to the idle device if specified
            to_idle(encoder)
        else:
            # If no input image, start with random noise in latent space
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # Load and move the diffusion model to the specified device
        diffusion = models["diffusion"]
        diffusion.to(device)

        # Iterate over each timestep in the sampling process
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # Get the time embedding for the current timestep
            time_embedding = get_time_embedding(timestep).to(device)

            # Prepare the model input
            model_input = latents

            if do_cfg:
                # Duplicate the model input for conditional and unconditional contexts
                model_input = model_input.repeat(2, 1, 1, 1)

            # Predict noise using the diffusion model
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                # Separate the model output into conditional and unconditional outputs
                output_cond, output_uncond = model_output.chunk(2)
                # Apply classifier-free guidance
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # Update the latents using the sampler step
            latents = sampler.step(timestep, latents, model_output)

        # Move the diffusion model to the idle device if specified
        to_idle(diffusion)

        # Load and move the decoder model to the specified device
        decoder = models["decoder"]
        decoder.to(device)
        # Decode the final latents into an image
        images = decoder(latents)
        # Move the decoder to the idle device if specified
        to_idle(decoder)

        # Rescale the image values to the range [0, 255] for display
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # Rearrange the image tensor dimensions for visualization
        images = images.permute(0, 2, 3, 1)
        # Convert the image tensor to a NumPy array and return the first image
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]
    
def rescale(x, old_range, new_range, clamp=False):
    # Rescale tensor 'x' from 'old_range' to 'new_range'
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        # Optionally clamp the values to the new range
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Calculate the frequency embeddings for the given timestep
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Multiply the timestep by the frequency values
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Return the concatenation of cosine and sine embeddings
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)