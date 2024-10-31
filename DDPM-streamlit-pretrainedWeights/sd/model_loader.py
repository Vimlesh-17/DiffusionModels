# Import necessary modules and classes
from clip import CLIP  # Importing the CLIP model class, which is used for zero-shot learning tasks.
from encoder import VAE_Encoder  # Importing the VAE Encoder class, which is part of a Variational Autoencoder (VAE).
from decoder import VAE_Decoder  # Importing the VAE Decoder class, complementing the VAE Encoder.
from diffusion import Diffusion  # Importing the Diffusion class, likely used for generative modeling or enhancing image generation.

import model_converter  # Importing a module named model_converter, presumably containing functions to handle model weights.

def preload_models_from_standard_weights(ckpt_path, device):
    """
    Preloads models with weights from a checkpoint file.

    Args:
        ckpt_path (str): Path to the checkpoint file containing the model weights.
        device (torch.device): The device (CPU or GPU) on which to load the models.

    Returns:
        dict: A dictionary containing the initialized models with loaded weights.
    """
    
    # Load the state dictionary containing model weights from the checkpoint file.
    # This function is expected to return a dictionary with keys corresponding to model names.
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    # Initialize the VAE Encoder model and move it to the specified device (CPU or GPU).
    encoder = VAE_Encoder().to(device)
    # Load the encoder weights from the state dictionary. The 'strict=True' ensures that the keys in the state_dict 
    # match exactly with the model's state_dict keys.
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    # Initialize the VAE Decoder model and move it to the specified device.
    decoder = VAE_Decoder().to(device)
    # Load the decoder weights from the state dictionary.
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    # Initialize the Diffusion model and move it to the specified device.
    diffusion = Diffusion().to(device)
    # Load the diffusion model weights from the state dictionary.
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    # Initialize the CLIP model and move it to the specified device.
    clip = CLIP().to(device)
    # Load the CLIP model weights from the state dictionary.
    clip.load_state_dict(state_dict['clip'], strict=True)

    # Return a dictionary containing all the initialized and loaded models.
    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }