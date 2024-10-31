import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer
import torch

def main():
    # Set the device for computation
    DEVICE = "cpu"
    
    ALLOW_CUDA = False
    ALLOW_MPS = False
    
    if torch.cuda.is_available() and ALLOW_CUDA:
        DEVICE = "cuda"
    
    # elif(torch.has_mps or torch.backend.mps.is_available()) and ALLOW_MPS:
    #     DEVICE = "mps"
    
    print(f"Using device: {DEVICE}")
    
    # Initialize the tokenizer with the vocabulary and merges files
    tokenizer = CLIPTokenizer(
        vocab_file=r"C:\Users\vimle\Desktop\vae\pytorch-stable-diffusion\data\tokenizer_vocab.json",
        merges_file=r"C:\Users\vimle\Desktop\vae\pytorch-stable-diffusion\data\tokenizer_merges.txt"
    )
    
    # Load the model weights from the checkpoint file
    model_file = r"C:\Users\vimle\Desktop\vae\pytorch-stable-diffusion\data\v1-5-pruned-emaonly.ckpt"
    models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)
    
    ## TEXT TO IMAGE
    
    # Define the prompt for image generation
    prompt = "A tall man with daggers, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
    uncond_prompt = ""  # Also known as negative prompt
    do_cfg = True
    cfg_scale = 8  # min: 1, max: 14
    
    ## IMAGE TO IMAGE
    
    input_image = None
    # Uncomment the following lines to enable image-to-image generation
    # image_path = r"C:\Users\vimle\Desktop\vae\pytorch-stable-diffusion\images\gojo.jpg"
    # input_image = Image.open(image_path)
    # Higher values means more noise will be added to the input image, so the result will further from the input image.
    # Lower values means less noise is added to the input image, so output will be closer to the input image.
    strength = 0.9
    
    ## SAMPLER
    
    sampler = "ddpm"
    num_inference_steps = 50
    seed = 42
    
    # Generate the output image
    output_image = pipeline.generate(
        prompt=prompt,
        uncond_prompt=uncond_prompt,
        input_image=input_image,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        n_inference_steps=num_inference_steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )
    
    # Convert the output image array to an image and display/save it
    result_image = Image.fromarray(output_image)
    result_image.show()  # Display the image
    # result_image.save("output.png")  # Save the image if needed

if __name__ == "__main__":
    main()