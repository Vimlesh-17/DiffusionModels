import streamlit as st
import model_loader
import pipeline
from PIL import Image
from transformers import CLIPTokenizer
import torch
from tqdm import tqdm

# Function to load models
@st.cache_resource
def load_models(model_file, device):
    return model_loader.preload_models_from_standard_weights(model_file, device)

# Main function to run the Streamlit app
def main():
    st.title("Stable Diffusion Image Generation")

    # Device selection
    device_option = st.selectbox("Select Device", ["cpu", "cuda"], index=0)
    device = torch.device(device_option if torch.cuda.is_available() else "cpu")

    st.write(f"Using device: {device}")

    # Load the tokenizer
    tokenizer = CLIPTokenizer(
        vocab_file="C:/Users/vimle/Desktop/vae/pytorch-stable-diffusion/data/tokenizer_vocab.json",
        merges_file="C:/Users/vimle/Desktop/vae/pytorch-stable-diffusion/data/tokenizer_merges.txt"
    )

    # Load models
    model_file = "C:/Users/vimle/Desktop/vae/pytorch-stable-diffusion/data/v1-5-pruned-emaonly.ckpt"
    models = load_models(model_file, device)

    # Input fields
    prompt = st.text_input("Prompt", "A tall man with daggers, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution.")
    uncond_prompt = st.text_input("Unconditional Prompt", "")
    strength = st.slider("Strength", 0.0, 1.0, 0.8)
    cfg_scale = st.slider("CFG Scale", 1, 14, 8)

    # Image upload
    input_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if input_image is not None:
        input_image = Image.open(input_image)
        st.image(input_image, caption="Uploaded Image", use_column_width=True)

    # Generate button
    if st.button("Generate"):
        with st.spinner("Generating image..."):
            # Initialize progress bar
            progress_bar = st.progress(0)
            total_steps = 50  # Assuming 50 inference steps

            # Define a custom tqdm class to update Streamlit progress bar
            class TqdmStreamlit(tqdm):
                def update_to(self, n):
                    progress_bar.progress(n / total_steps)
                    self.n = n

            # Use the custom tqdm class in the pipeline
            output_image = pipeline.generate(
                prompt=prompt,
                uncond_prompt=uncond_prompt,
                input_image=input_image,
                strength=strength,
                do_cfg=True,
                cfg_scale=cfg_scale,
                sampler_name="ddpm",
                n_inference_steps=total_steps,
                seed=42,
                models=models,
                device=device,
                idle_device="cpu",
                tokenizer=tokenizer,
            )
            st.image(output_image, caption="Generated Image", use_column_width=True)

if __name__ == "__main__":
    main()