# PyTorch Stable Diffusion

This repository contains a PyTorch implementation of Stable Diffusion, a state-of-the-art image generation model. The implementation is designed to run as a Streamlit app, allowing users to generate images from text prompts or modify existing images.

## Features

- **Text-to-Image Generation**: Generate high-quality images from textual descriptions.
- **Image-to-Image Translation**: Modify existing images based on textual prompts.
- **Streamlit Interface**: User-friendly web interface for easy interaction.

## Setup

### Prerequisites

- Python 3.11.3
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Vimlesh-17/DiffusionModels.git
   cd DDPM-stramlit-pretrainedWeights/sd
   cd 
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the necessary model weights and tokenizer files:

   - Download `vocab.json` and `merges.txt` from [Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/tokenizer) and save them in the `data` folder.
   - Download `v1-5-pruned-emaonly.ckpt` from [Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main) and save it in the `data` folder.

## Usage

### Running the Streamlit App

To start the Streamlit app, run the following command:

```bash
streamlit run sd/app.py
```

This will launch a web interface where you can input text prompts and upload images for generation.

### Hyperparameters and Their Impact

The app allows you to adjust several hyperparameters that influence the image generation process:

| Hyperparameter | Description | Impact |
|----------------|-------------|--------|
| **Strength**   | Controls the amount of noise added to the input image. | Higher values result in more deviation from the input image, while lower values keep the output closer to the input. |
| **CFG Scale**  | Classifier-Free Guidance scale. | Higher values increase adherence to the prompt, potentially at the cost of diversity. |
| **Device**     | Choose between CPU and CUDA for computation. | Affects the speed of image generation. CUDA is faster if a compatible GPU is available. |

### Visualizing Hyperparameter Effects

Below are some examples illustrating the effects of different hyperparameter settings:

- **Strength**: 
  - ![Input Image](DiffusionModels/DDPM-streamlit-pretrainedWeights/images/gojo.jpg) Input Image
  - ![Low Strength](DiffusionModels/DDPM-streamlit-pretrainedWeights/images/output2.png) Low Strength: More context retained with some distortions.
  - ![High Strength](DiffusionModels/DDPM-streamlit-pretrainedWeights/images/output1.png) High Strength: Better clarity but less context retained.


## File Structure

- `sd/app.py`: Main file to run the Streamlit app.
- `sd/pipeline.py`: Contains the image generation pipeline.
- `sd/model_loader.py`: Handles model loading and initialization.
- `sd/diffusion.py`: Implements the diffusion model.
- `sd/ddpm.py`: Contains the DDPM sampler class.
- `requirements.txt`: Lists the Python dependencies.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The implementation is inspired by the original Stable Diffusion model and its PyTorch adaptations.
- Special thanks to the Hugging Face community for providing pre-trained models and tokenizer files.