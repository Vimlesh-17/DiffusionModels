import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusion_model.unet import UNet
from diffusion_model.diffusion import Diffusion
from diffusion_model.utils import save_images, setup_logging
from tqdm import tqdm

def train(args, dataset):
    setup_logging(args.run_name)
    device = args.device
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = torch.nn.MSELoss()
    diffusion = Diffusion(noise_steps=args.noise_steps, img_size=args.image_size, device=device)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    loss_history = []

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader)
        epoch_loss = 0
        for i, (images,) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, f"results/{args.run_name}/{epoch}.jpg")

    torch.save(model.state_dict(), f"models/{args.run_name}/model.pth")
    return loss_history

def test_model(model, diffusion, device, n):
    model.eval()
    with torch.no_grad():
        sampled_images = diffusion.sample(model, n)
        save_images(sampled_images, f"results/test_output.jpg")
