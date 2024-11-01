import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from diffusion_model.grid_search import grid_search, save_results_as_html

def load_bag_dataset(train_path, test_path, batch_size=64):
    """Load and preprocess the bag dataset from .pt files."""
    bag_train = torch.load(train_path, weights_only=True) 
    bag_test = torch.load(test_path, weights_only=True)
    
    bag_train = torch.stack(bag_train).float() # Normalize
    bag_test = torch.stack(bag_test).float()

    train_dataset = TensorDataset(bag_train)
    test_dataset = TensorDataset(bag_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--run_name", type=str, default="default")
    args = parser.parse_args()

    # Load the dataset
    train_loader, test_loader = load_bag_dataset(
        r"C:\Users\vimle\Desktop\Diffusion_repo\Dataset\bag_train.pt",  # Path to the bag training dataset
        r"C:\Users\vimle\Desktop\Diffusion_repo\Dataset\bag_test.pt",   # Path to the bag test dataset
        batch_size=args.batch_size
    )

    # You can modify the dataset and transformations as needed for diffusion models
    transform = transforms.Compose([transforms.Resize(args.image_size), transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    
    # Run grid search for hyperparameter optimization
    results, best_run = grid_search(args, train_loader)

    # Save results as HTML
    save_results_as_html(results, best_run, "grid_search_results.html")

    print(f"Best run: {best_run}")
    print("Results saved to grid_search_results.html")

if __name__ == "__main__":
    main()

