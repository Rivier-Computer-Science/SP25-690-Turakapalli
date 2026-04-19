import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
from pathlib import Path
import sys
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from models.residual_cnn import ResidualCNN
from models.baseline_cnn import BaselineCNN


class NoisyDataset(Dataset):
    def __init__(self, clean_dataset, noise_std=25):
        self.clean_dataset = clean_dataset
        self.noise_std = noise_std

    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, idx):
        clean_img, _ = self.clean_dataset[idx]
        # Add Gaussian noise
        noise = torch.randn_like(clean_img) * (self.noise_std / 255.0)
        noisy_img = torch.clamp(clean_img + noise, 0, 1)
        return noisy_img, clean_img


def get_model(name):
    if name == "residual":
        return ResidualCNN()
    else:
        return BaselineCNN()


def train_model(model_name, epochs=10, batch_size=32, noise_std=25, lr=0.001):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    # Load CIFAR-10
    transform = T.Compose([T.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    noisy_train_dataset = NoisyDataset(train_dataset, noise_std=noise_std)
    train_loader = DataLoader(noisy_train_dataset, batch_size=batch_size, shuffle=True)

    model = get_model(model_name).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save model
    checkpoint_path = f"./training/{model_name}_cifar10_noise{noise_std}.pth"
    torch.save({'model_state': model.state_dict()}, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="residual", choices=["baseline", "residual"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--noise_std", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()
    train_model(args.model, args.epochs, args.batch_size, args.noise_std, args.lr)