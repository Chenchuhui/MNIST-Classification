import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from train import Trainer
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='mnist classification')
    parser.add_argument('--epochs', type=int, default=10, help="training epochs")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--bs', type=int, default=64, help="batch size")
    parser.add_argument('--load', type=bool, default=True, help="load model")
    args = parser.parse_args()

    return args
args = parse_args()

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fit ResNet50 input size
    transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Download and load the MNIST training data
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True)

# Download and load the MNIST test data
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=args.bs, shuffle=False)
model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

# Modify the final fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # 10 classes for MNIST

trainer = Trainer(model=model)

if args.load:
    trainer.load_model("./finetuned_save/mnist.pth")
else:
    trainer.fine_tune(train_loader=train_loader, epochs=args.epochs, lr=args.lr, save_dir="finetuned_save/")

trainer.eval(test_loader=test_loader)
