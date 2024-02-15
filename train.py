import os
import time

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from utils import AverageMeter


class Trainer:
    """ Trainer for MNIST classification """

    def __init__(self, model: nn.Module):
        self._model = model

    def train(
            self,
            train_loader: DataLoader,
            epochs: int,
            lr: float,
            save_dir: str,
    ) -> None:
        """ Model training, TODO: consider adding model evaluation into the training loop """

        optimizer = optim.SGD(params=self._model.parameters(), lr=lr)
        loss_track = AverageMeter()
        self._model.train()

        print("Start training...")
        for i in range(epochs):
            tik = time.time()
            loss_track.reset()
            for data, target in train_loader:
                optimizer.zero_grad()
                output = self._model(data)

                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                loss_track.update(loss.item(), n=data.size(0))
                print(loss_track.avg)

            elapse = time.time() - tik
            print("Epoch: [%d/%d]; Time: %.2f; Loss: %.5f" % (i + 1, epochs, elapse, loss_track.avg))

        print("Training completed, saving model to %s" % save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._model.state_dict(), os.path.join(save_dir, "mnist2.pth"))

        return

    def eval(self, test_loader: DataLoader) -> float:
        """ Model evaluation, return the model accuracy over test set """
        self._model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                outputs = self._model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        print(f"Accuracy of the network on test images: {100 * correct // total}%")
        return

    def infer(self, sample: Tensor) -> None:
        """ Model inference: input an image, return its class index """
        self._model.eval()
        with torch.no_grad():
            outputs = self._model(sample)
            _, predicted = torch.max(outputs.data, 1)
            print("Predicted of the first batch in the test dataset is", predicted.tolist())
        return

    def load_model(self, path: str) -> None:
        """ load model from a .pth file """
        self._model.load_state_dict(torch.load(path))
        return
    
    def fine_tune(self,
            train_loader: DataLoader,
            epochs: int,
            lr: float,
            save_dir: str,) -> None:
        optimizer = optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        loss_track = AverageMeter()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model.train()

        print("Start training...")
        for i in range(epochs):
            tik = time.time()
            loss_track.reset()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self._model(data)

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                loss_track.update(loss.item(), n=data.size(0))
                print(loss_track.avg)

            elapse = time.time() - tik
            print("Epoch: [%d/%d]; Time: %.2f; Loss: %.5f" % (i + 1, epochs, elapse, loss_track.avg))

        print("Training completed, saving model to %s" % save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._model.state_dict(), os.path.join(save_dir, "mnist.pth"))

        return
