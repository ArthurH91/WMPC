import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from model import Net


class NumpyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        goal_translation = self.data[idx, 0]
        goal_rotation = self.data[idx, 1]
        Q = np.array(self.data[idx, 2])

        inputs = np.concatenate((goal_translation, goal_rotation, Q[0]))
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(Q, dtype=torch.float32)

class Training:
    def __init__(self, result_path: str) -> None:
        self._data = np.load(result_path, allow_pickle=True)
        self._T = len(self._data[0, 2])
        self._nq = len(self._data[0, 2][0])
        self._dataset = NumpyDataset(self._data)

    def _create_data_loaders(self, batch_size=32) -> None:
        train_size = int(0.8 * len(self._dataset))
        val_size = len(self._dataset) - train_size
        train_dataset, val_dataset = random_split(self._dataset, [train_size, val_size])

        self._train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self._val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def _set_optimizer(self):
        self._net = Net(self._nq, self._T)
        self._criterion = nn.MSELoss()
        self._optimizer = optim.Adam(self._net.parameters())

    def _train(self, n_epoch: int):
        self._net.train()
        self._running_loss = 0.0
        for i, data in enumerate(self._train_loader, 0):
            inputs, labels = data
            self._optimizer.zero_grad()
            outputs = self._net(inputs)
            loss = self._criterion(outputs, labels)
            loss.backward()
            self._optimizer.step()
            self._running_loss += loss.item()
            if i % self._print_every == self._print_every - 1:
                print(
                    f"Epoch [{n_epoch + 1}/{self._N_epoch}], Step [{i + 1}/{len(self._train_loader)}], Training Loss: {self._running_loss / self._print_every:.4f}"
                )
                self._running_loss = 0.0

    def _eval(self, n_epoch: int):
        self._net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for self._data in self._val_loader:
                inputs, labels = self._data
                outputs = self._net(inputs)
                loss = self._criterion(outputs, labels)
                val_loss += loss.item()
        val_loss /= len(self._val_loader)
        print(f"Epoch [{n_epoch + 1}/{self._N_epoch}], Validation Loss: {val_loss:.4f}")

    def train_and_eval(self, N_epoch=250, print_every=120, batch_size=32, path=""):
        self._print_every = print_every
        self._batch_size = batch_size
        self._path = path
        self._N_epoch = N_epoch

        self._create_data_loaders(batch_size=batch_size)
        self._set_optimizer()

        print(f"Training data size = {len(self._train_loader.dataset)}")
        print(f"Validation data size = {len(self._val_loader.dataset)}")

        for epoch in range(N_epoch):
            self._train(epoch)
            self._eval(epoch)

        print("Finished Training")
        self._save_model()

    def _save_model(self):
        torch.save(self._net.state_dict(), self._path)

if __name__ == "__main__":
    epoch = 1000
    data_path = "/home/arthur/Desktop/Code/WMPC/traj-generation/results/results_box_5000.npy"
    model_path = "/home/arthur/Desktop/Code/WMPC/traj-generation/nn_models/box_5000" + str(epoch) + ".pth"
    training = Training(data_path)
    training.train_and_eval(epoch, path=model_path)