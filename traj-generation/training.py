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
        q0 = self.data[idx, 0]
        q1 = self.data[idx, 1]
        inputs = np.concatenate((q0, q1))
        traj = self.data[idx, 2]
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(
            traj, dtype=torch.float32
        )


class Training:
    def __init__(self, result_path: str) -> None:
        """Instanciate the class that trains the model.

        Args:
            result_path (str): path of the data to be trained.
        """
        self._data = np.load(result_path, allow_pickle=True)
        self._T = len(self._data[0, 2])
        self._nq = len(self._data[0, 0])

        # Creating the dataset
        self._dataset = NumpyDataset(self._data)

    def _create_data_loaders(self, batch_size=32) -> None:
        """Set the training variables to create the data loaders.

        Args:
            batch_size (int, optional): Batch size of the datasets for the training. Defaults to 32.
        """
        train_size = int(0.8 * len(self._dataset))
        val_size = len(self._dataset) - train_size
        train_dataset, val_dataset = random_split(self._dataset, [train_size, val_size])

        # Create dataloaders
        self._train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self._val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def _set_optimizer(self):
        """Setup the optimizer"""
        self._net = Net(self._nq, self._T)
        self._criterion = nn.MSELoss()
        self._optimizer = optim.Adam(self._net.parameters())

    def _train(self, n_epoch: int):
        """Train the current epoch

        Args:
            n_epoch (int): number of the current epoch.
        """
        self._net.train()  # Set the network to training mode
        self._running_loss = 0.0
        for i, self._data in enumerate(self._train_loader, 0):
            inputs, labels = self._data
            self._optimizer.zero_grad()
            outputs = self._net(inputs)
            loss = self._criterion(outputs, labels)
            loss.backward()
            self._optimizer.step()
            self._running_loss += loss.item()
            if i % self._print_every == self._print_every - 1:  # print every 10 mini-batches
                print(
                    f"Epoch [{n_epoch + 1}/{self._N_epoch}], Step [{i + 1}/{len(self._train_loader)}], Training Loss: {self._running_loss / self._print_every:.4f}"
                )
                self._running_loss = 0.0

    def _eval(self, n_epoch: int):
        """Eval the current epoch

        Args:
            n_epoch (int): number of the current epoch.
        """
        # Evaluation step
        self._net.eval()  # Set the network to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for self._data in self._val_loader:
                inputs, labels = self._data
                outputs = self._net(inputs)
                loss = self._criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(self._val_loader)
        print(f"Epoch [{n_epoch + 1}/{self._N_epoch}], Validation Loss: {val_loss:.4f}")

    def train_and_eval(self, N_epoch = 250, print_every=120, batch_size=32, path= ""):
        """Train and eval the NN.

        Args:
            N_epoch (int, optional): Number of epoch during with the network will be trained. Defaults to 250.
            print_every (int, optional): Number of iterations between each print. Defaults to 120.
            batch_size (int, optional): Batch size of the datasets for the training. Defaults to 32.

            
        """
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
        
        # Save the trained model
        torch.save(
            self._net.state_dict(),
            self._path,
        )

if __name__ == "__main__":

    data_path = "/home/arthur/Desktop/Code/warmstart-mpc/example/results/results_wall_6000.npy"
    
    model_path = "/home/arthur/Desktop/Code/warmstart-mpc/example/models/trained_model_wall_test.pth"

    training = Training(data_path)
    training.train_and_eval(10, path=model_path)