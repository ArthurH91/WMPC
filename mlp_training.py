import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


class MLP(nn.Module):
    def __init__(self, input_size=10, hidden_sizes=[128, 64], output_size=98):
        """
        Args:
            input_size (int): Size of the input layer.
            hidden_sizes (list of int): List where each element is the size of a hidden layer.
            output_size (int): Size of the output layer.
        """
        super(MLP, self).__init__()

        # Create a list to hold layers
        layers = []

        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())

        # Add hidden layers based on hidden_sizes
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())

        # Last hidden layer to output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Register layers as a ModuleList to make them part of the model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TrajectoriesDataset(Dataset):
    def __init__(self, data, T=15):
        self.data = data
        self.T = T

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.data[idx][0]  # target + initial configuration
        trajs = self.data[idx][1]  # list of configurations without the initial one
        return inputs, trajs


if __name__ == "__main__":
    import os.path as osp

    #### Load data ####
    data_filename = "trajectories_sc2_rs_n1000.pt"
    data_path = osp.join(
        osp.dirname(str(osp.abspath(__file__))), "results/trajectories", data_filename
    )
    data = torch.load(data_path, weights_only=True)
    
    
    #### Variables of the OCP ####
    nq = 7 # Number of robot joints
    T = int(len(data[0][1])/nq) # Number of timesteps


    ### Create dataset and dataloaders ###
    # Create dataset
    dataset = TrajectoriesDataset(data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    taset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    #### Train the model ####
    hidden_sizes=[128, 128, 64]
    net = MLP(hidden_sizes=hidden_sizes)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())

    # Variables for training
    N_epoch = 100 
    print_every = 124

    print(f"Training data size = {len(train_loader.dataset)}")
    print(f"Validation data size = {len(val_loader.dataset)}")

    for epoch in range(N_epoch):
        running_loss = 0.0
        net.train()  # Set the network to training mode
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % print_every == print_every - 1:  # print every 10 mini-batches
                print(
                    f"Epoch [{epoch + 1}/{N_epoch}], Step [{i + 1}/{len(train_loader)}], Training Loss: {running_loss / print_every:.4f}"
                )
                running_loss = 0.0

        # Evaluation step
        net.eval()  # Set the network to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Epoch [{epoch + 1}/{N_epoch}], Validation Loss: {val_loss:.4f}")

    print("Finished Training")

    # Save the trained model
    model_path = osp.join(
        osp.dirname(str(osp.abspath(__file__))),
        "results/models",
        data_filename[:-3] + "_model.pth",
    )
    torch.save(net.state_dict(), model_path)
