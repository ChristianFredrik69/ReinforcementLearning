import torch
import numpy
import wandb
from torch.utils.data import DataLoader
from torch import nn
import torchvision
from torchvision.transforms import transforms


class CIFARNetworkSimple(nn.Module):


    def __init__(self):
        
        super().__init__()


        
        self.f1 = nn.Conv2d(in_channel = 3, out_channel = 32, kernel_size = 3, stride = 2)
        self.f2 = nn.Conv2d(32, 64, 3, 2)

        self.flatten = nn.Flatten(start_dim=0)
        self.f3 = nn.Linear(in_features=8*8*64, out_features=100)
        self.f4 = nn.Linear(in_features=100, out_features=10)
        
        self.activation = nn.ReLU()

    def forward(self, x):
        
        # Layer 1
        z = self.f1(x)
        a = self.activation(z)

        # Layer 2
        z = self.f2(a)
        a = self.activation(z)

        # Flatten layer
        z = self.flatten(a)

        # Dense layer
        z = self.f3(z)
        a = self.activation(z)

        # Layer 3
        output = self.f4(a)

        return output


if __name__ == '__main__':
    
    # Check Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Running on", device)

    config = {
        "lr": 0.001,
        "architecture": "fc [500, 250, 100, 10]",
        "dataset": "CIFAR",
        "epochs": 25,
        "batch_size": 128
    }

    # wandb.init(project="CIFAR-example", config=config)

    train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, transform=transforms.ToTensor(), download = True)
    test_dataset = torchvision.datasets.CIFAR10(root="data", train=False, transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config["batch_size"], shuffle=True)

    model = CIFARNetworkSimple().to(device)

    lossfunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(1, config['epochs'] + 1):
        train_error = []
        test_acc = []
        for images, labels in train_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Forward pass
            output = model(images)
            loss = lossfunction(output, labels)
            error = loss.to("cpu")

            train_error.append(error)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for images, labels in train_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Forward pass
            output = model(images)
            _, y_pred = torch.max(output.data, 1)
            error = torch.sum(y_pred == labels).to("cpu")

            test_acc.append(error)

        total_train_error = sum(train_error) / len(train_error)
        total_acc = sum(test_acc) / train_dataset.data.shape[0]
        print("Epoch", epoch, "loss:", total_train_error, "\ttest_acc:", total_acc)

        # wandb.log({"train_loss": total_train_error, "test_acc": total_acc})

    torch.save(model.state_dict(), "model.ckpt")
    # wandb.finish()
