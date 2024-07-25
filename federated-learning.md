# Federated Learning with Ray and PyTorch

In this tutorial, we'll implement a basic federated learning setup using Ray and PyTorch. This example demonstrates how multiple clients can train local models and aggregate their updates on a central server.

## Prerequisites

Before you start, ensure you have the following packages installed:

```bash
pip install ray torch torchvision
```

## 1. Define the Neural Network Model

We start by defining a simple neural network using PyTorch:

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

## 2. Create a Synthetic Dataset

For demonstration purposes, we use a synthetic dataset:

```python
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = torch.randn(size, 1, 28, 28)
        self.labels = torch.randint(0, 10, (size,))
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

## 3. Define the Training Function

This function trains the model on the local dataset:

```python
import torch.optim as optim

def train_model(model, dataloader, epochs=1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
```

## 4. Setup Ray for Federated Learning

Initialize Ray and define remote functions for training and aggregation:

```python
import ray

# Initialize Ray
ray.init(ignore_reinit_error=True)

@ray.remote
def train_on_client(client_id, model_state_dict, data_loader):
    model = SimpleNN()
    model.load_state_dict(model_state_dict)
    train_model(model, data_loader)
    return model.state_dict()

@ray.remote
def aggregate_models(model_states):
    global_model = SimpleNN()
    global_state_dict = global_model.state_dict()
    
    num_models = len(model_states)
    
    # Aggregate models by averaging weights
    for key in global_state_dict:
        global_state_dict[key] = torch.mean(
            torch.stack([model_states[i][key].float() for i in range(num_models)]),
            dim=0
        )
    
    return global_state_dict
```

## 5. Run Federated Learning

Set up clients, train models, and aggregate updates:

```python
from torch.utils.data import DataLoader

# Create synthetic datasets for clients
num_clients = 3
datasets = [SyntheticDataset(size=100) for _ in range(num_clients)]
data_loaders = [DataLoader(dataset, batch_size=10, shuffle=True) for dataset in datasets]

# Initialize global model
global_model = SimpleNN()
global_state_dict = global_model.state_dict()

# Federated learning process
num_rounds = 5

for round in range(num_rounds):
    # Train on each client
    client_results = [train_on_client.remote(i, global_state_dict, data_loaders[i]) for i in range(num_clients)]
    client_states = ray.get(client_results)
    
    # Aggregate model updates
    global_state_dict = ray.get(aggregate_models.remote(client_states))
    global_model.load_state_dict(global_state_dict)
    
    print(f"Round {round + 1} completed")

print("Federated learning process completed.")
```

## Summary

In this tutorial, you learned how to set up a federated learning process using Ray and PyTorch. We covered defining a model, creating synthetic datasets, training locally, and aggregating model updates. This setup provides a foundation for more advanced federated learning systems, including secure aggregation and real-world data applications.

## References

For further reading on federated learning and distributed computing:

- [Ray Documentation](https://docs.ray.io/en/latest/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)

Feel free to customize and expand upon this example based on your specific use case and dataset!
