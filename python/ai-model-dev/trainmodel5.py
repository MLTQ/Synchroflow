import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import ai_edge_torch

# Define the positional encoding function
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Define the Transformer model with proper activation and layer normalization
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.layer_norm = nn.LayerNorm(d_model)  # Layer normalization added for stability

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.layer_norm(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, x)
        x = x.mean(dim=0)
        x = self.fc(x)
        return x

# Dataset preparation
class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, group_size=10):
        self.data = data
        self.labels = labels
        self.group_size = group_size
        if len(data) % group_size != 0:
            raise ValueError(f"The number of samples {len(data)} is not divisible by the group size {group_size}")
        if len(data) % group_size != 0:
            print(f"Warning: The last group will be dropped since the data is not divisible by {group_size}")

    def __len__(self):
        total_groups = len(self.data) // self.group_size
        return total_groups

    def __getitem__(self, idx):
        start_idx = idx * self.group_size
        end_idx = start_idx + self.group_size
        group_data = self.data[start_idx:end_idx]  # Make sure group_data has shape [seq_len, num_features]

        label = self.labels[idx * self.group_size]
        print(f"Label for group {idx}: {label}")
        return torch.tensor(group_data, dtype=torch.float32), label

# Load data from CSV file (assuming the first row now contains actual data)
csv_file = "testdata6_mini.csv"

# Load data assuming no header row, so we manually specify column names
df = pd.read_csv(csv_file, header=None)  # Avoid using the first row as the header

# Print the first 5 rows to check the data
print("First 5 rows of the dataset:")
print(df.head())

# Extract data (now no column headers)
data = df.values  # Get the data matrix (excluding labels)

# Normalize the data using Min-Max normalization (0-1 normalization)
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)  # Normalize the data by each column

num_groups = len(data_normalized) // 10
labels = np.repeat(np.arange(num_groups), 10)
print(f"Shape of data_normalized: {data_normalized.shape}")


train_loader = DataLoader(TimeSeriesDataset(data_normalized, labels), batch_size=1)

# Initialize the transformer model
d_model = 32
input_dim = data_normalized.shape[1]  # Number of features (columns in the dataset)
nhead = 4
num_encoder_layers = 2
num_classes = num_groups  # Based on number of groups
model = TransformerModel(input_dim, d_model, nhead, num_encoder_layers, num_classes)

# Loss and optimizer (AdamW with L2 regularization)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)  # Use a smaller learning rate

# Set the learning rate scheduler
def get_scheduler(optimizer):
    # Use a step scheduler with a decay factor
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Reduces LR by a factor of 0.1 every 5 epochs
    return scheduler

scheduler = get_scheduler(optimizer)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Print the loss for each batch
        print(f"Batch Loss: {loss.item()}")

        if torch.isnan(loss).any():
            print("Loss is NaN!")

        loss.backward()

        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    # Step the scheduler to adjust the learning rate
    scheduler.step()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# Save the trained model
model_save_path = "transformer_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
