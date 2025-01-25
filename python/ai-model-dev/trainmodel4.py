import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

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

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_classes):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x is [batch_size, seq_len, input_dim]
        x = self.embedding(x)  # [batch_size, seq_len, d_model]
        x = self.positional_encoding(x)  # Apply positional encoding
        x = x.permute(1, 0, 2)  # Transformer expects input as [seq_len, batch_size, d_model]
        x = self.transformer(x, x)  # The second input is used as memory in transformer
        x = x.mean(dim=0)  # Pooling the output (mean across sequence length)
        x = self.fc(x)  # Output layer
        return x

# Dataset preparation
class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels, group_size=10):
        self.data = data
        self.labels = labels
        self.group_size = group_size

    def __len__(self):
        return len(self.labels)  # Number of groups, not data rows

    def __getitem__(self, idx):
        start_idx = idx * self.group_size
        end_idx = start_idx + self.group_size
        group_data = self.data[start_idx:end_idx]
        label = self.labels[idx]
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

# Create labels for the groups (assuming there are num_groups unique labels)
num_groups = len(data_normalized) // 10  # Number of groups (10 rows per group)
labels = np.repeat(np.arange(num_groups), 10)  # Assign a label to each group of 10 rows

# Check the size of the data and the number of groups
print(f"Total rows in dataset: {len(data_normalized)}")
print(f"Number of groups: {num_groups}")
print("First 5 rows of the normalized dataset:")
print(data_normalized[:5])

# DataLoader setup with drop_last=True to ensure no incomplete batches
group_size = 10
train_dataset = TimeSeriesDataset(data_normalized, labels, group_size)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)

# Initialize the transformer model
input_dim = data_normalized.shape[1]  # Number of features (columns in the dataset)
d_model = 64
nhead = 4
num_encoder_layers = 3
num_classes = num_groups  # Based on number of groups
model = TransformerModel(input_dim, d_model, nhead, num_encoder_layers, num_classes)

# Loss and optimizer (AdamW with L2 regularization)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)  # AdamW with L2 regularization (weight decay)

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

        # Track gradients for each parameter
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradient for {name}: {param.grad.norm()}")

        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# Save the trained model
model_save_path = "transformer_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
